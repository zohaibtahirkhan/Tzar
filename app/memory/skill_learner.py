"""
app/memory/skill_learner.py

Skill Auto-Learning

Observes every successful multi-step task execution, logs the tool
sequence to a candidate table, and proposes a skill when the same
sequence appears 3+ times.

Flow:
    Task completes successfully
        ↓
    log_execution(tool_sequence, user_text, outcome)
        ↓
    detect_patterns()
        ↓
    Sequence appeared ≥ PATTERN_THRESHOLD times?
        ↓ yes
    generate_skill_proposal(sequence)
        ↓
    Pending proposal stored — user notified on next turn
        ↓
    User confirms → skill_create()   /   rejects → candidate cleared

Schema (data/memory.db):

    skill_candidates
    ┌──────────────────────────────────────────────────────┐
    │ id            INTEGER PRIMARY KEY AUTOINCREMENT       │
    │ sequence_hash TEXT   — hash of normalised tool list   │
    │ tool_sequence TEXT   — JSON list of tool names        │
    │ example_query TEXT   — one of the triggering queries  │
    │ count         INTEGER DEFAULT 1                       │
    │ proposed      INTEGER DEFAULT 0  (0=no, 1=yes)       │
    │ accepted      INTEGER DEFAULT 0  (0=pending, 1=yes, -1=no) │
    │ skill_name    TEXT                                    │
    │ last_seen     TEXT                                    │
    │ created_at    TEXT                                    │
    └──────────────────────────────────────────────────────┘

Integration in pipeline.py — after a successful plan execution:

    from app.memory.skill_learner import skill_learner
    await skill_learner.log_execution(
        tool_sequence=[r["tool"] for r in tool_results],
        user_text=user_text,
        outcome="success",
    )
    pending = skill_learner.get_pending_proposals()
    if pending:
        # Append to response: "I noticed you always do X — want me to save this as a skill?"
        response += "\\n\\n" + pending[0]["notification"]
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from app.config import settings

# ─── Config ───────────────────────────────────────────────────────────────────

PATTERN_THRESHOLD = settings.skill_pattern_threshold   # seen N times → propose as skill
MIN_SEQUENCE_LEN  = settings.skill_min_sequence_len    # single-tool sequences are too trivial
MAX_SEQUENCE_LEN  = 8        # cap to avoid over-specific skills

# Tools that are too noisy / common to form meaningful skills alone
_SKIP_TOOLS = {"memory_read", "memory_write", "session_list"}


# ─── DB ───────────────────────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.memory_db))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skill_candidates (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            sequence_hash TEXT    UNIQUE NOT NULL,
            tool_sequence TEXT    NOT NULL,
            example_query TEXT    DEFAULT '',
            count         INTEGER DEFAULT 1,
            proposed      INTEGER DEFAULT 0,
            accepted      INTEGER DEFAULT 0,
            skill_name    TEXT    DEFAULT '',
            last_seen     TEXT    DEFAULT (datetime('now')),
            created_at    TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_hash  ON skill_candidates(sequence_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sc_count ON skill_candidates(count DESC)")
    conn.commit()
    return conn


def _sequence_hash(tools: list[str]) -> str:
    key = "|".join(tools)
    return hashlib.md5(key.encode()).hexdigest()


def _normalise(tools: list[str]) -> list[str]:
    """Remove noise tools and deduplicate adjacent repeats."""
    clean = [t for t in tools if t not in _SKIP_TOOLS]
    # Remove adjacent duplicates (e.g. obsidian_search, obsidian_search → obsidian_search)
    result = []
    for t in clean:
        if not result or result[-1] != t:
            result.append(t)
    return result


# ─── Skill proposal builder ───────────────────────────────────────────────────

def _build_skill_name(tools: list[str]) -> str:
    """Derive a readable slug from the tool sequence."""
    # Map common tool suffixes to readable words
    _names = {
        "obsidian_create_note":   "note",
        "obsidian_append_daily":  "daily",
        "obsidian_search":        "search",
        "obsidian_keyword_search":"keyword-search",
        "doc_search":             "doc-search",
        "web_search":             "web-search",
        "kg_add":                 "kg-add",
        "kg_neighbors":           "kg-expand",
        "research_agent":         "research",
        "skill_load":             "skill",
        "obsidian_morning_briefing": "briefing",
    }
    parts = [_names.get(t, t.replace("_", "-").replace("obsidian-", "").replace("tool-", "")) for t in tools]
    name = "-then-".join(parts[:4])  # cap at 4 for readability
    return name[:50]


def _build_skill_content(tools: list[str], example_query: str) -> str:
    """Generate the skill markdown content."""
    tool_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(tools))
    return (
        f"## Auto-learned Skill\n\n"
        f"**Triggered by queries like:** {example_query}\n\n"
        f"**Tool sequence:**\n{tool_list}\n\n"
        f"**When to use:** When the user asks to do something that requires "
        f"this combination of tools in order.\n\n"
        f"**Notes:** This skill was automatically detected from repeated usage."
    )


def _build_notification(name: str, tools: list[str], count: int) -> str:
    """Natural-language notification for the user."""
    tool_str = " → ".join(tools)
    return (
        f"I noticed you've done '{tool_str}' {count} times. "
        f"Want me to save this as a skill called '{name}'? "
        f"Say 'yes save the skill' or 'no skip it'."
    )


# ─── Core learner ─────────────────────────────────────────────────────────────

@dataclass
class SkillProposal:
    sequence_hash: str
    tools: list[str]
    skill_name: str
    skill_content: str
    example_query: str
    count: int
    notification: str


class SkillLearner:
    """
    Observes tool executions, detects patterns, proposes skills.
    """

    def __init__(self):
        self._pending_proposals: list[SkillProposal] = []
        self._awaiting_confirmation: Optional[SkillProposal] = None

    # ── Logging ──────────────────────────────────────────────────────────────

    async def log_execution(
        self,
        tool_sequence: list[str],
        user_text: str = "",
        outcome: str = "success",
    ) -> None:
        """
        Record a tool sequence after successful execution.
        Automatically checks if it should be proposed as a skill.
        """
        if outcome != "success" or not settings.skill_learning_enabled:
            return

        tools = _normalise(tool_sequence)

        if len(tools) < MIN_SEQUENCE_LEN:
            return  # too trivial

        tools = tools[:MAX_SEQUENCE_LEN]
        seq_hash = _sequence_hash(tools)

        loop = asyncio.get_running_loop()

        def _upsert():
            conn = _get_db()
            existing = conn.execute(
                "SELECT id, count, proposed FROM skill_candidates WHERE sequence_hash=?",
                (seq_hash,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE skill_candidates SET count=count+1, last_seen=datetime('now'), "
                    "example_query=? WHERE sequence_hash=?",
                    (user_text[:200], seq_hash),
                )
                new_count = existing["count"] + 1
                already_proposed = existing["proposed"]
            else:
                conn.execute(
                    "INSERT INTO skill_candidates (sequence_hash, tool_sequence, example_query) "
                    "VALUES (?, ?, ?)",
                    (seq_hash, json.dumps(tools), user_text[:200]),
                )
                new_count = 1
                already_proposed = 0

            conn.commit()
            conn.close()
            return new_count, already_proposed

        new_count, already_proposed = await loop.run_in_executor(None, _upsert)

        # Check if we should propose this as a skill
        if new_count >= PATTERN_THRESHOLD and not already_proposed:
            await self._propose_skill(seq_hash, tools, user_text, new_count)

    # ── Proposal ─────────────────────────────────────────────────────────────

    async def _propose_skill(
        self,
        seq_hash: str,
        tools: list[str],
        example_query: str,
        count: int,
    ) -> None:
        """Create a skill proposal and add it to the pending queue."""
        name    = _build_skill_name(tools)
        content = _build_skill_content(tools, example_query)
        notif   = _build_notification(name, tools, count)

        proposal = SkillProposal(
            sequence_hash = seq_hash,
            tools         = tools,
            skill_name    = name,
            skill_content = content,
            example_query = example_query,
            count         = count,
            notification  = notif,
        )
        self._pending_proposals.append(proposal)

        # Mark as proposed in DB
        loop = asyncio.get_running_loop()

        def _mark():
            conn = _get_db()
            conn.execute(
                "UPDATE skill_candidates SET proposed=1, skill_name=? WHERE sequence_hash=?",
                (name, seq_hash),
            )
            conn.commit()
            conn.close()

        await loop.run_in_executor(None, _mark)

        logger.info(
            "SkillLearner: proposing skill '{}' (seq: {} tools, seen {}x)",
            name, len(tools), count,
        )

    # ── User response handling ────────────────────────────────────────────────

    def get_pending_proposals(self) -> list[SkillProposal]:
        """Return list of skill proposals waiting for user confirmation."""
        return list(self._pending_proposals)

    def pop_next_proposal(self) -> Optional[SkillProposal]:
        """Get and remove the next pending proposal (for showing to user)."""
        if self._pending_proposals:
            p = self._pending_proposals.pop(0)
            self._awaiting_confirmation = p
            return p
        return None

    async def confirm_skill(self, accepted: bool) -> str:
        """
        Called when user responds to a skill proposal.
        accepted=True → create the skill
        accepted=False → discard
        """
        proposal = self._awaiting_confirmation
        if proposal is None:
            return "No skill proposal is pending."

        self._awaiting_confirmation = None

        loop = asyncio.get_running_loop()

        def _update_db(accepted_val: int):
            conn = _get_db()
            conn.execute(
                "UPDATE skill_candidates SET accepted=? WHERE sequence_hash=?",
                (accepted_val, proposal.sequence_hash),
            )
            conn.commit()
            conn.close()

        if accepted:
            # Create the skill
            try:
                from app.memory.skills import skill_create
                result = skill_create(
                    name        = proposal.skill_name,
                    description = f"Auto-learned: {' → '.join(proposal.tools)}",
                    content     = proposal.skill_content,
                    category    = "auto-learned",
                )
                await loop.run_in_executor(None, _update_db, 1)
                logger.info("SkillLearner: skill '{}' created", proposal.skill_name)
                return f"Skill '{proposal.skill_name}' saved. I'll use it automatically next time."
            except Exception as e:
                logger.error("Failed to create skill: {}", e)
                return f"Sorry, I couldn't save the skill: {e}"
        else:
            await loop.run_in_executor(None, _update_db, -1)
            logger.info("SkillLearner: skill proposal '{}' rejected", proposal.skill_name)
            return "Got it, I won't suggest that skill again."

    # ── Stats ─────────────────────────────────────────────────────────────────

    async def skill_learning_summary(self) -> str:
        """Return a human-readable summary of the learning state."""
        loop = asyncio.get_running_loop()

        def _stats():
            conn = _get_db()
            total    = conn.execute("SELECT COUNT(*) FROM skill_candidates").fetchone()[0]
            proposed = conn.execute("SELECT COUNT(*) FROM skill_candidates WHERE proposed=1").fetchone()[0]
            accepted = conn.execute("SELECT COUNT(*) FROM skill_candidates WHERE accepted=1").fetchone()[0]
            top = conn.execute(
                "SELECT tool_sequence, count, skill_name FROM skill_candidates "
                "ORDER BY count DESC LIMIT 5"
            ).fetchall()
            conn.close()
            return total, proposed, accepted, top

        total, proposed, accepted, top = await loop.run_in_executor(None, _stats)

        lines = [
            f"Skill Learning — {total} observed patterns",
            f"  Proposed: {proposed}  Accepted: {accepted}  Pending: {len(self._pending_proposals)}",
        ]

        if top:
            lines.append("\nTop patterns:")
            for row in top:
                tools = json.loads(row["tool_sequence"])
                name  = row["skill_name"] or _build_skill_name(tools)
                lines.append(f"  {name} (seen {row['count']}x): {' → '.join(tools)}")

        if self._awaiting_confirmation:
            lines.append(f"\nAwaiting your response on: '{self._awaiting_confirmation.skill_name}'")

        return "\n".join(lines)

    async def reset_candidate(self, sequence_hash: str) -> str:
        """Reset a candidate so it can be proposed again (for testing)."""
        loop = asyncio.get_running_loop()

        def _reset():
            conn = _get_db()
            conn.execute(
                "UPDATE skill_candidates SET proposed=0, accepted=0 WHERE sequence_hash=?",
                (sequence_hash,),
            )
            conn.commit()
            conn.close()

        await loop.run_in_executor(None, _reset)
        return "Candidate reset."


# ─── Singleton ────────────────────────────────────────────────────────────────

skill_learner = SkillLearner()


# ─── Tool functions ───────────────────────────────────────────────────────────

async def skill_learning_stats() -> str:
    """Show the skill auto-learning status and top patterns."""
    return await skill_learner.skill_learning_summary()


async def confirm_skill_proposal(accepted: bool = True) -> str:
    """Accept or reject the pending skill proposal."""
    return await skill_learner.confirm_skill(accepted)
