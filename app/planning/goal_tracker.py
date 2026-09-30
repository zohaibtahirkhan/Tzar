"""
app/planning/goal_tracker.py — Long-Horizon Task Persistence

Stores multi-step goals in SQLite so they survive across sessions.
The pipeline checks for in-progress goals at the start of each turn
and resumes them when the user says "continue", "next step", "what's next", etc.

Schema:
    goals
      id, user_text, title, status, created_at, updated_at, project_name

    goal_steps
      id, goal_id, order, description, tool, params_json,
      result, status, created_at, completed_at

Status values:
    goals:      pending | in_progress | completed | abandoned
    goal_steps: pending | running | done | failed | skipped

Usage:
    goal = await goal_tracker.create_goal(user_text, steps)
    await goal_tracker.advance_step(goal.id, step_result)
    active = await goal_tracker.get_active_goals()
    await goal_tracker.resume_or_start(goal_id, pipeline)
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import aiosqlite
from loguru import logger

from app.config import settings

_DB_PATH = settings.data_dir / "goals.db"


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class GoalStep:
    id:           int
    goal_id:      int
    order:        int
    description:  str
    tool:         str
    params:       dict
    result:       str   = ""
    status:       str   = "pending"   # pending|running|done|failed|skipped
    completed_at: Optional[str] = None


@dataclass
class Goal:
    id:           int
    user_text:    str
    title:        str
    status:       str            # pending|in_progress|completed|abandoned
    steps:        list[GoalStep] = field(default_factory=list)
    created_at:   str = ""
    updated_at:   str = ""
    project_name: str = ""

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "done")

    @property
    def next_step(self) -> Optional[GoalStep]:
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def progress_str(self) -> str:
        return f"{self.completed_steps}/{self.total_steps} steps complete"

    def tree_str(self) -> str:
        """Render the goal as a tree for display in the UI."""
        lines = [f"Goal: {self.title}"]
        for s in self.steps:
            icon = {"done": "✓", "failed": "✗", "running": "▶", "skipped": "○", "pending": "·"}
            lines.append(f"  {icon.get(s.status, '·')} {s.description}")
        lines.append(f"  [{self.progress_str()}]")
        return "\n".join(lines)


# ─── DB helpers ───────────────────────────────────────────────────────────────

async def _init_db():
    async with aiosqlite.connect(str(_DB_PATH)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS goals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text    TEXT NOT NULL,
                title        TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'pending',
                project_name TEXT DEFAULT '',
                created_at   TEXT DEFAULT (datetime('now')),
                updated_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS goal_steps (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id      INTEGER NOT NULL REFERENCES goals(id),
                step_order   INTEGER NOT NULL,
                description  TEXT NOT NULL,
                tool         TEXT NOT NULL DEFAULT '',
                params_json  TEXT DEFAULT '{}',
                result       TEXT DEFAULT '',
                status       TEXT NOT NULL DEFAULT 'pending',
                completed_at TEXT
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON goals(status)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_steps_goal ON goal_steps(goal_id)")
        await db.commit()


async def _load_goal(db: aiosqlite.Connection, goal_id: int) -> Optional[Goal]:
    db.row_factory = aiosqlite.Row
    row = await db.execute_fetchall("SELECT * FROM goals WHERE id=?", (goal_id,))
    if not row:
        return None
    g = row[0]
    steps_rows = await db.execute_fetchall(
        "SELECT * FROM goal_steps WHERE goal_id=? ORDER BY step_order", (goal_id,)
    )
    steps = [
        GoalStep(
            id=s["id"], goal_id=s["goal_id"], order=s["step_order"],
            description=s["description"], tool=s["tool"],
            params=json.loads(s["params_json"] or "{}"),
            result=s["result"] or "", status=s["status"],
            completed_at=s["completed_at"],
        )
        for s in steps_rows
    ]
    return Goal(
        id=g["id"], user_text=g["user_text"], title=g["title"],
        status=g["status"], steps=steps, created_at=g["created_at"],
        updated_at=g["updated_at"], project_name=g["project_name"] or "",
    )


# ─── GoalTracker ──────────────────────────────────────────────────────────────

# Anchored: the whole utterance must be the command. A bare "next" inside a
# longer question ("what's the next train?") is not a goal continuation.
_CONTINUATION_RE = re.compile(
    r"(?:ok(?:ay)?[,\s]+|yes[,\s]+|please\s+)?"
    r"(continue|next step|what'?s next|proceed|go ahead|keep going|resume|"
    r"next|what do i do next|carry on|move on)"
    r"(?:\s+please)?[.!?]*",
    re.IGNORECASE,
)

_GOAL_TITLE_PROMPT = """\
Summarise this task request as a short goal title (5 words max).
Request: {text}
Title:"""


class GoalTracker:
    """Persists multi-step goals across sessions."""

    def __init__(self):
        self._initialised = False

    async def _ensure_init(self):
        if not self._initialised:
            await _init_db()
            self._initialised = True

    # ── Create ────────────────────────────────────────────────────────────────

    async def create_goal(
        self,
        user_text: str,
        steps: list[dict],          # list of {description, tool?, params?, status?}
        project_name: str = "",
        title: str = "",
    ) -> Optional[Goal]:
        """
        Persist a new goal with its steps. Returns the created Goal.
        steps: same format as ToolStep dicts from the coordinator plan.
        title: pass the planner's goal sentence to skip the title LLM call.
        """
        if not settings.goal_tracking_enabled:
            return None
        await self._ensure_init()
        title = title.strip() or await self._generate_title(user_text)

        statuses = [s.get("status", "pending") for s in steps]
        goal_status = "completed" if steps and all(st in ("done", "skipped") for st in statuses) else "in_progress"

        async with aiosqlite.connect(str(_DB_PATH)) as db:
            cur = await db.execute(
                "INSERT INTO goals (user_text, title, status, project_name) VALUES (?,?,?,?)",
                (user_text, title, goal_status, project_name),
            )
            goal_id = cur.lastrowid
            await db.executemany(
                "INSERT INTO goal_steps (goal_id, step_order, description, tool, params_json, status, completed_at) "
                "VALUES (?,?,?,?,?,?, CASE WHEN ?='done' THEN datetime('now') END)",
                [(goal_id, i, s.get("description", ""), s.get("tool", ""),
                  json.dumps(s.get("params", {})), st, st)
                 for i, (s, st) in enumerate(zip(steps, statuses))],
            )
            await db.commit()

        logger.info("GoalTracker: created goal #{} — '{}' ({} steps)", goal_id, title, len(steps))
        return await self.get_goal(goal_id)

    # ── Read ──────────────────────────────────────────────────────────────────

    async def get_goal(self, goal_id: int) -> Optional[Goal]:
        await self._ensure_init()
        async with aiosqlite.connect(str(_DB_PATH)) as db:
            return await _load_goal(db, goal_id)

    async def get_active_goals(self, project_name: str = "") -> list[Goal]:
        """Return in-progress goals, most recently updated first."""
        await self._ensure_init()
        async with aiosqlite.connect(str(_DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            if project_name:
                rows = await db.execute_fetchall(
                    "SELECT id FROM goals WHERE status='in_progress' AND project_name=? "
                    "ORDER BY updated_at DESC", (project_name,)
                )
            else:
                rows = await db.execute_fetchall(
                    "SELECT id FROM goals WHERE status='in_progress' ORDER BY updated_at DESC"
                )
            goals = []
            for row in rows:
                g = await _load_goal(db, row["id"])
                if g:
                    goals.append(g)
            return goals

    # ── Update ────────────────────────────────────────────────────────────────

    async def advance_step(
        self,
        goal_id: int,
        step_id: int,
        result: str,
        status: str = "done",
    ) -> Optional[Goal]:
        """Mark a step done/failed and update goal status."""
        await self._ensure_init()
        async with aiosqlite.connect(str(_DB_PATH)) as db:
            await db.execute(
                "UPDATE goal_steps SET status=?, result=?, completed_at=datetime('now') "
                "WHERE id=?",
                (status, result[:2000], step_id),
            )
            # Check if all steps are done
            all_steps = await db.execute_fetchall(
                "SELECT status FROM goal_steps WHERE goal_id=?", (goal_id,)
            )
            statuses = [s[0] for s in all_steps]
            if all(s in ("done", "skipped") for s in statuses):
                await db.execute(
                    "UPDATE goals SET status='completed', updated_at=datetime('now') WHERE id=?",
                    (goal_id,)
                )
                logger.info("GoalTracker: goal #{} completed", goal_id)
            elif "failed" in statuses:
                # Only mark abandoned if a required step failed — keep going otherwise
                pass
            else:
                await db.execute(
                    "UPDATE goals SET updated_at=datetime('now') WHERE id=?", (goal_id,)
                )
            await db.commit()
            return await _load_goal(db, goal_id)

    async def abandon_goal(self, goal_id: int) -> None:
        await self._ensure_init()
        async with aiosqlite.connect(str(_DB_PATH)) as db:
            await db.execute(
                "UPDATE goals SET status='abandoned', updated_at=datetime('now') WHERE id=?",
                (goal_id,)
            )
            await db.commit()
        logger.info("GoalTracker: goal #{} abandoned", goal_id)

    # ── Continuation detection ────────────────────────────────────────────────

    def is_continuation(self, user_text: str) -> bool:
        """True if the user is asking to continue the current goal."""
        if not settings.goal_tracking_enabled:
            return False
        return bool(_CONTINUATION_RE.fullmatch(user_text.strip()))

    async def resume_response(self, goal: Goal) -> str:
        """
        Build a response message for resuming a goal — shows progress tree
        and describes what the next step is.
        """
        next_step = goal.next_step
        if not next_step:
            return f"Goal '{goal.title}' is complete! {goal.progress_str()}"

        return (
            f"{goal.tree_str()}\n\n"
            f"Next up: {next_step.description}\n"
            f"Ready to proceed?"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _generate_title(self, user_text: str) -> str:
        try:
            from app.llm.engine import llm_engine
            raw = await asyncio.wait_for(
                llm_engine.generate(
                    [{"role": "user", "content": _GOAL_TITLE_PROMPT.format(text=user_text[:300])}],
                    system_prompt="Be concise. Return only the title, no quotes.",
                ),
                timeout=8.0,
            )
            title = raw.strip().strip('"').strip("'")[:80]
            return title if title else user_text[:60]
        except Exception:
            return user_text[:60]

    # ── Summary for UI ────────────────────────────────────────────────────────

    async def get_goals_summary(self, limit: int = 10) -> str:
        """Return a formatted summary of recent goals for display."""
        await self._ensure_init()
        async with aiosqlite.connect(str(_DB_PATH)) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT id, title, status, created_at, updated_at "
                "FROM goals ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
            if not rows:
                return "No goals tracked yet."
            lines = ["Recent Goals:"]
            icons = {"in_progress": "▶", "completed": "✓", "abandoned": "✗", "pending": "·"}
            for r in rows:
                icon = icons.get(r["status"], "·")
                lines.append(f"  {icon} [{r['id']}] {r['title']}  ({r['status']})")
            return "\n".join(lines)


# Singleton
goal_tracker = GoalTracker()