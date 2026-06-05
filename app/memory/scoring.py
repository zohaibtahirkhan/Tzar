"""
app/memory/scoring.py

Memory Scoring — Phase 6.

Every memory gets four scores:
  importance  — how consequential is this fact? (0.0–1.0)
  recency     — exponential decay from time of last access
  frequency   — how often has this been recalled?
  confidence  — how certain are we this is correct? (0.0–1.0)

Composite retrieval score:
  score = w_importance * importance
        + w_recency    * recency
        + w_frequency  * log(1 + frequency) / log(11)  # normalised to [0,1]
        + w_confidence * confidence

Weights are tunable via config (defaults give importance the most
influence, then recency, then confidence, then frequency).

Schema (added to data/memory.db):

    memory_scores
    ┌──────────────────────────────────────────────────────┐
    │ memory_id   TEXT PRIMARY KEY (FK → hot_memory rows)   │
    │ importance  REAL  DEFAULT 0.5                          │
    │ confidence  REAL  DEFAULT 0.8                          │
    │ frequency   INTEGER DEFAULT 0                          │
    │ last_access TEXT  (ISO datetime, updated on recall)    │
    │ created_at  TEXT                                       │
    └──────────────────────────────────────────────────────┘

Integration:
  1. On memory WRITE → call score_new_memory(memory_id, content)
     This calls a tiny LLM heuristic (or rule-based fallback) to
     estimate importance + confidence.

  2. On memory READ  → call recall_scored(query, top_k)
     Returns memories sorted by composite score, not just semantic
     similarity.

  3. score_decay_pass() — run periodically (or at startup) to
     update recency scores and prune very low-score memories.

Add to TOOL_REGISTRY:
  "memory_scores": memory_scores_summary   — show scoring stats
  "memory_prune":  prune_low_score_memories — remove stale memories
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from app.config import settings

# ─── Config ───────────────────────────────────────────────────────────────────

WEIGHT_IMPORTANCE = 0.45
WEIGHT_RECENCY    = 0.30
WEIGHT_CONFIDENCE = 0.15
WEIGHT_FREQUENCY  = 0.10

RECENCY_HALF_LIFE_DAYS = 14.0   # score halves every 14 days
PRUNE_THRESHOLD        = 0.08   # memories below this composite score get pruned
PRUNE_MIN_AGE_DAYS     = 7      # never prune a memory younger than this


# ─── Importance heuristics (rule-based, no LLM needed) ───────────────────────

_HIGH_IMPORTANCE_PATTERNS = [
    r"\bi prefer\b",
    r"\bmy name is\b",
    r"\bi am\b",
    r"\bi work\b",
    r"\bmy (job|role|profession|career)\b",
    r"\bmy (family|wife|husband|partner|son|daughter|child)\b",
    r"\bmy (project|goal|objective|deadline)\b",
    r"\bi (hate|love|dislike|enjoy|always|never)\b",
    r"\bremember (that|this|always|never)\b",
    r"\bimportant\b",
    r"\bcritical\b",
    r"\bpassword\b",  # high importance, handle carefully
    r"\bdeadline\b",
    r"\bbirthday\b",
    r"\banniversary\b",
]

_LOW_IMPORTANCE_PATTERNS = [
    r"\btoday i (ate|had|watched|saw)\b",
    r"\bi (just|am) (feeling|tired|hungry|bored)\b",
    r"\bthe weather\b",
    r"\bfunny\b",
    r"\blol\b",
    r"\bjoke\b",
]


def estimate_importance(content: str) -> float:
    """
    Rule-based importance estimator. Returns 0.0–1.0.
    Fast — no LLM call. Used as the default scorer.
    """
    lower = content.lower()

    # Check high-importance patterns
    for pattern in _HIGH_IMPORTANCE_PATTERNS:
        if re.search(pattern, lower):
            return 0.9

    # Check low-importance patterns
    for pattern in _LOW_IMPORTANCE_PATTERNS:
        if re.search(pattern, lower):
            return 0.1

    # Default: medium importance
    return 0.5


def estimate_confidence(content: str) -> float:
    """
    Estimate confidence in the memory's accuracy.
    Explicit statements ("my name is X") → high confidence.
    Hedged statements ("I think", "maybe") → lower confidence.
    """
    lower = content.lower()

    hedges = ["i think", "maybe", "perhaps", "probably", "might", "not sure", "i guess"]
    if any(h in lower for h in hedges):
        return 0.6

    assertions = ["my name is", "i am", "i work at", "i prefer", "always", "never"]
    if any(a in lower for a in assertions):
        return 0.95

    return 0.80


# ─── Recency scoring ──────────────────────────────────────────────────────────

def recency_score(last_access_iso: str) -> float:
    """
    Exponential decay based on days since last access.
    Score = 2^(-days / half_life)
    """
    try:
        last = datetime.fromisoformat(last_access_iso)
        days = (datetime.utcnow() - last).total_seconds() / 86400.0
        return math.pow(2.0, -days / RECENCY_HALF_LIFE_DAYS)
    except Exception:
        return 0.5


def frequency_score(frequency: int) -> float:
    """Normalise frequency to [0, 1] using log scale (never exceeds 1.0)."""
    if frequency <= 0:
        return 0.0
    # log(1+n) / log(1+n_max) — caps at 1.0 for any n >= n_max
    N_MAX = 50   # 50 recalls = full score
    return min(1.0, math.log1p(frequency) / math.log1p(N_MAX))


# ─── Composite score ──────────────────────────────────────────────────────────

def composite_score(
    importance: float,
    recency: float,
    confidence: float,
    frequency: int,
) -> float:
    return (
        WEIGHT_IMPORTANCE * importance
        + WEIGHT_RECENCY    * recency
        + WEIGHT_CONFIDENCE * confidence
        + WEIGHT_FREQUENCY  * frequency_score(frequency)
    )


# ─── DB layer ─────────────────────────────────────────────────────────────────

def _get_score_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.memory_db))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_scores (
            memory_id   TEXT PRIMARY KEY,
            importance  REAL    DEFAULT 0.5,
            confidence  REAL    DEFAULT 0.8,
            frequency   INTEGER DEFAULT 0,
            last_access TEXT    DEFAULT (datetime('now')),
            created_at  TEXT    DEFAULT (datetime('now'))
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ms_id ON memory_scores(memory_id)")
    conn.commit()
    return conn


# ─── Public API ───────────────────────────────────────────────────────────────

def score_memory(memory_id: str, content: str) -> dict:
    """
    Score a new memory and persist its scores.
    Called immediately after a memory is written.

    Returns the score dict for logging.
    """
    importance = estimate_importance(content)
    confidence = estimate_confidence(content)
    now        = datetime.utcnow().isoformat()

    conn = _get_score_db()
    conn.execute("""
        INSERT INTO memory_scores (memory_id, importance, confidence, frequency, last_access, created_at)
        VALUES (?, ?, ?, 0, ?, ?)
        ON CONFLICT(memory_id) DO UPDATE SET
            importance  = excluded.importance,
            confidence  = excluded.confidence,
            last_access = excluded.last_access
    """, (memory_id, importance, confidence, now, now))
    conn.commit()
    conn.close()

    scores = {
        "memory_id":  memory_id,
        "importance": importance,
        "confidence": confidence,
        "frequency":  0,
        "recency":    1.0,
        "composite":  composite_score(importance, 1.0, confidence, 0),
    }
    logger.debug("Memory scored: id={} importance={:.2f} confidence={:.2f}", memory_id, importance, confidence)
    return scores


def on_memory_recalled(memory_id: str) -> None:
    """
    Update frequency and last_access when a memory is recalled.
    Call this whenever a memory surfaces in a response.
    """
    conn = _get_score_db()
    conn.execute("""
        INSERT INTO memory_scores (memory_id, frequency, last_access)
        VALUES (?, 1, datetime('now'))
        ON CONFLICT(memory_id) DO UPDATE SET
            frequency   = frequency + 1,
            last_access = datetime('now')
    """, (memory_id,))
    conn.commit()
    conn.close()


def get_memory_score(memory_id: str) -> Optional[dict]:
    """Return the full score dict for a memory, or None if not found."""
    conn = _get_score_db()
    row = conn.execute(
        "SELECT * FROM memory_scores WHERE memory_id=?", (memory_id,)
    ).fetchone()
    conn.close()

    if not row:
        return None

    rec = recency_score(row["last_access"])
    return {
        "memory_id":  memory_id,
        "importance": row["importance"],
        "confidence": row["confidence"],
        "frequency":  row["frequency"],
        "recency":    rec,
        "composite":  composite_score(row["importance"], rec, row["confidence"], row["frequency"]),
    }


def rank_memories_by_score(memory_ids: list[str]) -> list[tuple[float, str]]:
    """
    Given a list of memory IDs, return them sorted by composite score desc.
    Returns [(composite_score, memory_id), ...].
    """
    conn = _get_score_db()
    scored = []
    for mid in memory_ids:
        row = conn.execute(
            "SELECT importance, confidence, frequency, last_access FROM memory_scores WHERE memory_id=?",
            (mid,)
        ).fetchone()
        if row:
            rec   = recency_score(row["last_access"])
            score = composite_score(row["importance"], rec, row["confidence"], row["frequency"])
        else:
            score = 0.5  # default for unscored memories
        scored.append((score, mid))
    conn.close()
    scored.sort(reverse=True)
    return scored


def prune_low_score_memories(dry_run: bool = False) -> str:
    """
    Find and remove memories with composite score below PRUNE_THRESHOLD
    that are older than PRUNE_MIN_AGE_DAYS.

    Returns a summary string.
    """
    cutoff_date = (datetime.utcnow() - timedelta(days=PRUNE_MIN_AGE_DAYS)).isoformat()

    conn = _get_score_db()
    rows = conn.execute(
        "SELECT memory_id, importance, confidence, frequency, last_access, created_at "
        "FROM memory_scores WHERE created_at < ?",
        (cutoff_date,)
    ).fetchall()

    to_prune = []
    for row in rows:
        rec   = recency_score(row["last_access"])
        score = composite_score(row["importance"], rec, row["confidence"], row["frequency"])
        if score < PRUNE_THRESHOLD:
            to_prune.append(row["memory_id"])

    if not to_prune:
        conn.close()
        return "No memories below prune threshold."

    if not dry_run:
        for mid in to_prune:
            conn.execute("DELETE FROM memory_scores WHERE memory_id=?", (mid,))
        conn.commit()

    conn.close()
    action = "Would prune" if dry_run else "Pruned"
    return f"{action} {len(to_prune)} low-score memories (threshold={PRUNE_THRESHOLD:.2f})."


def memory_scores_summary() -> str:
    """Return a human-readable summary of memory scoring stats."""
    conn = _get_score_db()
    total = conn.execute("SELECT COUNT(*) FROM memory_scores").fetchone()[0]
    if total == 0:
        conn.close()
        return "No scored memories yet."

    avg_imp = conn.execute("SELECT AVG(importance) FROM memory_scores").fetchone()[0]
    avg_con = conn.execute("SELECT AVG(confidence) FROM memory_scores").fetchone()[0]
    high    = conn.execute("SELECT COUNT(*) FROM memory_scores WHERE importance >= 0.8").fetchone()[0]
    low     = conn.execute("SELECT COUNT(*) FROM memory_scores WHERE importance <= 0.2").fetchone()[0]

    # Top 5 most important
    top = conn.execute(
        "SELECT memory_id, importance, frequency FROM memory_scores ORDER BY importance DESC LIMIT 5"
    ).fetchall()
    conn.close()

    lines = [
        f"Memory Scores — {total} total",
        f"  Avg importance: {avg_imp:.2f}  Avg confidence: {avg_con:.2f}",
        f"  High-importance (≥0.8): {high}  Low-importance (≤0.2): {low}",
        "\nTop memories by importance:",
    ]
    for row in top:
        lines.append(f"  [{row['memory_id'][:20]}] importance={row['importance']:.2f} recalled={row['frequency']}x")

    return "\n".join(lines)
