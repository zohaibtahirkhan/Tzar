"""
Session archive with FTS5 search.

Every conversation turn is stored permanently in SQLite.
The LLM can search past sessions by content, finding things
discussed weeks ago that are not in hot memory.

Distinct from long-term memory (which is curated facts).
Session search is the raw, unfiltered archive.
"""
import asyncio
import aiosqlite
from datetime import datetime
from pathlib import Path
from loguru import logger

from app.config import settings

SESSION_DB = settings.data_dir / "sessions.db"


# ─── Schema ───────────────────────────────────────────────────────────────────

async def init_session_db() -> None:
    async with aiosqlite.connect(str(SESSION_DB)) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role      TEXT NOT NULL,
                content   TEXT NOT NULL,
                ts        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_session ON sessions(session_id);
            CREATE INDEX IF NOT EXISTS idx_ts ON sessions(ts);

            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts
            USING fts5(content, role, content=sessions, content_rowid=id);

            CREATE TRIGGER IF NOT EXISTS sessions_ai
            AFTER INSERT ON sessions BEGIN
                INSERT INTO sessions_fts(rowid, content, role)
                VALUES (new.id, new.content, new.role);
            END;
        """)
        await db.commit()
    logger.info("Session store initialized at {}", SESSION_DB)


# ─── Write ────────────────────────────────────────────────────────────────────

# Module-level session ID — unique per process run
_CURRENT_SESSION = datetime.utcnow().strftime("%Y%m%d_%H%M%S")


async def log_turn(role: str, content: str) -> None:
    """Append a single turn to the session archive. Call after every exchange."""
    if not content.strip():
        return
    async with aiosqlite.connect(str(SESSION_DB)) as db:
        await db.execute(
            "INSERT INTO sessions (session_id, role, content) VALUES (?, ?, ?)",
            (_CURRENT_SESSION, role, content),
        )
        await db.commit()


# ─── Search ───────────────────────────────────────────────────────────────────

async def session_search(query: str, limit: int = 8) -> str:
    """
    FTS5 full-text search across all past conversations.
    Returns formatted results the LLM can reason over.
    """
    async with aiosqlite.connect(str(SESSION_DB)) as db:
        db.row_factory = aiosqlite.Row

        try:
            cursor = await db.execute(
                """
                SELECT s.session_id, s.role, s.content, s.ts
                FROM sessions s
                JOIN sessions_fts fts ON s.id = fts.rowid
                WHERE sessions_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, limit),
            )
        except Exception:
            # Fallback to LIKE
            cursor = await db.execute(
                """
                SELECT session_id, role, content, ts
                FROM sessions
                WHERE content LIKE ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )

        rows = await cursor.fetchall()

    if not rows:
        return f"No past conversations found matching '{query}'."

    lines = [f"Past conversations matching '{query}':"]
    current_session = None
    for row in rows:
        if row["session_id"] != current_session:
            current_session = row["session_id"]
            date = current_session[:8]
            time = current_session[9:] if len(current_session) > 8 else ""
            lines.append(f"\n── Session {date} {time} ──")
        role_label = "You" if row["role"] == "user" else "Assistant"
        lines.append(f"  {role_label}: {row['content'][:200]}")

    return "\n".join(lines)


async def session_list(limit: int = 10) -> str:
    """List recent sessions with message counts."""
    async with aiosqlite.connect(str(SESSION_DB)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT session_id,
                   COUNT(*) as turns,
                   MIN(ts) as started,
                   MAX(ts) as ended
            FROM sessions
            GROUP BY session_id
            ORDER BY started DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

    if not rows:
        return "No sessions recorded yet."

    lines = ["Recent sessions:"]
    for row in rows:
        sid   = row["session_id"]
        turns = row["turns"] // 2  # user+assistant pairs
        date  = row["started"][:10] if row["started"] else "?"
        lines.append(f"  {date} — {turns} exchanges  [{sid}]")

    return "\n".join(lines)