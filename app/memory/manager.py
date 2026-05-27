"""
Memory system: short-term conversation buffer + long-term SQLite store.
"""
import json
import asyncio
from datetime import datetime
from typing import Optional
from collections import deque

import aiosqlite
from loguru import logger

from app.config import settings


# ─── Long-Term Memory (SQLite) ────────────────────────────────────────────────

class LongTermMemory:
    """Persistent memory stored in SQLite."""

    def __init__(self, db_path: str = str(settings.memory_db)):
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON memories(category)
            """)
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(content, category, content=memories, content_rowid=id)
            """)
            await db.commit()
        self._initialized = True
        logger.info("Long-term memory initialized at {}", self.db_path)

    async def save(self, category: str, content: str) -> int:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "INSERT INTO memories (category, content) VALUES (?, ?)",
                (category, content),
            )
            await db.execute(
                "INSERT INTO memories_fts(rowid, content, category) VALUES (?, ?, ?)",
                (cursor.lastrowid, content, category),
            )
            await db.commit()
            logger.debug("Saved memory [{}]: {}", category, content[:60])
            return cursor.lastrowid

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Try FTS first
            try:
                cursor = await db.execute(
                    """
                    SELECT m.id, m.category, m.content, m.created_at
                    FROM memories m
                    JOIN memories_fts fts ON m.id = fts.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, limit),
                )
            except Exception:
                # Fallback to LIKE
                cursor = await db.execute(
                    """
                    SELECT id, category, content, created_at
                    FROM memories
                    WHERE content LIKE ? OR category LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", f"%{query}%", limit),
                )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def recall_by_category(self, category: str, limit: int = 10) -> list[dict]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id, category, content, created_at FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?",
                (category, limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete(self, memory_id: int) -> bool:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            await db.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
            await db.commit()
            return True

    async def format_for_context(self, query: str = "") -> str:
        if query:
            memories = await self.recall(query)
        else:
            memories = await self.recall_by_category("preference") + await self.recall_by_category("note")
        if not memories:
            return "No relevant memories found."
        lines = []
        for m in memories:
            ts = m["created_at"]
            lines.append(f"[{m['category']}] ({ts}): {m['content']}")
        return "\n".join(lines)


# ─── Short-Term Memory (in-process deque) ────────────────────────────────────

class ShortTermMemory:
    """Rolling conversation buffer kept in memory."""

    def __init__(self, max_turns: int = settings.memory_short_term_limit):
        self.max_turns = max_turns
        self._buffer: deque[dict] = deque(maxlen=max_turns * 2)  # user + assistant pairs

    def add_user(self, text: str) -> None:
        self._buffer.append({"role": "user", "content": text, "ts": datetime.utcnow().isoformat()})

    def add_assistant(self, text: str) -> None:
        self._buffer.append({"role": "assistant", "content": text, "ts": datetime.utcnow().isoformat()})

    def get_messages(self) -> list[dict]:
        """Return messages in LLM-compatible format."""
        return [{"role": m["role"], "content": m["content"]} for m in self._buffer]

    def format_for_prompt(self) -> str:
        if not self._buffer:
            return "No prior conversation."
        lines = []
        for m in self._buffer:
            role = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role}: {m['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._buffer.clear()

    def to_json(self) -> str:
        return json.dumps(list(self._buffer), indent=2)


# ─── Unified Memory Manager ────────────────────────────────────────────────────

class MemoryManager:
    def __init__(self):
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()

    async def initialize(self) -> None:
        await self.long_term.initialize()

    async def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.short_term.add_user(user_text)
        self.short_term.add_assistant(assistant_text)

    async def get_context(self, query: str = "") -> tuple[str, str]:
        """Returns (memory_context, conversation_history) for prompt building."""
        memory_ctx = await self.long_term.format_for_context(query)
        conv_history = self.short_term.format_for_prompt()
        return memory_ctx, conv_history


# Singleton
memory_manager = MemoryManager()
