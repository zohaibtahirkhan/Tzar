"""
Suite 2 — Memory Tests

Measures memory PRECISION, not quantity.
The curator must store durable facts and reject ephemeral ones.

Two layers tested:
  A. Hot memory (MEMORY.md / USER.md) — always-on curated facts
  B. Curator agent — store / reject / consolidate decisions
"""
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from tests.conftest import make_curator_response


# ─── 2.1  Hot memory CRUD ────────────────────────────────────────────────────

class TestHotMemory:
    """Unit tests for app.memory.hot_memory read/write/remove/replace."""

    @pytest.fixture(autouse=True)
    def patch_paths(self, tmp_data_dir, monkeypatch):
        """Redirect MEMORY_PATH and USER_PATH to tmp dir."""
        import app.memory.hot_memory as hm
        monkeypatch.setattr(hm, "MEMORY_PATH", tmp_data_dir / "MEMORY.md")
        monkeypatch.setattr(hm, "USER_PATH",   tmp_data_dir / "USER.md")

    def test_add_user_memory(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_add, hot_memory_for_prompt
        result = hot_memory_add("user", "User prefers Urdu for casual chat.")
        assert "Saved" in result
        prompt = hot_memory_for_prompt()
        assert "Urdu" in prompt

    def test_add_agent_memory(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_add, hot_memory_for_prompt
        result = hot_memory_add("memory", "Vault uses ISO date format in filenames.")
        assert "Saved" in result
        prompt = hot_memory_for_prompt()
        assert "ISO" in prompt

    def test_no_duplicates(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_add, _read_entries, USER_PATH
        hot_memory_add("user", "User dislikes bullet points.")
        hot_memory_add("user", "User dislikes bullet points.")
        entries = _read_entries(USER_PATH)
        assert entries.count("User dislikes bullet points.") == 1

    def test_remove_entry(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_add, hot_memory_remove, hot_memory_for_prompt
        hot_memory_add("user", "User prefers dark mode.")
        hot_memory_remove("user", "dark mode")
        prompt = hot_memory_for_prompt()
        assert "dark mode" not in prompt

    def test_remove_nonexistent_returns_error(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_remove
        result = hot_memory_remove("user", "something that was never added")
        assert "No entry found" in result

    def test_replace_entry(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_add, hot_memory_replace, hot_memory_for_prompt
        hot_memory_add("user", "User prefers concise answers.")
        hot_memory_replace("user", "concise answers", "User prefers very brief answers, one sentence max.")
        prompt = hot_memory_for_prompt()
        assert "very brief" in prompt
        assert "concise answers" not in prompt

    def test_char_limit_enforced(self, tmp_data_dir, monkeypatch):
        import app.memory.hot_memory as hm
        monkeypatch.setattr(hm, "USER_CHAR_LIMIT", 50)
        hm.hot_memory_add("user", "Short fact.")
        result = hm.hot_memory_add("user", "This entry alone exceeds the tiny fifty char limit we set.")
        assert "exceed" in result.lower() or "limit" in result.lower()

    def test_empty_file_returns_empty_string(self, tmp_data_dir):
        from app.memory.hot_memory import hot_memory_for_prompt
        prompt = hot_memory_for_prompt()
        assert isinstance(prompt, str)


# ─── 2.2  Curator — store vs reject decisions ─────────────────────────────────

class TestCuratorDecisions:
    """
    The curator LLM is mocked. We test that process_memory_candidate
    correctly routes each decision (store / reject / consolidate) to
    the right hot_memory function.
    """

    @pytest.fixture(autouse=True)
    def patch_paths(self, tmp_data_dir, monkeypatch):
        import app.memory.hot_memory as hm
        monkeypatch.setattr(hm, "MEMORY_PATH", tmp_data_dir / "MEMORY.md")
        monkeypatch.setattr(hm, "USER_PATH",   tmp_data_dir / "USER.md")

    # --- Should STORE ---

    @pytest.mark.asyncio
    async def test_stores_database_preference(self, tmp_data_dir):
        """'My favourite database is Snowflake' → curator stores as user preference."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        llm_fn = AsyncMock(return_value=make_curator_response(
            action="store", target="user",
            content="Favourite database: Snowflake.",
        ))
        await process_memory_candidate("[user] My favourite database is Snowflake.", llm_fn, hm)
        prompt = hm.hot_memory_for_prompt()
        assert "Snowflake" in prompt

    @pytest.mark.asyncio
    async def test_stores_concise_preference(self, tmp_data_dir):
        """'I prefer concise answers' → stored as user preference."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        llm_fn = AsyncMock(return_value=make_curator_response(
            action="store", target="user",
            content="Prefers concise answers.",
        ))
        await process_memory_candidate("[user] I prefer concise answers.", llm_fn, hm)
        prompt = hm.hot_memory_for_prompt()
        assert "concise" in prompt

    @pytest.mark.asyncio
    async def test_stores_environment_fact(self, tmp_data_dir):
        """Vault convention fact → stored as agent memory."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        llm_fn = AsyncMock(return_value=make_curator_response(
            action="store", target="memory",
            content="Vault uses ISO date format in filenames.",
        ))
        await process_memory_candidate("[memory] Vault uses ISO dates.", llm_fn, hm)
        prompt = hm.hot_memory_for_prompt()
        assert "ISO" in prompt

    # --- Should REJECT ---

    @pytest.mark.asyncio
    async def test_rejects_lunch(self, tmp_data_dir):
        """'I had chicken for lunch' → rejected (ephemeral)."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        llm_fn = AsyncMock(return_value=make_curator_response(
            action="reject", reason="ephemeral meal detail, not useful in future sessions",
        ))
        result = await process_memory_candidate("[user] I had chicken for lunch.", llm_fn, hm)
        assert "rejected" in result
        prompt = hm.hot_memory_for_prompt()
        assert "chicken" not in prompt

    @pytest.mark.asyncio
    async def test_rejects_movie(self, tmp_data_dir):
        """'I watched a movie yesterday' → rejected (ephemeral)."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        llm_fn = AsyncMock(return_value=make_curator_response(
            action="reject", reason="one-off activity, not a durable preference",
        ))
        result = await process_memory_candidate("[user] I watched a movie yesterday.", llm_fn, hm)
        assert "rejected" in result
        prompt = hm.hot_memory_for_prompt()
        assert "movie" not in prompt

    @pytest.mark.asyncio
    async def test_rejects_generic_thanks(self, tmp_data_dir):
        """'Thanks' → rejected."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        llm_fn = AsyncMock(return_value=make_curator_response(
            action="reject", reason="conversational filler",
        ))
        result = await process_memory_candidate("[user] Thanks.", llm_fn, hm)
        assert "rejected" in result

    # --- Should CONSOLIDATE ---

    @pytest.mark.asyncio
    async def test_consolidates_updated_preference(self, tmp_data_dir):
        """Updating an existing fact → consolidate, not duplicate store."""
        import app.memory.hot_memory as hm
        from app.memory.curator import process_memory_candidate

        # Pre-populate
        hm.hot_memory_add("user", "Prefers concise answers.")

        llm_fn = AsyncMock(return_value=json.dumps({
            "action": "consolidate",
            "target": "user",
            "old_substring": "concise answers",
            "new_content": "Prefers very brief answers, one sentence max.",
        }))
        await process_memory_candidate("[user] Actually I prefer one sentence max.", llm_fn, hm)
        prompt = hm.hot_memory_for_prompt()
        assert "one sentence" in prompt


# ─── 2.3  Long-term memory (SQLite FTS5) ─────────────────────────────────────

class TestLongTermMemory:
    """Tests for the SQLite-backed long-term memory store."""

    @pytest.fixture
    def fresh_db(self, tmp_path):
        return str(tmp_path / "test_memory.db")

    @pytest.mark.asyncio
    async def test_save_and_recall(self, fresh_db):
        from app.memory.manager import LongTermMemory
        mem = LongTermMemory(db_path=fresh_db)
        await mem.initialize()
        await mem.save("preference", "User prefers dark mode.")
        results = await mem.recall("dark mode")
        assert len(results) >= 1
        assert any("dark mode" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_recall_empty(self, fresh_db):
        from app.memory.manager import LongTermMemory
        mem = LongTermMemory(db_path=fresh_db)
        await mem.initialize()
        results = await mem.recall("nothing stored yet")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_by_category(self, fresh_db):
        from app.memory.manager import LongTermMemory
        mem = LongTermMemory(db_path=fresh_db)
        await mem.initialize()
        await mem.save("preference", "Prefers dark mode.")
        await mem.save("note", "Reminder: update vault.")
        prefs = await mem.recall_by_category("preference")
        assert all(r["category"] == "preference" for r in prefs)

    @pytest.mark.asyncio
    async def test_delete_memory(self, fresh_db):
        from app.memory.manager import LongTermMemory
        mem = LongTermMemory(db_path=fresh_db)
        await mem.initialize()
        mem_id = await mem.save("note", "Deletable note.")
        await mem.delete(mem_id)
        results = await mem.recall("Deletable note")
        assert not any("Deletable" in r["content"] for r in results)

    @pytest.mark.asyncio
    async def test_format_for_context_with_query(self, fresh_db):
        from app.memory.manager import LongTermMemory
        mem = LongTermMemory(db_path=fresh_db)
        await mem.initialize()
        await mem.save("preference", "Favourite database is Snowflake.")
        await mem.save("note", "Unrelated reminder about groceries.")
        ctx = await mem.format_for_context("Snowflake")
        assert "Snowflake" in ctx
