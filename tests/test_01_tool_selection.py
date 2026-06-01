"""
Suite 1 — Tool Selection Tests

Verifies that the LLM's JSON output is correctly interpreted and that
the right tool (or no tool) is selected for each input type.

Tests run against _parse_llm_json + the tool field only.
No real LLM calls — each test injects a pre-built JSON response.
"""
import json
import pytest
from unittest.mock import AsyncMock, patch

from tests.conftest import make_llm_response


# ─── Import the parser we're testing ─────────────────────────────────────────
from app.pipeline import _parse_llm_json


# ─── Helpers ──────────────────────────────────────────────────────────────────

def assert_tool(llm_json: str, expected_tool: str | None, label: str = ""):
    """Parse LLM JSON and assert the tool field matches expected."""
    data = _parse_llm_json(llm_json)
    actual = data.get("tool")
    assert actual == expected_tool, (
        f"[{label}] Expected tool={expected_tool!r}, got tool={actual!r}\n"
        f"Full data: {data}"
    )


# ─── 1.1  No-tool scenarios ───────────────────────────────────────────────────

class TestNoToolRequired:
    """Inputs that should be answered directly without any tool call."""

    def test_simple_math(self):
        """25 * 37 → no tool (LLM can compute internally)."""
        llm_out = make_llm_response(tool=None, response="Nine hundred and twenty-five.")
        assert_tool(llm_out, None, "simple_math")

    def test_factual_definition(self):
        """'What is Snowflake?' → no web_search (static knowledge)."""
        llm_out = make_llm_response(tool=None, response="Snowflake is a cloud data warehouse.")
        assert_tool(llm_out, None, "factual_definition")

    def test_greeting(self):
        """'Hello' → no tool."""
        llm_out = make_llm_response(tool=None, response="Hello! How can I help?", tone="warm")
        assert_tool(llm_out, None, "greeting")

    def test_thanks(self):
        """'Thanks' → no tool."""
        llm_out = make_llm_response(tool=None, response="You're welcome!", pace=1.15, tone="warm")
        assert_tool(llm_out, None, "thanks")

    def test_yes_or_ok(self):
        """'Okay' → no tool."""
        llm_out = make_llm_response(tool=None, response="Got it.", pace=1.15, pause_ms=50)
        assert_tool(llm_out, None, "affirmation")

    def test_simple_conversion(self):
        """Unit conversion → no tool."""
        llm_out = make_llm_response(tool=None, response="One kilometre is 0.621 miles.")
        assert_tool(llm_out, None, "unit_conversion")


# ─── 1.2  File tool scenarios ─────────────────────────────────────────────────

class TestFileTools:
    """Inputs that require filesystem tools."""

    def test_read_file(self):
        """'Read my todo.txt' → read_file."""
        llm_out = make_llm_response(tool="read_file", tool_params={"path": "todo.txt"})
        assert_tool(llm_out, "read_file", "read_todo")

    def test_write_file(self):
        """'Save this as notes.txt' → write_file."""
        llm_out = make_llm_response(
            tool="write_file",
            tool_params={"path": "notes.txt", "content": "content here"},
        )
        assert_tool(llm_out, "write_file", "write_notes")

    def test_list_directory(self):
        """'List my workspace files' → list_directory."""
        llm_out = make_llm_response(tool="list_directory", tool_params={"path": "."})
        assert_tool(llm_out, "list_directory", "list_dir")

    def test_delete_file(self):
        """'Delete old.txt' → delete_file."""
        llm_out = make_llm_response(tool="delete_file", tool_params={"path": "old.txt"})
        assert_tool(llm_out, "delete_file", "delete_file")

    def test_append_file(self):
        """'Add a line to log.txt' → append_file."""
        llm_out = make_llm_response(
            tool="append_file",
            tool_params={"path": "log.txt", "content": "new line"},
        )
        assert_tool(llm_out, "append_file", "append_file")


# ─── 1.3  Web search scenarios ────────────────────────────────────────────────

class TestWebSearch:
    """Web search should only fire for explicitly time-sensitive / online queries."""

    def test_ai_news_today(self):
        """'What happened in AI news today?' → web_search."""
        llm_out = make_llm_response(
            tool="web_search",
            tool_params={"query": "AI news today"},
        )
        assert_tool(llm_out, "web_search", "ai_news_today")

    def test_latest_release(self):
        """'What is the latest Snowflake release?' → web_search."""
        llm_out = make_llm_response(
            tool="web_search",
            tool_params={"query": "latest Snowflake release"},
        )
        assert_tool(llm_out, "web_search", "latest_release")

    def test_no_web_search_for_definitions(self):
        """'What is Python?' → NOT web_search (stable knowledge)."""
        llm_out = make_llm_response(tool=None, response="Python is a programming language.")
        assert_tool(llm_out, None, "no_search_for_definition")

    def test_no_web_search_for_math(self):
        """'What is 100 divided by 4?' → NOT web_search."""
        llm_out = make_llm_response(tool=None, response="Twenty-five.")
        assert_tool(llm_out, None, "no_search_for_math")


# ─── 1.4  Obsidian tool scenarios ─────────────────────────────────────────────

class TestObsidianTools:
    """Obsidian-specific tool routing."""

    def test_create_note(self):
        """'Create a note about RAG' → obsidian_create_note."""
        llm_out = make_llm_response(
            tool="obsidian_create_note",
            tool_params={"title": "RAG", "content": "..."},
        )
        assert_tool(llm_out, "obsidian_create_note", "create_note")

    def test_search_notes(self):
        """'What did I write about attention?' → obsidian_search."""
        llm_out = make_llm_response(
            tool="obsidian_search",
            tool_params={"query": "attention mechanisms"},
        )
        assert_tool(llm_out, "obsidian_search", "search_notes")

    def test_morning_briefing(self):
        """'Good morning' → obsidian_morning_briefing."""
        llm_out = make_llm_response(tool="obsidian_morning_briefing", tool_params={})
        assert_tool(llm_out, "obsidian_morning_briefing", "morning_briefing")

    def test_daily_note(self):
        """'Log this to my daily note' → obsidian_append_daily."""
        llm_out = make_llm_response(
            tool="obsidian_append_daily",
            tool_params={"content": "Reviewed RAG pipeline."},
        )
        assert_tool(llm_out, "obsidian_append_daily", "daily_note")

    def test_related_notes(self):
        """'What's related to Transformers?' → obsidian_get_related."""
        llm_out = make_llm_response(
            tool="obsidian_get_related",
            tool_params={"note_title": "Transformers"},
        )
        assert_tool(llm_out, "obsidian_get_related", "related_notes")


# ─── 1.5  Tool validation ─────────────────────────────────────────────────────

class TestToolValidation:
    """ToolRouter should reject calls with missing required parameters."""

    @pytest.mark.asyncio
    async def test_missing_path_rejected(self, tool_router):
        result = await tool_router.dispatch({"tool": "read_file"})  # missing path
        assert result["status"] == "error"
        assert "missing" in result["result"].lower()

    @pytest.mark.asyncio
    async def test_unknown_tool_rejected(self, tool_router):
        result = await tool_router.dispatch({"tool": "launch_missiles"})
        assert result["status"] == "error"
        assert "unknown" in result["result"].lower()

    @pytest.mark.asyncio
    async def test_valid_tool_dispatches(self, tool_router):
        """list_directory (no required params) should dispatch without error."""
        with patch("app.tools.filesystem.tool_list_directory", new=AsyncMock(return_value="file1.txt\nfile2.txt")):
            result = await tool_router.dispatch({"tool": "list_directory"})
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_save_memory_dispatches(self, tool_router):
        result = await tool_router.dispatch({
            "tool": "save_memory",
            "category": "preference",
            "content": "user prefers dark mode",
        })
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_recall_memory_no_results(self, tool_router):
        result = await tool_router.dispatch({"tool": "recall_memory", "query": "something obscure"})
        assert result["status"] == "ok"
        assert "no matching" in result["result"].lower()


# ─── 1.6  JSON parse robustness ──────────────────────────────────────────────

class TestJsonParsing:
    """_parse_llm_json should handle malformed output gracefully."""

    def test_plain_text_fallback(self):
        """Plain text (no JSON) should return a response field with the text."""
        data = _parse_llm_json("Sure, here you go.")
        assert data["tool"] is None
        assert "Sure" in data["response"] or data["response"] == "Done."

    def test_json_with_markdown_fence(self):
        """JSON wrapped in ```json``` fences should parse correctly."""
        raw = '```json\n{"tool": null, "response": "Hello!", "speech": {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"}}\n```'
        data = _parse_llm_json(raw)
        assert data["response"] == "Hello!"

    def test_missing_speech_defaults(self):
        """Missing speech field should get sensible defaults."""
        raw = '{"tool": null, "response": "Hi"}'
        data = _parse_llm_json(raw)
        assert data["speech"]["pace"] == 1.0
        assert data["speech"]["clause_pause_ms"] == 120
        assert data["speech"]["tone"] == "neutral"

    def test_tool_params_extracted(self):
        """tool_params should be present when tool is set."""
        raw = make_llm_response(tool="read_file", tool_params={"path": "x.txt"})
        data = _parse_llm_json(raw)
        assert data["tool"] == "read_file"
        assert data["tool_params"]["path"] == "x.txt"

    def test_no_thought_field(self):
        """'thought' field should not appear in parsed output (removed for security)."""
        raw = make_llm_response()
        data = _parse_llm_json(raw)
        assert "thought" not in data
