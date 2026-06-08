"""
Tests forIntent Classifier and Research Agent.

Run with:
    pytest tests/test_08_intent_research.py -v

All tests are offline — no LLM, no network, no vault required.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.intent import classify_intent, IntentType, needs_planning


# ─── Intent Classifier ──────────────────────────────────────────────

class TestChatIntent:
    """Short, conversational inputs should always → CHAT."""

    def test_hello(self):
        assert classify_intent("hello") == IntentType.CHAT

    def test_good_morning(self):
        assert classify_intent("good morning") == IntentType.CHAT

    def test_thanks(self):
        assert classify_intent("thanks") == IntentType.CHAT

    def test_yes(self):
        assert classify_intent("yes") == IntentType.CHAT

    def test_what_is_question(self):
        assert classify_intent("what is a transformer?") == IntentType.CHAT

    def test_who_is_question(self):
        assert classify_intent("who is Geoffrey Hinton?") == IntentType.CHAT

    def test_explain_request(self):
        assert classify_intent("explain how attention mechanisms work") == IntentType.CHAT

    def test_math_expression(self):
        assert classify_intent("what is 2+2") == IntentType.CHAT

    def test_very_short_always_chat(self):
        assert classify_intent("hi") == IntentType.CHAT
        assert classify_intent("ok cool") == IntentType.CHAT
        assert classify_intent("nice") == IntentType.CHAT


class TestMemoryIntent:
    """Memory read/write operations → MEMORY."""

    def test_remember_that(self):
        assert classify_intent("remember that I prefer Urdu for casual chat") == IntentType.MEMORY

    def test_my_name_is(self):
        assert classify_intent("my name is Zohaib") == IntentType.MEMORY

    def test_i_prefer(self):
        assert classify_intent("I prefer dark mode in all my apps") == IntentType.MEMORY

    def test_what_did_i_say(self):
        assert classify_intent("what did I say about Snowflake last week?") == IntentType.MEMORY

    def test_do_you_remember(self):
        assert classify_intent("do you remember our conversation about RAG?") == IntentType.MEMORY

    def test_recall(self):
        assert classify_intent("recall what I told you about my project setup") == IntentType.MEMORY


class TestToolIntent:
    """Single tool actions → TOOL."""

    def test_create_note(self):
        assert classify_intent("create a note about vector databases") == IntentType.TOOL

    def test_search_notes(self):
        assert classify_intent("search my notes for attention mechanisms") == IntentType.TOOL

    def test_morning_briefing(self):
        # "good morning" is 2 words → CHAT (via short-circuit)
        # but if phrased differently:
        assert classify_intent("give me my morning briefing") == IntentType.TOOL

    def test_list_files(self):
        assert classify_intent("list files in my workspace") == IntentType.TOOL

    def test_read_file(self):
        assert classify_intent("read the file called notes.txt") == IntentType.TOOL

    def test_obsidian_search(self):
        assert classify_intent("search the vault for transformers") == IntentType.TOOL

    def test_knowledge_graph(self):
        assert classify_intent("show me my knowledge graph stats") == IntentType.TOOL


class TestResearchIntent:
    """Multi-source research requests → RESEARCH."""

    def test_explicit_research(self):
        assert classify_intent("research the latest llama.cpp updates") == IntentType.RESEARCH

    def test_investigate(self):
        assert classify_intent("investigate the new Qwen3 models that just dropped") == IntentType.RESEARCH

    def test_whats_new(self):
        assert classify_intent("what's new in the Ollama project") == IntentType.RESEARCH

    def test_what_is_latest(self):
        assert classify_intent("what is the latest release of mistral") == IntentType.RESEARCH

    def test_compare_external(self):
        assert classify_intent("compare llama.cpp vs ollama performance benchmarks") == IntentType.RESEARCH

    def test_deep_dive(self):
        assert classify_intent("deep dive into how speculative decoding works") == IntentType.RESEARCH

    def test_recent_updates(self):
        assert classify_intent("recent updates to the transformers library") == IntentType.RESEARCH


class TestPlanningIntent:
    """Multi-step compound tasks → PLANNING."""

    def test_create_and_link(self):
        assert classify_intent(
            "create a note about RAG and add it to the knowledge graph"
        ) == IntentType.PLANNING

    def test_search_then_document(self):
        assert classify_intent(
            "find the snowflake release notes and then create a note"
        ) == IntentType.PLANNING

    def test_and_also(self):
        assert classify_intent(
            "write a summary of my project and also add it to the daily log"
        ) == IntentType.PLANNING

    def test_after_that(self):
        assert classify_intent(
            "read my notes on attention and after that create a summary"
        ) == IntentType.PLANNING

    def test_organise(self):
        assert classify_intent("organise all my notes into proper folders") == IntentType.PLANNING

    def test_for_each(self):
        assert classify_intent("for each project folder create a summary note") == IntentType.PLANNING


class TestNeedsPlanningShim:
    """needs_planning() backward-compat shim should return True for PLANNING+RESEARCH."""

    def test_planning_returns_true(self):
        assert needs_planning("find the notes and then add them to the daily log") is True

    def test_research_returns_true(self):
        assert needs_planning("research the latest qwen model releases") is True

    def test_chat_returns_false(self):
        assert needs_planning("what is 2+2") is False

    def test_tool_returns_false(self):
        assert needs_planning("create a note about RAG") is False

    def test_memory_returns_false(self):
        assert needs_planning("remember that I prefer dark mode") is False


class TestIntentLatency:
    """Classifier must be fast — no LLM calls."""

    def test_classify_is_fast(self):
        import time
        texts = [
            "hello",
            "research the latest llama.cpp updates and compare with ollama",
            "create a note about vector databases then add to graph",
            "what is 2+2",
            "remember that I prefer Urdu for casual chat",
        ]
        t0 = time.perf_counter()
        for t in texts * 100:   # 500 classifications
            classify_intent(t)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        # 500 classifications should finish in under 100ms
        assert elapsed_ms < 250, f"Classifier too slow: {elapsed_ms:.1f}ms for 500 calls"


# ─── Research Agent (offline / mock tests) ──────────────────────────

class TestResearchAgentParsing:
    """Unit tests for the JSON parsing helpers — no network needed."""

    def setup_method(self):
        from app.agents.researcher import ResearchAgent
        self.agent = ResearchAgent()

    def test_parse_json_list_clean(self):
        raw = '["query one", "query two", "query three"]'
        result = self.agent._parse_json_list(raw)
        assert result == ["query one", "query two", "query three"]

    def test_parse_json_list_with_markdown(self):
        raw = '```json\n["query one", "query two"]\n```'
        result = self.agent._parse_json_list(raw)
        assert result == ["query one", "query two"]

    def test_parse_json_list_fallback(self):
        """Falls back to regex extraction when JSON is malformed."""
        raw = 'search queries: "first query", "second query"'
        result = self.agent._parse_json_list(raw)
        assert len(result) >= 1

    def test_parse_json_dict_clean(self):
        import json
        raw = json.dumps({
            "findings": ["fact 1", "fact 2"],
            "contradictions": [],
            "summary": "This is the summary.",
            "note_title": "Test Research",
        })
        result = self.agent._parse_json_dict(raw)
        assert result["summary"] == "This is the summary."
        assert result["findings"] == ["fact 1", "fact 2"]

    def test_parse_json_dict_with_markdown(self):
        raw = '```json\n{"findings": ["f1"], "note_title": "title"}\n```'
        result = self.agent._parse_json_dict(raw)
        assert result["note_title"] == "title"

    def test_parse_json_dict_malformed_returns_empty(self):
        raw = "this is not json at all"
        result = self.agent._parse_json_dict(raw)
        assert result == {}

    def test_extract_text_strips_html(self):
        html = "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        text = self.agent._extract_text(html)
        assert "Title" in text
        assert "Content here." in text
        assert "<" not in text

    def test_extract_text_removes_scripts(self):
        html = "<script>alert('xss')</script><p>Real content</p>"
        text = self.agent._extract_text(html)
        assert "alert" not in text
        assert "Real content" in text

    def test_extract_text_decodes_entities(self):
        html = "<p>AT&amp;T &lt;rocks&gt;</p>"
        text = self.agent._extract_text(html)
        assert "AT&T" in text
        assert "<rocks>" in text


class TestResearchAgentDataModel:
    """Test the ResearchTask dataclass."""

    def test_default_fields(self):
        from app.agents.researcher import ResearchTask
        t = ResearchTask(question="test question")
        assert t.question == "test question"
        assert t.search_queries == []
        assert t.raw_results == []
        assert t.findings == []
        assert t.contradictions == []
        assert t.summary == ""
        assert t.sources == []

    def test_build_response_with_summary(self):
        from app.agents.researcher import ResearchAgent, ResearchTask
        agent = ResearchAgent()
        task = ResearchTask(question="test")
        task.summary = "This is the executive summary."
        task.findings = ["Fact one.", "Fact two.", "Fact three."]
        task.note_title = "Test Research Report"

        response = agent._build_response(task)
        assert "This is the executive summary." in response
        assert "Fact one." in response
        assert "Test Research Report" in response

    def test_build_response_with_contradictions(self):
        from app.agents.researcher import ResearchAgent, ResearchTask
        agent = ResearchAgent()
        task = ResearchTask(question="test")
        task.summary = "Summary."
        task.contradictions = ["Source A says X, Source B says Y."]
        task.note_title = "Report"

        response = agent._build_response(task)
        assert "conflicting" in response.lower() or "contradiction" in response.lower()

    def test_minimal_report_fallback(self):
        from app.agents.researcher import ResearchAgent, ResearchTask
        agent = ResearchAgent()
        task = ResearchTask(question="What is RAG?")
        task.note_title = "RAG Research"
        task.summary = "RAG combines retrieval with generation."
        task.findings = ["Finding one.", "Finding two."]
        task.sources = ["1. Source one — http://example.com"]

        report = agent._minimal_report(task)
        assert "# RAG Research" in report
        assert "## Summary" in report
        assert "RAG combines retrieval with generation." in report
        assert "## Key Findings" in report
        assert "- Finding one." in report
        assert "## Sources" in report


class TestSearchResultModel:
    """Test SearchResult dataclass."""

    def test_default_full_text(self):
        from app.agents.researcher import SearchResult
        r = SearchResult(title="Test", url="http://example.com", snippet="A snippet.")
        assert r.full_text == ""

    def test_fields(self):
        from app.agents.researcher import SearchResult
        r = SearchResult(
            title="My Title",
            url="http://example.com/page",
            snippet="Short snippet here.",
            full_text="Full page content extracted.",
        )
        assert r.title == "My Title"
        assert r.url == "http://example.com/page"
        assert r.full_text == "Full page content extracted."


# ─── Integration: classifier routes to research agent ────────────────────────

class TestIntentResearchIntegration:
    """Verify classifier correctly identifies research queries that should
    trigger the Research Agent."""

    RESEARCH_QUERIES = [
        "research the latest llama.cpp updates and compare with ollama",
        "investigate what's new in the mistral model family",
        "what are the recent updates to vllm",
        "deep dive into speculative decoding",
        "find out about the new qwen3 release",
        "compare llama.cpp vs ollama inference speed",
    ]

    NON_RESEARCH_QUERIES = [
        "create a note about vector databases",
        "what is RAG",
        "remember that I prefer Urdu",
        "search my notes for transformers",
        "hello",
        "good morning",
    ]

    def test_all_research_queries_classified_as_research(self):
        for query in self.RESEARCH_QUERIES:
            result = classify_intent(query)
            assert result == IntentType.RESEARCH, (
                f"Expected RESEARCH for: '{query}', got {result.value}"
            )

    def test_non_research_not_classified_as_research(self):
        for query in self.NON_RESEARCH_QUERIES:
            result = classify_intent(query)
            assert result != IntentType.RESEARCH, (
                f"Expected non-RESEARCH for: '{query}', got {result.value}"
            )
