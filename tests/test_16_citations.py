"""
Citation tests — a retrieval-grounded answer must say where it came from.

Run: pytest tests/test_16_citations.py -v

100% offline — LLM, memory, and search tools are mocked.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.conftest import make_llm_response, stream_llm_response, stub_memory
from app.tools.rag.citations import (
    Source, _collector, attribution_sentence, describe_sources, is_source_question,
    record_sources, start_source_collection,
)


# ═══════════════════════════════════════════════════════════════════════════════
# citations module
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceCollection:
    def test_record_without_collection_is_noop(self):
        _collector.set(None)
        record_sources([Source("A", "a.pdf", "document")])  # must not raise

    def test_record_dedupes_by_document(self):
        bucket = start_source_collection()
        record_sources([
            Source("Contract", "contract.pdf", "document"),
            Source("Contract", "contract.pdf", "document"),
            Source("Contract", "Notes/Contract.md", "note"),
        ])
        assert len(bucket) == 2

    @pytest.mark.asyncio
    async def test_child_task_records_into_parent_bucket(self):
        """Tools run under asyncio.wait_for — a child task must still see the list."""
        import asyncio

        bucket = start_source_collection()

        async def tool():
            record_sources([Source("Deep", "deep.md", "note")])

        await asyncio.wait_for(tool(), timeout=1)
        assert [s.title for s in bucket] == ["Deep"]


class TestAttributionSentence:
    def _src(self, *titles):
        return [Source(t, f"{t.lower()}.pdf", "document") for t in titles]

    def test_no_sources_no_attribution(self):
        assert attribution_sentence([], "The fee is 5%.") == ""

    def test_appends_single_source(self):
        out = attribution_sentence(self._src("Databricks Contract"), "The fee is 5%.")
        assert out == " That's from Databricks Contract."

    def test_joins_two_and_three(self):
        assert "A and B." in attribution_sentence(self._src("A", "B"), "x")
        assert "A, B, and C." in attribution_sentence(self._src("A", "B", "C"), "x")

    def test_caps_spoken_sources(self):
        out = attribution_sentence(self._src("A", "B", "C", "D", "E"), "x")
        assert out.endswith("and 2 others.")
        assert "D" not in out

    def test_skips_when_answer_already_cites_title(self):
        spoken = "According to the Databricks Contract, the fee is 5%."
        assert attribution_sentence(self._src("Databricks Contract"), spoken) == ""

    def test_skips_when_answer_cites_filename(self):
        srcs = [Source("Q3 Report", "q3-report.pdf", "document")]
        assert attribution_sentence(srcs, "It's in q3-report.pdf, page 2.") == ""

    def test_skips_when_answer_says_nothing_found(self):
        spoken = "I couldn't find anything about that in your documents."
        assert attribution_sentence(self._src("Unrelated"), spoken) == ""

    def test_falls_back_to_filename_stem_when_untitled(self):
        src = Source("", "meeting-notes.docx", "document")
        assert attribution_sentence([src], "x") == " That's from meeting-notes."
        assert src.as_dict() == {"title": "meeting-notes", "location": "meeting-notes.docx", "kind": "document"}


class TestSourceQuestion:
    @pytest.mark.parametrize("text", [
        "Where did that come from?",
        "where did you get that",
        "What's your source?",
        "which document was that",
        "Which note is this from?",
        "how do you know that?",
    ])
    def test_detects_source_questions(self, text):
        assert is_source_question(text)

    @pytest.mark.parametrize("text", [
        "Where did I put my keys?",
        "What's the source code for the router?",
        "Search my notes for attention",
    ])
    def test_ignores_unrelated(self, text):
        assert not is_source_question(text)

    def test_describe_with_sources(self):
        out = describe_sources([
            Source("Contract", "contract.pdf", "document"),
            Source("Genie", "AI/Genie.md", "note"),
        ])
        assert out == "That came from the document Contract, file contract.pdf and your note Genie."

    def test_describe_without_sources(self):
        assert "general knowledge" in describe_sources([])


# ═══════════════════════════════════════════════════════════════════════════════
# Search tools record sources
# ═══════════════════════════════════════════════════════════════════════════════

class TestSearchToolsRecord:
    @pytest.mark.asyncio
    async def test_doc_search_records_top_hits(self):
        from app.tools.rag import search

        hits = [
            {"title": "Contract", "source_path": "/docs/contract.pdf", "chunk_index": 2,
             "content": "Genie clause", "score": 0.03},
            {"title": "Contract", "source_path": "/docs/contract.pdf", "chunk_index": 5,
             "content": "More", "score": 0.02},
        ]
        bucket = start_source_collection()
        with patch.object(search, "_search_sync", return_value=hits), \
             patch("app.memory.projects._active_project", None):
            out = await search.doc_search("genie")

        assert "Contract" in out
        assert [(s.title, s.location, s.kind) for s in bucket] == [("Contract", "contract.pdf", "document")]

    @pytest.mark.asyncio
    async def test_keyword_search_records_notes(self, tmp_vault):
        from app.tools import obsidian

        (tmp_vault / "Transformers.md").write_text("Attention is all you need.")
        bucket = start_source_collection()
        with patch.object(obsidian, "_vault", return_value=tmp_vault):
            await obsidian.obsidian_keyword_search("attention")

        assert [(s.title, s.kind) for s in bucket] == [("Transformers", "note")]


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline wiring
# ═══════════════════════════════════════════════════════════════════════════════

def stub_memory(mock_mem):
    mock_mem.get_context = AsyncMock(return_value=("", ""))
    mock_mem.add_turn    = AsyncMock()


class TestPipelineCitations:
    @pytest.mark.asyncio
    async def test_prefetched_answer_gets_attribution(self):
        """needs_rag query → prefetch records sources → spoken answer names them."""
        from app.pipeline import AssistantPipeline

        async def fake_unified_search(query, top_k=5):
            record_sources([Source("Databricks Contract", "contract.pdf", "document")])
            return "── Local Documents ──\n[1] Databricks Contract (contract.pdf, chunk 1)\n Genie is included."

        pipeline = AssistantPipeline()
        with patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.pipeline.log_turn", new=AsyncMock()), \
             patch("app.pipeline.response_cache") as cache, \
             patch("app.tools.rag.unified_search", new=fake_unified_search):
            stub_memory(mock_mem)
            cache.get = AsyncMock(return_value=None); cache.set = AsyncMock()
            mock_llm.generate_stream = stream_llm_response(make_llm_response(response="Genie is included in the licence."))

            spoken = await pipeline.process_text_input("What did the contract say about Genie in my documents?")

        assert spoken == "Genie is included in the licence. That's from Databricks Contract."
        assert [s.as_dict()["location"] for s in pipeline._last_sources] == ["contract.pdf"]

    @pytest.mark.asyncio
    async def test_source_question_answers_from_last_turn(self):
        from app.pipeline import AssistantPipeline

        pipeline = AssistantPipeline()
        pipeline._last_sources = [Source("Databricks Contract", "contract.pdf", "document")]
        with patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.log_turn", new=AsyncMock()):
            stub_memory(mock_mem)
            spoken = await pipeline.process_text_input("Where did that come from?")

        assert spoken == "That came from the document Databricks Contract, file contract.pdf."
        mock_llm.generate.assert_not_called()
        # The question itself must not wipe the record — user may ask twice.
        assert pipeline._last_sources[0].title == "Databricks Contract"

    @pytest.mark.asyncio
    async def test_ungrounded_answer_has_no_attribution(self):
        from app.pipeline import AssistantPipeline

        pipeline = AssistantPipeline()
        with patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.pipeline.log_turn", new=AsyncMock()), \
             patch("app.pipeline.response_cache") as cache:
            stub_memory(mock_mem)
            cache.get = AsyncMock(return_value=None); cache.set = AsyncMock()
            mock_llm.generate_stream = stream_llm_response(make_llm_response(response="Hello there."))
            spoken = await pipeline.process_text_input("hi")

        assert spoken == "Hello there."
        assert pipeline._last_sources == []

    @pytest.mark.asyncio
    async def test_streaming_emits_attribution_and_sources(self):
        from app.pipeline import AssistantPipeline

        async def fake_unified_search(query, top_k=5):
            record_sources([Source("Lakehouse Notes", "AI/Lakehouse Notes.md", "note")])
            return "── Obsidian Notes ──\n[[Lakehouse Notes]] (AI/Lakehouse Notes.md) — relevance: 0.81"

        pipeline = AssistantPipeline()
        with patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.pipeline.log_turn", new=AsyncMock()), \
             patch("app.tools.rag.unified_search", new=fake_unified_search):
            stub_memory(mock_mem)
            mock_llm.generate_stream = stream_llm_response(
                make_llm_response(response="You wrote that Genie needs Unity Catalog."))
            events = [e async for e in pipeline.stream("What did I write about Genie in my notes?")]

        text_events = [e for e in events if e.kind == "text"]
        assert len(text_events) > 2, "answer should arrive in pieces as tokens stream"
        assert "".join(e.text for e in text_events) == (
            "You wrote that Genie needs Unity Catalog. That's from Lakehouse Notes.")
        assert events[-1].kind == "done" and events[-1].result.sources[0]["kind"] == "note"
        assert pipeline._last_sources[0].kind == "note"

    def test_followup_prompt_asks_for_citation_only_after_retrieval(self):
        from app.pipeline import AssistantPipeline
        p = AssistantPipeline()
        assert "which note or document" not in p._followup_prompt("read_file", "x")
        p._last_sources = [Source("A", "a.pdf", "document")]
        assert "which note or document" in p._followup_prompt("doc_search", "x")

    @pytest.mark.asyncio
    async def test_prefetch_without_sources_injects_nothing(self):
        """unified_search's 'Nothing found' text must not reach the prompt."""
        from app.pipeline import AssistantPipeline

        async def empty_search(query, top_k=5):
            return "Nothing found matching 'x' in notes or documents."

        p = AssistantPipeline()
        p._last_sources = start_source_collection()
        with patch("app.tools.rag.unified_search", new=empty_search):
            assert await p._prefetch_retrieval("x") == ""
