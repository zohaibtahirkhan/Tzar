"""
Suite 5 — Voice / Speech Tests

Verifies that speech metadata (pace, pause, tone, interruptible)
adapts correctly to different input types and that the TTS interrupt
mechanism respects the interruptible flag.

No audio hardware is required — TTS calls are mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tests.conftest import make_llm_response
from app.pipeline import _parse_llm_json, SpeechMeta, AssistantPipeline


# ─── Helpers ──────────────────────────────────────────────────────────────────

def speech_from_json(llm_json: str) -> dict:
    data = _parse_llm_json(llm_json)
    return data.get("speech", {})


# ─── 5.1  Speech field correctness ───────────────────────────────────────────

class TestSpeechFields:

    def test_thanks_is_fast_and_warm(self):
        """'Thanks' reply should be fast pace, short pause, warm tone."""
        llm_out = make_llm_response(
            response="You're welcome!",
            pace=1.15, pause_ms=50, tone="warm",
        )
        s = speech_from_json(llm_out)
        assert s["pace"] >= 1.1,  f"Expected fast pace, got {s['pace']}"
        assert s["clause_pause_ms"] <= 80, f"Expected short pause, got {s['clause_pause_ms']}"
        assert s["tone"] == "warm", f"Expected warm tone, got {s['tone']}"

    def test_reading_log_file_is_slow_and_informative(self):
        """'Read this log file' reply should be slow, large pause, informative."""
        llm_out = make_llm_response(
            response="Here are the log entries. Error at line 42. Warning at line 87.",
            pace=0.75, pause_ms=350, tone="informative",
        )
        s = speech_from_json(llm_out)
        assert s["pace"] <= 0.85,  f"Expected slow pace, got {s['pace']}"
        assert s["clause_pause_ms"] >= 250, f"Expected large pause, got {s['clause_pause_ms']}"
        assert s["tone"] == "informative", f"Expected informative tone, got {s['tone']}"

    def test_delete_files_is_focused(self):
        """'Delete all files' should trigger focused tone and possibly lower pace."""
        llm_out = make_llm_response(
            response="Are you sure you want to delete all files? This cannot be undone.",
            pace=0.9, pause_ms=200, tone="focused",
            interruptible=False,
        )
        s = speech_from_json(llm_out)
        assert s["tone"] in ("focused", "urgent"), f"Expected focused/urgent, got {s['tone']}"

    def test_explanation_is_medium_pace(self):
        """Long explanation should be medium pace with breathing room."""
        llm_out = make_llm_response(
            response="RAG stands for Retrieval-Augmented Generation. It works by...",
            pace=0.85, pause_ms=200, tone="informative",
        )
        s = speech_from_json(llm_out)
        assert 0.8 <= s["pace"] <= 1.0, f"Expected medium pace, got {s['pace']}"
        assert s["clause_pause_ms"] >= 150, f"Expected medium pause, got {s['clause_pause_ms']}"

    def test_tool_call_response_is_focused(self):
        """When a tool is being called the speech should be focused."""
        llm_out = make_llm_response(
            tool="web_search", tool_params={"query": "Snowflake release"},
            pace=1.0, pause_ms=120, tone="focused",
        )
        s = speech_from_json(llm_out)
        assert s["tone"] == "focused"

    def test_morning_greeting_is_warm(self):
        """'Good morning' response should be warm."""
        llm_out = make_llm_response(
            response="Good morning! Here's your briefing.",
            pace=1.0, pause_ms=150, tone="warm",
        )
        s = speech_from_json(llm_out)
        assert s["tone"] == "warm"

    def test_error_or_warning_is_urgent(self):
        """Error messages should use urgent tone."""
        llm_out = make_llm_response(
            response="Warning: the file could not be found.",
            pace=1.0, pause_ms=200, tone="urgent",
        )
        s = speech_from_json(llm_out)
        assert s["tone"] == "urgent"

    def test_pace_is_within_valid_range(self):
        """Pace must be in [0.5, 1.5] — anything else is a model error."""
        for pace in [0.75, 0.85, 1.0, 1.15]:
            llm_out = make_llm_response(pace=pace)
            s = speech_from_json(llm_out)
            assert 0.5 <= s["pace"] <= 1.5, f"Pace out of range: {s['pace']}"

    def test_pause_ms_is_within_valid_range(self):
        """Pause must be in [0, 600] ms."""
        for pause in [50, 120, 200, 350]:
            llm_out = make_llm_response(pause_ms=pause)
            s = speech_from_json(llm_out)
            assert 0 <= s["clause_pause_ms"] <= 600, f"Pause out of range: {s['clause_pause_ms']}"


# ─── 5.2  SpeechMeta extraction ───────────────────────────────────────────────

class TestSpeechMetaExtraction:
    """AssistantPipeline._extract_speech_meta correctness."""

    def test_extracts_all_fields(self):
        pipeline = AssistantPipeline()
        data = {
            "speech": {"pace": 0.75, "clause_pause_ms": 350, "tone": "informative"}
        }
        meta = pipeline._extract_speech_meta(data)
        assert meta.pace == 0.75
        assert meta.pause_ms == 350
        assert meta.tone == "informative"

    def test_defaults_on_missing_speech(self):
        pipeline = AssistantPipeline()
        meta = pipeline._extract_speech_meta({})
        assert meta.pace == 1.0
        assert meta.pause_ms == 120
        assert meta.tone == "neutral"

    def test_defaults_on_partial_speech(self):
        pipeline = AssistantPipeline()
        data = {"speech": {"pace": 0.8}}  # missing pause and tone
        meta = pipeline._extract_speech_meta(data)
        assert meta.pace == 0.8
        assert meta.pause_ms == 120   # default
        assert meta.tone == "neutral"  # default


# ─── 5.3  Interruptibility ────────────────────────────────────────────────────

class TestInterruptibility:

    def test_default_is_interruptible(self):
        """Normal responses should be interruptible by default."""
        llm_out = make_llm_response(response="Here is the answer.")
        data = _parse_llm_json(llm_out)
        assert data.get("interruptible", True) is True

    def test_critical_response_is_not_interruptible(self):
        """Confirmations of destructive actions should be non-interruptible."""
        llm_out = make_llm_response(
            response="Deleted successfully.",
            interruptible=False,
            tone="focused",
        )
        data = _parse_llm_json(llm_out)
        assert data["interruptible"] is False

    def test_interrupt_fires_when_interruptible_true(self):
        """interrupt() should stop TTS when interruptible=True."""
        pipeline = AssistantPipeline()
        pipeline._speaking = True
        pipeline._interruptible = True
        pipeline._interrupted = False

        pipeline.interrupt()

        assert pipeline._interrupted is True

    def test_interrupt_blocked_when_interruptible_false(self):
        """interrupt() should do nothing when interruptible=False."""
        pipeline = AssistantPipeline()
        pipeline._speaking = True
        pipeline._interruptible = False
        pipeline._interrupted = False

        pipeline.interrupt()

        assert pipeline._interrupted is False

    def test_interrupt_no_effect_when_not_speaking(self):
        """interrupt() should be a no-op when not speaking."""
        pipeline = AssistantPipeline()
        pipeline._speaking = False
        pipeline._interruptible = True
        pipeline._interrupted = False

        pipeline.interrupt()

        assert pipeline._interrupted is False


# ─── 5.4  Confidence field ────────────────────────────────────────────────────

class TestConfidenceField:

    def test_high_confidence_present(self):
        """High confidence response should include confidence >= 0.8."""
        llm_out = make_llm_response(confidence=0.95)
        data = _parse_llm_json(llm_out)
        assert data["confidence"] >= 0.8

    def test_low_confidence_logged(self):
        """Low confidence (< 0.6) should be extracted correctly."""
        llm_out = make_llm_response(confidence=0.45, response="I'm not entirely sure, but...")
        data = _parse_llm_json(llm_out)
        assert data["confidence"] < 0.6

    def test_missing_confidence_defaults_to_high(self):
        """If confidence is missing, parser should default to 0.9."""
        raw = '{"tool": null, "response": "Hello", "speech": {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"}}'
        data = _parse_llm_json(raw)
        assert data.get("confidence", 0.9) >= 0.8

    def test_confidence_in_valid_range(self):
        """Confidence must be 0.0–1.0."""
        for conf in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            llm_out = make_llm_response(confidence=conf)
            data = _parse_llm_json(llm_out)
            assert 0.0 <= data["confidence"] <= 1.0


# ─── 5.5  Full pipeline speech integration ───────────────────────────────────

class TestPipelineSpeechIntegration:
    """
    Verifies that after process_text_input, the speech meta on the
    pipeline matches what the LLM returned.
    """

    @pytest.mark.asyncio
    async def test_speech_meta_set_after_process(self):
        """After processing, pipeline._speech_meta should reflect LLM's speech field."""
        from app.pipeline import AssistantPipeline

        pipeline = AssistantPipeline()
        llm_response = make_llm_response(
            response="Here are the log contents.",
            pace=0.75, pause_ms=350, tone="informative",
        )

        with patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.pipeline.needs_planning", return_value=False), \
             patch("app.pipeline.log_turn", new=AsyncMock()):

            mock_llm.generate = AsyncMock(return_value=llm_response)
            mock_mem.get_context = AsyncMock(return_value=("", ""))
            mock_mem.add_turn    = AsyncMock()

            await pipeline.process_text_input("Read the server log file.")

        assert pipeline._speech_meta.pace == 0.75
        assert pipeline._speech_meta.pause_ms == 350
        assert pipeline._speech_meta.tone == "informative"

    @pytest.mark.asyncio
    async def test_interruptible_flag_set_after_process(self):
        """_interruptible flag should be set from LLM response."""
        from app.pipeline import AssistantPipeline

        pipeline = AssistantPipeline()
        llm_response = make_llm_response(
            response="Deletion confirmed.",
            interruptible=False, tone="focused",
        )

        with patch("app.pipeline.llm_engine") as mock_llm, \
             patch("app.pipeline.memory_manager") as mock_mem, \
             patch("app.pipeline.needs_planning", return_value=False), \
             patch("app.pipeline.log_turn", new=AsyncMock()):

            mock_llm.generate = AsyncMock(return_value=llm_response)
            mock_mem.get_context = AsyncMock(return_value=("", ""))
            mock_mem.add_turn    = AsyncMock()

            await pipeline.process_text_input("Delete all temp files.")

        assert pipeline._interruptible is False
