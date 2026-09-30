"""
Pipeline tests — the single turn implementation.

Covers what the old two-path pipeline could not be tested for: real token
streaming of the JSON "response" field, the bounded tool loop keeping its
context, one result object per turn, clause-by-clause speech, and barge-in in
the voice loop. All offline: the LLM, tools, microphone, STT and TTS are scripted.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from tests.conftest import make_llm_response, stream_llm_response


# ─── Incremental "response" extraction ───────────────────────────────────────

def _drain(text: str, chunk: int) -> str:
    from app.pipeline import _ResponseFieldStream
    ex = _ResponseFieldStream()
    return "".join(ex.feed(text[i:i + chunk]) for i in range(0, len(text), chunk))


@pytest.mark.parametrize("chunk", [1, 3, 7, 50])
def test_response_stream_decodes_escapes_across_chunk_boundaries(chunk):
    reply = json.dumps({"tool": None, "tool_params": None,
                        "response": 'He said "hi"\nthen café \\ done', "confidence": 0.9})
    assert _drain(reply, chunk) == 'He said "hi"\nthen café \\ done'


def test_response_stream_is_silent_for_tool_calls():
    reply = json.dumps({"tool": "doc_search", "tool_params": {"query": "x"}, "response": "Searching..."})
    assert _drain(reply, 5) == ""


def test_response_stream_is_silent_when_tool_key_is_missing_or_late():
    # No "tool" before "response": we cannot know yet whether this is a tool
    # call, so nothing is forwarded (the final text still arrives at the end).
    assert _drain('{"response": "hello", "tool": null}', 4) == ""


def test_response_stream_exposes_the_head_and_stops_at_closing_quote():
    from app.pipeline import _ResponseFieldStream
    ex = _ResponseFieldStream()
    assert ex.feed('{"tool": null, "speech": {"pace": 0.8}, "interruptible": false, "response": "ab') == "ab"
    assert ex.head == {"tool": None, "speech": {"pace": 0.8}, "interruptible": False}
    assert ex.feed('c", "confidence": 0.9}') == "c"
    assert ex.feed(' trailing garbage "response": "no"') == ""


def test_parse_llm_json_salvages_malformed_replies():
    from app.pipeline import _parse_llm_json

    cut_off = '{"tool": null, "speech": {"pace": 1.0}, "response": "I found three files'
    assert _parse_llm_json(cut_off)["response"] == "I found three files"

    dup_keys = '{"tool": "read_file", "tool_params": {"path": "a"}, "tool": "read_file", "response": ""}'
    assert _parse_llm_json(dup_keys)["tool"] == "read_file"

    prose = "Sure! Here is the answer."
    assert _parse_llm_json(prose)["response"] == prose
    assert _parse_llm_json("")["response"] == "" and _parse_llm_json("")["speech"]["pace"] == 1.0


# ─── One turn: streaming, tool loop, result ──────────────────────────────────

def _needs_tools():
    from app.intent import Capabilities
    return Capabilities(needs_tools=True)


@pytest.mark.asyncio
async def test_stream_forwards_only_the_final_answer_and_keeps_tool_context(pipeline_mocks):
    from app.pipeline import AssistantPipeline, SpeechMeta

    p = AssistantPipeline()
    pipeline_mocks.generate_stream = MagicMock(side_effect=stream_llm_response(
        make_llm_response(tool="list_directory", tool_params={"path": "."}, response="Looking..."),
        make_llm_response(tool="read_file", tool_params={"path": "a.txt"}, response="Reading..."),
        make_llm_response(response="a.txt says hello.", pace=0.85, tone="informative", interruptible=False),
        chunk_size=6,
    ))
    with patch("app.pipeline.classify", return_value=_needs_tools()), \
         patch.object(p, "_run_tool", new=AsyncMock(side_effect=[
             ({"tool": "list_directory", "status": "ok", "result": "a.txt"}, 3),
             ({"tool": "read_file", "status": "ok", "result": "hello"}, 2),
         ])):
        events = [e async for e in p.stream("read the files")]

    kinds = [e.kind for e in events]
    assert kinds[:2] == ["status", "status"]                       # one per tool, before any text
    assert [e.text for e in events if e.kind == "status"] == ["[Running list_directory...]\n", "[Running read_file...]\n"]
    meta = [e for e in events if e.kind == "meta"]
    assert len(meta) == 1 and meta[0].speech == SpeechMeta(0.85, 120, "informative") and meta[0].interruptible is False
    assert kinds.index("meta") < kinds.index("text")                # how to speak arrives before what to speak
    text = [e.text for e in events if e.kind == "text"]
    assert len(text) > 1 and "".join(text) == "a.txt says hello."    # streamed; tool replies never leaked

    result = events[-1].result
    assert result.response == "a.txt says hello." == "".join(text)   # what was streamed is what is recorded
    assert [r["tool"] for r in result.tool_results] == ["list_directory", "read_file"]
    assert result.speech == SpeechMeta(0.85, 120, "informative") and result.interruptible is False

    # Each follow-up carries the whole exchange so far, not just the last result.
    calls = [c.args[0] for c in pipeline_mocks.generate_stream.call_args_list]
    assert [len(m) for m in calls] == [1, 3, 5]
    assert "a.txt" in calls[1][-1]["content"] and "hello" in calls[2][-1]["content"]


@pytest.mark.asyncio
async def test_attribution_is_streamed_as_its_own_piece(pipeline_mocks):
    from app.pipeline import AssistantPipeline
    from app.tools.rag.citations import Source, record_sources

    async def fake_search(query, top_k=5):
        record_sources([Source("Lakehouse Notes", "AI/Lakehouse Notes.md", "note")])
        return "[[Lakehouse Notes]]"

    p = AssistantPipeline()
    pipeline_mocks.generate_stream = stream_llm_response(make_llm_response(response="Genie needs Unity Catalog."))
    with patch("app.tools.rag.unified_search", new=fake_search):
        events = [e async for e in p.stream("what did I write about Genie in my notes?")]

    text = [e.text for e in events if e.kind == "text"]
    assert text[-1] == " That's from Lakehouse Notes."
    assert "".join(text) == events[-1].result.response == "Genie needs Unity Catalog. That's from Lakehouse Notes."


@pytest.mark.asyncio
async def test_repeated_identical_tool_call_is_cut_short(pipeline_mocks):
    from app.pipeline import AssistantPipeline, FALLBACK_RESPONSE

    p = AssistantPipeline()
    same = make_llm_response(tool="kg_summary", tool_params={}, response="")
    pipeline_mocks.generate_stream = stream_llm_response(same, same, same)
    with patch.object(p, "_run_tool", new=AsyncMock(return_value=({"tool": "kg_summary", "status": "ok", "result": "x"}, 1))) as run_tool:
        result = await p.run_turn("graph stats")

    assert run_tool.await_count == 1                      # second identical call was refused
    assert result.response == FALLBACK_RESPONSE           # no spoken text came back
    assert len(result.tool_results) == 1


@pytest.mark.asyncio
async def test_zero_llm_answers_are_archived_once_and_carry_default_speech(pipeline_mocks):
    from app.pipeline import AssistantPipeline, SpeechMeta
    import app.pipeline as pl

    p = AssistantPipeline()
    pl.response_cache.get = AsyncMock(return_value="Hello!")
    result = await p.run_turn("hi")
    assert (result.response, result.speech, result.tool_results) == ("Hello!", SpeechMeta(), [])
    pl.memory_manager.add_turn.assert_awaited_once_with("hi", "Hello!")
    pipeline_mocks.generate_stream = MagicMock()
    pipeline_mocks.generate_stream.assert_not_called()


# ─── Speaking as the answer streams ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_respond_aloud_speaks_clauses_as_they_close(pipeline_mocks):
    from app.pipeline import AssistantPipeline, SpeechMeta

    p = AssistantPipeline()
    pipeline_mocks.generate_stream = stream_llm_response(make_llm_response(
        response="I found three files in the workspace. The largest one is notes.txt, which you edited yesterday.",
        pace=0.8, tone="informative", interruptible=False,
    ), chunk_size=5)
    played: list[tuple[str, SpeechMeta, bool]] = []

    async def fake_play(text, meta, interruptible):
        played.append((text, meta, interruptible))

    with patch.object(p, "_play", new=fake_play):
        await p._respond_aloud("what's in my workspace")

    assert [t for t, _, _ in played] == [
        "I found three files in the workspace.",
        "The largest one is notes.txt,",
        "which you edited yesterday.",
    ]
    assert all(m == SpeechMeta(0.8, 120, "informative") and i is False for _, m, i in played)


# ─── Voice loop: barge-in ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_voice_loop_wake_word_cancels_the_answer_in_progress():
    """
    Script: wake → "hello" → assistant starts a slow answer → wake word again
    while it talks → answer is cut → "goodbye" → loop ends. The old loop
    awaited the whole turn inline, so the second wake word was never heard.
    """
    import app.pipeline as pl
    from app.pipeline import AssistantPipeline, TurnEvent, TurnResult

    p = AssistantPipeline()
    frames = np.ones(512, dtype=np.float32)

    async def chunks():
        for step in ["wake", "speech", "wake", "speech"]:
            yield step
            for _ in range(3):
                await asyncio.sleep(0)                  # let the response task run a bit

    class FakeCollector:
        def __init__(self, vad): pass
        def reset(self): pass
        def push(self, chunk): return frames if chunk == "speech" else None

    async def slow_stream(user_text):                   # a turn that never finishes on its own
        yield TurnEvent("meta")
        yield TurnEvent("text", "Hi there, I found your files. The largest one")
        await asyncio.sleep(10)
        yield TurnEvent("done", result=TurnResult("Hi there, I found your files. The largest one is big."))

    spoken: list[str] = []

    async def fake_play(text, meta, interruptible):
        spoken.append(text)
        p._speaking, p._interruptible = True, interruptible
        try:
            while not p._interrupted and text != "Goodbye!":   # the answer talks until cut; goodbye finishes
                await asyncio.sleep(0)
        finally:
            p._speaking = False

    with patch.object(pl, "microphone") as mic, \
         patch.object(pl, "wake_word_detector") as wake, \
         patch.object(pl, "SpeechCollector", FakeCollector), \
         patch.object(pl, "stt_engine") as stt, \
         patch.object(pl, "audio_player") as player, \
         patch.object(p, "stream", new=slow_stream), \
         patch.object(p, "_play", new=fake_play):
        mic.chunks = chunks
        wake.check_chunk = lambda chunk: 1.0 if chunk == "wake" else 0.0
        wake.detected = lambda conf: conf >= 0.5
        stt.transcribe_async = AsyncMock(side_effect=["hello", "goodbye"])
        await asyncio.wait_for(p.run_voice_loop(), timeout=5)

    assert spoken == ["Hi there, I found your files.", "Goodbye!"]   # first clause started, then cut
    player.stop.assert_called_once()                    # the second wake word stopped the audio
    assert p._active is False
