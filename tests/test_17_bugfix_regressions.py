"""
Regression tests for the bugs fixed in the whole-codebase review.

Each test names the failure it guards against. All offline: no models, no
network, temp directories only.
"""
import asyncio
import json
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ─── Long-term memory: FTS5 external-content delete order ─────────────────────

@pytest.mark.asyncio
async def test_memory_delete_leaves_no_stale_fts_row(tmp_path):
    """Deleting the content row before the FTS row left a phantom index entry."""
    from app.memory.manager import LongTermMemory

    mem = LongTermMemory(db_path=str(tmp_path / "m.db"))
    await mem.initialize()
    mid = await mem.save("note", "apple pie recipe")
    assert await mem.recall("apple")
    await mem.delete(mid)

    conn = sqlite3.connect(str(tmp_path / "m.db"))
    stale = conn.execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'apple'").fetchall()
    assert stale == []


# ─── Pipeline: untrusted LLM fields must not crash the turn ───────────────────

def test_parse_llm_json_coerces_untrusted_field_types():
    """One normalisation point: "pace": "fast" / "confidence": "high" used to crash the turn."""
    from app.pipeline import _parse_llm_json

    data = _parse_llm_json(json.dumps({
        "response": "hi", "confidence": "high", "interruptible": "false",
        "speech": {"pace": "fast", "clause_pause_ms": -5, "tone": None},
    }))
    assert data["confidence"] == 0.9 and data["interruptible"] is False
    assert data["speech"] == {"pace": 1.0, "clause_pause_ms": 0, "tone": "neutral"}

    data = _parse_llm_json(json.dumps({"response": "x", "speech": "quick", "confidence": 0.3}))
    assert data["speech"]["pace"] == 1.0 and data["confidence"] == 0.3

    clamped = _parse_llm_json(json.dumps({"response": "x", "speech": {"pace": 9, "clause_pause_ms": 99999}}))
    assert clamped["speech"]["pace"] == 2.0 and clamped["speech"]["clause_pause_ms"] == 2000


# ─── Goal tracker ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_goal_records_step_status_and_completes(tmp_path, monkeypatch):
    """Goals used to be persisted with every step pending and never advanced."""
    import app.planning.goal_tracker as gt

    monkeypatch.setattr(gt, "_DB_PATH", tmp_path / "goals.db")
    tracker = gt.GoalTracker()
    monkeypatch.setattr(tracker, "_generate_title", AsyncMock(return_value="Title"))

    done = await tracker.create_goal("do three things", [
        {"description": "a", "tool": "x", "status": "done"},
        {"description": "b", "tool": "y", "status": "done"},
        {"description": "c", "tool": "z", "status": "done"},
    ])
    assert done.status == "completed" and done.completed_steps == 3
    assert await tracker.get_active_goals() == []

    partial = await tracker.create_goal("do three things", [
        {"description": "a", "tool": "x", "status": "done"},
        {"description": "b", "tool": "y"},
        {"description": "c", "tool": "z"},
    ])
    assert partial.status == "in_progress"
    assert partial.next_step.description == "b"


@pytest.mark.parametrize("tool_results, expected", [
    ([{"tool": "a", "status": "ok"}, {"tool": "b", "status": "ok"}], "done"),
    ([{"tool": "a", "status": "ok"}, {"tool": "b", "status": "error"}], "pending"),
    ([], "pending"),
])
def test_pipeline_persists_goal_after_turn(monkeypatch, tool_results, expected):
    """Goal is recorded after the plan ran, titled from the plan, no extra LLM call."""
    from app.pipeline import AssistantPipeline
    from app.planner import Plan

    p = AssistantPipeline()
    plan = Plan(goal="Tidy the vault", steps=["s1", "s2", "s3"])
    captured = {}

    async def fake_create(user_text, steps, title=""):
        captured.update(steps=steps, title=title)

    monkeypatch.setattr("app.pipeline.goal_tracker.create_goal", fake_create)

    asyncio.run(_run_persist(p, plan, tool_results))
    assert captured["title"] == "Tidy the vault"
    assert [s["status"] for s in captured["steps"]] == [expected] * 3

    captured.clear()
    asyncio.run(_run_persist(p, None, []))                         # no plan → nothing persisted
    asyncio.run(_run_persist(p, Plan(goal="g", steps=["a"]), []))  # < 3 steps → nothing persisted
    assert captured == {}


async def _run_persist(p, plan, tool_results):
    p._persist_goal("q", plan, tool_results)
    await asyncio.sleep(0)   # let the background task run


# ─── Project switch must not hijack tool requests ────────────────────────────

@pytest.mark.parametrize("text", ["open note Meeting", "load skill capture", "open the file a.txt"])
def test_tool_requests_are_not_project_switches(text):
    from app.memory.projects import extract_project_name_from_query
    assert extract_project_name_from_query(text) is None


def test_active_project_context_reaches_system_prompt(monkeypatch):
    """project_switch loaded context that build_system_prompt never injected."""
    import app.memory.projects as pr
    from app.prompts.templates import build_system_prompt

    ctx = pr.ProjectContext(name="Thesis", slug="thesis", description="d", vault_folder="thesis")
    monkeypatch.setattr(pr, "_active_project", ctx)
    with patch("app.memory.hot_memory.hot_memory_for_prompt", return_value=""), \
         patch("app.memory.skills.skills_list", return_value=""):
        assert "ACTIVE PROJECT: Thesis" in build_system_prompt()


# ─── Coordinator runs each step once and leaves its params alone ──────────────

@pytest.mark.asyncio
async def test_coordinator_step_runs_once_even_when_critic_unsatisfied():
    """The old retry injected `_retry_context` into tool kwargs (TypeError) and
    re-ran identical calls; now a step runs exactly once."""
    import app.agents.coordinator as co

    step = co.ToolStep(description="d", tool="t", params={"a": 1})
    run = AsyncMock(return_value=co.StepResult("t", "ok", "meh"))
    with patch.object(co.tool_executor, "run", run), \
         patch.object(co.critic, "evaluate", AsyncMock(return_value=co.CriticResult(False, "no", "try"))):
        out = await co.Coordinator()._execute_step(step, co.CoordinatorContext("q", None))
    assert out.result == "meh" and run.await_count == 1
    assert step.params == {"a": 1}


# ─── Obsidian: LLM-supplied folder names cannot leave the vault ───────────────

@pytest.mark.asyncio
async def test_obsidian_folder_traversal_is_blocked(tmp_vault, monkeypatch):
    import app.tools.obsidian as obs

    monkeypatch.setattr(obs, "VAULT_PATH", tmp_vault)
    monkeypatch.setattr(obs, "_index_note", AsyncMock())
    monkeypatch.setattr(obs.settings, "kg_auto_extract", False)

    with pytest.raises(PermissionError):
        await obs.obsidian_create_note("Escape", "x", folder="../outside")
    assert not (tmp_vault.parent / "outside").exists()

    with pytest.raises(PermissionError):
        await obs.obsidian_list_vault(folder="../../")
    with pytest.raises(PermissionError):
        await obs.obsidian_get_project_context("../../..")

    # Normal folders still work.
    assert "created" in (await obs.obsidian_create_note("Ok", "x", folder="AI Notes")).lower()


# ─── RAG: remove_document resolves titles with either path separator ─────────

@pytest.mark.asyncio
async def test_remove_document_matches_windows_paths(tmp_path, monkeypatch):
    import app.tools.rag.ingestor as ing

    monkeypatch.setattr(ing.settings, "obsidian_vector_db", tmp_path / "v.db")
    conn = ing._get_doc_db()
    conn.execute(
        "INSERT INTO doc_vectors (source_path, title, chunk_index, content, content_hash, embedding) "
        "VALUES (?, ?, 0, 'c', 'h', X'00')",
        (r"C:\Users\me\docs\contract.pdf", "contract"),
    )
    conn.commit(); conn.close()

    assert "Removed 1 chunk" in await ing.remove_document("contract.pdf")


# ─── SpeechCollector re-frames any chunk size to what the VAD accepts ─────────

def test_speech_collector_reframes_to_vad_frames():
    """Silero VAD requires exactly 512 samples @ 16 kHz; browsers (/ws/audio)
    send whatever their resampler produced. The collector re-frames for every caller."""
    from app.audio.vad import SpeechCollector

    assert SpeechCollector.FRAME_SAMPLES == 512
    vad = MagicMock()
    vad.is_speech.side_effect = lambda frame: (seen.append(len(frame)), 0.0)[1]
    seen: list[int] = []
    collector = SpeechCollector(vad)

    for size in (170, 170, 170, 900, 30):          # what a resampled ScriptProcessor emits
        assert collector.push(np.ones(size, dtype=np.float32)) is None
    assert seen == [512, 512]
    assert len(collector._pending) == (170 * 3 + 900 + 30) - 1024
    collector.reset()
    assert len(collector._pending) == 0


# ─── MCP stdio: replies are matched by id, notifications are skipped ─────────

@pytest.mark.asyncio
async def test_mcp_stdio_skips_notifications_and_detects_eof():
    from app.tools.mcp_client import MCPRegistry, MCPServer

    class FakeStdout:
        def __init__(self, lines): self.lines = list(lines)
        async def readline(self): return self.lines.pop(0) if self.lines else b""

    class FakeStdin:
        def write(self, b): pass
        async def drain(self): pass

    srv = MCPServer(name="s", transport="stdio", command="x")
    srv._stdin = FakeStdin()
    srv._stdout = FakeStdout([
        b'{"jsonrpc":"2.0","method":"notifications/message","params":{}}\n',
        b'not json\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n',
    ])
    assert await MCPRegistry()._stdio_request(srv, "tools/list", {}) == {"tools": []}

    srv._stdout = FakeStdout([])
    with pytest.raises(RuntimeError, match="closed stdout"):
        await MCPRegistry()._stdio_request(srv, "tools/list", {})


# ─── Test isolation: nothing in the suite touches the real data/ directory ────

def test_settings_point_at_throwaway_data_dir():
    from app.config import settings
    real = Path(__file__).resolve().parent.parent / "data"
    for name in ("data_dir", "memory_db", "kg_db", "projects_db", "obsidian_vector_db", "workspace_dir"):
        assert not Path(getattr(settings, name)).resolve().is_relative_to(real), name


# ─── API: browser requests from other sites must not reach the tools ─────────

def test_api_refuses_foreign_origins_and_rebound_hosts():
    """CORS never covered /ws/audio, and a DNS-rebound page passes CORS as same-origin."""
    from starlette.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from app.router import app

    client = TestClient(app, base_url="http://127.0.0.1:8000")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/audio", headers={"Origin": "https://evil.example"}):
            pass
    assert client.get("/cache/stats", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/cache/stats", headers={"Host": "evil.example:8000"}).status_code == 403
    assert client.get("/cache/stats").status_code == 200                                    # curl, scripts
    assert client.get("/cache/stats", headers={"Origin": "tauri://localhost"}).status_code == 200


# ─── Pipeline: escapes small models emit that JSON doesn't allow ─────────────

@pytest.mark.parametrize("chunk", [1, 4, 100])
def test_response_stream_survives_invalid_escapes(chunk):
    """`It\\'s` crashed the whole turn with JSONDecodeError; `C:\\users` stalled the stream."""
    from app.pipeline import _ResponseFieldStream, _parse_llm_json

    raw = r'{"tool": null, "response": "It\'s in C:\users\me, ok"}'
    ex = _ResponseFieldStream()
    assert "".join(ex.feed(raw[i:i + chunk]) for i in range(0, len(raw), chunk)) == r"It's in C:\users\me, ok"
    assert _parse_llm_json(raw[:-2])["response"] == r"It's in C:\users\me, ok"      # cut-off reply, salvage path


# ─── Pipeline: cached answers must not cross conversations ───────────────────

def test_cache_key_includes_the_conversation(monkeypatch):
    """A cached "why?" about Postgres was replayed as the answer to "why?" about Python."""
    import app.pipeline as pl
    from app.memory.manager import MemoryManager

    mm = MemoryManager()
    monkeypatch.setattr(pl, "memory_manager", mm)
    mm.short_term.add_user("Should I use Postgres?"); mm.short_term.add_assistant("Yes.")
    after_postgres = pl.AssistantPipeline._cache_key("why?")
    mm.short_term.clear()
    mm.short_term.add_user("Is Python slow?"); mm.short_term.add_assistant("Sometimes.")
    assert pl.AssistantPipeline._cache_key("why?") != after_postgres


# ─── Voice commands: whole utterances only ───────────────────────────────────

def test_voice_commands_match_whole_utterances(monkeypatch):
    """"search the web for X" flipped a setting and dropped the question; "Goodbye." (STT punctuation) was ignored."""
    from app.config import settings
    from app.pipeline import AssistantPipeline

    monkeypatch.setattr(settings, "web_search_enabled", False)
    p = AssistantPipeline()
    assert p._voice_command("Search the web for cheap flights to Lahore") is None
    assert settings.web_search_enabled is False
    assert p._voice_command("Enable web search.") and settings.web_search_enabled is True
    assert p._voice_command("Goodbye.") == "Goodbye!" and p._active is False


# ─── llama.cpp: one inference at a time, even when a caller walks away ──────

@pytest.mark.asyncio
async def test_llamacpp_abandoned_stream_holds_lock_until_inference_stops(monkeypatch):
    """Cancelling a stream released the lock while the thread kept generating → concurrent llama.cpp calls."""
    import threading
    import time
    from app.llm import engine as eng

    monkeypatch.setattr(eng.settings, "LLM_BACKEND", "llamacpp")
    running, peak, lock = [0], [0], threading.Lock()

    class FakeLlama:
        def create_chat_completion(self, **_):
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            try:
                for _ in range(30):
                    time.sleep(0.01)
                    yield {"choices": [{"delta": {"content": "t"}}]}
            finally:
                with lock:
                    running[0] -= 1

    e = eng.LLMEngine()
    e._llm = FakeLlama()
    msgs = [{"role": "user", "content": "hi"}]
    stream = e.generate_stream(msgs, "sys")
    await stream.__anext__()
    await stream.aclose()                      # caller walks away mid-answer
    assert await e.generate(msgs, "sys")
    assert peak[0] == 1


# ─── Ollama down / model missing: say how to fix it ─────────────────────────

def test_ollama_problem_names_the_fix(monkeypatch):
    """/health said "loaded" and every answer was a generic apology while Ollama was down."""
    from app.llm import engine as eng

    e = eng.LLMEngine()
    e._base_url, e._model = "http://127.0.0.1:9", "qwen2.5:3b"          # nothing listens on port 9
    assert "ollama serve" in e.ollama_problem()

    tags = MagicMock(json=lambda: {"models": [{"name": "llama3:latest"}, {"name": "qwen2.5:latest"}]})
    monkeypatch.setattr(eng.httpx, "get", lambda *a, **k: tags)
    assert "ollama pull qwen2.5:3b" in e.ollama_problem()
    e._model = "qwen2.5"                                                   # untagged → ":latest"
    assert e.ollama_problem() is None


@pytest.mark.asyncio
async def test_ollama_unreachable_raises_instead_of_returning_nothing(monkeypatch):
    from app.llm import engine as eng

    monkeypatch.setattr(eng.settings, "LLM_BACKEND", "ollama")
    e = eng.LLMEngine()
    e._base_url, e._ollama_available = "http://127.0.0.1:9", True
    with pytest.raises(eng.LLMUnavailableError, match="ollama serve"):
        async for _ in e.generate_stream([{"role": "user", "content": "hi"}], "sys"):
            pass
    assert await e.generate([{"role": "user", "content": "hi"}], "sys") == ""   # callers of generate() still never see it


@pytest.mark.asyncio
async def test_turn_answers_with_the_setup_fix_and_does_not_cache_it(pipeline_mocks):
    import app.pipeline as pl
    from app.llm.engine import LLMUnavailableError

    async def down(*a, **k):
        raise LLMUnavailableError("I can't reach Ollama at http://localhost:11434. Start it with: ollama serve")
        yield

    pipeline_mocks.generate_stream = down
    result = await pl.AssistantPipeline().run_turn("what's in my notes about rust?")
    assert "ollama serve" in result.response
    pl.response_cache.set.assert_not_awaited()
    pl.memory_manager.add_turn.assert_not_awaited()


# ─── .env.example must start as-is ───────────────────────────────────────────

def test_env_example_loads_as_is(tmp_path, monkeypatch):
    """`cp .env.example .env` crashed startup: blank AUDIO_INPUT_DEVICE= isn't an int."""
    from app.config import Settings, BASE_DIR

    (tmp_path / ".env").write_text((BASE_DIR / ".env.example").read_text())
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.audio_input_device is None and s.audio_output_device is None
    assert "~" not in str(s.obsidian_vault_path)
