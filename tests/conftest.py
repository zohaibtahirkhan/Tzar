"""
Shared pytest fixtures for the local voice assistant test suite.

All LLM calls are mocked — tests run entirely offline with no models loaded.
The mock LLM is controlled per-test via the `llm_response` fixture parameter.
"""
import asyncio
import atexit
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Prevent real model loading at import time ─────────────────────────────────
os.environ.setdefault("LLM_MODEL_PATH", "models/fake.gguf")

# ── Keep every DB / vault / workspace write out of the real data/ directory ───
# Modules capture these paths from settings at import time, so they must be set
# before anything under app/ is imported. Per-test fixtures still narrow further.
_SESSION_DATA = Path(tempfile.mkdtemp(prefix="tzar-tests-"))
atexit.register(shutil.rmtree, _SESSION_DATA, ignore_errors=True)
for _key, _sub in {
    "DATA_DIR":            "data",
    "LOG_DIR":             "data/logs",
    "WORKSPACE_DIR":       "workspace",
    "MEMORY_DB":           "data/memory.db",
    "KG_DB":               "data/knowledge_graph.db",
    "PROJECTS_DB":         "data/projects.db",
    "OBSIDIAN_VECTOR_DB":  "data/obsidian_vectors.db",
    "OBSIDIAN_VAULT_PATH": "vault",
}.items():
    os.environ.setdefault(_key, str(_SESSION_DATA / _sub))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_llm_response(
    tool: str | None = None,
    tool_params: dict | None = None,
    response: str = "Done.",
    confidence: float = 0.9,
    interruptible: bool = True,
    pace: float = 1.0,
    pause_ms: int = 120,
    tone: str = "neutral",
) -> str:
    """Build a valid JSON string the way the real LLM would return it."""
    return json.dumps({
        "tool": tool,
        "tool_params": tool_params,
        "speech": {
            "pace": pace,
            "clause_pause_ms": pause_ms,
            "tone": tone,
        },
        "confidence": confidence,
        "interruptible": interruptible,
        "response": response,          # last, as the prompt asks — lets TTS start early
    })


def stream_llm_response(*replies: str, chunk_size: int = 7):
    """
    Stand-in for llm_engine.generate_stream: each call serves the next reply
    in small pieces, the way tokens arrive from a real model. Wrap it in
    MagicMock(side_effect=...) to also record the messages each call received.
    """
    queue = list(replies)

    async def _gen(*args, **kwargs):
        text = queue.pop(0)
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]
    return _gen


def stub_memory(mock_mem) -> None:
    """Quiet memory manager: no context, no writes."""
    mock_mem.get_context = AsyncMock(return_value=("", ""))
    mock_mem.add_turn = AsyncMock()


def make_curator_response(action: str = "store", target: str = "user",
                          content: str = "", reason: str = "") -> str:
    if action == "reject":
        return json.dumps({"action": "reject", "reason": reason or "not worth storing"})
    if action == "consolidate":
        return json.dumps({"action": "consolidate", "target": target,
                           "old_substring": "old", "new_content": content})
    return json.dumps({"action": "store", "target": target, "content": content})


def make_plan_response(
    goal: str = "",
    steps: list | None = None,
    tools: list | None = None,
    complexity: str = "low",
    can_answer_directly: bool = False,
) -> str:
    return json.dumps({
        "goal": goal,
        "complexity": complexity,
        "steps": steps or [],
        "required_tools": tools or [],
        "can_answer_directly": can_answer_directly,
    })


# ─── Core fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Isolated data directory for each test."""
    (tmp_path / "skills").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def tmp_vault(tmp_path):
    """Isolated Obsidian vault directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def pipeline_mocks():
    """
    Everything a pipeline turn touches outside its own logic, mocked: yields the
    LLM mock so a test can script replies with
    `mock_llm.generate_stream = stream_llm_response(reply, ...)`.
    """
    with patch("app.pipeline.llm_engine") as mock_llm, \
         patch("app.pipeline.memory_manager") as mock_mem, \
         patch("app.pipeline.log_turn", new=AsyncMock()), \
         patch("app.pipeline.build_system_prompt", return_value="SYSTEM"), \
         patch("app.pipeline.response_cache") as cache:
        stub_memory(mock_mem)
        cache.get = AsyncMock(return_value=None)
        cache.set = AsyncMock()
        cache.clear = AsyncMock()
        mock_llm.generate_stream = stream_llm_response(make_llm_response())
        yield mock_llm


@pytest.fixture
def tool_router():
    """ToolRouter with a fake memory manager — no DB, no real tools."""
    from app.tools.router import ToolRouter

    fake_memory = MagicMock()
    fake_memory.long_term.save = AsyncMock(return_value=1)
    fake_memory.long_term.recall = AsyncMock(return_value=[])

    return ToolRouter(memory_manager=fake_memory)


@pytest.fixture
def event_loop():
    """Single event loop for all async tests in a session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()