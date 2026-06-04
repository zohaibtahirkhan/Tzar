"""
Shared pytest fixtures for the local voice assistant test suite.

All LLM calls are mocked — tests run entirely offline with no models loaded.
The mock LLM is controlled per-test via the `llm_response` fixture parameter.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Make project root importable ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Prevent real model loading at import time ─────────────────────────────────
os.environ.setdefault("LLM_MODEL_PATH", "models/fake.gguf")
os.environ.setdefault("OBSIDIAN_VAULT_PATH", "/tmp/fake_vault")


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
        "response": response,
        "confidence": confidence,
        "interruptible": interruptible,
        "speech": {
            "pace": pace,
            "clause_pause_ms": pause_ms,
            "tone": tone,
        },
    })


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
def mock_llm():
    """
    Mock LLM engine. Set mock_llm.generate.return_value to control responses.
    Supports both .generate() and .generate_stream() (returns full string).
    """
    engine = MagicMock()
    engine.generate = AsyncMock(return_value=make_llm_response())
    engine.generate_stream = AsyncMock(return_value=make_llm_response())
    return engine


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