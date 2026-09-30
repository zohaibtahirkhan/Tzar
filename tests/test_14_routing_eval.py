"""
Routing eval — runs tests/eval/routing_cases.json against every zero-LLM
routing decision the pipeline makes before a model is called.

Routing here is regex-based, so it is both deterministic (perfectly evaluable
offline) and fragile (a pattern tweak can silently re-route whole classes of
input). This harness is what makes such a tweak visible.

Cases carrying "known_gap" describe correct behaviour the classifier does not
yet deliver. They run as xfail(strict=True): still failing is fine, but the
moment the classifier is fixed they XPASS — which strict mode turns into a hard
failure, so the gap marker gets removed rather than rotting.
"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

CASES_PATH = Path(__file__).resolve().parent / "eval" / "routing_cases.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _params(section: str):
    """Turn a fixture section into pytest params, honouring known_gap → xfail."""
    out = []
    for case in CASES[section]:
        marks = ()
        if gap := case.get("known_gap"):
            marks = (pytest.mark.xfail(strict=True, reason=f"known gap: {gap}"),)
        out.append(pytest.param(case, id=case["input"], marks=marks))
    return out


# ─── Capabilities ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", _params("capabilities"))
def test_capabilities(case):
    from app.intent import classify

    caps = classify(case["input"])
    actual = {**caps.as_dict(), "is_complex": caps.is_complex()}
    mismatches = {
        key: (expected, actual.get(key))
        for key, expected in case["expect"].items()
        if actual.get(key) != expected
    }
    assert not mismatches, (
        f"{case['input']!r}: expected {case['expect']}, got "
        f"{ {k: actual[k] for k in case['expect']} } — mismatches {mismatches}"
    )


# ─── Morning briefing fast path ───────────────────────────────────────────────

@pytest.mark.parametrize("case", _params("morning_greeting_fast_path"))
def test_morning_greeting_fast_path(case):
    from app.pipeline import _is_morning_greeting

    assert _is_morning_greeting(case["input"]) is case["expect"], case.get("_why", "")


# ─── Memory-save short-circuit ────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("case", _params("memory_save_shortcircuit"))
async def test_memory_save_shortcircuit(case):
    """
    Exercises the real _try_memory_save() gate with the DB write and turn
    archiving mocked out, so the decision itself is what's under test.
    """
    from app.intent import classify
    from app.pipeline import AssistantPipeline

    pipeline = AssistantPipeline()
    with patch("app.pipeline.memory_manager") as mem, \
         patch.object(pipeline, "_archive_turn", new=AsyncMock()), \
         patch("app.pipeline.response_cache") as cache:
        mem.long_term.save = AsyncMock(return_value=1)
        cache.clear = AsyncMock()

        saved = await pipeline._try_memory_save(case["input"], classify(case["input"])) is not None

    assert saved is case["expect"], case.get("_why", "")


# ─── Project continuity ───────────────────────────────────────────────────────

@pytest.mark.parametrize("case", _params("project_switch_detection"))
def test_project_switch_detection(case):
    from app.memory.projects import extract_project_name_from_query

    assert extract_project_name_from_query(case["input"]) == case["expect"], case.get("_why", "")


# ─── Goal continuation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", _params("goal_continuation"))
def test_goal_continuation(case):
    from app.planning.goal_tracker import goal_tracker

    assert goal_tracker.is_continuation(case["input"]) is case["expect"]


# ─── Fixture hygiene ──────────────────────────────────────────────────────────

def test_fixture_has_no_duplicate_inputs():
    """Two cases for one input with different expectations would silently
    contradict each other."""
    for section, cases in CASES.items():
        if section.startswith("_"):
            continue
        inputs = [c["input"] for c in cases]
        dupes = sorted({i for i in inputs if inputs.count(i) > 1})
        assert not dupes, f"{section}: duplicate inputs {dupes}"
