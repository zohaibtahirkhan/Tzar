"""
Tests for Memory Scoring, Project Continuity, Skill Auto-Learning.

Run: pytest tests/test_10_scoring_projects_skills.py -v

100% offline — no LLM, no network, no Obsidian vault required.
"""

import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Scoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestImportanceEstimator:
    def setup_method(self):
        from app.memory.scoring import estimate_importance
        self._est = estimate_importance

    def test_preference_is_high_importance(self):
        assert self._est("I prefer dark mode in all my apps") >= 0.8

    def test_name_is_high_importance(self):
        assert self._est("my name is Zohaib") >= 0.8

    def test_project_goal_is_high(self):
        assert self._est("my project deadline is next Friday") >= 0.8

    def test_casual_food_is_low(self):
        assert self._est("today I had pizza for lunch") <= 0.2

    def test_weather_is_low(self):
        assert self._est("the weather today is rainy") <= 0.2

    def test_neutral_content_is_medium(self):
        score = self._est("I was reading about transformers last night")
        assert 0.2 < score < 0.8

    def test_empty_string(self):
        score = self._est("")
        assert 0.0 <= score <= 1.0


class TestConfidenceEstimator:
    def setup_method(self):
        from app.memory.scoring import estimate_confidence
        self._est = estimate_confidence

    def test_assertion_is_high_confidence(self):
        assert self._est("my name is Zohaib") >= 0.9

    def test_preference_is_high_confidence(self):
        assert self._est("I prefer using dark mode always") >= 0.9

    def test_hedged_statement_is_lower(self):
        assert self._est("I think maybe I prefer dark mode") < 0.8

    def test_maybe_lowers_confidence(self):
        assert self._est("maybe I should use Rust for this project") < 0.8

    def test_default_confidence_is_mid(self):
        c = self._est("I went to the gym this morning")
        assert 0.6 <= c <= 0.95


class TestRecencyScore:
    def setup_method(self):
        from app.memory.scoring import recency_score
        self._score = recency_score

    def test_just_accessed_is_near_one(self):
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        assert self._score(now) > 0.99

    def test_two_weeks_ago_is_half(self):
        from datetime import datetime, timedelta
        two_weeks_ago = (datetime.utcnow() - timedelta(days=14)).isoformat()
        score = self._score(two_weeks_ago)
        assert 0.45 <= score <= 0.55   # should be ~0.5

    def test_one_month_ago_is_lower(self):
        from datetime import datetime, timedelta
        old = (datetime.utcnow() - timedelta(days=30)).isoformat()
        assert self._score(old) < 0.5

    def test_invalid_date_returns_default(self):
        score = self._score("not-a-date")
        assert 0.0 <= score <= 1.0   # should not crash

    def test_score_is_between_zero_and_one(self):
        from datetime import datetime, timedelta
        for days in [0, 7, 14, 30, 90, 365]:
            dt = (datetime.utcnow() - timedelta(days=days)).isoformat()
            s = self._score(dt)
            assert 0.0 <= s <= 1.0


class TestFrequencyScore:
    def setup_method(self):
        from app.memory.scoring import frequency_score
        self._score = frequency_score

    def test_zero_frequency_is_zero(self):
        assert self._score(0) == 0.0

    def test_high_frequency_is_high(self):
        # With N_MAX=50: freq=50 → 1.0, freq=10 → ~0.6
        assert self._score(50) >= 0.99
        assert self._score(10) >= 0.5

    def test_monotonically_increasing(self):
        scores = [self._score(n) for n in range(11)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_always_between_zero_and_one(self):
        for n in [0, 1, 3, 5, 10, 100]:
            assert 0.0 <= self._score(n) <= 1.0


class TestCompositeScore:
    def setup_method(self):
        from app.memory.scoring import composite_score
        self._score = composite_score

    def test_all_max_is_near_one(self):
        score = self._score(importance=1.0, recency=1.0, confidence=1.0, frequency=10)
        assert score >= 0.9

    def test_all_zero_is_zero(self):
        score = self._score(importance=0.0, recency=0.0, confidence=0.0, frequency=0)
        assert score == 0.0

    def test_high_importance_dominates(self):
        high = self._score(importance=1.0, recency=0.0, confidence=0.0, frequency=0)
        low  = self._score(importance=0.0, recency=1.0, confidence=0.0, frequency=0)
        assert high > low   # importance weight > recency weight

    def test_result_in_valid_range(self):
        score = self._score(0.5, 0.5, 0.5, 3)
        assert 0.0 <= score <= 1.0


class TestMemoryScoreDB:
    """Test DB write/read operations with a temp DB."""

    def setup_method(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self._tmp.close()
        self._db_path = self._tmp.name

    def teardown_method(self):
        Path(self._db_path).unlink(missing_ok=True)

    def _patch_db(self):
        """Return context manager that patches settings.memory_db."""
        return patch("app.memory.scoring.settings.memory_db", Path(self._db_path))

    def test_score_memory_writes_to_db(self):
        with self._patch_db():
            from app.memory.scoring import score_memory
            result = score_memory("mem_001", "I prefer dark mode always")
            assert result["memory_id"] == "mem_001"
            assert result["importance"] >= 0.8
            assert 0.0 <= result["composite"] <= 1.0

    def test_get_memory_score_retrieves(self):
        with self._patch_db():
            from app.memory.scoring import score_memory, get_memory_score
            score_memory("mem_002", "my name is Alice")
            result = get_memory_score("mem_002")
            assert result is not None
            assert result["importance"] >= 0.8

    def test_get_memory_score_missing(self):
        with self._patch_db():
            from app.memory.scoring import get_memory_score
            result = get_memory_score("nonexistent_id_xyz")
            assert result is None

    def test_on_memory_recalled_increments_frequency(self):
        with self._patch_db():
            from app.memory.scoring import score_memory, on_memory_recalled, get_memory_score
            score_memory("mem_003", "I work at Databricks")
            on_memory_recalled("mem_003")
            on_memory_recalled("mem_003")
            result = get_memory_score("mem_003")
            assert result["frequency"] == 2

    def test_rank_memories_by_score(self):
        with self._patch_db():
            from app.memory.scoring import score_memory, rank_memories_by_score
            score_memory("high", "my name is Alice — very important")
            score_memory("low",  "today I had pizza — not important")
            ranked = rank_memories_by_score(["high", "low"])
            # High-importance memory should rank first
            assert ranked[0][1] == "high"

    def test_rank_unscored_memory_gets_default(self):
        with self._patch_db():
            from app.memory.scoring import rank_memories_by_score
            ranked = rank_memories_by_score(["unknown_mem"])
            assert len(ranked) == 1
            assert 0.0 <= ranked[0][0] <= 1.0

    def test_prune_low_score_memories_dry_run(self):
        with self._patch_db():
            from app.memory.scoring import score_memory, _sync_prune_low_score_memories
            # Write a memory that will have low score
            score_memory("low_mem", "lol the weather is nice")
            result = _sync_prune_low_score_memories(dry_run=True)
            assert isinstance(result, str)

    def test_memory_scores_summary_empty(self):
        with self._patch_db():
            from app.memory.scoring import _sync_memory_scores_summary
            result = _sync_memory_scores_summary()
            assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# Project Continuity
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlugify:
    def test_basic(self):
        from app.memory.projects import _slugify
        assert _slugify("AI Assistant") == "ai-assistant"

    def test_special_chars(self):
        from app.memory.projects import _slugify
        assert _slugify("Workers' Welfare Fund") == "workers-welfare-fund"

    def test_already_slug(self):
        from app.memory.projects import _slugify
        assert _slugify("my-project") == "my-project"

    def test_numbers_preserved(self):
        from app.memory.projects import _slugify
        assert "3" in _slugify("Phase 3 Implementation")


class TestProjectNameExtraction:
    def setup_method(self):
        from app.memory.projects import extract_project_name_from_query
        self._extract = extract_project_name_from_query

    def test_continue_the_project(self):
        result = self._extract("continue the AI assistant project")
        assert result is not None
        assert "AI assistant" in result or "assistant" in result.lower()

    def test_switch_to(self):
        result = self._extract("switch to workers welfare")
        assert result is not None
        assert "workers welfare" in result.lower() or "welfare" in result.lower()

    def test_resume_work(self):
        result = self._extract("resume the Databricks contract work")
        assert result is not None

    def test_no_match_returns_none(self):
        result = self._extract("what is the weather today?")
        assert result is None

    def test_no_match_simple_chat(self):
        result = self._extract("hello")
        assert result is None

    def test_open_project(self):
        result = self._extract("open my job search project")
        assert result is not None


@pytest.mark.asyncio
class TestProjectCRUD:
    """Test project creation, listing, switching — with temp DB."""

    def _patch_db(self, tmp_path):
        db = tmp_path / "test_projects.db"
        return patch("app.memory.projects._PROJECT_DB", db)

    async def test_create_project(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.projects import project_new
            result = await project_new("AI Assistant", description="Building Jarvis")
            assert "AI Assistant" in result
            assert "created" in result.lower()

    async def test_create_duplicate_returns_message(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.projects import project_new
            await project_new("Test Project")
            result = await project_new("Test Project")
            assert "already exists" in result.lower()

    async def test_list_empty(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.projects import project_list
            result = await project_list()
            assert "no projects" in result.lower()

    async def test_list_after_create(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.projects import project_new, project_list
            await project_new("Project Alpha")
            await project_new("Project Beta")
            result = await project_list()
            assert "Project Alpha" in result
            assert "Project Beta" in result

    async def test_archive_project(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.projects import project_new, project_archive
            await project_new("Old Project")
            result = await project_archive("Old Project")
            assert "updated" in result.lower() or "archive" in result.lower()

    async def test_update_description(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.projects import project_new, project_update
            await project_new("My Project")
            result = await project_update("My Project", description="New description")
            assert "updated" in result.lower()

    async def test_project_status_no_active(self, tmp_path):
        with self._patch_db(tmp_path):
            import app.memory.projects as pm
            from app.memory.projects import project_status
            pm._active_project = None
            result = await project_status()
            assert "no project" in result.lower() or "active" in result.lower()

    async def test_project_status_with_active(self, tmp_path):
        with self._patch_db(tmp_path):
            import app.memory.projects as pm
            from app.memory.projects import ProjectContext, project_status
            pm._active_project = ProjectContext(
                name="Test Project",
                slug="test-project",
                description="A test",
                vault_folder="test-project",
                loaded_at="10:00",
            )
            result = await project_status()
            assert "Test Project" in result
            pm._active_project = None


class TestProjectContext:
    """Test ProjectContext data model."""

    def _make_ctx(self):
        from app.memory.projects import ProjectContext
        ctx = ProjectContext(
            name="AI Assistant",
            slug="ai-assistant",
            description="Building a local Jarvis",
            vault_folder="AI Assistant",
            loaded_at="09:30",
        )
        ctx.open_tasks    = ["Implement Phase", "Write tests"]
        ctx.notes         = [{"title": "Architecture", "snippet": "Pipeline design doc"}]
        ctx.memories      = ["User prefers Python", "Project started March 2024"]
        ctx.graph_entities = ["LLM", "RAG", "Obsidian", "Pipeline"]
        return ctx

    def test_to_context_string_contains_project_name(self):
        ctx = self._make_ctx()
        s = ctx.to_context_string()
        assert "AI Assistant" in s

    def test_to_context_string_contains_tasks(self):
        ctx = self._make_ctx()
        s = ctx.to_context_string()
        assert "Implement Phase" in s

    def test_to_context_string_contains_notes(self):
        ctx = self._make_ctx()
        s = ctx.to_context_string()
        assert "Architecture" in s

    def test_to_context_string_contains_entities(self):
        ctx = self._make_ctx()
        s = ctx.to_context_string()
        assert "LLM" in s or "RAG" in s

    def test_to_spoken_summary(self):
        ctx = self._make_ctx()
        s = ctx.to_spoken_summary()
        assert "AI Assistant" in s
        assert "task" in s.lower() or "note" in s.lower()

    def test_empty_context_string(self):
        from app.memory.projects import ProjectContext
        ctx = ProjectContext(
            name="Empty", slug="empty", description="",
            vault_folder="empty", loaded_at="10:00",
        )
        s = ctx.to_context_string()
        assert "Empty" in s   # should still have name


# ═══════════════════════════════════════════════════════════════════════════════
# Skill Auto-Learning
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceNormalisation:
    def setup_method(self):
        from app.memory.skill_learner import _normalise
        self._norm = _normalise

    def test_removes_skip_tools(self):
        tools = ["memory_write", "obsidian_create_note", "memory_read"]
        result = self._norm(tools)
        assert "memory_write" not in result
        assert "memory_read" not in result
        assert "obsidian_create_note" in result

    def test_removes_adjacent_duplicates(self):
        tools = ["obsidian_search", "obsidian_search", "obsidian_create_note"]
        result = self._norm(tools)
        assert result.count("obsidian_search") == 1

    def test_preserves_order(self):
        tools = ["web_search", "obsidian_create_note", "kg_add"]
        result = self._norm(tools)
        assert result == tools

    def test_empty_sequence(self):
        assert self._norm([]) == []

    def test_all_skip_tools_returns_empty(self):
        tools = ["memory_read", "memory_write"]
        result = self._norm(tools)
        assert result == []


class TestSkillNameBuilder:
    def setup_method(self):
        from app.memory.skill_learner import _build_skill_name
        self._build = _build_skill_name

    def test_basic_sequence(self):
        name = self._build(["web_search", "obsidian_create_note"])
        assert isinstance(name, str)
        assert len(name) > 0

    def test_name_contains_recognisable_parts(self):
        name = self._build(["web_search", "obsidian_create_note"])
        # Should include web-search or note somewhere
        assert "search" in name.lower() or "note" in name.lower()

    def test_long_sequence_capped(self):
        tools = ["t1", "t2", "t3", "t4", "t5", "t6"]
        name = self._build(tools)
        assert len(name) <= 50

    def test_empty_sequence_doesnt_crash(self):
        name = self._build([])
        assert isinstance(name, str)


class TestSkillHasher:
    def setup_method(self):
        from app.memory.skill_learner import _sequence_hash
        self._hash = _sequence_hash

    def test_same_sequence_same_hash(self):
        tools = ["web_search", "obsidian_create_note"]
        assert self._hash(tools) == self._hash(tools)

    def test_different_sequence_different_hash(self):
        h1 = self._hash(["web_search", "obsidian_create_note"])
        h2 = self._hash(["obsidian_create_note", "web_search"])
        assert h1 != h2

    def test_hash_is_string(self):
        assert isinstance(self._hash(["tool_a"]), str)


@pytest.mark.asyncio
class TestSkillLearner:
    """Test the SkillLearner with a temp DB."""

    def _patch_db(self, tmp_path):
        db = tmp_path / "test_mem.db"
        return patch("app.memory.skill_learner.settings.memory_db", db)

    async def test_log_short_sequence_ignored(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner
            learner = SkillLearner()
            await learner.log_execution(["single_tool"])
            assert len(learner._pending_proposals) == 0

    async def test_log_empty_sequence_ignored(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner
            learner = SkillLearner()
            await learner.log_execution([])
            assert len(learner._pending_proposals) == 0

    async def test_failed_execution_not_logged(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner
            learner = SkillLearner()
            await learner.log_execution(["web_search", "obsidian_create_note"], outcome="error")
            assert len(learner._pending_proposals) == 0

    async def test_proposal_generated_at_threshold(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner, PATTERN_THRESHOLD
            learner = SkillLearner()
            tools = ["web_search", "obsidian_create_note", "kg_add"]
            for _ in range(PATTERN_THRESHOLD):
                await learner.log_execution(tools, user_text="research and note")
            assert len(learner._pending_proposals) == 1

    async def test_proposal_not_duplicate(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner, PATTERN_THRESHOLD
            learner = SkillLearner()
            tools = ["doc_search", "obsidian_create_note"]
            # Log more than threshold times
            for _ in range(PATTERN_THRESHOLD + 2):
                await learner.log_execution(tools)
            # Should only propose once
            assert len(learner._pending_proposals) == 1

    async def test_pop_next_proposal(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner, PATTERN_THRESHOLD
            learner = SkillLearner()
            tools = ["web_search", "obsidian_append_daily"]
            for _ in range(PATTERN_THRESHOLD):
                await learner.log_execution(tools)

            proposal = learner.pop_next_proposal()
            assert proposal is not None
            assert len(learner._pending_proposals) == 0
            assert learner._awaiting_confirmation is proposal

    async def test_confirm_skill_no_pending(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner
            learner = SkillLearner()
            result = await learner.confirm_skill(accepted=True)
            assert "no skill proposal" in result.lower() or "pending" in result.lower()

    async def test_reject_skill(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner, PATTERN_THRESHOLD
            learner = SkillLearner()
            tools = ["web_search", "obsidian_create_note"]
            for _ in range(PATTERN_THRESHOLD):
                await learner.log_execution(tools)

            learner.pop_next_proposal()
            result = await learner.confirm_skill(accepted=False)
            assert "won't" in result.lower() or "skip" in result.lower() or "got it" in result.lower()

    async def test_skill_learning_summary_empty(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner
            learner = SkillLearner()
            result = await learner.skill_learning_summary()
            assert isinstance(result, str)
            assert "pattern" in result.lower() or "skill" in result.lower()

    async def test_notification_in_proposal(self, tmp_path):
        with self._patch_db(tmp_path):
            from app.memory.skill_learner import SkillLearner, PATTERN_THRESHOLD
            learner = SkillLearner()
            tools = ["doc_search", "obsidian_create_note", "kg_add"]
            for _ in range(PATTERN_THRESHOLD):
                await learner.log_execution(tools, user_text="search docs and take notes")

            proposal = learner._pending_proposals[0]
            assert "skill" in proposal.notification.lower()
            assert len(proposal.tools) > 0


class TestSkillContentBuilder:
    def setup_method(self):
        from app.memory.skill_learner import _build_skill_content
        self._build = _build_skill_content

    def test_contains_tools(self):
        tools = ["web_search", "obsidian_create_note"]
        content = self._build(tools, "research and save notes")
        assert "web_search" in content
        assert "obsidian_create_note" in content

    def test_contains_example_query(self):
        tools = ["web_search"]
        content = self._build(tools, "look up the latest news")
        assert "look up" in content.lower() or "news" in content.lower()

    def test_contains_section_headers(self):
        tools = ["web_search", "kg_add"]
        content = self._build(tools, "example")
        assert "##" in content   # markdown headers

    def test_non_empty(self):
        content = self._build([], "")
        assert isinstance(content, str)
        assert len(content) > 0
