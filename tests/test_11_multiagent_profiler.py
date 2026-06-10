"""
Tests for Multi-Agent Orchestrator and System Profiler.

Run: pytest tests/test_11_multiagent_profiler.py -v

100% offline — no LLM, no GPU, no Ollama, no network.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Agent Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentContext:
    def setup_method(self):
        from app.agents.orchestrator import AgentContext, AgentTask, TaskType
        self.AgentContext = AgentContext
        self.AgentTask    = AgentTask
        self.TaskType     = TaskType

    def test_add_and_get_task(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        t = self.AgentTask(id="t1", type=self.TaskType.CHAT, description="do thing")
        ctx.add_task(t)
        assert ctx.get_task("t1") is t

    def test_get_missing_task_returns_none(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        assert ctx.get_task("nonexistent") is None

    def test_all_done_empty(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        assert ctx.all_done()   # no tasks = trivially done

    def test_all_done_pending(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        t = self.AgentTask(id="t1", type=self.TaskType.TOOL, description="pending")
        ctx.add_task(t)
        assert not ctx.all_done()

    def test_all_done_mixed(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        t1 = self.AgentTask(id="t1", type=self.TaskType.TOOL, description="d1", status="done")
        t2 = self.AgentTask(id="t2", type=self.TaskType.TOOL, description="d2", status="failed")
        ctx.add_task(t1)
        ctx.add_task(t2)
        assert ctx.all_done()

    def test_task_outputs_only_done(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        t1 = self.AgentTask(id="t1", type=self.TaskType.TOOL, description="done task",
                            result="result A", status="done")
        t2 = self.AgentTask(id="t2", type=self.TaskType.TOOL, description="failed task",
                            result="oops", status="failed")
        ctx.add_task(t1)
        ctx.add_task(t2)
        output = ctx.task_outputs()
        assert "result A" in output
        assert "oops" not in output

    def test_elapsed_ms_is_positive(self):
        ctx = self.AgentContext(user_text="test", intent="chat")
        assert ctx.elapsed_ms() >= 0


class TestAgentTask:
    def test_default_status_is_pending(self):
        from app.agents.orchestrator import AgentTask, TaskType
        t = AgentTask(id="x", type=TaskType.CHAT, description="desc")
        assert t.status == "pending"

    def test_default_depends_on_empty(self):
        from app.agents.orchestrator import AgentTask, TaskType
        t = AgentTask(id="x", type=TaskType.CHAT, description="desc")
        assert t.depends_on == []

    def test_task_result_starts_empty(self):
        from app.agents.orchestrator import AgentTask, TaskType
        t = AgentTask(id="x", type=TaskType.TOOL, description="desc")
        assert t.result == ""


@pytest.mark.asyncio
class TestPlannerAgent:
    async def test_research_intent_creates_research_task(self):
        from app.agents.orchestrator import PlannerAgent, AgentContext, TaskType
        agent = PlannerAgent()
        ctx = AgentContext(user_text="research llama.cpp", intent="research")
        await agent.run(ctx)
        assert len(ctx.tasks) == 1
        assert ctx.tasks[0].type == TaskType.RESEARCH

    async def test_chat_intent_creates_chat_task(self):
        from app.agents.orchestrator import PlannerAgent, AgentContext, TaskType
        agent = PlannerAgent()
        ctx = AgentContext(user_text="hello", intent="chat")
        await agent.run(ctx)
        assert len(ctx.tasks) == 1
        assert ctx.tasks[0].type == TaskType.CHAT

    async def test_memory_intent_creates_chat_task(self):
        from app.agents.orchestrator import PlannerAgent, AgentContext, TaskType
        agent = PlannerAgent()
        ctx = AgentContext(user_text="remember that I prefer dark mode", intent="memory")
        await agent.run(ctx)
        assert len(ctx.tasks) == 1
        assert ctx.tasks[0].type == TaskType.CHAT

    async def test_planning_intent_with_failed_planner_falls_back(self):
        from app.agents.orchestrator import PlannerAgent, AgentContext, TaskType
        import app.agents.orchestrator as orch_mod
        agent = PlannerAgent()
        ctx = AgentContext(user_text="create note then update graph", intent="planning")
        orig = orch_mod.make_plan
        orch_mod.make_plan = AsyncMock(side_effect=Exception("LLM not loaded"))
        try:
            await agent.run(ctx)
        finally:
            orch_mod.make_plan = orig
        assert len(ctx.tasks) == 1
        assert ctx.tasks[0].type == TaskType.CHAT


@pytest.mark.asyncio
class TestResearcherAgent:
    async def test_no_research_tasks_does_nothing(self):
        from app.agents.orchestrator import ResearcherAgent, AgentContext, AgentTask, TaskType
        agent = ResearcherAgent()
        ctx = AgentContext(user_text="test", intent="chat")
        ctx.add_task(AgentTask(id="t1", type=TaskType.CHAT, description="chat"))
        await agent.run(ctx)
        assert ctx.research_summary == ""

    async def test_research_task_calls_research_agent(self):
        from app.agents.orchestrator import ResearcherAgent, AgentContext, AgentTask, TaskType
        import app.agents.orchestrator as orch_mod
        agent = ResearcherAgent()
        ctx = AgentContext(user_text="research llama", intent="research")
        t = AgentTask(id="r0", type=TaskType.RESEARCH, description="research llama")
        ctx.add_task(t)

        mock_ra = MagicMock()
        mock_ra.run = AsyncMock(return_value="Research summary here.")
        orig = orch_mod.research_agent
        orch_mod.research_agent = mock_ra
        try:
            await agent.run(ctx)
        finally:
            orch_mod.research_agent = orig

        assert t.status == "done"
        assert ctx.research_summary == "Research summary here."

    async def test_research_task_handles_failure(self):
        from app.agents.orchestrator import ResearcherAgent, AgentContext, AgentTask, TaskType
        import app.agents.orchestrator as orch_mod
        agent = ResearcherAgent()
        ctx = AgentContext(user_text="research", intent="research")
        t = AgentTask(id="r0", type=TaskType.RESEARCH, description="research")
        ctx.add_task(t)

        mock_ra = MagicMock()
        mock_ra.run = AsyncMock(side_effect=Exception("network error"))
        orig = orch_mod.research_agent
        orch_mod.research_agent = mock_ra
        try:
            await agent.run(ctx)
        finally:
            orch_mod.research_agent = orig

        assert t.status == "failed"
        assert "failed" in t.result.lower()


@pytest.mark.asyncio
class TestExecutorAgent:
    async def test_no_tool_tasks_does_nothing(self):
        from app.agents.orchestrator import ExecutorAgent, AgentContext, AgentTask, TaskType
        agent = ExecutorAgent()
        ctx = AgentContext(user_text="test", intent="chat")
        ctx.add_task(AgentTask(id="c0", type=TaskType.CHAT, description="chat"))
        await agent.run(ctx)
        assert ctx.tool_results == []

    async def test_tool_task_dispatches_to_router(self):
        from app.agents.orchestrator import ExecutorAgent, AgentContext, AgentTask, TaskType
        agent = ExecutorAgent()
        ctx = AgentContext(user_text="create note", intent="tool")
        t = AgentTask(id="t0", type=TaskType.TOOL, description="create note",
                      tool_name="obsidian_create_note",
                      tool_params={"title": "Test", "content": "body"})
        ctx.add_task(t)

        import app.agents.orchestrator as orch_mod
        mock_result = {"tool": "obsidian_create_note", "status": "ok", "result": "Created."}
        MockRouter = MagicMock()
        MockRouter.return_value.dispatch = AsyncMock(return_value=mock_result)
        orig = orch_mod.ToolRouter
        orch_mod.ToolRouter = MockRouter
        try:
            await agent.run(ctx)
        finally:
            orch_mod.ToolRouter = orig

        assert t.status == "done"
        assert "Created." in t.result
        assert len(ctx.tool_results) == 1


@pytest.mark.asyncio
class TestMemoryCuratorAgent:
    async def test_runs_without_error_on_empty_context(self):
        from app.agents.orchestrator import MemoryCuratorAgent, AgentContext
        agent = MemoryCuratorAgent()
        ctx = AgentContext(user_text="test", intent="chat")
        # Should not raise even with no tools or memory
        await agent.run(ctx)

    async def test_scores_memory_write_results(self):
        from app.agents.orchestrator import MemoryCuratorAgent, AgentContext
        import app.agents.orchestrator as orch_mod
        agent = MemoryCuratorAgent()
        ctx = AgentContext(user_text="remember I prefer dark mode", intent="memory")
        ctx.tool_results = [{"tool": "memory_write", "status": "ok", "result": "mem_001"}]

        mock_score = MagicMock()
        orig = orch_mod.score_memory
        orch_mod.score_memory = mock_score
        try:
            await agent.run(ctx)
        finally:
            orch_mod.score_memory = orig
        mock_score.assert_called_once()

    async def test_logs_tool_sequence_to_skill_learner(self):
        from app.agents.orchestrator import MemoryCuratorAgent, AgentContext
        import app.agents.orchestrator as orch_mod
        agent = MemoryCuratorAgent()
        ctx = AgentContext(user_text="create note and add to graph", intent="planning")
        ctx.tool_results = [
            {"tool": "obsidian_create_note", "status": "ok", "result": "Created"},
            {"tool": "kg_add", "status": "ok", "result": "Added"},
        ]

        mock_sl = MagicMock()
        mock_sl.log_execution = AsyncMock()
        orig = orch_mod.skill_learner
        orch_mod.skill_learner = mock_sl
        try:
            await agent.run(ctx)
        finally:
            orch_mod.skill_learner = orig
        mock_sl.log_execution.assert_called_once()
        args = mock_sl.log_execution.call_args[0]
        assert "obsidian_create_note" in args[0]


@pytest.mark.asyncio
class TestSkillBuilderAgent:
    async def test_no_pending_proposal_does_nothing(self):
        from app.agents.orchestrator import SkillBuilderAgent, AgentContext
        import app.agents.orchestrator as orch_mod
        agent = SkillBuilderAgent()
        ctx = AgentContext(user_text="test", intent="chat")
        ctx.final_response = "Original response."

        mock_sl = MagicMock()
        mock_sl.pop_next_proposal = MagicMock(return_value=None)
        orig = orch_mod.skill_learner
        orch_mod.skill_learner = mock_sl
        try:
            await agent.run(ctx)
        finally:
            orch_mod.skill_learner = orig
        assert ctx.final_response == "Original response."

    async def test_pending_proposal_appended_to_response(self):
        from app.agents.orchestrator import SkillBuilderAgent, AgentContext
        from app.memory.skill_learner import SkillProposal
        import app.agents.orchestrator as orch_mod
        agent = SkillBuilderAgent()
        ctx = AgentContext(user_text="test", intent="chat")
        ctx.final_response = "Here is your answer."

        proposal = SkillProposal(
            sequence_hash="abc",
            tools=["web_search", "obsidian_create_note"],
            skill_name="research-then-note",
            skill_content="content",
            example_query="research and note",
            count=3,
            notification="I noticed you always do X. Save as skill?",
        )

        mock_sl = MagicMock()
        mock_sl.pop_next_proposal = MagicMock(return_value=proposal)
        orig = orch_mod.skill_learner
        orch_mod.skill_learner = mock_sl
        try:
            await agent.run(ctx)
        finally:
            orch_mod.skill_learner = orig
        assert "I noticed you always do X" in ctx.final_response


# ═══════════════════════════════════════════════════════════════════════════════
# System Profiler
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemProfilerDetectors:
    def test_get_ram_returns_two_ints(self):
        from app.system_profiler import _get_ram
        total, avail = _get_ram()
        assert isinstance(total, int)
        assert isinstance(avail, int)
        assert total >= 0
        assert avail >= 0

    def test_get_cpu_returns_dict(self):
        from app.system_profiler import _get_cpu
        cpu = _get_cpu()
        assert "name" in cpu
        assert "cores_logical" in cpu
        assert "avx" in cpu
        assert isinstance(cpu["cores_logical"], int)
        assert cpu["cores_logical"] >= 1

    def test_get_disk_free_returns_float(self):
        from app.system_profiler import _get_disk_free
        free = _get_disk_free("/tmp")
        assert isinstance(free, float)
        assert free >= 0.0

    def test_get_disk_free_bad_path(self):
        from app.system_profiler import _get_disk_free
        free = _get_disk_free("/nonexistent/path/xyz")
        assert free == 0.0

    def test_get_ollama_returns_tuple(self):
        from app.system_profiler import _get_ollama
        available, version, models = _get_ollama()
        assert isinstance(available, bool)
        assert isinstance(version, str)
        assert isinstance(models, list)


class TestProfileSystem:
    def test_profile_system_returns_profile(self):
        from app.system_profiler import profile_system, SystemProfile
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            profile = profile_system()
        assert isinstance(profile, SystemProfile)
        assert profile.cpu_cores_logical >= 1
        assert profile.ram_total_mb >= 0
        assert isinstance(profile.gpus, list)

    def test_profile_summary_is_string(self):
        from app.system_profiler import profile_system
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            profile = profile_system()
        summary = profile.summary()
        assert isinstance(summary, str)
        assert "CPU" in summary
        assert "RAM" in summary


class TestModelRecommender:
    """Test the recommender with synthetic profiles."""

    def _make_profile(
        self,
        ram_total_gb=16, ram_avail_gb=10,
        vram_mb=0, has_gpu=False,
        is_apple=False, ollama=True,
        cpu_cores=8,
    ):
        from app.system_profiler import SystemProfile, GPUInfo
        p = SystemProfile()
        p.ram_total_mb     = ram_total_gb * 1024
        p.ram_available_mb = ram_avail_gb * 1024
        p.primary_vram_mb  = vram_mb
        p.has_gpu          = has_gpu
        p.is_apple_silicon = is_apple
        p.cpu_cores_physical = cpu_cores
        p.cpu_cores_logical  = cpu_cores * 2
        p.has_avx2         = True
        p.ollama_available  = ollama
        p.ollama_version    = "0.5.0" if ollama else ""
        p.disk_free_gb      = 50.0
        p.os_name           = "Linux"
        p.os_version        = "6.8"
        p.python_version    = "3.12.0"
        if has_gpu and not is_apple:
            p.gpus = [GPUInfo(vendor="nvidia", name="RTX 3080", vram_mb=vram_mb)]
        elif is_apple:
            p.gpus = [GPUInfo(vendor="apple", name="M2 Pro", vram_mb=ram_total_gb*1024)]
        return p

    def test_high_ram_system_gets_high_tier(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=64, ram_avail_gb=48)
        rec = recommend_models(profile)
        assert rec.tier in ("high", "mid")
        assert rec.llm_size_gb >= 4.0

    def test_low_ram_system_gets_small_model(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=4, ram_avail_gb=2)
        rec = recommend_models(profile)
        assert rec.llm_size_gb <= 2.0

    def test_ollama_backend_when_available(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ollama=True)
        rec = recommend_models(profile)
        assert rec.backend == "ollama"

    def test_llamacpp_backend_when_ollama_absent(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ollama=False)
        rec = recommend_models(profile)
        assert rec.backend == "llamacpp"
        assert any("Ollama" in w for w in rec.warnings)

    def test_apple_silicon_gets_full_gpu_offload(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=32, ram_avail_gb=24,
                                     is_apple=True, has_gpu=True, vram_mb=32*1024)
        rec = recommend_models(profile)
        assert rec.n_gpu_layers == -1

    def test_cpu_only_system_gets_zero_gpu_layers(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(has_gpu=False, vram_mb=0)
        rec = recommend_models(profile)
        assert rec.n_gpu_layers == 0

    def test_thread_count_is_cores_minus_two(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(cpu_cores=8)
        rec = recommend_models(profile)
        assert rec.n_threads == 6

    def test_thread_count_minimum_one(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(cpu_cores=2)
        rec = recommend_models(profile)
        assert rec.n_threads >= 1

    def test_stt_model_is_valid(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert rec.stt_model in ("tiny", "base", "small", "medium", "large-v3")

    def test_high_ram_gets_better_stt(self):
        from app.system_profiler import recommend_models
        low  = recommend_models(self._make_profile(ram_total_gb=4,  ram_avail_gb=2))
        high = recommend_models(self._make_profile(ram_total_gb=32, ram_avail_gb=24))
        stt_order = ["tiny", "base", "small", "medium", "large-v3"]
        assert stt_order.index(high.stt_model) >= stt_order.index(low.stt_model)

    def test_env_snippet_contains_backend(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert "LLM_BACKEND" in rec.env_snippet

    def test_env_snippet_contains_threads(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert "LLM_THREADS" in rec.env_snippet

    def test_reasoning_is_nonempty(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile()
        rec = recommend_models(profile)
        assert len(rec.reasoning) > 0

    def test_alternatives_list(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=32, ram_avail_gb=20)
        rec = recommend_models(profile)
        assert isinstance(rec.alternatives, list)

    def test_format_report_is_string(self):
        from app.system_profiler import recommend_models, format_report
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            from app.system_profiler import profile_system
            profile = profile_system()
        rec = recommend_models(profile)
        report = format_report(profile, rec)
        assert isinstance(report, str)
        assert "Tzar System Profile" in report
        assert "Recommendation" in report
        assert "env snippet" in report.lower() or ".env" in report

    def test_very_low_ram_does_not_crash(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(ram_total_gb=2, ram_avail_gb=1)
        rec = recommend_models(profile)   # should not raise
        assert rec.llm_size_gb > 0

    def test_nvidia_gpu_partial_offload(self):
        from app.system_profiler import recommend_models
        profile = self._make_profile(has_gpu=True, vram_mb=8192)
        rec = recommend_models(profile)
        assert rec.n_gpu_layers > 0


class TestSystemProfilerTool:
    @pytest.mark.asyncio
    async def test_tool_returns_string(self):
        from app.system_profiler import run_system_profile_tool
        with patch("app.system_profiler._get_ollama", return_value=(False, "", [])):
            result = await run_system_profile_tool()
        assert isinstance(result, str)
        assert len(result) > 100
