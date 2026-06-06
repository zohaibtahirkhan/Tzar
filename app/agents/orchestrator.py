"""
app/agents/orchestrator.py

Multi-Agent Layer — Phase 10.

Five specialised agents coordinate through a shared AgentContext.
No external message bus needed at this scale — pure async Python.

Agents:
  PlannerAgent      — decomposes the user goal into AgentTasks
  ResearcherAgent   — wraps the Phase 2 ResearchAgent for web research
  ExecutorAgent     — runs tool calls from the plan, one step at a time
  MemoryCuratorAgent— runs after every response: scores, prunes, archives
  SkillBuilderAgent — observes executor sequences, feeds skill_learner

Flow:
  User text
      ↓
  MultiAgentOrchestrator.run(user_text)
      ↓
  PlannerAgent.plan()          → AgentPlan {tasks[]}
      ↓
  For each task in parallel or sequence:
      ResearcherAgent.run()    if task.type == RESEARCH
      ExecutorAgent.run()      if task.type == TOOL / CHAT
      ↓
  MemoryCuratorAgent.curate()  → scores + prunes in background
  SkillBuilderAgent.observe()  → logs sequence to skill_learner
      ↓
  Final response assembled from all task outputs

Integration — in pipeline.py, replace the manual plan/execute block with:

    from app.agents.orchestrator import orchestrator
    if settings.multi_agent_enabled:
        return await orchestrator.run(user_text, intent)
    # else: existing single-agent path (unchanged)

This is opt-in. Set MULTI_AGENT_ENABLED=true in .env to activate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

# Module-level imports so agents can be patched in tests.
# These are imported lazily at runtime if not yet available.
try:
    from app.planner import make_plan
except ImportError:
    make_plan = None  # type: ignore

try:
    from app.agents.researcher import research_agent
except ImportError:
    research_agent = None  # type: ignore

try:
    from app.tools.router import ToolRouter
except ImportError:
    ToolRouter = None  # type: ignore

try:
    from app.memory.scoring import score_memory
except ImportError:
    score_memory = None  # type: ignore

try:
    from app.memory.skill_learner import skill_learner
except ImportError:
    skill_learner = None  # type: ignore


# ─── Task types ───────────────────────────────────────────────────────────────

class TaskType(Enum):
    CHAT     = "chat"
    TOOL     = "tool"
    RESEARCH = "research"
    MEMORY   = "memory"


# ─── Shared context ───────────────────────────────────────────────────────────

@dataclass
class AgentTask:
    """A single unit of work assigned to one agent."""
    id: str
    type: TaskType
    description: str
    tool_name: Optional[str] = None
    tool_params: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)   # task IDs this waits for
    result: str = ""
    status: str = "pending"   # pending / running / done / failed


@dataclass
class AgentContext:
    """
    Shared mutable state passed between all agents in one orchestration run.
    Agents read from and write to this — the message bus equivalent.
    """
    user_text: str
    intent: str

    tasks: list[AgentTask] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    research_summary: str = ""
    memory_context: str = ""
    project_context: str = ""
    final_response: str = ""

    started_at: float = field(default_factory=time.perf_counter)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000

    def add_task(self, task: AgentTask) -> None:
        self.tasks.append(task)

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def all_done(self) -> bool:
        return all(t.status in ("done", "failed") for t in self.tasks)

    def task_outputs(self) -> str:
        """Concatenate all task results for LLM synthesis."""
        parts = []
        for t in self.tasks:
            if t.result and t.status == "done":
                parts.append(f"[{t.description}]: {t.result}")
        return "\n\n".join(parts)


# ─── Base agent ───────────────────────────────────────────────────────────────

class BaseAgent:
    name: str = "BaseAgent"

    async def run(self, ctx: AgentContext) -> None:
        raise NotImplementedError


# ─── Planner Agent ────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Decomposes the user goal into a sequence of AgentTasks.
    Uses the existing make_plan() from planner.py, then maps steps → tasks.
    Falls back to a single CHAT task if planning fails.
    """
    name = "PlannerAgent"

    async def run(self, ctx: AgentContext) -> None:
        from app.intent import IntentType

        intent = IntentType(ctx.intent)

        # Research intent → single research task, no planner LLM call needed
        if intent == IntentType.RESEARCH:
            ctx.add_task(AgentTask(
                id="research-0",
                type=TaskType.RESEARCH,
                description=f"Research: {ctx.user_text}",
            ))
            logger.debug("PlannerAgent: RESEARCH task created")
            return

        # Chat/memory intent → single chat task
        if intent in (IntentType.CHAT, IntentType.MEMORY):
            ctx.add_task(AgentTask(
                id="chat-0",
                type=TaskType.CHAT,
                description="Direct LLM response",
            ))
            return

        # Tool/planning intent → call make_plan() and map steps to tasks
        try:
            from app.llm.engine import llm_engine
            plan = await make_plan(ctx.user_text, llm_engine.generate)

            if plan.can_answer_directly or not plan.steps:
                ctx.add_task(AgentTask(id="chat-0", type=TaskType.CHAT,
                                       description="Direct answer"))
                return

            prev_id = None
            for i, (step, tool) in enumerate(
                zip(plan.steps, plan.required_tools + [""] * len(plan.steps))
            ):
                task_id = f"task-{i}"
                task_type = TaskType.TOOL if tool else TaskType.CHAT
                t = AgentTask(
                    id=task_id,
                    type=task_type,
                    description=step,
                    tool_name=tool or None,
                    depends_on=[prev_id] if prev_id else [],
                )
                ctx.add_task(t)
                prev_id = task_id

            logger.debug("PlannerAgent: {} tasks created from plan", len(ctx.tasks))

        except Exception as e:
            logger.error("PlannerAgent failed: {} — falling back to CHAT", e)
            ctx.add_task(AgentTask(id="chat-0", type=TaskType.CHAT,
                                   description="Fallback direct response"))


# ─── Researcher Agent ─────────────────────────────────────────────────────────

class ResearcherAgent(BaseAgent):
    """
    Handles RESEARCH tasks by delegating to the Phase 2 ResearchAgent.
    Stores the summary in ctx.research_summary.
    """
    name = "ResearcherAgent"

    async def run(self, ctx: AgentContext) -> None:
        research_tasks = [t for t in ctx.tasks if t.type == TaskType.RESEARCH]
        if not research_tasks:
            return

        for task in research_tasks:
            task.status = "running"
            try:
                result = await research_agent.run(task.description)
                task.result = result
                task.status = "done"
                ctx.research_summary = result
                logger.debug("ResearcherAgent: task '{}' done", task.id)
            except Exception as e:
                task.result = f"Research failed: {e}"
                task.status = "failed"
                logger.error("ResearcherAgent: task '{}' failed: {}", task.id, e)


# ─── Executor Agent ───────────────────────────────────────────────────────────

class ExecutorAgent(BaseAgent):
    """
    Executes TOOL tasks in dependency order.
    CHAT tasks are handled by the final LLM synthesis step, not here.
    """
    name = "ExecutorAgent"

    async def run(self, ctx: AgentContext) -> None:
        tool_tasks = [t for t in ctx.tasks if t.type == TaskType.TOOL]
        if not tool_tasks:
            return

        router = ToolRouter()

        for task in tool_tasks:
            # Wait for dependencies
            await self._wait_deps(task, ctx)

            task.status = "running"
            try:
                if task.tool_name:
                    result = await router.dispatch(task.tool_name, task.tool_params)
                    task.result = str(result.get("result", ""))
                    ctx.tool_results.append(result)
                task.status = "done"
                logger.debug("ExecutorAgent: task '{}' ({}) done", task.id, task.tool_name)
            except Exception as e:
                task.result = f"Tool error: {e}"
                task.status = "failed"
                logger.error("ExecutorAgent: task '{}' failed: {}", task.id, e)

    @staticmethod
    async def _wait_deps(task: AgentTask, ctx: AgentContext, timeout: float = 30.0) -> None:
        deadline = time.perf_counter() + timeout
        for dep_id in task.depends_on:
            while time.perf_counter() < deadline:
                dep = ctx.get_task(dep_id)
                if dep and dep.status in ("done", "failed"):
                    break
                await asyncio.sleep(0.05)


# ─── Memory Curator Agent ─────────────────────────────────────────────────────

class MemoryCuratorAgent(BaseAgent):
    """
    Runs after every response. Non-blocking — fires as a background task.

    Responsibilities:
      1. Score any newly written memories (Phase 6)
      2. Log the tool sequence to skill_learner (Phase 8)
      3. Periodically prune low-score memories (every 50 runs)
    """
    name = "MemoryCuratorAgent"
    _run_count = 0

    async def run(self, ctx: AgentContext) -> None:
        MemoryCuratorAgent._run_count += 1

        # 1. Score new memories written during this turn
        for r in ctx.tool_results:
            if r.get("tool") in ("memory_write", "save_memory", "hot_memory_add"):
                try:
                    if score_memory:
                        score_memory(
                            memory_id=r.get("result", f"mem-{time.time()}"),
                            content=ctx.user_text,
                        )
                except Exception as e:
                    logger.debug("MemoryCurator: scoring failed: {}", e)

        # 2. Feed tool sequence to skill learner
        try:
            tool_seq = [r["tool"] for r in ctx.tool_results if r.get("status") == "ok"]
            if tool_seq and skill_learner:
                await skill_learner.log_execution(tool_seq, ctx.user_text, "success")
        except Exception as e:
            logger.debug("MemoryCurator: skill logging failed: {}", e)

        # 3. Prune every 50 runs (background, non-blocking)
        if MemoryCuratorAgent._run_count % 50 == 0:
            try:
                from app.memory.scoring import prune_low_score_memories  # noqa: PLC0415
                result = prune_low_score_memories()
                logger.info("MemoryCurator: {}", result)
            except Exception as e:
                logger.debug("MemoryCurator: prune failed: {}", e)


# ─── Skill Builder Agent ──────────────────────────────────────────────────────

class SkillBuilderAgent(BaseAgent):
    """
    Checks for pending skill proposals after each turn and appends
    the notification to the response if one is ready.
    """
    name = "SkillBuilderAgent"

    async def run(self, ctx: AgentContext) -> None:
        try:
            if not skill_learner:
                return
            proposal = skill_learner.pop_next_proposal()
            if proposal:
                ctx.final_response += "\n\n" + proposal.notification
                logger.info("SkillBuilderAgent: appended proposal for '{}'", proposal.skill_name)
        except Exception as e:
            logger.debug("SkillBuilderAgent: {}", e)


# ─── Response Synthesiser ─────────────────────────────────────────────────────

async def _synthesise(ctx: AgentContext) -> str:
    """
    Final LLM call that assembles all task outputs into a coherent response.
    Skipped if there's only a research result (already a full response).
    """
    # Pure research response — already formatted
    if ctx.research_summary and not ctx.tool_results:
        return ctx.research_summary

    task_output = ctx.task_outputs()

    # If nothing was produced, do a direct LLM call
    if not task_output and not ctx.tool_results:
        from app.llm.engine import llm_engine
        from app.prompts.templates import build_system_prompt
        system = build_system_prompt(
            memory_context=ctx.memory_context,
            plan_context="",
        )
        return await llm_engine.generate(
            [{"role": "user", "content": ctx.user_text}],
            system_prompt=system,
        )

    # Synthesise tool outputs into a spoken response
    from app.llm.engine import llm_engine
    from app.prompts.templates import build_system_prompt

    system = build_system_prompt(
        memory_context=ctx.memory_context,
        plan_context="",
    )
    synthesis_prompt = (
        f"The following tool results were gathered:\n\n{task_output}\n\n"
        f"Answer the user's original request naturally and concisely: {ctx.user_text}"
    )
    return await llm_engine.generate(
        [{"role": "user", "content": synthesis_prompt}],
        system_prompt=system,
    )


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class MultiAgentOrchestrator:
    """
    Coordinates all agents for a single user turn.

    Usage (in pipeline.py):
        from app.agents.orchestrator import orchestrator
        response = await orchestrator.run(user_text, intent.value)
    """

    def __init__(self):
        self.planner  = PlannerAgent()
        self.researcher = ResearcherAgent()
        self.executor = ExecutorAgent()
        self.curator  = MemoryCuratorAgent()
        self.skill_builder = SkillBuilderAgent()

    async def run(self, user_text: str, intent: str) -> str:
        ctx = AgentContext(user_text=user_text, intent=intent)

        # Load context (project + memory) in parallel with planning
        ctx.project_context = self._get_project_context()
        ctx.memory_context  = await self._get_memory_context(user_text)

        # 1. Plan
        await self.planner.run(ctx)
        logger.info(
            "Orchestrator: {} task(s) planned for intent={}",
            len(ctx.tasks), intent,
        )

        # 2. Research + Executor run concurrently (independent tasks)
        await asyncio.gather(
            self.researcher.run(ctx),
            self.executor.run(ctx),
        )

        # 3. Synthesise final response
        ctx.final_response = await _synthesise(ctx)

        # 4. Background: curator + skill builder (don't await — fire and forget)
        asyncio.create_task(self.curator.run(ctx))
        asyncio.create_task(self.skill_builder.run(ctx))

        elapsed = ctx.elapsed_ms()
        logger.info(
            "Orchestrator: done in {:.0f}ms — {} tool(s) executed",
            elapsed, len(ctx.tool_results),
        )
        return ctx.final_response

    @staticmethod
    def _get_project_context() -> str:
        try:
            from app.memory.projects import get_active_project_context
            return get_active_project_context()
        except Exception:
            return ""

    @staticmethod
    async def _get_memory_context(user_text: str) -> str:
        try:
            from app.memory.manager import memory_manager
            return await memory_manager.recall_for_context(user_text)
        except Exception:
            return ""


# ─── Singleton ────────────────────────────────────────────────────────────────

orchestrator = MultiAgentOrchestrator()
