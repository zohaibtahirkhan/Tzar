"""
app/agents/coordinator.py — Lean 2-Agent Coordinator

Replaces the 5-agent orchestrator as the DEFAULT multi-step execution path.

Architecture:
    Coordinator       — reads capabilities, builds a plan, synthesises the answer
    ToolExecutor      — runs exactly one tool per call, returns structured result

    Researcher/Curator/SkillBuilder are background services, not blocking agents.
    The 5-agent MultiAgentOrchestrator stays available via multi_agent_experimental=True.

Flow:
    User text + Capabilities
        ↓
    Coordinator._plan()           → ordered list of ToolStep
        ↓
    for step in plan:
        ToolExecutor.run(step)    → StepResult
        Critic.evaluate()         → satisfied? retry?
        ↓
    Coordinator._synthesise()     → spoken response string

Integration (pipeline.py):
    from app.agents.coordinator import coordinator
    if settings.multi_agent_enabled:
        return await coordinator.run(user_text, caps)
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.config import settings


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ToolStep:
    """One planned action in the execution sequence."""
    description: str
    tool:        str
    params:      dict = field(default_factory=dict)
    required:    bool = True   # if False, failure is non-fatal


@dataclass
class StepResult:
    tool:    str
    status:  str    # "ok" | "error" | "skipped"
    result:  str
    elapsed_ms: float = 0.0


@dataclass
class CriticResult:
    satisfied:       bool
    reason:          str = ""
    retry_suggestion: str = ""


@dataclass
class CoordinatorContext:
    user_text:    str
    caps:         object          # Capabilities
    memory_ctx:   str = ""
    conv_history: str = ""
    step_results: list[StepResult] = field(default_factory=list)


# ─── Tool Executor ────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Runs exactly one tool call.
    Wraps the existing ToolRouter so no new dispatch logic is needed.
    """

    def __init__(self):
        self._router = None   # lazy import to avoid circular

    def _get_router(self):
        if self._router is None:
            from app.tools.router import ToolRouter
            from app.memory.manager import memory_manager
            self._router = ToolRouter(memory_manager=memory_manager)
        return self._router

    async def run(self, step: ToolStep) -> StepResult:
        t0 = time.perf_counter()
        router = self._get_router()
        try:
            result = await asyncio.wait_for(
                router.dispatch({"tool": step.tool, **step.params}),
                timeout=30.0,
            )
            return StepResult(
                tool=step.tool,
                status=result.get("status", "ok"),
                result=str(result.get("result", ""))[:2000],
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except asyncio.TimeoutError:
            logger.error("Tool '{}' timed out", step.tool)
            return StepResult(tool=step.tool, status="error", result="Tool timed out.",
                              elapsed_ms=(time.perf_counter() - t0) * 1000)
        except Exception as exc:
            logger.error("Tool '{}' raised: {}", step.tool, exc)
            return StepResult(tool=step.tool, status="error", result=str(exc),
                              elapsed_ms=(time.perf_counter() - t0) * 1000)


tool_executor = ToolExecutor()


# ─── Critic ───────────────────────────────────────────────────────────────────

_CRITIC_PROMPT = """\
You are a quality critic for an AI assistant.

Original request: {request}
Step taken: {action}
Result received: {result}

Did this result fully address what was needed for this step?
Answer ONLY with JSON — no prose.
{{"satisfied": true_or_false, "reason": "one sentence", "retry_suggestion": "what to try instead if not satisfied"}}
"""

class Critic:
    """
    Evaluates whether a tool result satisfied the step's intent.
    Uses a quick LLM call — kept short on purpose.
    """

    async def evaluate(
        self,
        request: str,
        step: ToolStep,
        result: StepResult,
    ) -> CriticResult:
        # Skip critic for errors — already failed, no value in evaluating
        if result.status == "error":
            return CriticResult(satisfied=False, reason="Tool returned error.",
                                retry_suggestion=f"Try a different approach for: {step.description}")

        try:
            from app.llm.engine import llm_engine
            raw = await asyncio.wait_for(
                llm_engine.generate(
                    [{"role": "user", "content": _CRITIC_PROMPT.format(
                        request=request,
                        action=step.description,
                        result=result.result[:600],
                    )}],
                    system_prompt="Return only valid JSON. Be strict but fair.",
                ),
                timeout=15.0,
            )
            clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
            m     = re.search(r"\{.*\}", clean, re.DOTALL)
            if m:
                data = json.loads(m.group())
                return CriticResult(
                    satisfied=bool(data.get("satisfied", True)),
                    reason=str(data.get("reason", "")),
                    retry_suggestion=str(data.get("retry_suggestion", "")),
                )
        except Exception as exc:
            logger.warning("Critic failed (non-fatal): {}", exc)

        # If critic call fails, assume satisfied — don't block the pipeline
        return CriticResult(satisfied=True, reason="Critic unavailable — assuming ok.")


critic = Critic()


# ─── Plan generation ──────────────────────────────────────────────────────────

_PLAN_PROMPT = """\
You are a task planner for a local voice assistant.
Given a user request and the capabilities needed, produce an execution plan.

Respond ONLY in JSON — no prose.

{{
  "goal": "one sentence describing the end goal",
  "steps": [
    {{"description": "what this step does", "tool": "tool_name", "params": {{}}, "required": true}},
    ...
  ]
}}

Available tools (use EXACT names):
  read_file, write_file, append_file, list_directory, delete_file,
  web_search, save_memory, recall_memory, memory_write, memory_remove,
  obsidian_create_note, obsidian_append_daily, obsidian_capture_idea,
  obsidian_search, obsidian_keyword_search, obsidian_read_note, obsidian_list_vault,
  obsidian_morning_briefing, obsidian_get_related, obsidian_get_project, obsidian_reindex,
  kg_summary, kg_path, kg_neighbors, kg_clusters, kg_add, kg_extract_and_index,
  doc_search, ingest_document, ingest_directory, remove_document, list_documents,
  unified_search, skill_load, skill_create, session_search,
  memory_scores, memory_prune, project_list, project_switch,
  system_profile, mcp_status,
  browser_action

Capabilities needed: {caps}
Request: {request}

Rules:
- Keep steps minimal (2–5 is usually right).
- research steps that need web search: use web_search tool.
- For RAG / local document lookups: use doc_search or unified_search.
- params should only include fields relevant to the tool.
- required: false for optional/enrichment steps.
"""


async def _build_plan(ctx: CoordinatorContext) -> list[ToolStep]:
    """Ask the LLM to produce an ordered list of ToolSteps."""
    try:
        from app.llm.engine import llm_engine
        caps_summary = ", ".join(
            k for k, v in ctx.caps.as_dict().items()
            if isinstance(v, bool) and v
        ) or "chat"

        raw = await asyncio.wait_for(
            llm_engine.generate(
                [{"role": "user", "content": _PLAN_PROMPT.format(
                    caps=caps_summary,
                    request=ctx.user_text,
                )}],
                system_prompt="Return only valid JSON.",
            ),
            timeout=20.0,
        )
        clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        m     = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return []
        data  = json.loads(m.group())
        steps = []
        for s in data.get("steps", []):
            if not s.get("tool"):
                continue
            steps.append(ToolStep(
                description=s.get("description", s["tool"]),
                tool=s["tool"],
                params=s.get("params", {}),
                required=s.get("required", True),
            ))
        logger.info("Coordinator plan: {} steps — {}", len(steps), [s.tool for s in steps])
        return steps

    except Exception as exc:
        logger.error("Plan generation failed: {}", exc)
        return []


# ─── Synthesis ────────────────────────────────────────────────────────────────

_SYNTH_PROMPT = """\
You are a helpful voice assistant.
Original request: {request}

Results from completed steps:
{results}

Synthesise a clear, concise spoken response that directly answers the request.
If any steps failed, acknowledge it briefly and provide what you can.
Do NOT include JSON — respond in plain natural language only.
"""


async def _synthesise(ctx: CoordinatorContext) -> str:
    """Turn the collected step results into a final spoken response."""
    if not ctx.step_results:
        return "I wasn't able to complete that task."

    results_text = "\n".join(
        f"[{r.tool}] ({r.status}): {r.result[:400]}"
        for r in ctx.step_results
    )
    try:
        from app.llm.engine import llm_engine
        from app.prompts.templates import build_system_prompt
        system = build_system_prompt(
            memory_context=ctx.memory_ctx,
            conversation_history=ctx.conv_history,
            plan_context="",
        )
        response = await asyncio.wait_for(
            llm_engine.generate(
                [{"role": "user", "content": _SYNTH_PROMPT.format(
                    request=ctx.user_text,
                    results=results_text,
                )}],
                system_prompt=system,
            ),
            timeout=30.0,
        )
        return response.strip()
    except Exception as exc:
        logger.error("Synthesis failed: {}", exc)
        # Fallback: return the last successful result
        ok = [r for r in ctx.step_results if r.status == "ok"]
        return ok[-1].result if ok else "Task completed with errors."


# ─── Coordinator ─────────────────────────────────────────────────────────────

class Coordinator:
    """
    Lean 2-agent coordinator.
    Plans → Executes (with Critic retry) → Synthesises.
    """

    async def run(self, user_text: str, caps) -> str:
        """
        Main entry point.
        caps: Capabilities object from app.intent.classify()
        """
        t0 = time.perf_counter()
        logger.info("Coordinator.run — caps: {}", caps.as_dict())

        # Build context
        from app.memory.manager import memory_manager
        memory_ctx, conv_history = await memory_manager.get_context(user_text)

        ctx = CoordinatorContext(
            user_text=user_text,
            caps=caps,
            memory_ctx=memory_ctx,
            conv_history=conv_history,
        )

        # Research short-circuit — delegate to Research Agent for pure research
        if caps.needs_research and not caps.needs_tools and not caps.needs_planning:
            logger.info("Coordinator: pure research — delegating to ResearchAgent")
            from app.agents.researcher import research_agent
            response = await research_agent.run(user_text)
            await self._archive(user_text, response)
            return response

        # Build plan
        steps = await _build_plan(ctx)
        if not steps:
            # No plan — fall back to direct LLM
            logger.info("Coordinator: no plan generated, falling back to direct LLM")
            return await self._direct_llm(ctx)

        # Execute each step with critic evaluation
        for step in steps:
            result = await self._execute_with_retry(step, ctx)
            ctx.step_results.append(result)
            if result.status == "error" and step.required:
                logger.warning("Required step '{}' failed — stopping plan", step.tool)
                break

        # Synthesise response from all results
        response = await _synthesise(ctx)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Coordinator done in {:.0f}ms — {} steps, {} ok, {} errors",
            elapsed,
            len(ctx.step_results),
            sum(1 for r in ctx.step_results if r.status == "ok"),
            sum(1 for r in ctx.step_results if r.status == "error"),
        )

        await self._archive(user_text, response)

        # Background: run memory curator and skill observer
        asyncio.create_task(self._background_tasks(user_text, ctx.step_results))

        return response

    async def _execute_with_retry(
        self,
        step: ToolStep,
        ctx: CoordinatorContext,
        max_retries: int = 1,
    ) -> StepResult:
        """Run a step. If critic is unsatisfied, retry once with the suggestion."""
        for attempt in range(max_retries + 1):
            result = await tool_executor.run(step)

            verdict = await critic.evaluate(ctx.user_text, step, result)
            logger.debug(
                "Critic [{}] attempt {}: satisfied={} — {}",
                step.tool, attempt + 1, verdict.satisfied, verdict.reason[:80],
            )

            if verdict.satisfied:
                return result

            if attempt < max_retries and verdict.retry_suggestion:
                logger.info("Critic unsatisfied — retrying '{}': {}", step.tool, verdict.retry_suggestion)
                # Inject the retry suggestion into params
                step.params["_retry_context"] = verdict.retry_suggestion

        return result   # return last result even if still unsatisfied

    async def _direct_llm(self, ctx: CoordinatorContext) -> str:
        """Fallback: single LLM call with memory context."""
        from app.llm.engine import llm_engine
        from app.prompts.templates import build_system_prompt
        system = build_system_prompt(ctx.memory_ctx, ctx.conv_history, "")
        response = await llm_engine.generate(
            [{"role": "user", "content": ctx.user_text}],
            system_prompt=system,
        )
        await self._archive(ctx.user_text, response)
        return response.strip()

    @staticmethod
    async def _archive(user_text: str, response: str) -> None:
        from app.memory.manager import memory_manager
        from app.memory.session_store import log_turn
        await memory_manager.add_turn(user_text, response)
        await log_turn("user", user_text)
        await log_turn("assistant", response)

    @staticmethod
    async def _background_tasks(user_text: str, results: list[StepResult]) -> None:
        """Non-blocking post-turn housekeeping."""
        try:
            from app.memory.scoring import prune_low_score_memories_async
            await prune_low_score_memories_async()
        except Exception as exc:
            logger.debug("Background curator skipped: {}", exc)

        # Skill observation — log tool sequences for pattern detection
        try:
            tool_sequence = [r.tool for r in results if r.status == "ok"]
            if len(tool_sequence) >= 2:
                from app.memory.skill_learner import skill_learner
                skill_learner.observe_sequence(user_text, tool_sequence)
        except Exception as exc:
            logger.debug("Skill observer skipped: {}", exc)


# Singleton
coordinator = Coordinator()