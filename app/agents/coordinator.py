"""
app/agents/coordinator.py — Lean 2-Agent Coordinator

The single multi-step execution path (the direct tool loop in pipeline.py is
its fallback when it times out or fails).

Architecture:
    Coordinator       — reads capabilities, builds a plan, synthesises the answer
    ToolExecutor      — runs exactly one tool per call, returns structured result

    Researcher/Curator/SkillBuilder are background services, not blocking agents.

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


# ─── Constants ────────────────────────────────────────────────────────────────

from app.tools.router import TOOL_EXECUTION_TIMEOUT_SECONDS   # re-exported; enforced inside dispatch()

# Execution limits
PLAN_GENERATION_TIMEOUT_SECONDS = 45.0  # Timeout for plan LLM call (increased from 25s)
SYNTHESIS_TIMEOUT_SECONDS = 35.0  # Timeout for synthesis LLM call (increased from 25s)
DIRECT_LLM_TIMEOUT_SECONDS = 45.0  # Timeout for fallback direct LLM (increased from 30s)

# Result size limits
MAX_STEP_RESULT_CHARS = 2000  # Max chars per step result
MAX_SYNTHESIS_RESULT_CHARS = 800  # Threshold for direct return
MAX_SYNTHESIS_CONTEXT_CHARS = 400  # Max chars per result in synthesis context

# Coordinator wait times
FALLBACK_DELAY_SECONDS = 0.2  # Wait before falling back to direct LLM

# Critic timeouts
CRITIC_EVALUATION_TIMEOUT_SECONDS = 15.0
MAX_CRITIC_RESULT_PREVIEW_CHARS = 600  # Max chars shown to critic


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
        """dispatch() never raises and enforces the per-tool timeout itself."""
        t0 = time.perf_counter()
        result = await self._get_router().dispatch({"tool": step.tool, **step.params})
        return StepResult(
            tool=step.tool,
            status=result.get("status", "ok"),
            result=str(result.get("result", ""))[:MAX_STEP_RESULT_CHARS],
            elapsed_ms=(time.perf_counter() - t0) * 1000,
        )


tool_executor = ToolExecutor()


# ─── Critic ───────────────────────────────────────────────────────────────────

_CRITIC_PROMPT = """\
Request: {request}
Action: {action}
Result: {result}

Did this satisfy the request? Answer in JSON only:
{{"satisfied": true_or_false, "reason": "brief reason", "retry_suggestion": "what to try if failed"}}
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

        # Critic disabled: accept every successful step without an LLM call.
        if not settings.critic_enabled:
            return CriticResult(satisfied=True, reason="Critic disabled.")

        try:
            from app.llm.engine import llm_engine
            raw = await llm_engine.generate_safe(
                [{"role": "user", "content": _CRITIC_PROMPT.format(
                    request=request,
                    action=step.description,
                    result=result.result[:MAX_CRITIC_RESULT_PREVIEW_CHARS],
                )}],
                system_prompt="Return only valid JSON. Be strict but fair.",
                timeout=CRITIC_EVALUATION_TIMEOUT_SECONDS,
            )
            if not raw:
                return CriticResult(satisfied=True, reason="Critic returned empty — assuming ok.")
            
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
Create a task plan in JSON format. Output ONLY JSON starting with {{ and ending with }}

FORMAT:
{{
  "goal": "brief goal description",
  "steps": [
    {{"description": "what to do", "tool": "tool_name", "params": {{"key": "value"}}, "required": true}}
  ]
}}

AVAILABLE TOOLS:
obsidian_search, obsidian_create_note, obsidian_append_daily, obsidian_read_note, obsidian_list_vault
web_search, doc_search, unified_search, ingest_document, list_documents
kg_summary, kg_add, kg_neighbors, kg_path, kg_clusters
read_file, write_file, append_file, list_directory, delete_file
save_memory, recall_memory
project_list, project_switch, project_status
system_profile, memory_scores, memory_prune, mcp_status

User request: {request}

Output JSON only. Start with {{ and end with }}
"""

def _parse_plan_json(raw: str) -> dict | None:
    """
    Robust JSON parser for LLM plan output with multiple fallback strategies.
    
    Handles common LLM output issues:
      1. Valid JSON with proper braces
      2. Missing outer braces (interior content only)
      3. Markdown fences (```json...```)
      4. Truncated output (hit max_tokens)
      5. Mixed prose + JSON
    
    Returns None if all parsing strategies fail.
    """
    if not raw or not raw.strip():
        return None
 
    # Strip markdown fences first
    clean = re.sub(r'```[a-zA-Z]*\n?|```', '', raw.strip()).strip()
 
    # Strategy 1: Standard JSON with outer braces
    if clean.startswith('{'):
        end = clean.rfind('}')
        if end != -1:
            try:
                return json.loads(clean[:end + 1])
            except json.JSONDecodeError:
                pass
 
    # Strategy 2: Extract JSON from prose (find first { to last })
    match = re.search(r'\{.*\}', clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
 
    # Strategy 3: Missing outer braces - wrap and try
    if '"goal"' in clean and '"steps"' in clean:
        try:
            return json.loads('{' + clean + '}')
        except json.JSONDecodeError:
            pass
 
    # Strategy 4: Extract steps array only (minimal recovery)
    steps_match = re.search(r'"steps"\s*:\s*(\[.*?\])', clean, re.DOTALL)
    if steps_match:
        try:
            steps = json.loads(steps_match.group(1))
            return {'goal': 'auto', 'steps': steps}
        except json.JSONDecodeError:
            pass
 
    # All strategies failed
    logger.error("Plan JSON parsing failed after 4 strategies. Raw output: {}", raw[:200])
    return None
 
 
async def _build_plan(ctx) -> list:
    """
    Ask the LLM to produce an ordered list of ToolSteps.
    Uses generate_safe() — never raises, never segfaults.
    Sequential: waits for the LLM to fully finish before returning.
    """
    try:
        from app.llm.engine import llm_engine
 
        caps_summary = ", ".join(
            k for k, v in ctx.caps.as_dict().items()
            if isinstance(v, bool) and v
        ) or "chat"
 
        raw = await llm_engine.generate_safe(
            [{"role": "user", "content": _PLAN_PROMPT.format(
                request=ctx.user_text,
            )}],
            system_prompt="Output only valid JSON. Start with { and end with }",
            max_tokens=500,
            timeout=PLAN_GENERATION_TIMEOUT_SECONDS,
        )
 
        if not raw:
            logger.warning("_build_plan: empty LLM response")
            return []
 
        data = _parse_plan_json(raw)
        if not data:
            logger.warning("_build_plan: could not parse JSON from: {}", raw[:120])
            return []
 
        from app.agents.coordinator import ToolStep
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
        logger.error("_build_plan failed: {}", exc)
        return []


# ─── Synthesis ────────────────────────────────────────────────────────────────

_SYNTH_PROMPT = """\
User asked: {request}

Tool results:
{results}

Provide a clear, natural spoken response. If steps failed, acknowledge briefly and provide what you can.
"""


async def _synthesise(ctx: CoordinatorContext) -> str:
    """Turn collected step results into a final spoken response."""
 
    # If no results at all — return fast, don't waste a 35s LLM call
    if not ctx.step_results:
        return "I wasn't able to complete that task. Please try again."
 
    ok_results  = [r for r in ctx.step_results if r.status == "ok"]
    err_results = [r for r in ctx.step_results if r.status != "ok"]
 
    # All steps failed — return error immediately, no LLM synthesis needed
    if not ok_results:
        tools_tried = ", ".join(r.tool for r in err_results)
        return (
            f"I tried to help but ran into errors with: {tools_tried}. "
            "Please check the logs or try rephrasing your request."
        )
 
    # If only one step succeeded and result is already human-readable — return it directly
    if len(ok_results) == 1 and len(ok_results[0].result) < MAX_SYNTHESIS_RESULT_CHARS:
        return ok_results[0].result
 
    # Multiple steps — synthesise with LLM
    results_text = "\n".join(
        f"[{r.tool}] ({r.status}): {r.result[:MAX_SYNTHESIS_CONTEXT_CHARS]}"
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
        response = await llm_engine.generate_safe(
            [{"role": "user", "content": _SYNTH_PROMPT.format(
                request=ctx.user_text,
                results=results_text,
            )}],
            system_prompt=system,
            timeout=SYNTHESIS_TIMEOUT_SECONDS,    # tighter timeout — synthesis should be fast
        )
        if response:
            return response.strip()
    except Exception as exc:
        logger.error("Synthesis LLM call failed: {}", exc)
 
    # Fallback — return the best result we have
    return ok_results[-1].result if ok_results else "Task completed."


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
            return await research_agent.run(user_text)

        # Build plan
        steps = await _build_plan(ctx)
        if not steps:
            logger.warning("Coordinator: plan generation failed or timed out")
            # CRITICAL SAFETY: Do NOT make another LLM call immediately after a timeout.
            # This causes segfaults in llama.cpp due to concurrent access.
            # Instead, return a simple error response without using the LLM.
            return "I'm having trouble processing that request right now. Please try again or rephrase your question."

        # Execute each step with critic evaluation
        for step in steps:
            result = await self._execute_step(step, ctx)
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

        # Background: run memory curator and skill observer
        from app.pipeline import spawn_background
        spawn_background(self._background_tasks(user_text, ctx.step_results))

        return response   # the pipeline archives the turn with what it actually speaks

    async def _execute_step(self, step: ToolStep, ctx: CoordinatorContext) -> StepResult:
        """
        Run a step once and let the critic grade it. An unsatisfied verdict is
        logged, not retried: re-running the identical call cannot change the
        result, and nothing re-plans params from the critic's suggestion.
        """
        result = await tool_executor.run(step)
        verdict = await critic.evaluate(ctx.user_text, step, result)
        logger.debug("Critic [{}]: satisfied={} — {}", step.tool, verdict.satisfied, verdict.reason[:80])
        if not verdict.satisfied:
            logger.info("Critic unsatisfied with '{}': {}", step.tool, verdict.retry_suggestion or verdict.reason)
        return result

    async def _direct_llm(self, ctx: CoordinatorContext) -> str:
        """Fallback: single LLM call with memory context."""
        from app.llm.engine import llm_engine
        from app.prompts.templates import build_system_prompt
        system = build_system_prompt(ctx.memory_ctx, ctx.conv_history, "")
        response = await llm_engine.generate_safe(
            [{"role": "user", "content": ctx.user_text}],
            system_prompt=system,
            timeout=DIRECT_LLM_TIMEOUT_SECONDS,
        )
        return response.strip()

    @staticmethod
    async def _background_tasks(user_text: str, results: list[StepResult]) -> None:
        """Non-blocking post-turn housekeeping."""
        try:
            from app.memory.scoring import prune_low_score_memories
            await prune_low_score_memories()
        except Exception as exc:
            logger.debug("Background curator skipped: {}", exc)

        # Skill observation — log tool sequences for pattern detection
        try:
            tool_sequence = [r.tool for r in results if r.status == "ok"]
            if len(tool_sequence) >= 2:
                from app.memory.skill_learner import skill_learner
                await skill_learner.log_execution(tool_sequence, user_text, "success")
        except Exception as exc:
            logger.debug("Skill observer skipped: {}", exc)


# Singleton
coordinator = Coordinator()