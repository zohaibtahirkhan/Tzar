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


# ─── Constants ────────────────────────────────────────────────────────────────

# Execution limits
MAX_STEP_RETRIES = 1  # Max retry attempts per step
TOOL_EXECUTION_TIMEOUT_SECONDS = 30.0  # Timeout per tool execution
PLAN_GENERATION_TIMEOUT_SECONDS = 25.0  # Timeout for plan LLM call
SYNTHESIS_TIMEOUT_SECONDS = 25.0  # Timeout for synthesis LLM call
DIRECT_LLM_TIMEOUT_SECONDS = 30.0  # Timeout for fallback direct LLM

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
        t0 = time.perf_counter()
        router = self._get_router()
        try:
            result = await asyncio.wait_for(
                router.dispatch({"tool": step.tool, **step.params}),
                timeout=TOOL_EXECUTION_TIMEOUT_SECONDS,
            )
            return StepResult(
                tool=step.tool,
                status=result.get("status", "ok"),
                result=str(result.get("result", ""))[:MAX_STEP_RESULT_CHARS],
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
You are a task planner for a local voice assistant.
Produce a minimal execution plan for the user request.
 
CRITICAL: Output ONLY valid JSON. No markdown fences, no prose, no explanations.
Start your response with { and end with }
 
EXACT FORMAT (copy this structure):
{
  "goal": "one sentence describing the end goal",
  "steps": [
    {"description": "what this step does", "tool": "tool_name", "params": {"param_name": "value"}, "required": true}
  ]
}
 
EXAMPLE OUTPUT:
{
  "goal": "Create a note about machine learning",
  "steps": [
    {"description": "Search vault for related notes", "tool": "obsidian_search", "params": {"query": "machine learning"}, "required": false},
    {"description": "Create new note with content", "tool": "obsidian_create_note", "params": {"title": "Machine Learning Overview", "content": "Key concepts..."}, "required": true}
  ]
}
 
TOOL REFERENCE — always include the params shown:
 
  obsidian_search          params: {"query": "<search terms>"}
  obsidian_keyword_search  params: {"query": "<keyword>"}
  obsidian_read_note       params: {"title": "<note title>"}
  obsidian_create_note     params: {"title": "<title>", "content": "<body>"}
  obsidian_append_daily    params: {"content": "<text to append>"}
  obsidian_capture_idea    params: {"content": "<idea text>"}
  obsidian_list_vault      params: {}
  obsidian_morning_briefing params: {}
  obsidian_get_related     params: {"title": "<note title>"}
 
  web_search               params: {"query": "<search query>"}
 
  doc_search               params: {"query": "<search terms>"}
  unified_search           params: {"query": "<search terms>"}
  ingest_document          params: {"path": "<file path>"}
  list_documents           params: {}
 
  kg_summary               params: {}
  kg_add                   params: {"text": "<triples text>"}
  kg_extract_and_index     params: {"text": "<raw text>", "note_title": "<title>"}
  kg_path                  params: {"source": "<node>", "target": "<node>"}
  kg_neighbors             params: {"node": "<node name>", "depth": 1}
  kg_clusters              params: {}
 
  read_file                params: {"path": "<file path>"}
  write_file               params: {"path": "<file path>", "content": "<text>"}
  append_file              params: {"path": "<file path>", "content": "<text>"}
  list_directory           params: {"path": "<directory path>"}
  delete_file              params: {"path": "<file path>"}
 
  save_memory              params: {"category": "<category>", "content": "<fact>"}
  recall_memory            params: {"query": "<search terms>"}
 
  project_list             params: {}
  project_switch           params: {"name": "<project name>"}
  system_profile           params: {}
  memory_scores            params: {}
  memory_prune             params: {}
  mcp_status               params: {}
  browser_action           params: {"task": "<what to do in browser>"}
 
Capabilities needed: {caps}
Request: {request}
 
Rules:
- Use EXACT tool names from the reference above.
- ALWAYS include ALL required params — never use empty {{}} for tools that need a query.
- Keep steps to 2–5. Single-tool tasks need exactly 1 step.
- For note searches: use obsidian_search with the search terms as query.
- For document/memory searches: use unified_search with the search terms as query.
- required: false only for optional enrichment steps.
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
                caps=caps_summary,
                request=ctx.user_text,
            )}],
            system_prompt="You are a JSON-only output agent. Return ONLY valid JSON starting with { and ending with }. No markdown, no explanations.",
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
You are a helpful voice assistant.
Original request: {request}

Results from completed steps:
{results}

Synthesise a clear, concise spoken response that directly answers the request.
If any steps failed, acknowledge it briefly and provide what you can.
Do NOT include JSON — respond in plain natural language only.
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
            response = await research_agent.run(user_text)
            await self._archive(user_text, response)
            return response

        # Build plan
        steps = await _build_plan(ctx)
        if not steps:
            logger.info("Coordinator: no plan generated, waiting {}ms then falling back to direct LLM", int(FALLBACK_DELAY_SECONDS * 1000))
            await asyncio.sleep(FALLBACK_DELAY_SECONDS)   # give lock time to fully release
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
        max_retries: int = MAX_STEP_RETRIES,
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
        response = await llm_engine.generate_safe(
            [{"role": "user", "content": ctx.user_text}],
            system_prompt=system,
            timeout=DIRECT_LLM_TIMEOUT_SECONDS,
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