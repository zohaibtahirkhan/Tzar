"""
Core pipeline — ties together VAD → STT → LLM → Tool Router → TTS → Playback.

Fixes applied in this version:
  1. Uses classify() (multi-label Capabilities) instead of classify_intent()
  2. Coordinator routing with proper capability check
  3. Goal tracking continuation check
  4. Memory save short-circuit (actually writes to DB)
  5. Tool result truncation (prevents context overflow from large results)
  6. _clean_llm_response() strips "Assistant: Assistant:" prefix pollution
  7. Fixed project_switch false positive via updated extract_project_name_from_query
  8. "Good morning" routes to obsidian_morning_briefing tool
  9. generate_safe() used in coordinator; generate() always returns str
 10. print(full_messages) removed from engine
 11. Repeated tool loop detection (seen_calls set)
 12. Tool duration tracking for SSE
 13. strip_tool_calls_safe() None/empty guard
 14. zip_longest for plan step dicts (handles length mismatch)
 15. Try/except around _parse_llm_json() in streaming
 16. interrupt() logs when blocked due to non-interruptible flag
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass
from itertools import zip_longest
from typing import Optional, AsyncIterator

import numpy as np
from loguru import logger

from app.config import settings
from app.audio.stt import stt_engine
from app.audio.tts import tts_engine, audio_player, split_into_sentences
from app.audio.vad import vad_engine, SpeechCollector
from app.audio.wake_word import wake_word_detector
from app.audio.microphone import microphone
from app.llm.engine import llm_engine
from app.memory.manager import memory_manager
from app.tools.router import ToolRouter, extract_tool_calls, strip_tool_calls
from app.prompts.templates import build_system_prompt, format_tool_result
from app.utils.timing import LatencyMetrics, Timer
from app.memory.session_store import log_turn
from app.intent import classify, classify_intent, IntentType, Capabilities
from app.planner import make_plan, needs_planning
from app.planning.goal_tracker import goal_tracker


# ─── Speech metadata ─────────────────────────────────────────────────────────

@dataclass
class SpeechMeta:
    pace:     float = 1.0
    pause_ms: int   = 120
    tone:     str   = "neutral"


_DEFAULT_SPEECH = SpeechMeta()


# ─── Response cleaning ────────────────────────────────────────────────────────

_ASSISTANT_PREFIX_RE = re.compile(r'^(Assistant:\s*)+', re.IGNORECASE | re.MULTILINE)
_USER_SAYS_RE        = re.compile(r'\n+User\s+says?\s*["\']?.*', re.IGNORECASE | re.DOTALL)


def _clean_llm_response(text) -> str:
    """
    Strip hallucinated role prefixes and fake conversation continuations.
    Safe to call with None or empty string — always returns a str.
    """
    if not text:
        return ""
    text = str(text).strip()
    text = _ASSISTANT_PREFIX_RE.sub('', text).strip()
    text = _USER_SAYS_RE.sub('', text).strip()
    return text


def strip_tool_calls_safe(raw) -> str:
    """
    None/empty-safe wrapper around strip_tool_calls.
    The LLM can return "" (or, in edge cases, None) from generate(),
    which would crash the bare strip_tool_calls(raw) call.
    """
    if not raw:
        return ""
    return strip_tool_calls(str(raw))


# ─── Tool result truncation ───────────────────────────────────────────────────

_TOOL_RESULT_MAX_CHARS = 1200   # ~340 tokens — safe budget for followup call


def _truncate_tool_result(result_text: str, tool_name: str) -> str:
    """
    Cap large tool results so they don't blow the 4096-token context.
    list_documents (667 docs) and system_profile are the main offenders.
    """
    if not result_text or len(result_text) <= _TOOL_RESULT_MAX_CHARS:
        return result_text or ""

    truncated  = result_text[:_TOOL_RESULT_MAX_CHARS]
    last_nl    = truncated.rfind('\n')
    if last_nl > _TOOL_RESULT_MAX_CHARS * 0.6:
        truncated = truncated[:last_nl]

    total_lines = result_text.count('\n')
    shown_lines = truncated.count('\n')
    return truncated + f"\n[...{total_lines - shown_lines} more lines truncated]"


# ─── JSON parser ─────────────────────────────────────────────────────────────

def _parse_llm_json(raw: str) -> dict:
    """
    Parse JSON from LLM output robustly.
    Always returns a dict — never raises.
    """
    if not raw:
        return {
            "tool": None, "tool_params": None,
            "response": "",
            "speech": {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"},
        }

    clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()

    # Try outermost braces
    if clean.startswith('{'):
        end = clean.rfind('}')
        if end != -1:
            try:
                data = json.loads(clean[:end + 1])
                if "speech" not in data or not isinstance(data["speech"], dict):
                    data["speech"] = {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"}
                if "confidence" not in data:
                    data["confidence"] = 0.9
                return data
            except json.JSONDecodeError:
                pass

    logger.warning("JSON parse failed. Treating raw output as plain response. Raw: {}", raw[:200])
    return {
        "tool": None, "tool_params": None,
        "response": _clean_llm_response(clean),
        "speech": {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"},
        "confidence": 0.9,
    }


# ─── Morning briefing detection ───────────────────────────────────────────────

_MORNING_RE = re.compile(
    r'^(good\s+morning|morning|buenos\s+dias|bonjour|sabah\s+al[\s-]khayr|صباح\s+الخير)\s*[!.,]?\s*$',
    re.IGNORECASE,
)


def _is_morning_greeting(text: str) -> bool:
    return bool(_MORNING_RE.match(text.strip()))


# ─── Memory save triggers (compiled once) ────────────────────────────────────

_SAVE_TRIGGERS = re.compile(
    r'\b(remember that|remember this|save this|note that|'
    r'my name is|i prefer|i use|i am working on|i\'m working on|'
    r'my .{1,20} is|i like|i dislike|i hate|i love|i always|i never)\b',
    re.IGNORECASE,
)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class AssistantPipeline:

    def __init__(self):
        self.tool_router      = ToolRouter(memory_manager=memory_manager)
        self._active          = False
        self._speaking        = False
        self._interrupted     = False
        self._speech_meta     = SpeechMeta()
        self._interruptible   = True
        self._last_tool_results: list[dict] = []   # exposed for SSE streaming

    def _extract_speech_meta(self, data: dict) -> SpeechMeta:
        s = data.get("speech", {})
        return SpeechMeta(
            pace=float(s.get("pace", 1.0)),
            pause_ms=int(s.get("clause_pause_ms", 120)),
            tone=str(s.get("tone", "neutral")),
        )

    # ── Shared helpers ────────────────────────────────────────────────────────

    async def _run_tool(self, tool_name: str, tool_params: dict) -> tuple[dict, int]:
        """Run one tool with timeout. Returns (result_dict, duration_ms)."""
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.tool_router.dispatch({"tool": tool_name, **tool_params}),
                timeout=30.0,
            )
            return result, int((time.monotonic() - start) * 1000)
        except asyncio.TimeoutError:
            logger.error("Tool '{}' timed out after 30s", tool_name)
            return {"tool": tool_name, "status": "error", "result": "Tool timed out."}, 30000
        except Exception:
            logger.exception("Tool '{}' failed", tool_name)
            return {"tool": tool_name, "status": "error", "result": "Tool execution failed."}, int((time.monotonic() - start) * 1000)

    async def _archive_turn(self, user_text: str, spoken: str) -> None:
        await memory_manager.add_turn(user_text, spoken)
        await log_turn("user", user_text)
        await log_turn("assistant", spoken)

    # ── Memory save short-circuit ─────────────────────────────────────────────

    async def _try_memory_save(self, user_text: str) -> Optional[str]:
        """
        If the user is explicitly saving a fact, write it to long-term memory
        directly and return a confirmation string.
        Returns None if this isn't a save request.
        """
        from app.intent import classify
        caps = classify(user_text)
        if not caps.needs_memory:
            return None

        if not _SAVE_TRIGGERS.search(user_text):
            return None

        category = "preference" if any(w in user_text.lower() for w in
                   ("prefer", "like", "dislike", "hate", "love", "always", "never")) else "fact"
        try:
            await memory_manager.long_term.save(category=category, content=user_text.strip())
            spoken = "Got it, I've saved that."
            await self._archive_turn(user_text, spoken)
            return spoken
        except Exception as exc:
            logger.warning("Memory save short-circuit failed: {} — continuing to LLM", exc)
            return None

    # ── Main text pipeline ────────────────────────────────────────────────────

    async def process_text_input(self, user_text: str) -> str:
        logger.info("User: {}", user_text)

        # ── Morning briefing fast-path ────────────────────────────────────────
        if _is_morning_greeting(user_text):
            logger.info("Morning greeting detected — running morning briefing tool")
            result, _ = await self._run_tool("obsidian_morning_briefing", {})
            spoken = _clean_llm_response(result.get("result", "Good morning!"))
            if not spoken:
                spoken = "Good morning! Your workspace is ready."
            await self._archive_turn(user_text, spoken)
            return spoken

        memory_ctx, conv_history = await memory_manager.get_context(user_text)

        # ── Multi-label capabilities classification ───────────────────────────
        from app.intent import classify
        caps = classify(user_text)
        logger.info("Capabilities: {}", caps.as_dict())

        # ── Memory save short-circuit ─────────────────────────────────────────
        saved = await self._try_memory_save(user_text)
        if saved:
            return saved

        # ── Project continuity ────────────────────────────────────────────────
        from app.memory.projects import extract_project_name_from_query, project_switch
        project_name = extract_project_name_from_query(user_text)
        if project_name:
            spoken = await project_switch(project_name)
            await self._archive_turn(user_text, spoken)
            return spoken

        # ── Goal continuation ─────────────────────────────────────────────────
        try:
            if goal_tracker.is_continuation(user_text):
                active = await goal_tracker.get_active_goals()
                if active:
                    spoken = await goal_tracker.resume_response(active[0])
                    await self._archive_turn(user_text, spoken)
                    return spoken
        except Exception:
            pass

        # ── Coordinator (multi-step complex tasks) ────────────────────────────
        if settings.multi_agent_enabled and caps.is_complex():
            from app.agents.coordinator import coordinator
            logger.info("Routing to Coordinator — capabilities: {}", caps.primary)
            try:
                result = await asyncio.wait_for(
                    coordinator.run(user_text, caps),
                    timeout=120.0,
                )
                return result
            except asyncio.TimeoutError:
                logger.error("Coordinator timed out — falling back to direct pipeline")
            except Exception as exc:
                logger.error("Coordinator error: {} — falling back", exc)

        # ── Research agent ────────────────────────────────────────────────────
        if caps.needs_research and not caps.needs_tools:
            from app.agents.researcher import research_agent
            logger.info("Routing to ResearchAgent")
            response = await research_agent.run(user_text)
            await self._archive_turn(user_text, response)
            return response

        # ── Planning ─────────────────────────────────────────────────────────
        plan_context = ""
        if caps.needs_planning:
            plan = await make_plan(user_text, llm_engine.generate)
            plan_context = plan.to_context_string()
            if plan_context:
                logger.info("Plan injected into prompt.")
                if len(plan.steps) >= 3:
                    step_dicts = [
                        {"description": step, "tool": tool or "", "params": {}}
                        for step, tool in zip_longest(plan.steps, plan.required_tools, fillvalue="")
                    ]
                    asyncio.create_task(goal_tracker.create_goal(user_text, step_dicts))

        system_prompt = build_system_prompt(memory_ctx, conv_history, plan_context)
        messages      = [{"role": "user", "content": user_text}]

        # ── First LLM call ────────────────────────────────────────────────────
        raw = await llm_engine.generate(messages, system_prompt)
        logger.info("LLM raw: {}", raw[:300])

        data        = _parse_llm_json(raw)
        tool_name   = data.get("tool")
        tool_params = data.get("tool_params") or {}

        # ── Tool execution loop ───────────────────────────────────────────────
        self._last_tool_results = []
        iterations  = 0
        seen_calls  = set()

        while tool_name and iterations < 4:
            iterations += 1

            # Detect repeated tool loops
            try:
                call_sig = (tool_name, json.dumps(tool_params, sort_keys=True))
            except Exception:
                call_sig = (tool_name, str(tool_params))

            if call_sig in seen_calls:
                logger.warning("Repeated tool loop detected: {}", tool_name)
                break
            seen_calls.add(call_sig)

            logger.info("Tool #{}: {} {}", iterations, tool_name, tool_params)

            result, duration_ms = await self._run_tool(tool_name, tool_params)
            result_text = _truncate_tool_result(str(result.get("result", "")), tool_name)
            logger.info("Tool result ({} | {}ms): {}", result.get("status", "ok"), duration_ms, result_text[:200])

            self._last_tool_results.append({
                "tool":        result.get("tool", tool_name),
                "status":      result.get("status", "ok"),
                "result":      result_text[:800],
                "duration_ms": duration_ms,
            })

            followup = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"Tool '{tool_name}' returned:\n{result_text}\n\n"
                    "If you need another tool, output JSON with tool and tool_params. "
                    "If done, output JSON with thought, response, and speech fields only."
                )},
            ]
            raw         = await llm_engine.generate(followup, system_prompt)
            logger.info("LLM followup: {}", raw[:300])
            data        = _parse_llm_json(raw)
            tool_name   = data.get("tool")
            tool_params = data.get("tool_params") or {}

        # ── Confidence monitoring ─────────────────────────────────────────────
        confidence = data.get("confidence", 0.9)
        if confidence < 0.6:
            logger.warning("Low confidence ({:.0%}) — '{}...'", confidence, (data.get("response") or "")[:60])

        # ── Build spoken response ─────────────────────────────────────────────
        spoken = data.get("response") or ""
        spoken = str(spoken).strip()
        if not spoken:
            spoken = strip_tool_calls_safe(raw).strip()
        spoken = _clean_llm_response(spoken)
        if not spoken or spoken.startswith("{"):
            spoken = "I'm sorry, I wasn't able to generate a response. Please try again."

        self._speech_meta   = self._extract_speech_meta(data)
        self._interruptible = data.get("interruptible", True)

        logger.info("Speech meta: pace={} pause_ms={} tone={} interruptible={}",
                    self._speech_meta.pace, self._speech_meta.pause_ms,
                    self._speech_meta.tone, self._interruptible)
        logger.info("Assistant: {}", spoken[:120])
        await self._archive_turn(user_text, spoken)

        if iterations >= 3:
            logger.info("Complex task ({} tool calls).", iterations)

        return spoken

    # ── Streaming pipeline ────────────────────────────────────────────────────

    async def process_text_input_streaming(self, user_text: str) -> AsyncIterator[str]:
        logger.info("User (stream): {}", user_text)

        # Reset per-request state
        self._last_tool_results = []

        # ── Morning briefing fast-path ────────────────────────────────────────
        if _is_morning_greeting(user_text):
            result, _ = await self._run_tool("obsidian_morning_briefing", {})
            spoken = _clean_llm_response(result.get("result", "Good morning!")) or "Good morning!"
            await self._archive_turn(user_text, spoken)
            for clause in split_into_sentences(spoken):
                if clause.strip():
                    yield clause
            return

        memory_ctx, conv_history = await memory_manager.get_context(user_text)

        from app.intent import classify
        caps = classify(user_text)
        logger.info("Capabilities (stream): {}", caps.as_dict())

        # ── Memory save short-circuit ─────────────────────────────────────────
        saved = await self._try_memory_save(user_text)
        if saved:
            yield saved
            return

        # ── Project continuity ────────────────────────────────────────────────
        from app.memory.projects import extract_project_name_from_query, project_switch
        project_name = extract_project_name_from_query(user_text)
        if project_name:
            result = await project_switch(project_name)
            await self._archive_turn(user_text, result)
            for clause in split_into_sentences(result):
                if clause.strip():
                    yield clause
            return

        # ── Goal continuation ─────────────────────────────────────────────────
        try:
            if goal_tracker.is_continuation(user_text):
                active = await goal_tracker.get_active_goals()
                if active:
                    spoken = await goal_tracker.resume_response(active[0])
                    await self._archive_turn(user_text, spoken)
                    for clause in split_into_sentences(spoken):
                        if clause.strip():
                            yield clause
                    return
        except Exception:
            pass

        # ── Coordinator ───────────────────────────────────────────────────────
        if settings.multi_agent_enabled and caps.is_complex():
            from app.agents.coordinator import coordinator
            logger.info("Routing to Coordinator (stream) — {}", caps.primary)
            try:
                result = await asyncio.wait_for(
                    coordinator.run(user_text, caps), timeout=120.0,
                )
                await self._archive_turn(user_text, result)
                for clause in split_into_sentences(result):
                    if clause.strip():
                        yield clause
                return
            except asyncio.TimeoutError:
                logger.error("Coordinator timed out — falling back to direct pipeline")
            except Exception as exc:
                logger.error("Coordinator error (stream): {}", exc)

        # ── Research agent ────────────────────────────────────────────────────
        if caps.needs_research and not caps.needs_tools:
            from app.agents.researcher import research_agent
            yield "[Researching...]"
            response = await research_agent.run(user_text)
            await self._archive_turn(user_text, response)
            for clause in split_into_sentences(response):
                if clause.strip():
                    yield clause
            return

        # ── Planning ─────────────────────────────────────────────────────────
        plan_context = ""
        if caps.needs_planning:
            plan = await make_plan(user_text, llm_engine.generate)
            plan_context = plan.to_context_string()
            if plan_context:
                logger.info("Plan injected into prompt (stream)")
                if len(plan.steps) >= 3:
                    step_dicts = [
                        {"description": step, "tool": tool or "", "params": {}}
                        for step, tool in zip_longest(plan.steps, plan.required_tools, fillvalue="")
                    ]

                    async def _persist_goal():
                        try:
                            await goal_tracker.create_goal(user_text, step_dicts)
                        except Exception:
                            logger.exception("Failed to persist goal")

                    asyncio.create_task(_persist_goal())

        # ── Prompt build ─────────────────────────────────────────────────────
        system_prompt = build_system_prompt(memory_ctx, conv_history, plan_context)
        messages      = [{"role": "user", "content": user_text}]

        # ── First LLM call ────────────────────────────────────────────────────
        raw = await llm_engine.generate(messages, system_prompt)
        logger.info("LLM raw (stream): {}", raw[:300])

        try:
            data = _parse_llm_json(raw)
        except Exception:
            logger.exception("Failed to parse LLM JSON response")
            data = {"response": strip_tool_calls_safe(raw).strip(), "confidence": 0.3}

        tool_name   = data.get("tool")
        tool_params = data.get("tool_params") or {}

        # ── Tool execution loop ───────────────────────────────────────────────
        iterations = 0
        seen_calls = set()

        while tool_name and iterations < 4:
            iterations += 1

            # Detect repeated tool loops
            try:
                call_sig = (tool_name, json.dumps(tool_params, sort_keys=True))
            except Exception:
                call_sig = (tool_name, str(tool_params))

            if call_sig in seen_calls:
                logger.warning("Repeated tool loop detected: {}", tool_name)
                break
            seen_calls.add(call_sig)

            logger.info("Tool #{} (stream): {} {}", iterations, tool_name, tool_params)
            yield f"[Running {tool_name}...]"

            result, duration_ms = await self._run_tool(tool_name, tool_params)
            result_text = _truncate_tool_result(str(result.get("result", "")), tool_name)

            logger.info("Tool result ({} | {}ms): {}", result.get("status", "ok"), duration_ms, result_text[:200])

            self._last_tool_results.append({
                "tool":        result.get("tool", tool_name),
                "status":      result.get("status", "ok"),
                "result":      result_text[:800],
                "duration_ms": duration_ms,
            })

            followup = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": (
                    f"Tool '{tool_name}' returned:\n{result_text}\n\n"
                    "If you need another tool, output JSON with tool and tool_params. "
                    "If done, output JSON with thought, response, and speech fields only."
                )},
            ]
            raw = await llm_engine.generate(followup, system_prompt)

            try:
                data = _parse_llm_json(raw)
            except Exception:
                logger.exception("Failed to parse follow-up LLM JSON")
                data = {"response": strip_tool_calls_safe(raw).strip(), "confidence": 0.3}
                break

            tool_name   = data.get("tool")
            tool_params = data.get("tool_params") or {}

        # ── Confidence monitoring ─────────────────────────────────────────────
        confidence = data.get("confidence", 0.9)
        if confidence < 0.6:
            logger.warning("Low confidence ({:.0%}) — '{}...'", confidence, (data.get("response") or "")[:60])

        # ── Final spoken response ─────────────────────────────────────────────
        spoken = data.get("response") or ""
        spoken = str(spoken).strip()
        if not spoken:
            spoken = strip_tool_calls_safe(raw).strip()
        spoken = _clean_llm_response(spoken)
        if not spoken or spoken.startswith("{"):
            spoken = "I'm sorry, I wasn't able to generate a response. Please try again."

        # ── Speech metadata ───────────────────────────────────────────────────
        self._speech_meta   = self._extract_speech_meta(data)
        self._interruptible = data.get("interruptible", True)

        logger.info("Speech meta: pace={} pause_ms={} tone={} interruptible={}",
                    self._speech_meta.pace, self._speech_meta.pause_ms,
                    self._speech_meta.tone, self._interruptible)

        # ── Memory + logs ─────────────────────────────────────────────────────
        await self._archive_turn(user_text, spoken)

        # ── Stream clause-by-clause ───────────────────────────────────────────
        for clause in split_into_sentences(spoken):
            if clause.strip():
                yield clause

    # ── TTS / Voice ───────────────────────────────────────────────────────────

    async def speak(self, text: str) -> None:
        if not tts_engine.is_loaded():
            logger.warning("TTS not loaded; skipping audio output")
            return
        if not text or not text.strip():
            logger.warning("speak() called with empty text — skipping")
            return

        meta = self._speech_meta
        logger.debug("Speaking (pace={} pause={}ms tone={}): {}",
                     meta.pace, meta.pause_ms, meta.tone, text[:80])

        self._speaking    = True
        self._interrupted = False
        t0 = time.perf_counter()
        first_chunk = True

        try:
            async for audio_chunk in tts_engine.stream_sentences(text, speed=meta.pace):
                if self._interrupted:
                    logger.info("TTS interrupted mid-speech")
                    break
                if first_chunk:
                    logger.debug("TTS first audio: {:.0f}ms", (time.perf_counter() - t0) * 1000)
                    first_chunk = False
                await audio_player.play_async(audio_chunk)
                if meta.pause_ms > 0 and not self._interrupted:
                    await asyncio.sleep(meta.pause_ms / 1000)
        finally:
            self._speaking = False

    def interrupt(self) -> None:
        if self._speaking and self._interruptible:
            self._interrupted = True
            logger.debug("TTS interrupted (interruptible=True)")
        elif self._speaking and not self._interruptible:
            logger.debug("Interrupt blocked — response marked non-interruptible")

    async def run_voice_loop(self) -> None:
        self._active = True
        collector = SpeechCollector(vad_engine)

        logger.info("=" * 50)
        logger.info("Voice assistant active. Say '{}' to begin.", settings.wake_word_phrase)
        logger.info("=" * 50)

        awaiting_speech = False
        woke_at: Optional[float] = None

        async for chunk in microphone.chunks():
            if not self._active:
                break

            if not awaiting_speech:
                confidence = wake_word_detector.check_chunk(chunk)
                if wake_word_detector.detected(confidence):
                    self.interrupt()
                    logger.info("Wake word detected! (confidence={:.2f})", confidence)
                    print("\n🎙️  Listening...", flush=True)
                    awaiting_speech = True
                    woke_at = time.perf_counter()
                    collector.reset()
            else:
                utterance = collector.push(chunk)
                if utterance is not None:
                    awaiting_speech = False
                    logger.debug("Utterance collected in {:.0f}ms",
                                 (time.perf_counter() - woke_at) * 1000)
                    print("⏳  Transcribing...", flush=True)
                    user_text = await stt_engine.transcribe_async(utterance)
                    logger.info("STT: {}", user_text)

                    if not user_text.strip():
                        print("(no speech detected)", flush=True)
                        continue

                    print(f"You: {user_text}", flush=True)
                    lower = user_text.lower().strip()

                    if any(x in lower for x in ["enable web search", "search the web", "go online"]):
                        from app.tools.web_search import enable_web_search
                        self._speech_meta = SpeechMeta(pace=1.0, pause_ms=80, tone="neutral")
                        msg = enable_web_search()
                        print(f"Assistant: {msg}", flush=True)
                        await self.speak(msg)
                        continue

                    if any(x in lower for x in ["disable web search", "go offline"]):
                        from app.tools.web_search import disable_web_search
                        self._speech_meta = SpeechMeta(pace=1.0, pause_ms=80, tone="neutral")
                        msg = disable_web_search()
                        print(f"Assistant: {msg}", flush=True)
                        await self.speak(msg)
                        continue

                    if lower in ("quit", "exit", "goodbye", "bye"):
                        self._speech_meta = SpeechMeta(pace=1.0, pause_ms=80, tone="warm")
                        await self.speak("Goodbye!")
                        self._active = False
                        break

                    print("🧠  Thinking...", flush=True)
                    response = await self.process_text_input(user_text)
                    print(f"Assistant: {response}", flush=True)
                    await self.speak(response)

                    print("\n" + "─" * 40, flush=True)
                    print(f"Say '{settings.wake_word_phrase}' again when ready.", flush=True)

    def stop(self) -> None:
        self._active = False


# Singleton
pipeline = AssistantPipeline()