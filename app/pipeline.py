"""
Core pipeline: ties together VAD → STT → LLM → Tool Router → TTS → Playback.

Streaming architecture:
  User Speaking
    ↓
  Silero VAD (end-of-utterance detection)
    ↓
  Faster-Whisper STT
    ↓
  Qwen2.5 3B (llama.cpp) — streaming tokens
    ↓
  Tool Router (if tool calls present)
    ↓
  Kokoro TTS — sentence-by-sentence
    ↓
  Speaker
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass
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


# ── Speech state set by LLM each turn ────────────────────────────────────────

@dataclass
class SpeechMeta:
    pace: float = 1.0        # TTS speed multiplier
    pause_ms: int = 120      # silence between clauses in ms
    tone: str = "neutral"


_DEFAULT_SPEECH = SpeechMeta()


def _parse_llm_json(raw: str) -> dict:
    """
    Parse JSON from LLM output robustly.
    Strips markdown fences, finds the first {...} block.
    Always returns a dict with keys: thought, tool, tool_params, response, speech.
    """
    clean = re.sub(r"^```[a-zA-Z]*\n?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if "speech" not in data or not isinstance(data["speech"], dict):
                data["speech"] = {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"}
            if "confidence" not in data:
                data["confidence"] = 0.9
            return data
        except json.JSONDecodeError:
            pass
    logger.warning("JSON parse failed. Treating raw output as plain response. Raw: {}", raw[:200])
    return {
        "tool": None,
        "tool_params": None,
        "response": clean,
        "speech": {"pace": 1.0, "clause_pause_ms": 120, "tone": "neutral"},
    }


class AssistantPipeline:
    """
    Stateful pipeline for one session.
    Manages the full listen → think → speak loop.
    """

    def __init__(self):
        self.tool_router = ToolRouter(memory_manager=memory_manager)
        self._active = False
        self._speaking = False
        self._interrupted = False
        self._speech_meta = SpeechMeta()   # updated each turn by LLM
        self._interruptible = True
        self._last_tool_results: list[dict] = []

    def _extract_speech_meta(self, data: dict) -> SpeechMeta:
        s = data.get("speech", {})
        return SpeechMeta(
            pace=float(s.get("pace", 1.0)),
            pause_ms=int(s.get("clause_pause_ms", 120)),
            tone=str(s.get("tone", "neutral")),
        )

    async def process_text_input(self, user_text: str) -> str:
        """
        Full pipeline with multi-label capabilities routing and goal tracking.
        """
        logger.info("User: {}", user_text)
    
        memory_ctx, conv_history = await memory_manager.get_context(user_text)
    
        # ── Multi-label capabilities classification (replaces single-label intent) ─
        caps = classify(user_text)
        logger.info("Capabilities: {}", caps.as_dict())
        
        # ── Memory write short-circuit ────────────────────────────────────────────

        _SAVE_TRIGGERS = re.compile(
            r"\b("
            r"remember that|"
            r"remember this|"
            r"save this|"
            r"note that|"
            r"my name is|"
            r"i prefer|"
            r"i use|"
            r"i am working on"
            r")\b",
            re.IGNORECASE,
        )

        if caps.needs_memory and _SAVE_TRIGGERS.search(user_text):

            fact = user_text.strip()

            category = (
                "preference"
                if "prefer" in user_text.lower()
                else "fact"
            )

            try:
                await memory_manager.long_term.save(
                    category=category,
                    content=fact,
                )
            except Exception:
                logger.exception("Failed to save memory")
                return "I couldn't save that memory."

            response = "Got it, I've saved that."

            await memory_manager.add_turn(
                user_text,
                response,
            )

            await log_turn("user", user_text)
            await log_turn("assistant", response)

            return response
    
        # ── Project continuity ────────────────────────────────────────────────────
        from app.memory.projects import extract_project_name_from_query, project_switch
        project_name = extract_project_name_from_query(user_text)
        if project_name:
            return await project_switch(project_name)
    
        # ── Goal continuation check ───────────────────────────────────────────────
        if goal_tracker.is_continuation(user_text):
            active_goals = await goal_tracker.get_active_goals()
            if active_goals:
                return await goal_tracker.resume_response(active_goals[0])
    
        # ── Coordinator (multi-step, multi-capability) ────────────────────────────
        if settings.multi_agent_enabled and caps.is_complex():
            from app.agents.coordinator import coordinator
            logger.info("Routing to Coordinator — capabilities: {}", caps.primary)
            return await coordinator.run(user_text, caps)
    
        # ── Legacy 5-agent orchestrator (experimental) ────────────────────────────
        if getattr(settings, "multi_agent_experimental", False):
            from app.agents.orchestrator import orchestrator
            return await orchestrator.run(user_text, classify_intent(user_text).value)
    
        # ── Research Agent ─────────────────────────────────────────────────────────
        if caps.needs_research and not caps.needs_tools:
            from app.agents.researcher import research_agent
            logger.info("Routing to ResearchAgent")
            response = await research_agent.run(user_text)
            await memory_manager.add_turn(user_text, response)
            await log_turn("user", user_text)
            await log_turn("assistant", response)
            return response
    
        # ── Planning: persist goal if multi-step ─────────────────────────────────
        plan_context = ""
        if caps.needs_planning:
            plan = await make_plan(user_text, llm_engine.generate)
            plan_context = plan.to_context_string()
            if plan_context:
                logger.info("Plan injected into prompt")
                # Persist the goal for long-horizon tracking
                if len(plan.steps) >= 3:
                    step_dicts = [
                        {"description": s, "tool": t, "params": {}}
                        for s, t in zip(plan.steps, plan.required_tools + [""] * len(plan.steps))
                    ]
                    asyncio.create_task(
                        goal_tracker.create_goal(user_text, step_dicts)
                    )
    
        system_prompt = build_system_prompt(memory_ctx, conv_history, plan_context)
        messages = [{"role": "user", "content": user_text}]
    
        # ── First LLM call ────────────────────────────────────────────────────────
        raw = await llm_engine.generate(messages, system_prompt)
        logger.info("LLM raw: {}", raw[:300])
        data = _parse_llm_json(raw)
        tool_name   = data.get("tool")
        tool_params = data.get("tool_params") or {}
    
        # ── Tool execution loop (up to 4 steps) ───────────────────────────────────
        self._last_tool_results = []
        iterations = 0
        while tool_name and iterations < 4:
            iterations += 1
            logger.info("Tool #{}: {} {}", iterations, tool_name, tool_params)
            try:
                result = await asyncio.wait_for(
                    self.tool_router.dispatch({"tool": tool_name, **tool_params}),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                result = {"tool": tool_name, "status": "error", "result": "Tool timed out."}
    
            result_text = result["result"]
            logger.info("Tool result ({}): {}", result["status"], result_text[:200])
    
            # Track for SSE emission
            self._last_tool_results.append({
                "tool":   result.get("tool", tool_name),
                "status": result.get("status", "ok"),
                "result": str(result_text)[:800],
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
            data        = _parse_llm_json(raw)
            tool_name   = data.get("tool")
            tool_params = data.get("tool_params") or {}
    
        confidence = data.get("confidence", 0.9)
        if confidence < 0.6:
            logger.warning("Low confidence ({:.0%})", confidence)
    
        spoken = (data.get("response") or "").strip()
        if not spoken:
            spoken = strip_tool_calls(raw).strip()
        if spoken.startswith("{"):
            spoken = "Done."
    
        self._speech_meta   = self._extract_speech_meta(data)
        self._interruptible = data.get("interruptible", True)
    
        logger.info("Assistant: {}", spoken[:120])
        await memory_manager.add_turn(user_text, spoken)
        await log_turn("user", user_text)
        await log_turn("assistant", spoken)
    
        return spoken

    async def process_text_input_streaming(
        self,
        user_text: str
    ) -> AsyncIterator[str]:
        """
        Full streaming pipeline with:
        - capability-based routing
        - project continuity
        - goal continuation
        - coordinator escalation
        - planning support
        - research short-circuit
        - resilient tool execution
        - SSE-safe tool tracking
        - low-latency clause streaming
        """

        logger.info("User (stream): {}", user_text)

        # IMPORTANT:
        # This assumes pipeline instances are request-scoped.
        # If not, move these into request-local state.
        self._last_tool_results = []

        memory_ctx, conv_history = await memory_manager.get_context(user_text)

        # ─────────────────────────────────────────────────────────────
        # Multi-label capability classification
        # ─────────────────────────────────────────────────────────────

        caps = classify(user_text)
        logger.info("Capabilities (stream): {}", caps.as_dict())

        # ── Memory write short-circuit ────────────────────────────────────────────
        _SAVE_TRIGGERS = re.compile(
            r"\b("
            r"remember that|"
            r"remember this|"
            r"save this|"
            r"note that|"
            r"my name is|"
            r"i prefer|"
            r"i use|"
            r"i am working on"
            r")\b",
            re.IGNORECASE,
        )

        if caps.needs_memory and _SAVE_TRIGGERS.search(user_text):

            fact = user_text.strip()

            category = (
                "preference"
                if "prefer" in user_text.lower()
                else "fact"
            )

            try:
                await memory_manager.long_term.save(
                    category=category,
                    content=fact,
                )
            except Exception:
                logger.exception("Failed to save memory")
                return "I couldn't save that memory."

            response = "Got it, I've saved that."

            await memory_manager.add_turn(
                user_text,
                response,
            )

            await log_turn("user", user_text)
            await log_turn("assistant", response)

            return response

        # ─────────────────────────────────────────────────────────────
        # Project continuity auto-detect
        # ─────────────────────────────────────────────────────────────

        from app.memory.projects import (
            extract_project_name_from_query,
            project_switch,
        )

        project_name = extract_project_name_from_query(user_text)

        if project_name:
            result = await project_switch(project_name)

            for clause in split_into_sentences(result):
                if clause.strip():
                    yield clause

            return

        # ─────────────────────────────────────────────────────────────
        # Goal continuation
        # ─────────────────────────────────────────────────────────────

        if goal_tracker.is_continuation(user_text):

            active_goals = await goal_tracker.get_active_goals()

            if active_goals:
                result = await goal_tracker.resume_response(active_goals[0])

                for clause in split_into_sentences(result):
                    if clause.strip():
                        yield clause

                return

        # ─────────────────────────────────────────────────────────────
        # Multi-agent coordinator
        # ─────────────────────────────────────────────────────────────

        if settings.multi_agent_enabled and caps.is_complex():

            from app.agents.coordinator import coordinator

            logger.info(
                "Routing to Coordinator (stream) — capabilities: {}",
                caps.primary,
            )

            result = await coordinator.run(user_text, caps)

            await memory_manager.add_turn(user_text, result)
            await log_turn("user", user_text)
            await log_turn("assistant", result)

            for clause in split_into_sentences(result):
                if clause.strip():
                    yield clause

            return

        # ─────────────────────────────────────────────────────────────
        # Legacy experimental orchestrator
        # ─────────────────────────────────────────────────────────────

        if getattr(settings, "multi_agent_experimental", False):

            from app.agents.orchestrator import orchestrator

            result = await orchestrator.run(
                user_text,
                classify_intent(user_text).value,
            )

            await memory_manager.add_turn(user_text, result)
            await log_turn("user", user_text)
            await log_turn("assistant", result)

            for clause in split_into_sentences(result):
                if clause.strip():
                    yield clause

            return

        # ─────────────────────────────────────────────────────────────
        # Research agent short-circuit
        # ─────────────────────────────────────────────────────────────

        if caps.needs_research and not caps.needs_tools:

            from app.agents.researcher import research_agent

            logger.info("Routing to ResearchAgent (stream)")

            yield "[Researching...]"

            response = await research_agent.run(user_text)

            await memory_manager.add_turn(user_text, response)
            await log_turn("user", user_text)
            await log_turn("assistant", response)

            for clause in split_into_sentences(response):
                if clause.strip():
                    yield clause

            return

        # ─────────────────────────────────────────────────────────────
        # Planning
        # ─────────────────────────────────────────────────────────────

        plan_context = ""

        if caps.needs_planning:

            plan = await make_plan(user_text, llm_engine.generate)

            plan_context = plan.to_context_string()

            if plan_context:
                logger.info("Plan injected into prompt (stream)")

            # Persist long-horizon goals
            if len(plan.steps) >= 3:

                from itertools import zip_longest

                step_dicts = [
                    {
                        "description": step,
                        "tool": tool or "",
                        "params": {},
                    }
                    for step, tool in zip_longest(
                        plan.steps,
                        plan.required_tools,
                        fillvalue="",
                    )
                ]

                async def _persist_goal():
                    try:
                        await goal_tracker.create_goal(
                            user_text,
                            step_dicts,
                        )
                    except Exception:
                        logger.exception(
                            "Failed to persist goal"
                        )

                asyncio.create_task(_persist_goal())

        # ─────────────────────────────────────────────────────────────
        # Prompt build
        # ─────────────────────────────────────────────────────────────

        system_prompt = build_system_prompt(
            memory_ctx,
            conv_history,
            plan_context,
        )

        messages = [
            {
                "role": "user",
                "content": user_text,
            }
        ]

        # ─────────────────────────────────────────────────────────────
        # First LLM call
        # ─────────────────────────────────────────────────────────────

        raw = await llm_engine.generate(
            messages,
            system_prompt,
        )

        logger.info("LLM raw (stream): {}", raw[:300])

        try:
            data = _parse_llm_json(raw)

        except Exception:

            logger.exception(
                "Failed to parse LLM JSON response"
            )

            data = {
                "response": strip_tool_calls(raw).strip(),
                "confidence": 0.3,
            }

        tool_name = data.get("tool")
        tool_params = data.get("tool_params") or {}

        # ─────────────────────────────────────────────────────────────
        # Tool execution loop
        # ─────────────────────────────────────────────────────────────

        iterations = 0
        seen_calls = set()

        while tool_name and iterations < 4:

            iterations += 1

            # Detect repeated tool loops
            try:
                call_sig = (
                    tool_name,
                    json.dumps(tool_params, sort_keys=True),
                )
            except Exception:
                call_sig = (
                    tool_name,
                    str(tool_params),
                )

            if call_sig in seen_calls:

                logger.warning(
                    "Repeated tool loop detected: {}",
                    tool_name,
                )

                break

            seen_calls.add(call_sig)

            logger.info(
                "Tool #{} (stream): {} {}",
                iterations,
                tool_name,
                tool_params,
            )

            yield f"[Running {tool_name}...]"

            import time

            start = time.monotonic()

            try:

                result = await asyncio.wait_for(
                    self.tool_router.dispatch(
                        {
                            "tool": tool_name,
                            **tool_params,
                        }
                    ),
                    timeout=30.0,
                )

            except asyncio.TimeoutError:

                logger.error(
                    "Tool '{}' timed out after 30s",
                    tool_name,
                )

                result = {
                    "tool": tool_name,
                    "status": "error",
                    "result": "Tool timed out.",
                }

            except Exception:

                logger.exception(
                    "Tool '{}' failed",
                    tool_name,
                )

                result = {
                    "tool": tool_name,
                    "status": "error",
                    "result": "Tool execution failed.",
                }

            duration_ms = int(
                (time.monotonic() - start) * 1000
            )

            result_text = str(
                result.get("result", "")
            )

            logger.info(
                "Tool result ({} | {}ms): {}",
                result.get("status", "ok"),
                duration_ms,
                result_text[:200],
            )

            # Track tool results for SSE/UI
            self._last_tool_results.append(
                {
                    "tool": result.get(
                        "tool",
                        tool_name,
                    ),
                    "status": result.get(
                        "status",
                        "ok",
                    ),
                    "result": result_text[:800],
                    "duration_ms": duration_ms,
                }
            )

            # ─────────────────────────────────────────────────────────
            # Follow-up LLM call
            # ─────────────────────────────────────────────────────────

            followup = messages + [
                {
                    "role": "assistant",
                    "content": raw,
                },
                {
                    "role": "user",
                    "content": (
                        f"Tool '{tool_name}' returned:\n"
                        f"{result_text}\n\n"
                        "If you need another tool, "
                        "output JSON with tool and tool_params. "
                        "If done, output JSON with "
                        "thought, response, and speech fields only."
                    ),
                },
            ]

            raw = await llm_engine.generate(
                followup,
                system_prompt,
            )

            try:

                data = _parse_llm_json(raw)

            except Exception:

                logger.exception(
                    "Failed to parse follow-up LLM JSON"
                )

                data = {
                    "response": strip_tool_calls(raw).strip(),
                    "confidence": 0.3,
                }

                break

            tool_name = data.get("tool")
            tool_params = data.get("tool_params") or {}

        # ─────────────────────────────────────────────────────────────
        # Confidence monitoring
        # ─────────────────────────────────────────────────────────────

        confidence = data.get("confidence", 0.9)

        if confidence < 0.6:

            logger.warning(
                "Low confidence response ({:.0%}) — '{}...'",
                confidence,
                (data.get("response") or "")[:60],
            )

        # ─────────────────────────────────────────────────────────────
        # Final spoken response
        # ─────────────────────────────────────────────────────────────

        spoken = (data.get("response") or "").strip()

        if not spoken:
            spoken = strip_tool_calls(raw).strip()

        if spoken.startswith("{"):
            spoken = "Done."

        # ─────────────────────────────────────────────────────────────
        # Speech metadata
        # ─────────────────────────────────────────────────────────────

        self._speech_meta = self._extract_speech_meta(
            data
        )

        self._interruptible = data.get(
            "interruptible",
            True,
        )

        logger.info(
            "Speech meta: "
            "pace={} pause_ms={} tone={} "
            "interruptible={}",
            self._speech_meta.pace,
            self._speech_meta.pause_ms,
            self._speech_meta.tone,
            self._interruptible,
        )

        # ─────────────────────────────────────────────────────────────
        # Memory + logs
        # ─────────────────────────────────────────────────────────────

        await memory_manager.add_turn(
            user_text,
            spoken,
        )

        await log_turn("user", user_text)
        await log_turn("assistant", spoken)

        # ─────────────────────────────────────────────────────────────
        # Stream clause-by-clause
        # ─────────────────────────────────────────────────────────────

        for clause in split_into_sentences(spoken):

            if clause.strip():
                yield clause
            
    async def speak(self, text: str) -> None:
        """
        Synthesize and play text using the pacing the LLM decided this turn.
        pace and pause_ms come from self._speech_meta, set by process_text_input.
        For system messages (not going through LLM), defaults are used.
        """
        if not tts_engine.is_loaded():
            logger.warning("TTS not loaded; skipping audio output")
            return

        if not text or not text.strip():
            logger.warning("speak() called with empty text — skipping")
            return

        meta = self._speech_meta
        logger.debug("Speaking (pace={} pause={}ms tone={}): {}", meta.pace, meta.pause_ms, meta.tone, text[:80])

        self._speaking = True
        self._interrupted = False

        # Apply LLM-decided speed to TTS engine for this utterance
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
        """Stop TTS immediately — only if the current response is interruptible."""
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
                    self.interrupt()  # stop TTS if speaking
                    logger.info("Wake word detected! (confidence={:.2f})", confidence)
                    print("\n🎙️  Listening...", flush=True)
                    awaiting_speech = True
                    woke_at = time.perf_counter()
                    collector.reset()
            else:
                utterance = collector.push(chunk)
                if utterance is not None:
                    awaiting_speech = False
                    logger.debug("Utterance collected in {:.0f}ms", (time.perf_counter() - woke_at) * 1000)

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