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
from app.intent import classify_intent, IntentType
from app.planner import make_plan


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

    def _extract_speech_meta(self, data: dict) -> SpeechMeta:
        s = data.get("speech", {})
        return SpeechMeta(
            pace=float(s.get("pace", 1.0)),
            pause_ms=int(s.get("clause_pause_ms", 120)),
            tone=str(s.get("tone", "neutral")),
        )

    async def process_text_input(self, user_text: str) -> str:
        """
        Full pipeline:
          1. Build prompt with selective memory
          2. Classify Intent (NEW: Research vs Planning vs Chat)
          3. If tool requested -> execute -> second LLM call for final response
          4. Extract spoken response text + store speech pacing metadata
          5. Return ONLY the spoken text (never raw JSON)
        """
        logger.info("User: {}", user_text)

        memory_ctx, conv_history = await memory_manager.get_context(user_text)

        # ── NEW: Intent Classification (replaces needs_planning) ──
        intent = classify_intent(user_text)

        # ── NEW: Research Agent short-circuit ─────────────────────
        if intent == IntentType.RESEARCH:
            from app.agents.researcher import research_agent
            logger.info("Intent: RESEARCH - routing to Research Agent")
            
            # Run the specialized research agent
            response = await research_agent.run(user_text)
            
            # Archive the turn to memory (consistent with standard flow)
            await memory_manager.add_turn(user_text, response)
            await log_turn("user", user_text)
            await log_turn("assistant", response)
            
            # Return the result directly (skips standard tool loop)
            return response

        # ── Existing planner branch (updated check) ───────────────────
        plan_context = ""
        if intent == IntentType.PLANNING:
            plan = await make_plan(user_text, llm_engine.generate)
            plan_context = plan.to_context_string()
            if plan_context:
                logger.info("Plan injected into prompt.")

        system_prompt = build_system_prompt(memory_ctx, conv_history, plan_context)
        messages = [{"role": "user", "content": user_text}]

        # ── First LLM call ────────────────────────────────────────────────────
        raw = await llm_engine.generate(messages, system_prompt)
        logger.info("LLM raw: {}", raw[:300])

        data = _parse_llm_json(raw)
        tool_name   = data.get("tool")
        tool_params = data.get("tool_params") or {}

        # ── Tool execution loop (up to 4 steps) ───────────────────────────────
        iterations = 0
        while tool_name and iterations < 4:
            iterations += 1
            logger.info("Tool #{}: {} {}", iterations, tool_name, tool_params)

            try:
                result = await asyncio.wait_for(
                    self.tool_router.dispatch({"tool": tool_name, **tool_params}),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error("Tool '{}' timed out after 30s", tool_name)
                result = {"tool": tool_name, "status": "error", "result": "Tool timed out."}
                
            result_text = result["result"]
            logger.info("Tool result ({}): {}", result["status"], result_text[:200])

            followup = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Tool '{tool_name}' returned:\n{result_text}\n\n"
                        "If you need another tool, output JSON with tool and tool_params. "
                        "If done, output JSON with thought, response, and speech fields only."
                    ),
                },
            ]
            raw = await llm_engine.generate(followup, system_prompt)
            logger.info("LLM followup: {}", raw[:300])

            data = _parse_llm_json(raw)
            tool_name   = data.get("tool")
            tool_params = data.get("tool_params") or {}

        # ── Confidence: log low-confidence answers ────────────────────────────
        confidence = data.get("confidence", 0.9)
        if confidence < 0.6:
            logger.warning(
                "Low confidence response ({:.0%}) — '{}...'",
                confidence, (data.get("response") or "")[:60],
            )
            
        # ── Extract spoken response (NEVER the raw JSON) ──────────────────────
        spoken = (data.get("response") or "").strip()
        if not spoken:
            # Model didn't follow format — use the raw output as fallback
            spoken = strip_tool_calls(raw).strip()
            # If it still looks like JSON, pull just the response field value out
            if spoken.startswith("{"):
                spoken = "Done."

        # ── Store LLM pacing decisions for speak() ────────────────────────────
        self._speech_meta = self._extract_speech_meta(data)
        self._interruptible = data.get("interruptible", True)
        logger.info(
            "Speech meta: pace={} pause_ms={} tone={} interruptible={}",
            self._speech_meta.pace, self._speech_meta.pause_ms,
            self._speech_meta.tone, self._interruptible,
        )

        logger.info("Assistant: {}", spoken[:120])
        await memory_manager.add_turn(user_text, spoken)
        await log_turn("user", user_text)
        await log_turn("assistant", spoken)

        # Auto-suggest skill creation after complex multi-tool tasks
        if iterations >= 3:
            logger.info("Complex task ({} tool calls) — LLM may want to save a skill.", iterations)

        return spoken

    async def process_text_input_streaming(self, user_text: str) -> AsyncIterator[str]:
        """
        Resolves the full response (including any tool calls), then yields
        clause-by-clause for low-latency TTS chaining.
        Never yields raw JSON — only the spoken response text.
        """
        logger.info("User (stream): {}", user_text)

        memory_ctx, conv_history = await memory_manager.get_context(user_text)

        # ── NEW: Intent Classification ─────────────────────────────────────────
        intent = classify_intent(user_text)

        # ── NEW: Research Agent short-circuit ─────────────────────────────────
        if intent == IntentType.RESEARCH:
            from app.agents.researcher import research_agent
            logger.info("Intent: RESEARCH (stream) - routing to Research Agent")
            
            # Yield a status update so UI knows work is happening
            yield "[Researching...]"
            
            response = await research_agent.run(user_text)
            
            # Archive
            await memory_manager.add_turn(user_text, response)
            await log_turn("user", user_text)
            await log_turn("assistant", response)
            
            # Stream the result out sentence by sentence
            for clause in split_into_sentences(response):
                if clause.strip():
                    yield clause
            return # Stop execution here

        # ── Planner for complex multi-step ────────────────────────────────
        plan_context = ""
        if intent == IntentType.PLANNING:
            plan = await make_plan(user_text, llm_engine.generate)
            plan_context = plan.to_context_string()
            if plan_context:
                logger.info("Plan injected into prompt.")

        system_prompt = build_system_prompt(memory_ctx, conv_history, plan_context)
        messages = [{"role": "user", "content": user_text}]

        raw = await llm_engine.generate(messages, system_prompt)
        logger.info("LLM raw (stream): {}", raw[:300])

        data = _parse_llm_json(raw)
        tool_name   = data.get("tool")
        tool_params = data.get("tool_params") or {}

        iterations = 0
        while tool_name and iterations < 4:
            iterations += 1
            yield f"[Running {tool_name}...]"   # status token — caller can display/ignore

            try:
                result = await asyncio.wait_for(
                    self.tool_router.dispatch({"tool": tool_name, **tool_params}),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error("Tool '{}' timed out after 30s", tool_name)
                result = {"tool": tool_name, "status": "error", "result": "Tool timed out."}
                
            result_text = result["result"]

            followup = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Tool '{tool_name}' returned:\n{result_text}\n\n"
                        "If done, output JSON with thought, response, and speech fields only."
                    ),
                },
            ]
            raw = await llm_engine.generate(followup, system_prompt)
            data = _parse_llm_json(raw)
            tool_name   = data.get("tool")
            tool_params = data.get("tool_params") or {}

        # ── Confidence: log low-confidence answers ────────────────────────────
        confidence = data.get("confidence", 0.9)
        if confidence < 0.6:
            logger.warning(
                "Low confidence response ({:.0%}) — '{}...'",
                confidence, (data.get("response") or "")[:60],
            )
            
        spoken = (data.get("response") or "").strip()
        if not spoken:
            spoken = strip_tool_calls(raw).strip()
        if spoken.startswith("{"):
            spoken = "Done."

        self._speech_meta = self._extract_speech_meta(data)
        self._interruptible = data.get("interruptible", True)
        logger.info(
            "Speech meta: pace={} pause_ms={} tone={} interruptible={}",
            self._speech_meta.pace, self._speech_meta.pause_ms,
            self._speech_meta.tone, self._interruptible,
        )
        await memory_manager.add_turn(user_text, spoken)
        await log_turn("user", user_text)
        await log_turn("assistant", spoken)

        # Yield clause by clause so caller can pipe directly into TTS
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
        woke_at: Optional[float] = time.perf_counter()

        async for chunk in microphone.chunks():
            if not self._active:
                break

            if not awaiting_speech:
                awaiting_speech = True
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