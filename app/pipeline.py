"""
Core pipeline — ties together VAD → STT → LLM → Tool Router → TTS → Playback.

There is one implementation of a turn, `_turn()`, an async generator of
TurnEvents. Everything else is a view over it:

    stream(text)             → TurnEvents as they happen (SSE, live TTS)
    run_turn(text)           → the final TurnResult (API, WebSocket, tests)
    process_text_input(text) → just the spoken string (terminal)

A turn is: answer without the LLM if we can (previous-turn sources, cache,
morning briefing, memory save, project switch, goal continuation, coordinator,
research agent); otherwise build the prompt, call the LLM, run the tools it
asks for (bounded), and speak its final JSON "response". The "response" string
is forwarded as it is generated — the prompt puts the speech metadata before
it, so the voice loop can start talking before the model has finished.
"""
import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

from loguru import logger

from app.config import settings
from app.audio.stt import stt_engine
from app.audio.tts import tts_engine, audio_player, split_into_clauses
from app.audio.vad import vad_engine, SpeechCollector
from app.audio.wake_word import wake_word_detector
from app.audio.microphone import microphone
from app.llm.engine import llm_engine, parse_json_object, LLMUnavailableError
from app.memory.manager import memory_manager
from app.tools.router import ToolRouter, TOOL_EXECUTION_TIMEOUT_SECONDS   # re-exported; enforced in dispatch()
from app.prompts.templates import build_system_prompt, CITE_INSTRUCTION
from app.memory.session_store import log_turn
from app.intent import classify, Capabilities
from app.planner import make_plan, Plan
from app.planning.goal_tracker import goal_tracker
from app.cache import response_cache
from app.tools.rag.citations import (
    Source, start_source_collection, attribution_sentence, is_source_question, describe_sources,
)


# ─── Constants ────────────────────────────────────────────────────────────────

TOOL_RESULT_MAX_CHARS = 1200          # ~340 tokens — safe budget for the follow-up call
TOOL_RESULT_MAX_DISPLAY_CHARS = 800   # for the UI's expandable tool rows
MAX_TOOL_ITERATIONS = 4               # tool calls per turn
COORDINATOR_TIMEOUT_SECONDS = 120.0
MAX_RESPONSE_PREVIEW_CHARS = 120      # log preview length

# Retrieval prefetch for needs_rag queries
RETRIEVAL_PREFETCH_TIMEOUT_SECONDS = 8.0
MAX_RETRIEVAL_CONTEXT_CHARS = 2500    # keep the prompt within the token budget

# Tools that mutate remembered state. The response cache is keyed on query
# text alone, so after one of these runs a cached answer may be stale.
_CACHE_INVALIDATING_TOOLS = {
    "save_memory", "memory_write", "memory_remove", "memory_replace", "memory_prune",
    "project_switch", "project_new", "project_update", "project_archive",
}

FALLBACK_RESPONSE = "I'm sorry, I wasn't able to generate a response. Please try again."

_background_tasks: set[asyncio.Task] = set()


def spawn_background(coro) -> asyncio.Task:
    """create_task() that keeps a reference: the loop holds tasks weakly, so an unreferenced one can vanish mid-run."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


# ─── Turn data ────────────────────────────────────────────────────────────────

@dataclass
class SpeechMeta:
    pace:     float = 1.0
    pause_ms: int   = 120
    tone:     str   = "neutral"


@dataclass
class TurnResult:
    """Everything a caller may want from one finished turn."""
    response: str
    tool_results: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    speech: SpeechMeta = field(default_factory=SpeechMeta)
    interruptible: bool = True


@dataclass
class TurnEvent:
    """
    kind = "status": progress line for the UI ("[Running doc_search...]\\n")
    kind = "meta":   how to speak what follows (`speech`, `interruptible`)
    kind = "text":   a piece of the spoken answer, in order
    kind = "done":   the turn is over; `result` is set
    """
    kind: str
    text: str = ""
    speech: SpeechMeta = field(default_factory=SpeechMeta)
    interruptible: bool = True
    result: Optional[TurnResult] = None


# ─── LLM output handling ──────────────────────────────────────────────────────

_ASSISTANT_PREFIX_RE = re.compile(r'^(Assistant:\s*)+', re.IGNORECASE | re.MULTILINE)
_USER_SAYS_RE        = re.compile(r'\n+User\s+says?\s*["\']?.*', re.IGNORECASE | re.DOTALL)


def _clean_llm_response(text) -> str:
    """For non-JSON output only: strip hallucinated role prefixes and fake continuations."""
    if not text:
        return ""
    text = str(text).strip()
    text = _ASSISTANT_PREFIX_RE.sub('', text).strip()
    text = _USER_SAYS_RE.sub('', text).strip()
    return text


def _truncate_tool_result(result_text: str) -> str:
    """Cap large tool results (list_documents, system_profile) so they don't blow the context."""
    if not result_text or len(result_text) <= TOOL_RESULT_MAX_CHARS:
        return result_text or ""
    truncated = result_text[:TOOL_RESULT_MAX_CHARS]
    last_nl = truncated.rfind('\n')
    if last_nl > TOOL_RESULT_MAX_CHARS * 0.6:
        truncated = truncated[:last_nl]
    hidden = result_text.count('\n') - truncated.count('\n')
    return truncated + f"\n[...{hidden} more lines truncated]"


def _clamp(value, default: float, lo: float, hi: float) -> float:
    try:
        return min(max(float(value), lo), hi)
    except (TypeError, ValueError):
        return default


def _normalise_reply(data: dict) -> dict:
    """
    Coerce the LLM's JSON to the types the pipeline relies on. Every field is
    untrusted: "pace": "fast" or "confidence": "high" must not crash a turn.
    Defaults come from SpeechMeta so they are defined once.
    """
    speech = data.get("speech") if isinstance(data.get("speech"), dict) else {}
    data["speech"] = {
        "pace":            _clamp(speech.get("pace"), SpeechMeta.pace, 0.5, 2.0),
        "clause_pause_ms": int(_clamp(speech.get("clause_pause_ms"), SpeechMeta.pause_ms, 0, 2000)),
        "tone":            str(speech.get("tone") or SpeechMeta.tone),
    }
    data["confidence"]    = _clamp(data.get("confidence"), 0.9, 0.0, 1.0)
    data["interruptible"] = data.get("interruptible", True) not in (False, 0, "false", "False")
    data.setdefault("tool", None)
    data.setdefault("tool_params", None)
    data["response"] = str(data.get("response") or "")
    return data


def _extract_speech_meta(data: dict) -> SpeechMeta:
    """SpeechMeta from a reply dict (normalised or raw)."""
    s = _normalise_reply(dict(data))["speech"]
    return SpeechMeta(pace=s["pace"], pause_ms=s["clause_pause_ms"], tone=s["tone"])


class _ResponseFieldStream:
    """
    Incrementally extracts the JSON "response" string while the LLM is still
    generating, so the answer can be forwarded token by token.

    The prompt puts every other field before "response", so when the string
    opens we already know whether this reply is a tool call (then nothing is
    forwarded — the text belongs to a follow-up turn, not the user) and how it
    should be spoken (`head`). Only complete escape sequences are decoded; a
    partial `\\u12` waits for more.
    """
    _OPEN      = re.compile(r'"response"\s*:\s*"')
    _TOOL_NULL = re.compile(r'"tool"\s*:\s*null')
    # A malformed \u (as in C:\users) is consumed as a pair so it can't stall the stream.
    _BODY      = re.compile(r'(?:[^"\\]|\\u[0-9a-fA-F]{4}|\\[^u]|\\u(?=[0-9a-fA-F]{0,3}[^0-9a-fA-F]))*')

    def __init__(self):
        self._buf = ""          # unconsumed text only — trimmed as we go
        self._active = False
        self._closed = False
        self.head: dict = {}    # the fields that preceded "response", once it opens

    def feed(self, token: str) -> str:
        """Return the newly available decoded response text (often "")."""
        if self._closed:
            return ""
        self._buf += token
        if not self._active:
            m = self._OPEN.search(self._buf)
            if not m:
                return ""
            prefix = self._buf[:m.start()]
            if not self._TOOL_NULL.search(prefix):
                self._closed = True          # tool call (or unknown) — don't stream
                return ""
            self.head = parse_json_object(prefix.rstrip().rstrip(",") + "}") or {}
            self._active = True
            self._buf = self._buf[m.end():]
        end = self._BODY.match(self._buf).end()
        segment, self._buf = self._buf[:end], self._buf[end:]
        if self._buf.startswith('"'):
            self._closed = True
        return _decode_json_string(segment) if "\\" in segment else segment


_ESCAPE_RE = re.compile(r'\\(u[0-9a-fA-F]{4}|.)', re.DOTALL)


def _repair_escape(m: re.Match) -> str:
    esc = m.group(1)
    if len(esc) == 5 or esc in '"\\/bfnrt':
        return m.group(0)                        # valid JSON escape
    return esc if esc == "'" else "\\\\" + esc   # \' → '; anything else keeps its backslash


def _decode_json_string(body: str) -> str:
    """Decode a JSON string body. Small models emit escapes JSON lacks (\\', C:\\users) — keep their text."""
    try:
        return json.loads(f'"{body}"', strict=False)
    except json.JSONDecodeError:
        return json.loads('"' + _ESCAPE_RE.sub(_repair_escape, body) + '"', strict=False)


def _parse_llm_json(raw: str) -> dict:
    """
    The LLM's reply as a normalised dict — never raises. Valid JSON is the
    normal case; a malformed reply (duplicate keys, cut-off tail) is salvaged
    with the same grammar the streamer uses; as a last resort the whole
    output is treated as the spoken response.
    """
    data = parse_json_object(raw)
    if data is None:
        data = _salvage_reply(raw or "")
    return _normalise_reply(data)


def _salvage_reply(raw: str) -> dict:
    tool = re.search(r'"tool"\s*:\s*"([^"]+)"', raw)
    if tool:
        params = re.search(r'"tool_params"\s*:\s*(\{[^}]*\})', raw)
        logger.warning("JSON parse failed; salvaged tool call {}", tool.group(1))
        return {"tool": tool.group(1), "tool_params": (parse_json_object(params.group(1)) if params else None) or {}}
    response = _ResponseFieldStream().feed(raw).strip()
    if response and not response.startswith("{"):
        logger.warning("JSON parse failed; salvaged response '{}...'", response[:60])
        return {"response": response}
    logger.warning("JSON parse failed completely. Raw: {}", raw[:200])
    return {"response": _clean_llm_response(raw)}


# ─── Zero-LLM triggers ────────────────────────────────────────────────────────

_MORNING_RE = re.compile(
    r'^(good\s+morning|morning|buenos\s+dias|bonjour|sabah\s+al[\s-]khayr|صباح\s+الخير)\s*[!.,]?\s*$',
    re.IGNORECASE,
)


def _is_morning_greeting(text: str) -> bool:
    return bool(_MORNING_RE.match(text.strip()))


_SAVE_TRIGGERS = re.compile(
    r'\b(remember that|remember this|save this|note that|'
    r'my name is|i prefer|i use|i am working on|i\'m working on|'
    r'my .{1,20} is|i like|i dislike|i hate|i love|i always|i never)\b',
    re.IGNORECASE,
)

_PREFERENCE_WORDS = ("prefer", "like", "dislike", "hate", "love", "always", "never")


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class AssistantPipeline:

    def __init__(self):
        self.tool_router = ToolRouter(memory_manager=memory_manager)
        self._active = False
        # Speaking state, owned by _play()/interrupt().
        self._speaking = False
        self._interrupted = False
        self._interruptible = True
        # The one cross-turn value: "where did that come from?" asks about it.
        self._last_sources: list[Source] = []

    _extract_speech_meta = staticmethod(_extract_speech_meta)   # kept callable on the instance

    # ── Public views over one turn ────────────────────────────────────────────

    def stream(self, user_text: str) -> AsyncIterator[TurnEvent]:
        """Events as they happen, with the LLM's answer forwarded token by token."""
        return self._turn(user_text)

    async def run_turn(self, user_text: str) -> TurnResult:
        """Run a whole turn and return its result."""
        async for event in self._turn(user_text):
            if event.kind == "done":
                return event.result
        raise RuntimeError("turn ended without a result")   # _turn always emits done

    async def process_text_input(self, user_text: str) -> str:
        return (await self.run_turn(user_text)).response

    # ── The one turn implementation ───────────────────────────────────────────

    async def _turn(self, user_text: str) -> AsyncIterator[TurnEvent]:
        logger.info("User: {}", user_text)

        # "Where did that come from?" is about the previous turn's sources, so
        # it is answered before those are reset (and before the cache).
        spoken = await self._answer_source_question(user_text)
        if spoken is None:
            self._last_sources = start_source_collection()
            caps = classify(user_text)
            logger.info("Capabilities: {}", caps.as_dict())
            cache_key = self._cache_key(user_text)     # before this turn joins the history
            spoken = (await self._cached(cache_key)
                      or await self._morning_briefing(user_text)
                      or await self._try_memory_save(user_text, caps)
                      or await self._project_switch(user_text)
                      or await self._try_goal_continuation(user_text))
            if spoken is None and settings.multi_agent_enabled and caps.is_complex():
                yield TurnEvent("status", "[Planning...]\n")
                spoken = await self._coordinate(user_text, caps)
            if spoken is None and caps.needs_research and not caps.needs_tools:
                yield TurnEvent("status", "[Researching...]\n")
                spoken = await self._research(user_text)
        if spoken is not None:
            await self._archive_turn(user_text, spoken)
            yield TurnEvent("text", spoken)
            yield TurnEvent("done", result=TurnResult(spoken, sources=self._sources()))
            return

        # ── Prompt (the three inputs are independent I/O — overlap them) ─────
        (memory_ctx, conv_history), plan, retrieval_ctx = await asyncio.gather(
            memory_manager.get_context(user_text),
            make_plan(user_text, llm_engine.generate) if caps.needs_planning else asyncio.sleep(0, None),
            self._prefetch_retrieval(user_text) if caps.needs_rag else asyncio.sleep(0, ""),
        )
        plan_context = plan.to_context_string() if plan else ""
        system_prompt = build_system_prompt(memory_ctx, conv_history, plan_context, retrieval_ctx)
        messages = [{"role": "user", "content": user_text}]

        # ── LLM + bounded tool loop ───────────────────────────────────────────
        tool_results: list[dict] = []
        seen_calls: set[tuple] = set()
        for iteration in range(MAX_TOOL_ITERATIONS + 1):
            raw, streamed, extractor = "", "", _ResponseFieldStream()
            try:
                async for token in llm_engine.generate_stream(messages, system_prompt):
                    raw += token
                    if delta := extractor.feed(token):
                        if not streamed:     # first text: the head tells us how to speak it
                            head = _normalise_reply(extractor.head)
                            yield TurnEvent("meta", speech=_extract_speech_meta(head), interruptible=head["interruptible"])
                        streamed += delta
                        yield TurnEvent("text", delta)
            except LLMUnavailableError as exc:
                # Tell the user how to fix it, on every channel. Not cached or
                # archived: it's about the setup, not the conversation.
                logger.error("LLM unavailable: {}", exc)
                yield TurnEvent("text", str(exc))
                yield TurnEvent("done", result=TurnResult(str(exc), tool_results, self._sources()))
                return
            logger.info("LLM raw: {}", raw[:300])
            data = _parse_llm_json(raw)
            tool_name, tool_params = data["tool"], data["tool_params"] or {}
            if not tool_name or iteration == MAX_TOOL_ITERATIONS:
                break

            call_sig = (tool_name, json.dumps(tool_params, sort_keys=True, default=str))
            if call_sig in seen_calls:
                logger.warning("Repeated tool loop detected: {}", tool_name)
                break
            seen_calls.add(call_sig)

            logger.info("Tool #{}: {} {}", iteration + 1, tool_name, tool_params)
            yield TurnEvent("status", f"[Running {tool_name}...]\n")
            result, duration_ms = await self._run_tool(tool_name, tool_params)
            result_text = _truncate_tool_result(str(result.get("result", "")))
            logger.info("Tool result ({} | {}ms): {}", result.get("status", "ok"), duration_ms, result_text[:200])
            tool_results.append({
                "tool":        result.get("tool", tool_name),
                "status":      result.get("status", "ok"),
                "result":      result_text[:TOOL_RESULT_MAX_DISPLAY_CHARS],
                "duration_ms": duration_ms,
            })
            # Keep the whole exchange: a later step may need an earlier result.
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": self._followup_prompt(tool_name, result_text)},
            ]

        # ── Final answer: exactly what was streamed, plus attribution ─────────
        if data["confidence"] < 0.6:
            logger.warning("Low confidence ({:.0%}) — '{}...'", data["confidence"], data["response"][:60])
        spoken = data["response"].strip()
        if not spoken or spoken.startswith("{"):
            spoken = FALLBACK_RESPONSE
        if not streamed.strip():
            yield TurnEvent("text", spoken)
        if attribution := attribution_sentence(self._last_sources, spoken):
            yield TurnEvent("text", attribution)
            spoken += attribution

        speech, interruptible = _extract_speech_meta(data), data["interruptible"]
        logger.info("Assistant: {}  ({}, interruptible={})", spoken[:MAX_RESPONSE_PREVIEW_CHARS], speech, interruptible)
        await self._archive_turn(user_text, spoken)
        self._persist_goal(user_text, plan, tool_results)
        # Cache conversational answers only: tool and retrieval answers depend
        # on state that changes underneath the cache.
        if not tool_results and not caps.needs_tools and not caps.needs_rag:
            await response_cache.set(cache_key, spoken)

        yield TurnEvent("done", result=TurnResult(spoken, tool_results, self._sources(), speech, interruptible))

    # ── Zero-LLM answers (each returns the spoken text or None) ──────────────

    async def _answer_source_question(self, user_text: str) -> Optional[str]:
        if not is_source_question(user_text):
            return None
        logger.info("Source question — {} source(s) on record", len(self._last_sources))
        return describe_sources(self._last_sources)

    @staticmethod
    def _cache_key(user_text: str) -> str:
        # The answer depends on the conversation so far — "why?" means something
        # different after every turn — so the key includes it.
        return f"{memory_manager.short_term.format_for_prompt()}\n{user_text}"

    async def _cached(self, cache_key: str) -> Optional[str]:
        spoken = await response_cache.get(cache_key)
        if spoken:
            logger.info("Cache HIT")
        return spoken

    async def _morning_briefing(self, user_text: str) -> Optional[str]:
        if not _is_morning_greeting(user_text):
            return None
        logger.info("Morning greeting — running the briefing tool")
        result, _ = await self._run_tool("obsidian_morning_briefing", {})
        return str(result.get("result") or "").strip() or "Good morning! Your workspace is ready."

    async def _try_memory_save(self, user_text: str, caps: Capabilities) -> Optional[str]:
        """Explicit 'remember that ...' — write straight to long-term memory."""
        if not caps.needs_memory or not _SAVE_TRIGGERS.search(user_text):
            return None
        lowered = user_text.lower()
        category = "preference" if any(w in lowered for w in _PREFERENCE_WORDS) else "fact"
        try:
            await memory_manager.long_term.save(category=category, content=user_text.strip())
            await response_cache.clear()      # "what are my preferences?" is now stale
        except Exception as exc:
            logger.warning("Memory save short-circuit failed: {} — continuing to LLM", exc)
            return None
        return "Got it, I've saved that."

    async def _project_switch(self, user_text: str) -> Optional[str]:
        from app.memory.projects import extract_project_name_from_query, project_switch
        name = extract_project_name_from_query(user_text)
        return await project_switch(name) if name else None

    async def _try_goal_continuation(self, user_text: str) -> Optional[str]:
        """'continue' / 'what's next' — resume the most recent in-progress goal."""
        try:
            if not goal_tracker.is_continuation(user_text):
                return None
            active = await goal_tracker.get_active_goals()
            return await goal_tracker.resume_response(active[0]) if active else None
        except Exception as exc:
            logger.warning("Goal continuation check failed: {} — continuing", exc)
            return None

    async def _coordinate(self, user_text: str, caps: Capabilities) -> Optional[str]:
        """Multi-step tasks go to the Coordinator; on failure fall through to the direct loop."""
        from app.agents.coordinator import coordinator
        logger.info("Routing to Coordinator — {}", caps.primary)
        try:
            result = await asyncio.wait_for(coordinator.run(user_text, caps), timeout=COORDINATOR_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.error("Coordinator timed out — falling back to direct pipeline")
            return None
        except Exception as exc:
            logger.error("Coordinator error: {} — falling back", exc)
            return None
        return result + attribution_sentence(self._last_sources, result)

    async def _research(self, user_text: str) -> str:
        from app.agents.researcher import research_agent
        logger.info("Routing to ResearchAgent")
        return await research_agent.run(user_text)

    # ── Turn helpers ──────────────────────────────────────────────────────────

    def _sources(self) -> list[dict]:
        return [s.as_dict() for s in self._last_sources]

    async def _run_tool(self, tool_name: str, tool_params: dict) -> tuple[dict, int]:
        """Run one tool (dispatch() enforces the timeout and never raises). Returns (result, duration_ms)."""
        start = time.monotonic()
        result = await self.tool_router.dispatch({"tool": tool_name, **tool_params})
        if tool_name in _CACHE_INVALIDATING_TOOLS and result.get("status") == "ok":
            await response_cache.clear()
        return result, int((time.monotonic() - start) * 1000)

    async def _prefetch_retrieval(self, user_text: str) -> str:
        """
        For retrieval queries, search notes and documents up front and hand the
        passages to the LLM directly: one fewer round-trip and a grounded answer.
        """
        try:
            from app.tools.rag import unified_search
            result = await asyncio.wait_for(unified_search(user_text, top_k=5), timeout=RETRIEVAL_PREFETCH_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            logger.warning("Retrieval prefetch timed out — LLM will call search tools itself")
            return ""
        except Exception as exc:
            logger.warning("Retrieval prefetch failed: {} — continuing without it", exc)
            return ""
        if not self._last_sources:      # nothing retrieved — search tools recorded no sources
            return ""
        return result[:MAX_RETRIEVAL_CONTEXT_CHARS]

    def _followup_prompt(self, tool_name: str, result_text: str) -> str:
        # A retrieval tool has run when the request's source bucket is non-empty.
        cite = f"When the answer comes from a search result, {CITE_INSTRUCTION}. " if self._last_sources else ""
        return (
            f"Tool '{tool_name}' returned:\n{result_text}\n\n"
            "If you need another tool, output JSON with tool and tool_params. "
            f"{cite}If done, output JSON with the speech fields first and response last."
        )

    async def _archive_turn(self, user_text: str, spoken: str) -> None:
        await memory_manager.add_turn(user_text, spoken)
        await log_turn("user", user_text)
        await log_turn("assistant", spoken)

    def _persist_goal(self, user_text: str, plan: Optional[Plan], tool_results: list[dict]) -> None:
        """
        Record a multi-step plan as a goal once the turn has run it. The plan
        was executed in this turn, so its steps are done when every tool call
        succeeded; otherwise they stay pending for "continue". Background task:
        the DB write never delays the answer.
        """
        if plan is None or plan.can_answer_directly or len(plan.steps) < 3:
            return
        status = "done" if tool_results and all(r.get("status") == "ok" for r in tool_results) else "pending"
        step_dicts = [{"description": step, "status": status} for step in plan.steps]

        async def _run():
            try:
                await goal_tracker.create_goal(user_text, step_dicts, title=plan.goal[:80])
            except Exception:
                logger.exception("Failed to persist goal")

        spawn_background(_run())

    # ── TTS ───────────────────────────────────────────────────────────────────

    async def speak(self, text: str, meta: Optional[SpeechMeta] = None, interruptible: bool = True) -> None:
        """Speak a whole text. interrupt() cuts it short unless interruptible=False."""
        self._interrupted = False
        await self._play(text, meta or SpeechMeta(), interruptible)

    async def _play(self, text: str, meta: SpeechMeta, interruptible: bool) -> None:
        """Synthesise and play one piece of text clause by clause; stops when interrupted."""
        if not tts_engine.is_loaded():
            logger.warning("TTS not loaded; skipping audio output")
            return
        if not text.strip() or self._interrupted:
            return
        logger.debug("Speaking (pace={} pause={}ms tone={}): {}", meta.pace, meta.pause_ms, meta.tone, text[:80])
        self._speaking, self._interruptible = True, interruptible
        try:
            async for audio_chunk in tts_engine.stream_sentences(text, speed=meta.pace):
                if self._interrupted:
                    logger.info("TTS interrupted mid-speech")
                    break
                await audio_player.play_async(audio_chunk)
                if meta.pause_ms > 0 and not self._interrupted:
                    await asyncio.sleep(meta.pause_ms / 1000)
        finally:
            self._speaking = False

    def interrupt(self) -> None:
        if self._speaking and self._interruptible:
            self._interrupted = True
            audio_player.stop()
            logger.debug("TTS interrupted (interruptible=True) - audio stopped and cleaned up")
        elif self._speaking and not self._interruptible:
            logger.debug("Interrupt blocked — response marked non-interruptible")

    # ── Voice loop ────────────────────────────────────────────────────────────

    async def run_voice_loop(self) -> None:
        """
        Wake word → collect an utterance → transcribe → respond (as a task).
        The response task runs the turn and speaks each clause as soon as the
        model has finished it, while this loop keeps reading the microphone:
        saying the wake word again cancels the answer — unless the LLM marked
        it non-interruptible, in which case it is allowed to finish first.
        """
        self._active = True
        collector = SpeechCollector(vad_engine)
        responding: Optional[asyncio.Task] = None
        woke_at: Optional[float] = None          # set while waiting for the utterance

        logger.info("=" * 50)
        logger.info("Voice assistant active. Say '{}' to begin.", settings.wake_word_phrase)
        logger.info("=" * 50)

        async for chunk in microphone.chunks():
            if not self._active:
                break

            if woke_at is None:
                if wake_word_detector.detected(wake_word_detector.check_chunk(chunk)):
                    if responding and not responding.done():
                        self.interrupt()
                        if not (self._speaking and not self._interruptible):
                            responding.cancel()
                        await asyncio.gather(responding, return_exceptions=True)
                    logger.info("Wake word detected")
                    print("\n🎙️  Listening...", flush=True)
                    woke_at = time.perf_counter()
                    collector.reset()
                continue

            # A false wake-word trigger with no speech after it would otherwise
            # hold this branch forever, so time out back to wake-word listening.
            if time.perf_counter() - woke_at > settings.wake_listen_timeout_s:
                logger.info("No speech within {}s — returning to wake word", settings.wake_listen_timeout_s)
                print("(timed out — say the wake word again)", flush=True)
                woke_at = None
                collector.reset()
                continue

            utterance = collector.push(chunk)
            if utterance is None:
                continue
            logger.debug("Utterance collected in {:.0f}ms", (time.perf_counter() - woke_at) * 1000)
            woke_at = None

            print("⏳  Transcribing...", flush=True)
            user_text = await stt_engine.transcribe_async(utterance)
            logger.info("STT: {}", user_text)
            if not user_text.strip():
                print("(no speech detected)", flush=True)
                continue
            print(f"You: {user_text}", flush=True)

            responding = asyncio.create_task(self._respond_aloud(user_text))

        if responding:                           # "goodbye" — let it finish speaking, then leave
            await asyncio.gather(responding, return_exceptions=True)

    async def _respond_aloud(self, user_text: str) -> None:
        """Answer one utterance out loud, speaking each clause as soon as it is complete."""
        if (command := self._voice_command(user_text)) is not None:
            print(f"Assistant: {command}", flush=True)
            await self.speak(command)
            return

        print("🧠  Thinking...", flush=True)
        self._interrupted = False
        meta, interruptible, pending = SpeechMeta(), True, ""
        try:
            async for event in self.stream(user_text):
                if event.kind == "meta":
                    meta, interruptible = event.speech, event.interruptible
                elif event.kind == "text":
                    pending += event.text
                    *ready, pending = split_into_clauses(pending) or [""]   # last part may be unfinished
                    for clause in ready:
                        await self._play(clause, meta, interruptible)
                elif event.kind == "done":
                    print(f"Assistant: {event.result.response}", flush=True)
            await self._play(pending, meta, interruptible)
        except asyncio.CancelledError:
            print("(interrupted)", flush=True)
            raise
        except Exception:
            # This runs as a task nobody awaits until the next wake word — log now or it's lost.
            logger.exception("Voice turn failed")
            print("[Error — see log]", flush=True)
        print("\n" + "─" * 40, flush=True)
        print(f"Say '{settings.wake_word_phrase}' again when ready.", flush=True)

    def _voice_command(self, user_text: str) -> Optional[str]:
        """Voice-only commands that bypass the LLM. Returns what to say, or None."""
        from app.tools.web_search import enable_web_search, disable_web_search
        # Whole-utterance matches only: "search the web for X" is a question, not
        # a toggle. STT adds punctuation ("Goodbye."), so strip it first.
        lower = user_text.lower().strip().strip(".!?,")
        if lower in ("enable web search", "go online"):
            return enable_web_search()
        if lower in ("disable web search", "go offline"):
            return disable_web_search()
        if lower in ("quit", "exit", "goodbye", "bye"):
            self._active = False
            return "Goodbye!"
        return None

    def stop(self) -> None:
        self._active = False


# Singleton
pipeline = AssistantPipeline()
