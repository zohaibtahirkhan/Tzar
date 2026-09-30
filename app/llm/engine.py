"""
LLM Engine — Ollama backend.

Talks to a locally running Ollama instance via its HTTP API.
Supports streaming and non-streaming generation.
Swap LLM_MODEL in config to change models with zero code changes.
"""
import asyncio
import json
import re
import threading
from typing import AsyncIterator, Optional
import time

import httpx
from loguru import logger

from app.config import settings


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)

# Text the model must never continue past: fake conversation turns.
STOP_SEQUENCES = ["User:", "Human:", "\nUser:", "\nHuman:"]


def parse_json_object(raw: str) -> Optional[dict]:
    """
    Best-effort JSON object from LLM output: strips markdown fences, takes the
    outermost {...}, parses it. None when there is no valid object — callers
    decide what to salvage. Every prompt in this app asks for a JSON object,
    so this is the one place that reads model output.
    """
    clean = _FENCE_RE.sub("", (raw or "").strip()).strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class LLMUnavailableError(RuntimeError):
    """The backend can't answer at all (Ollama down, model not pulled). The message says how to fix it."""


class LLMEngine:
    def __init__(self):
        # Use semaphore instead of lock to allow controlled concurrent access
        # For CPU: limit=1, for GPU: limit=num_gpu_cores
        self._semaphore = asyncio.Semaphore(1)  # CPU-only: 1 concurrent inference
        self._llm = None
        self._base_url = settings.LLM_OLLAMA_HOST
        self._model = settings.llm_model
        logger.info("Using Ollama URL: {}", self._base_url)

    def load(self) -> None:
        """Load the model synchronously (call once at startup)."""
        # Skip load for Ollama backend (uses HTTP API instead)
        if settings.LLM_BACKEND == "ollama":
            # Nothing to load; report now so a first-run user sees the fix in the log.
            # Not fatal: Ollama may be started after the assistant.
            problem = self.ollama_problem()
            if problem:
                logger.warning("Ollama not ready: {}", problem)
            else:
                logger.info("Ollama ready: {} at {}", self._model, self._base_url)
            self._ollama_available = True
            return
        
        # llamacpp backend - load model file
        model_path = str(settings.llm_model_path)
        if not settings.llm_model_path.exists():
            logger.error(
                "Model not found at {}. "
                "Download Qwen2.5-3B-Instruct-Q4_K_M.gguf and place it in the models/ directory.",
                model_path,
            )
            raise FileNotFoundError(f"LLM model not found: {model_path}")

        logger.info("Loading LLM model: {}", model_path)
        t0 = time.perf_counter()

        try:
            from llama_cpp import Llama  # type: ignore

            self._llm = Llama(
                model_path=model_path,
                n_ctx=settings.llm_context_length,
                n_threads=settings.llm_threads,
                n_gpu_layers=settings.llm_n_gpu_layers,
                verbose=False,
                use_mlock=True,          # keep model in RAM
                use_mmap=True,
            )
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
            )
            
    def _not_running(self) -> str:
        return f"I can't reach Ollama at {self._base_url}. Start it with: ollama serve"

    def _not_pulled(self) -> str:
        return f"The model {self._model} isn't downloaded. Run: ollama pull {self._model}"

    def ollama_problem(self) -> Optional[str]:
        """None when Ollama is up and the model is pulled; otherwise what the user should do. Blocking."""
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=3.0)
            r.raise_for_status()
            names = {m.get("name") for m in r.json().get("models", [])}
        except Exception as exc:
            logger.debug("Ollama check failed: {}", exc)
            return self._not_running()
        # Ollama stores an untagged name as "<name>:latest".
        if self._model not in names and f"{self._model}:latest" not in names:
            return self._not_pulled()
        return None

    def is_loaded(self) -> bool:
        """Check if LLM has been loaded successfully."""
        if settings.LLM_BACKEND == "ollama":
            return getattr(self, "_ollama_available", False)
        elif settings.LLM_BACKEND == "llamacpp":
            return self._llm is not None
        return False

    # async def generate_stream(
    #     Streaming generation. Yields text chunks as they arrive.
    #     Uses a queue internally so the lock is released between chunks,
    #     allowing other coroutines to run.
    #     """
    #     async with self._lock:
    
    async def generate_stream(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = settings.llm_max_tokens,
        temperature: float = settings.llm_temperature,
    ) -> AsyncIterator[str]:
        """
        Async streaming generator. Yields token strings as they are produced.
 
        The lock is held for the ENTIRE duration of inference.
        If the generator is abandoned mid-stream (caller breaks or throws),
        aclose() on the generator will release the lock via the finally block.
        """
        # ── Backend-specific check ──
        if settings.LLM_BACKEND == "ollama":
            if not getattr(self, "_ollama_available", False):
                raise RuntimeError("Ollama not available. Check service is running.")
        elif settings.LLM_BACKEND == "llamacpp":
            if self._llm is None:
                raise RuntimeError("LLM not loaded. Call load() first.")
        else:
            raise RuntimeError(f"Unknown backend: {settings.LLM_BACKEND}")
 
        # ── Token budget guard: prevent context overflow before calling llamacpp ──
        def _token_budget_guard(system_prompt: str, messages: list, max_tokens: int) -> list:
            """
            Estimate token usage and truncate messages if over budget.
            Returns (possibly shortened) messages list.
        
            Token estimate: chars / 3.5 (conservative — Qwen tokenises ~3.5 chars/token)
            Budget: context_length - max_tokens - system_tokens - 64 (safety buffer)
            """
            from app.config import settings
        
            chars_per_token = 3.5
            system_tokens   = len(system_prompt) / chars_per_token
            max_out         = max_tokens
            safety          = 64
            budget          = settings.llm_context_length - system_tokens - max_out - safety
        
            if budget <= 0:
                # System prompt alone is too long — nothing we can do except truncate it
                logger.error("System prompt exceeds entire context budget!")
                return [messages[-1]] if messages else messages
        
            # Calculate tokens used by messages
            msg_tokens = sum(len(m.get("content", "")) / chars_per_token for m in messages)
        
            if msg_tokens <= budget:
                return messages  # fine, no truncation needed
        
            # Need to truncate. Strategy:
            # 1. Always keep the last user message (most recent request)
            # 2. Truncate individual message content that is unusually long
            # 3. Drop oldest turns first
        
            last_user = messages[-1]
            # Truncate last message if it alone exceeds budget (e.g. tool result dump)
            last_tokens = len(last_user.get("content", "")) / chars_per_token
            if last_tokens > budget:
                max_chars = int(budget * chars_per_token)
                truncated_content = last_user["content"][:max_chars] + "\n[...truncated]"
                logger.warning(
                    "Tool result too large ({:.0f} tokens) — truncated to {:.0f} tokens",
                    last_tokens, budget,
                )
                return [{"role": last_user["role"], "content": truncated_content}]
        
            # Drop oldest messages until we fit
            kept = [last_user]
            remaining = budget - last_tokens
            for msg in reversed(messages[:-1]):
                t = len(msg.get("content", "")) / chars_per_token
                if t <= remaining:
                    kept.insert(0, msg)
                    remaining -= t
                else:
                    break  # drop this and all older messages
        
            logger.warning(
                "Token budget: dropped {} old messages to fit in context",
                len(messages) - len(kept),
            )
            return kept
        
        messages = _token_budget_guard(system_prompt, messages, max_tokens)
 
        # ── Ollama Backend: HTTP streaming ───────────────────────────────────
        if settings.LLM_BACKEND == "ollama":
            async with self._semaphore:
                try:
                    logger.info("Ollama: creating fresh client...")
                    # Use fresh client each time with proper timeout for model loading
                    timeout = httpx.Timeout(180.0, connect=10.0, read=180.0)
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        full_messages = [{"role": "system", "content": system_prompt}] + messages
                        
                        payload = {
                            "model": self._model,
                            "messages": full_messages,
                            "stream": True,
                            "options": {
                                # Without this Ollama uses its own default (4096) and
                                # silently drops the start of the prompt — the system
                                # prompt — while the budget guard above assumes our value.
                                "num_ctx": settings.llm_context_length,
                                "temperature": temperature,
                                "num_predict": max_tokens,
                                "top_p": settings.llm_top_p,
                                "repeat_penalty": settings.llm_repeat_penalty,
                                "stop": STOP_SEQUENCES,
                            }
                        }
                        
                        logger.info("Ollama: starting stream request...")
                        logger.debug(f"URL: {self._base_url}/api/chat")
                        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
                        
                        async with client.stream("POST", f"{self._base_url}/api/chat", json=payload) as response:
                            logger.info(f"Ollama: entered stream context")
                            logger.info(f"Ollama: got response status {response.status_code}")
                            if response.status_code == 404:
                                raise LLMUnavailableError(self._not_pulled())
                            response.raise_for_status()
                            
                            logger.info("Ollama: reading text...")
                            # aiter_lines() reassembles JSON lines split across
                            # HTTP chunks; aiter_text() + split() dropped those tokens.
                            async for line in response.aiter_lines():
                                if not line.strip():
                                    continue
                                try:
                                    chunk = json.loads(line)
                                except json.JSONDecodeError:
                                    logger.warning("Ollama: skipping unparseable line: {}", line[:120])
                                    continue
                                if chunk.get("error"):      # e.g. out of memory loading the model
                                    raise LLMUnavailableError(f"Ollama error: {chunk['error']}")
                                token = chunk.get("message", {}).get("content")
                                if token:
                                    yield token
                                if chunk.get("done", False):
                                    logger.info("Ollama: stream done")
                                    break
                except LLMUnavailableError:
                    raise
                except httpx.ConnectError as exc:
                    raise LLMUnavailableError(self._not_running()) from exc
                except Exception as exc:
                    logger.error("Ollama streaming error: {} - {}", type(exc).__name__, exc)
                return  # Exit early for Ollama

        # ── LlamaCPP Backend: Thread-based streaming ─────────────────────────
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        loop   = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        abandoned = threading.Event()

        def _run_inference():
            try:
                stream = self._llm.create_chat_completion(
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=settings.llm_top_p,
                    repeat_penalty=settings.llm_repeat_penalty,
                    stream=True,
                    stop=STOP_SEQUENCES,
                )
                for chunk in stream:
                    if abandoned.is_set():
                        break
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                logger.error("LLM inference error: {}", exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel

        # One inference at a time: concurrent llama.cpp calls crash it.
        try:
            async with self._semaphore:
                inference_task = loop.run_in_executor(None, _run_inference)
                try:
                    while True:
                        token = await queue.get()
                        if token is None:
                            break
                        yield token
                finally:
                    # A cancelled or abandoned caller must not release the
                    # semaphore while the thread is still generating — the next
                    # call would run concurrently. Stop it and wait for it.
                    abandoned.set()
                    await asyncio.shield(inference_task)
        except Exception as exc:
            logger.error("LlamaCPP streaming error: {}", exc)

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = settings.llm_max_tokens,
        temperature: float = settings.llm_temperature,
    ) -> str:
        """Non-streaming generation. Always returns a string, never None."""
        parts: list[str] = []
        try:
            async for token in self.generate_stream(messages, system_prompt, max_tokens, temperature):
                if token:
                    parts.append(token)
        except Exception as exc:
            logger.error("generate() error: {}", exc)
        return "".join(parts)   # always a str, even if empty

    async def generate_safe(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = settings.llm_max_tokens,
        timeout: float = 60.0,
    ) -> str:
        """
        Timeout-protected generate(). Returns "" on timeout; generate() itself
        never raises. wait_for() cancels and awaits the stream, which holds the
        semaphore until inference has stopped, so no settle delay is needed.
        """
        try:
            return await asyncio.wait_for(self.generate(messages, system_prompt, max_tokens), timeout) or ""
        except asyncio.TimeoutError:
            logger.error("generate_safe() timed out after {}s", timeout)
            return ""


# Module-level singleton
llm_engine = LLMEngine()