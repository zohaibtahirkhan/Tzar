"""
LLM Engine — Ollama backend.

Talks to a locally running Ollama instance via its HTTP API.
Supports streaming and non-streaming generation.
Swap LLM_MODEL in config to change models with zero code changes.
"""
import asyncio
import json
from typing import AsyncIterator, Optional
import time

import httpx
from loguru import logger

from app.config import settings


class LLMEngine:
    def __init__(self):
        # Use semaphore instead of lock to allow controlled concurrent access
        # For CPU: limit=1, for GPU: limit=num_gpu_cores
        self._semaphore = asyncio.Semaphore(1)  # CPU-only: 1 concurrent inference
        self._client: Optional[httpx.AsyncClient] = None
        self._llm = None
        self._base_url = settings.LLM_OLLAMA_HOST
        self._model = settings.llm_model
        logger.info("Using Ollama URL: {}", self._base_url)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(60.0, connect=5.0)
            )
        return self._client

    def load(self) -> None:
        """Load the model synchronously (call once at startup)."""
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
                n_gpu_layers=0,          # CPU-only
                verbose=False,
                use_mlock=True,          # keep model in RAM
                use_mmap=True,
            )
        except ImportError:
            raise ImportError(
                "llama-cpp-python is not installed. Run: pip install llama-cpp-python"
            )
            
    async def health_check(self) -> bool:
        """Returns True if Ollama is running and the model is available."""
        try:
            client = await self._get_client()
            r = await client.get("/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            available = any(self._model in m for m in models)
            if not available:
                logger.warning(
                    "Model '{}' not found in Ollama. Run: ollama pull {}",
                    self._model, self._model
                )
            self._ollama_available = True
            return available
        except Exception as e:
            logger.error("Ollama health check failed: {}", e)
            return False

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
        if self._llm is None:
            raise RuntimeError("LLM not loaded. Call load() first.")
 
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
 
        full_messages = [{"role": "system", "content": system_prompt}] + messages
 
        loop   = asyncio.get_event_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
 
        def _run_inference():
            try:
                stream = self._llm.create_chat_completion(
                    messages=full_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=settings.llm_top_p,
                    repeat_penalty=settings.llm_repeat_penalty,
                    stream=True,
                    stop=["User:", "Human:", "\nUser:", "\nHuman:"],
                )
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as exc:
                logger.error("LLM inference error: {}", exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)   # sentinel
 
        # ── Use semaphore for controlled concurrent access ──────────────────
        # Semaphore allows N concurrent inferences (configurable)
        # Using async context manager ensures release even if:
        #   - the caller abandons the generator (aclose())
        #   - _run_inference raises
        #   - queue.get() raises
        async with self._semaphore:
            await loop.run_in_executor(None, _run_inference)
 
            t_first = None
            while True:
                token = await queue.get()
                if token is None:
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                yield token

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
        Timeout-protected generate(). Never raises. Returns "" on any failure.
        Used by Coordinator and Critic to prevent segfaults on concurrent calls.
        """
        try:
            result = await asyncio.wait_for(
                self.generate(messages, system_prompt, max_tokens),
                timeout=timeout,
            )
            return result or ""
        except asyncio.TimeoutError:
            logger.error("generate_safe() timed out after {}s", timeout)
            return ""
        except Exception as exc:
            logger.error("generate_safe() error: {}", exc)
            return ""

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Module-level singleton
llm_engine = LLMEngine()