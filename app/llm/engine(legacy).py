"""
LLM engine wrapping llama-cpp-python.
Supports streaming generation and chat-formatted prompts.
"""
import asyncio
import time
from typing import AsyncIterator, Optional

from loguru import logger

from app.config import settings


class LLMEngine:
    """
    Thin async wrapper around llama_cpp.Llama.
    The model is loaded once and reused across requests.
    """

    def __init__(self):
        self._llm = None
        self._lock = asyncio.Lock()

    def is_loaded(self) -> bool:
        return self._llm is not None

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

        elapsed = time.perf_counter() - t0
        logger.info("LLM loaded in {:.2f}s", elapsed)

    async def generate_stream(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = settings.llm_max_tokens,
        temperature: float = settings.llm_temperature,
    ) -> AsyncIterator[str]:
        """
        Async streaming generator. Yields token strings as they are produced.
        Messages should be in [{"role": ..., "content": ...}] format.
        """
        if self._llm is None:
            raise RuntimeError("LLM not loaded. Call load() first.")

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        loop = asyncio.get_event_loop()
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
                )
                for chunk in stream:
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
            except Exception as e:
                logger.error("LLM inference error: {}", e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        # Run blocking inference in thread pool
        loop = asyncio.get_event_loop()
        async with self._lock:
            await loop.run_in_executor(None, _run_inference)

            t_first = None
            while True:
                token = await queue.get()
                if token is None:
                    break
                if t_first is None:
                    t_first = time.perf_counter()
                    logger.debug("LLM first token latency: N/A (streaming started)")
                yield token

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str,
        max_tokens: int = settings.llm_max_tokens,
        temperature: float = settings.llm_temperature,
    ) -> str:
        """Non-streaming generation — accumulates and returns full text."""
        parts = []
        async for token in self.generate_stream(messages, system_prompt, max_tokens, temperature):
            parts.append(token)
        return "".join(parts)


# Singleton
llm_engine = LLMEngine()
