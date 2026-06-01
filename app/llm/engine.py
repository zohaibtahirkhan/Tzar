"""
LLM Engine — Ollama backend.

Talks to a locally running Ollama instance via its HTTP API.
Supports streaming and non-streaming generation.
Swap LLM_MODEL in config to change models with zero code changes.
"""
import asyncio
import json
from typing import AsyncIterator, Optional

import httpx
from loguru import logger

from app.config import settings


class LLMEngine:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = settings.LLM_OLLAMA_HOST
        self._model = settings.llm_model

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(60.0, connect=5.0)
            )
        return self._client

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
            return available
        except Exception as e:
            logger.error("Ollama health check failed: {}", e)
            return False

    def _build_payload(
        self,
        messages: list[dict],
        system_prompt: str = "",
        stream: bool = True
    ) -> dict:
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        ollama_messages.extend(messages)

        return {
            "model": self._model,
            "messages": ollama_messages,
            "stream": stream,
            "options": {
                "temperature": settings.llm_temperature,
                "num_ctx": settings.llm_context_length,
                "num_predict": settings.llm_max_tokens,
            }
        }

    async def generate(
        self,
        messages: list[dict],
        system_prompt: str = ""
    ) -> str:
        """Non-streaming generation. Returns full response text."""
        async with self._lock:
            client = await self._get_client()
            payload = self._build_payload(messages, system_prompt, stream=False)

            try:
                r = await client.post("/api/chat", json=payload)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except httpx.HTTPStatusError as e:
                logger.error("Ollama HTTP error: {} — {}", e.response.status_code, e.response.text)
                raise
            except Exception as e:
                logger.error("Ollama generate failed: {}", e)
                raise

    async def generate_stream(
        self,
        messages: list[dict],
        system_prompt: str = ""
    ) -> AsyncIterator[str]:
        """
        Streaming generation. Yields text chunks as they arrive.
        Uses a queue internally so the lock is released between chunks,
        allowing other coroutines to run.
        """
        async with self._lock:
            client = await self._get_client()
            payload = self._build_payload(messages, system_prompt, stream=True)

            try:
                async with client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield token
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
            except httpx.HTTPStatusError as e:
                logger.error("Ollama stream error: {}", e)
                raise
            except Exception as e:
                logger.error("Ollama stream failed: {}", e)
                raise

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Module-level singleton
llm_engine = LLMEngine()