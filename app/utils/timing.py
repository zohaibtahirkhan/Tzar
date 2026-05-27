"""
Latency measurement utilities.
"""
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

from app.config import settings


@dataclass
class LatencyMetrics:
    wake_ms: Optional[float] = None
    stt_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    tts_start_ms: Optional[float] = None
    full_response_ms: Optional[float] = None

    def log(self) -> None:
        targets = {
            "wake": settings.target_wake_latency_ms,
            "stt": settings.target_stt_latency_ms,
            "llm_first_token": settings.target_llm_first_token_ms,
            "tts_start": settings.target_tts_start_ms,
            "full_response": settings.target_full_response_ms,
        }
        values = {
            "wake": self.wake_ms,
            "stt": self.stt_ms,
            "llm_first_token": self.llm_first_token_ms,
            "tts_start": self.tts_start_ms,
            "full_response": self.full_response_ms,
        }
        lines = []
        for key, val in values.items():
            if val is not None:
                target = targets.get(key, 9999)
                status = "✓" if val <= target else "✗"
                lines.append(f"  {status} {key}: {val:.0f}ms (target: {target}ms)")
        if lines:
            logger.info("Latency report:\n{}", "\n".join(lines))


class Timer:
    def __init__(self, name: str = ""):
        self.name = name
        self._start: Optional[float] = None
        self.elapsed_ms: Optional[float] = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        if self.name:
            logger.debug("{}: {:.0f}ms", self.name, self.elapsed_ms)

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return self.elapsed_ms
