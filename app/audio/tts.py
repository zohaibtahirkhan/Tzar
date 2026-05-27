"""
Text-to-speech using Kokoro TTS.
Supports sentence-level streaming so audio starts playing before
the full response has been generated.
"""
import asyncio
import io
import re
import time
from typing import AsyncIterator, Optional

import numpy as np
from loguru import logger

from app.config import settings


# Sentence boundary splitter
_CLAUSE_RE = re.compile(r'(?<=[.!?,;:])\s+|(?<=\.\.\.) ')

def split_into_clauses(text: str) -> list[str]:
    """
    Split text at clause boundaries (commas, semicolons, colons, ellipses,
    sentence endings) for more natural TTS pacing.
    Merges very short fragments (< 4 words) with the next clause.
    """
    parts = _CLAUSE_RE.split(text.strip())
    merged = []
    carry = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        combined = (carry + " " + part).strip() if carry else part
        if len(combined.split()) < 4:
            carry = combined  # too short, merge forward
        else:
            merged.append(combined)
            carry = ""
    if carry:
        merged.append(carry)
    return merged

split_into_sentences = split_into_clauses

class TTSEngine:
    """Kokoro TTS wrapper with streaming audio output."""

    def __init__(self):
        self._pipeline = None

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        try:
            from kokoro import KPipeline  # type: ignore
        except ImportError:
            raise ImportError(
                "kokoro TTS not installed. Run: pip install kokoro"
            )

        logger.info("Loading Kokoro TTS (voice: {})", settings.tts_voice)
        try:
            self._pipeline = KPipeline(lang_code="a")  # 'a' = American English
            logger.info("Kokoro TTS loaded")
        except Exception as e:
            logger.error("Kokoro TTS load failed: {}", e)
            self._pipeline = None

    def synthesize(self, text: str, speed: float = None) -> Optional[np.ndarray]:
        """
        Synthesize text to a float32 numpy audio array.
        Returns None on failure.
        """
        if speed is None:
            speed = settings.tts_speed
            
        if self._pipeline is None:
            logger.warning("TTS not available")
            return None

        t0 = time.perf_counter()
        try:
            audio_chunks = []
            for _, _, audio in self._pipeline(
                text,
                voice=settings.tts_voice,
                speed=speed,
                split_pattern=None,
            ):
                if audio is not None:
                    audio_chunks.append(audio)

            if not audio_chunks:
                return None

            result = np.concatenate(audio_chunks)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.debug("TTS synthesis [{:.0f}ms] for {} chars", elapsed_ms, len(text))
            return result
        except Exception as e:
            logger.error("TTS synthesis error: {}", e)
            return None

    async def synthesize_async(self, text: str, speed: float = None) -> Optional[np.ndarray]:
        """Async Wrapper"""
        import functools
        loop = asyncio.get_event_loop()
        fn = functools.partial(self.synthesize, text, speed)
        return await loop.run_in_executor(None, fn)

    async def stream_sentences(self, text: str, speed: float = None) -> AsyncIterator[np.ndarray]:
        """Yield audio arrays clause-by-clause for low-latency natural playback."""
        clauses = split_into_clauses(text)
        for clause in clauses:
            if not clause:
                continue
            audio = await self.synthesize_async(clause, speed=speed)
            if audio is not None:
                yield audio


class AudioPlayer:
    """Plays numpy float32 audio arrays through the system speaker."""

    def __init__(self):
        self._sd = None

    def _get_sd(self):
        if self._sd is None:
            try:
                import sounddevice as sd  # type: ignore
                self._sd = sd
            except ImportError:
                logger.error("sounddevice not installed. Run: pip install sounddevice")
        return self._sd

    def play_sync(self, audio: np.ndarray, sample_rate: int = settings.tts_sample_rate) -> None:
        sd = self._get_sd()
        if sd is None:
            return
        sd.play(audio, samplerate=sample_rate, device=settings.audio_output_device)
        sd.wait()

    async def play_async(self, audio: np.ndarray, sample_rate: int = settings.tts_sample_rate) -> None:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.play_sync, audio, sample_rate)

    async def play_stream(self, audio_gen: AsyncIterator[np.ndarray]) -> None:
        """Play audio chunks as they arrive from the TTS stream."""
        async for chunk in audio_gen:
            await self.play_async(chunk)


# Singletons
tts_engine = TTSEngine()
audio_player = AudioPlayer()
