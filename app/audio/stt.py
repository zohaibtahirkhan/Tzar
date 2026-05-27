"""
Speech-to-text using Faster-Whisper.
Supports both file transcription and streaming chunked audio.
"""
import asyncio
import io
import time
import tempfile
from pathlib import Path
from typing import AsyncIterator

import numpy as np
from loguru import logger

from app.config import settings


class STTEngine:
    """Faster-Whisper based STT."""

    def __init__(self):
        self._model = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

        logger.info("Loading Whisper STT model: {}", settings.stt_model)
        t0 = time.perf_counter()
        self._model = WhisperModel(
            settings.stt_model,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
        )
        logger.info("Whisper loaded in {:.2f}s", time.perf_counter() - t0)

    def transcribe_numpy(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a numpy float32 audio array.
        Returns the transcription string.
        """
        if self._model is None:
            raise RuntimeError("STT model not loaded.")

        t0 = time.perf_counter()
        if settings.stt_language == '':
            settings.stt_language = 'en'
        
        segments, info = self._model.transcribe(
            audio_array,
            language=settings.stt_language,
            beam_size=settings.stt_beam_size,
            vad_filter=settings.stt_vad_filter,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("STT [{:.0f}ms] detected lang={}: {}", elapsed_ms, info.language, text[:80])
        return text

    async def transcribe_async(self, audio_array: np.ndarray) -> str:
        """Async wrapper — runs transcription in thread pool to avoid blocking."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.transcribe_numpy, audio_array)

    def transcribe_file(self, path: str | Path) -> str:
        """Transcribe an audio file (wav, mp3, etc.)."""
        if self._model is None:
            raise RuntimeError("STT model not loaded.")
        segments, info = self._model.transcribe(
            str(path),
            language=settings.stt_language,
            beam_size=settings.stt_beam_size,
            vad_filter=settings.stt_vad_filter,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()


# Singleton
stt_engine = STTEngine()
