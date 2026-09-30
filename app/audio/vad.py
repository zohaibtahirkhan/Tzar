"""
Voice Activity Detection using Silero VAD.
Detects speech boundaries to trigger STT only when user is speaking.
"""
import collections
import time
from typing import Optional

import numpy as np
from loguru import logger

from app.config import settings


class VADEngine:
    """Silero VAD wrapper with speech segment buffering."""

    def __init__(self):
        self._model = None
        self._get_speech_timestamps = None
        self._is_speaking = False
        self._silence_start: Optional[float] = None

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            import torch  # type: ignore
            from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore
        except ImportError:
            raise ImportError("silero-vad not installed. Run: pip install silero-vad")

        logger.info("Loading Silero VAD...")
        self._model = load_silero_vad()
        self._get_speech_timestamps = get_speech_timestamps
        logger.info("Silero VAD loaded")

    def is_speech(self, chunk: np.ndarray, sample_rate: int = 16000) -> float:
        """
        Returns speech probability for an audio chunk (float32, 1D).
        chunk should be ~32ms (512 samples @ 16kHz).
        """
        if self._model is None:
            raise RuntimeError("VAD model not loaded.")

        import torch  # type: ignore

        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        tensor = torch.from_numpy(chunk)
        confidence = self._model(tensor, sample_rate).item()
        return confidence

    def get_speech_segments(self, audio: np.ndarray, sample_rate: int = 16000) -> list[dict]:
        """
        Return speech timestamp segments from a full audio array.
        Each segment: {"start": int, "end": int} in samples.
        """
        if self._model is None or self._get_speech_timestamps is None:
            raise RuntimeError("VAD model not loaded.")
        import torch  # type: ignore

        tensor = torch.from_numpy(audio.astype(np.float32))
        return self._get_speech_timestamps(
            tensor,
            self._model,
            threshold=settings.vad_threshold,
            min_speech_duration_ms=settings.vad_min_speech_duration_ms,
            min_silence_duration_ms=settings.vad_min_silence_duration_ms,
            sampling_rate=sample_rate,
        )


class SpeechCollector:
    """
    Stateful buffer that accumulates audio chunks and detects
    when a complete utterance has been spoken.
    """

    # Silero VAD accepts exactly this many samples per call (512 @ 16 kHz).
    FRAME_SAMPLES = int(settings.audio_sample_rate * settings.audio_chunk_ms / 1000)

    def __init__(self, vad: VADEngine):
        self.vad = vad
        self._buffer: list[np.ndarray] = []
        self._pending = np.zeros(0, dtype=np.float32)   # partial frame carried over
        self._is_speaking = False
        self._silence_frames = 0
        # How many consecutive silent frames before we declare end-of-speech
        self._silence_threshold_frames = int(
            settings.vad_min_silence_duration_ms / settings.audio_chunk_ms
        )

    def push(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Push audio of any length (the microphone sends exact frames, a browser
        sends whatever its resampler produced). Feeds the VAD in FRAME_SAMPLES
        slices and returns the utterance array when speech ends, else None.
        """
        self._pending = np.concatenate([self._pending, chunk]) if len(self._pending) else chunk
        utterance = None
        while len(self._pending) >= self.FRAME_SAMPLES and utterance is None:
            frame, self._pending = self._pending[:self.FRAME_SAMPLES], self._pending[self.FRAME_SAMPLES:]
            utterance = self._push_frame(frame)
        return utterance

    def _push_frame(self, chunk: np.ndarray) -> Optional[np.ndarray]:
        prob = self.vad.is_speech(chunk)
        is_speech = prob >= settings.vad_threshold

        if is_speech:
            self._is_speaking = True
            self._silence_frames = 0
            self._buffer.append(chunk)
        elif self._is_speaking:
            self._buffer.append(chunk)
            self._silence_frames += 1
            if self._silence_frames >= self._silence_threshold_frames:
                # End of utterance
                utterance = np.concatenate(self._buffer)
                self._buffer.clear()
                self._is_speaking = False
                self._silence_frames = 0
                logger.debug("VAD: utterance captured ({:.2f}s)", len(utterance) / settings.vad_sample_rate)
                return utterance

        return None

    def reset(self) -> None:
        self._buffer.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._is_speaking = False
        self._silence_frames = 0


# Singleton
vad_engine = VADEngine()
