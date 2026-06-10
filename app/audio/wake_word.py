"""
Wake word detection using OpenWakeWord 0.6.x.

Key facts about OWW 0.6 on Linux:
  - Default inference framework is 'tflite' (more efficient on Linux)
  - Models are NOT bundled in the pip package — must be downloaded first
  - wakeword_models= accepts bare names ('hey_jarvis') OR full file paths
  - Bare names are resolved via openwakeword.get_pretrained_model_paths()
    which looks in ~/.local/share/openwakeword/ (downloaded via utils.download_models)
"""
import os
import time
from typing import Optional

import numpy as np
from loguru import logger

from app.config import settings


def _ensure_models_downloaded() -> None:
    """
    OWW 0.6 ships no model files in the pip package.
    Download them once into ~/.local/share/openwakeword/ if not already present.
    """
    try:
        import openwakeword
        paths = openwakeword.get_pretrained_model_paths("onnx")
        if paths:
            return  # already downloaded
        logger.info("Downloading OpenWakeWord pretrained models (one-time setup)...")
        openwakeword.utils.download_models()
        logger.info("OpenWakeWord models downloaded.")
    except Exception as exc:
        logger.warning("Could not download OWW models: {}", exc)


class WakeWordDetector:
    """
    Wraps OpenWakeWord 0.6 for continuous wake phrase detection.
    Uses the ONNX inference framework (works without tflite-runtime).
    """

    FRAME_SAMPLES = 1280   # exactly 80 ms @ 16 kHz — required by OWW

    def __init__(self):
        self._model = None
        self._model_key: Optional[str] = None
        self._last_detection: float = 0.0
        self._cooldown_s: float = 2.0
        self._buffer: list = []
        self._buffered_samples: int = 0

    # ─── Loading ──────────────────────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            from openwakeword.model import Model
        except ImportError:
            raise ImportError("openwakeword not installed. Run: pip install openwakeword")

        _ensure_models_downloaded()

        model_name = settings.wake_word_model   # e.g. "hey_jarvis"

        # Try the configured model, then fall back to hey_jarvis
        for attempt in [model_name, "hey_jarvis"]:
            try:
                self._model = Model(
                    wakeword_models=[attempt],
                    inference_framework="onnx",   # avoid tflite-runtime dependency
                )
                # prediction_buffer is populated after the first predict() call;
                # grab the key now from the models dict instead
                self._model_key = list(self._model.models.keys())[0]
                logger.info(
                    "Wake word model loaded: '{}' (phrase='{}')",
                    self._model_key, settings.wake_word_phrase,
                )
                return
            except Exception as exc:
                logger.warning("Failed to load wake word model '{}': {}", attempt, exc)

        logger.error(
            "All wake word model load attempts failed — wake word detection disabled.\n"
            "Run: python3 -c \"import openwakeword; openwakeword.utils.download_models()\"\n"
            "Available models after download: alexa, hey_jarvis, hey_marvin, hey_mycroft, hey_rhasspy"
        )
        self._model = None

    # ─── Per-chunk inference ──────────────────────────────────────────────────

    def check_chunk(self, chunk: np.ndarray) -> float:
        """
        Feed one microphone chunk (any size). Returns confidence in [0, 1].
        Accumulates samples until 1280 are ready, then runs one OWW frame.

        OWW requires int16 audio @ 16 kHz.
        If the microphone delivers float32 in [-1, 1] we convert here.
        """
        if self._model is None:
            return 0.0

        # Normalise to int16 — OWW's mel-spectrogram pipeline requires it
        if chunk.dtype != np.int16:
            chunk = (chunk * 32767).clip(-32768, 32767).astype(np.int16)

        self._buffer.append(chunk)
        self._buffered_samples += len(chunk)

        if self._buffered_samples < self.FRAME_SAMPLES:
            return 0.0

        # Consume exactly one frame; keep the remainder for the next call
        accumulated = np.concatenate(self._buffer)
        frame    = accumulated[:self.FRAME_SAMPLES]
        leftover = accumulated[self.FRAME_SAMPLES:]

        self._buffer = [leftover] if len(leftover) else []
        self._buffered_samples = len(leftover)

        # predict() returns {model_key: score} directly in OWW 0.6
        scores = self._model.predict(frame)
        confidence = float(scores.get(self._model_key, 0.0))

        if confidence > 0.05:
            logger.debug(
                "Wake word score: {:.3f}  (threshold: {})",
                confidence, settings.wake_word_threshold,
            )

        return confidence

    # ─── Detection decision ───────────────────────────────────────────────────

    def detected(self, confidence: float) -> bool:
        now = time.monotonic()
        if confidence >= settings.wake_word_threshold:
            if (now - self._last_detection) >= self._cooldown_s:
                self._last_detection = now
                return True
        return False


# Singleton
wake_word_detector = WakeWordDetector()