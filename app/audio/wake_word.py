"""
Wake word detection using OpenWakeWord.
Continuously monitors audio and signals when the wake phrase is detected.
"""
import asyncio
import time
from typing import Optional, Callable, Awaitable

import numpy as np
from loguru import logger

from app.config import settings


class WakeWordDetector:
    """
    Wraps OpenWakeWord for continuous wake phrase detection.
    Calls an async callback when the wake word is detected above threshold.
    """

    def __init__(self):
        self._model = None
        self._last_detection: float = 0
        self._cooldown_s = 2.0  # seconds between detections
        self._buffer = []          # ADD: accumulation buffer
        self._buffer_target = 1280 # ADD: ~80ms at 16kHz

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        try:
            from openwakeword.model import Model  # type: ignore
        except ImportError:
            raise ImportError("openwakeword not installed. Run: pip install openwakeword")

        logger.info("Loading OpenWakeWord model: {}", settings.wake_word_model)
        try:
            self._model = Model(
                wakeword_models=[settings.wake_word_model],
                inference_framework="onnx",
            )
            logger.info("Wake word model loaded: '{}'", settings.wake_word_phrase)
        except Exception as e:
            logger.error("Failed to load wake word model '{}': {}. Falling back to 'alexa'.", settings.wake_word_model, e)
            try:
                self._model = Model(wakeword_models=["alexa"], inference_framework="onnx")
                logger.warning("Using 'alexa' as wake word fallback. Change settings.wake_word_model to your preferred model.")
            except Exception as e2:
                logger.error("Wake word model load failed entirely: {}", e2)
                self._model = None

    def check_chunk(self, chunk: np.ndarray) -> float:
        if self._model is None:
            return 0.0

        self._buffer.append(chunk.astype(np.float32))
        if sum(len(c) for c in self._buffer) < self._buffer_target:
            return 0.0

        accumulated = np.concatenate(self._buffer)
        self._buffer = []

        self._model.predict(accumulated)  # float32 — no int16 conversion
        scores = self._model.prediction_buffer
        if not scores:
            return 0.0
        model_key = list(scores.keys())[0]
        history = scores[model_key]
        return float(history[-1]) if history else 0.0

    def detected(self, confidence: float) -> bool:
        now = time.monotonic()
        if confidence >= settings.wake_word_threshold:
            if (now - self._last_detection) >= self._cooldown_s:
                self._last_detection = now
                return True
        return False


# Singleton
wake_word_detector = WakeWordDetector()
