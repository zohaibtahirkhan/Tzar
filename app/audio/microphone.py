"""
Microphone input: continuous audio capture using PyAudio.
Feeds chunks to VAD and wake word detector.
"""
import asyncio
import queue
import threading
import time
from typing import Optional, Callable

import numpy as np
from loguru import logger

from app.config import settings


class MicrophoneStream:
    """
    Continuously captures audio from the microphone in a background thread.
    Chunks are placed into an asyncio-safe queue for async consumers.
    """

    def __init__(self):
        self._pa = None
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._chunk_samples = int(settings.audio_sample_rate * settings.audio_chunk_ms / 1000)
        self._queue: queue.Queue = queue.Queue(maxsize=200)

    def _get_pyaudio(self):
        try:
            import pyaudio  # type: ignore
            return pyaudio
        except ImportError:
            raise ImportError("pyaudio not installed. Run: pip install pyaudio")

    def start(self) -> None:
        pa_module = self._get_pyaudio()
        self._pa = pa_module.PyAudio()
        self._running = True

        self._stream = self._pa.open(
            format=pa_module.paFloat32,
            channels=settings.audio_channels,
            rate=settings.audio_sample_rate,
            input=True,
            input_device_index=settings.audio_input_device,
            frames_per_buffer=self._chunk_samples,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        logger.info(
            "Microphone started: {}Hz, {}ms chunks ({} samples)",
            settings.audio_sample_rate,
            settings.audio_chunk_ms,
            self._chunk_samples,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudio  # type: ignore

        if status:
            logger.warning("Audio input status: {}", status)

        chunk = np.frombuffer(in_data, dtype=np.float32)
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            pass  # Drop oldest — we're behind
        return (None, pyaudio.paContinue)

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pa:
            self._pa.terminate()
        logger.info("Microphone stopped")

    def read_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Blocking read with timeout (for non-async consumers)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    async def read_chunk_async(self) -> Optional[np.ndarray]:
        """Async-friendly read — yields control while waiting."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read_chunk, 0.05)

    async def chunks(self):
        """Async generator yielding audio chunks continuously."""
        while self._running:
            chunk = await self.read_chunk_async()
            if chunk is not None:
                yield chunk
            else:
                await asyncio.sleep(0.001)


# Singleton
microphone = MicrophoneStream()
