"""
Microphone input: continuous audio capture.
Feeds chunks to VAD and wake word detector.

Two capture backends are supported so the same code runs on Windows, macOS
and Linux: `sounddevice` (ships prebuilt PortAudio wheels on all platforms)
and `pyaudio` (needs a PortAudio toolchain, painful on Windows).
Selected via settings.audio_backend — "auto" prefers sounddevice.
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
        self._backend = ""              # "sounddevice" | "pyaudio", set by start()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._chunk_samples = int(settings.audio_sample_rate * settings.audio_chunk_ms / 1000)
        self._queue: queue.Queue = queue.Queue(maxsize=200)

    def _resolve_backend(self) -> str:
        """Pick a capture backend that is actually importable on this machine."""
        preference = (settings.audio_backend or "auto").lower()
        if preference == "auto":
            order = ["sounddevice", "pyaudio"]
        else:
            # Try the requested backend first, then fall back to the other one.
            order = [preference] + [b for b in ("sounddevice", "pyaudio") if b != preference]

        for backend in order:
            try:
                __import__(backend)
                if backend != preference and preference != "auto":
                    logger.warning(
                        "Audio backend '{}' unavailable — falling back to '{}'",
                        preference, backend,
                    )
                return backend
            except ImportError:
                continue

        raise ImportError(
            "No audio capture backend available. Install one of:\n"
            "  pip install sounddevice   (recommended — works on Windows/macOS/Linux)\n"
            "  pip install pyaudio       (needs PortAudio dev libraries)"
        )

    def start(self) -> None:
        self._backend = self._resolve_backend()
        self._running = True

        if self._backend == "sounddevice":
            self._start_sounddevice()
        else:
            self._start_pyaudio()

        logger.info(
            "Microphone started ({}): {}Hz, {}ms chunks ({} samples)",
            self._backend,
            settings.audio_sample_rate,
            settings.audio_chunk_ms,
            self._chunk_samples,
        )

    # ── sounddevice backend (cross-platform default) ─────────────────────────

    def _start_sounddevice(self) -> None:
        import sounddevice as sd  # type: ignore

        self._stream = sd.InputStream(
            samplerate=settings.audio_sample_rate,
            channels=settings.audio_channels,
            dtype="float32",
            blocksize=self._chunk_samples,
            device=settings.audio_input_device,
            callback=self._sd_callback,
        )
        self._stream.start()

    def _sd_callback(self, indata, frame_count, time_info, status) -> None:
        if status:
            logger.warning("Audio input status: {}", status)

        # indata is (frames, channels) and its buffer is reused by PortAudio,
        # so it must be copied before being handed to another thread.
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self._enqueue(chunk)

    # ── pyaudio backend ──────────────────────────────────────────────────────

    def _start_pyaudio(self) -> None:
        import pyaudio  # type: ignore

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=settings.audio_channels,
            rate=settings.audio_sample_rate,
            input=True,
            input_device_index=settings.audio_input_device,
            frames_per_buffer=self._chunk_samples,
            stream_callback=self._pa_callback,
        )
        self._stream.start_stream()

    def _pa_callback(self, in_data, frame_count, time_info, status):
        import pyaudio  # type: ignore

        if status:
            logger.warning("Audio input status: {}", status)

        self._enqueue(np.frombuffer(in_data, dtype=np.float32))
        return (None, pyaudio.paContinue)

    # ── shared ───────────────────────────────────────────────────────────────

    def _enqueue(self, chunk: np.ndarray) -> None:
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            pass  # Consumer is behind — drop this chunk rather than block audio

    def stop(self) -> None:
        self._running = False
        if self._stream:
            if self._backend == "sounddevice":
                self._stream.stop()
            else:
                self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa:
            self._pa.terminate()
            self._pa = None
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
