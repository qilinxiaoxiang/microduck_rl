"""Audio-only wake-window prototype. It never opens a microphone or controls a robot."""
from __future__ import annotations

import math
import re
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

INPUT_HZ = 48_000
ASR_HZ = 16_000
WINDOW_S = 1.5
PREROLL_S = 1.0
MAX_RESULT_AGE_S = 3.0


def local_intent(text: str) -> str:
    normalized = " ".join(re.sub(r"[,.!?;:]", " ", text.lower()).split())
    normalized = re.sub(r"^(?:hey )?mochi(?:\s+|$)", "", normalized)
    if normalized in ("fetch", "please fetch", "fetch please"):
        return "fetch"
    if normalized in ("stop", "please stop", "stop please", "stopped"):
        return "stop"
    return "none"


class StreamingDecimator:
    """Keep FIR history and decimation phase across capture blocks."""

    def __init__(self):
        n = np.arange(97) - 48
        h = np.sinc(2 * 7000 / INPUT_HZ * n) * np.blackman(97)
        self.h = (h / h.sum()).astype(np.float32)
        self.reset()

    def reset(self):
        self.history = np.zeros(96, dtype=np.float32)
        self.seen = 0

    def process(self, block: np.ndarray) -> tuple[int, np.ndarray]:
        mono = block[:, :2].mean(axis=1, dtype=np.float32)
        padded = np.concatenate((self.history, mono))
        filtered = np.convolve(padded, self.h, mode="valid")
        offset = -self.seen % 3
        self.history = padded[-96:]
        self.seen += len(mono)
        return offset, filtered[offset::3]


def to_16k(clip: np.ndarray) -> np.ndarray:
    """Downmix the two ears and apply the name-call-style FIR decimator."""
    return StreamingDecimator().process(clip)[1]


class Whisper:
    """Load one local sherpa-onnx Whisper model for a single background worker."""

    def __init__(self, model_dir: Path):
        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(model_dir / "tiny.en-encoder.int8.onnx"),
            decoder=str(model_dir / "tiny.en-decoder.int8.onnx"),
            tokens=str(model_dir / "tiny.en-tokens.txt"),
            language="en", task="transcribe", num_threads=4,
        )

    def __call__(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(ASR_HZ, samples)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


@dataclass(frozen=True)
class Window:
    session: int
    epoch: int
    wake_time: float
    deadline: float


class VoiceShadow:
    """Consumes copies of an existing capture stream and native wake hits.

    Input timestamps, wake times, and poll times must share one monotonic clock.
    All public methods run on the capture thread; only transcription is threaded.
    """

    def __init__(self, transcribe: Callable[[np.ndarray], str],
                 emit: Callable[[dict], None]):
        self.transcribe = transcribe
        self.emit = emit
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-asr")
        self.pending: tuple[Future, Window] | None = None
        self.ring: deque[tuple[float, np.ndarray]] = deque()
        self.decimator = StreamingDecimator()
        self.window: Window | None = None
        self.parts: list[tuple[float, np.ndarray]] = []
        self.session = 0
        self.epoch = 0
        self.active_fetch = False
        self.next_time: float | None = None
        self.playback_active = False
        self.closed = False

    def _emit(self, state: str, **fields: object) -> None:
        self.emit({"state": state, "shadow_only": True, **fields})

    def invalidate(self, reason: str) -> None:
        if not reason:
            raise ValueError("invalidation reason is required")
        self.epoch += 1
        self.window = None
        self.parts = []
        self.ring.clear()
        self.decimator.reset()
        self._emit("invalidated", reason=reason, epoch=self.epoch)

    def restart_capture(self, reason: str) -> None:
        self.next_time = None
        self.playback_active = False
        self.invalidate(reason)

    def set_fetch_active(self, active: bool) -> None:
        if not isinstance(active, bool):
            raise TypeError("fetch state must be boolean")
        if active != self.active_fetch:
            self.active_fetch = active
            self.invalidate("fetch state changed")

    def feed(self, t: float, block: np.ndarray, *, wake_time: float | None = None,
             playback: bool = False, poll_time: float | None = None) -> None:
        if self.closed:
            raise RuntimeError("voice shadow is closed")
        block = np.asarray(block, dtype=np.float32)
        if not math.isfinite(t) or block.ndim != 2 or block.shape[1] < 2 or not 0 < len(block) <= INPUT_HZ // 10:
            raise ValueError("expected finite timestamp and multichannel audio blocks of at most 100 ms")
        if not np.isfinite(block).all():
            raise ValueError("audio contains non-finite samples")
        if self.next_time is not None and abs(t - self.next_time) > 1 / INPUT_HZ:
            self.restart_capture("capture discontinuity")
            raise ValueError("capture timestamp discontinuity; restart stream explicitly")
        end = t + len(block) / INPUT_HZ
        if poll_time is None:
            poll_time = end
        if not math.isfinite(poll_time) or poll_time < end - 1e-9:
            raise ValueError("poll time must be finite and not precede the end of the block")
        if wake_time is not None and (not math.isfinite(wake_time) or not t <= wake_time <= end):
            raise ValueError("wake timestamp must belong to the current block")
        self.next_time = end
        if playback:
            if not self.playback_active:
                self.invalidate("playback suppression")
            self.playback_active = True
            self.poll(poll_time)
            return
        self.playback_active = False

        offset, samples = self.decimator.process(block)
        part = (t + offset / INPUT_HZ, samples)
        self.ring.append(part)
        cutoff = (wake_time if self.window is None and wake_time is not None else end) - PREROLL_S
        while self.ring and self.ring[0][0] + len(self.ring[0][1]) / ASR_HZ <= cutoff:
            self.ring.popleft()
        if self.window is not None:
            self.parts.append(part)
        elif wake_time is not None:
            self.session += 1
            self.window = Window(self.session, self.epoch, wake_time, wake_time + WINDOW_S)
            self.parts = list(self.ring)
            self._emit("listening", session=self.session, wake_time=wake_time,
                       deadline=self.window.deadline)

        self.poll(poll_time)
        if self.window is not None and end >= self.window.deadline:
            window = self.window
            segments = []
            for start, part_samples in self.parts:
                first = max(0, math.ceil((window.wake_time - PREROLL_S - start) * ASR_HZ - 1e-8))
                last = min(len(part_samples), math.ceil((window.deadline - start) * ASR_HZ - 1e-8))
                if last > first:
                    segments.append(part_samples[first:last])
            samples = np.concatenate(segments)
            self.window = None
            self.parts = []
            if self.pending is not None:
                self._emit("error", session=window.session, error="ASR worker busy; window dropped")
                return
            self.pending = (self.executor.submit(self.transcribe, samples), window)
            self._emit("transcribing", session=window.session)

    def poll(self, now: float) -> None:
        if not math.isfinite(now):
            raise ValueError("poll time must be finite")
        if self.pending is None or not self.pending[0].done():
            return
        future, window = self.pending
        self.pending = None
        try:
            text = future.result()
            if not isinstance(text, str) or not text.strip():
                raise ValueError("ASR returned an empty or invalid transcript")
        except Exception as exc:
            self._emit("error", session=window.session,
                       error=f"ASR failed ({type(exc).__name__})")
            return
        if window.epoch != self.epoch or window.session != self.session:
            self._emit("expired", session=window.session, reason="session or job changed")
            return
        if now < window.deadline or now - window.deadline > MAX_RESULT_AGE_S:
            self._emit("expired", session=window.session, reason="ASR result outside deadline")
            return
        intent = local_intent(text)
        if intent == "fetch" and self.active_fetch:
            self._emit("ignored", session=window.session, reason="fetch already active")
            return
        self._emit("intent", session=window.session, intent=intent, source="local_rules",
                   capture_end=window.deadline, result_time=now, epoch=window.epoch)

    def close(self) -> None:
        if not self.closed:
            self.invalidate("shadow stopped")
            self.closed = True
            if self.pending is not None and not self.pending[0].done():
                self._emit("error", error="ASR worker still running at shutdown; process may not exit")
            self.executor.shutdown(wait=False, cancel_futures=True)
