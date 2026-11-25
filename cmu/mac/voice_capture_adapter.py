"""Shadow-only input contract for a future single-owner name-call integration."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from voice_shadow import INPUT_HZ, VoiceShadow

MAX_BLOCK_LAG_S = 0.25
# Historical ownkws rounds t_hit to three decimal places.
HIT_ROUNDING_S = 0.000501


class VoiceCaptureAdapter:
    """Consume already-captured audio and already-accepted wake hits; no device I/O."""

    def __init__(self, shadow: VoiceShadow):
        self.shadow = shadow

    def on_block(self, start: float, block: np.ndarray, *, accepted_hit: Mapping | None,
                 playback: bool, now: float) -> None:
        """Times must use the same monotonic clock, including the spotter's t_hit."""
        try:
            samples = np.asarray(block)
            if samples.ndim != 2 or not 0 < len(samples) <= INPUT_HZ // 10:
                raise ValueError("invalid capture block")
            if not isinstance(playback, bool):
                raise ValueError("playback state must be boolean")
            if not math.isfinite(start) or not math.isfinite(now):
                raise ValueError("capture and arrival timestamps must be finite")
            end = start + len(samples) / INPUT_HZ
            if now < end - 1e-9 or now - end > MAX_BLOCK_LAG_S:
                raise ValueError("capture block is from the future or too delayed")
            wake_time = None
            if accepted_hit is not None:
                if not isinstance(accepted_hit, Mapping) or accepted_hit.get("keyword") != "MOCHI":
                    raise ValueError("accepted wake hit must identify MOCHI")
                wake_time = accepted_hit.get("t_hit")
                if (isinstance(wake_time, bool) or not isinstance(wake_time, (int, float))
                        or not math.isfinite(wake_time)
                        or not start - HIT_ROUNDING_S <= wake_time <= end + HIT_ROUNDING_S):
                    raise ValueError("wake hit is not from this capture block")
                wake_time = min(end, max(start, wake_time))
            self.shadow.feed(start, samples, wake_time=wake_time, playback=playback, poll_time=now)
        except (ValueError, TypeError):
            self.shadow.restart_capture("invalid or delayed capture input")
            raise

    def capture_lost(self) -> None:
        self.shadow.restart_capture("capture lost")

    def manual_takeover(self) -> None:
        self.shadow.invalidate("manual takeover")

    def fetch_active(self, active: bool) -> None:
        self.shadow.set_fetch_active(active)

    def close(self) -> None:
        self.shadow.close()
