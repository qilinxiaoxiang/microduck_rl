"""Replay a WAV with an explicitly labelled wake time. No microphone or robot I/O."""
import argparse
import json
import math
import sys
import wave
from pathlib import Path

import numpy as np

from voice_shadow import INPUT_HZ, VoiceShadow, Whisper


def replay(path, wake_at, transcribe):
    if not math.isfinite(wake_at) or wake_at < 0:
        raise ValueError("wake time must be finite and nonnegative")
    events = []
    voice = VoiceShadow(transcribe, events.append)
    try:
        with wave.open(str(path), "rb") as wav:
            channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
            if channels < 2 or rate != INPUT_HZ or width not in (2, 4) or wav.getcomptype() != "NONE":
                raise ValueError("expected 48 kHz, at least two channels, 16/32-bit PCM WAV")
            t = 0.0
            marked = False
            while data := wav.readframes(960):
                samples = np.frombuffer(data, dtype="<i2" if width == 2 else "<i4")
                if len(samples) % channels:
                    raise ValueError("incomplete WAV frame")
                block = samples.astype(np.float32).reshape(-1, channels) / (2 ** (8 * width - 1))
                end = t + len(block) / INPUT_HZ
                hit = wake_at if not marked and t <= wake_at <= end else None
                marked |= hit is not None
                voice.feed(t, block, wake_time=hit)
                t = end
        if not marked:
            raise ValueError("wake time is outside the WAV")
        if voice.window is not None:
            raise ValueError("WAV ends before the wake window closes")
        if voice.pending is not None:
            try:
                voice.pending[0].result(timeout=120)
            except Exception:
                # poll reports ASR failures without silently treating them as none.
                if not voice.pending[0].done():
                    raise
            voice.poll(t)
        return events
    finally:
        voice.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--wake-at", type=float, required=True,
                        help="labelled wake time in seconds; not automatic wake detection")
    parser.add_argument("--asr-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        events = replay(args.wav, args.wake_at, Whisper(args.asr_dir))
        print(json.dumps({"scope": "labelled wake WAV replay; no live capture or robot control",
                          "events": events}, indent=2))
        return 1 if any(event["state"] == "error" for event in events) else 0
    except (ImportError, OSError, ValueError, RuntimeError, wave.Error) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
