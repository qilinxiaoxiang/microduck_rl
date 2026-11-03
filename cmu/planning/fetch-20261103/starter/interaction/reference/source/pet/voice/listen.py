"""Utterance capture: a stream of audio blocks in, one clip per spoken sentence out (16 kHz mono, 1-8 s).

    pet/.venv/bin/python -m pet.voice.listen --wav out/clips/zh01.robot.wav          # a file (any rate/channels)
    pet/.venv/bin/python -m pet.voice.listen --mic [--device N]                      # this Mac's microphone, 16 kHz
    pet/.venv/bin/python -m pet.voice.listen --cmd "ssh a7z arecord -q -D hw:CARD=sndi2s0,DEV=0 -c 4 -r 48000 -f S32_LE -t raw -"

Sources yield (t, mono block) at the source rate in 20 ms blocks. A 4-channel robot stream (CURRENT-STATE
a7z_audio_capture: ch0 left ear, ch1 right ear, ch2 back mic, ch3 zero, S32_LE 48 kHz, INMP441 24-bit left-justified)
is downmixed to the mean of the two ears; a stereo file to the mean of both channels. Segmentation runs at the source
rate: the onset comes from namecall.OnsetDetector (the same detector that turns the duck's head to a call; a voice-band
energy rise over a tracked room floor, confirmed by the next block), the end is TAIL_S of blocks back under floor +
END_DB, bounded to MIN_S..MAX_S with PRE_S of pre-roll. The finished utterance is resampled to 16 kHz once (windowed-sinc
decimation for 48 kHz, scipy or linear interpolation otherwise) and handed on as an Utterance.

Nothing here opens the servo bus or the IMU, so no robot I/O lock is involved (namecall.py says the same); the one
shared resource is the capture device itself: one arecord at a time on hw:CARD=sndi2s0 (namecall.py or this, not both).
"""
from __future__ import annotations

import collections
import json
import math
import queue
import shlex
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MAC = ROOT / "microduck-sim" / "guide" / "firmware" / "mac"
if str(MAC) not in sys.path:
    sys.path.insert(0, str(MAC))
try:  # the shared onset detector (microduck-sim/guide/firmware/mac/namecall.py, owned by the name-call work)
    from namecall import OnsetDetector as _NamecallOnset
    from namecall import band_energy_db as _band_energy_db
except Exception:  # pragma: no cover - namecall imports pet.express and duck_sound; off this checkout fall back
    _NamecallOnset = None
    _band_energy_db = None

ROBOT_FS = 48000
ROBOT_CHANNELS = 4
EARS = (0, 1)
STT_FS = 16000
BLOCK_S = 0.02
MIN_S, MAX_S = 1.0, 8.0
PRE_S = 0.3           # pre-roll kept before the onset block
TAIL_S = 0.7          # this long under the floor ends the utterance
END_DB = 6.0          # "under the floor" means less than floor + END_DB
BAND = (300.0, 4000.0)


# ----------------------------------------------------------------------------------------------- resampling

def _band_energy(block: np.ndarray, fs: int, window: np.ndarray) -> float:
    if _band_energy_db is not None:
        return _band_energy_db(block, fs, BAND, window)[0]
    n = len(block)
    p = np.abs(np.fft.rfft(block * window)) ** 2 / n
    f = np.fft.rfftfreq(n, 1 / fs)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return 10 * math.log10(max(float(p[m].sum()), 1e-14))


def to_16k(x: np.ndarray, fs: int) -> np.ndarray:
    """Mono float samples at fs -> 16 kHz. Integer ratios use a windowed-sinc low-pass then decimation."""
    x = np.asarray(x, np.float32)
    if fs == STT_FS:
        return x
    if fs % STT_FS == 0:
        r = fs // STT_FS
        taps = 32 * r + 1
        n = np.arange(taps) - (taps - 1) / 2
        cutoff = 0.45 / r                                 # cycles per input sample
        h = 2 * cutoff * np.sinc(2 * cutoff * n) * np.hamming(taps)
        h /= h.sum()
        y = np.convolve(x, h, mode="same")
        return y[::r].astype(np.float32)
    try:
        from scipy.signal import resample_poly
        g = math.gcd(fs, STT_FS)
        return resample_poly(x, STT_FS // g, fs // g).astype(np.float32)
    except ImportError:  # pragma: no cover
        n_out = int(round(len(x) * STT_FS / fs))
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def downmix(block: np.ndarray, ears=EARS) -> np.ndarray:
    """(n, channels) -> mono: the ears of a robot stream, else the mean of all channels."""
    if block.ndim == 1:
        return block
    if block.shape[1] >= ROBOT_CHANNELS:
        return block[:, list(ears)].mean(axis=1)
    return block.mean(axis=1)


# ----------------------------------------------------------------------------------------------- utterance

@dataclass
class Utterance:
    samples: np.ndarray                   # float32, 16 kHz mono, -1..1
    t_start: float                        # source time of the first kept sample (s)
    t_end: float
    onset: dict = field(default_factory=dict)
    source: str = ""
    fs: int = STT_FS

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.fs

    @property
    def peak_dbfs(self) -> float:
        return 20 * math.log10(max(float(np.max(np.abs(self.samples))), 1e-9))

    def write_wav(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        q = np.clip(self.samples * 32767, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.fs)
            w.writeframes(q.tobytes())
        return path

    def summary(self) -> dict:
        return {"t_start": round(self.t_start, 3), "t_end": round(self.t_end, 3), "seconds": round(self.seconds, 2),
                "peak_dbfs": round(self.peak_dbfs, 1), "onset": self.onset, "source": self.source}


class _MiniOnset:
    """Fallback when namecall.py is not importable: voice-band energy THRESH over a tracked floor, confirmed once."""

    def __init__(self, fs, block, thresh_db=12.0, confirm_db=6.0, refractory_s=2.0, floor_tau_s=3.0):
        self.fs, self.block, self.thresh, self.confirm, self.refractory = fs, block, thresh_db, confirm_db, refractory_s
        self.alpha = block / fs / floor_tau_s
        self.window = np.hanning(block)
        self.floor, self.n, self.pending, self.last = None, 0, None, -1e9
        self.deaf_until = -1e9
        self.hist = collections.deque(maxlen=25)

    def level(self, block):
        return _band_energy(block, self.fs, self.window), 0.0

    def update(self, t, block):
        e = self.level(block)[0]
        self.n += 1
        if self.n <= int(0.5 * self.fs / self.block):
            self.hist.append(e)
            self.floor = float(np.median(self.hist))
            return None
        over = e - self.floor
        if over < self.thresh:
            self.floor += self.alpha * (e - self.floor)
        out = None
        if t < self.deaf_until or t - self.last < self.refractory:
            self.pending = None
        elif self.pending is not None:
            if over >= self.thresh - self.confirm:
                out, self.last = dict(self.pending), self.pending["t"]
            self.pending = None
        elif over >= self.thresh:
            self.pending = dict(t=t, e_db=round(e, 1), floor_db=round(self.floor, 1), over_db=round(over, 1))
        return out


THRESH_DB = 12.0      # namecall's default: an onset is this far over the tracked room floor


def make_onset_detector(fs: int, block: int, thresh_db: float = THRESH_DB):
    if _NamecallOnset is not None:
        return _NamecallOnset(fs=fs, block=block, thresh_db=thresh_db)
    return _MiniOnset(fs, block, thresh_db=thresh_db)


class Segmenter:
    """feed(t, mono_block) -> Utterance when a sentence has ended, else None. Blocks are BLOCK_S long at fs."""

    def __init__(self, fs: int, block: int | None = None, min_s=MIN_S, max_s=MAX_S, pre_s=PRE_S, tail_s=TAIL_S,
                 end_db=END_DB, thresh_db=THRESH_DB, source=""):
        self.fs = fs
        self.block = block or int(round(fs * BLOCK_S))
        self.min_s, self.max_s, self.tail_s, self.end_db, self.source = min_s, max_s, tail_s, end_db, source
        self.det = make_onset_detector(fs, self.block, thresh_db)
        self.pre = collections.deque(maxlen=max(1, int(round(pre_s / BLOCK_S))))
        self.cur: list[np.ndarray] | None = None
        self.cur_t0 = 0.0
        self.onset: dict = {}
        self.quiet_blocks = 0
        self.n_utt = 0

    @property
    def floor_db(self):
        return self.det.floor

    def deaf_until(self, t: float):
        """Ignore sound until t (own quack, own servos)."""
        self.det.deaf_until = t

    def feed(self, t: float, block: np.ndarray) -> Utterance | None:
        block = np.asarray(block, np.float32)
        onset = self.det.update(t, block)
        if self.cur is None:
            self.pre.append((t, block))
            if onset is None:
                return None
            self.cur = [b for _, b in self.pre]
            self.cur_t0 = self.pre[0][0]
            self.onset = onset
            self.quiet_blocks = 0
            return None
        self.cur.append(block)
        length = len(self.cur) * self.block / self.fs
        e = self.det.level(block)[0]
        floor = self.det.floor if self.det.floor is not None else -80.0
        self.quiet_blocks = self.quiet_blocks + 1 if e < floor + self.end_db else 0
        ended = self.quiet_blocks * self.block / self.fs >= self.tail_s and length >= self.min_s
        if not ended and length < self.max_s:
            return None
        return self._finish(t + self.block / self.fs)

    def flush(self) -> Utterance | None:
        """End of stream: whatever is being collected, if it is long enough to be speech."""
        if self.cur is None:
            return None
        if len(self.cur) * self.block / self.fs < 0.4:
            self.cur = None
            return None
        return self._finish(self.cur_t0 + len(self.cur) * self.block / self.fs)

    def _finish(self, t_end: float) -> Utterance:
        x = np.concatenate(self.cur)
        self.cur = None
        self.n_utt += 1
        need = int(self.min_s * self.fs) - len(x)
        if need > 0:
            x = np.concatenate([x, np.zeros(need, np.float32)])
        return Utterance(to_16k(x, self.fs), self.cur_t0, t_end, dict(self.onset), self.source)


# ----------------------------------------------------------------------------------------------- sources

def wav_blocks(path: Path | str, ears=EARS):
    """(t, mono block) from a 16- or 32-bit WAV of any rate and channel count; 32-bit frames are INMP441-style."""
    with wave.open(str(path)) as w:
        fs, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 4:
        x = (np.frombuffer(raw, "<i4").reshape(-1, ch) >> 8).astype(np.float32) / 2 ** 23
    elif sw == 2:
        x = np.frombuffer(raw, "<i2").reshape(-1, ch).astype(np.float32) / 2 ** 15
    else:
        raise ValueError(f"{path}: {sw * 8}-bit samples")
    mono = downmix(x, ears)
    block = int(round(fs * BLOCK_S))
    for i in range(0, len(mono) - block + 1, block):
        yield i / fs, mono[i:i + block]
    return


def raw_blocks(stream, fs=ROBOT_FS, channels=ROBOT_CHANNELS, ears=EARS, sample_width=4):
    """(t, mono block) from a raw S32_LE (or S16_LE) interleaved byte stream, e.g. arecord -t raw on stdin."""
    block = int(round(fs * BLOCK_S))
    need = block * channels * sample_width
    n = 0
    t0 = time.time()
    while True:
        buf = b""
        while len(buf) < need:
            chunk = stream.read(need - len(buf))
            if not chunk:
                return
            buf += chunk
        if sample_width == 4:
            x = (np.frombuffer(buf, "<i4").reshape(-1, channels) >> 8).astype(np.float32) / 2 ** 23
        else:
            x = np.frombuffer(buf, "<i2").reshape(-1, channels).astype(np.float32) / 2 ** 15
        yield t0 + n * block / fs, downmix(x, ears)
        n += 1


def command_blocks(cmd: list[str] | str, fs=ROBOT_FS, channels=ROBOT_CHANNELS, ears=EARS):
    """Run a command (arecord, or ssh host arecord) and read its raw S32_LE output as blocks."""
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    try:
        yield from raw_blocks(proc.stdout, fs, channels, ears)
        err = (proc.stderr.read() or b"").decode(errors="replace").strip()
        if err:
            print(f"capture command ended: {err[-400:]}", file=sys.stderr)
    finally:
        if proc.poll() is None:
            proc.kill()


def mic_blocks(device=None, fs=STT_FS):
    """(t, mono block) from this machine's microphone through sounddevice, opened at fs (CoreAudio resamples)."""
    import sounddevice as sd
    block = int(round(fs * BLOCK_S))
    q: queue.Queue = queue.Queue(maxsize=500)
    t0 = time.time()
    n = [0]

    def cb(indata, frames, t, status):
        try:
            q.put_nowait((t0 + n[0] * block / fs, indata[:, 0].copy()))
        except queue.Full:
            pass
        n[0] += 1

    with sd.InputStream(device=device, channels=1, samplerate=fs, blocksize=block, dtype="float32", callback=cb):
        while True:
            yield q.get()


def utterances(blocks, fs: int, source: str = "", **kw):
    """Segment a block source into Utterances."""
    seg = Segmenter(fs, source=source, **kw)
    for t, b in blocks:
        u = seg.feed(t, b)
        if u is not None:
            yield u
    u = seg.flush()
    if u is not None:
        yield u


def wav_rate(path) -> int:
    with wave.open(str(path)) as w:
        return w.getframerate()


# ----------------------------------------------------------------------------------------------- CLI

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="a WAV file (any rate; 4-channel robot format uses the ears)")
    src.add_argument("--mic", action="store_true", help="this machine's microphone")
    src.add_argument("--cmd", help="a command whose stdout is raw S32_LE 4-channel 48 kHz (arecord, ssh ... arecord)")
    ap.add_argument("--device", default=None, help="sounddevice input device index or name for --mic")
    ap.add_argument("--out", default=str(HERE / "out" / "utt"), help="where utterance WAVs go")
    ap.add_argument("--max-utterances", type=int, default=0, help="stop after N (0 = until the source ends)")
    ap.add_argument("--threshold-db", type=float, default=THRESH_DB, help="onset threshold over the room floor")
    a = ap.parse_args(argv)
    if a.wav:
        fs, blocks, source = wav_rate(a.wav), wav_blocks(a.wav), f"wav:{a.wav}"
    elif a.mic:
        dev = int(a.device) if a.device and a.device.isdigit() else a.device
        fs, blocks, source = STT_FS, mic_blocks(dev), "mic"
    else:
        fs, blocks, source = ROBOT_FS, command_blocks(a.cmd), f"cmd:{a.cmd}"
    out = Path(a.out)
    n = 0
    for u in utterances(blocks, fs, source=source, thresh_db=a.threshold_db):
        n += 1
        p = u.write_wav(out / f"utt-{time.strftime('%Y%m%d-%H%M%S')}-{n:03d}.wav")
        print(json.dumps({**u.summary(), "wav": str(p)}, ensure_ascii=False), flush=True)
        if a.max_utterances and n >= a.max_utterances:
            break
    if n == 0:
        print("no utterance found", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
