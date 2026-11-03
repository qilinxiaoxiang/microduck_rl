"""The duck's own name spotter: a small log-mel time-convolution network trained on Shawn's recorded "Mochi" calls (namecall-20261029/
train_mochi.py), run in numpy on the A7Z. It replaces the public sherpa-onnx keyword model, which transcribed his
calls as ONCE / MONTY / I and never fired on the robot (PET_ROBOT_LOG 2026-10-30).

Features: the two ears downmixed to 16 kHz (namecall.Decimate3), 40 log-mel bands over 25 ms frames every 10 ms,
a WINDOW_S window, the mean over time removed per band (level and channel colour drop out). Model (the recipe of the
Edge Impulse keyword lab, 18-763 lecture 21: spectral features, a small time convolution, the keyword against
"other speech" and "noise / silence" classes): three conv + max-pool layers over time, a global max over time (so the
word may sit anywhere in the window), a dense softmax; weights in an .npz (OwnModel). Scoring runs
every SCORE_EVERY blocks (100 ms) on the newest window; a hit is the first score at or over the threshold, then
REFRACTORY_S of quiet. OwnSpotter.feed() has the KeywordGate interface (t, block, silent) -> hit dict or None, so
namecall.NameCall uses it unchanged (its detector onsets still fix the word start and the direction).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

FS16 = 16000
N_FFT = 512
WIN = 400            # 25 ms
HOP = 160            # 10 ms
N_MELS = 40
F_LO, F_HI = 100.0, 7000.0
WINDOW_S = 1.0       # scored window; a call is 0.3-0.6 s, PRE_S of it before the detector onset
PRE_S = 0.15
N_FRAMES = (int(WINDOW_S * FS16) - WIN) // HOP + 1           # 98, what logmel() yields for a WINDOW_S signal
SCORE_EVERY = 5      # blocks of 20 ms
REFRACTORY_S = 1.0
REPEAT_THRESHOLD = 0.7   # second chance: two score runs >= this, starting REPEAT_GAP_S apart, also fire (None = off);
REPEAT_GAP_S = (1.0, 5.0)  # chosen offline 2026-10-30 (namecall-20261029/results/operating_point_20261030.json)
MODEL_DEFAULT = os.path.join(HERE, "namecall-20261029", "models", "mochi_own.npz")


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, fs=FS16, f_lo=F_LO, f_hi=F_HI):
    edges = _mel_to_hz(np.linspace(_hz_to_mel(f_lo), _hz_to_mel(f_hi), n_mels + 2))
    bins = np.floor((n_fft + 1) * edges / fs).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for m in range(n_mels):
        a, b, c = bins[m], bins[m + 1], bins[m + 2]
        if b == a:
            b = a + 1
        if c == b:
            c = b + 1
        fb[m, a:b] = (np.arange(a, b) - a) / (b - a)
        fb[m, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


_FB = mel_filterbank()
_HANN = np.hanning(WIN).astype(np.float32)


def logmel(y):
    """(frames, N_MELS) log-mel of a 16 kHz float signal, mean over time removed per band."""
    y = np.asarray(y, np.float32)
    n = (len(y) - WIN) // HOP + 1
    if n < 1:
        return np.zeros((0, N_MELS), np.float32)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx] * _HANN
    spec = np.abs(np.fft.rfft(frames, N_FFT, axis=1)) ** 2
    mel = spec @ _FB.T
    lm = np.log10(mel + 1e-10)
    return (lm - lm.mean(axis=0, keepdims=True)).astype(np.float32)


def window_features(y16, start_s):
    """Flattened log-mel of the WINDOW_S window starting at start_s (zero padded), or None when it runs past the end."""
    i0 = int(round(start_s * FS16))
    n = int(WINDOW_S * FS16)
    if i0 < 0:
        y = np.concatenate([np.zeros(-i0, np.float32), y16[: max(0, n + i0)]])
    else:
        y = y16[i0:i0 + n]
    if len(y) < n:
        if len(y) < 0.6 * n:
            return None
        y = np.concatenate([y, np.zeros(n - len(y), np.float32)])
    lm = logmel(y)
    return lm[:N_FRAMES].reshape(-1)


class OwnModel:
    """Time-convolution classifier over a log-mel window, numpy only.

    npz keys: conv{i}_w (k, c_in, c_out), conv{i}_b, pool{i} (time max-pool after conv i, 1 = none), dense_w (c, n_cls),
    dense_b, mu / sd (per mel band, applied before conv0), threshold, classes (str array; "mochi" is the target).
    Layers: conv (same padding) -> ReLU -> max-pool, repeated; global max over time; dense; softmax."""

    def __init__(self, path=MODEL_DEFAULT, _d=None, _prefix=""):
        d = np.load(path, allow_pickle=False) if _d is None else _d
        self.path = path
        self.threshold = float(d["threshold"]) if "threshold" in d else None
        if not _prefix and "members" in d:
            # an ensemble: members m0_, m1_, ... trained with different seeds; probabilities are averaged (2026-10-30:
            # single seeds of one recipe differed by several false hits and calls on the same held-out audio)
            self.members = [OwnModel(path, _d=d, _prefix=f"m{j}_") for j in range(int(d["members"]))]
            self.classes = self.members[0].classes
            self.target = self.classes.index("mochi")
            return
        self.members = None
        k = lambda name: _prefix + name
        self.convs = []
        i = 0
        while k(f"conv{i}_w") in d:
            self.convs.append((d[k(f"conv{i}_w")].astype(np.float32), d[k(f"conv{i}_b")].astype(np.float32), int(d[k(f"pool{i}")])))
            i += 1
        self.dw, self.db = d[k("dense_w")].astype(np.float32), d[k("dense_b")].astype(np.float32)
        self.mu, self.sd = d[k("mu")].astype(np.float32), d[k("sd")].astype(np.float32)
        self.classes = [str(c) for c in d[k("classes")]]
        self.target = self.classes.index("mochi")

    @staticmethod
    def _conv(x, w, b):
        """x (B, T, C_in), w (k, C_in, C_out), same padding."""
        k = w.shape[0]
        p = k // 2
        xp = np.pad(x, ((0, 0), (p, k - 1 - p), (0, 0)))
        out = np.zeros((x.shape[0], x.shape[1], w.shape[2]), np.float32)
        for j in range(k):
            out += xp[:, j:j + x.shape[1], :] @ w[j]
        return out + b

    def probs(self, feats):
        if self.members is not None:
            return np.mean([m.probs(feats) for m in self.members], axis=0)
        x = np.asarray(feats, np.float32).reshape(-1, N_FRAMES, N_MELS)
        x = (x - self.mu) / self.sd
        for w, b, pool in self.convs:
            x = np.maximum(self._conv(x, w, b), 0.0)
            if pool > 1:
                t = x.shape[1] // pool * pool
                x = x[:, :t].reshape(x.shape[0], t // pool, pool, x.shape[2]).max(axis=2)
        z = x.max(axis=1) @ self.dw + self.db
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def score(self, feats):
        """P(mochi) for one flattened window (float) or a batch of rows (array)."""
        p = self.probs(feats)[:, self.target]
        return p if p.shape[0] > 1 else float(p[0])


class OwnSpotter:
    """KeywordGate stand-in: feed(t, block, silent) -> None or dict(t_hit, keyword, tokens, span_s)."""

    def __init__(self, model_path=MODEL_DEFAULT, ears=(0, 1), threshold=None, gain_db=0.0,
                 repeat_threshold=REPEAT_THRESHOLD, repeat_gap=REPEAT_GAP_S):
        import namecall as nc
        self.nc = nc
        self.model = OwnModel(model_path)
        self.threshold = self.model.threshold if threshold is None else float(threshold)
        self.ears = list(ears)
        self.repeat_threshold = None if repeat_threshold is None else float(repeat_threshold)
        self.repeat_gap = tuple(repeat_gap)
        self.repeats = 0
        self.gain = 10 ** (gain_db / 20.0)
        self.model_dir, self.keywords_file = model_path, None
        self.hits = 0
        self.reset()

    def reset(self):
        self.dec = self.nc.Decimate3()
        self.ring = np.zeros(int(WINDOW_S * FS16) + 3200, np.float32)
        self.n_blocks = 0
        self.last_hit_t = -1e9
        self.last_p = 0.0
        self.runs = []           # start times of recent score runs >= repeat_threshold that did not fire
        self.in_run = False

    def feed(self, t, block, silent=False):
        x = block[:, self.ears].mean(axis=1) if block.ndim == 2 else block
        y = self.dec(x) * self.gain
        if silent:
            y = np.zeros_like(y)
        self.ring = np.concatenate([self.ring[len(y):], y.astype(np.float32)])
        self.n_blocks += 1
        if self.n_blocks % SCORE_EVERY:
            return None
        t_end = t + len(x) / self.nc.FS
        if t_end - self.last_hit_t < REFRACTORY_S:
            return None
        n = int(WINDOW_S * FS16)
        feats = logmel(self.ring[-n:])[:N_FRAMES].reshape(-1)
        return self.decide(self.model.score(feats), t_end)

    def decide(self, p, t_end):
        """One score p at time t_end -> hit dict or None: the threshold, or the second chance on a repeated near miss."""
        self.last_p = p
        if p >= self.threshold:
            self.hits += 1
            self.last_hit_t = t_end
            self.runs, self.in_run = [], False
            return dict(t_hit=round(t_end, 3), keyword="MOCHI", tokens=[round(float(p), 3)], span_s=None, repeat=False)
        if self.repeat_threshold is None or p < self.repeat_threshold:
            self.in_run = False
            return None
        if self.in_run:
            return None
        self.in_run = True
        lo, hi = self.repeat_gap
        if any(lo <= t_end - t0 <= hi for t0 in self.runs):
            # second chance (Apple, "Hey Siri: An On-device DNN-powered Voice Trigger"): a caller who repeats the name
            # after a near miss is answered on the repeat (2026-10-30 18:53: two calls at 0.78-0.83, answer 7.6 s late)
            self.hits += 1
            self.repeats += 1
            self.last_hit_t = t_end
            self.runs, self.in_run = [], False
            return dict(t_hit=round(t_end, 3), keyword="MOCHI", tokens=[round(float(p), 3)], span_s=None, repeat=True)
        self.runs = [t0 for t0 in self.runs if t_end - t0 <= hi] + [t_end]
        return None


OWW_DIR = os.path.join(HERE, "namecall-20261029", "models", "oww")
OWW_FRAME = 1280         # 80 ms at 16 kHz: openWakeWord's step
OWW_MEL_CTX = 480        # the 3 extra 10 ms hops openWakeWord feeds the mel model before each new frame


class OwwSpotter(OwnSpotter):
    """openWakeWord head on the duck (round 2, 2026-10-30): the frozen Google speech-embedding front end
    (melspectrogram.onnx -> 32-band mel every 10 ms, /10 + 2; embedding_model.onnx -> 96 features per 80 ms from the
    last 76 mel frames) and a small head trained on the duck-ear calls, web and TTS "mochi" against 2000 h of speech
    (namecall-20261029/oww/run_v2.sh). Same streaming arithmetic as openwakeword.utils.AudioFeatures, in
    onnxruntime only; same feed() / decide() contract as OwnSpotter, scored every 80 ms.
    Each ear is its own stream and the score is the larger of the two (per_ear=True): on the eight known-direction
    calls of 2026-10-30 20:18 the ear mix missed the call from +135 deg (0.01) that the right ear alone heard at 0.87
    (namecall-20261029/eightdir_20261030.json); the mix of both ears is the single stream when per_ear is False."""

    def __init__(self, head_path, ears=(0, 1), threshold=0.5, gain_db=0.0, repeat_threshold=REPEAT_THRESHOLD,
                 repeat_gap=REPEAT_GAP_S, front_dir=OWW_DIR, threads=1, per_ear=True):
        import namecall as nc
        import onnxruntime as ort
        self.nc = nc
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = threads
        prov = ["CPUExecutionProvider"]
        self.mel = ort.InferenceSession(os.path.join(front_dir, "melspectrogram.onnx"), so, providers=prov)
        self.emb = ort.InferenceSession(os.path.join(front_dir, "embedding_model.onnx"), so, providers=prov)
        self.head = ort.InferenceSession(head_path, so, providers=prov)
        self.mel_in, self.emb_in, self.head_in = (m.get_inputs()[0].name for m in (self.mel, self.emb, self.head))
        self.n_feat = int(self.head.get_inputs()[0].shape[1])      # 16 feature frames = 1.28 s
        self.threshold = float(threshold)
        self.ears = list(ears)
        self.repeat_threshold = None if repeat_threshold is None else float(repeat_threshold)
        self.repeat_gap = tuple(repeat_gap)
        self.repeats = 0
        self.gain = 10 ** (gain_db / 20.0)
        self.model_dir, self.keywords_file, self.model = head_path, None, None
        self.per_ear = bool(per_ear) and len(self.ears) > 1
        self.hits = 0
        self.reset()

    class _Stream:
        """One microphone's streaming state: decimator, pending 16 kHz samples, mel context, mel and feature rows."""

        def __init__(self, dec):
            self.dec = dec
            self.pending = np.zeros(0, np.float32)
            self.raw = np.zeros(OWW_MEL_CTX, np.float32)          # the tail kept for the mel model's context
            self.melbuf = np.ones((76, 32), np.float32)
            self.feats = np.zeros((0, 96), np.float32)

    def reset(self):
        n = len(self.ears) if self.per_ear else 1
        self.streams = [self._Stream(self.nc.Decimate3()) for _ in range(n)]
        self.last_hit_t = -1e9
        self.last_p = 0.0
        self.runs = []
        self.in_run = False
        # like AudioFeatures: a start-up buffer from 4 s of low noise, so the first real frames have a full history
        noise = (np.random.default_rng(0).integers(-1000, 1000, 4 * FS16) / 32768.0).astype(np.float32)
        for k in range(0, len(noise) - OWW_FRAME + 1, OWW_FRAME):
            for st in self.streams:
                self._frame(noise[k:k + OWW_FRAME], st)

    @property
    def feats(self):
        return self.streams[0].feats

    def _frame(self, y16, st=None):
        """One 80 ms frame of 16 kHz float audio -> one new 96-feature row of the stream (the first by default)."""
        st = self.streams[0] if st is None else st
        x = np.concatenate([st.raw[-OWW_MEL_CTX:], y16]) * 32767.0
        spec = np.squeeze(self.mel.run(None, {self.mel_in: x[None, :].astype(np.float32)})[0]) / 10.0 + 2.0
        st.melbuf = np.vstack([st.melbuf, spec.astype(np.float32)])[-970:]
        e = self.emb.run(None, {self.emb_in: st.melbuf[-76:][None, :, :, None].astype(np.float32)})[0]
        st.feats = np.vstack([st.feats, e.reshape(1, 96)])[-120:]
        st.raw = y16[-OWW_MEL_CTX:] if len(y16) >= OWW_MEL_CTX else np.concatenate([st.raw, y16])[-OWW_MEL_CTX:]

    def score(self, st=None):
        """P(mochi) of one stream, or the largest over the ear streams when none is given."""
        sts = self.streams if st is None else [st]
        return max(float(np.squeeze(self.head.run(None, {self.head_in: s.feats[-self.n_feat:][None]})[0])) for s in sts)

    def feed(self, t, block, silent=False):
        if block.ndim == 2:
            chans = [block[:, e] for e in self.ears] if self.per_ear else [block[:, self.ears].mean(axis=1)]
        else:
            chans = [block] * len(self.streams)
        framed = False
        for st, x in zip(self.streams, chans):
            y = st.dec(x) * self.gain
            if silent:
                y = np.zeros_like(y)
            st.pending = np.concatenate([st.pending, y.astype(np.float32)])
            if len(st.pending) >= OWW_FRAME:
                self._frame(st.pending[:OWW_FRAME], st)
                st.pending = st.pending[OWW_FRAME:]
                framed = True
        if not framed:
            return None
        t_end = t + len(chans[0]) / self.nc.FS
        if t_end - self.last_hit_t < REFRACTORY_S:
            return None
        return self.decide(self.score(), t_end)


def pack_ensemble(paths, out, threshold):
    """One .npz holding several single models (m0_..., m1_...) and the ensemble threshold."""
    d = {"members": np.int32(len(paths)), "threshold": np.float32(threshold)}
    for j, p in enumerate(paths):
        src = np.load(p, allow_pickle=False)
        for key in src.files:
            if key != "threshold":
                d[f"m{j}_{key}"] = src[key]
    np.savez(out, **d)
    return out
