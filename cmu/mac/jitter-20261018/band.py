"""Band-limited (2-8 Hz) view of an oscillation window: per joint action and q band rms, realised ratio, q-vs-action
phase/lag at the window's dominant action frequency, gyro band rms, and per 0.5 s which joint's action band power is largest.

  python3 band.py LOG T0 A B        (T0 'first' = first row + 4 s for sim logs)
"""
import sys

import numpy as np

import osc_metrics as om


def bp(x, lo=2.0, hi=8.0):
    x = np.nan_to_num(x - np.nanmean(x))
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), om.DT)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def main():
    path, t0, a, b = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    rows = om.load(path)
    t0 = rows[0]["t"] + 4.0 if t0 == "first" else float(t0)
    w = om.grid(rows, t0, a, b)
    A = np.array([bp(w["action"][:, j]) for j in range(14)])
    Q = np.array([bp(w["q"][:, j]) for j in range(14)])
    G = np.array([bp(w["gyro"][:, i]) for i in range(3)])
    tot = np.sum(A ** 2, axis=0)
    f = np.fft.rfftfreq(A.shape[1], om.DT)
    P = np.sum(np.abs(np.fft.rfft(A * np.hanning(A.shape[1]), axis=1)) ** 2, axis=0)
    f0 = f[np.argmax(P)]
    print(f"window {a}-{b}s  dominant action frequency (all joints) {f0:.2f} Hz; gyro band rms xyz {np.round(G.std(axis=1), 2)}")
    for j in range(14):
        ph = om.phase_at(w["action"][:, j], w["q"][:, j], f0)
        lag_ms = None if ph is None else -ph / 360 / f0 * 1000
        print(f"  {om.DOF[j]:16s} a_band {A[j].std():.3f}  q_band {Q[j].std():.3f}  q/a {Q[j].std() / max(A[j].std(), 1e-6):.2f}"
              f"  q phase vs a {ph:+5.0f} deg ({lag_ms:+4.0f} ms)" if ph is not None else "")
    for s in np.arange(0, A.shape[1], 25):
        seg = A[:, s:s + 25]
        e = np.sqrt(np.mean(seg ** 2, axis=1))
        top = np.argsort(e)[::-1][:3]
        print(f"  t {a + s * om.DT:5.1f}  top action-band joints " + ", ".join(f"{om.DOF[k]} {e[k]:.2f}" for k in top))


if __name__ == "__main__":
    main()
