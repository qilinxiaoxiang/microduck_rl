"""Oscillation metrics for a walk_wired or brain_sim_check --tick-log JSONL (same row shape).

Per window: per-second mean leg |dq| and |action|, dominant frequency per leg joint (q and action) and trunk gyro,
lead/lag of each leg joint's q against the roll/pitch gyro, action-vs-q phase at the dominant frequency, and the
exponential growth rate of the leg |dq| envelope.

  python3 osc_metrics.py LOG --t0 T0 --win A B [--win C D ...] [--relative]   (A..D in s from T0)
"""
import argparse
import json

import numpy as np

DOF = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
       "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
LEG = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
DT = 0.02


def load(path):
    rows = []
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows.append(r)
    return rows


def grid(rows, t0, a, b):
    """Resample a window onto the 20 ms grid (holding the last row), so dropped ticks do not shift the spectrum."""
    sel = [r for r in rows if t0 + a <= r["t"] < t0 + b and r.get("q")]
    if not sel:
        return dict(t=np.zeros(0), n=0, modes=[], missing=None)
    ts = np.array([r["t"] for r in sel])
    tg = np.arange(t0 + a, t0 + b, DT)
    ix = np.clip(np.searchsorted(ts, tg, side="right") - 1, 0, len(sel) - 1)
    g = lambda k, n: np.array([sel[i][k] if sel[i].get(k) is not None else [np.nan] * n for i in ix], dtype=float)
    out = dict(t=tg - t0, q=g("q", 14), dq=g("dq", 14), gyro=g("gyro_body", 3), gb=g("gb", 3), action=g("action", 14),
               modes=sorted({r["mode"] for r in sel}), n=len(sel),
               missing=float(np.mean([bool(r.get("missing")) or r.get("fresh") is False for r in sel])) if sel else None)
    return out


def dom_freq(x):
    x = np.nan_to_num(x - np.nanmean(x))
    if len(x) < 16 or not np.any(x):
        return None, 0.0
    w = np.hanning(len(x))
    f = np.fft.rfftfreq(len(x), DT)
    P = np.abs(np.fft.rfft(x * w)) ** 2
    P[f < 0.5] = 0
    k = int(np.argmax(P))
    return float(f[k]), float(P[k] / max(P.sum(), 1e-12))


def phase_at(x, y, f):
    """Phase of y relative to x at frequency f, degrees (positive: y leads x)."""
    n = len(x)
    tt = np.arange(n) * DT
    e = np.exp(-2j * np.pi * f * tt) * np.hanning(n)
    X, Y = np.sum(np.nan_to_num(x - np.nanmean(x)) * e), np.sum(np.nan_to_num(y - np.nanmean(y)) * e)
    return float(np.degrees(np.angle(Y / X))) if abs(X) > 0 else None


def xcorr_lag(x, y, maxlag=10):
    """Lag (ticks) maximising corr(x[t], y[t+lag]); positive: y lags x."""
    x, y = np.nan_to_num(x - np.nanmean(x)), np.nan_to_num(y - np.nanmean(y))
    best, bl = -2, 0
    for L in range(-maxlag, maxlag + 1):
        a, b = (x[:len(x) - L], y[L:]) if L >= 0 else (x[-L:], y[:len(y) + L])
        if a.std() == 0 or b.std() == 0:
            continue
        c = abs(np.corrcoef(a, b)[0, 1])
        if c > best:
            best, bl = c, L
    return bl, round(best, 2)


def window(rows, t0, a, b):
    w = grid(rows, t0, a, b)
    dq, act, q = w["dq"][:, LEG], w["action"][:, LEG], w["q"][:, LEG]
    per_s = []
    for s in range(int(a), int(np.ceil(b))):
        m = (w["t"] >= s) & (w["t"] < s + 1)
        if m.any():
            per_s.append(dict(s=s, leg_dq=round(float(np.nanmean(np.abs(dq[m]))), 2),
                              act=round(float(np.nanmean(np.abs(act[m]))), 2) if not np.all(np.isnan(act[m])) else None,
                              gyro_max=round(float(np.nanmax(np.abs(w["gyro"][m]))), 2)))
    env = np.array([p["leg_dq"] for p in per_s])
    growth = None
    if len(env) >= 3 and np.all(env > 0):
        growth = round(float(np.polyfit(np.arange(len(env)), np.log(env), 1)[0]), 3)  # 1/s
    joints = {}
    for k, j in enumerate(LEG):
        fq, sh = dom_freq(w["q"][:, j])
        fa, _ = dom_freq(w["action"][:, j]) if not np.all(np.isnan(w["action"][:, j])) else (None, 0)
        joints[DOF[j]] = dict(f_q=None if fq is None else round(fq, 2), share=round(sh, 2),
                              f_a=None if fa is None else round(fa, 2),
                              q_rms=round(float(np.nanstd(w["q"][:, j])), 3),
                              a_rms=None if np.all(np.isnan(w["action"][:, j])) else round(float(np.nanstd(w["action"][:, j])), 3),
                              act_vs_q_deg=None if fq is None or np.all(np.isnan(w["action"][:, j])) else round(phase_at(w["q"][:, j], w["action"][:, j], fq)),
                              lag_vs_pitch=xcorr_lag(w["gyro"][:, 1], w["dq"][:, j]),
                              )
    gy = {ax: dict(zip(("f", "share"), (lambda r: (None if r[0] is None else round(r[0], 2), round(r[1], 2)))(dom_freq(w["gyro"][:, i]))),
                   rms=round(float(np.nanstd(w["gyro"][:, i])), 3)) for i, ax in enumerate("xyz")}
    return dict(win=[a, b], modes=w["modes"], rows=w["n"], missing_share=None if w["missing"] is None else round(w["missing"], 3),
                tilt_deg=round(float(np.degrees(np.arccos(np.clip(-np.nanmean(w["gb"][:, 2]), -1, 1)))), 1),
                growth_per_s=growth, per_s=per_s, gyro=gy, joints=joints)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--t0", type=float, default=None, help="absolute t0; default first row")
    ap.add_argument("--win", nargs=2, type=float, action="append", required=True)
    ap.add_argument("--brief", action="store_true", help="skip the per-joint table")
    a = ap.parse_args()
    rows = load(a.log)
    t0 = a.t0 if a.t0 is not None else rows[0]["t"]
    for lo, hi in a.win:
        r = window(rows, t0, lo, hi)
        if a.brief:
            r.pop("joints")
        print(json.dumps(r))


if __name__ == "__main__":
    main()
