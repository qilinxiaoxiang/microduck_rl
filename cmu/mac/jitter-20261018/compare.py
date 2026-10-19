"""Side-by-side jitter summary for real walk_wired logs and jitter_sim tick logs (same row shape).

For each log: per window (s after the walk start t0) mean |dq| over legs (0-4, 9-13) and head (5-8) separately,
max |action| legs/head, trunk gyro rms, dominant action and gyro frequency, leg q_rms/action_rms ratio (how much of
the commanded oscillation the joints realise), envelope growth of leg |dq| (1/s), and the walk_wired action guard
(3 of the last 10 policy ticks with any |action| > 3 rad) trip time.

  python3 compare.py LOG:T0 [LOG:T0 ...] --win 5 25 --win 34 40      (T0 = 'first' for sim logs: walk_at)
"""
import argparse
import json

import numpy as np

import osc_metrics as om

LEG, HEADI = om.LEG, [5, 6, 7, 8]


def summary(rows, t0, a, b):
    w = om.grid(rows, t0, a, b)
    dq, act, q = w["dq"], w["action"], w["q"]
    out = dict(win=[a, b])
    if w["n"] == 0:
        return out
    out["leg_dq"] = round(float(np.nanmean(np.abs(dq[:, LEG]))), 3)
    out["head_dq"] = round(float(np.nanmean(np.abs(dq[:, HEADI]))), 3)
    ok = ~np.all(np.isnan(act), axis=1)
    if ok.any():
        out["leg_amax"] = round(float(np.nanmax(np.abs(act[ok][:, LEG]))), 2)
        out["head_amax"] = round(float(np.nanmax(np.abs(act[ok][:, HEADI]))), 2)
        a_rms = np.nanstd(act[:, LEG], axis=0)
        out["leg_q_over_a"] = round(float(np.nanmedian(np.nanstd(q[:, LEG], axis=0) / np.maximum(a_rms, 1e-4))), 2)
        fa = [om.dom_freq(act[:, j])[0] for j in LEG]
        out["leg_act_f_median"] = round(float(np.median([f for f in fa if f])), 2) if any(fa) else None
    out["gyro_rms"] = [round(float(np.nanstd(w["gyro"][:, i])), 3) for i in range(3)]
    out["gyro_f"] = [None if om.dom_freq(w["gyro"][:, i])[0] is None else round(om.dom_freq(w["gyro"][:, i])[0], 2) for i in range(3)]
    env = []
    for s in np.arange(a, b, 1.0):
        m = (w["t"] >= s) & (w["t"] < s + 1)
        if m.any():
            env.append(float(np.nanmean(np.abs(dq[m][:, LEG]))))
    out["leg_dq_per_s"] = [round(e, 2) for e in env]
    if len(env) >= 3 and min(env) > 0:
        out["growth_per_s"] = round(float(np.polyfit(np.arange(len(env)), np.log(env), 1)[0]), 3)
    out["modes"] = w["modes"]
    out["tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(-np.nanmean(w["gb"][:, 2]), -1, 1)))), 1)
    return out


def guard_trip(rows, t0):
    hist = []
    for r in rows:
        if r.get("action") is None or r["t"] < t0:
            continue
        hist.append(max(abs(x) for x in r["action"]) > 3.0)
        hist = hist[-10:]
        if sum(hist) >= 3:
            return round(r["t"] - t0, 2)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--win", nargs=2, type=float, action="append", required=True)
    ap.add_argument("--walk-at", type=float, default=4.0)
    a = ap.parse_args()
    for spec in a.logs:
        path, t0 = spec.rsplit(":", 1)
        rows = om.load(path)
        t0 = rows[0]["t"] + a.walk_at if t0 == "first" else float(t0)
        print(json.dumps(dict(log=path, guard_trip_s=guard_trip(rows, t0), end_s=round(rows[-1]["t"] - t0, 2))))
        for lo, hi in a.win:
            print("   ", json.dumps(summary(rows, t0, lo, hi)))


if __name__ == "__main__":
    main()
