#!/usr/bin/env python
"""Static grasp pose search: the most stable pose whose beak comes down to H mm above the floor without touching.

For a target height H (default 1 mm) the search looks for 14 servo angles + a jaw opening such that
  - the lowest point of the whole head (real CAD meshes AND the simulator's beak boxes, whichever is lower) is H,
  - that lowest point is a lip ("lips" mode: BOTH lips, the upper lip and the jaw tip, within H .. H + --both-tol;
    "lowest" mode: the jaw tip or the upper lip, the other one anywhere),
  - nothing else of the body comes within 4 mm of the floor, no self-collision, soles flat (< 3 deg) and both down,
  - every joint inside its real range (MJCF ranges, hip roll +-0.52) with 0.02 rad to spare,
  - holding torques <= --tau-max N.m, and the commanded targets needed to hold it on kP 200 servos
    (q + torque / 0.554 N.m/rad) at most --overrun rad past a range,
and maximises the centre-of-mass margin inside the support polygon. Differential evolution, then a local polish.

Families: "home" = symmetric, hip roll / yaw at the standing values (the earlier study's family); "sym" = symmetric
with hip roll out and hip yaw free; "asym" = every leg joint free (both feet must stay flat and down), head roll free.

  PYTHONPATH=src .venv-mjlab/bin/python scripts/static_grasp/sg_search.py --family sym --mode lips --h 1 --name sym_lips_1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sg_model as S

OUT = Path("/Users/remi/microduck/notes/static-grasp/poses")
K_SERVO = 0.554           # N.m/rad: BAM XL330 at firmware kP 200, 7.4 V (kt * vin * kp * error_gain / R)
HOME = S.DEFAULT_POSE
PAD = 0.02


def decode(x, family):
    q = HOME.copy()
    if family in ("home", "sym"):
        hp, kn, an, neck, hpitch, th = x[:6]
        ro, yo = (x[6], x[7]) if family == "sym" else (0.0, 0.0)
        q[0], q[9] = yo, -yo
        q[1], q[10] = HOME[1] - ro, HOME[10] + ro
        q[2], q[3], q[4] = hp, kn, an
        q[11], q[12], q[13] = -hp, -kn, -an
        q[5], q[6] = neck, hpitch
    else:   # asym: left leg (yaw, roll, pitch, knee, ankle), right leg in the left convention, neck, head pitch, roll, th
        q[0:5] = x[0:5]
        q[9:14] = -np.asarray(x[5:10])
        q[5], q[6], q[8] = x[10], x[11], x[12]
        th = x[13]
    return q, float(th)


def bounds(sm, family, theta_rng):
    lo, hi = sm.lo + PAD, sm.hi - PAD
    if family in ("home", "sym"):
        b = [(lo[2], hi[2]), (lo[3], hi[3]), (lo[4], hi[4]), (lo[5], hi[5]), (lo[6], hi[6]), theta_rng]
        if family == "sym":
            b += [(-0.1, 0.41), (-0.4, 0.4)]
        return b
    left = [(lo[i], hi[i]) for i in range(5)]
    right = [(-hi[i], -lo[i]) for i in range(9, 14)]
    return left + right + [(lo[5], hi[5]), (lo[6], hi[6]), (-0.3, 0.3), theta_rng]


def score(x, sm, a, detail=False):
    q, th = decode(x, a.family)
    viol = float(np.clip(sm.lo + PAD - q, 0, None).sum() + np.clip(q - (sm.hi - PAD), 0, None).sum())
    if viol > 1e-9:                     # out of range: worse than any in-range pose, graded to lead back inside
        return (1e9 * (1 + viol), None) if detail else 1e9 * (1 + viol)
    r = sm.evaluate(q, th)
    H = a.h / 1000
    beak_low = min(r["head_low_vis"], r["head_low_col"])
    lip_lo, lip_hi = min(r["lip_up"], r["jaw_tip"]), max(r["lip_up"], r["jaw_tip"])
    c = -min(r["margin"], 0.03) / 0.005
    c += ((beak_low - H) / 0.0002) ** 2
    c += (max(0.0, lip_lo - beak_low - 0.0003) / 0.0003) ** 2           # the lowest head point is a lip
    if a.mode == "lips":
        c += (max(0.0, lip_hi - H - a.both_tol / 1000) / 0.0005) ** 2     # and the other lip close to it
    c += (max(0.0, 0.004 - r["body_low"]) / 0.001) ** 2
    c += 10.0 * r["selfc"]
    c += (max(0.0, r["tau_max"] - a.tau_max) / 0.02) ** 2
    target = q + r["tau"] / K_SERVO
    over = float(max((target - sm.hi).max(), (sm.lo - target).max(), 0.0))
    r["target_overrun"] = over
    r["targets"] = target
    c += (max(0.0, over - a.overrun) / 0.03) ** 2
    c += sum((max(0.0, t - 3.0) / 1.0) ** 2 for t in r["sole_tilt_deg"])
    c += sum((max(0.0, g - 0.5) / 0.2) ** 2 for g in r["foot_gap_mm"])
    c += (max(0.0, -r["margin"]) / 0.002) ** 2
    c += 0.05 * r["tau_max"] / a.tau_max
    if a.family == "asym" and a.mode == "lips":
        c += (np.degrees(abs(q[8])) / 5.0) ** 2 * 0.1                   # head roll costs a little
    return (c, r) if detail else c


def summary(sm, x, a):
    c, r = score(x, sm, a, detail=True)
    q, th = decode(x, a.family)
    out = dict(name=a.name, family=a.family, mode=a.mode, h_target_mm=a.h, pad_rad=PAD, cost=float(c), q=q.tolist(), theta=th,
               margin_cm=100 * r["margin"], com_h_cm=100 * r["com_h"], tip_push_N=r["tip_push_N"],
               tip_angle_deg=r["tip_angle_deg"],
               beak_low_mm=1000 * min(r["head_low_vis"], r["head_low_col"]), cad_low_mm=1000 * r["head_low_vis"],
               box_low_mm=1000 * r["head_low_col"], lip_up_mm=1000 * r["lip_up"], jaw_tip_mm=1000 * r["jaw_tip"],
               body_low_mm=1000 * r["body_low"], head_pitch_deg=r["head_pitch_deg"], bisector_deg=r["bisector_deg"],
               trunk_pitch_deg=r["trunk_pitch_deg"], reach_cm=100 * r["reach"], tau=list(map(float, r["tau"])),
               tau_max=r["tau_max"], tau_arg=r["tau_arg"], targets=list(map(float, r["targets"])),
               target_overrun=r["target_overrun"], range_margin=r["range_margin"], selfc=r["selfc"],
               sole_tilt_deg=r["sole_tilt_deg"], foot_gap_mm=r["foot_gap_mm"], jaw_deg=float(np.degrees(th)),
               grasp_pt=list(map(float, r["grasp_pt"])), mouth_xy=list(map(float, r["mouth_xy"])))
    return out


def line(s):
    return (f"{s['name']:22s} margin {s['margin_cm']:5.2f} cm (tip push {s['tip_push_N']:.2f} N)  beak low "
            f"{s['beak_low_mm']:5.2f} mm [CAD {s['cad_low_mm']:5.2f} box {s['box_low_mm']:5.2f}]  lips up/jaw "
            f"{s['lip_up_mm']:5.1f}/{s['jaw_tip_mm']:5.1f}  pitch {s['head_pitch_deg']:5.1f} jaw {s['jaw_deg']:4.1f} "
            f"bis {s['bisector_deg']:5.1f}  tau {s['tau_max']:.2f} {s['tau_arg']}  overrun {s['target_overrun']:.2f}  "
            f"body {s['body_low_mm']:4.1f}  tilt {np.round(s['sole_tilt_deg'], 1)}  self {s['selfc']}  "
            f"trunk {s['trunk_pitch_deg']:.0f}  reach {s['reach_cm']:.1f}  cost {s['cost']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="sym", choices=["home", "sym", "asym"])
    ap.add_argument("--mode", default="lips", choices=["lips", "lowest"],
                    help="lips: both lips low (use --theta 0.35,0.52 for a grasp-ready open mouth)")
    ap.add_argument("--h", type=float, default=1.0, help="target height of the lowest head point, mm")
    ap.add_argument("--both-tol", type=float, default=1.5, help="lips mode: the higher lip within H + this, mm")
    ap.add_argument("--theta", default="0.10,0.52", help="jaw opening range, rad (lo,hi)")
    ap.add_argument("--tau-max", type=float, default=0.30)
    ap.add_argument("--overrun", type=float, default=0.20)
    ap.add_argument("--maxiter", type=int, default=250)
    ap.add_argument("--popsize", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x0", default=None, help="json file of a previous result to start from (its q, theta)")
    ap.add_argument("--from-default", action="store_true", help="seed with the default grasp's lowest pose")
    ap.add_argument("--pad", type=float, default=0.02, help="clearance kept from every joint limit, rad")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    global PAD
    PAD = a.pad
    a.name = a.name or f"{a.family}_{a.mode}_{a.h:g}mm"
    OUT.mkdir(parents=True, exist_ok=True)
    sm = S.StaticModel()
    b = bounds(sm, a.family, tuple(float(v) for v in a.theta.split(",")))
    x0 = None
    if a.from_default or a.x0:
        if a.x0:
            j = json.load(open(a.x0)); q0, th0 = np.array(j["q"]), j["theta"]
        else:
            q0, th0, _, _ = S.default_grasp_low_pose()
        x0 = encode(q0, th0, a.family)
        x0 = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(x0, b)])
    t0 = time.time()
    res = differential_evolution(score, b, args=(sm, a), seed=a.seed, maxiter=a.maxiter, popsize=a.popsize,
                                 tol=1e-8, mutation=(0.5, 1.0), recombination=0.7, polish=False, x0=x0,
                                 init="sobol")
    x = res.x
    pol = minimize(score, x, args=(sm, a), method="Nelder-Mead", bounds=b,
                   options=dict(maxiter=4000, xatol=1e-5, fatol=1e-6, adaptive=True))
    if pol.fun < res.fun and score(pol.x, sm, a) < 1e9:
        x = pol.x
    s = summary(sm, x, a)
    s["search_s"] = time.time() - t0
    s["nfev"] = int(res.nfev + pol.nfev)
    json.dump(s, open(OUT / f"{a.name}.json", "w"), indent=1)
    print(line(s), flush=True)


def encode(q, th, family):
    if family in ("home", "sym"):
        x = [q[2], q[3], q[4], q[5], q[6], th]
        if family == "sym":
            x += [HOME[1] - q[1], q[0]]
        return np.array(x)
    return np.array(list(q[0:5]) + list(-np.asarray(q[9:14])) + [q[5], q[6], q[8], th])


if __name__ == "__main__":
    main()
