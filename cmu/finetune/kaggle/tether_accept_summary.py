#!/usr/bin/env python3
"""Summarise tether_accept.sh output against walk61: turn rates, walking distance, falls, jitter.

  python3 cmu/finetune/kaggle/tether_accept_summary.py <out-dir> [out.json]

Acceptance (FINETUNE-DELTA.md, run plan 'tether'): with the lead, in-place turns faster than walk61; without it, no
slower; walking distance and falls no worse with or without the lead; action jumps and peak tilt no worse (the
sim_suite real-case verdict, plus the same limits on the lead cases: tilt at most +2 deg, jumps at most +2).
"""
import json
import os
import sys

TURN_S, WALK_S = 25.0, 12.0
TILT_SLACK, JUMP_SLACK, DIST_FRAC = 2.0, 2, 0.9


def rows(d):
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            try:
                out[f[:-5]] = json.load(open(os.path.join(d, f)))
            except json.JSONDecodeError:
                out[f[:-5]] = None
    return out


def summary(r):
    def pick(prefix):
        return [v for k, v in r.items() if k.startswith(prefix) and v]
    s = {}
    for side, sign in (("R", -1), ("L", 1)):
        t = pick(f"turn{side}_tether_s")
        s[f"turn{side}_tether_dps"] = [round(sign * v["yaw_walk_deg"] / TURN_S, 1) for v in t]
        n = r.get(f"turn{side}_none")
        s[f"turn{side}_none_dps"] = round(sign * n["yaw_walk_deg"] / TURN_S, 1) if n else None
    w = pick("walk_tether_s")
    s["walk_tether_m"] = [v["dist_x"] for v in w]
    s["walk_none_m"] = r["walk_none"]["dist_x"] if r.get("walk_none") else None
    lead = [v for k, v in r.items() if v and ("tether" in k or k.endswith("_none"))]
    s["falls"] = sum(v["fell_at"] is not None for v in lead)
    s["cases"] = len(lead)
    s["max_tilt"] = max(v["max_tilt"] for v in lead)
    s["action_jumps_max"] = max(v["action_jumps"] for v in lead)
    s["action_jumps_sum"] = sum(v["action_jumps"] for v in lead)
    return s


def verdict(c, ref, cr, rr):
    fails = []
    for side in "RL":
        if sum(c[f"turn{side}_tether_dps"]) <= sum(ref[f"turn{side}_tether_dps"]):
            fails.append(f"turn{side} with the lead not faster")
        if c[f"turn{side}_none_dps"] < ref[f"turn{side}_none_dps"] - 1.0:
            fails.append(f"turn{side} without the lead slower")
    if min(c["walk_tether_m"]) < DIST_FRAC * min(ref["walk_tether_m"]) or c["walk_none_m"] < DIST_FRAC * ref["walk_none_m"]:
        fails.append("walks shorter")
    if c["falls"] > ref["falls"]:
        fails.append("more falls")
    for k, v in cr.items():
        b = rr.get(k)
        if not (v and b):
            continue
        if v["max_tilt"] > b["max_tilt"] + TILT_SLACK:
            fails.append(f"{k} tilt {v['max_tilt']} vs {b['max_tilt']}")
        if v["action_jumps"] > b["action_jumps"] + JUMP_SLACK:
            fails.append(f"{k} jumps {v['action_jumps']} vs {b['action_jumps']}")
    return fails


def main():
    d = sys.argv[1]
    names = sorted(n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n)))
    raw = {n: rows(os.path.join(d, n)) for n in names}
    res = {n: summary(r) for n, r in raw.items()}
    for n in names:
        res[n]["suite_verdict"] = open(os.path.join(d, n, "suite.out")).read().strip().splitlines()[-12:]
        if n != "walk61":
            res[n]["lead_fails"] = verdict(res[n], res["walk61"], raw[n], raw["walk61"])
    print(json.dumps(res, indent=1))
    if len(sys.argv) > 2:
        json.dump(res, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
