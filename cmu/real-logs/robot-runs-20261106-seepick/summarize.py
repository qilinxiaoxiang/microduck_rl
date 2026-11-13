#!/usr/bin/env python3
"""Per-attempt numbers from the sliced tick logs: max tilt, fall (tilt > 45 deg outside the pick's own bow), guard stops, largest action, in-place
turns at vx 0, jaw holding time, battery. Prints one line per run and writes summary.json."""
import gzip
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def tilt(gb):
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2]))))


def main():
    runs = json.load(open(os.path.join(HERE, "runs.json")))
    out = []
    for r in runs:
        rows = [json.loads(l) for l in gzip.open(os.path.join(HERE, r["file"]), "rt")]
        rows = [x for x in rows if "gb" in x and r["t_start"] - 1 <= x["t"] <= r["t_end"] + 15]
        tilts = [tilt(x["gb"]) for x in rows]
        # the pick itself bows the trunk 35-50 deg (mode seepick); a fall is over 45 deg in any other mode
        fall_tilts = [tilt(x["gb"]) for x in rows if x["mode"] != "seepick"] or [0.0]
        acts = [max((abs(a) for a in (x.get("action") or [0])), default=0) for x in rows]
        stands, spins, prev_mode, prev_spin = 0, 0, None, False
        hold_s = 0.0
        for i, x in enumerate(rows):
            if x["mode"] == "stand" and prev_mode == "walk":
                stands += 1
            prev_mode = x["mode"]
            c = x.get("cmd") or {}
            spin = x["mode"] == "walk" and not c.get("vx") and bool(c.get("vtheta"))
            if spin and not prev_spin:
                spins += 1
            prev_spin = spin
            if x.get("jaw_holding") and i:
                hold_s += x["t"] - rows[i - 1]["t"]
        bat = [x["battery_dv"] / 10 for x in rows if isinstance(x.get("battery_dv"), (int, float))]  # pack, decivolts
        s = dict(run=r["run"], tag=r["tag"], max_tilt_deg=round(max(tilts), 1), max_tilt_outside_pick_deg=round(max(fall_tilts), 1), fell=max(fall_tilts) > 45,
                 walk_to_stand_switches=stands, max_abs_action_rad=round(max(acts), 2), in_place_turns=spins,
                 jaw_holding_s=round(hold_s, 1), battery_v=[round(min(bat), 2), round(max(bat), 2)] if bat else None)
        out.append(s)
        print(s)
    with open(os.path.join(HERE, "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
