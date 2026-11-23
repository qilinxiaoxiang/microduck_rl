"""Burst distances from the servos: walk_wired tick logs -> one row per burst (leg odometry), and the burst table.

Replaces the ruler test (Michael, 2026-11-22: trust the servos). The distance is walk_brain.LegOdometry's, replayed
over the log: the planted foot is taken as fixed on the floor, so a sliding foot reads as no motion.

    uv run --no-sync --with pyserial python cmu/step/odom_report.py <brain log .jsonl[.gz]> [...] [--out r.json]

Prints each burst (commanded seconds, cm at the stop and once settled, turn, foot switches, peak tilt, whether a guard
ended it) and, per duration, the furthest settled distance: what arms.json "bursts" holds.
"""
import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mac"), str(ROOT / "pi")]
import walk_brain as wb  # noqa: E402

SETTLE_S = 2.5      # walk61 keeps stepping 1-2 s after a burst
LEAD_TICKS = 3


def read(path):
    rows = []
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue        # a line cut by a crash or a full disk
            if isinstance(r, dict) and r.get("q") and r.get("gb") and "t" in r:
                rows.append(r)
    return rows


def commanded_vx(r):
    c = r.get("cmd") or {}
    return float(c.get("vx") or 0.0) if c.get("mode") == "walk" else 0.0


def yaw_of(r):
    return wb.yaw_of(wb.body_quat(r["quat"])) if r.get("quat") else 0.0


def bursts(rows):
    starts = [i for i in range(1, len(rows)) if commanded_vx(rows[i]) > 0 >= commanded_vx(rows[i - 1])]
    out = []
    for n, i in enumerate(starts):
        j = i
        while j < len(rows) and commanded_vx(rows[j]) > 0:
            j += 1
        if j >= len(rows):
            continue            # the log ends inside the burst
        limit = starts[n + 1] if n + 1 < len(starts) else len(rows)
        end = j
        while end < limit and rows[end]["t"] - rows[j]["t"] <= SETTLE_S and rows[end].get("mode", "walk") == "walk":
            end += 1
        seg, odom, m0, yaw0 = rows[max(i - LEAD_TICKS, 0):max(end, j + 1)], wb.LegOdometry(), None, None
        track, switches, last = [], 0, None
        for r in seg:
            yaw = yaw_of(r)
            mid = odom.update(dict(zip(wb.NAMES, r["q"])), np.asarray(r["gb"]), yaw)
            if m0 is None:
                m0, yaw0 = mid.copy(), yaw
            d = mid - m0
            track.append((r["t"], d[0] * math.cos(yaw0) + d[1] * math.sin(yaw0), yaw - yaw0))
            switches += last is not None and odom.stance != last
            last = odom.stance
        at_stop = min(track, key=lambda p: abs(p[0] - rows[j]["t"]))
        after = rows[end] if end < len(rows) else rows[-1]
        out.append(dict(
            t=round(rows[i]["t"], 2), seconds=round(rows[j]["t"] - rows[i]["t"], 2),
            for_s=(rows[i].get("cmd") or {}).get("for_s"),
            cm_at_stop=round(100 * at_stop[1], 1), cm=round(100 * track[-1][1], 1),
            turn_deg=round(math.degrees(math.atan2(math.sin(track[-1][2]), math.cos(track[-1][2]))), 1),
            foot_switches=int(switches), settled_s=round(track[-1][0] - rows[j]["t"], 1),
            peak_tilt=round(max(math.degrees(math.acos(max(-1.0, min(1.0, -r["gb"][2])))) for r in seg), 1),
            guard=after.get("mode") in ("stand", "monitor") and after["t"] - rows[j]["t"] < SETTLE_S))
    return out


def table(rows):
    """[[seconds, the furthest settled distance m]] rising, per duration rounded to 0.1 s; guard-ended bursts left out."""
    far = {}
    for b in rows:
        if not b["guard"] and b["settled_s"] >= 1.0:
            k = max(0.1, round(b["seconds"], 1))
            far[k] = max(far.get(k, 0.0), b["cm"] / 100.0)
    out, high = [], 0.0
    for k in sorted(far):
        high = max(high, far[k])
        out.append([k, round(high, 3)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    found = []
    for path in a.logs:
        for b in bursts(read(path)):
            found.append(dict(b, log=Path(path).name))
            print(f"{Path(path).name[:28]:29} {b['seconds']:5.2f} s  stop {b['cm_at_stop']:6.1f}  settled {b['cm']:6.1f} cm"
                  f"  turn {b['turn_deg']:6.1f}  switches {b['foot_switches']:2d}  tilt {b['peak_tilt']:4.1f}"
                  f"{'  GUARD' if b['guard'] else ''}")
    print("table", json.dumps(table(found)))
    if a.out:
        Path(a.out).write_text(json.dumps(dict(bursts=found, table=table(found)), indent=1) + "\n")


if __name__ == "__main__":
    main()
