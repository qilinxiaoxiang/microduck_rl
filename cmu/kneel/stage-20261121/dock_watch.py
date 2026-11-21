"""Live view of the dock from walk_wired's tick log, for the held-off-the-ground check (run on the A7Z).

    ~/.venvs/duck-policy/bin/python ~/dock-stage/dock_watch.py            # newest duck_wired_log_*.jsonl
    ~/.venvs/duck-policy/bin/python ~/dock-stage/dock_watch.py --hz 2 LOG

One line per 1/hz s: mode, policy, the commander's camera slots (visible, age, u, v), its twist, tilt, the largest
policy action and the leg load. In mode dock it also keeps the share of ticks with visible = 1. The 2026-11-20 failure
reads "vis 0" with the age climbing to 2.0 for the whole dock; after the fix, visible should hold 1 while the ball is
in the picture.
"""
import argparse
import glob
import json
import math
import os
import time


def newest_log():
    logs = glob.glob(os.path.expanduser("~/duck-logs/duck_wired_log_*.jsonl"))
    return max(logs, key=os.path.getmtime) if logs else None


def tilt_deg(gb):
    n = math.sqrt(sum(x * x for x in gb)) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2] / n))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default=None)
    ap.add_argument("--hz", type=float, default=4.0)
    a = ap.parse_args()
    path = a.log or newest_log()
    if not path:
        raise SystemExit("no ~/duck-logs/duck_wired_log_*.jsonl")
    print("watching", path, flush=True)
    f = open(path)
    f.seek(0, 2)
    buf, last_print, dock_n, dock_vis, amax = "", 0.0, 0, 0, 0.0
    while True:
        chunk = f.read()
        if not chunk:
            time.sleep(0.02)
            continue
        buf += chunk
        lines = buf.split("\n")
        buf = lines.pop()
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            act = r.get("action")
            if isinstance(act, list) and act:
                amax = max(amax, max(abs(x) for x in act))
            cam = r.get("dock_cam")
            if r.get("mode") == "dock" and cam:
                dock_n += 1
                dock_vis += cam[4] > 0.5
            now = time.time()
            if now - last_print < 1.0 / a.hz:
                continue
            last_print = now
            hold = r.get("hold") or {}
            cam_s = ("vis %d age %.2f u %+.2f v %+.2f" % (cam[4] > 0.5, cam[3], cam[0], cam[1])) if cam else "cam -"
            cmd = r.get("dock_cmd")
            cmd_s = ("cmd %+.2f %+.2f %+.2f" % tuple(cmd)) if cmd else "cmd -"
            share = (" | dock vis %d/%d (%.0f%%)" % (dock_vis, dock_n, 100.0 * dock_vis / dock_n)) if dock_n else ""
            print("%-6s %-6s %s %s | tilt %4.1f amax %4.2f load %.2f%s" % (
                r.get("mode"), r.get("policy"), cam_s, cmd_s, tilt_deg(r.get("gb") or [0, 0, -1]), amax,
                hold.get("load", float("nan")) if isinstance(hold, dict) else float("nan"), share), flush=True)
            amax = 0.0


if __name__ == "__main__":
    main()
