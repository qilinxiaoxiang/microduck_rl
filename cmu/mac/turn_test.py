"""One timed walk on the A7Z brain, then what it did: yaw rate, stepping, tilt, head reached, guard trips.

Runs on the A7Z beside walk_wired.py (robot in walk mode at zero command, i.e. stand61). Holds the walk the way the app
does (the command rewritten every 0.3 s with for_s 0.6, so the robot stops by itself if this script dies), then reads
the rows walk_wired logged meanwhile.

  ~/.venvs/duck-policy/bin/python turn_test.py --vx 0.25 --vtheta 0.6 --secs 4
  ~/.venvs/duck-policy/bin/python turn_test.py --vx 0 --vtheta 0.5 --secs 4                     turn in place
  ~/.venvs/duck-policy/bin/python turn_test.py --vx 0.25 --secs 5 --head-pitch 0.8 --head-after 1   head down once going

Yaw rate is the gyro about gravity (+ = left), averaged from 1 s after the start to the end (the gait's first second
excluded). Stepping: the peak-to-peak hip pitch swing of both legs over that window (stepping on the as-built sim is
0.2-0.4 rad; a policy that stands still stays under 0.05). As-built sim at vx 0.25 (2026-10-24): vtheta 0.3 / 0.6 / 1.0
-> 17 / 34 / 57 deg/s; vx 0 vtheta 0.5 -> about 29 deg/s.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import time

NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
         "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")


def find_log():
    ps = subprocess.run(["pgrep", "-fa", "walk_wired.py"], capture_output=True, text=True).stdout
    for tok in ps.split():
        if tok.endswith(".jsonl"):
            return tok
    logs = sorted(glob.glob(os.path.expanduser("~/duck-logs/duck_wired_log_*.jsonl")), key=os.path.getmtime)
    return logs[-1] if logs else None


def write(path, cmd):
    with open(path + ".tt", "w") as f:
        json.dump(cmd, f)
    os.replace(path + ".tt", path)


def yaw_rate(r):
    g, gb = r.get("gyro_body"), r.get("gb")
    n = math.sqrt(sum(x * x for x in gb)) or 1.0
    return -sum(a * b for a, b in zip(g, gb)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--vtheta", type=float, default=0.0)
    ap.add_argument("--secs", type=float, default=4.0)
    ap.add_argument("--head-pitch", type=float, default=None, help="rad, absolute head_pitch target")
    ap.add_argument("--head-yaw", type=float, default=None)
    ap.add_argument("--head-after", type=float, default=1.0, help="s after the start before the head command")
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    args = ap.parse_args()
    log = find_log()
    if not subprocess.run(["pgrep", "-f", "python walk_wired.py"], capture_output=True).stdout:
        raise SystemExit("walk_wired.py is not running: start the brain from the app first")
    with open(log, "rb") as f:
        f.seek(0, os.SEEK_END)
        start_off = f.tell()
    t0 = time.time()
    head = {k: v for k, v in (("head_pitch", args.head_pitch), ("head_yaw", args.head_yaw)) if v is not None}
    try:
        while time.time() - t0 < args.secs:
            cmd = {"mode": "walk", "vx": args.vx, "vy": 0.0, "vtheta": args.vtheta, "for_s": 0.6}
            if head and time.time() - t0 >= args.head_after:
                cmd["head"] = head
            write(args.cmd, cmd)
            time.sleep(0.3)
    finally:
        write(args.cmd, {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
    time.sleep(1.5)
    rows = []
    with open(log, "rb") as f:
        f.seek(start_off)
        for ln in f.read().split(b"\n"):
            try:
                rows.append(json.loads(ln))
            except ValueError:
                continue
    win = [r for r in rows if t0 + 1.0 <= r["t"] <= t0 + args.secs]
    modes = sorted({r["mode"] for r in rows if t0 <= r["t"] <= t0 + args.secs + 1.5})
    if len(win) < 10:
        raise SystemExit(f"only {len(win)} log rows in the window (log {log})")
    w = [math.degrees(yaw_rate(r)) for r in win]
    q = {n: [r["q"][i] for r in win] for i, n in enumerate(NAMES)}
    swing = max(max(q[n]) - min(q[n]) for n in ("left_hip_pitch", "right_hip_pitch"))
    tilt = [math.degrees(math.acos(min(1.0, -r["gb"][2] / (math.sqrt(sum(x * x for x in r["gb"])) or 1)))) for r in win]
    tail = win[-max(1, len(win) // 4):]
    out = dict(cmd=dict(vx=args.vx, vtheta=args.vtheta, secs=args.secs, head=head or None),
               yaw_rate_deg_s=round(sum(w) / len(w), 1), commanded_deg_s=round(math.degrees(args.vtheta), 1),
               stepping=swing > 0.1, hip_pitch_swing_rad=round(swing, 3), max_tilt_deg=round(max(tilt), 1),
               head_reached={k: round(sum(r["q"][NAMES.index(k)] for r in tail) / len(tail), 2) for k in head} or None,
               modes=modes, guard_trip=any(m != "walk" for m in modes), rows=len(win), log=log)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
