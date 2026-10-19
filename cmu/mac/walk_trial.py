"""One guarded walk trial against a running walk_wired.py: stand61 balance, check, timed walk, report.

Only call it once Shawn says the robot is guarded. It writes the control file like a person would:
  1. {"mode": "walk", vx 0} (stand61 before the first step), then waits for log rows at least --settle-s old;
     the jsonl log is buffered about 1 s, so the check waits for fresh rows instead of reading an empty window.
  2. Balance check on the last 0.5 s: policy stand, max |action| < 0.3, tilt < 5 deg, IMU age < 0.1 s.
     A failed check leaves stand61 running, prints why, and exits 1 without walking.
  3. {"mode": "walk", "vx": ..., "for_s": ...}; walk_wired sets zero velocity after for_s and walk61 stops itself.
  4. Report for the walk window and the 3 s after the stop: actions, tilt, yaw rate, IMU gaps, servo voltage, policy,
     plus the Pi's under-voltage log tail (duck-pwrlog.service).

  ~/.venvs/microduck-walk/bin/python walk_trial.py --log /tmp/duck_wired_log_1015w.jsonl
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time


def put(path, cmd):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cmd, f)
    os.replace(tmp, path)


class Tail:
    """New rows of the jsonl log since this object was made."""

    def __init__(self, path):
        self.path, self.pos, self.rows = path, os.path.getsize(path), []

    def poll(self):
        with open(self.path) as f:
            f.seek(self.pos)
            chunk = f.read()
        done = chunk.rfind("\n") + 1
        self.pos += len(chunk[:done].encode())
        for line in chunk[:done].splitlines():
            try:
                self.rows.append(json.loads(line))
            except ValueError:
                pass
        return self.rows

    def wait_until(self, t, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if self.poll() and self.rows[-1].get("t", 0) >= t:
                return True
            time.sleep(0.2)
        return False


def tilt(r):
    return math.degrees(math.acos(max(-1.0, min(1.0, -r["gb"][2]))))


def max_act(rows):
    acts = [max(abs(a) for a in r["action"]) for r in rows if r.get("action")]
    return round(max(acts), 2) if acts else None


def summary(rows):
    if not rows:
        return {"rows": 0}
    yaw = sorted(abs((r.get("gyro_body") or [0, 0, 0])[2]) for r in rows)
    volts = [r["battery_dv"] for r in rows if r.get("battery_dv")]
    lost = [r for r in rows if (r.get("imu_age") or 0) > 0.25]
    return {"rows": len(rows), "max_action": max_act(rows), "max_tilt_deg": round(max(map(tilt, rows)), 1),
            "yaw_p95": round(yaw[int(0.95 * (len(yaw) - 1))], 2), "yaw_max": round(yaw[-1], 2),
            "imu_max_age_s": round(max(r.get("imu_age") or 0 for r in rows), 3),
            "imu_lost_from_s": None if not lost else round(lost[0]["t"] - rows[0]["t"], 2),
            "servo_v_min": None if not volts else min(volts) / 10,
            "policies": sorted({r.get("policy") for r in rows if r.get("action")} - {None}),
            "modes": sorted({r["mode"] for r in rows})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="walk_wired.py --log file")
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--vx", type=float, default=0.25)
    ap.add_argument("--for-s", type=float, default=2.0)
    ap.add_argument("--settle-s", type=float, default=2.5)
    ap.add_argument("--pi", default="duck", help="ssh host for the under-voltage log; '' skips it")
    args = ap.parse_args()

    tail = Tail(args.log)
    t0 = time.time()
    put(args.cmd, {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
    if not tail.wait_until(t0 + args.settle_s, timeout=args.settle_s + 6):
        print("NO WALK: the log has no rows past the settle time; is walk_wired.py running with this --log?")
        return 1
    last = [r for r in tail.rows if r["t"] >= tail.rows[-1]["t"] - 0.5]
    bal = summary(last)
    why = [w for w, bad in (("policy is not stand61", bal["policies"] != ["stand"]),
                            (f"action {bal['max_action']} >= 0.3", (bal["max_action"] or 9) >= 0.3),
                            (f"tilt {bal['max_tilt_deg']} >= 5 deg", bal["max_tilt_deg"] >= 5),
                            (f"IMU age {bal['imu_max_age_s']} s", bal["imu_max_age_s"] >= 0.1)) if bad]
    print("balance", json.dumps(bal))
    if why:
        print("NO WALK, stand61 keeps running:", "; ".join(why))
        return 1

    t1 = time.time()
    put(args.cmd, {"mode": "walk", "vx": args.vx, "vy": 0.0, "vtheta": 0.0, "for_s": args.for_s})
    tail.wait_until(t1 + args.for_s + 3.0, timeout=args.for_s + 8)
    walk = [r for r in tail.rows if t1 <= r["t"] < t1 + args.for_s]
    after = [r for r in tail.rows if t1 + args.for_s <= r["t"] < t1 + args.for_s + 3.0]
    report = {"vx": args.vx, "for_s": args.for_s, "walk": summary(walk), "after_stop": summary(after),
              "fell": any(tilt(r) > 45 for r in walk + after)}
    print(json.dumps(report, indent=1))
    if args.pi:
        try:
            out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", args.pi, "tail -3 ~/duck/pwrlog.txt; uptime"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
        except subprocess.TimeoutExpired:
            out = ""
        print("Pi:", out or "NOT REACHABLE (power cut or reboot?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
