#!/usr/bin/env python3
"""Find every yaw disturbance in the brain logs and say whether the robot settled or the oscillation grew.

  python3 yaw_events.py LOG.jsonl [LOG.jsonl ...] > yaw_events.json

Events (robot upright, tilt < 20 deg, at the event):
  turn_stop  the command's vtheta goes from non-zero to zero with vx = vy = 0 (a turn in place ends)
  head_step  the commanded head yaw moves more than 0.2 rad within 1 s while vx = vy = vtheta = 0
For each event, the 4 s after it: peak |gyro z|, peak |gyro x,y|, peak tilt, peak leg current, whether it fell
(tilt > 60 deg), the policy and mode, the turn's rate and length, the yaw rate at the stop, the head pose, battery,
the share of ticks with stale IMU or joint reads, and the growth ratio (peak |gyro z| in s 1-3 over s 0-1; above 1
means the oscillation grew). Streams the log line by line; memory stays small.
"""
import gzip
import json
import math
import sys

POST_S = 4.0


def tilt_deg(gb):
    n = math.sqrt(sum(x * x for x in gb)) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2] / n))))


def head_yaw(cmd):
    h = cmd.get("head") if isinstance(cmd, dict) else None
    return float(h.get("head_yaw", 0.0)) if isinstance(h, dict) else 0.0


def twist(cmd):
    if not isinstance(cmd, dict):
        return 0.0, 0.0, 0.0
    return (float(cmd.get("vx") or 0.0), float(cmd.get("vy") or 0.0), float(cmd.get("vtheta") or 0.0))


def scan(path):
    events, open_ev = [], []
    prev_vth, turn_t0, turn_peak = 0.0, None, 0.0
    hist = []  # (t, head_yaw) over the last second
    last_head_ev = -1e9
    last_move_t = None
    for line in (gzip.open(path, "rt") if path.endswith(".gz") else open(path)):
        if not line.startswith("{"):
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t, gb, gy = r.get("t"), r.get("gb"), r.get("gyro_body")
        if t is None or not gb or not gy:
            continue
        cmd = r.get("cmd") or {}
        vx, vy, vth = twist(cmd)
        tl = tilt_deg(gb)
        # feed open events; an event's window closes early when the twist command changes (the next turn or walk)
        for ev in open_ev:
            dt = t - ev["t"]
            if ev["closed"] is None and (abs(vth) > 1e-3 or math.hypot(vx, vy) > 1e-3):
                ev["closed"] = round(dt, 2)
            if ev["closed"] is not None:
                continue
            gz, gxy = abs(gy[2]), max(abs(gy[0]), abs(gy[1]))
            ev["peak_gz"] = max(ev["peak_gz"], gz)
            ev["peak_gxy"] = max(ev["peak_gxy"], gxy)
            ev["peak_tilt"] = max(ev["peak_tilt"], tl)
            ev["peak_legs_ma"] = max(ev["peak_legs_ma"], float(r.get("legs_ma") or 0))
            ev["gz_0_1" if dt < 1.0 else "gz_1_3"] = max(ev["gz_0_1" if dt < 1.0 else "gz_1_3"], gz if dt < 3.0 else 0)
            ev["n"] += 1
            ev["stale"] += (r.get("fresh") is False) + bool(r.get("missing"))
            if tl > 60 and ev["fell_at"] is None:
                ev["fell_at"] = round(dt, 2)
        done = [ev for ev in open_ev if t - ev["t"] > POST_S]
        for ev in done:
            open_ev.remove(ev)
            ev["growth"] = round(ev["gz_1_3"] / max(ev["gz_0_1"], 0.05), 2)
            ev["stale_frac"] = round(ev.pop("stale") / max(ev.pop("n"), 1), 3)
            for k in ("peak_gz", "peak_gxy", "peak_tilt", "gz_0_1", "gz_1_3"):
                ev[k] = round(ev[k], 2)
            events.append(ev)

        def new(kind, **kw):
            q = r.get("q") or []
            ev = dict(log=path.rsplit("/", 1)[-1], kind=kind, t=t, mode=r.get("mode"), policy=r.get("policy"),
                      tilt0=round(tl, 1), gz0=round(gy[2], 2), battery_dv=r.get("battery_dv"),
                      head_yaw_q=round(q[12], 3) if len(q) > 12 else None, neck_q=round(q[11], 3) if len(q) > 11 else None,
                      peak_gz=0.0, peak_gxy=0.0, peak_tilt=0.0, peak_legs_ma=0.0, gz_0_1=0.0, gz_1_3=0.0,
                      fell_at=None, closed=None, n=0, stale=0, **kw)
            open_ev.append(ev)

        moving = math.hypot(vx, vy) > 1e-3
        if moving:
            last_move_t = t
        prev_moving_s = None if last_move_t is None else round(t - last_move_t, 2)  # s since the last walk command
        if abs(vth) > 1e-3 and not moving:
            if turn_t0 is None:
                turn_t0, turn_peak = t, 0.0
            turn_peak = max(turn_peak, abs(vth))
        if abs(prev_vth) > 1e-3 and abs(vth) <= 1e-3 and not moving and turn_t0 is not None and tl < 20:
            new("turn_stop", turn_vtheta=round(prev_vth, 3), turn_s=round(t - turn_t0, 2),
                stepped_before=prev_moving_s)
        if abs(vth) <= 1e-3 or moving:
            turn_t0 = None
        prev_vth = vth
        hy = head_yaw(cmd)
        hist.append((t, hy))
        while hist and t - hist[0][0] > 1.0:
            hist.pop(0)
        if not moving and abs(vth) <= 1e-3 and tl < 20 and t - last_head_ev > POST_S:
            lo, hi = min(h for _, h in hist), max(h for _, h in hist)
            if hi - lo > 0.2:
                last_head_ev = t
                new("head_step", head_yaw_cmd=round(hy, 3), head_yaw_swing=round(hi - lo, 3))
    return events


def main(paths):
    out = []
    for p in paths:
        ev = scan(p)
        print(f"{p.rsplit('/', 1)[-1]}: {len(ev)} events", file=sys.stderr)
        out += ev
    json.dump(out, sys.stdout, indent=0)


if __name__ == "__main__":
    main(sys.argv[1:])
