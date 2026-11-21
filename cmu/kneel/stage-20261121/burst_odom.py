"""Replay a walk61 burst from walk_wired's tick log through walk_brain.LegOdometry (the small step's estimator)."""
import json, math, sys
import numpy as np
import walk_brain as wb
L, t0 = sys.argv[1], float(sys.argv[2])
span = float(sys.argv[3]) if len(sys.argv) > 3 else 4.5
rows = []
for l in open(L):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    if t0 - 0.5 <= r.get("t", 0) <= t0 + span and r.get("q") and r.get("quat"):
        rows.append(r)
odom = wb.LegOdometry()
def tilt(g):
    n = math.sqrt(sum(v * v for v in g)) or 1
    return math.degrees(math.acos(max(-1, min(1, -g[2] / n))))
out, switches, last_st = [], [], None
for r in rows:
    q = dict(zip(wb.NAMES, r["q"]))
    yaw = wb.yaw_of(wb.body_quat(r["quat"]))
    mid = odom.update(q, np.array(r["gb"]), yaw)
    if last_st is not None and odom.stance != last_st:
        switches.append(r["t"] - t0)
    last_st = odom.stance
    out.append((r["t"] - t0, mid.copy(), yaw, r))
m0, y0 = out[0][1], out[0][2]
c, s = math.cos(-y0), math.sin(-y0)
R0 = np.array([[c, -s], [s, c]])
def rel(m):
    return R0 @ (m - m0)
print("stance switches (foot changes): %d, at %s s" % (len(switches), ", ".join("%.2f" % x for x in switches)))
fin = rel(out[-1][1]); dyaw = math.degrees(((out[-1][2] - y0 + math.pi) % (2 * math.pi)) - math.pi)
print("odometry, start -> end: forward %+.1f cm, left %+.1f cm, heading %+.1f deg (+ = left / counter-clockwise)" % (fin[0] * 100, fin[1] * 100, dyaw))
print("  t     fwd cm  left cm  yaw deg  tilt  load  legs_mA  touch")
for t, m, yaw, r in out[::6]:
    p = rel(m)
    print("%+5.2f  %+6.1f  %+6.1f  %+6.1f  %5.1f  %.2f  %6s  %s" % (
        t, p[0] * 100, p[1] * 100, math.degrees(((yaw - y0 + math.pi) % (2 * math.pi)) - math.pi), tilt(r["gb"]),
        (r.get("hold") or {}).get("load", -1), r.get("legs_ma"), r.get("touch")))
