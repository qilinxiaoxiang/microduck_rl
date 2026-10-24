"""approach.Approach closed in the as-built MuJoCo model: the same Brain and walk_wired command semantics as
pickup_sim.SimDuck (for_s, head command, guards) plus walk_wired's command smoothing (CMD_ALPHA 0.2), the head camera
rendered upright at 480 x 640 with the Radxa 13M's vertical FOV, frames at 8 fps that arrive CAM_LAT_S after capture,
orange_target.detect, and duck_kin's numpy projection with the logged state nearest each frame's capture time.

The robot's left drift is in by default as on the robot: a yaw torque on the trunk while the walk policy runs
(DRIFT_NM, the as-built sim stand-in brain_sim_check.py --yaw-torque uses) and the brain's yaw trim at the robot's
learned value (TRIM, ~/.duck_yaw_trim.json on the A7Z, 2026-10-24). Without them the first suite passed while the
robot turned the wrong way (approach.py docstring). The drift was fitted on straight walks only; whether the robot also
drifts while turning in place or standing on walk61 at zero command is not measured (2026-10-24), so --drift-when
picks: forward (default, only while vx is commanded) or walk61 (whenever walk61 runs, in place and at zero command too).

  ~/.venvs/microduck-dyn/bin/python approach_sim.py --obj 0.6 -0.2 [--drift 0.023] [--move 3 0 -0.2] [--out run.json]
  ~/.venvs/microduck-dyn/bin/python approach_sim.py --suite [--drift D] [--out-dir dir]   the cases below, one line each

Pass, an object in front: "arrived" and the true object (foot frame below the trunk origin, as the controller's)
0.05-0.17 m away within 25 deg of straight ahead, no fall, no guard trip. An object behind: the controller reports it
not found without walking (trunk path under 3 cm).
"""
import argparse
import collections
import json
import math
import os
import sys

SIM_PYTHON = os.path.expanduser("~/.venvs/microduck-dyn/bin/python")
if os.path.realpath(sys.prefix) != os.path.realpath(os.path.dirname(os.path.dirname(SIM_PYTHON))):
    os.execv(SIM_PYTHON, [SIM_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as apc  # noqa: E402
import duck_kin  # noqa: E402
from orange_target import detect  # noqa: E402
from pickup_sim import TICK, SimDuck  # noqa: E402

CMD_ALPHA = 0.2       # walk_wired: cmd += CMD_ALPHA * (target - cmd) every tick
FPS = 8.0
CAM_LAT_S = 0.15
W, H = 480, 640
DRIFT_NM = 0.015   # needs the robot's trim on a straight walk with the 4 deg IMU pitch (2026-10-24)
TRIM = -1.1079
# (object x, y, move): move = (s after start, dx, dy) or None; two far to the side (turn in place first), one moved
# beside the robot once it is close (turn in place there), the last one behind
SUITE = (((0.6, 0.0), None), ((0.86, -0.31), None), ((0.5, -0.2), None), ((0.55, 0.2), None), ((0.8, 0.35), None),
         ((0.8, 0.0), (3.0, 0.0, -0.2)), ((0.7, -0.15), (3.0, 0.0, 0.3)),
         ((0.35, -0.55), None), ((0.25, 0.45), None), ((0.45, 0.0), (5.0, 0.0, 0.15)), ((-0.2, -0.4), None))


class ApproachSimDuck(SimDuck):
    def __init__(self, obj_xy, obj_h, obj_r, volt, drift=DRIFT_NM, trim=TRIM, drift_when="forward"):
        super().__init__(obj_xy, obj_h, obj_r, volt)
        self.drift, self.drift_when = drift, drift_when
        self.brain.yaw_trim = trim
        self.imu_pitch_deg = apc_imu_pitch()   # the robot's sensor error, which the brain's trim undoes
        m = self.m
        f = (H / 2) / math.tan(math.radians(66.2) / 2)   # the render has square pixels: fx = fy
        self.K = duck_kin.Intrinsics(f, f, W / 2, H / 2, W, H)
        m.cam_fovy[self.cid] = self.K.vfov_deg
        self.renderer = mujoco.Renderer(m, height=H, width=W)
        self.raw = {"mode": "stand"}
        self.cmd_s = [0.0, 0.0, 0.0]
        self.hist = collections.deque(maxlen=150)
        self.frames_q = collections.deque()
        self.delivered = None
        self.next_frame_t = 0.0
        self.trips = []

    def send(self, cmd):
        super().send(cmd)
        self.raw = dict(self.want)

    def yaw(self):
        q = self.d.xquat[self.trunk]
        w, x, y, z = (float(v) for v in q)
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def tick(self):
        if self.burst_end is not None and self.t >= self.burst_end:  # walk_wired for_s expiry, on the raw command
            self.raw = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            if self.last_cmd.get("head"):
                self.raw["head"] = self.last_cmd["head"]
            self.burst_end = None
        raw = self.raw
        if raw.get("mode") == "walk":
            for k, key in enumerate(("vx", "vy", "vtheta")):
                self.cmd_s[k] += CMD_ALPHA * (float(raw.get(key, 0)) - self.cmd_s[k])
                if abs(self.cmd_s[k]) < 1e-4:
                    self.cmd_s[k] = 0.0
            self.want = dict(raw, vx=self.cmd_s[0], vy=self.cmd_s[1], vtheta=self.cmd_s[2])
        else:
            self.cmd_s = [0.0, 0.0, 0.0]
            self.want = dict(raw)
        n_events = len(self.events)
        on = (self.info or {}).get("policy") == "walk" and (self.drift_when == "walk61" or abs(self.cmd_s[0]) > 0.02)
        self.d.xfrc_applied[self.trunk, 5] = self.drift if on else 0.0
        super().tick()
        if len(self.events) > n_events:
            self.trips += self.events[n_events:]
        if self.want.get("mode") != raw.get("mode"):   # a guard switched the mode: it holds, as in walk_wired
            self.raw = dict(self.want)
        self.hist.append(self.full_state())
        if self.t >= self.next_frame_t:
            self.next_frame_t += 1.0 / FPS
            self.frames_q.append((self.render(), self.t, self.frames))
        while self.frames_q and self.frames_q[0][1] + CAM_LAT_S <= self.t:
            self.delivered = self.frames_q.popleft()

    def object_trunk_frame(self):
        """The object's true position in the controller's foot frame: origin on the floor below the trunk origin."""
        R = self.d.xmat[self.trunk].reshape(3, 3)
        fwd = R[:, 0].copy()
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        left = np.array([-fwd[1], fwd[0], 0.0])
        rel = self.obj - self.d.xpos[self.trunk]
        return float(rel @ fwd), float(rel @ left)

    def full_state(self):
        return dict(self.state(), yaw=self.yaw())

    def state_at(self, t):
        return min(self.hist, key=lambda r: abs(r["t"] - t)) if self.hist else None

    def latest_frame(self):
        return self.delivered

    def render(self):
        self.renderer.update_scene(self.d, camera=self.cid)
        scn = self.renderer.scene
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, self.obj_half, self.obj, np.eye(3).ravel(),
                            np.array((1.0, 0.45, 0.0, 1.0), np.float32))
        self.frames += 1
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)


def apc_imu_pitch():
    from walk_brain import IMU_PITCH_TRIM_DEG
    return IMU_PITCH_TRIM_DEG


def reachable_case(obj):
    return obj[0] > 0   # in front: walked to (turning in place first if far to the side); behind: not searched for


def run(obj, move=None, obj_h=0.06, obj_r=0.02, volt=7.0, pitch_offset=0.0, max_s=90.0, quiet=False, drift=DRIFT_NM,
        trim=TRIM, drift_when="forward"):
    duck = ApproachSimDuck(obj, obj_h, obj_r, volt, drift, trim, drift_when)
    for _ in range(int(1.0 / TICK)):
        duck.tick()
    duck.send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})   # stand61, as the app leaves it
    for _ in range(int(2.0 / TICK)):
        duck.tick()
    logs = []

    def log(s):
        logs.append(s)
        if not quiet:
            print(s, flush=True)
    kin = duck_kin.DuckKin(duck.K, pitch_offset_deg=pitch_offset)
    ctl = apc.Approach(kin, duck.latest_frame, duck.full_state, duck.state_at, duck.send,
                       lambda f, accept=None: detect(f, min_area=apc.MIN_AREA, accept=accept), now=lambda: duck.t,
                       log=log, obj_radius=obj_r)
    path, last, t0 = 0.0, None, duck.t
    yaw0, yaw_min, yaw_max, yaw_prev, yaw_acc = duck.yaw(), 0.0, 0.0, duck.yaw(), 0.0
    moved = move is None
    while ctl.phase not in ("done", "failed") and duck.t - t0 < max_s:
        if not moved and duck.t - t0 >= move[0]:
            duck.obj[0] += move[1]
            duck.obj[1] += move[2]
            moved = True
            log(f"[sim] object moved by ({move[1]:+.2f}, {move[2]:+.2f})")
        for _ in range(2):
            duck.tick()
        ctl.step()
        y = duck.yaw()
        yaw_acc += apc.wrap(y - yaw_prev)
        yaw_prev = y
        yaw_min, yaw_max = min(yaw_min, yaw_acc), max(yaw_max, yaw_acc)
        xy = (float(duck.d.xpos[duck.trunk, 0]), float(duck.d.xpos[duck.trunk, 1]))
        if last is None or math.hypot(xy[0] - last[0], xy[1] - last[1]) > 0.02:
            path += 0.0 if last is None else math.hypot(xy[0] - last[0], xy[1] - last[1])
            last = xy
    if ctl.phase not in ("done", "failed"):
        ctl.finish(False, f"time limit {max_s} s")
    for _ in range(int(2.0 / TICK)):
        duck.tick()
    ox, oy = duck.object_trunk_frame()
    r_true, b_true = math.hypot(ox, oy), math.degrees(math.atan2(oy, ox))
    res = ctl.result or {}
    meas_err = None if "x" not in res or not res.get("ok") else round(math.hypot(res["x"] - ox, res["y"] - oy), 3)
    reach = reachable_case(obj)
    if reach:
        ok = bool(res.get("ok")) and apc.DONE_R[0] <= r_true <= apc.DONE_R[1] and abs(b_true) <= 25 and \
            duck.fell is None and not duck.trips
    else:
        ok = not res.get("ok") and path < 0.03 and duck.fell is None
    return dict(ok=ok, obj=list(obj), move=move, reachable=reach, ctl_ok=res.get("ok"), why=res.get("why"),
                elapsed_s=round(duck.t - t0, 1), approaches=res.get("approaches"), true_r=round(r_true, 3),
                true_bearing_deg=round(b_true, 1), meas=[res.get("x"), res.get("y")], meas_err_m=meas_err,
                path_m=round(path, 2), yaw_range_deg=[round(math.degrees(yaw_min), 1), round(math.degrees(yaw_max), 1)],
                fell_at=duck.fell, trips=duck.trips, max_tilt=round(duck.max_tilt, 1), pitch_offset=pitch_offset,
                drift_nm=drift, trim=trim, drift_when=drift_when), logs, ctl.trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=float, nargs=2, default=(0.6, -0.2))
    ap.add_argument("--move", type=float, nargs=3, default=None, metavar=("S", "DX", "DY"))
    ap.add_argument("--obj-height", type=float, default=0.06)
    ap.add_argument("--volt", type=float, default=7.0)
    ap.add_argument("--pitch-offset", type=float, default=0.0, help="deg; a wrong projection pitch (robustness)")
    ap.add_argument("--drift", type=float, default=DRIFT_NM, help="N m yaw torque while walking (+ = left)")
    ap.add_argument("--trim", type=float, default=TRIM, help="brain yaw trim, rad/s")
    ap.add_argument("--drift-when", choices=("forward", "walk61"), default="forward")
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    kw = dict(obj_h=args.obj_height, volt=args.volt, pitch_offset=args.pitch_offset, max_s=args.max_s,
              drift=args.drift, trim=args.trim, drift_when=args.drift_when)
    if args.suite:
        rows = []
        for obj, move in SUITE:
            out, logs, trace = run(obj, move, quiet=True, **kw)
            rows.append(out)
            print(json.dumps(out), flush=True)
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                tag = f"obj_{obj[0]:+.2f}_{obj[1]:+.2f}" + ("_moved" if move else "")
                with open(os.path.join(args.out_dir, tag + ".json"), "w") as f:
                    json.dump(dict(out, log=logs, trace=trace), f)
        print(json.dumps(dict(passed=sum(r["ok"] for r in rows), of=len(rows), drift_nm=args.drift,
                              drift_when=args.drift_when, pitch_offset=args.pitch_offset)))
        return
    out, logs, trace = run(tuple(args.obj), tuple(args.move) if args.move else None, **kw)
    print(json.dumps(out))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(out, log=logs, trace=trace), f)


if __name__ == "__main__":
    main()
