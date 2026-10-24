"""approach.Approach closed in the as-built MuJoCo model: the same Brain and walk_wired command semantics as
pickup_sim.SimDuck (for_s, head command, guards) plus walk_wired's command smoothing (CMD_ALPHA 0.2), the head camera
rendered upright at 480 x 640 with the Radxa 13M's vertical FOV, frames at 8 fps that arrive CAM_LAT_S after capture,
orange_target.detect, and duck_kin's numpy projection with the logged state nearest each frame's capture time.

  ~/.venvs/microduck-dyn/bin/python approach_sim.py --obj 0.6 -0.2 [--pitch-offset 3] [--out run.json]
  ~/.venvs/microduck-dyn/bin/python approach_sim.py --suite [--out-dir dir]      the placements below, one line each

Pass: the controller reports "arrived" and the true object (foot frame below the trunk origin, as the controller's) is 0.08-0.30 m away within
25 deg of straight ahead, with no fall and no guard trip.
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
SUITE = ((0.6, 0.0), (0.5, -0.25), (0.45, 0.3), (0.35, -0.55), (0.8, 0.35), (0.25, 0.45), (-0.2, -0.4))


class ApproachSimDuck(SimDuck):
    def __init__(self, obj_xy, obj_h, obj_r, volt):
        super().__init__(obj_xy, obj_h, obj_r, volt)
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


def run(obj, obj_h=0.06, obj_r=0.02, volt=7.0, pitch_offset=0.0, max_s=90.0, quiet=False, vth_max=None):
    duck = ApproachSimDuck(obj, obj_h, obj_r, volt)
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
    if vth_max is not None:
        apc.VTH_MAX = vth_max
    ctl = apc.Approach(kin, duck.latest_frame, duck.full_state, duck.state_at, duck.send,
                       lambda f, accept=None: detect(f, min_area=apc.MIN_AREA, accept=accept), now=lambda: duck.t, log=log, obj_radius=obj_r)
    path, last, t0 = 0.0, None, duck.t
    while ctl.phase not in ("done", "failed") and duck.t - t0 < max_s:
        for _ in range(2):
            duck.tick()
        ctl.step()
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
    meas_err = None if "x" not in res else round(math.hypot(res["x"] - ox, res["y"] - oy), 3)
    ok = bool(res.get("ok")) and 0.08 <= r_true <= 0.30 and abs(b_true) <= 25 and duck.fell is None and not duck.trips
    return dict(ok=ok, obj=list(obj), ctl_ok=res.get("ok"), why=res.get("why"), elapsed_s=round(duck.t - t0, 1),
                approaches=res.get("approaches"), true_r=round(r_true, 3), true_bearing_deg=round(b_true, 1),
                meas=[res.get("x"), res.get("y")], meas_err_m=meas_err, path_m=round(path, 2), fell_at=duck.fell,
                trips=duck.trips, max_tilt=round(duck.max_tilt, 1), pitch_offset=pitch_offset,
                vth_max=apc.VTH_MAX), logs, ctl.trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=float, nargs=2, default=(0.6, -0.2))
    ap.add_argument("--obj-height", type=float, default=0.06)
    ap.add_argument("--volt", type=float, default=7.0)
    ap.add_argument("--pitch-offset", type=float, default=0.0, help="deg; a wrong projection pitch (robustness)")
    ap.add_argument("--vth-max", type=float, default=None)
    ap.add_argument("--max-s", type=float, default=90.0)
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.suite:
        rows = []
        for obj in SUITE:
            out, logs, trace = run(obj, args.obj_height, volt=args.volt, pitch_offset=args.pitch_offset,
                                   max_s=args.max_s, quiet=True, vth_max=args.vth_max)
            rows.append(out)
            print(json.dumps(out), flush=True)
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                with open(os.path.join(args.out_dir, f"obj_{obj[0]:+.2f}_{obj[1]:+.2f}.json"), "w") as f:
                    json.dump(dict(out, log=logs, trace=trace), f)
        print(json.dumps(dict(passed=sum(r["ok"] for r in rows), of=len(rows))))
        return
    out, logs, trace = run(tuple(args.obj), args.obj_height, volt=args.volt, pitch_offset=args.pitch_offset,
                           max_s=args.max_s, vth_max=args.vth_max)
    print(json.dumps(out))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(out, log=logs, trace=trace), f)


if __name__ == "__main__":
    main()
