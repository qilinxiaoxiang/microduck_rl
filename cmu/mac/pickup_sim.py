"""The whole beak-pickup chain closed in the as-built MuJoCo model: the same Brain (walk61 / stand61 / groundpick
policies), the same walk_wired.py command semantics (for_s bursts, head command, ground-pick hand-back, guards), the
head camera rendered from the model with an orange box in the scene, orange_target.detect on that frame, the
target_ground.DuckGeometry floor projection, and the same pickup.Pickup phase machine. Only the jaw is faked (the model
has no jaw): a "grip" holds when the mouth tip is within GRIP_REACH of the object centre, and the object then follows
the mouth tip.

  ~/.venvs/microduck-dyn/bin/python pickup_sim.py [--obj 0.5 -0.25] [--obj-height 0.09] [--volt 7.4] [--out trace.json]

Prints the pickup log, then one JSON line: ok, why, elapsed, bursts, trips, grip distance, object lift, end tilt.
"""
import argparse
import collections
import json
import math
import os
import sys

# cmu/ copy: runs in the repository's uv environment (uv run <script>); no private venv re-exec.
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
WHOLE = os.path.join(HERE, "..", "whole")
sys.path.insert(0, os.path.join(WHOLE, "dynamics", "realgap-20261014"))
sys.path.insert(0, os.path.join(WHOLE, "delay-repair-20260912"))
import evaluate_delay as ev  # noqa: E402
import pickup as pk  # noqa: E402
from orange_target import detect  # noqa: E402
from static_gap import cfg, feet_flat_rot, setup, sole_points  # noqa: E402
from target_ground import DuckGeometry  # noqa: E402
from walk_brain import (IMU_MOUNT_QUAT, MOUNT_FIX, NAMES, SIGN, VEL_UNIT, Brain, apply_inverse, qmul,  # noqa: E402
                        to_tick)
from walk_brain import STAND as BRAIN_STAND  # noqa: E402

MODELS = os.path.join(WHOLE, "runtime-overlay", "models")
TICK = 0.02
GRIP_REACH = 0.035        # m, mouth tip to object centre for the faked grip to hold
MAX_TILT_WALK = 30.0      # walk_wired guards
MAX_TILT_GP = 55.0
HARD_ACTION = 3.0
OVER_TICKS, HARD_WINDOW = 3, 10
GP_DONE = 0.9
ORANGE = (1.0, 0.45, 0.0, 1.0)


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def apply(q, v):
    return apply_inverse(conj(q), v)


class SimDuck:
    """The robot, its runtime and its camera in the model; pickup.Pickup drives it through send/state/observe."""

    def __init__(self, obj_xy, obj_h=0.09, obj_r=0.02, volt=7.4, pitch_offset=0.0, sensor_delay=1, action_delay=1):
        m, d, c, names, bam = ev.load(cfg(0.09, model="asbuilt", kp=200, voltage=volt))
        self.m, self.d, self.c, self.names, self.bam = m, d, c, names, bam
        self.idx = {n: names.index(n) for n in NAMES}
        p0 = np.array([BRAIN_STAND[n] for n in names])
        true0 = dict(zip(names, p0))
        Rt = feet_flat_rot(m, d, names, true0)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rt.ravel())
        ji = setup(m, d, names, true0, q)
        zmin = np.vstack(sole_points(m, d))[:, 2].min()
        d.qpos[2] -= zmin - 0.0005
        mujoco.mj_forward(m, d)
        self.qa, self.va = m.jnt_qposadr[ji], m.jnt_dofadr[ji]
        self.trunk = m.body("trunk_base").id
        self.gyro_adr = m.sensor("imu_ang_vel").adr[0]
        self.mouth = m.site("mouth_tip").id
        c.q_target = p0.copy()
        self.sub = round(TICK / m.opt.timestep)
        self.brain = Brain(MODELS, compensation=False)
        self.aq = collections.deque([p0.copy()] * (action_delay + 1), maxlen=action_delay + 1)
        self.sq = collections.deque(maxlen=sensor_delay + 1)
        self.age = sensor_delay * TICK + 0.004
        # the projection model the real pipeline uses (its own copy), and the sim camera aimed exactly like it
        self.geom = DuckGeometry(model="asbuilt", pitch_offset_deg=pitch_offset)
        true_cam = DuckGeometry(model="asbuilt", pitch_offset_deg=0.0)
        self.cid = m.camera("head_camera").id
        m.cam_fovy[self.cid] = true_cam.m.cam_fovy[true_cam.cid]
        m.cam_quat[self.cid] = true_cam.m.cam_quat[true_cam.cid]
        self.renderer = mujoco.Renderer(m, height=480, width=640)
        # the object: a visual box standing on the floor (no contact; the robot cannot push it)
        self.obj = np.array([obj_xy[0], obj_xy[1], obj_h / 2], float)
        self.obj_half = np.array([obj_r, obj_r, obj_h / 2])
        self.held = False
        # runtime state (walk_wired semantics)
        self.t = 0.0
        self.want = {"mode": "stand"}
        self.last_cmd = {}
        self.burst_end = None
        self.jaw_goal, self.jaw_deg, self.holding = None, 0.0, False
        self.hard_hits = collections.deque(maxlen=HARD_WINDOW)
        self.info = None
        self.tilt, self.max_tilt, self.fell = 0.0, 0.0, None
        self.sent, self.events = [], []
        self.grip_dist = None
        self.frames = 0
        self.dump_dir = None

    # --- walk_wired.py side -----------------------------------------------------------------------------------
    def send(self, cmd):
        self.sent.append((round(self.t, 2), dict(cmd)))
        self.last_cmd = dict(cmd)
        self.want = dict(cmd)
        self.burst_end = self.t + float(cmd["for_s"]) if cmd.get("mode") == "walk" and cmd.get("for_s") else None
        if "jaw" in cmd:
            self.jaw_goal = cmd["jaw"]

    def state(self):
        info = self.info or {}
        return dict(t=self.t, mode=self.brain.mode, policy=info.get("policy"),
                    torque=self.brain.mode in Brain.TORQUE_MODES, gb=info.get("gb", [0.0, 0.0, -1.0]),
                    q={n: float(v) for n, v in zip(NAMES, info.get("q", [BRAIN_STAND[n] for n in NAMES]))},
                    imu_age=self.age, gp_phase=info.get("gp_phase"), jaw_deg=self.jaw_deg, jaw_holding=self.holding,
                    jaw_goal=self.jaw_goal, fresh=True)

    def tick(self):
        m, d, brain = self.m, self.d, self.brain
        t = self.t
        if self.burst_end is not None and t >= self.burst_end:  # for_s expired: zero command, head kept if given
            self.want = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            if self.last_cmd.get("head"):
                self.want["head"] = self.last_cmd["head"]
            self.burst_end = None
        # sensors as the wired link reports them (one tick of delay)
        q_trunk = d.qpos[3:7].copy()
        q_sensor = qmul(qmul(tuple(q_trunk), conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))
        w_body = d.sensordata[self.gyro_adr:self.gyro_adr + 3]
        g_sensor = apply(MOUNT_FIX, apply(IMU_MOUNT_QUAT, w_body))
        sample = (q_sensor, tuple(float(x) for x in g_sensor), d.qpos[self.qa].copy(), d.qvel[self.va].copy())
        if not self.sq:
            self.sq.extend([sample] * self.sq.maxlen)
        self.sq.append(sample)
        q_sensor, g_sensor, qm, dqm = self.sq[0]
        pos_t = [to_tick(n, float(qm[self.idx[n]])) for n in NAMES]
        vel_t = [int(round(float(dqm[self.idx[n]]) / VEL_UNIT / SIGN[n])) for n in NAMES]
        want = self.want
        if want.get("mode") == "walk":  # walk_wired smooths the velocity command (CMD_ALPHA) inside the Brain path
            want_b = want
        else:
            want_b = want
        goal, kp, info = brain.tick(want_b, t, pos_t, vel_t, tuple(q_sensor), g_sensor, True, self.age, self.age)
        self.info = info
        # guards as walk_wired.py applies them
        act = info.get("action")
        if brain.mode in Brain.POLICY_MODES and act is not None:
            amax = max(abs(a) for a in act)
            self.hard_hits.append(amax > HARD_ACTION)
            if sum(self.hard_hits) >= OVER_TICKS:
                self.events.append((round(t, 2), f"STOP: action {amax:.2f} over {HARD_ACTION} rad"))
                self.want = {"mode": "stand"}
                self.hard_hits.clear()
        limit = MAX_TILT_GP if brain.mode == "groundpick" else MAX_TILT_WALK
        if brain.mode in Brain.POLICY_MODES and self.tilt > limit:
            self.events.append((round(t, 2), f"STOP: tilt {self.tilt:.1f} over {limit} deg"))
            self.want = {"mode": "monitor"} if brain.mode == "groundpick" else {"mode": "stand"}
        if brain.mode == "groundpick" and (info.get("gp_phase") or 0.0) >= GP_DONE and \
                float(want.get("phase", 0.0)) >= GP_DONE:
            self.want = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            self.events.append((round(t, 2), "ground pick complete, stand61"))
        # goals to the servo model
        if goal is not None and all(n in goal for n in self.names):
            target = np.array([goal[n] for n in self.names])
            if kp:
                self.bam.actuator.kp = kp
        else:
            target = self.aq[-1].copy()
        self.aq.append(target.copy())
        self.c.q_target = self.aq[0]
        for _ in range(self.sub):
            self.c.update()
            mujoco.mj_step(m, d)
        R = d.xmat[self.trunk].reshape(3, 3)
        self.tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
        self.max_tilt = max(self.max_tilt, self.tilt)
        if self.fell is None and self.tilt > 80:
            self.fell = round(t, 2)
        # the faked jaw and the object
        mouth = d.site_xpos[self.mouth]
        if self.jaw_goal == "open":
            self.jaw_deg, self.holding, self.held = 20.0, False, False
        elif self.jaw_goal == "grip":
            if not self.held:
                dist = float(np.linalg.norm(mouth - self.obj))
                self.grip_dist = dist if self.grip_dist is None else min(self.grip_dist, dist)
                self.held = dist < GRIP_REACH
            self.holding = self.held
            self.jaw_deg = 14.0 if self.held else -3.0
        elif self.jaw_goal is not None:
            self.jaw_deg, self.holding, self.held = float(self.jaw_goal), False, False
        if self.held:
            self.obj = mouth.copy() + np.array([0.0, 0.0, -self.obj_half[2] * 0.5])
        self.t += TICK

    # --- the camera ---------------------------------------------------------------------------------------------
    def render(self):
        self.renderer.update_scene(self.d, camera=self.cid)
        scn = self.renderer.scene
        g = scn.geoms[scn.ngeom]
        scn.ngeom += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, self.obj_half, self.obj, np.eye(3).ravel(),
                            np.array(ORANGE, np.float32))
        rgb = self.renderer.render()
        self.frames += 1
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def observe(self, q, gb, object_z):
        frame = self.render()
        det = detect(frame, min_area=80)
        if self.dump_dir:
            ox, oy = self.object_foot_frame()
            cv2.imwrite(os.path.join(self.dump_dir, f"f{self.frames:03d}_t{self.t:05.1f}_x{ox:.3f}_y{oy:.3f}_"
                                     f"{'hit' if det else 'none'}.png"), frame)
        if det is None:
            return None
        h, w = frame.shape[:2]
        xy = pk.project(det, self.geom, q, gb, w, h, float(self.obj_half[0]))
        if xy is None or xy[0] < 0.03:
            return None
        return dict(x=float(xy[0]), y=float(xy[1]), cut=bool(xy[2]), area=int(det["area"]), cx=det["cx"], cy=det["cy"],
                    frame=self.frames)

    def object_foot_frame(self):
        """The object's true position in the foot frame (x forward along the trunk heading, y left)."""
        R = self.d.xmat[self.trunk].reshape(3, 3)
        feet = [self.m.geom(g).id for g in ("left_foot_collision", "right_foot_collision")]
        mid = np.mean([self.d.geom_xpos[g] for g in feet], axis=0)
        fwd = R[:, 0].copy()
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        left = np.array([-fwd[1], fwd[0], 0.0])
        rel = self.obj - mid
        return float(rel @ fwd), float(rel @ left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", type=float, nargs=2, default=(0.5, -0.25), help="object x y in the start foot frame (m)")
    ap.add_argument("--obj-height", type=float, default=0.09)
    ap.add_argument("--obj-radius", type=float, default=0.02)
    ap.add_argument("--volt", type=float, default=7.4)
    ap.add_argument("--pitch-offset", type=float, default=0.0, help="deg; a wrong projection pitch, to test robustness")
    ap.add_argument("--max-s", type=float, default=150.0)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default=None, help="JSON trace path")
    ap.add_argument("--dump-dir", default=None, help="save every camera frame here (debug)")
    args = ap.parse_args()
    duck = SimDuck(tuple(args.obj), args.obj_height, args.obj_radius, args.volt, args.pitch_offset)
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        duck.dump_dir = args.dump_dir
    for _ in range(int(1.0 / TICK)):  # a second on the open-loop stand pose, as walk_wired starts
        duck.tick()
    logs = []

    def log(s):
        logs.append(s)
        print(s, flush=True)
    p = pk.Pickup(duck.observe, duck.state, duck.send, now=lambda: duck.t, log=log, object_z=args.obj_height / 2,
                  dry=args.dry)
    truth = []
    while p.phase not in ("done", "failed") and duck.t < args.max_s:
        for _ in range(5):
            duck.tick()
        p.step()
        if len(truth) == 0 or duck.t - truth[-1][0] >= 1.0:
            ox, oy = duck.object_foot_frame()
            truth.append((round(duck.t, 1), p.phase, round(ox, 3), round(oy, 3), round(duck.tilt, 1)))
    end_t = duck.t
    for _ in range(int(3.0 / TICK)):  # settle after the machine stops
        duck.tick()
    ox, oy = duck.object_foot_frame()
    bursts = [c for _, c in duck.sent if c.get("mode") == "walk" and c.get("for_s")]
    out = dict(ok=bool(p.result and p.result.get("ok")), why=p.result.get("why") if p.result else None, phase=p.phase,
               elapsed_s=round(end_t, 1), bursts=len(bursts), trips=p.trips, grip_dist_m=None if duck.grip_dist is None
               else round(duck.grip_dist, 4), held=duck.held, object_z=round(float(duck.obj[2]), 3),
               object_foot_frame=[round(ox, 3), round(oy, 3)], fell_at=duck.fell, max_tilt=round(duck.max_tilt, 1),
               end_tilt=round(duck.tilt, 1), end_mode=duck.brain.mode, frames=duck.frames, events=duck.events,
               obj=list(args.obj), obj_height=args.obj_height, volt=args.volt, pitch_offset=args.pitch_offset)
    print(json.dumps(out))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(out, log=logs, sent=duck.sent, truth=truth, trace=p.trace), f)


if __name__ == "__main__":
    main()
