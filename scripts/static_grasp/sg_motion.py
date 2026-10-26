#!/usr/bin/env python
"""A static grasp played the way robotd would play it on the real duck, with NO grasp policy, in simulation.

Exact ground-grasp training model, 5 ms physics, BAM XL330 servos. The sequence uses robotd's own semantics
(microduck repo, robotd/src/main.rs):
  - stand: the shipped stand policy (alpha_stand.onnx) drives for --pre s, as when Rémi has pressed Start twice;
  - robot.poseJoints {targets, gain, ramp_s}: the policy stops; every servo ramps LINEARLY from its MEASURED
    position (not its previous target) to the target over ramp_s, at that servo gain, then holds;
  - robot.mouth {open}: the jaw target = -5 deg + open x 35 deg (live during a scripted pose);
  - robot.init out of a scripted pose: ramp from the measured positions to home over 2 s at gain 200, then hold home;
  - Start: the stand policy drives again.
An object (box / lying cylinder / upright capsule) can sit on the floor where the open beak will come down.

  PYTHONPATH=src .venv-mjlab/bin/python scripts/static_grasp/sg_motion.py POSE.json --object lego --gain 600
Writes notes/static-grasp/motion/<name>.json (stats + per-tick log) and <name>.npz (qpos per tick, for films).
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sg_hold as H
import sg_model as S

OUT = Path("/Users/remi/microduck/notes/static-grasp/motion")
STAND_ONNX = Path("/Users/remi/microduck/microduck/policies/alpha_stand.onnx")
PICK_ONNX = Path("/Users/remi/microduck/microduck/policies/alpha_ground_pick.onnx")   # the default grasp
PICK_PERIOD = 4.0
HOME = S.DEFAULT_POSE
MOUTH_CLOSED, MOUTH_OPEN = np.radians(-5.0), np.radians(30.0)
HEAD = range(5, 9)
OBJECTS = {
    # name: (shape, half sizes, mass kg)
    "lego": ("box", (0.008, 0.008, 0.0057), 0.002),          # 2x2 Lego brick, 1.6 x 1.6 x 1.14 cm
    "block": ("box", (0.010, 0.010, 0.008), 0.012),         # 2 x 2 x 1.6 cm block
    "die": ("box", (0.008, 0.008, 0.008), 0.004),           # 1.6 cm die
    "aa": ("cyl_lying", (0.007, 0.025, 0.0), 0.023),        # AA battery lying across the beak
    "pawn": ("capsule_up", (0.008, 0.005, 0.0), 0.004),     # small upright piece, 1.6 x 2.6 cm
    "coin": ("cyl_up", (0.0115, 0.001, 0.0), 0.005),        # a 2.3 cm coin lying flat (2 mm thick)
}


def mouth_rad(open_frac):
    return MOUTH_CLOSED + float(np.clip(open_frac, 0, 1)) * (MOUTH_OPEN - MOUTH_CLOSED)


def open_frac(theta):
    return float((theta - MOUTH_CLOSED) / (MOUTH_OPEN - MOUTH_CLOSED))


class Robotd:
    """The bits of robotd's loop a static grasp touches, on top of sg_hold.Sim."""

    def __init__(self, sim, seed=0):
        self.sim, self.m, self.d = sim, sim.m, sim.d
        self.sm = sim.sm
        self.st = ort.InferenceSession(str(STAND_ONNX))
        self.st_in = self.st.get_inputs()[0].name
        self.gp = ort.InferenceSession(str(PICK_ONNX))
        self.gp_in = self.gp.get_inputs()[0].name
        self.pick_t0 = None
        self.rng = np.random.default_rng(seed)
        self.mode = "policy"
        self.ramp = None              # (t0, from, to, dur)
        self.gain = 200.0
        self.mouth = MOUTH_CLOSED
        self.tbuf = deque([HOME.copy()] * 7, maxlen=7)
        self.jv_prev = np.zeros(14, np.float32)
        self.last_action = np.zeros(14, np.float32)
        self.prev_target = None
        self.t = 0.0
        self.gyro = int(self.m.sensor_adr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")])

    def measured(self):
        return self.d.qpos[self.sm.qadr].copy()

    def pose_joints(self, targets, gain=200.0, ramp_s=2.0):
        self.mode = "scripted"
        self.ramp = (self.t, self.measured(), np.array(targets, float), max(ramp_s, 1e-6))
        self.gain = gain

    def init(self):
        self.mode = "homing"
        self.ramp = (self.t, self.measured(), HOME.copy(), 2.0)
        self.gain = 200.0

    def pick(self):
        """robot.do ground_pick: the default grasp policy, phase 0 -> 0.7 over 2.8 s, then the stand policy."""
        self.mode = "pick"
        self.pick_t0 = self.t
        self.gain = 200.0

    def start(self):
        self.mode = "policy"
        self.prev_target = None
        self.gain = 200.0

    def _policy_target(self, net="stand"):
        d = self.d
        R = d.xmat[self.sm.trunk].reshape(3, 3)
        grav = (R.T @ np.array([0, 0, -1.0])).astype(np.float32)
        gyro = d.sensordata[self.gyro:self.gyro + 3].astype(np.float32)
        jpos = (self.measured() - HOME).astype(np.float32)
        jv = d.qvel[self.sm.dadr].astype(np.float32)
        cmd = np.zeros(13, np.float32)
        if net == "pick":
            ph = (self.t - self.pick_t0) / PICK_PERIOD
            cmd[0], cmd[1] = np.cos(2 * np.pi * ph), np.sin(2 * np.pi * ph)
        obs = np.concatenate([gyro, grav, jpos, self.jv_prev, self.last_action, cmd])[None]
        self.jv_prev = jv
        sess, inp = (self.gp, self.gp_in) if net == "pick" else (self.st, self.st_in)
        act = sess.run(None, {inp: obs.astype(np.float32)})[0].reshape(-1)
        self.last_action = act.astype(np.float32)
        target = HOME + act
        if self.prev_target is not None:          # robotd's low-pass on the targets
            for j in range(14):
                al = 0.5 if j in HEAD else 0.7
                target[j] = al * target[j] + (1 - al) * self.prev_target[j]
        self.prev_target = target.copy()
        return target

    def tick(self, push=None):
        """One 20 ms control tick."""
        if self.mode == "pick" and (self.t - self.pick_t0) / PICK_PERIOD >= 0.7:
            self.mode = "policy"
        if self.mode in ("policy", "pick"):
            target = self._policy_target("pick" if self.mode == "pick" else "stand")
            lag = int(self.rng.integers(3, 7))
        else:
            t0, a, b, dur = self.ramp
            u = np.clip((self.t - t0) / dur, 0, 1)
            target, lag = a + (b - a) * u, 1
            if self.mode == "homing" and u >= 1:
                self.mode = "held"
                self.ramp = (self.t, HOME.copy(), HOME.copy(), 1e-6)
        self.sim.bam.model.actuator.kp = self.gain
        self.d.ctrl[self.sim.mouth_act] = self.mouth
        self.d.xfrc_applied[self.sm.trunk, :3] = 0 if push is None else push
        for _ in range(4):
            self.tbuf.append(target)
            self.sim.bam.q_target[:] = self.tbuf[-1 - lag]
            self.sim.bam.update()
            mujoco.mj_step(self.m, self.d)
        self.t += H.DT_CTRL
        return target


def setup_object(sim, kind):
    m = sim.m
    ob = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object/object")
    og = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object/object_geom")
    oj = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[j] == ob][0]
    shape, half, mass = OBJECTS[kind]
    if shape == "cyl_lying":
        m.geom_type[og] = mujoco.mjtGeom.mjGEOM_CYLINDER
        m.geom_quat[og] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
        m.geom_sameframe[og] = 0
        z_rest, ext = half[0], (half[0], half[1], half[0])
    elif shape == "capsule_up":
        m.geom_type[og] = mujoco.mjtGeom.mjGEOM_CAPSULE
        z_rest, ext = half[0] + half[1], (half[0], half[0], half[0] + half[1])
    elif shape == "cyl_up":
        m.geom_type[og] = mujoco.mjtGeom.mjGEOM_CYLINDER
        z_rest, ext = half[1], (half[0], half[0], half[1])
    else:
        z_rest, ext = half[2], half
    m.geom_size[og] = half
    m.body_mass[ob] = mass
    a_ = ext
    m.body_inertia[ob] = [mass / 3 * (a_[1] ** 2 + a_[2] ** 2), mass / 3 * (a_[0] ** 2 + a_[2] ** 2),
                          mass / 3 * (a_[0] ** 2 + a_[1] ** 2)]
    m.geom_rbound[og] = float(np.linalg.norm(ext))
    m.geom_aabb[og] = [0, 0, 0, *ext]
    return dict(body=ob, geom=og, qadr=m.jnt_qposadr[oj], dadr=m.jnt_dofadr[oj], z_rest=z_rest)


def grasp_spot_in_feet_frame(sm, q, theta):
    """Where the grasp point (between the pads) lands, relative to the feet midpoint and heading, in the static pose."""
    sm.place(q, theta)
    d = sm.d
    gp = 0.5 * (d.site_xpos[sm.s_pad_up] + d.site_xpos[sm.s_pad_low])
    lip = 0.5 * (d.site_xpos[sm.s_lip_up] + d.site_xpos[mujoco.mj_name2id(sm.m, mujoco.mjtObj.mjOBJ_SITE, "robot/lip_low")])
    mid = 0.5 * (d.xpos[sm.feet[0]] + d.xpos[sm.feet[1]])
    return gp[:2] - mid[:2], lip[:2] - mid[:2]


def feet_frame(sim):
    d = sim.d
    mid = 0.5 * (d.xpos[sim.sm.feet[0]] + d.xpos[sim.sm.feet[1]])
    R = d.xmat[sim.sm.trunk].reshape(3, 3)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return mid[:2], yaw


def run(hold_json, obj=None, pre=2.5, hand_t=1.3, hand_ramp=0.1, open_t=0.3, settle_s=1.5, close_s=0.6,
        up="poses", up_s=2.0, gain=600.0, seed=0, name="run", obj_offset=(0.0, 0.0), qlog=True, target_noise=0.0,
        vin=7.4, final_targets=None):
    """The default grasp policy takes the duck down, then one robot.poseJoints switches to the static grasp pose.
    Timeline (robotd calls; t_pick = pre):
      0                    stand policy drives (Start pressed twice)
      t_pick               robot.do ground_pick (the default grasp policy)
      t_pick + open_t      robot.mouth open (the jaw opens while the head is still high)
      t_pick + hand_t      robot.poseJoints(static grasp targets, gain, hand_ramp): the policy stops, the servos
                           ramp from where they are to the static pose and hold it
      + settle_s           robot.mouth 0 (the jaw closes and squeezes)
      + close_s            way up: "poses" = robot.poseJoints(standing, gain, up_s) then robot.init;
                           "init" = robot.init straight away (2 s ramp at gain 200)
      + 2.5                Start: the stand policy drives, 2.5 s"""
    P = json.load(open(hold_json))
    q, theta = np.array(P["q"]), float(P["theta"])
    k_g = H.K_SERVO * gain / 200.0
    sim = H.Sim(vin=vin, kp=200.0)
    rd = Robotd(sim, seed)
    m, d = sim.m, sim.d
    targets = np.array(final_targets) if final_targets is not None else q + sim.sm.evaluate(q, theta)["tau"] / k_g
    stand_targets = HOME + sim.sm.evaluate(HOME, MOUTH_CLOSED)["tau"] / k_g
    if target_noise:
        rng = np.random.default_rng(seed + 100)
        targets = targets + rng.normal(0, target_noise, 14)
    o = setup_object(sim, obj) if obj else None
    qpos0 = H.placed_qpos(sim, HOME, MOUTH_CLOSED)
    sim.reset(qpos0, HOME, MOUTH_CLOSED)
    if o is not None:
        d.qpos[o["qadr"]:o["qadr"] + 3] = [0.5, 0.5, o["z_rest"] + 0.0005]
    else:
        d.qpos[sim.sm.obj_q:sim.sm.obj_q + 3] = [2.0, 2.0, 0.01]
    mujoco.mj_forward(m, d)
    rd.start()
    spot, _ = grasp_spot_in_feet_frame(sim.sm, q, theta)
    T = dict(pick=pre, open=pre + open_t, hand=pre + (hand_t if hand_t is not None else 1.3))
    T["close"] = T["hand"] + hand_ramp + settle_s
    T["up"] = T["close"] + close_s
    T["init"] = T["up"] + (up_s + 0.5 if up == "poses" else 0.0)
    T["start"] = T["init"] + 2.5
    T["end"] = T["start"] + 2.5
    tick_of = {k: int(round(v / H.DT_CTRL)) for k, v in T.items()}
    log, qs, events, placed = [], [], [], None
    k = 0
    while k < tick_of["end"]:
        if k == tick_of["pick"]:
            if o is not None:     # the object goes where the static pose puts the open beak, from the feet now
                mid, yaw = feet_frame(sim)
                c, s_ = np.cos(yaw), np.sin(yaw)
                off = spot + np.asarray(obj_offset)
                xy = mid + np.array([c * off[0] - s_ * off[1], s_ * off[0] + c * off[1]])
                d.qpos[o["qadr"]:o["qadr"] + 3] = [xy[0], xy[1], o["z_rest"] + 0.0003]
                d.qpos[o["qadr"] + 3:o["qadr"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                d.qvel[o["dadr"]:o["dadr"] + 6] = 0
                mujoco.mj_forward(m, d)
                placed = d.xpos[o["body"]].copy()
            rd.pick()
            events.append((rd.t, "robot.do ground_pick (default grasp policy)"))
        if k == tick_of["open"]:
            rd.mouth = theta
            events.append((rd.t, f"robot.mouth {open_frac(theta):.2f} (jaw {np.degrees(theta):.0f} deg)"))
        if k == tick_of["hand"] and hand_t is not None:
            rd.pose_joints(targets, gain, hand_ramp)
            events.append((rd.t, f"robot.poseJoints(static grasp, gain {gain:g}, ramp {hand_ramp:g} s)"))
        if k == tick_of["close"] and hand_t is not None:
            rd.mouth = MOUTH_CLOSED
            events.append((rd.t, "robot.mouth 0 (closes, squeezes)"))
        if hand_t is None and k == int(round((pre + 0.36 * PICK_PERIOD) / H.DT_CTRL)):
            rd.mouth = MOUTH_CLOSED            # the default grasp: f9's beak timeline closes at phase 0.30-0.36
            events.append((rd.t, "robot.mouth 0 (closes, squeezes)"))
        if hand_t is None:
            pass
        elif k == tick_of["up"] and up == "poses":
            rd.pose_joints(stand_targets, gain, up_s)
            events.append((rd.t, f"robot.poseJoints(standing, gain {gain:g}, ramp {up_s:g} s)"))
        if k == tick_of["init"] and hand_t is not None:
            rd.init()
            events.append((rd.t, "robot.init (home ramp 2 s, gain 200)"))
        if k == tick_of["start"] and hand_t is not None:
            rd.start()
            events.append((rd.t, "Start (stand policy)"))
        rd.tick()
        k += 1
        snap = sim.snapshot()
        z = snap["zaxis"]
        row = dict(t=round(rd.t, 3), mode=rd.mode, beak=1000 * snap["beak_low"], lip_up=1000 * snap["lip_up"],
                   jaw_tip=1000 * snap["jaw_tip"], margin=100 * snap["margin"], head_N=snap["head_force"],
                   nonfoot=snap["nonfoot"], tilt=float(np.degrees(np.arccos(np.clip(z[2], -1, 1)))),
                   tau_max=float(np.abs(snap["tau"]).max()), mouth=float(np.degrees(d.qpos[sim.sm.mouth_q])))
        if o is not None:
            row["obj"] = d.xpos[o["body"]].tolist()
            cj = cu = cf = 0
            for c_ in d.contact[:d.ncon]:
                pair = (c_.geom1, c_.geom2)
                if o["geom"] in pair:
                    other = pair[1] if pair[0] == o["geom"] else pair[0]
                    nmo = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
                    cj |= nmo.endswith("jaw_plate_collision"); cu |= nmo.endswith("upper_beak_collision")
                    cf |= other == sim.floor
            row.update(cj=int(cj), cu=int(cu), cf=int(cf))
        log.append(row)
        if qlog:
            qs.append(d.qpos.copy())

    def phase(a, b):
        return [r for r in log if T[a] <= r["t"] < T[b]]
    pick, static, up_ph = phase("pick", "hand"), phase("hand", "up"), phase("up", "end")
    settled = [r for r in static if r["t"] >= T["hand"] + hand_ramp + 0.5]
    touch = [r for r in static if r["head_N"] > 0]
    st = dict(name=name, object=obj, gain=gain, hand_t=hand_t, hand_ramp=hand_ramp, up=up, up_s=up_s,
              target_noise=target_noise, vin=vin, seed=seed,
              fell=bool(max(r["tilt"] for r in log[-25:]) > 45 or any(r["nonfoot"] for r in log)),
              beak_min_pick_mm=min(r["beak"] for r in pick),
              beak_static_mean_mm=float(np.mean([r["beak"] for r in settled])),
              beak_static_min_mm=min(r["beak"] for r in settled),
              touch_static_s=len(touch) * H.DT_CTRL, touch_static_peak_N=max([r["head_N"] for r in touch] or [0.0]),
              touch_after_settle=bool(any(r["head_N"] > 0 for r in settled)),
              head_floor_pick=bool(max(r["head_N"] for r in pick) > 0),
              head_floor_up=bool(max(r["head_N"] for r in up_ph) > 0),
              margin_min_static_cm=min(r["margin"] for r in static),
              margin_settled_cm=float(np.mean([r["margin"] for r in settled])),
              margin_min_up_cm=min(r["margin"] for r in up_ph),
              tau_max=max(r["tau_max"] for r in log), upright_end=bool(log[-1]["tilt"] < 30),
              times=T, events=events)
    if o is not None:
        end = log[-1]
        lift = end["obj"][2] - placed[2]
        st.update(success=bool(end["cj"] and end["cu"] and not end["cf"] and lift > 0.01 and end["tilt"] < 45),
                  lift_cm=100 * lift, pinched_end=bool(end["cj"] and end["cu"]),
                  push_before_close_cm=100 * max(np.linalg.norm(np.array(r["obj"][:2]) - placed[:2])
                                                 for r in log if T["pick"] < r["t"] < T["close"]))
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(stats=st, log=log[::2]), open(OUT / f"{name}.json", "w"), default=float)
    if qlog:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)]
        np.savez(OUT / f"{name}.npz", t=np.array([r["t"] for r in log]), qpos=np.array(qs),
                 joint_names=np.array(names), events=np.array([str(e) for e in events]),
                 object_size=np.array(OBJECTS[obj][1] if obj else [0, 0, 0]),
                 object_shape=np.array(OBJECTS[obj][0] if obj else "none"))
    return st, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", help="a pose json (sg_search or sg_hold output): q + theta")
    ap.add_argument("--object", default=None, choices=list(OBJECTS))
    ap.add_argument("--gain", type=float, default=600.0)
    ap.add_argument("--hand-t", type=float, default=1.3, help="switch time after robot.do ground_pick, s")
    ap.add_argument("--no-switch", action="store_true", help="the default grasp alone (no static pose)")
    ap.add_argument("--targets", default=None, help="sg_hold json whose calibrated targets to use (same gain)")
    ap.add_argument("--hand-ramp", type=float, default=0.1)
    ap.add_argument("--up", default="poses", choices=["poses", "init"])
    ap.add_argument("--up-s", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="target noise, rad (calibration error)")
    ap.add_argument("--vin", type=float, default=7.4)
    ap.add_argument("--dx", type=float, default=0.0, help="object offset forward, m")
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    name = a.name or f"{Path(a.pose).stem}_{a.object or 'none'}_g{a.gain:g}_h{a.hand_t:g}_r{a.hand_ramp:g}_{a.up}{a.up_s:g}"
    ft = None
    if a.targets:
        HJ = json.load(open(a.targets))
        assert abs(HJ.get("kp", 200) - a.gain) < 1e-6, "targets calibrated at another gain"
        ft = HJ["targets"]
    if a.no_switch:
        name = a.name or f"default_grasp_{a.object or 'none'}"
    st, _ = run(a.pose, a.object, hand_t=None if a.no_switch else a.hand_t, final_targets=ft, hand_ramp=a.hand_ramp, up=a.up, up_s=a.up_s, gain=a.gain,
                seed=a.seed, name=name, obj_offset=(a.dx, a.dy), target_noise=a.noise, vin=a.vin)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in st.items()
                      if k not in ("events", "times")}), flush=True)


if __name__ == "__main__":
    main()
