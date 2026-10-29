"""Dry run of walk_brain.Brain (the exact Mac-side mode logic) in the as-built MuJoCo model.

The Brain receives what the wired link would hand it: raw encoder ticks and velocity LSBs in this build's servo
conventions (SIGN/OFFSET), the BNO085 quaternion in its physical mount frame (IMU_MOUNT_QUAT and MOUNT_FIX undone),
gyro in that frame, and sample ages. Its goals go back through to_tick/to_model like walk_wired.py.

Schedule: mode "stand" for --stand-s (starts feet flat at walk_brain.STAND, i.e. pi/stand_pose.py STAND_POSE, and holds it), then {"mode": "walk", "vx": 0} (stand61) until
--walk-at, then vx for --walk-s, then zero velocity again (walk61 brings its stride to rest, walk_brain.Brain.stepped). Injected gaps as in dynamics/realgap-20261014/policy_test.py.

--kick-at T sends {"mode": "kick", "foot": --kick-foot, "id": 1} at T and rewrites it to walk at zero command once the
Brain reports the kick done, as walk_wired.py does. --sit-at T / --unsit-at T2 send {"mode": "sit"} from T and walk at
zero command from T2 (the Brain stands the robot back up first). Each reports fell or not, peak tilt and the time
from the hand-back to a stable stand (tilt within 2 deg of the stance before it and |gyro| < 0.3 rad/s for 1 s).

--battery-stop-at T replaces the schedule from T s with pi/battery.py stop_command, exactly as walk_wired.py does after
a battery stop, and runs --after-stop s past T (default 10). --stop-ramp also switches to stand mode (open-loop ramp
to STAND) that many seconds after the stop, the sequence first planned and rejected after these runs.

Lifted by the head (hold.py, 2026-10-22): --grip-at T --release-at T2 covers the head pads L1, R1 and M (synthetic raw
counts, full level) from T to T2; --lift-at T --lower-at T2 pulls the head body (jaw_soft, which carries head_roll) with a
spring-damper hand toward a point that rises --lift-height over 0.6 s at T and returns to the start height over 0.8 s at
T2; the hand lets go at --release-at (or 1.2 s after --lower-at without a grip). hold.HoldEstimator runs every tick on the
pad counts and the foot-floor load share (normal contact force / m g); --hold passes its estimate to the Brain as
want["hold"] (policy authority s and per-joint P gains, which reach the simulated servos as a 14-element kp), without
it the estimate is only logged (shadow, today's behaviour). The fall check (tilt > 45 deg) pauses while the hand carries
the robot. --wired-guards adds walk_wired.py's walk-mode stops: an action over 3.0 rad in 3 of 10 ticks or tilt over
30 deg holds the last goals and switches to stand mode for the rest of the run, a single tick over 3.0 is clipped; with
--hold the tilt stop is skipped while the grip engagement e > 0.3, and the checks use the applied (scaled) action.
--wired-hold (with --hold) gates the hold as walk_wired.py does (2026-10-23): the velocity command is smoothed with
CMD_ALPHA, a walk meeting a head grip is stopped (GripStop), the hold acts only where hold_allowed says so (at once
after a grip stop) and the authority ramps across every on/off edge; without it the estimate reaches the Brain in any
walk-mode tick, as in the 2026-10-22 runs.

  ~/.venvs/microduck-dyn/bin/python brain_sim_check.py [--volt 5.0] [--walk-ramp 0] [--com-x 5] [--zero left_hip_pitch -8 ...]
"""
import argparse
import collections
import json
import math
import os
import sys

# The as-built model uses bam at the commit locked in uv.lock (the one Pollen trains with).
# any other interpreter re-runs this script under that one.
# cmu/ copy: runs in the repository's uv environment (uv run <script>); no private venv re-exec.

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
WHOLE = os.path.join(HERE, "..", "whole")
sys.path.insert(0, os.path.join(WHOLE, "dynamics", "realgap-20261014"))
sys.path.insert(0, os.path.join(WHOLE, "delay-repair-20260912"))
import evaluate_delay as ev  # noqa: E402
from static_gap import cfg, feet_flat_rot, setup, sole_points  # noqa: E402
from walk_brain import IDS, IMU_MOUNT_QUAT, MOUNT_FIX, NAMES, VEL_UNIT, SIGN, Brain, apply_inverse, Breakaway, qmul, to_model, to_tick  # noqa: E402
from walk_brain import STAND as BRAIN_STAND  # noqa: E402
from battery import stop_command, stop_settled  # noqa: E402
from walk_wired import HARD_MAX_ACTION, HARD_WINDOW, MAX_TILT_DEG, MAX_TILT_GP_DEG, MAX_TILT_SIT_DEG, OVER_LIMIT_TICKS  # noqa: E402
from walk_wired import QUACK_HEAD_S, quack_head_block, quack_head_pose  # noqa: E402
from walk_wired import CMD_ALPHA, GripStop, hold_allowed  # noqa: E402
from walk_wired import SP_DONE_PHASE  # noqa: E402
import walk_brain  # noqa: E402
import jaw_ctl  # noqa: E402
import jaw_limits as jl  # noqa: E402
from hold import DOWN_RATE, PAD_HI, PADS, UP_RATE, HoldEstimator  # noqa: E402

MODELS = os.path.join(WHOLE, "runtime-overlay", "models")


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


# --foot-play axes in each ankle body's frame: the ankle joint axis (local z), and world forward / world up at the Brain's
# stand pose with the feet flat (computed 2026-10-21 from the as-built model: world x -> left [-1, 0.002, 0], right [1, 0.002, 0];
# world z -> left [0.002, 0.996, -0.087], right [-0.002, 0.996, -0.088]).
FOOT_PLAY_AXES = {"pitch": {"left": [0, 0, 1], "right": [0, 0, 1]},
                  "roll": {"left": [-1, 0.002, 0], "right": [1, 0.002, 0]},
                  "yaw": {"left": [0.002, 0.996, -0.087], "right": [-0.002, 0.996, -0.088]}}


def apply(q, v):
    return apply_inverse(conj(q), v)


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


GRIP_PADS = ("L1", "R1", "M")  # a palm over the head top: the two large side pads and the middle one
# --sit-head: the seat is reached at SIT_PREP_S + SIT_SETTLE_S (2.8 s) after the sit command; the head moves after it
SIT_HEAD_DELAY_S = 3.0
PAD_BASE = 1000.0  # synthetic rest count of every pad


def smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Hand:
    """Spring-damper hand on the head body: vertical to a rising and falling target height, horizontal to the start point."""

    def __init__(self, m, d, args):
        self.m, self.d, self.a = m, d, args
        self.head = m.body("jaw_soft").id
        assert m.body(m.jnt_bodyid[m.joint("head_roll").id]).name == "jaw_soft"
        self.p0, self.R0 = None, None
        self.off_at = args.release_at if args.release_at is not None and args.lower_at is not None \
            and args.release_at > args.lower_at else (None if args.lower_at is None else args.lower_at + 1.2)
        self.v6 = np.zeros(6)

    def active(self, t):
        return self.a.lift_at is not None and t >= self.a.lift_at and (self.off_at is None or t < self.off_at)

    def target_z(self, t):
        a = self.a
        if a.lower_at is not None and t >= a.lower_at:
            return self.p0[2] + a.lift_height * (1.0 - smooth((t - a.lower_at) / 0.8))
        return self.p0[2] + a.lift_height * smooth((t - a.lift_at) / 0.6)

    def apply(self, t):
        d = self.d
        if not self.active(t):
            d.xfrc_applied[self.head] = 0
            return
        p = d.xipos[self.head].copy()
        if self.p0 is None:
            self.p0 = p.copy()
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.head, self.v6, 0)
        w, v = self.v6[:3], self.v6[3:]
        a = self.a
        f = np.zeros(6)
        f[:2] = a.hand_kxy * (self.p0[:2] - p[:2]) - a.hand_cxy * v[:2]
        f[2] = a.hand_k * (self.target_z(t) - p[2]) - a.hand_c * v[2]
        if self.R0 is None:
            self.R0 = d.xmat[self.head].reshape(3, 3).copy()
        # --hand-krot 0 (default): the palm only damps rotation, the head pivots in the hand; > 0: the grip holds the
        # head's orientation at the lift with this angular stiffness
        rv = np.zeros(3)
        if a.hand_krot:
            qe = np.zeros(4)
            mujoco.mju_mat2Quat(qe, (d.xmat[self.head].reshape(3, 3) @ self.R0.T).ravel())
            mujoco.mju_quat2Vel(rv, qe, 1.0)
        f[3:] = -a.hand_krot * rv - a.hand_crot * w
        d.xfrc_applied[self.head] = f


# ── Seeing pick (--sp-at) ─────────────────────────────────────────────────────────────────────────────────────────────
# The as-built model (microduck_rl robot_groundcontact.xml) has no jaw joint: the jaw mesh is part of the head body
# jaw_soft. The policy was trained on microduck_rl-grasp's ground-grasp robot (src/mjlab_microduck/robot/ground_grasp.py
# add_mouth), which gives the jaw its own body on hinge passive_mouth (-0.0873..0.55 rad, + opens) and replaces the
# underside hulls by boxes. SP_* below copy those numbers (that module imports mjlab, which this venv lacks). The jaw
# here is driven like the robot's ID15: an XL330 BAM model at walk_wired's jaw gain (jaw_ctl.KP_JAW) with the current
# limit of its current-based position mode (jaw_ctl.GOAL_CURRENT_MA), not the training env's 1.0 N m/rad, 0.5 N m servo.
SP_MOUTH_PIVOT = np.array([0.0037, 0.0, -0.0182])  # head-local: up, left, back
SP_MOUTH_RANGE = (-0.0873, 0.55)
SP_JAW_MASS = 0.010
SP_UNDERSIDE_HULLS = ("bottom_head_shell", "soft_mouth_top", "jaw", "jaw_soft")
SP_CLOSED_RAD = 0.0  # the CAD closed jaw (plate 0.2 mm under the upper beak box): jaw_ctl's "closed" reference
# training (cmu_seeing_pick_env_cfg.py at microduck_rl-grasp 0a520062): object, camera, landing spot
SP_LANDING_XY = (0.087, 0.0)  # microduck_static_grasp_env_cfg.OBJECT_NOMINAL_XY: the static pose's pad midpoint
SP_BALL_RADIUS = (0.0053, 0.0072)
SP_BOX_HALF_XY, SP_BOX_HALF_Z = (0.005, 0.010), (0.005, 0.009)
SP_CYL_RADIUS, SP_CYL_HALF_H = (0.005, 0.009), (0.004, 0.008)
SP_MASS = (0.0005, 0.010)
SP_FRICTION = (0.35, 1.1)
SP_CAM_PERIOD_TICKS = 5
SP_CAM_DELAY_S = (0.10, 0.30)
SP_CAM_DROP = (0.0, 0.3)
SP_CAM_STALL_PROB, SP_CAM_STALL_S = 0.02, (0.5, 1.5)
SP_CAM_BIAS, SP_CAM_NOISE, SP_CAM_NOISE_SIZE = 0.01, 0.004, 0.001
SP_CAM_HALF_FOV = (math.radians(51.0), math.radians(35.0))
SP_CAM_MIN_DEPTH = 0.015
SP_INIT_AGE_S = (0.1, 0.4)
SP_HELD_LIFT = 0.01
SP_HELD_FROM = 0.44  # microduck_static_grasp_env_cfg.TIMELINE held_from: grasp_held pays from here
SP_SUCCESS_TICKS = 20  # seeing_audit.SUCCESS_STEPS: held on at least this many ticks = a training-env success
SP_STOPPED_DEG_PER_TICK = 0.3  # --sp-holding stopped: 15 deg/s  # mdp.grasp_state "held": pinched, off the floor, lifted > 1 cm


def _uf(u, f):
    return np.array([u, 0.0, -f])


def sp_add_mouth(spec, beak_mu, beak_priority=False):
    """ground_grasp.add_mouth on the as-built spec: jaw body on passive_mouth, beak boxes, lip and pad sites.
    beak_priority: the beak boxes get MuJoCo priority 1, so their friction is the beak-object friction whatever the
    object's own (a lined lip; without it MuJoCo takes the larger of the pair)."""
    head = spec.body("jaw_soft")
    jaw = head.add_body(name="jaw", pos=SP_MOUTH_PIVOT)
    jaw.mass = SP_JAW_MASS
    jaw.inertia = [4e-6, 4e-6, 4e-6]
    jaw.ipos = _uf(-0.0134, 0.054) - SP_MOUTH_PIVOT
    jaw.explicitinertial = True
    head.mass = head.mass - SP_JAW_MASS
    j = jaw.add_joint(name="passive_mouth", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], range=list(SP_MOUTH_RANGE))
    j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    j.damping = np.zeros((3, 1))  # BAM sets damping and friction every step (as ev.load does for the 14 servos)
    j.frictionloss = 0.0
    j.solref_friction = [-5e4, -2e2]
    j.solimp_friction = [.99, .9999, .001, .5, 2]
    removed = []
    for g in list(head.geoms):
        if g.meshname in SP_UNDERSIDE_HULLS and not (g.contype == 0 and g.conaffinity == 0):
            removed.append(g.meshname)
            spec.delete(g)
    kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, group=3, contype=1, conaffinity=1)
    beak = dict(condim=4, friction=[beak_mu, 0.02, 0.0001])
    if beak_priority:
        beak["priority"] = 1
    head.add_geom(name="head_rear_collision", pos=_uf(-0.00685, 0.00175), size=[0.01005, 0.0459, 0.04025], **kw)
    head.add_geom(name="upper_beak_collision", pos=_uf(-0.00525, 0.060), size=[0.00395, 0.0459, 0.018], **kw, **beak)
    jaw.add_geom(name="jaw_plate_collision", pos=_uf(-0.0134, 0.05985) - SP_MOUTH_PIVOT,
                 size=[0.0040, 0.0440, 0.01785], **kw, **beak)
    for other in ("yaw_roll_motion", "neck_pitch"):
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = "jaw", other
    head.add_site(name="lip_up", pos=_uf(-0.0092, 0.0770), group=4)
    head.add_site(name="pad_up", pos=_uf(-0.0092, 0.0680), group=4)
    jaw.add_site(name="lip_low", pos=_uf(-0.0134, 0.0777) - SP_MOUTH_PIVOT, group=4)
    jaw.add_site(name="jaw_tip_bottom", pos=_uf(-0.0174, 0.0777) - SP_MOUTH_PIVOT, group=4)
    jaw.add_site(name="pad_low", pos=_uf(-0.0094, 0.0680) - SP_MOUTH_PIVOT, group=4)
    return removed


def sp_add_object(spec, obj):
    body = spec.worldbody.add_body(name="sp_object", pos=[-1.0, 0.0, obj["half"][2] + 0.0005])
    body.add_freejoint(name="sp_object_free")
    kw = dict(name="sp_object_geom", mass=obj["mass"], condim=4, friction=[obj["mu"], 0.02, 0.0001],
              contype=1, conaffinity=1, rgba=[1.0, 0.45, 0.05, 1.0])
    if obj["shape"] == "ball":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=list(obj["half"]), **kw)
    elif obj["shape"] == "box":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(obj["half"]), **kw)
    else:
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[obj["half"][0], obj["half"][2], 0.0], **kw)


def sp_add_jaw_actuator(spec, limit):
    a = spec.add_actuator(name="jaw_servo", target="passive_mouth", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    a.set_to_motor()
    a.ctrllimited = False
    a.forcelimited = True
    a.forcerange = [-limit, limit]
    a.gear = [1, 0, 0, 0, 0, 0]


def sp_object_draw(args, rng):
    """The object of one run: shape from --sp-shape, sizes, mass and friction from the flags or drawn per seed from
    the training ranges."""
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
    shape = args.sp_shape
    if shape == "ball":
        half = [u(SP_BALL_RADIUS) for _ in range(3)]
    elif shape == "box":
        half = [u(SP_BOX_HALF_XY), u(SP_BOX_HALF_XY), u(SP_BOX_HALF_Z)]
    else:
        r = u(SP_CYL_RADIUS)
        half = [r, r, u(SP_CYL_HALF_H)]
    if args.sp_half is not None:  # exact half extents (m): x, y, z (ball: three radii; cylinder: r, r, half height)
        half = list(args.sp_half)
    if args.sp_size is not None:  # largest horizontal width (m): scale all three half extents
        half = [h * args.sp_size / (2 * max(half[:2])) for h in half]
    mass = args.sp_mass / 1000 if args.sp_mass is not None else u(SP_MASS)
    mu = args.sp_friction if args.sp_friction is not None else u(SP_FRICTION)
    return dict(shape=shape, half=[round(h, 5) for h in half], mass=round(mass, 5), mu=round(mu, 3),
                size=round(2 * max(half[:2]), 5))


class SimCamera:
    """The training env's head camera (_camera_update) in the runtime's terms: a frame every 5 ticks (10 fps), a
    detection when the object centre is inside the 102 x 70 deg cone of the head_camera site and more than 1.5 cm in
    front of it, dropped with probability `drop`, frozen through stalls, delivered `delay` later with a per-run bias
    and per-frame noise. What the control file would hold: the newest delivered detection with its capture time."""

    def __init__(self, rng, delay_s, drop, bias, noise, size_noise, stalls=True):
        self.rng, self.drop, self.bias, self.noise, self.size_noise = rng, drop, np.array(bias), noise, size_noise
        self.delay = int(min(15, max(1, round(delay_s / 0.02))))
        self.offset = int(rng.integers(0, SP_CAM_PERIOD_TICKS))
        self.stalls = stalls
        self.frames = collections.deque(maxlen=32)  # (capture tick, capture t, measurement or None)
        self.last = None
        self.stall_until = -1
        self.n = dict(frames=0, visible=0, detected=0, delivered=0, stall_frames=0)
        self.last_delivery_t, self.last_visible_t = None, None

    def measure(self, xy, size, t):
        xy = np.asarray(xy) + self.bias + self.rng.normal(0, self.noise, 2)
        return {"x": round(float(xy[0]), 5), "y": round(float(xy[1]), 5),
                "size": round(float(size + self.rng.normal(0, self.size_noise)), 5), "t": round(float(t), 4)}

    def step(self, k, t, visible, xy, size):
        if (k + self.offset) % SP_CAM_PERIOD_TICKS == 0:
            self.n["frames"] += 1
            if self.stalls and k >= self.stall_until and self.rng.random() < SP_CAM_STALL_PROB:
                self.stall_until = k + int(round(self.rng.uniform(*SP_CAM_STALL_S) / 0.02))
            stalled = k < self.stall_until
            self.n["stall_frames"] += stalled
            self.n["visible"] += visible
            if visible:
                self.last_visible_t = t
            ok = visible and not stalled and self.rng.random() >= self.drop
            self.n["detected"] += ok
            self.frames.append((k, t, self.measure(xy, size, t) if ok else None))
        for fk, ft, meas in self.frames:
            if fk == k - self.delay and meas is not None:
                self.last = meas
                self.n["delivered"] += 1
                self.last_delivery_t = t
        return self.last


def sp_contacts(m, d, obj_geom, named):
    """{name: summed normal force (N)} of the object's contacts with each named geom set, plus 'floor'."""
    f6, out = np.zeros(6), {k: 0.0 for k in named}
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if obj_geom not in (g1, g2):
            continue
        other = g2 if g1 == obj_geom else g1
        for k, gs in named.items():
            if other in gs:
                mujoco.mj_contactForce(m, d, i, f6)
                out[k] += float(f6[0])
    return out


def floor_force(m, d, floor, geoms):
    f6, tot = np.zeros(6), 0.0
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if (g1 == floor and g2 in geoms) or (g2 == floor and g1 in geoms):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += float(f6[0])
    return tot


def sp_install_model_hook(args, obj):
    """ev.load builds the robot from the MJCF and applies microduck_rl/cmu/model/visual.py to the spec before it
    compiles; this wraps that call to add the jaw, the object and the jaw actuator (evaluate_delay.py unchanged)."""
    sys.path.insert(0, str(ev.MODEL_DIR.parents[3] / "cmu" / "model"))
    import visual  # noqa: E402  (the module ev.load imports the hook from)
    from bam.model import load_model
    bj = load_model(motor_name="xl330", model="m6")
    limit = min(args.volt * bj.kt.value / bj.R.value, args.sp_jaw_current / 1000 * bj.kt.value)
    orig = visual.apply_visual_to_spec
    report = {}

    def hooked(spec, *a, **kw):
        n = orig(spec, *a, **kw)
        report["removed_hulls"] = sp_add_mouth(spec, args.sp_beak_mu, args.sp_beak_priority)
        sp_add_object(spec, obj)
        sp_add_jaw_actuator(spec, limit)
        for k in list(spec.keys):  # the MJCF keyframes do not list the new joints; nothing here uses them
            spec.delete(k)
        return n

    visual.apply_visual_to_spec = hooked
    report["jaw_torque_limit_nm"] = round(limit, 4)
    return report


class SeePick:
    """--sp-at: the object, the camera stand-in, the jaw as walk_wired.py drives it, and the run's numbers."""

    def __init__(self, m, d, args, obj, rng, model_report):
        from bam.model import load_model
        from bam.mujoco import MujocoController
        self.m, self.d, self.a, self.obj, self.rng = m, d, args, obj, rng
        bj = load_model(motor_name="xl330", model="m6")
        bj.actuator.kp = args.sp_jaw_kp
        bj.actuator.vin = args.volt
        bj.actuator.max_current = args.sp_jaw_current / 1000  # Goal Current of mode 5 (current-based position)
        qpos, qvel = d.qpos.copy(), d.qvel.copy()
        self.jctl = MujocoController(bj, ["jaw_servo"], m, d)  # its mj_setConst leaves d at qpos0: restore the pose
        d.qpos[:], d.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, d)
        jid = m.joint("passive_mouth").id
        self.jq = m.jnt_qposadr[jid]
        rest = SP_CLOSED_RAD + math.radians(walk_brain.SP_JAW_REST_DEG)
        d.qpos[self.jq] = rest
        self.jctl.q_target[:] = rest
        self.jaw_queue = collections.deque([rest] * (args.action_delay + 1), maxlen=args.action_delay + 1)
        self.jaw_meas = collections.deque(maxlen=args.sensor_delay + 1)
        # before the seeing pick the jaw rests at SP_JAW_REST_DEG with torque on (assumed; walk_wired leaves the jaw
        # untouched until a jaw command or the seeing pick, and its state before that is not modelled)
        self.goal_deg = walk_brain.SP_JAW_REST_DEG
        self.zero = jl.JawZero(SP_CLOSED_RAD, SP_CLOSED_RAD + 0.3)  # sim joint rad = "hardware rad", + opens
        self.body = m.body("sp_object").id
        self.oq = m.jnt_qposadr[m.joint("sp_object_free").id]
        self.odof = m.jnt_dofadr[m.joint("sp_object_free").id]
        self.ogeom = m.geom("sp_object_geom").id
        self.floor = m.geom("floor").id
        self.named = {"jaw": {m.geom("jaw_plate_collision").id}, "upper": {m.geom("upper_beak_collision").id},
                      "floor": {self.floor}}
        heads = {m.body("jaw_soft").id, m.body("jaw").id}
        self.head_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in heads and m.geom_contype[g]}
        self.site = {s: m.site(s).id for s in ("lip_up", "jaw_tip_bottom", "pad_up", "pad_low", "head_camera")}
        self.trunk = m.body("trunk_base").id
        u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731
        delay = args.sp_camera_delay if args.sp_camera_delay is not None else u(SP_CAM_DELAY_S)
        drop = args.sp_camera_drop if args.sp_camera_drop is not None else u(SP_CAM_DROP)
        bias = args.sp_camera_bias if args.sp_camera_bias is not None else list(rng.uniform(-SP_CAM_BIAS, SP_CAM_BIAS, 2))
        self.cam = SimCamera(rng, delay, drop, bias, args.sp_camera_noise, SP_CAM_NOISE_SIZE, stalls=not args.sp_no_stalls)
        self.cam_cfg = dict(delay_s=round(delay, 3), delay_ticks=self.cam.delay, drop=round(drop, 3),
                            bias_m=[round(float(b), 4) for b in bias], noise_m=args.sp_camera_noise, stalls=not args.sp_no_stalls,
                            half_fov_deg=[round(math.degrees(x), 1) for x in SP_CAM_HALF_FOV], min_depth_m=SP_CAM_MIN_DEPTH)
        self.model_report = model_report
        self.switched, self.done_t, self.spawn, self.k = None, None, None, 0
        self.rows = []
        self.cap = dict(ticks_open=0, min_cap_deg=None, min_cap_at=None, ticks_cap_below_trained=0,
                        first_below_trained=None, ticks_clamped=0, first_clamp=None, closed_zone_ticks=0,
                        clamp_names={}, max_head_pitch_open_deg=None)
        self.ev = dict(first_pinched_t=None, first_holding_bit_t=None, close_at=None, lip_min_m=None,
                       lip_min_phase=None, beak_floor_force_max_n=0.0, beak_floor_ticks=0, max_tilt=0.0,
                       min_trunk_z=None, max_action=0.0, first_lost_view_t=None,
                       # the object's travel in the foot frame: first beak touch -> pinch, and over the trained close
                       # window (phase 0.34 -> 0.40); the jaw command when --sp-close-on-contact started the close
                       first_contact=None, push_contact_to_pinch_mm=None, close_window=None, contact_close=None)
        self.close_shift = 0.0  # --sp-close-on-contact: phase advance of the jaw timeline after the first touch
        self.holding_now = False
        self.state = {}

    # -- the scene -------------------------------------------------------------------------------------------------
    def trunk_frame(self):
        p = self.d.xpos[self.trunk].copy()
        yaw = _yaw(self.d.xquat[self.trunk])
        return p, yaw

    def foot_xy(self, pw):
        p, yaw = self.trunk_frame()
        dx, dy = pw[0] - p[0], pw[1] - p[1]
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def place(self):
        p, yaw = self.trunk_frame()
        off = np.array(SP_LANDING_XY) + np.array(self.a.sp_offset)
        c, s = math.cos(yaw), math.sin(yaw)
        x, y = p[0] + c * off[0] - s * off[1], p[1] + s * off[0] + c * off[1]
        z = self.obj["half"][2] + 0.0005
        oyaw = float(self.rng.uniform(-math.pi, math.pi)) if self.a.sp_object_yaw is None else math.radians(self.a.sp_object_yaw)
        self.d.qpos[self.oq:self.oq + 7] = [x, y, z, math.cos(oyaw / 2), 0, 0, math.sin(oyaw / 2)]
        self.d.qvel[self.odof:self.odof + 6] = 0
        mujoco.mj_forward(self.m, self.d)
        self.spawn = np.array([x, y, z])
        self.obj_yaw_deg = round(math.degrees(oyaw), 1)
        self.root0 = (p.copy(), yaw)  # the standing root at the switch: seeing_audit's frame for pad_at_close

    def root0_xy(self, pw):
        p, yaw = self.root0
        dx, dy = pw[0] - p[0], pw[1] - p[1]
        c, s = math.cos(yaw), math.sin(yaw)
        return [round(c * dx + s * dy, 4), round(-s * dx + c * dy, 4)]

    def visible(self):
        cp = self.d.site_xpos[self.site["head_camera"]]
        R = self.d.site_xmat[self.site["head_camera"]].reshape(3, 3)
        p = R.T @ (self.d.xpos[self.body] - cp)  # site frame: x optical axis, y left, z up
        depth = p[0]
        return bool(depth > SP_CAM_MIN_DEPTH and abs(p[1]) < depth * math.tan(SP_CAM_HALF_FOV[0])
                    and abs(p[2]) < depth * math.tan(SP_CAM_HALF_FOV[1]))

    def opening_deg(self):
        q = self.jaw_meas[0] if self.jaw_meas else float(self.d.qpos[self.jq])
        return self.zero.opening_deg(q), q

    def holding(self):
        """jaw_ctl.JawCtl.holding: a grip command and the measured jaw at least GRIP_MIN_DEG from closed. --sp-holding
        stopped also asks the jaw to have stopped (under SP_STOPPED_DEG_PER_TICK over the last two reads)."""
        deg, _ = self.opening_deg()
        bit = self.goal_deg == "grip" and deg >= jaw_ctl.GRIP_MIN_DEG
        if bit and self.a.sp_holding == "stopped" and len(self.jaw_meas) >= 2:
            step = abs(math.degrees(self.jaw_meas[-1] - self.jaw_meas[-2]))
            self._jaw_steps = (getattr(self, "_jaw_steps", [])[-1:] + [step])
            bit = max(self._jaw_steps) < SP_STOPPED_DEG_PER_TICK
        return bit

    # -- per tick --------------------------------------------------------------------------------------------------
    def sense(self):
        """Start of a tick: the jaw's present position as the next read returns it (same delay as the body servos)."""
        q = float(self.d.qpos[self.jq])
        if not self.jaw_meas:
            self.jaw_meas.extend([q] * self.jaw_meas.maxlen)
        self.jaw_meas.append(q)

    def want(self, t):
        """The control-file command while the seeing pick runs: mode, the newest delivered detection, holding bit."""
        if self.switched is None:
            self.place()
            self.switched = t
            init_age = float(self.rng.uniform(*SP_INIT_AGE_S))
            self.cam.last = self.cam.measure(self.foot_xy(self.d.xpos[self.body]), self.obj["size"], t - init_age)
            self.cam_cfg["init_age_s"] = round(init_age, 3)
        self.k = int(round((t - self.switched) / 0.02))
        vis = self.visible()
        if not vis and self.ev["first_lost_view_t"] is None:
            self.ev["first_lost_view_t"] = round(t - self.switched, 2)
        last = self.cam.step(self.k, t, vis, self.foot_xy(self.d.xpos[self.body]), self.obj["size"])
        self.holding_now = self.holding()
        return {"mode": "seepick", "object": dict(last), "_holding": self.holding_now}

    def active(self, t):
        return self.a.sp_at is not None and t >= self.a.sp_at and self.done_t is None

    def after_tick(self, t, brain, info):
        """walk_wired.py after brain.tick: jaw goal from info["sp_jaw"], the hand-back at SP_DONE_PHASE, and later in
        the same tick jaw.write(link, brain.goal or measured_head, measured_head) through jaw_limits."""
        phase = info.get("sp_phase")
        if brain.mode == "seepick" and "sp_jaw" in info:
            self.goal_deg = info["sp_jaw"]
            if self.a.sp_close_on_contact and phase is not None:
                # start the trained close (0.34 -> 0.40, then grip) at the first beak-object touch when that comes
                # before phase 0.34; the policy's inputs are untouched (only the holding bit may come earlier)
                sj = walk_brain.SP_JAW
                if self.close_shift == 0.0 and sj["open_end"] <= phase < sj["close_start"] and self.ev["first_contact"]:
                    self.close_shift = sj["close_start"] - phase
                    self.ev["contact_close"] = dict(t_from_switch=round(t - self.switched, 2), phase=phase,
                                                    advance_phase=round(self.close_shift, 4),
                                                    advance_s=round(self.close_shift * walk_brain.SP_PERIOD, 2))
                if self.close_shift:
                    self.goal_deg = walk_brain.seepick_jaw(phase + self.close_shift)
        if brain.mode == "seepick" and info.get("action") is not None:  # walk_brain.clamp_goal at work
            for k, n in enumerate(walk_brain.OBSERVATION_DOF_ORDER):
                lo, hi = walk_brain.JOINT_RANGE.get(n, (-9, 9))
                v = brain.default[n] + float(info["action"][k])
                if v < lo - 1e-9 or v > hi + 1e-9:
                    gc = self.ev.setdefault("goal_clamped_ticks", {})
                    gc[n] = gc.get(n, 0) + 1
                    pk = self.ev.setdefault("goal_clamp_peak_rad", {})
                    over = v - hi if v > hi else v - lo
                    pk[n] = round(over, 3) if abs(over) > abs(pk.get(n, 0.0)) else pk.get(n, 0.0)
        if brain.mode == "seepick" and phase is not None and phase >= SP_DONE_PHASE and self.done_t is None:
            self.done_t = round(t, 2)
            self.state["handback"] = self.snapshot(brain)
        if brain.q is None:
            return
        measured_head = {n: float(brain.q[n]) for n in jl.POSE_JOINTS}
        commanded = brain.goal or measured_head
        deg = jaw_ctl.SQUEEZE_DEG if self.goal_deg == "grip" else float(self.goal_deg)
        targets = {"jaw": self.zero.jaw_rad_at(deg)}
        targets.update({n: float(commanded[n]) for n in jl.POSE_JOINTS if n in commanded})
        _, meas_q = self.opening_deg()
        if self.a.sp_no_cap:
            out, clamps = dict(targets), []
        else:
            out, clamps = jl.limit_targets(targets, self.zero, measured_jaw_rad=meas_q, measured_head=measured_head)
        head_used = {n: out[n] for n in jl.POSE_JOINTS if n in out}
        cap, closed, contact = jl.pose_jaw_cap_deg(head_used, measured_head)
        meant_open = brain.mode == "seepick" and phase is not None and \
            walk_brain.SP_JAW["open_start"] < phase < walk_brain.SP_JAW["close_end"]
        if meant_open:
            c = self.cap
            c["ticks_open"] += 1
            hp = math.degrees(commanded["head_pitch"])
            c["max_head_pitch_open_deg"] = round(hp if c["max_head_pitch_open_deg"] is None else max(hp, c["max_head_pitch_open_deg"]), 1)
            if c["min_cap_deg"] is None or cap < c["min_cap_deg"]:
                c["min_cap_deg"] = round(cap, 2)
                c["min_cap_at"] = dict(t_from_switch=round(t - self.switched, 2), phase=phase, contact_deg=round(contact, 2),
                                       commanded_head={n: round(float(commanded[n]), 3) for n in jl.POSE_JOINTS},
                                       measured_head={n: round(v, 3) for n, v in measured_head.items()})
            full_open = walk_brain.SP_JAW["open_end"] <= phase <= walk_brain.SP_JAW["close_start"]
            if full_open and cap < walk_brain.SP_JAW_OPEN_DEG - 1e-6:
                c["ticks_cap_below_trained"] += 1
                if c["first_below_trained"] is None:
                    c["first_below_trained"] = dict(t_from_switch=round(t - self.switched, 2), phase=phase, cap_deg=round(cap, 2))
            jaw_clamps = [x for x in clamps if x.startswith("jaw")]
            if jaw_clamps:
                c["ticks_clamped"] += 1
                if c["first_clamp"] is None:
                    c["first_clamp"] = dict(t_from_switch=round(t - self.switched, 2), phase=phase, wanted_deg=round(deg, 2),
                                            cap_deg=round(cap, 2), clamps=clamps)
            c["closed_zone_ticks"] += closed
            for x in clamps:
                c["clamp_names"][x] = c["clamp_names"].get(x, 0) + 1
        self.jaw_queue.append(out["jaw"])
        self.jctl.q_target[0] = self.jaw_queue[0]
        if self.a.sp_log is not None and self.switched is not None:
            self.rows.append(dict(t=round(t, 2), mode=brain.mode, phase=phase, jaw_goal=self.goal_deg,
                                  jaw_written_deg=round(self.zero.opening_deg(out["jaw"]), 2),
                                  jaw_measured_deg=round(self.zero.opening_deg(meas_q), 2), holding=self.holding_now,
                                  cap_deg=round(cap, 2), contact_deg=round(contact, 2), clamps=clamps,
                                  head_cmd={n: round(float(commanded[n]), 3) for n in jl.POSE_JOINTS},
                                  head_meas={n: round(v, 3) for n, v in measured_head.items()},
                                  body=info.get("sp_body"),
                                  obj_xyz=[round(float(v), 4) for v in self.d.xpos[self.body]],
                                  pad_up_xyz=[round(float(v), 4) for v in self.d.site_xpos[self.site["pad_up"]]],
                                  pad_low_xyz=[round(float(v), 4) for v in self.d.site_xpos[self.site["pad_low"]]],
                                  obj_contact_n={k: round(v, 3) for k, v in
                                                 sp_contacts(self.m, self.d, self.ogeom, self.named).items()},
                                  action_max=None if info.get("action") is None else
                                  round(max(abs(a) for a in info["action"]), 3)))

    def substep(self):
        self.jctl.update()

    def snapshot(self, brain):
        d = self.d
        f = sp_contacts(self.m, d, self.ogeom, self.named)
        pinched = f["jaw"] > 0 and f["upper"] > 0
        lift = float(d.xpos[self.body][2] - self.spawn[2])
        return dict(pinched=pinched, floor=f["floor"] > 0, lift_m=round(lift, 4),
                    held=bool(pinched and f["floor"] <= 0 and lift > SP_HELD_LIFT),
                    object_z_m=round(float(d.xpos[self.body][2]), 4), holding_bit=self.holding(),
                    jaw_opening_deg=round(self.opening_deg()[0], 2), squeeze_n={k: round(v, 3) for k, v in f.items()},
                    object_foot_xy=[round(float(v), 4) for v in self.foot_xy(d.xpos[self.body])])

    def observe(self, t, brain, info, tilt, trunk_z):
        """After the physics of a tick: the grasp and posture numbers."""
        if self.switched is None:
            return
        d, e = self.d, self.ev
        e["max_tilt"] = max(e["max_tilt"], tilt)
        e["min_trunk_z"] = round(trunk_z if e["min_trunk_z"] is None else min(e["min_trunk_z"], trunk_z), 4)
        if info.get("action") is not None:
            e["max_action"] = round(max(e["max_action"], max(abs(a) for a in info["action"])), 3)
        if self.done_t is not None and brain.mode != "seepick":
            return
        phase = info.get("sp_phase")
        lip = 0.5 * (d.site_xpos[self.site["lip_up"], 2] + d.site_xpos[self.site["jaw_tip_bottom"], 2])
        if self.rows and self.rows[-1]["t"] == round(t, 2):
            pad = 0.5 * (d.site_xpos[self.site["pad_up"]] + d.site_xpos[self.site["pad_low"]])
            fo = sp_contacts(self.m, d, self.ogeom, self.named)
            self.rows[-1].update(tilt=round(tilt, 1), lip_z=round(float(lip), 4), pad_z=round(float(pad[2]), 4),
                                 pad_foot_xy=[round(float(v), 4) for v in self.foot_xy(pad)],
                                 obj_foot_xy=[round(float(v), 4) for v in self.foot_xy(d.xpos[self.body])],
                                 obj_z=round(float(d.xpos[self.body][2]), 4), visible=self.visible(),
                                 obj_force={k: round(v, 3) for k, v in fo.items()},
                                 beak_floor_n=round(floor_force(self.m, d, self.floor, self.head_geoms), 3))
        if e["lip_min_m"] is None or lip < e["lip_min_m"]:
            e["lip_min_m"], e["lip_min_phase"] = round(float(lip), 4), phase
        ff = floor_force(self.m, d, self.floor, self.head_geoms)
        if ff > 0:
            e["beak_floor_ticks"] += 1
            e["beak_floor_force_max_n"] = round(max(e["beak_floor_force_max_n"], ff), 3)
        f = sp_contacts(self.m, d, self.ogeom, self.named)
        # the object's travel is reported three ways: in the foot frame (forward/left, moves with the robot), in the
        # world (the roll on the floor) and relative to the pad centre (out of the beak); mm
        pad_c = 0.5 * (d.site_xpos[self.site["pad_up"]] + d.site_xpos[self.site["pad_low"]])
        where = dict(object_foot_xy=[round(float(v), 4) for v in self.foot_xy(d.xpos[self.body])],
                     object_world_xy=[round(float(v), 4) for v in d.xpos[self.body][:2]],
                     object_rel_pad_xy=[round(float(v), 4) for v in self.foot_xy(d.xpos[self.body]) - self.foot_xy(pad_c)])

        def travel(since):
            out = {}
            for k, name in (("object_foot_xy", "foot"), ("object_world_xy", "world"), ("object_rel_pad_xy", "rel_pad")):
                dxy = np.asarray(where[k]) - np.asarray(since[k])
                out[name + "_mm"] = dict(x=round(float(dxy[0]) * 1000, 1), y=round(float(dxy[1]) * 1000, 1),
                                         norm=round(float(np.linalg.norm(dxy)) * 1000, 1))
            return out

        if (f["jaw"] > 0 or f["upper"] > 0) and e["first_contact"] is None:
            e["first_contact"] = dict(t_from_switch=round(t - self.switched, 2), phase=phase,
                                      which="both" if f["jaw"] > 0 and f["upper"] > 0 else ("jaw" if f["jaw"] > 0 else "upper"),
                                      jaw_measured_deg=round(self.opening_deg()[0], 2), **where)
        if f["jaw"] > 0 and f["upper"] > 0 and e["first_pinched_t"] is None:
            e["first_pinched_t"] = round(t - self.switched, 2)
            if e["first_contact"] is not None:
                e["push_contact_to_pinch_mm"] = dict(s=round(t - self.switched - e["first_contact"]["t_from_switch"], 2),
                                                     **travel(e["first_contact"]))
        if phase is not None and e["close_at"] is not None and e["close_window"] is None and \
                phase >= walk_brain.SP_JAW["close_end"]:
            e["close_window"] = dict(phase=phase, pinched=f["jaw"] > 0 and f["upper"] > 0,
                                     jaw_measured_deg=round(self.opening_deg()[0], 2), **where, **travel(e["close_at"]))
        if self.holding_now and e["first_holding_bit_t"] is None:
            e["first_holding_bit_t"] = round(t - self.switched, 2)
        pinched = f["jaw"] > 0 and f["upper"] > 0
        # the training success measure (seeing_audit.SUCCESS_STEPS): grasp_held pays (pinched, off the floor, lifted
        # more than 1 cm, phase >= TIMELINE held_from 0.44) on at least 20 ticks before the hand-back
        if phase is not None and phase >= SP_HELD_FROM and pinched and f["floor"] <= 0 and \
                float(d.xpos[self.body][2] - self.spawn[2]) > SP_HELD_LIFT:
            e["held_ticks"] = e.get("held_ticks", 0) + 1
        if self.holding_now and not pinched:
            e["holding_bit_without_pinch_ticks"] = e.get("holding_bit_without_pinch_ticks", 0) + 1
        if pinched and not self.holding_now:
            e["pinch_without_holding_bit_ticks"] = e.get("pinch_without_holding_bit_ticks", 0) + 1
        if phase is not None and phase >= walk_brain.SP_JAW["close_start"] and e["close_at"] is None:
            pad = pad_c
            e["close_at"] = dict(t_from_switch=round(t - self.switched, 2), phase=phase,
                                 pad_xy_err_m=round(float(np.linalg.norm(pad[:2] - d.xpos[self.body][:2])), 4),
                                 pad_z_m=round(float(pad[2]), 4),
                                 pad_foot_xy=[round(float(v), 4) for v in self.foot_xy(pad)],
                                 pad_root0_xy=self.root0_xy(pad), object_root0_xy=self.root0_xy(d.xpos[self.body]),
                                 jaw_measured_deg=round(self.opening_deg()[0], 2), tilt=round(tilt, 1),
                                 body_slots=info.get("sp_body"), **where)

    def summary(self, brain):
        out = dict(object=dict(self.obj, yaw_deg=getattr(self, "obj_yaw_deg", None)), offset_m=list(self.a.sp_offset),
                   landing_xy=list(SP_LANDING_XY), spawn_world=None if self.spawn is None else [round(float(v), 4) for v in self.spawn],
                   camera=dict(self.cam_cfg, **self.cam.n, last_delivery_from_switch_s=None if self.cam.last_delivery_t is None
                               else round(self.cam.last_delivery_t - self.switched, 2),
                               last_visible_from_switch_s=None if self.cam.last_visible_t is None
                               else round(self.cam.last_visible_t - self.switched, 2)),
                   jaw=dict(kp=self.a.sp_jaw_kp, goal_current_ma=self.a.sp_jaw_current, cap_applied=not self.a.sp_no_cap,
                            beak_priority=bool(self.a.sp_beak_priority), close_on_contact=bool(self.a.sp_close_on_contact),
                            beak_mu=self.a.sp_beak_mu, **self.model_report),
                   switched_at=self.switched, handback_at=self.done_t,
                   handback_from_switch_s=None if self.done_t is None or self.switched is None else round(self.done_t - self.switched, 2),
                   events=self.ev, jaw_cap=self.cap, at_handback=self.state.get("handback"))
        if self.switched is not None:
            out["at_end"] = self.snapshot(brain)
        return out


class SpTrace:
    """--sp-trace FILE: per seepick tick the 61 obs and 14 actions the policy saw and gave (captured at the ONNX call),
    the goals written, the true joint positions after the tick's physics (Brain order), trunk height and tilt, lip and
    pad heights, pad and object in the foot frame, beak-floor force and object contacts. Same fields as
    microduck_rl-grasp cmu/grasp/gap_trace.py writes from the training env (gap study, 2026-10-27)."""

    def __init__(self, brain, path):
        self.f = open(path, "w")
        self.last = None
        sess = brain.sessions["seepick"]
        tracer = self

        class Wrapped:
            def get_inputs(self):
                return sess.get_inputs()

            def get_modelmeta(self):
                return sess.get_modelmeta()

            def run(self, out, feed):
                res = sess.run(out, feed)
                tracer.last = (np.asarray(list(feed.values())[0])[0].tolist(), np.asarray(res[0])[0].tolist())
                return res

        brain.sessions["seepick"] = Wrapped()

    @staticmethod
    def box_min_z(m, d, name):
        """Lowest corner of a box geom (m, world z)."""
        g = m.geom(name).id
        R, c, h = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        return round(min(float((c + R @ (np.array([sx, sy, sz]) * h))[2]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)), 4)

    @staticmethod
    def head_contacts(m, d, sp):
        """Every active contact of a head or jaw geom: [geom, other geom, distance mm, normal force N]."""
        out, f6 = [], np.zeros(6)
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            if g1 in sp.head_geoms or g2 in sp.head_geoms:
                g, o = (g1, g2) if g1 in sp.head_geoms else (g2, g1)
                mujoco.mj_contactForce(m, d, i, f6)
                out.append([m.geom(g).name or f"g{g}", m.geom(o).name or f"g{o}", round(float(d.contact[i].dist) * 1000, 2),
                            round(float(f6[0]), 3)])
        return out

    def row(self, t, brain, info, goal, m, d, sp, qa, z0, names, tilt, tz):
        if brain.mode != "seepick" or info.get("sp_phase") is None or self.last is None or info.get("action") is None:
            return
        obs, act = self.last
        self.last = None
        order = walk_brain.OBSERVATION_DOF_ORDER
        qd = {n: float(d.qpos[qa[k]] - z0[k]) for k, n in enumerate(names)}
        s = sp.site
        pad = 0.5 * (d.site_xpos[s["pad_up"]] + d.site_xpos[s["pad_low"]])
        f = sp_contacts(m, d, sp.ogeom, sp.named)
        self.f.write(json.dumps(dict(
            t=round(t, 3), phase=info["sp_phase"], obs=[round(v, 4) for v in obs], action=[round(v, 4) for v in act],
            target=None if goal is None else [round(float(goal[n]), 4) for n in order],
            q=[round(qd[n], 4) for n in order], jaw_rad=round(float(d.qpos[sp.jq]), 4),
            trunk_z=round(tz, 4), tilt_deg=round(tilt, 2),
            lip_up_z=round(float(d.site_xpos[s["lip_up"], 2]), 4),
            lip_low_z=round(float(d.site_xpos[m.site("lip_low").id, 2]), 4),
            jaw_tip_bottom_z=round(float(d.site_xpos[s["jaw_tip_bottom"], 2]), 4), pad_z=round(float(pad[2]), 4),
            pad_foot_xy=[round(float(v), 4) for v in sp.foot_xy(pad)],
            obj_foot_xy=[round(float(v), 4) for v in sp.foot_xy(d.xpos[sp.body])],
            obj_z=round(float(d.xpos[sp.body][2]), 4),
            beak_floor_n=round(floor_force(m, d, sp.floor, sp.head_geoms), 3),
            jaw_contact=f["jaw"] > 0, upper_contact=f["upper"] > 0, pinched=f["jaw"] > 0 and f["upper"] > 0,
            head_contacts=self.head_contacts(m, d, sp),
            box_min_z={g: self.box_min_z(m, d, g) for g in ("upper_beak_collision", "jaw_plate_collision", "head_rear_collision")},
            floor_dist_mm={g: round(1000 * mujoco.mj_geomDistance(m, d, m.geom(g).id, sp.floor, 0.05, None), 2)
                           for g in ("upper_beak_collision", "jaw_plate_collision")})) + "\n")
        self.f.flush()


# The body-BVH refit now runs inside evaluate_delay.load for every caller (2026-10-29,
# bvh-refit-baselines-20261029/README.md). --bvh-refit stays an accepted flag so older command lines run; it repeats
# the refit, which finds nothing to change ("bvh refit: {}").
refit_body_bvh = ev.refit_body_bvh


def foot_load(m, d, floor, foot_geoms):
    """Summed normal force (N) of the foot-floor contacts."""
    f6, tot = np.zeros(6), 0.0
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if (g1 == floor and g2 in foot_geoms) or (g2 == floor and g1 in foot_geoms):
            mujoco.mj_contactForce(m, d, i, f6)
            tot += float(f6[0])
    return tot


def run(args):
    zero = {k: float(v) for k, v in args.zero}
    c0 = cfg(0.09, model=args.model, kp=200, voltage=args.volt)
    # sliding friction of the floor and of every geom on both foot bodies: all geoms share priority 0, so MuJoCo takes
    # the larger of the two and the foot-floor contact gets exactly this value (1.0 is the model's own)
    c0["friction"] = args.floor_mu
    if args.foot_play:  # sim-ID 2026-10-21: free play between ankle horn and foot (deg, centring stiffness N m/rad, damping)
        fp = args.foot_play + [0.05, 0.002][len(args.foot_play) - 1:]
        c0["foot_play"] = dict(deg=fp[0], stiffness=fp[1], damping=fp[2], axis=FOOT_PLAY_AXES[args.foot_play_axis])
    if args.manifest:  # another signed-part manifest on the same loader (the 90 g NP-F970 removal is still applied by the loader)
        c0["mass_manifest"] = os.path.abspath(args.manifest)
    if args.com_x:
        c0["mass_parts"] = [dict(body="trunk_base", mass_kg=-0.05, position_m=[0, 0, 0.0]),
                            dict(body="trunk_base", mass_kg=0.05, position_m=[args.com_x / 1000 * 0.76 / 0.05, 0, 0.0])]
    sp_rng = np.random.default_rng([args.seed, 7]) if args.sp_at is not None else None
    sp_obj = sp_object_draw(args, sp_rng) if args.sp_at is not None else None
    sp_model = sp_install_model_hook(args, sp_obj) if args.sp_at is not None else None
    m, d, c, names, bam = ev.load(c0)
    if args.bvh_refit:  # no-op since ev.load refits (2026-10-29); a second refit must find nothing to change
        print("bvh refit:", json.dumps(refit_body_bvh(m)), flush=True)
    if sp_model is not None:  # the training env's contact solver settings on request (gap study 2026-10-29)
        if args.sp_cone == "elliptic":
            m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        if args.sp_impratio is not None:
            m.opt.impratio = args.sp_impratio
        sp_model["contact"] = dict(cone=args.sp_cone, impratio=float(m.opt.impratio), bvh_refit=True)  # ev.load refits
        if args.sp_cone != "elliptic" or (args.sp_impratio or 1.0) < 10:
            print("seepick: WITHOUT --sp-cone elliptic --sp-impratio 10 the sim is not the one the policy was trained in "
                  "(a pinched object creeps out of the beak): seepick-gap-20261029/README.md", flush=True)
    if args.bam_friction_scale != 1.0:  # sim-ID knob (2026-10-17): scale BAM's stiction/load-friction budget, as the training env's friction DR does
        for pname in ("friction_base", "friction_stribeck", "load_friction_base", "load_friction_stribeck", "load_friction_motor",
                      "load_friction_external", "load_friction_motor_stribeck", "load_friction_external_stribeck"):
            prm = getattr(bam, pname, None)
            if prm is not None:
                prm.value = prm.value * args.bam_friction_scale
    if args.sole_solref:  # compliant sole (sim-ID 2026-10-21): the foot-floor contact's timeconst/dampratio, foot side dominant
        for g in ("left_foot_collision", "right_foot_collision"):
            m.geom_solref[m.geom(g).id] = args.sole_solref
            m.geom_solmix[m.geom(g).id] = 1e3
    idx = {n: names.index(n) for n in NAMES}  # Brain order (servo IDs 1..14) -> model actuator order
    z0 = np.array([math.radians(zero.get(n, 0)) for n in names])
    p0 = np.array([BRAIN_STAND[n] for n in names])  # the Brain's stand pose (pi/stand_pose.py), not static_gap.STAND
    true0 = dict(zip(names, p0 + z0))
    Rt = feet_flat_rot(m, d, names, true0)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Rt.ravel())
    ji = setup(m, d, names, true0, q)
    zmin = np.vstack(sole_points(m, d))[:, 2].min()
    d.qpos[2] -= zmin - 0.0005
    mujoco.mj_forward(m, d)
    qa, va = m.jnt_qposadr[ji], m.jnt_dofadr[ji]
    trunk = m.body("trunk_base").id
    gyro_adr = m.sensor("imu_ang_vel").adr[0]
    c.q_target = p0 + z0
    brain = Brain(args.models, compensation=args.compensation, walk_ramp=args.walk_ramp, yaw_trim_file=args.yaw_trim_file)
    comp = Breakaway(first=args.ba_first, ramp=args.ba_ramp, cap=args.ba_cap, min_err_ticks=args.ba_min_err, stuck_ticks=args.ba_stuck, cap_ticks=args.ba_cap_ticks)
    Rimu = rot_y(args.imu_pitch)
    Mq = qmul(IMU_MOUNT_QUAT, (1, 0, 0, 0))
    sub = round(0.02 / m.opt.timestep)
    aq = collections.deque([p0.copy()] * (args.action_delay + 1), maxlen=args.action_delay + 1)
    sq = collections.deque(maxlen=args.sensor_delay + 1)
    fell, fall_dir, maxtilt, x0, rows, first_policy = None, None, 0.0, None, [], None
    yaw_walk, yaw_prev, turns = 0.0, None, []
    path_m, path_xy = 0.0, None  # trunk path over the walk window, sampled every 0.5 s (stride sway averages out)
    rng = np.random.default_rng(args.seed)
    imu_hold, imu_t = None, 0.0
    max_action, action_jumps, prev_action = 0.0, 0, None
    T = args.walk_at + args.walk_s + 3.0
    dance = None
    if args.dance:
        with open(args.dance) as f:
            dance = json.load(f)
    dance_vth = [0.0]
    sit_keys = None
    restarted = False
    if args.sit_head_keys:
        with open(args.sit_head_keys[0]) as f:
            sit_keys = json.load(f)[args.sit_head_keys[1]]
    if args.quack_at:
        T = max(T, max(args.quack_at) + QUACK_HEAD_S + 1.0)
    gp_at = args.gp_at
    if gp_at is not None:
        # ground pick: stand61 until gp_at, phase gp_phase (descent at 1/4 per s) held gp_hold_s after it is reached,
        # then phase 1.0 (rise, rest), then back to walk mode at zero command (stand61) as walk_wired.py does
        gp_reached = gp_at + args.gp_phase * 4.0
        gp_rise_at = gp_reached + args.gp_hold_s
        gp_done_at = gp_rise_at + (1.0 - args.gp_phase) * 4.0 + 0.5
        T = max(T, gp_done_at + 3.0)
    sp = None
    if args.sp_at is not None:
        # seeing pick: stand61 until sp_at, then {"mode": "seepick", "object": ...} until the phase reaches
        # walk_wired.SP_DONE_PHASE, then walk mode at zero command (stand61) for --sp-after s
        sp = SeePick(m, d, args, sp_obj, sp_rng, sp_model)
        args.vx = 0.0
        T = max(T, args.sp_at + walk_brain.SP_PERIOD * SP_DONE_PHASE / (1.0 - args.drop) + 0.3 + args.sp_after)
    sp_trace = SpTrace(brain, args.sp_trace) if args.sp_trace and sp is not None else None
    mouth = m.site("mouth_tip").id
    feet = [m.geom(g).id for g in ("left_foot_collision", "right_foot_collision")]
    gp = dict(mouth_min_z=None, mouth_at_min=None, pose_at_min=None, tilt_at_min=None, max_tilt=0.0, phase_at_min=None)
    stop_at = args.battery_stop_at
    if stop_at is not None:
        T = max(T, stop_at + args.after_stop)
    settled_at, tilt_after_stop = None, 0.0
    if args.kick_at is not None:
        T = max(T, args.kick_at + 4.0)
    if args.sit_at is not None:
        T = max(T, (args.unsit_at if args.unsit_at is not None else args.sit_at + 6.0) + 6.0)
    if args.resit_at is not None:
        T = max(T, args.resit_at + args.resit_s)
    kick_done_t, kick_start_t, kick_foot_z0, kick_foot_max = None, None, None, None
    sit = dict(start=None, flag1=None, rise=None, done=None, min_trunk_z=None, seated_tilt=None, seated_trunk_z=None)
    zero_t, quack_ok, quack_reach = None, {}, {}  # quack head moves: zero-command start, quack time -> ran to the end
    hist = []  # (t, tilt, |gyro|, trunk z) every tick, for the stable-stand search
    kick_geom = m.geom(f"{args.kick_foot}_foot_collision").id
    # lifted by the head (hold.py): scripted grip, spring-damper hand, the estimator every tick
    scen = args.grip_at is not None or args.lift_at is not None
    T = max(T, args.until)
    est = HoldEstimator(kp_rl=walk_brain.KP_RL)
    hand = Hand(m, d, args)
    floor = m.geom("floor").id
    foot_bodies = {m.body(b).id for b in ("ankle_left", "ankle_right") if m.body(b) is not None}
    foot_bodies |= {i for i in range(m.nbody) if m.body(i).name.startswith("foot_play_")}
    foot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in foot_bodies}
    mg = float(m.body_subtreemass[0]) * 9.81
    act_id = {n: m.actuator(n).id for n in names}
    leg_names = [n for n in names if n not in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
    hrows = []  # per tick: t, load, est record, raw/applied action, neck torques, leg q, tilt, gyro, hand force, feet z
    guard = dict(stop=None, why=None, hard_hits=collections.deque(maxlen=HARD_WINDOW), last_goal=None, clipped=0,
                 tilt_skipped=0)
    load_now, tau_legs = 1.0, 0.0
    grip_stop, cmd_s, hold_s, wz_t, grip_stop_at = GripStop(), [0.0, 0.0, 0.0], 1.0, None, None  # --wired-hold
    settled_at = None
    for tick in range(int(T / 0.02)):
        t = tick * 0.02
        # what the sensors report, in commanded (encoder) frame and the BNO's own frame
        q_trunk = d.qpos[3:7].copy()
        if args.imu_pitch:
            qp = np.zeros(4)
            mujoco.mju_mat2Quat(qp, Rimu.ravel())
            q_trunk = np.array(qmul(q_trunk, qp))
        q_sensor = qmul(qmul(tuple(q_trunk), conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))
        w_body = Rimu.T @ d.sensordata[gyro_adr:gyro_adr + 3]
        g_sensor = apply(MOUNT_FIX, apply(IMU_MOUNT_QUAT, w_body))
        qm = d.qpos[qa] - z0
        dqm = d.qvel[va].copy()
        sample = (q_sensor, tuple(float(x) for x in g_sensor), qm.copy(), dqm)
        if not sq:
            sq.extend([sample] * sq.maxlen)
        sq.append(sample)
        if sp is not None:
            sp.sense()
        # --imu-hz holds the last IMU sample between samples (a sensor stress case). The robot's current IMU rate lives in microduck-sim/CURRENT-STATE.json, never in this comment
        imu_new = args.imu_hz <= 0 or tick == 0 or int(t * args.imu_hz) != int((t - 0.02) * args.imu_hz)
        if imu_new:
            imu_hold, imu_t = sample[:2], t
        sample = imu_hold + sample[2:]
        q_sensor, g_sensor, qm, dqm = sq[0]
        pos_t = [to_tick(n, float(qm[idx[n]])) for n in NAMES]
        vel_t = [int(round(float(dqm[idx[n]]) / VEL_UNIT / SIGN[n])) for n in NAMES]
        if t < args.stand_s:
            want = {"mode": "stand"}
        elif gp_at is not None and gp_at <= t < gp_done_at:
            want = {"mode": "groundpick", "phase": args.gp_phase if t < gp_rise_at else 1.0}
        else:
            walking = args.walk_at <= t < args.walk_at + args.walk_s
            vth = args.vtheta
            for at, v in args.vtheta_seq:  # the last step at or before this walk time wins
                if t - args.walk_at >= at:
                    vth = v
            want = {"mode": "walk", "vx": args.vx if walking else 0.0, "vy": 0.0,
                    "vtheta": vth if walking else 0.0}
            if (args.head_yaw or args.head_pitch is not None) and t >= args.walk_at:
                want["head"] = {"head_yaw": args.head_yaw, "head_pitch": args.head_pitch}
            if dance is not None and t >= args.walk_at:  # shoot.py's singing take: the step in force at this time
                step = None
                for st_ in dance["steps"]:
                    if t - args.walk_at >= st_["t"]:
                        step = st_
                if step is not None:
                    dv = step.get("vtheta", dance_vth[0]) if "vtheta" in step else dance_vth[0]
                    dance_vth[0] = dv
                    want = dict(want, vx=0.0, vtheta=dv if walking else 0.0,
                                head={n: brain.default[n] + d for n, d in step.get("head", {}).items()})
            if args.kick_at is not None and t >= args.kick_at and kick_done_t is None:
                want = {"mode": "kick", "foot": args.kick_foot, "id": 1}
            if args.sit_at is not None and t >= args.sit_at and (args.unsit_at is None or t < args.unsit_at):
                want = {"mode": "sit"}
                seat_t = args.sit_at + walk_brain.SIT_PREP_S + walk_brain.SIT_SETTLE_S + 0.2
                if sit_keys is not None and t >= seat_t and args.restart_at is None:  # shoot.py sitfind: seated head keyframes, linear ramps
                    want["head"] = {n: brain.default[n] + d for n, d in head_keys(sit_keys, t - seat_t).items()}
                if args.sit_head and t >= args.sit_at + SIT_HEAD_DELAY_S:  # robot-remote.js head buttons while seated
                    want["head"] = {n: brain.default[n] + d for n, d in args.sit_head.items()}
            if args.restart_at is not None and t >= args.restart_at:
                # the brain is stopped on the seated robot (the servos keep the last goals) and started again: a fresh
                # Brain in monitor mode, then {"mode": "sit"} from --resit-at, which enters the seat directly
                if not restarted:
                    brain, restarted = Brain(args.models, compensation=args.compensation, walk_ramp=args.walk_ramp,
                                             yaw_trim_file=args.yaw_trim_file), True
                want = {"mode": "monitor"}
                if args.resit_at is not None and t >= args.resit_at:
                    want = {"mode": "sit"}
                    if sit_keys is not None and t >= args.resit_at + 0.2:
                        want["head"] = {n: brain.default[n] + d for n, d in head_keys(sit_keys, t - args.resit_at - 0.2).items()}
            zero_t = None if walking else (t if zero_t is None else zero_t)
            for qt in args.quack_at:  # walk_wired.py's quack head move: the same gate and profile
                if qt <= t < qt + QUACK_HEAD_S:
                    # gated at the quack and on every tick after it, as walk_wired.py does
                    ok = quack_head_block(want, brain, None if zero_t is None else t - zero_t) is None
                    quack_ok[qt] = quack_ok.get(qt, ok) and ok
                    if quack_ok[qt]:
                        want["head"] = quack_head_pose({"head_roll": args.quack_roll, "head_yaw": args.quack_yaw,
                                                        "head_pitch": args.quack_pitch}, t - qt, brain)
        if sp is not None and sp.active(t) and guard["stop"] is None:
            want = sp.want(t)
        if stop_at is not None and t >= stop_at:
            want = stop_command(True, brain.mode)
            if args.stop_ramp is not None and t - stop_at >= args.stop_ramp:
                want = {"mode": "stand"}
        # the hold estimate: pad counts of a palm on L1 + R1 + M while gripped, foot load share from the floor contacts
        level = 1.0 if args.grip_at is not None and args.grip_at <= t and (args.release_at is None or t < args.release_at) else 0.0
        counts = [PAD_BASE + (level * PAD_HI if p in GRIP_PADS else 0.0) for p in PADS]
        # --hold-uncal: no load share, as walk_wired before hold_session.py calibrates the leg currents (grip only)
        # --load-from-legs: the share the robot would read from its leg currents (summed |torque| of the ten leg
        # actuators between the calibrated hanging and standing sums), not the true foot force
        load_in = load_now if args.load_from_legs is None else min(1.0, max(0.0, (tau_legs - args.load_from_legs[1]) / (args.load_from_legs[0] - args.load_from_legs[1])))
        est.update(0.02, counts=counts, age=0.0, load=None if args.hold_uncal else load_in)
        if guard["stop"] is not None:  # walk_wired wrote stand (monitor out of a sit or crouch) to the control file
            want = {"mode": guard["stop_to"]}
        if args.wired_hold:  # walk_wired.py's order: grip stop on the file command, then the smoothing, then the gate
            if grip_stop.update(want, args.hold and not args.hold_uncal, est.e) == "stop" and grip_stop_at is None:
                grip_stop_at = round(t, 2)
            want = grip_stop.apply(want)
            if want.get("mode") == "walk":
                for k, key in enumerate(("vx", "vy", "vtheta")):
                    cmd_s[k] += CMD_ALPHA * (float(want.get(key, 0)) - cmd_s[k])
                    cmd_s[k] = 0.0 if abs(cmd_s[k]) < 1e-4 else cmd_s[k]
                moving = any(cmd_s) or any(float(want.get(k, 0)) for k in ("vx", "vy", "vtheta"))
                want = dict(want, vx=cmd_s[0], vy=cmd_s[1], vtheta=cmd_s[2])
                wz_t = None if moving else (t if wz_t is None else wz_t)
            else:
                cmd_s, wz_t = [0.0, 0.0, 0.0], None
            on = args.hold and not args.hold_uncal and hold_allowed(want, brain, None if wz_t is None else t - wz_t,
                                                                     grip_stop.on)
            target = est.s if on else 1.0
            hold_s = max(target, hold_s - DOWN_RATE * 0.02) if target < hold_s else min(target, hold_s + UP_RATE * 0.02)
            if on or hold_s < 1.0:
                want["hold"] = {"s": hold_s, "kp": est.gains() if on else None}
            if not args.no_settle and grip_stop.settle(brain, hold_s) and settled_at is None:
                settled_at = round(t, 2)
        elif args.hold and want.get("mode") == "walk":
            want["hold"] = {"s": est.s, "kp": est.gains()}
        age = args.sensor_delay * 0.02 + 0.004
        imu_age = age + (t - imu_t)
        fresh = rng.random() >= args.drop  # a dropped sync read (8 % on 10-15) leaves the brain holding its goal
        goal, kp, info = brain.tick(want, t, pos_t, vel_t, tuple(q_sensor), g_sensor, fresh, age, imu_age)
        if args.wired_guards:  # walk_wired.py's walk-mode stops, checked on the applied action before the goals go out
            if brain.mode not in Brain.POLICY_MODES:
                guard["hard_hits"].clear()
            elif guard["stop"] is None:
                act, why = info.get("action"), None
                if act is not None:
                    amax = max(abs(a) for a in act)
                    guard["hard_hits"].append(amax > HARD_MAX_ACTION)
                    if sum(guard["hard_hits"]) >= OVER_LIMIT_TICKS:
                        why = f"action over {HARD_MAX_ACTION} rad in {sum(guard['hard_hits'])} of {len(guard['hard_hits'])} ticks"
                    elif amax > HARD_MAX_ACTION:
                        goal = walk_brain.clamp_goal({n: (brain.default[n] + max(-HARD_MAX_ACTION, min(HARD_MAX_ACTION, float(act[k])))
                                                          if n in brain.default else goal[n])
                                                      for k, n in enumerate(walk_brain.OBSERVATION_DOF_ORDER)})
                        brain.goal = dict(goal)
                        guard["clipped"] += 1
                max_tilt = {"groundpick": MAX_TILT_GP_DEG, "seepick": MAX_TILT_GP_DEG,
                            "sit": MAX_TILT_SIT_DEG}.get(brain.mode, MAX_TILT_DEG)
                if why is None and info["gb"][2] > -math.cos(math.radians(max_tilt)):
                    if args.hold and est.e > 0.3:
                        guard["tilt_skipped"] += 1
                    else:
                        why = f"body tilt over {max_tilt} deg"
                if why is not None:
                    guard["stop"], guard["why"] = round(t, 2), why
                    guard["stop_to"] = "monitor" if brain.mode in ("groundpick", "seepick", "sit") else "stand"
                    if guard["stop_to"] == "monitor":
                        brain.hold_head()  # walk_wired.py: the legs hold their goals, the head where it is
                    if guard["last_goal"] is not None:
                        goal = dict(guard["last_goal"])
                        brain.goal = dict(guard["last_goal"])
            if goal is not None:
                guard["last_goal"] = dict(goal)
        if sp is not None:
            sp.after_tick(t, brain, info)  # walk_wired: jaw goal, hand-back, jaw.write under jaw_limits
        if info.get("kick_done") is not None and kick_done_t is None:
            kick_done_t = round(t, 2)  # walk_wired.py rewrites the control file to walk at zero command here
        if brain.mode == "kick" and kick_start_t is None:
            kick_start_t = round(t, 2)
        if tick_log is not None:  # same row shape as the walk_wired log, for sim-vs-real statistics
            row = dict(t=round(t, 3), mode=brain.mode, fresh=fresh, pos_tick=pos_t, **info)
            if goal is not None:
                row["goal_tick"] = {str(i): to_tick(n, goal[n]) for i, n in zip(IDS, NAMES)}
                row["kp"] = kp
            tick_log.write(json.dumps(row) + "\n")
        if args.breakaway and goal is not None and kp:
            goal = comp.apply(goal, {n: to_model(n, tk) for n, tk in zip(NAMES, pos_t)}, kp)
        if info["action"] is not None:
            max_action = max(max_action, max(abs(a) for a in info["action"]))
            if prev_action is not None and max(abs(a - b) for a, b in zip(info["action"], prev_action)) > 0.5:
                action_jumps += 1  # tick-to-tick action jump > 0.5 rad, the real-log jitter measure (2026-10-16)
            prev_action = list(info["action"])
        if info["action"] is not None and first_policy is None:
            first_policy = round(t, 2)
        if goal is not None:
            target = np.array([ev.HOME[i] for i in range(14)])  # placeholder shape
            target = np.array([goal[n] for n in names]) if all(n in goal for n in names) else target
            if isinstance(kp, dict):  # graded hold: one P gain per joint, in the model's actuator order
                bam.actuator.kp = np.array([float(kp[n]) for n in names])
            elif kp:
                bam.actuator.kp = kp
        else:
            target = aq[-1].copy()
        aq.append(target.copy())
        c.q_target = aq[0] + z0
        # --yaw-torque: a steady turn about the vertical on the trunk while walking, a stand-in for whatever turns the
        # real robot left (10/21-22: +22 to +42 deg/s at vx 0.25, cause not isolated)
        d.xfrc_applied[trunk, 5] = args.yaw_torque if args.walk_at <= t < args.walk_at + args.walk_s else 0.0
        if args.walk_at <= t < args.walk_at + args.walk_s and "turn" in info:
            turns.append(info["turn"])
        for k_sub in range(sub):
            hand.apply(t + k_sub * m.opt.timestep)
            c.update()
            if sp is not None:
                sp.substep()
            mujoco.mj_step(m, d)
        load_now = min(1.0, max(0.0, foot_load(m, d, floor, foot_geoms) / mg))
        tau_legs = float(sum(abs(d.actuator_force[act_id[n]]) for n in leg_names))
        if scen:
            Rh = d.xmat[trunk].reshape(3, 3)
            raw = info.get("action_raw", info.get("action"))
            hrows.append(dict(t=round(t, 2), load=load_now, tau_legs=tau_legs, est=est.record(), h_load=est.h_load,
                              s_applied=want.get("hold", {}).get("s", 1.0) if want.get("mode") == "walk" else 1.0,
                              grip_stop=grip_stop.on, vx=want.get("vx"),
                              raw=None if raw is None else list(raw),
                              act=None if info.get("action") is None else list(info["action"]),
                              tau={n: float(d.actuator_force[act_id[n]]) for n in ("neck_pitch", "head_pitch")},
                              legq=[float(d.qpos[qa[names.index(n)]]) for n in leg_names],
                              legdq=[float(d.qvel[va[names.index(n)]]) for n in leg_names],
                              tilt=math.degrees(math.acos(float(np.clip(Rh[2, 2], -1, 1)))),
                              gyro=float(np.linalg.norm(d.sensordata[gyro_adr:gyro_adr + 3])),
                              hand=hand.active(t), fz=float(d.xfrc_applied[hand.head, 2]),
                              feet_z=float(min(d.geom_xpos[g, 2] for g in feet)), mode=brain.mode,
                              kp_applied=None if kp is None else (dict(kp) if isinstance(kp, dict) else int(kp))))
        for qt in quack_ok:  # largest head joint excursion from the policy default during each quack move
            if qt <= t < qt + QUACK_HEAD_S + 0.5:
                rq = quack_reach.setdefault(qt, {})
                for k, n in enumerate(names):
                    if n in ("head_roll", "head_yaw", "head_pitch"):
                        dv = float(d.qpos[qa[k]] - z0[k]) - brain.default[n]
                        rq[n] = dv if abs(dv) > abs(rq.get(n, 0.0)) else rq.get(n, 0.0)
        R = d.xmat[trunk].reshape(3, 3)
        tilt = math.degrees(math.acos(float(np.clip(R[2, 2], -1, 1))))
        maxtilt = max(maxtilt, tilt)
        tz = float(d.xpos[trunk, 2])
        hist.append((round(t, 2), tilt, float(np.linalg.norm(d.sensordata[gyro_adr:gyro_adr + 3])), tz))
        if sp is not None:
            sp.observe(t, brain, info, tilt, tz)
        if sp_trace is not None:
            sp_trace.row(t, brain, info, goal, m, d, sp, qa, z0, names, tilt, tz)
        if args.kick_at is not None:
            fz = float(d.geom_xpos[kick_geom, 2])
            if kick_start_t is None:
                kick_foot_z0 = fz
            elif kick_done_t is None or t <= kick_done_t + 0.2:
                kick_foot_max = fz if kick_foot_max is None else max(kick_foot_max, fz)
        if brain.mode == "sit":
            ph = brain.sit["phase"] if brain.sit else None
            if sit["start"] is None:
                sit["start"] = round(t, 2)
            if ph in ("down", "seated") and sit["flag1"] is None:
                sit["flag1"] = round(t, 2)
            if ph == "rise" and sit["rise"] is None:
                sit["rise"] = round(t, 2)
            sit["min_trunk_z"] = round(tz if sit["min_trunk_z"] is None else min(sit["min_trunk_z"], tz), 4)
            if ph == "seated":
                sit["seated_tilt"], sit["seated_trunk_z"] = round(tilt, 1), round(tz, 4)
                sit["seated_max_tilt"] = max(sit.get("seated_max_tilt", 0.0), round(tilt, 1))
                mz = round(float(d.site_xpos[mouth, 2]), 4)
                sit["seated_mouth_min_z"] = min(sit.get("seated_mouth_min_z", mz), mz)
                if brain.q is not None:  # head joints at the end of the seat, model rad (--sit-head targets default + delta)
                    sit["seated_head_q"] = {n: round(float(brain.q[n]), 3) for n in ("head_pitch", "head_yaw", "head_roll")}
                    sit["head_default"] = {n: round(float(brain.default[n]), 3) for n in ("head_pitch", "head_yaw", "head_roll")}
        elif sit["start"] is not None and sit["done"] is None:
            sit["done"] = round(t, 2)
        if brain.mode == "groundpick":
            # mouth tip relative to the point on the floor midway between the feet, x forward along the trunk heading
            gp["max_tilt"] = max(gp["max_tilt"], tilt)
            mz = float(d.site_xpos[mouth, 2])
            if gp["mouth_min_z"] is None or mz < gp["mouth_min_z"]:
                mid = np.mean([d.geom_xpos[g] for g in feet], axis=0)
                rel = d.site_xpos[mouth] - mid
                fwd = R[:, 0].copy()
                fwd[2] = 0
                fwd /= np.linalg.norm(fwd)
                left = np.array([-fwd[1], fwd[0], 0.0])
                gp.update(mouth_min_z=round(mz, 4), mouth_at_min=[round(float(rel @ fwd), 4), round(float(rel @ left), 4)],
                          pose_at_min={n: round(float(d.qpos[qa[k]] - z0[k]), 3) for k, n in enumerate(names)},
                          tilt_at_min=round(tilt, 1), phase_at_min=info.get("gp_phase"),
                          trunk_z_at_min=round(float(d.xpos[trunk, 2]), 4))
        if stop_at is not None and t >= stop_at:
            tilt_after_stop = max(tilt_after_stop, tilt)
            holding_stand = brain.ramp is None and brain.goal is not None and \
                all(abs(brain.goal[n] - BRAIN_STAND[n]) < 1e-6 for n in NAMES)
            if settled_at is None and stop_settled(True, brain.mode, holding_stand, t - stop_at):
                settled_at = round(t, 2)
        if t >= args.walk_at and x0 is None:
            x0 = float(d.xpos[trunk, 0])
            yaw0 = yaw_prev = _yaw(d.xquat[trunk])
        if x0 is not None and t < args.walk_at + args.walk_s:  # heading change over the walk window, unwrapped
            yaw_now = _yaw(d.xquat[trunk])
            yaw_walk += (yaw_now - yaw_prev + math.pi) % (2 * math.pi) - math.pi
            yaw_prev = yaw_now
            if tick % 25 == 0:
                xy = (float(d.xpos[trunk, 0]), float(d.xpos[trunk, 1]))
                if path_xy is not None:
                    path_m += math.hypot(xy[0] - path_xy[0], xy[1] - path_xy[1])
                path_xy = xy
        if tick % 25 == 0:
            rows.append((round(t, 1), brain.mode, info["gb"][0], round(float(d.xpos[trunk, 0]), 3), round(tilt, 1)))
        low = brain.mode in ("groundpick", "seepick", "sit")
        carried = hand.active(t)  # the hand holds it up: a tilt is the hang, not a fall
        if (tilt > (80 if low else 45) and not carried) or d.xpos[trunk, 2] < (0.03 if brain.mode in ("sit", "seepick") else 0.04):
            fell, fall_dir = round(t, 2), _lean_dir(R)
            break
    if gp_at is not None:
        gp["end_tilt"] = round(tilt, 1)

    def window(a, b):
        return [h for h in hist if a <= h[0] < b]

    def stable_after(t_ref, base_t):
        """Seconds from t_ref to the start of the first 1 s window that stays within 2 deg of the stance tilt before
        base_t with |gyro| < 0.3 rad/s; None if the run ends first."""
        pre = window(base_t - 1.0, base_t)
        if t_ref is None or not pre:
            return None, None
        base = float(np.median([h[1] for h in pre]))
        after = [h for h in hist if h[0] >= t_ref]
        for i, h in enumerate(after):
            w = [x for x in after[i:] if x[0] < h[0] + 1.0]
            if after[-1][0] < h[0] + 1.0:
                break
            if all(x[1] <= base + 2.0 and x[2] < 0.3 for x in w):
                return round(h[0] - t_ref, 2), round(base, 1)
        return None, round(base, 1)

    if gp_at is not None:
        gp["stable_after_s"], gp["stance_tilt"] = stable_after(gp_done_at, gp_at)
    kick = None
    if args.kick_at is not None:
        stable_s, base = stable_after(kick_done_t, args.kick_at)
        seg = window(args.kick_at, (kick_done_t or args.kick_at) + 3.0)
        kick = dict(at=args.kick_at, foot=args.kick_foot, started=kick_start_t, done=kick_done_t,
                    peak_tilt=round(max(h[1] for h in seg), 1) if seg else None, stance_tilt=base,
                    foot_lift_m=None if kick_foot_max is None else round(kick_foot_max - kick_foot_z0, 4),
                    stable_after_s=stable_s)
    sitr = None
    if args.sit_at is not None:
        stable_s, base = stable_after(sit["done"], args.sit_at)
        seg = window(args.sit_at, (sit["done"] or T) + 3.0)
        sitr = dict(sit, at=args.sit_at, unsit_at=args.unsit_at, stance_tilt=base,
                    peak_tilt=round(max(h[1] for h in seg), 1) if seg else None, stable_after_s=stable_s,
                    end_trunk_z=round(hist[-1][3], 4) if hist else None)
    if sp is not None and args.sp_log:
        with open(args.sp_log, "w") as f:
            for r in sp.rows:
                f.write(json.dumps(r) + "\n")
    holdr = hold_summary(args, hrows, hand, guard, stable_after, leg_names) if scen else None
    if holdr is not None and args.wired_hold:
        g0 = args.grip_at
        half = next((r["t"] for r in hrows if g0 is not None and r["t"] >= g0 and r["s_applied"] < 0.5), None)
        holdr.update(wired_hold=True, grip_stop_at=grip_stop_at, grip_stop_end=grip_stop.on, stand61_from=settled_at,
                     applied_s_below_0p5_after_grip_s=None if half is None else round(half - g0, 2),
                     min_applied_s=round(min(r["s_applied"] for r in hrows), 3))
    return dict(model=args.model, volt=args.volt, walk_ramp=args.walk_ramp, com_x_mm=args.com_x, zero=zero, imu_pitch=args.imu_pitch,
                kick=kick, sit=sitr, hold=holdr, quack_head_ran={str(k): v for k, v in quack_ok.items()},
                quack_head_reached={str(k): {n: round(v, 3) for n, v in r.items()} for k, r in quack_reach.items()},
                quack_peak_tilt={str(k): round(max(h[1] for h in window(k, k + QUACK_HEAD_S + 1.0)), 1) for k in quack_ok},
                seepick=None if sp is None else dict(sp.summary(brain), guard=dict(
                    wired=bool(args.wired_guards), stop_at=guard["stop"], why=guard["why"], clipped_ticks=guard["clipped"],
                    hard_max_action=HARD_MAX_ACTION, max_tilt_deg=MAX_TILT_GP_DEG)),
                groundpick=None if gp_at is None else dict(gp, at=gp_at, phase=args.gp_phase, hold_s=args.gp_hold_s,
                                                           done_at=round(gp_done_at, 2)),
                head=None if not (args.head_yaw or args.head_pitch is not None) else dict(head_yaw=args.head_yaw, head_pitch=args.head_pitch,
                                                         final={n: round(float(d.qpos[qa[k]] - z0[k]), 3) for k, n in enumerate(names) if "head" in n or "neck" in n}),
                compensation=args.compensation, first_policy_tick_s=first_policy, max_action=round(max_action, 2), action_jumps=action_jumps,
                drop=args.drop, imu_hz=args.imu_hz, floor_mu=args.floor_mu, fell_at=fell, fall_dir=fall_dir,
                max_tilt=round(maxtilt, 1), battery_stop_at=stop_at, stop_ramp=args.stop_ramp, settled_s=settled_at, end_mode=brain.mode,
                max_tilt_after_stop=round(tilt_after_stop, 1) if stop_at is not None else None,
                dist_x=None if x0 is None else round(float(d.xpos[trunk, 0] - x0), 3),
                yaw_deg=None if x0 is None else round(math.degrees((_yaw(d.xquat[trunk]) - yaw0 + math.pi) % (2 * math.pi) - math.pi), 1),
                yaw_walk_deg=None if x0 is None else round(math.degrees(yaw_walk), 1),
                path_m=None if x0 is None else round(path_m, 3),
                turn_mean=round(float(np.mean(turns)), 3) if turns else None, yaw_trim=round(brain.yaw_trim, 3),
                trace=rows)


def hold_summary(args, rows, hand, guard, stable_after, leg_names):
    """Numbers for the lifted-by-the-head scenario (see the module docstring)."""
    r3 = lambda x: None if x is None else round(float(x), 3)  # noqa: E731
    t_ref = min(x for x in (args.grip_at, args.lift_at) if x is not None)
    ref = [r for r in rows if t_ref - 1.0 <= r["t"] < t_ref]
    if args.lift_at is not None:
        win = (args.lift_at, (args.lower_at if args.lower_at is not None else rows[-1]["t"]) + 0.8)
    else:
        win = (args.grip_at, args.release_at if args.release_at is not None else rows[-1]["t"])
    lw = [r for r in rows if win[0] <= r["t"] < win[1]]
    air = [r for r in rows if r["hand"] and r["load"] < 0.05]
    hand_off = args.release_at if args.grip_at is not None and args.release_at is not None else hand.off_at
    amax = lambda key, rs: max((max(abs(a) for a in r[key]) for r in rs if r[key] is not None), default=None)  # noqa: E731

    def jump(key, rs):
        best, prev = None, None
        for r in rs:
            if r[key] is not None and prev is not None:
                j = max(abs(a - b) for a, b in zip(r[key], prev))
                best = j if best is None or j > best else best
            prev = r[key] if r[key] is not None else prev
        return best

    def torque(rs, n):
        v = [abs(r["tau"][n]) for r in rs]
        return None if not v else dict(peak=r3(max(v)), mean=r3(np.mean(v)))

    out = dict(hold_set={k: v for k, v in args.hold_set}, hold_uncal=bool(args.hold_uncal), hand_krot=args.hand_krot, grip_at=args.grip_at, release_at=args.release_at, lift_at=args.lift_at, lower_at=args.lower_at,
               lift_height=args.lift_height, hold_on=bool(args.hold), hand_off_at=hand_off, lift_window=list(win),
               wired_guards=bool(args.wired_guards), guard_stop_at=guard["stop"], guard_why=guard["why"],
               guard_clipped_ticks=guard["clipped"], guard_tilt_skipped_ticks=guard["tilt_skipped"],
               max_raw_action_lift=r3(amax("raw", lw)), max_applied_action_lift=r3(amax("act", lw)),
               min_s=r3(min(r["est"]["s"] for r in rows)), max_e=r3(max(r["est"]["e"] for r in rows)),
               min_load=r3(min(r["load"] for r in rows)), min_kp_leg=min(r["est"]["kp_leg"] for r in rows),
               max_kp_neck=max(r["est"]["kp_neck"] for r in rows),
               torque_standing={n: torque(ref, n) for n in ("neck_pitch", "head_pitch")},
               legs_tau_standing=r3(np.mean([r["tau_legs"] for r in ref])) if ref else None)
    if air:
        a0, a1 = air[0]["t"], air[-1]["t"]
        q = np.array([r["legq"] for r in air])
        dq = np.array([r["legdq"] for r in air])
        rng_ = q.max(0) - q.min(0)
        # the struggle once the legs had 0.5 s to settle into the hang
        late = np.array([r["legq"] for r in air if r["t"] >= a0 + 0.5]) if a1 - a0 > 0.6 else None
        out.update(air_from=a0, air_to=a1, air_s=round(a1 - a0 + 0.02, 2),
                   feet_clearance_m=r3(max(r["feet_z"] for r in air) - np.mean([r["feet_z"] for r in ref])),
                   torque_hang={n: torque(air, n) for n in ("neck_pitch", "head_pitch")},
                   max_tilt_carried=round(max(r["tilt"] for r in rows if r["hand"]), 1),
                   leg_range_air_rad=r3(rng_.max()), leg_range_air_joint=leg_names[int(rng_.argmax())],
                   leg_range_air_after_0p5s_rad=None if late is None or not len(late) else r3((late.max(0) - late.min(0)).max()),
                   leg_rms_speed_air=r3(np.sqrt(np.mean(dq ** 2))),
                   max_raw_action_air=r3(amax("raw", air)), max_applied_action_air=r3(amax("act", air)),
                   action_ticks_air=sum(r["act"] is not None for r in air),
                   legs_tau_air=r3(np.mean([r["tau_legs"] for r in air])), legs_tau_air_max=r3(max(r["tau_legs"] for r in air)),
                   est_load_air_mean=r3(np.mean([r["est"]["load"] for r in air if r["est"]["load"] is not None] or [np.nan])),
                   first_air_est_load_below_0p5_s=next((round(r["t"] - a0, 2) for r in air if (r["est"]["load"] if r["est"]["load"] is not None else 1) < 0.5), None))
    if hand_off is not None:
        post = [r for r in rows if hand_off <= r["t"] < hand_off + 0.3]
        prev = [r for r in rows if r["t"] < hand_off][-1:]
        s_back = next((r["t"] for r in rows if r["t"] >= hand_off and r["est"]["s"] >= 0.999), None)
        stable_s, base = stable_after(hand_off, t_ref)
        after = [r for r in rows if r["t"] >= hand_off]
        out.update(stable_after_release_s=stable_s, stance_tilt=base,
                   max_action_jump_0p3s_after_release=r3(jump("act", prev + post)),
                   max_raw_action_jump_0p3s_after_release=r3(jump("raw", prev + post)),
                   max_applied_action_after_release=r3(amax("act", after)),
                   s_back_to_1_after_release_s=None if s_back is None else round(s_back - hand_off, 2),
                   max_tilt_after_release=round(max(r["tilt"] for r in after), 1) if after else None)
    out["trace"] = [dict(t=r["t"], load=r3(r["load"]), est_load=r["est"]["load"], tau_legs=r3(r["tau_legs"]), s=r["est"]["s"], e=r["est"]["e"], h_load=r3(r["h_load"]),
                         kp_leg=r["est"]["kp_leg"], kp_neck=r["est"]["kp_neck"], tilt=round(r["tilt"], 1), mode=r["mode"],
                         raw=r3(None if r["raw"] is None else max(abs(a) for a in r["raw"])),
                         act=r3(None if r["act"] is None else max(abs(a) for a in r["act"])),
                         tau_neck=r3(r["tau"]["neck_pitch"]), tau_head=r3(r["tau"]["head_pitch"]), fz=r3(r["fz"]),
                         feet_z=round(r["feet_z"], 4))
                    for r in rows[::5]]
    return out


tick_log = None


def head_keys(keys, t):
    """Head deltas at time t from [{"t", "head": {joint: delta}}] keyframes, linear between them, held after the last."""
    names = {n for k in keys for n in k["head"]}
    out = {}
    for n in names:
        pts = [(k["t"], k["head"].get(n, 0.0)) for k in keys]
        if t <= pts[0][0]:
            out[n] = pts[0][1]
        elif t >= pts[-1][0]:
            out[n] = pts[-1][1]
        else:
            for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
                if t0 <= t <= t1:
                    out[n] = v0 + (v1 - v0) * (t - t0) / max(1e-9, t1 - t0)
                    break
    return out


def _yaw(q):
    """Heading (rad) of a MuJoCo wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _lean_dir(R):
    """Way the trunk's up axis leans in its own heading frame (x forward): forward, backward, left or right."""
    fwd = R[:, 0].copy()
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    f, l = float(R[:, 2] @ fwd), float(R[:, 2] @ np.array([-fwd[1], fwd[0], 0.0]))
    if abs(f) >= abs(l):
        return "forward" if f > 0 else "backward"
    return "left" if l > 0 else "right"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volt", type=float, default=None,
                    help="servo supply; default: servo_rail_v in microduck-sim/CURRENT-STATE.json (the as-run rail)")
    ap.add_argument("--model", choices=("v19", "asbuilt"), default="v19",
                    help="mass model: v19 (default) or asbuilt (2026-10-15, D-007 bench backpack)")
    ap.add_argument("--manifest", default=None,
                    help="signed-part mass manifest to use instead of the model's own (same schema; e.g. a plugged/battery variant)")
    ap.add_argument("--walk-ramp", type=float, default=0.0, help="Brain walk_ramp; 1.5 reproduces the old behaviour")
    ap.add_argument("--com-x", type=float, default=0.0, help="whole-robot COM shift forward, mm")
    ap.add_argument("--zero", nargs=2, action="append", default=[], metavar=("JOINT", "DEG"))
    ap.add_argument("--imu-pitch", type=float, default=0.0)
    ap.add_argument("--compensation", action="store_true")
    ap.add_argument("--stand-s", type=float, default=1.0)
    ap.add_argument("--walk-at", type=float, default=4.0)
    ap.add_argument("--walk-s", type=float, default=10.0)
    ap.add_argument("--vx", type=float, default=0.3)
    ap.add_argument("--vtheta", type=float, default=0.0, help="rad/s yaw rate command while walking")
    ap.add_argument("--vtheta-seq", nargs=2, type=float, action="append", default=[], metavar=("S", "VTHETA"),
                    help="from S seconds into the walk, command this vtheta (repeatable; e.g. a right arc then a left arc)")
    ap.add_argument("--yaw-torque", type=float, default=0.0, help="N m about world up on the trunk while walking (+ = left)")
    ap.add_argument("--yaw-trim-file", default=None, help="walk_brain yaw trim file, read at start and updated after the walk")
    ap.add_argument("--sensor-delay", type=int, default=1)
    ap.add_argument("--action-delay", type=int, default=1)
    ap.add_argument("--battery-stop-at", type=float, default=None, help="s; apply the battery stop sequence from here")
    ap.add_argument("--after-stop", type=float, default=10.0, help="s simulated after the battery stop")
    ap.add_argument("--stop-ramp", type=float, default=None, help="s after the stop to switch to stand (rejected design)")
    ap.add_argument("--drop", type=float, default=0.0, help="fraction of ticks with a dropped servo read")
    ap.add_argument("--imu-hz", type=float, default=0.0, help="IMU sample rate; 0 = every 20 ms tick")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--breakaway", action="store_true", help="apply walk_brain.breakaway to the goals (no stiction in the sim: checks the push alone does not upset the policy)")
    ap.add_argument("--ba-first", type=float, default=31.0)
    ap.add_argument("--ba-ramp", type=float, default=15.0)
    ap.add_argument("--ba-cap", type=float, default=120.0)
    ap.add_argument("--ba-min-err", type=float, default=6.0)
    ap.add_argument("--ba-stuck", type=float, default=0.5)
    ap.add_argument("--ba-cap-ticks", type=float, default=115.0)
    ap.add_argument("--kp-rl", type=int, default=None, help="firmware P gain the brain uses in walk mode (default walk_brain.KP_RL 200)")
    ap.add_argument("--models", default=MODELS, help="directory holding walk61.onnx and stand61.onnx (default runtime-overlay/models)")
    ap.add_argument("--gp-at", type=float, default=None, help="s; switch stand61 to groundpick (Pollen alpha_ground_pick) here")
    ap.add_argument("--gp-phase", type=float, default=0.4, help="phase to descend to and hold (0.375..0.425 is the lowest)")
    ap.add_argument("--gp-hold-s", type=float, default=2.0, help="s to hold the low pose before the rise")
    ap.add_argument("--dance", default=None, help="dance.json: head and in-place turn steps from --walk-at on (shoot.py sing)")
    ap.add_argument("--head-yaw", type=float, default=0.0, help="rad; head pose command in walk mode from --walk-at on")
    ap.add_argument("--head-pitch", type=float, default=None, help="rad; absolute head_pitch target with --head-yaw")
    ap.add_argument("--quack-at", type=float, nargs="*", default=[],
                    help="s; walk_wired.py quack head tilts (QUACK_HEAD_S raised-cosine profile) starting at these times")
    ap.add_argument("--quack-roll", type=float, default=0.18, help="rad; head_roll delta of the --quack-at tilts")
    ap.add_argument("--quack-yaw", type=float, default=0.25, help="rad; head_yaw delta of the --quack-at tilts")
    ap.add_argument("--quack-pitch", type=float, default=0.0, help="rad; head_pitch delta of the --quack-at moves")
    ap.add_argument("--kick-at", type=float, default=None, help="s; one kick command (id 1) at this time")
    ap.add_argument("--kick-foot", choices=("left", "right"), default="left")
    ap.add_argument("--sit-at", type=float, default=None, help="s; {\"mode\": \"sit\"} from here")
    ap.add_argument("--sit-head", type=json.loads, default=None,
                    help=f'JSON head deltas from the default, e.g. \'{{"head_pitch": 1.1}}\', sent with the sit command from --sit-at + {SIT_HEAD_DELAY_S} s (seated)')
    ap.add_argument("--sit-head-keys", nargs=2, default=None, metavar=("FILE", "KEY"),
                    help="seated head keyframes (sitlook.json sim_aim) from the seat on, as shoot.py sitfind sends them")
    ap.add_argument("--restart-at", type=float, default=None,
                    help="s; a fresh Brain in monitor mode from here (the brain restarted on a seated robot)")
    ap.add_argument("--resit-at", type=float, default=None, help="s; {\"mode\": \"sit\"} again from here, after --restart-at")
    ap.add_argument("--resit-s", type=float, default=30.0, help="s simulated after --resit-at")
    ap.add_argument("--unsit-at", type=float, default=None, help="s; walk at zero command from here (stands up first)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--bam-friction-scale", type=float, default=1.0, help="multiply BAM Coulomb/Stribeck/load friction (training DR range 0.9-1.1; measured real servos sit above it)")
    ap.add_argument("--floor-mu", type=float, default=1.0,
                    help="foot-floor sliding friction (tilt test: tan of the slide angle); 1.0 = the model's own, training randomised 0.7-1.3")
    ap.add_argument("--foot-play", type=float, nargs="+", default=None, metavar="DEG [STIFF DAMP]",
                    help="free play about the ankle axis between servo horn and foot, +-DEG (sim-ID 2026-10-21)")
    ap.add_argument("--foot-play-axis", choices=sorted(FOOT_PLAY_AXES), default="pitch",
                    help="axis of --foot-play: pitch (the ankle axis), roll (forward) or yaw (vertical), at the stand pose")
    ap.add_argument("--sole-solref", type=float, nargs=2, default=None, metavar=("TIMECONST", "DAMPRATIO"),
                    help="foot-floor contact softness (MuJoCo solref; the model's own is 0.02 1): a compliant sole")
    ap.add_argument("--grip-at", type=float, default=None, help="s; head pads L1, R1, M covered from here (hold.py estimator input)")
    ap.add_argument("--release-at", type=float, default=None, help="s; the hand leaves the pads (and lets go of the head)")
    ap.add_argument("--lift-at", type=float, default=None, help="s; the hand lifts the head --lift-height over 0.6 s from here")
    ap.add_argument("--lower-at", type=float, default=None, help="s; the hand lowers the head back to its start height over 0.8 s")
    ap.add_argument("--lift-height", type=float, default=0.04, help="m; rise of the hand's target height")
    ap.add_argument("--hand-k", type=float, default=2000.0, help="N/m; vertical stiffness of the hand (0.84 kg sags ~4 mm)")
    ap.add_argument("--hand-c", type=float, default=60.0, help="N s/m; vertical damping of the hand")
    ap.add_argument("--hand-kxy", type=float, default=500.0, help="N/m; horizontal stiffness holding the head where it was lifted")
    ap.add_argument("--hand-cxy", type=float, default=20.0, help="N s/m; horizontal damping")
    ap.add_argument("--hand-crot", type=float, default=0.02, help="N m s/rad; angular damping of the palm on the head")
    ap.add_argument("--hand-krot", type=float, default=0.0, help="N m/rad; grip holding the head's orientation (0 = head pivots in the hand)")
    ap.add_argument("--hold", action="store_true", help="pass hold.HoldEstimator's s and gains to the Brain (else shadow only)")
    ap.add_argument("--load-from-legs", type=float, nargs=2, default=None, metavar=("STAND", "AIR"),
                    help="feed the estimator the load share from the summed |leg actuator torque| (N m) between these two sums, as walk_wired reads it from the leg currents")
    ap.add_argument("--hold-uncal", action="store_true", help="feed the estimator no load share (uncalibrated legs: the grip term alone)")
    ap.add_argument("--hold-set", nargs=2, action="append", default=[], metavar=("NAME", "VALUE"),
                    help="override a hold.py module constant for this run (sensitivity study), e.g. --hold-set UP_RATE 2")
    ap.add_argument("--wired-hold", action="store_true",
                    help="gate the hold as walk_wired.py does: command smoothing, grip stop, hold_allowed, authority ramp")
    ap.add_argument("--no-settle", action="store_true", help="with --wired-hold: keep walk61 after a grip stop (GripStop.settle off)")
    ap.add_argument("--wired-guards", action="store_true", help="walk_wired.py's walk-mode stops (3.0 rad hard limit, 30 deg tilt)")
    ap.add_argument("--until", type=float, default=0.0, help="s; run at least this long")
    ap.add_argument("--bvh-refit", action="store_true", help="accepted for older command lines, no effect: evaluate_delay.load refits the collision bounding volumes itself since 2026-10-29 (prints the empty change set)")
    ap.add_argument("--tick-log", default=None, help="write one JSON row per 20 ms tick (q, dq, gb, gyro_body, action) like /tmp/duck_wired_log_*.jsonl")
    sp = ap.add_argument_group("seeing pick (walk_brain mode seepick, seepick61.onnx)")
    sp.add_argument("--sp-at", type=float, default=None, help="s; switch stand61 to seepick here (forces vx 0)")
    sp.add_argument("--sp-offset", type=float, nargs=2, default=[0.0, 0.0], metavar=("DX", "DY"),
                    help="m; object offset from the trained landing spot SP_LANDING_XY in the foot frame")
    sp.add_argument("--sp-shape", choices=("ball", "box", "cylinder"), default="ball")
    sp.add_argument("--sp-half", type=float, nargs=3, default=None, metavar=("HX", "HY", "HZ"), help="m; exact half extents (overrides the draw)")
    sp.add_argument("--sp-size", type=float, default=None, help="m; largest horizontal width (default: drawn from the training range)")
    sp.add_argument("--sp-mass", type=float, default=None, help="g (default: drawn from 0.5-10 g)")
    sp.add_argument("--sp-friction", type=float, default=None, help="object sliding friction (default: drawn from 0.35-1.1)")
    sp.add_argument("--sp-object-yaw", type=float, default=None, help="deg (default: drawn)")
    sp.add_argument("--sp-beak-mu", type=float, default=0.01,
                    help="beak box friction; MuJoCo takes the larger of the pair, so the default 0.01 makes the beak-object "
                         "friction the object's own (the training env's 1.5 overrode every object friction)")
    sp.add_argument("--sp-camera-delay", type=float, default=None, help="s capture to policy (default: drawn from 0.1-0.3)")
    sp.add_argument("--sp-camera-drop", type=float, default=None, help="per-frame detection drop probability (default: drawn from 0-0.3)")
    sp.add_argument("--sp-camera-bias", type=float, nargs=2, default=None, metavar=("BX", "BY"), help="m (default: drawn within +-1 cm)")
    sp.add_argument("--sp-camera-noise", type=float, default=SP_CAM_NOISE, help="m per frame (training 0.004)")
    sp.add_argument("--sp-no-stalls", action="store_true", help="no camera stalls (training: 2 %% of frames start a 0.5-1.5 s stall)")
    sp.add_argument("--sp-jaw-kp", type=int, default=jaw_ctl.KP_JAW, help="ID15 position P gain (walk_wired --jaw-kp)")
    sp.add_argument("--sp-jaw-current", type=float, default=float(jaw_ctl.GOAL_CURRENT_MA), help="mA; ID15 Goal Current in mode 5")
    sp.add_argument("--sp-holding", choices=("runtime", "stopped"), default="stopped",
                    help="holding bit: jaw_ctl.holding (stopped: open >= GRIP_MIN_DEG and still), or the rule before 2026-10-27 without the still check (runtime)")
    sp.add_argument("--sp-no-cap", action="store_true", help="do not apply jaw_limits to the jaw goal (it is still evaluated)")
    sp.add_argument("--sp-beak-priority", action="store_true",
                    help="beak boxes at MuJoCo priority 1: --sp-beak-mu is the beak-object friction whatever the object's "
                         "(a lined lip: inner tube about 1.1, bare PLA 0.35-0.5)")
    sp.add_argument("--sp-close-on-contact", action="store_true",
                    help="start the trained close (0.34->0.40) at the first beak-object touch when it comes earlier "
                         "(study of the close timing; the deployed timeline is walk_brain.SP_JAW)")
    sp.add_argument("--sp-trace", default=None, help="write SpTrace rows (61 obs, 14 actions, goals, joints, beak) per seepick tick")
    sp.add_argument("--sp-cone", choices=("pyramidal", "elliptic"), default="pyramidal",
                    help="MuJoCo friction cone of the whole model (the as-built MJCF compiles pyramidal; the training env runs elliptic)")
    sp.add_argument("--sp-impratio", type=float, default=None,
                    help="MuJoCo impratio (the as-built MJCF compiles 1; the training env runs 10, its sticky friction: "
                         "microduck_ground_grasp_env_cfg, a pinched block creeps out of the wedge at pyramidal / 1)")
    sp.add_argument("--sp-after", type=float, default=3.0, help="s simulated after the hand-back to stand61")
    sp.add_argument("--sp-log", default=None, help="JSONL: per tick jaw goal/written/measured, cap, head pose, body slots")
    args = ap.parse_args()
    global tick_log
    tick_log = open(args.tick_log, "w") if args.tick_log else None
    if args.volt is None:
        _state = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "CURRENT-STATE.json")
        args.volt = float(json.load(open(_state))["facts"]["servo_rail_v"]["value"])
        print(f"--volt {args.volt} from CURRENT-STATE.json servo_rail_v", file=sys.stderr)
    if args.kp_rl:
        import walk_brain as _wb
        _wb.KP_RL = args.kp_rl
    import hold as _hold
    for name, val in args.hold_set:
        assert hasattr(_hold, name) and name.isupper(), name
        setattr(_hold, name, type(getattr(_hold, name))(float(val)))
    out = run(args)
    out["manifest"] = args.manifest  # as given on the command line (None = the model's own manifest)
    if not args.trace:
        out.pop("trace")
    print(json.dumps(out))


if __name__ == "__main__":
    import sim_host_guard
    sim_host_guard.one_sim_on_laptop("brain_sim_check")
    main()
