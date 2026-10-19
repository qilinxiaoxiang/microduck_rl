"""Headless MuJoCo check of the community walk policy under this build's control rate and latency.

Runs the community runtime unchanged (Scheduler, Observer, WalkMove, MuJoCoController with the BAM XL330 model)
on a simulated clock, so a 20 s walk takes a few seconds. Reports whether the robot stays up and how far it goes.

  ~/.venvs/microduck-walk/bin/python sim_walk_check.py --hz 44 --delay-act 6 --vx 0.5
"""
import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_XML = os.path.join(HERE, "..", "..", "..", "microduck_rl", "src", "mjlab_microduck", "robot", "microduck", "scene_walk.xml")
# MICRODUCK_ROOT: a community runtime tree, e.g. a copy with the 61-input runtime overlay installed
ROOT = os.environ.get("MICRODUCK_ROOT") or os.path.join(HERE, "..", "..", "..", "Microduck-build-tutorial", "microduck")
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402


class _Viewer:
    opt = mujoco.MjvOption()

    def is_running(self):
        return True

    def sync(self):
        pass


mujoco.viewer.launch_passive = lambda *a, **k: _Viewer()

# simulated clock shared by the scheduler and observer
CLOCK = {"t": 0.0}
fake_time = types.SimpleNamespace(perf_counter=lambda: CLOCK["t"], monotonic=lambda: CLOCK["t"],
                                  time=lambda: CLOCK["t"], sleep=lambda s: CLOCK.update(t=CLOCK["t"] + max(s, 0)))

sys.modules.setdefault("placo", types.ModuleType("placo"))  # only the squat move uses it
import scheduler as sched_mod  # noqa: E402
import observer as obs_mod  # noqa: E402
from input.input_source import UserInput  # noqa: E402
from moves.walk import WalkMove  # noqa: E402
from sim.mujoco_controller import MuJoCoController  # noqa: E402

sched_mod.time = fake_time
obs_mod.time = fake_time


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


class Script:
    """Stand for --settle s, then walk with the given command for --walk s, then stop."""

    def __init__(self, args, ctrl, stop_path):
        self.a, self.ctrl, self.stop_path, self.n = args, ctrl, stop_path, 0
        self.track = []

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        t = CLOCK["t"]
        d = self.ctrl._data
        w, x, y, z = d.qpos[3:7]
        up_z = 1 - 2 * (x * x + y * y)  # world z of the trunk's z axis; 1 = upright
        self.track.append((t, float(d.qpos[0]), float(d.qpos[1]), float(up_z)))
        if t > self.a.settle + self.a.walk:
            open(self.stop_path, "w").write("stop\n")
        moves = {"walk"} if t > self.a.settle else set()
        return UserInput(active_moves=moves, velocity={"vx": self.a.vx, "vy": self.a.vy, "vtheta": self.a.vth})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=50.0)
    ap.add_argument("--delay-act", type=int, default=2, help="sim steps of 5 ms (community default 2)")
    ap.add_argument("--delay-pos", type=int, default=0)
    ap.add_argument("--delay-vel", type=int, default=1)
    ap.add_argument("--delay-gyro", type=int, default=3)
    ap.add_argument("--delay-quat", type=int, default=4)
    ap.add_argument("--vx", type=float, default=0.5, help="normalised [-1, 1] like the gamepad")
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--vth", type=float, default=0.0)
    ap.add_argument("--mjcf", default=TRAIN_XML,
                    help="training model of walk.onnx (the tutorial robot.xml is a different duck)")
    ap.add_argument("--com", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="trunk CoM offset, m")
    ap.add_argument("--vin", type=float, default=None, help="servo supply, V (community 7.5; this build 5.0)")
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--walk", type=float, default=15.0)
    args = ap.parse_args()

    stop_path = "/tmp/sim_walk_check.stop"
    ctrl = MuJoCoController(mjcf_path=os.path.abspath(args.mjcf), stop_flag_path=stop_path,
                            delay_act_steps=args.delay_act, delay_pos_ticks=args.delay_pos,
                            delay_vel_ticks=args.delay_vel, delay_gyro_ticks=args.delay_gyro,
                            delay_quat_ticks=args.delay_quat, trunk_com_offset=tuple(args.com))
    # training: 5 ms physics steps, 4 per 50 Hz tick; at another rate a tick still spans 4 steps of 1/(4 hz)
    ctrl._steps_per_tick = 4
    # the training XML declares position servos (kp 0.55); BAM writes motor torque into ctrl, as mjlab's BAM
    # actuator does, so make every actuator a plain torque motor
    m = ctrl._model
    m.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    m.actuator_gainprm[:, :] = 0
    m.actuator_gainprm[:, 0] = 1
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    m.actuator_biasprm[:, :] = 0
    m.actuator_ctrllimited[:] = 0
    # the runtime drops the tutorial duck from 0.165 m; the training duck stands at 0.12 m (STAND keyframe)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    mujoco.mj_resetDataKeyframe(m, ctrl._data, key)
    mujoco.mj_forward(m, ctrl._data)
    ctrl._bam.reset(ctrl._data.qpos)
    if args.vin is not None:
        ctrl._bam.model.actuator.vin = args.vin
        ctrl._bam.vin_min = min(ctrl._bam.vin_min or args.vin, args.vin - 1.0)
    ctrl._model.opt.timestep = (1.0 / args.hz) / 4
    # The training XML's imu site is aligned with the trunk; training rotates body readings by BNO080_MOUNT_QUAT
    # (mounted_body_projected_gravity / mounted_body_ang_vel). Report what that physical BNO would read:
    # q_sensor = q_body * M^-1, gyro_sensor = M gyro_body.
    from constants import IMU_MOUNT_QUAT as M
    from imu_reader import quat_apply_inverse
    Minv = (M[0], -M[1], -M[2], -M[3])
    raw_quat, raw_gyro = ctrl.read_quat, ctrl.read_gyro
    ctrl.read_quat = lambda dt: tuple(_qmul(raw_quat(dt), Minv))
    ctrl.read_gyro = lambda: tuple(float(v) for v in quat_apply_inverse(Minv, raw_gyro()))
    script = Script(args, ctrl, stop_path)
    walk = WalkMove(controller=ctrl)
    actions = []
    orig_step = walk.step

    def step(obs, cmd):
        orig_step(obs, cmd)
        la = getattr(walk, "_last_action", None)
        if la is not None and len(la):
            actions.append(max(abs(float(v)) for v in la))

    walk.step = step
    s = sched_mod.Scheduler(frequency_hz=args.hz, controller=ctrl, stop_flag_path=stop_path, input_source=script,
                            moves={"walk": walk})
    s._cleanup = lambda: None
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        s.run()

    tr = script.track
    if os.environ.get("DBG"):
        [print(round(t, 2), round(x, 3), round(z, 3)) for t, x, y, z in tr[::10]]
    fell = next((t for t, x, y, z in tr if z < 0.5), None)  # trunk tilted past 60 deg
    t_end, x_end, y_end, _ = tr[-1]
    t0 = next(t for t, *_ in tr if t > args.settle)
    x0 = next(x for t, x, *_ in tr if t > args.settle)
    out = {"hz": args.hz, "delay_act_ms": args.delay_act * 1000 * ctrl._model.opt.timestep, "delay_gyro": args.delay_gyro,
           "delay_quat": args.delay_quat, "vin": ctrl._bam.model.actuator.vin, "vx_cmd": args.vx, "com": args.com, "fell_at_s": None if fell is None else round(fell - t0, 2),
           "dist_x_m": round(x_end - x0, 3), "drift_y_m": round(y_end, 3), "walk_s": round(t_end - t0, 1),
           "max_action_rad": round(max(actions), 3) if actions else None,
           "p95_action_rad": round(sorted(actions)[int(0.95 * len(actions))], 3) if actions else None}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
