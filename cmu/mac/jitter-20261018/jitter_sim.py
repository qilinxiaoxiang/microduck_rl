"""brain_sim_check.py with the 2026-10-18 jitter hypotheses as switches (sim only; robot code untouched).

Schedule: stand --stand-s, stand61 until --walk-at, walk --vx for --cmd-s, then zero command (walk61 at rest, as on
the robot after a walk) for --tail s. Everything else (as-built model, Brain, sensor path, drops, IMU rate) is
brain_sim_check.run unchanged; real-condition flags come from robot_state_check.py --print-sim-args.

Hypothesis switches (all off by default):
  --deadband RAD     servo position deadband on the leg joints: inside it the XL330 P loop outputs nothing, outside it
                     acts on the excess (E1 10/15: loaded legs ignored 0.03-0.05 rad at P 200/400; unloaded P 200 ~0.01)
  --deadband-head    apply the deadband to the head joints too
  --stiction-scale K multiply only BAM's Stribeck (low-speed/static) friction terms, leaving Coulomb/viscous alone
  --lowpass A_LEG A_HEAD   Pollen robotd target low-pass t = a*t + (1-a)*prev (control.rs: legs 0.7, head 0.5)
  --action-scale S   Pollen robotd walking action scale 0.9 (targets = home + S*action)
  2026-10-30 observation-side hypotheses (the robot's trunk gyro is 4-6x the sim's and broadband 5-25 Hz, 10/17):
  --gyro-noise RMS   white noise (rad/s, per axis, per tick) added to the gyro the Brain receives
  --grav-noise RMS   random small rotation (rad, per axis) applied to the IMU quaternion the Brain receives
  --obs-delay N      N extra ticks of delay on the IMU quaternion and gyro only (joint reads unchanged)
  --act-noise RMS    white noise (rad) on every policy action before it becomes a goal (motor-side noise floor)

  ~/.venvs/microduck-dyn/bin/python jitter_sim.py --model asbuilt --volt 7.0 --drop 0.04 --tick-log out.jsonl [switches]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAC = os.path.dirname(HERE)
# cmu/ copy: runs in the repository's uv environment (uv run <script>); no private venv re-exec.
sys.path.insert(0, MAC)

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import brain_sim_check as bsc  # noqa: E402
import walk_brain  # noqa: E402

HEAD = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--cmd-s", type=float, default=3.0)
    pre.add_argument("--tail", type=float, default=30.0)
    pre.add_argument("--deadband", type=float, default=0.0)
    pre.add_argument("--deadband-head", action="store_true")
    pre.add_argument("--stiction-scale", type=float, default=1.0)
    pre.add_argument("--lowpass", nargs=2, type=float, default=None)
    pre.add_argument("--action-scale", type=float, default=1.0)
    pre.add_argument("--push", nargs=4, type=float, action="append", default=[], metavar=("T", "FX", "FY", "DUR"),
                     help="horizontal force (N) on trunk_base from T s for DUR s: the touch the 10/18 robot may have had")
    pre.add_argument("--head-mass-g", type=float, default=0.0,
                     help="extra point mass on jaw_soft at the Pi board position (A7Z vs Pi Zero sensitivity; unsourced)")
    pre.add_argument("--gyro-noise", type=float, default=0.0, help="rad/s rms white noise per axis on the gyro the Brain sees")
    pre.add_argument("--grav-noise", type=float, default=0.0, help="rad rms random rotation per axis on the quaternion the Brain sees")
    pre.add_argument("--obs-delay", type=int, default=0, help="extra ticks of delay on quaternion and gyro only")
    pre.add_argument("--act-noise", type=float, default=0.0, help="rad rms white noise on every policy action before the goal")
    pre.add_argument("--gyro-noise-dq", type=float, default=0.0,
                     help="motion-driven vibration: gyro noise rms per axis = K x rms leg joint speed (rad/s) this tick "
                          "(foot impacts and play shake the trunk in proportion to how hard the legs move)")
    mine, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    # brain_sim_check runs to walk_at + walk_s + 3; walking is [walk_at, walk_at + walk_s), so stretch walk_s
    # and zero the command after cmd_s in the Brain wrapper below
    walk_at = float(rest[rest.index("--walk-at") + 1]) if "--walk-at" in rest else 4.0
    seed = int(rest[rest.index("--seed") + 1]) if "--seed" in rest else 0
    sys.argv += ["--walk-s", str(mine.cmd_s + mine.tail - 3.0)]
    nrng = np.random.default_rng(1000 + seed)  # observation/action noise stream, separate from the sim's drop rng
    imu_fifo = []

    orig_tick = walk_brain.Brain.tick
    prev = {}

    def tick(self, want, t, pos_t, vel_t, quat, gyro, imu_valid, joint_age, imu_age):
        if want.get("mode") == "walk" and t >= walk_at + mine.cmd_s:
            want = dict(want, vx=0.0, vy=0.0, vtheta=0.0)
        if mine.obs_delay > 0:
            imu_fifo.append((tuple(quat), tuple(gyro)))
            while len(imu_fifo) > mine.obs_delay + 1:
                imu_fifo.pop(0)
            quat, gyro = imu_fifo[0]
            imu_age = imu_age + 0.02 * (len(imu_fifo) - 1)
        if mine.gyro_noise > 0:
            gyro = tuple(float(g) + float(x) for g, x in zip(gyro, nrng.normal(0.0, mine.gyro_noise, 3)))
        if mine.gyro_noise_dq > 0:
            legs = [v * walk_brain.VEL_UNIT for n, v in zip(walk_brain.NAMES, vel_t) if n not in HEAD]
            sd = mine.gyro_noise_dq * float(np.sqrt(np.mean(np.square(legs))))
            gyro = tuple(float(g) + float(x) for g, x in zip(gyro, nrng.normal(0.0, sd, 3)))
        if mine.grav_noise > 0:
            ax = nrng.normal(0.0, mine.grav_noise, 3)
            dq = np.array([1.0, 0.5 * ax[0], 0.5 * ax[1], 0.5 * ax[2]])
            dq = dq / np.linalg.norm(dq)
            quat = tuple(float(x) for x in walk_brain.qmul(tuple(quat), tuple(dq)))
        goal, kp, info = orig_tick(self, want, t, pos_t, vel_t, quat, gyro, imu_valid, joint_age, imu_age)
        if goal is not None and self.mode == "walk" and info.get("action") is not None and mine.act_noise > 0:
            # noise on the goal only (the logged action stays the policy's own output, as on the robot the log
            # holds the raw action); applied before the robotd-style filters below
            goal = {n: g + float(nrng.normal(0.0, mine.act_noise)) if n in self.default else g for n, g in goal.items()}
        if goal is not None and self.mode == "walk" and info.get("action") is not None:
            if mine.action_scale != 1.0:
                goal = {n: self.default[n] + mine.action_scale * (g - self.default[n]) if n in self.default else g
                        for n, g in goal.items()}
            if mine.lowpass:
                out = {}
                for n, g in goal.items():
                    al = mine.lowpass[1] if n in HEAD else mine.lowpass[0]
                    out[n] = al * g + (1 - al) * prev.get(n, g)
                goal = out
        elif goal is not None and self.mode == "walk" and (mine.lowpass or mine.action_scale != 1.0) and prev:
            goal = dict(prev)  # held tick: keep the last filtered/scaled target, not the Brain's raw goal
        if goal is not None:
            prev.clear()
            prev.update(goal)
        return goal, kp, info

    walk_brain.Brain.tick = tick

    orig_load = bsc.ev.load

    def load(c0):
        m, d, c, names, bam = orig_load(c0)
        if mine.stiction_scale != 1.0:
            for pname in ("friction_stribeck", "load_friction_stribeck", "load_friction_motor_stribeck",
                          "load_friction_external_stribeck"):
                prm = getattr(bam, pname, None)
                if prm is not None:
                    prm.value = prm.value * mine.stiction_scale
        if mine.deadband > 0:
            mask = np.array([(n not in HEAD) or mine.deadband_head for n in names])
            upd = c.update

            def update():
                qt = c.q_target.copy()
                q = d.qpos[c.qpos_indexes]
                e = qt - q
                eff = q + np.sign(e) * np.maximum(np.abs(e) - mine.deadband, 0.0)
                c.q_target = np.where(mask, eff, qt)
                upd()
                c.q_target = qt
            c.update = update
        return m, d, c, names, bam

    bsc.ev.load = load
    if mine.head_mass_g:
        cfg0 = bsc.cfg

        def cfg(*a, **k):
            c = cfg0(*a, **k)
            c.setdefault("mass_parts", []).append(dict(body="jaw_soft", mass_kg=mine.head_mass_g / 1000,
                                                       position_m=[0.0139, 0.0, -0.0505]))
            return c
        bsc.cfg = cfg
    if mine.push:
        import mujoco
        step = mujoco.mj_step

        def mj_step(m, d):
            tb = m.body("trunk_base").id
            d.xfrc_applied[tb, :] = 0
            for t, fx, fy, dur in mine.push:
                if t <= d.time < t + dur:
                    d.xfrc_applied[tb, 0] += fx
                    d.xfrc_applied[tb, 1] += fy
            step(m, d)
        bsc.mujoco.mj_step = mj_step
    print(json.dumps(dict(hypothesis=vars(mine))), file=sys.stderr)
    bsc.main()


if __name__ == "__main__":
    main()
