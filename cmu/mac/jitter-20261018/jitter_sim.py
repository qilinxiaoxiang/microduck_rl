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
    mine, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    # brain_sim_check runs to walk_at + walk_s + 3; walking is [walk_at, walk_at + walk_s), so stretch walk_s
    # and zero the command after cmd_s in the Brain wrapper below
    walk_at = float(rest[rest.index("--walk-at") + 1]) if "--walk-at" in rest else 4.0
    sys.argv += ["--walk-s", str(mine.cmd_s + mine.tail - 3.0)]

    orig_tick = walk_brain.Brain.tick
    prev = {}

    def tick(self, want, t, *a, **k):
        if want.get("mode") == "walk" and t >= walk_at + mine.cmd_s:
            want = dict(want, vx=0.0, vy=0.0, vtheta=0.0)
        goal, kp, info = orig_tick(self, want, t, *a, **k)
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
