"""Replay of the 2026-10-14 walk-mode start in the as-built sim: hold STAND (KP 400), ramp 1.5 s smoothstep to the
policy default (HOME), then hold HOME. No policy. Reports max trunk tilt and whether the robot falls.

Hypotheses are injected as: --zero JOINT DEG (true joint = commanded + DEG, i.e. a calibration error),
--com-x MM (trunk point mass shift forward), --imu-pitch DEG (sensor only, reported in gb).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_gap import cfg, NAMES, HOME, STAND, feet_flat_rot, sole_points, setup
import evaluate_delay as ev


def run(zero, com_x_mm, ramp_s=1.5, hold_s=3.0, pre_s=2.0, battery=0.09, target=None, start=None):
    extra = []
    if com_x_mm:
        extra = [dict(body='trunk_base', mass_kg=0.03, position_m=[com_x_mm / 1000 / 0.03 * 0.03 * 0 + 0, 0, 0])]
    c0 = cfg(battery, kp=400)
    if com_x_mm:  # move 50 g of trunk mass forward/back by com_x * total/0.05 so the whole COM moves com_x
        c0['mass_parts'] = [dict(body='trunk_base', mass_kg=-0.05, position_m=[0, 0, 0.0]),
                            dict(body='trunk_base', mass_kg=0.05, position_m=[com_x_mm / 1000 * 0.76 / 0.05, 0, 0.0])]
    m, d, c, names, bam = ev.load(c0)
    start = start or STAND; target = target or HOME
    true_start = {n: start[n] + math.radians(zero.get(n, 0)) for n in names}
    Rt = feet_flat_rot(m, d, names, true_start)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rt.ravel())
    ji = setup(m, d, names, true_start, q)
    z = np.vstack(sole_points(m, d))[:, 2].min(); d.qpos[2] -= z - 0.0005; mujoco.mj_forward(m, d)
    qa = m.jnt_qposadr[ji]; trunk = m.body('trunk_base').id
    dt = m.opt.timestep; t = 0.0; maxtilt = 0; fell = None; trace = []
    T = pre_s + ramp_s + hold_s
    while t < T:
        s = min(max((t - pre_s) / ramp_s, 0), 1); s = s * s * (3 - 2 * s)
        cmd = {n: start[n] + (target[n] - start[n]) * s for n in names}
        c.q_target = np.array([cmd[n] + math.radians(zero.get(n, 0)) for n in names])
        c.update(); mujoco.mj_step(m, d); t += dt
        R = d.xmat[trunk].reshape(3, 3)
        tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
        g = R.T @ np.array([0, 0, -1.])
        maxtilt = max(maxtilt, tilt)
        if fell is None and tilt > 45: fell = round(t, 2)
        if abs(t / 0.25 - round(t / 0.25)) < dt / 0.25 / 2: trace.append((round(t, 2), round(float(g[0]), 3)))
    return dict(fell_at=fell, max_tilt=round(maxtilt, 1), gbx_trace=trace)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero', nargs=2, action='append', default=[])
    ap.add_argument('--com-x', type=float, default=0)
    a = ap.parse_args()
    zero = {k: float(v) for k, v in a.zero}
    print(json.dumps(run(zero, a.com_x)))
