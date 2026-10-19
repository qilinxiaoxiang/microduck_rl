"""walk61/stand61 in the as-built sim with injected real-robot gaps, started the way the real robot starts.

Gaps: joint zero errors (true = commanded + deg), COM shift forward (mm), IMU pitch mount error (deg, sensor only).
Start: feet-flat at STAND or HOME, policy on from t=0 (no open-loop ramp) or after a ramp. KP 200 at 5 V.
Schedule: stand policy until --walk-at, then walk command vx for --walk-s, then stand again.
"""
import argparse, collections, json, math, sys
from pathlib import Path
import numpy as np, mujoco, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_gap import cfg, NAMES, HOME, STAND, feet_flat_rot, sole_points, setup
import evaluate_delay as ev
H = np.array([HOME[n] for n in NAMES])
MODELS = Path(__file__).resolve().parents[2] / 'runtime-overlay/models'


def rot_y(deg):
    a = math.radians(deg); return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def run(zero=None, com_x_mm=0.0, imu_pitch=0.0, start='STAND', ramp_s=0.0, vx=0.3, walk_at=3.0, walk_s=10.0,
        T=16.0, sensor_delay=1, action_delay=1, seed=0, kp=200, battery=0.09, joint_noise=0.0, push=None, head=None, volt=5.0):
    zero = zero or {}
    rng = np.random.default_rng(seed)
    c0 = cfg(battery, kp=kp, voltage=volt)
    if com_x_mm:
        c0['mass_parts'] = [dict(body='trunk_base', mass_kg=-0.05, position_m=[0, 0, 0.0]),
                            dict(body='trunk_base', mass_kg=0.05, position_m=[com_x_mm / 1000 * 0.76 / 0.05, 0, 0.0])]
    m, d, c, names, bam = ev.load(c0)
    z0 = np.array([math.radians(zero.get(n, 0)) for n in names])
    P0 = STAND if start == 'STAND' else HOME
    p0 = np.array([P0[n] for n in names])
    true0 = dict(zip(names, p0 + z0))
    Rt = feet_flat_rot(m, d, names, true0); q = np.zeros(4); mujoco.mju_mat2Quat(q, Rt.ravel())
    ji = setup(m, d, names, true0, q)
    zmin = np.vstack(sole_points(m, d))[:, 2].min(); d.qpos[2] -= zmin - 0.0005; mujoco.mj_forward(m, d)
    qa, va = m.jnt_qposadr[ji], m.jnt_dofadr[ji]
    trunk = m.body('trunk_base').id; gyro = m.sensor('imu_ang_vel').adr[0]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = {k: ort.InferenceSession(str(MODELS / f'{k}61.onnx'), sess_options=so) for k in ('walk', 'stand')}
    Rimu = rot_y(imu_pitch)
    last = np.zeros(14); aq = collections.deque([p0.copy()] * (action_delay + 1), maxlen=action_delay + 1)
    sq = collections.deque(maxlen=sensor_delay + 1)
    c.q_target = p0 + z0
    sub = round(0.02 / m.opt.timestep); fell = None; maxtilt = 0; x0 = None; rows = []
    for tick in range(int(T / 0.02)):
        t = tick * 0.02
        R = d.xmat[trunk].reshape(3, 3)
        g = Rimu.T @ (R.T @ np.array([0, 0, -1.]))
        w = Rimu.T @ d.sensordata[gyro:gyro + 3]
        qm = d.qpos[qa] - z0 + rng.normal(0, joint_noise, 14)  # what the encoders report in commanded frame
        s = np.r_[w, g, qm - H, d.qvel[va]]
        if not sq: sq.extend([s] * sq.maxlen)
        sq.append(s); s = sq[0]
        if t < ramp_s:
            a = min(t / ramp_s, 1); a = a * a * (3 - 2 * a); target = p0 + (H - p0) * a; last[:] = 0
        else:
            cmd = np.zeros(13)
            walking = walk_at <= t < walk_at + walk_s
            if walking: cmd[0] = vx
            if head is not None: cmd[3:7] = head(t)
            k = 'walk' if walking else 'stand'
            obs = np.r_[s, last, cmd].astype(np.float32)
            last = sess[k].run(None, {sess[k].get_inputs()[0].name: obs[None]})[0].reshape(-1)
            target = H + last
        aq.append(target.copy()); c.q_target = aq[0] + z0
        if push and abs(t - push[0]) < 1e-6: d.qvel[0] += push[1]
        for _ in range(sub): c.update(); mujoco.mj_step(m, d)
        tilt = math.degrees(math.acos(np.clip(d.xmat[trunk].reshape(3, 3)[2, 2], -1, 1)))
        maxtilt = max(maxtilt, tilt)
        if t >= walk_at and x0 is None: x0 = d.xpos[trunk, 0]
        if tilt > 45 or d.xpos[trunk, 2] < 0.04:
            fell = round(t, 2); break
        if tick % 25 == 0: rows.append((round(t, 1), round(float(g[0]), 3), round(float(d.xpos[trunk, 0]), 3),
                                        [round(float(x), 3) for x in (d.qpos[qa] - z0 - H)[5:9]]))
    return dict(fell_at=fell, max_tilt=round(maxtilt, 1), dist_x=None if x0 is None else round(float(d.xpos[trunk, 0] - x0), 3),
                trace=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--zero', nargs=2, action='append', default=[])
    ap.add_argument('--com-x', type=float, default=0)
    ap.add_argument('--imu-pitch', type=float, default=0)
    ap.add_argument('--start', default='STAND')
    ap.add_argument('--ramp', type=float, default=0)
    ap.add_argument('--vx', type=float, default=0.3)
    a = ap.parse_args()
    print(json.dumps(run({k: float(v) for k, v in a.zero}, a.com_x, a.imu_pitch, a.start, a.ramp, a.vx)))
