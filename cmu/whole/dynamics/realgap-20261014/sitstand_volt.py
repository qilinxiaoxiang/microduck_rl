"""Can the community sit->stand policy rise at this build's 5 V (vs the 7.4 V it was trained for)? As-built mass."""
import json, math, sys
from pathlib import Path
import numpy as np, mujoco, onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parent))
from static_gap import cfg, NAMES, HOME
import evaluate_delay as ev
POL = ev.POLICY_DIR / 'BEST_alpha_sitstand.onnx'
H = np.array([HOME[n] for n in NAMES])

def run(volt, com_x_mm=0, T=8.0):
    c0 = cfg(0.09, kp=200, voltage=volt)
    if com_x_mm:
        c0['mass_parts'] = [dict(body='trunk_base', mass_kg=-0.05, position_m=[0, 0, 0.0]),
                            dict(body='trunk_base', mass_kg=0.05, position_m=[com_x_mm / 1000 * 0.76 / 0.05, 0, 0.0])]
    m, d, c, names, bam = ev.load(c0)
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'SIT'); mujoco.mj_resetDataKeyframe(m, d, k); mujoco.mj_forward(m, d)
    ji = np.array([m.actuator(n).trnid[0] for n in names]); qa, va = m.jnt_qposadr[ji], m.jnt_dofadr[ji]
    trunk = m.body('trunk_base').id; gy = m.sensor('imu_ang_vel').adr[0]
    s = ort.InferenceSession(str(POL)); last = np.zeros(14); c.q_target = d.qpos[qa].copy()
    sub = round(0.02 / m.opt.timestep); zs = []; maxtau = 0
    for tick in range(int(T / 0.02)):
        t = tick * 0.02; R = d.xmat[trunk].reshape(3, 3)
        cmd = np.zeros(13); cmd[0] = 1.0 if t < 1.0 else 0.0  # sit 1 s, then stand
        obs = np.r_[d.sensordata[gy:gy + 3], R.T @ [0, 0, -1.], d.qpos[qa] - H, d.qvel[va], last, cmd].astype(np.float32)
        last = s.run(None, {s.get_inputs()[0].name: obs[None]})[0].reshape(-1); c.q_target = H + last
        for _ in range(sub): c.update(); mujoco.mj_step(m, d); maxtau = max(maxtau, float(np.abs(d.actuator_force).max()))
        tilt = math.degrees(math.acos(np.clip(d.xmat[trunk].reshape(3, 3)[2, 2], -1, 1)))
        if tick % 25 == 0: zs.append((round(t, 1), round(float(d.xpos[trunk, 2]), 3), round(tilt, 1)))
    return dict(volt=volt, com=com_x_mm, final_z=zs[-1][1], final_tilt=zs[-1][2], peak_torque_nm=round(maxtau, 3), trace=zs[::2])

if __name__ == '__main__':
    print('SIT key exists; HOME stand z ~0.12 m')
    for v, com in [(7.4, 0), (5.0, 0), (5.0, 10)]:
        print(json.dumps(run(v, com)))
