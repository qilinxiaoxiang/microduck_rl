"""Sim-vs-real static balance check for the as-built (V1.9) MicroDuck model.

For a pose (model rad, policy joint order) it finds the feet-flat trunk orientation kinematically and reports the
trunk-frame projected gravity (what the policy/IMU see) and the whole-robot COM relative to the foot soles.
Then it simulates the real 2026-10-14 sequence: hold STAND at KP 400, ramp to the policy default over 1.5 s.
"""
import json, math, sys
from pathlib import Path
import numpy as np, mujoco
HERE = Path(__file__).resolve().parent
WHOLE = HERE.parents[1]
sys.path.insert(0, str(WHOLE / 'delay-repair-20260912'))
import evaluate_delay as ev

ORDER = ev.HOME  # policy order: left leg 5, neck, head x3, right leg 5
NAMES = ['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch',
         'head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']
HOME = dict(zip(NAMES, ev.HOME))
STAND = dict(HOME, right_ankle=-0.5403, left_ankle=0.5403, neck_pitch=0.5236, head_pitch=0.5236)


MODELS = {  # mass model -> run config; 'v19' stays the default
    'v19': WHOLE / 'dynamics/v19-20261014/runs/const-bat90-sensor0-nocomp-seed0.config.json',
    'asbuilt': WHOLE / 'dynamics/asbuilt-20261015/asbuilt.config.json',  # D-007 bench backpack, 2026-10-15
}


def cfg(battery=0.09, model='v19', **kw):
    c = json.loads(MODELS[model].read_text())
    c.update(remove_old_battery_kg=battery, **kw)
    if c.get('mass_manifest') and not Path(c['mass_manifest']).is_absolute():  # cmu/ copy: paths relative to the config
        c['mass_manifest'] = str(MODELS[model].parent / c['mass_manifest'])
    return c


def setup(m, d, names, pose, trunk_quat=(1, 0, 0, 0)):
    ji = np.array([m.actuator(n).trnid[0] for n in names])
    d.qpos[:3] = [0, 0, .12]; d.qpos[3:7] = trunk_quat
    d.qpos[m.jnt_qposadr[ji]] = [pose[n] for n in names]
    mujoco.mj_forward(m, d)
    return ji


def sole_points(m, d):
    pts = []
    for g in ('left_foot_collision', 'right_foot_collision'):
        gid = m.geom(g).id
        mid = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n]
        R = d.geom_xmat[gid].reshape(3, 3)
        pts.append(v @ R.T + d.geom_xpos[gid])
    return pts


def sole_normal(m, d, g):
    """Unit normal of the sole plate (smallest-variance axis of its mesh vertices), pointing down out of the foot."""
    gid = m.geom(g).id; mid = m.geom_dataid[gid]
    v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    R = d.geom_xmat[gid].reshape(3, 3); w = v @ R.T
    c = w - w.mean(0); n = np.linalg.svd(c)[2][-1]
    body_c = d.xpos[m.geom_bodyid[gid]] - (w.mean(0))
    return n if n @ body_c < 0 else -n


def feet_flat_rot(m, d, names, pose):
    """World rotation of the trunk that makes the mean sole normal point straight down (yaw kept)."""
    setup(m, d, names, pose)
    n = sole_normal(m, d, 'left_foot_collision') + sole_normal(m, d, 'right_foot_collision')
    n /= np.linalg.norm(n); t = np.array([0, 0, -1.])
    ax = np.cross(n, t); s = np.linalg.norm(ax); cth = n @ t
    if s < 1e-9: return np.eye(3)
    ax /= s; K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + math.sin(math.atan2(s, cth)) * K + (1 - cth) * K @ K


def report(m, d, names, pose, Rref=None):
    Rt = feet_flat_rot(m, d, names, pose)
    q = np.zeros(4); mujoco.mju_mat2Quat(q, Rt.ravel())
    setup(m, d, names, pose, q)
    pts = sole_points(m, d)
    allp = np.vstack(pts)
    zmin = allp[:, 2].min()
    d.qpos[2] -= zmin; mujoco.mj_forward(m, d)
    pts = sole_points(m, d); allp = np.vstack(pts)
    contact = allp[allp[:, 2] < 1.5e-3]
    com = d.subtree_com[0]
    g = Rt.T @ np.array([0, 0, -1.])
    fx = (contact[:, 0].min(), contact[:, 0].max())
    return dict(gb=np.round(g, 3).tolist(), trunk_pitch_deg=round(math.degrees(math.atan2(g[0], -g[2])), 2),
                com_x_mm=round(com[0] * 1000, 1), sole_x_mm=[round(fx[0] * 1000, 1), round(fx[1] * 1000, 1)],
                com_margin_front_mm=round((fx[1] - com[0]) * 1000, 1), com_margin_back_mm=round((com[0] - fx[0]) * 1000, 1))


def settle_ref(m, d, c, names):
    ji = setup(m, d, names, HOME)
    qa = m.jnt_qposadr[ji]
    c.q_target = d.qpos[qa].copy()
    for _ in range(int(1.0 / m.opt.timestep)):
        c.update(); mujoco.mj_step(m, d)
    return d.xmat[m.body('ankle_left').id].reshape(3, 3).copy()


if __name__ == '__main__':
    out = {}
    for bat in (0.06, 0.09, 0.12):
        m, d, c, names, bam = ev.load(cfg(bat, kp=400))
        assert names == NAMES, names
        Rref = None
        out[f'bat{int(bat*1000)}'] = dict(mass_g=round(float(m.body_mass.sum()) * 1000, 1),
                                          HOME=report(m, d, names, HOME, Rref), STAND=report(m, d, names, STAND, Rref))
    print(json.dumps(out, indent=1))
