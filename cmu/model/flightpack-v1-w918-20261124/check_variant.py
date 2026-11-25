"""Checks of the flightpack-v1-w918-20261124 body (build only, no training). Writes check.json beside this file.

1. Load: the manifest parts equal what make_variants.resolve_variant derives from the variant file (the check
   tests/test_cmu_asbuilt_cfg.py::test_manifests_match_their_variant_files and test_current_model.py run on the dated
   variants), unique ids, apply_parts_to_spec on the official scene leaves every inertia positive and physical
   (test_apply_parts_leaves_other_bodies_identical_and_inertias_positive_definite), a 2 s rollout stays finite.
2. Mass, COM in the trunk frame and inertia about the COM at the STAND keyframe, against the 918 g base.
3. Stand pose of the tip test (guide/firmware/pi/stand_pose.py STAND_POSE, soles flat): COM x from the heel in
   predict_tip.py's sole frame, against the real limits 6.25 mm (back) and 28.5 mm (front) measured on the 918 g robot.
4. Static joint torque at that pose for hip roll, knee, ankle: two feet (least-norm wrench split) and one foot (all
   weight, foot wrench with any moment), with 0 / 30 / 60 % of the weight as lift at the fan sites (thrust_sites.json).
5. thrust_sites.json against extensions/flight-20261110/sim/flight.py FANS.
Run on the Mac mini: ~/.venvs/flight-sim/bin/python cmu/model/flightpack-v1-w918-20261124/check_variant.py
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
REPO = MODEL_DIR.parents[1]
SIM = REPO.parent
FL = SIM / "extensions/flight-20261110"
NAME = "flightpack-v1-w918-20261124"
BASE = "battery-v2m5-w918-20261123"
LIMIT_NM = 0.538
G = 9.81
BACK_LIMIT_MM, FRONT_LIMIT_MM = 6.25, 28.5   # com-tip-test-20261123/com_solution.txt (918 g robot, model COM at the tip)
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(SIM / "guide/firmware/pi"))
import apply_masses as am  # noqa: E402
import make_variants  # noqa: E402
from stand_pose import STAND_POSE  # noqa: E402


def build(name):
    parts = [dict(p) for p in am.load_manifest(am.MANIFEST_DIR / f"{name}.json")["parts"]] + [am.battery_removal_part()]
    spec = mujoco.MjSpec.from_file(str(am.SCENE_XML))
    am.apply_ankle_heel_stop(spec)
    rep = am.apply_parts_to_spec(spec, parts)
    return spec, spec.compile(), rep


def totals(m):
    d = mujoco.MjData(m)
    am.home_qpos(m, d)
    M = float(m.body_mass.sum())
    com = d.subtree_com[1].copy()
    I = np.zeros((3, 3))
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3); r = d.xipos[b] - com
        I += R @ np.diag(m.body_inertia[b]) @ R.T + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
    t = m.body("trunk_base").id
    Rt = d.xmat[t].reshape(3, 3)
    return M, Rt.T @ (com - d.xpos[t]), Rt.T @ I @ Rt


class Stand:
    """predict_tip.py's sole frame and stand pose, on any model."""

    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.feet = [g for g in range(m.ngeom) if (m.geom(g).name or "").endswith("foot_collision")]
        self.fb = m.geom_bodyid[self.feet[0]]
        self.pose()
        d = self.d
        allp = np.vstack([self.mesh_world(g) for g in self.feet])
        zmin = allp[:, 2].min(); low = allp[allp[:, 2] < zmin + 0.003]
        coef, *_ = np.linalg.lstsq(np.c_[low[:, 0], low[:, 1], np.ones(len(low))], low[:, 2], rcond=None)
        n = np.array([-coef[0], -coef[1], 1.0]); n /= np.linalg.norm(n)
        xf = np.array([1.0, 0, 0]) - n[0] * n; xf /= np.linalg.norm(xf)
        Rsw = np.vstack([xf, np.cross(n, xf), n]); o_w = low.mean(0)
        Rb0 = d.xmat[self.fb].reshape(3, 3).copy(); pb0 = d.xpos[self.fb].copy()
        self.Rsb, self.ob = Rsw @ Rb0, Rb0.T @ (o_w - pb0)
        self.Rsw = Rsw
        sole = np.vstack([self.to_sole(p) for p in allp])
        self.heel = sole[:, 0].min()

    def pose(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d); d.qpos[3] = 1.0
        for n, v in STAND_POSE.items():
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mujoco.mj_forward(m, d)

    def mesh_world(self, g):
        m, d = self.m, self.d
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T

    def to_sole(self, p):
        Rb = self.d.xmat[self.fb].reshape(3, 3); pb = self.d.xpos[self.fb]
        return self.Rsb @ (Rb.T @ (np.asarray(p) - pb) - self.ob)

    def com_from_heel_mm(self):
        c = self.to_sole(self.d.subtree_com[1]); c[0] -= self.heel
        return c * 1000


def static_torques(m, lift_frac, sites, feet_bodies):
    """Quasi-static joint torques at STAND_POSE: qfrc_bias - qfrc_ext = J_c^T W + tau (tau = 0 on the free joint).
    Gravity is along the sole normal (soles flat on the floor), so the model is rotated to put the sole frame level."""
    d = mujoco.MjData(m)
    st = Stand(m)
    # rotate the floating base so the sole is level (gravity along -sole normal)
    q0 = st.d.qpos.copy()
    Rw = st.Rsw            # rows: sole axes in world at the stand pose; new world = sole frame
    quat = np.zeros(4); mujoco.mju_mat2Quat(quat, (Rw @ np.eye(3)).flatten())
    d.qpos[:] = q0
    qb = np.zeros(4); mujoco.mju_mulQuat(qb, quat, q0[3:7]); d.qpos[3:7] = qb
    d.qpos[0:3] = Rw @ q0[0:3]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    M = float(m.body_mass.sum())
    qext = np.zeros(m.nv)
    t = m.body("trunk_base").id
    R = d.xmat[t].reshape(3, 3)
    for f in sites:
        F = (lift_frac * M * G / len(sites)) * (R @ np.array(f["thrust_axis"]))
        p = d.xpos[t] + R @ np.array(f["position_m"])
        mujoco.mj_applyFT(m, d, F, np.zeros(3), p, t, qext)
    need = d.qfrc_bias - qext
    Js = []
    for b in feet_bodies:
        jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
        mujoco.mj_jacBody(m, d, jp, jr, b)
        Js.append(np.vstack([jp, jr]))
    J = np.vstack(Js)                                   # (6k, nv)
    W = np.linalg.lstsq(J[:, :6].T, need[:6], rcond=None)[0]   # least-norm wrench split over the feet
    tau = need - J.T @ W
    assert np.abs(tau[:6]).max() < 1e-6
    out = {}
    for jn in ("hip_roll", "knee", "ankle"):
        vals = {s: float(tau[m.jnt_dofadr[m.joint(f"{s}_{jn}").id]]) for s in ("left", "right")}
        out[jn] = vals
    return out


def main():
    res = {}
    # 1. load
    man = am.load_manifest(am.MANIFEST_DIR / f"{NAME}.json")
    _, regen, removed, added, chain = make_variants.resolve_variant(make_variants.VARIANT_DIR / f"{NAME}.json")
    ids = [p["id"] for p in man["parts"]]
    res["load"] = dict(manifest_equals_variant=man["parts"] == regen, unique_ids=len(ids) == len(set(ids)),
                       chain_last=chain[-1], removed=len(removed), added=len(added))
    spec, m, rep = build(NAME)
    ref = mujoco.MjSpec.from_file(str(am.SCENE_XML)); am.apply_ankle_heel_stop(ref); mref = ref.compile()
    touched = set(rep["bodies"])
    ok_pd, untouched_same = True, True
    for i in range(m.nbody):
        if m.body_mass[i] > 0:
            I = m.body_inertia[i]
            ok_pd &= bool(np.all(I > 0) and I.max() <= I.sum() - I.max() + 1e-12)
        if m.body(i).name not in touched:
            untouched_same &= bool(m.body_mass[i] == mref.body_mass[i])
    d = mujoco.MjData(m); am.home_qpos(m, d)
    d.ctrl[:] = d.qpos[7:7 + m.nu] if m.nu == m.nq - 7 else d.ctrl
    for _ in range(int(2.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    res["load"].update(inertias_physical=ok_pd, untouched_bodies_identical=untouched_same, bodies_touched=sorted(touched),
                       rollout_2s_finite=bool(np.all(np.isfinite(d.qpos))), nu=int(m.nu), njnt=int(m.njnt),
                       njnt_equals_official=int(m.njnt) == int(mref.njnt), ngeom_equals=int(m.ngeom) == int(mref.ngeom))
    try:
        import visual
        res["load"]["visual_geoms_swapped_as_base"] = visual.apply_visual_to_spec(spec, BASE)
    except Exception as e:  # noqa: BLE001
        res["load"]["visual"] = f"not applied: {type(e).__name__}: {e}"
    # 2. mass, COM, inertia
    _, mb, _ = build(BASE)
    Mp, cp, Ip = totals(m)
    Mb, cb, Ib = totals(mb)
    res["mass"] = dict(pack_model_g=round(Mp * 1000, 2), base_g=round(Mb * 1000, 2), added_g=round((Mp - Mb) * 1000, 2),
                       com_trunk_mm=(cp * 1000).round(1).tolist(), base_com_trunk_mm=(cb * 1000).round(1).tolist(),
                       inertia_kgm2=dict(xx=round(Ip[0, 0], 5), yy=round(Ip[1, 1], 5), zz=round(Ip[2, 2], 5)),
                       inertia_ratio_vs_base=dict(roll=round(Ip[0, 0] / Ib[0, 0], 2), pitch=round(Ip[1, 1] / Ib[1, 1], 2),
                                                  yaw=round(Ip[2, 2] / Ib[2, 2], 2)))
    # 3. stand margins
    sp, sb = Stand(m), Stand(mb)
    xp, xb = sp.com_from_heel_mm(), sb.com_from_heel_mm()
    res["stand"] = dict(pose="guide/firmware/pi/stand_pose.py STAND_POSE, soles flat (predict_tip.py frame)",
                        com_x_from_heel_mm=round(float(xp[0]), 2), com_z_mm=round(float(xp[2]), 2),
                        base_com_x_from_heel_mm=round(float(xb[0]), 2), base_com_z_mm=round(float(xb[2]), 2),
                        front_margin_mm=round(FRONT_LIMIT_MM - float(xp[0]), 2), back_margin_mm=round(float(xp[0]) - BACK_LIMIT_MM, 2),
                        base_front_margin_mm=round(FRONT_LIMIT_MM - float(xb[0]), 2),
                        base_back_margin_mm=round(float(xb[0]) - BACK_LIMIT_MM, 2),
                        limits="real robot tips back at model COM 6.25 mm, forward at 28.5 mm (918 g, 2026-11-23)")
    # 4. torques
    sites = json.loads((HERE / "thrust_sites.json").read_text())["fans"]
    fb = sorted({m.geom_bodyid[g] for g in sp.feet})
    tq = {}
    for label, model in (("pack", m), ("base", mb)):
        st = Stand(model)
        fbm = sorted({model.geom_bodyid[g] for g in st.feet})
        left = [b for b in fbm if model.body(b).name.endswith(("_l", "left")) or "left" in model.body(b).name]
        right = [b for b in fbm if b not in left]
        for lf in ((0.0, 0.3, 0.6) if label == "pack" else (0.0,)):
            two = static_torques(model, lf, sites if label == "pack" else [], fbm)
            one = static_torques(model, lf, sites if label == "pack" else [], left[:1])
            row = {}
            for jn in ("hip_roll", "knee", "ankle"):
                row[jn] = dict(two_feet_frac=round(max(abs(v) for v in two[jn].values()) / LIMIT_NM, 3),
                               one_foot_left_stance_frac=round(abs(one[jn]["left"]) / LIMIT_NM, 3),
                               two_feet_nm=two[jn], one_foot_nm=one[jn])
            tq[f"{label}_lift{int(lf * 100)}"] = row
    res["torque"] = dict(limit_nm=LIMIT_NM, feet_bodies=[m.body(b).name for b in fb], rows=tq,
                         method="quasi-static at STAND_POSE with gravity along the sole normal; two feet = least-norm "
                                "6D wrench split; one foot = all weight on the left foot (foot may carry any moment); "
                                "lift = fraction of the total weight shared equally by the four fans along their axes")
    # 5. thrust sites vs flight.py
    try:
        sys.path.insert(0, str(FL / "sim"))
        import flight
        ff = {f["name"]: f for f in flight.FANS}
        dev = max(max(np.abs(np.array(s["position_m"]) - ff[s["name"]]["p"]).max(),
                      np.abs(np.array(s["thrust_axis"]) - ff[s["name"]]["a"]).max()) for s in sites)
        res["thrust_sites_vs_flight_py_max_dev"] = float(dev)
    except Exception as e:  # noqa: BLE001
        res["thrust_sites_vs_flight_py"] = f"import failed: {type(e).__name__}: {e}"
    (HERE / "check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
