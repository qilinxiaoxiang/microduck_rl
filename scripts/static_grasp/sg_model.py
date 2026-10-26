"""Static grasp study: the exact ground-grasp training model, posed rigidly on the floor, and its static numbers.

A pose = the 14 servo angles (robotd order) + the jaw opening. `StaticModel.evaluate` places that rigid shape on the
floor the way gravity would rest it on its feet and returns:
  - support: the convex hull of the foot points that touch the floor (the support polygon), the centre-of-mass (CoM)
    projection and its margin = signed distance to the polygon edge (positive = inside, the duck does not tip),
    the CoM height and the push that would tip it (weight x margin / CoM height);
  - clearances: lowest point of the head on the real CAD meshes (visual) and on the simulator's collision boxes,
    the upper-lip and jaw-tip heights (sites lip_up / jaw_tip_bottom), the lowest non-foot point of the body;
  - holding torques: gravity with each foot carrying the weight share that balances the CoM (mj_jac), per joint;
  - joint-range margins (MJCF ranges = the real joint limits, robotd's safety.rs says so; hip roll +-0.52 as in
    training), self-collisions of the collision geoms, sole tilt of each foot against the floor.

Placement: the base is rotated so that the BOTTOM FACET of the convex hull of both feet (the one whose normal is
closest to the soles' own down axes) lies on z = 0, heading +x. If the CoM projects outside that facet, the margin
is negative: the duck would tip off it.
"""
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WT / "src"))
sys.path.insert(0, str(WT / "scripts"))

TASK = "Mjlab-GroundGrasp-Flat-MicroDuck"
SERVOS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
          "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
DEFAULT_POSE = np.array([0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                         0.0, 0.0873, 0.4579, 0.0049, -0.4530])
G = 9.81
TOUCH_TOL = 0.001   # a foot point within 1 mm of the floor counts as touching (rubber sole, backlash)
_CACHE = {}


def build_model(task=TASK):
    """The exact training model (mjlab Scene compile + sim options), as gg_eval builds it. Cached; returns a copy."""
    if task not in _CACHE:
        import mjlab_microduck.tasks  # noqa: F401
        import mjlab.tasks  # noqa: F401
        from mjlab.scene import Scene
        from mjlab.tasks.registry import load_env_cfg
        cfg = load_env_cfg(task, play=True)
        cfg.scene.num_envs = 1
        m = Scene(cfg.scene, device="cpu").compile()
        cfg.sim.mujoco.apply(m)
        _CACHE[task] = m
    return copy.deepcopy(_CACHE[task])


def _quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def _hull2d(pts):
    if len(pts) < 3:
        return None
    try:
        h = ConvexHull(pts)
    except Exception:
        return None
    return pts[h.vertices]                     # CCW


def _margin(poly, p):
    """Signed distance from p to the boundary of the CCW polygon (positive inside)."""
    if poly is None:
        return -1.0
    dmin = np.inf
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        dmin = min(dmin, (e[0] * (p[1] - a[1]) - e[1] * (p[0] - a[0])) / np.linalg.norm(e))
    return float(dmin)


class StaticModel:
    def __init__(self, task=TASK):
        self.m = m = build_model(task)
        self.d = mujoco.MjData(m)
        nm = lambda t, n: mujoco.mj_name2id(m, t, "robot/" + n)
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.jid = np.array([nm(J, n) for n in SERVOS])
        self.qadr = m.jnt_qposadr[self.jid]
        self.dadr = m.jnt_dofadr[self.jid]
        self.lo, self.hi = m.jnt_range[self.jid, 0].copy(), m.jnt_range[self.jid, 1].copy()
        self.mouth_q = m.jnt_qposadr[nm(J, "passive_mouth")]
        self.free_q = m.jnt_qposadr[nm(J, "trunk_base_freejoint")]
        B = mujoco.mjtObj.mjOBJ_BODY
        self.trunk = nm(B, "trunk_base")
        self.head, self.jaw = nm(B, "jaw_soft"), nm(B, "jaw")
        self.feet = [nm(B, "ankle_left"), nm(B, "ankle_right")]
        self.obj_q = m.jnt_qposadr[mujoco.mj_name2id(m, J, "object/object_free")]
        S = mujoco.mjtObj.mjOBJ_SITE
        self.s_lip_up, self.s_jaw_tip = nm(S, "lip_up"), nm(S, "jaw_tip_bottom")
        self.s_lip_low = nm(S, "lip_low")
        self.s_pad_up, self.s_pad_low = nm(S, "pad_up"), nm(S, "pad_low")
        self.s_feet = [nm(S, "left_foot"), nm(S, "right_foot")]
        self.floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE][0]
        self.mass = float(m.body_subtreemass[self.trunk])
        self.floor_at_zero = False      # measure_qpos of a simulated state: touching = within TOUCH_TOL of z = 0
        robot_bodies = {b for b in range(m.nbody) if self._is_robot(b)}
        self.robot_bodies = robot_bodies
        coll = lambda g: m.geom_contype[g] or m.geom_conaffinity[g]
        self.geom_pts = {g: self._local_pts(g) for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies}
        self.foot_geoms = [g for g in self.geom_pts if m.geom_bodyid[g] in self.feet and coll(g)]
        self.sole_geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{s}_foot_collision")
                           for s in ("left", "right")]
        self.head_vis = [g for g in self.geom_pts if m.geom_bodyid[g] in (self.head, self.jaw) and not coll(g)]
        self.head_col = [g for g in self.geom_pts if m.geom_bodyid[g] in (self.head, self.jaw) and coll(g)]
        self.body_col = [g for g in self.geom_pts if m.geom_bodyid[g] not in (self.head, self.jaw, *self.feet) and coll(g)]
        # per-body point clouds in the body frame, reduced to their convex hull (+ 0.3 mm voxel thinning)
        mujoco.mj_kinematics(m, self.d)
        self.sets = {}
        for key, geoms in (("foot_l", [g for g in self.foot_geoms if m.geom_bodyid[g] == self.feet[0]]),
                           ("foot_r", [g for g in self.foot_geoms if m.geom_bodyid[g] == self.feet[1]]),
                           ("head_vis", [g for g in self.head_vis if m.geom_bodyid[g] == self.head]),
                           ("jaw_vis", [g for g in self.head_vis if m.geom_bodyid[g] == self.jaw]),
                           ("head_col", [g for g in self.head_col if m.geom_bodyid[g] == self.head]),
                           ("jaw_col", [g for g in self.head_col if m.geom_bodyid[g] == self.jaw])):
            self.sets[key] = (int(m.geom_bodyid[geoms[0]]), self._body_cloud(geoms))
        self.other_sets = []
        for b in sorted({int(m.geom_bodyid[g]) for g in self.body_col}):
            self.other_sets.append((b, self._body_cloud([g for g in self.body_col if m.geom_bodyid[g] == b])))
        self.sole_sets = [self._body_cloud([g]) for g in self.sole_geoms]
        # the support search only needs the bottom of each foot: points within 8 mm of the sole plane (body frame)
        self.foot_bottom = {}
        for key, fb, s in (("foot_l", self.feet[0], self.s_feet[0]), ("foot_r", self.feet[1], self.s_feet[1])):
            P = self.sets[key][1]
            Rb = self.d.xmat[fb].reshape(3, 3); Rs = self.d.site_xmat[s].reshape(3, 3)
            down = Rb.T @ -Rs[:, 2]                  # the sole's down axis in the foot body frame
            h = P @ down
            Pb = P[h > h.max() - 0.008]
            _, idx = np.unique(np.round(Pb / 0.0015).astype(np.int64), axis=0, return_index=True)   # 1.5 mm voxels:
            Pb = Pb[np.sort(idx)]                    # only the facet orientation comes from these; heights from P
            self.foot_bottom[key] = (fb, Pb[ConvexHull(Pb).vertices])

    def _is_robot(self, b):
        while b > 0:
            if b == self.trunk:
                return True
            b = self.m.body_parentid[b]
        return False

    def _local_pts(self, g):
        m = self.m
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].copy()
            if len(v) > 8:                       # lowest points and support hulls only need the hull vertices
                try:
                    v = v[ConvexHull(v).vertices]
                except Exception:
                    pass
            return v
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = m.geom_size[g]
            return np.array([[a * s[0], b * s[1], c * s[2]] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        return np.zeros((1, 3))

    def _body_cloud(self, geoms):
        m, d = self.m, self.d
        b = m.geom_bodyid[geoms[0]]
        pts = []
        for g in geoms:
            # geom frame -> body frame (model constants)
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3, 3)
            pts.append(m.geom_pos[g] + self.geom_pts[g] @ R.T)
        P = np.concatenate(pts)
        if len(P) > 8:
            P = P[ConvexHull(P).vertices]
            key = np.round(P / 0.0003).astype(np.int64)
            _, idx = np.unique(key, axis=0, return_index=True)
            P = P[np.sort(idx)]
            if len(P) > 8:
                P = P[ConvexHull(P).vertices]
        return P

    def body_world(self, b, P):
        d = self.d
        return d.xpos[b] + P @ d.xmat[b].reshape(3, 3).T

    def world_pts(self, g):
        d = self.d
        return d.geom_xpos[g] + self.geom_pts[g] @ d.geom_xmat[g].reshape(3, 3).T

    # ---------------------------------------------------------------------------------------------------------
    def set_joints(self, q, theta):
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[self.free_q:self.free_q + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[self.qadr] = q
        d.qpos[self.mouth_q] = theta
        d.qpos[self.obj_q:self.obj_q + 3] = [2.0, 2.0, 0.01]       # object parked far away
        d.qvel[:] = 0

    def place(self, q, theta):
        """Rest the rigid pose on its feet (see module docstring). Returns the support facet normal tilt info."""
        m, d = self.m, self.d
        self.set_joints(q, theta)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        # everything in the base frame (base at identity)
        foot_pts = np.concatenate([self.body_world(*self.foot_bottom[k]) for k in ("foot_l", "foot_r")])
        hull = ConvexHull(foot_pts)
        down = np.zeros(3)
        for s in self.s_feet:
            down -= d.site_xmat[s].reshape(3, 3)[:, 2]
        down /= np.linalg.norm(down)
        # outward normals of hull facets; pick the one closest to the soles' down axis
        eq = hull.equations                     # n . x + off <= 0 inside, n outward unit
        k = int(np.argmax(eq[:, :3] @ down))
        n, off = eq[k, :3], eq[k, 3]
        # rotation that sends n to world -z, heading of the trunk x-axis kept along +x
        z_w = -n
        hx = np.array([1.0, 0, 0]) - np.dot([1.0, 0, 0], z_w) * z_w
        hx /= np.linalg.norm(hx)
        y_w = np.cross(z_w, hx)
        R = np.vstack([hx, y_w, z_w])           # rows: world axes expressed in the base frame -> base->world = R
        # height: the facet plane (n . x = -off) goes to z = 0
        p0 = R @ (-off * n)                      # a point of the plane, rotated
        pos = np.array([0.0, 0.0, -p0[2]])
        d.qpos[self.free_q:self.free_q + 3] = pos
        d.qpos[self.free_q + 3:self.free_q + 7] = _quat_from_mat(R)
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        # recentre x, y on the feet midpoint
        mid = 0.5 * (d.xpos[self.feet[0]] + d.xpos[self.feet[1]])
        d.qpos[self.free_q] -= mid[0]
        d.qpos[self.free_q + 1] -= mid[1]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        return n

    def evaluate(self, q, theta, collide=True, torques=True):
        self.place(q, theta)
        return self._measure(q, theta, collide, torques)

    def measure_qpos(self, qpos, collide=False):
        """The same numbers for a simulated state (full qpos of this model), without re-placing it."""
        m, d = self.m, self.d
        d.qpos[:] = qpos
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        self.floor_at_zero = True
        try:
            return self._measure(d.qpos[self.qadr].copy(), float(d.qpos[self.mouth_q]), collide, torques=False)
        finally:
            self.floor_at_zero = False

    def _measure(self, q, theta, collide, torques):
        m, d = self.m, self.d
        r = {}
        fl, fr = self.body_world(*self.sets["foot_l"]), self.body_world(*self.sets["foot_r"])
        foot_pts = np.concatenate([fl, fr])
        zmin = max(foot_pts[:, 2].min(), 0.0) if self.floor_at_zero else foot_pts[:, 2].min()
        touch = foot_pts[foot_pts[:, 2] < zmin + TOUCH_TOL][:, :2]
        poly = _hull2d(touch)
        com = d.subtree_com[self.trunk].copy()
        r["com"] = com
        r["poly"] = poly
        r["margin"] = _margin(poly, com[:2])
        r["com_h"] = float(com[2])
        r["tip_push_N"] = float(self.mass * G * max(r["margin"], 0.0) / max(com[2], 1e-3))
        r["tip_angle_deg"] = float(np.degrees(np.arctan2(max(r["margin"], 0.0), com[2])))
        # which feet touch, and how flat they are
        r["foot_gap_mm"] = [1000 * float(p[:, 2].min() - zmin) for p in (fl, fr)]
        r["sole_tilt_deg"] = [self._sole_tilt(self.body_world(fb, P)) for fb, P in zip(self.feet, self.sole_sets)]
        # clearances
        hv = np.concatenate([self.body_world(*self.sets[k]) for k in ("head_vis", "jaw_vis")])
        hc = np.concatenate([self.body_world(*self.sets[k]) for k in ("head_col", "jaw_col")])
        bc = np.concatenate([self.body_world(b, P) for b, P in self.other_sets])
        r["head_low_vis"] = float(hv[:, 2].min())
        r["head_low_col"] = float(hc[:, 2].min())
        r["body_low"] = float(bc[:, 2].min())
        H = d.xmat[self.head].reshape(3, 3)
        lowest = hv[np.argmin(hv[:, 2])]
        r["head_low_fwd"] = float(-(H.T @ (lowest - d.xpos[self.head]))[2])     # head-local forward coord, m
        r["lip_up"] = float(d.site_xpos[self.s_lip_up][2])
        r["jaw_tip"] = float(d.site_xpos[self.s_jaw_tip][2])
        fwd, up = -H[:, 2], H[:, 0]
        r["head_pitch_deg"] = float(np.degrees(np.arctan2(-fwd[2], up[2])))
        r["bisector_deg"] = r["head_pitch_deg"] + np.degrees(theta) / 2
        mouth = 0.5 * (d.site_xpos[self.s_lip_up] + d.site_xpos[self.s_lip_low])
        r["mouth_xy"] = mouth[:2].copy()
        feet_mid = 0.5 * (d.xpos[self.feet[0]] + d.xpos[self.feet[1]])
        r["reach"] = float(mouth[0] - feet_mid[0])
        r["grasp_pt"] = 0.5 * (d.site_xpos[self.s_pad_up] + d.site_xpos[self.s_pad_low])
        r["trunk_pitch_deg"] = float(np.degrees(np.arcsin(np.clip(-d.xmat[self.trunk].reshape(3, 3)[2, 0], -1, 1))))
        r["range_margin"] = float(min((q - self.lo).min(), (self.hi - q).min()))
        if collide:
            mujoco.mj_collision(m, d)
            selfc, worst = 0, 0.0
            for c in d.contact[:d.ncon]:
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                if b1 in self.robot_bodies and b2 in self.robot_bodies and c.dist < -0.0005:
                    selfc += 1
                    worst = min(worst, c.dist)
            r["selfc"], r["self_pen_mm"] = selfc, -1000 * worst
        if torques:
            r["tau"] = self.static_torques(com, poly)
            r["tau_max"] = float(np.abs(r["tau"]).max())
            r["tau_arg"] = SERVOS[int(np.argmax(np.abs(r["tau"])))]
        return r

    def _sole_tilt(self, sp):
        # plane fit of the sole's bottom face: points within 3 mm of its lowest point
        low = sp[sp[:, 2] < sp[:, 2].min() + 0.003]
        if len(low) < 3:
            return 90.0
        c = low - low.mean(0)
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        n = vt[2]
        return float(np.degrees(np.arccos(min(1.0, abs(n[2])))))

    def static_torques(self, com, poly):
        """Holding torques (N.m) at the servos: gravity minus the floor forces. The weight is split between the feet
        so the forces balance the CoM: each foot pushes at the point of its own contact patch closest to the CoM line
        (x of the CoM), shares chosen so the total moment about the CoM is zero in y."""
        m, d = self.m, self.d
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)
        tau = d.qfrc_bias.copy()
        W = self.mass * G
        ys = [d.xpos[f][1] for f in self.feet]
        # shares from the lever rule on y (two point supports at the feet's y)
        if abs(ys[0] - ys[1]) > 1e-4:
            wl = np.clip((com[1] - ys[1]) / (ys[0] - ys[1]), 0, 1)
        else:
            wl = 0.5
        for fb, share, y in zip(self.feet, (wl, 1 - wl), ys):
            p = np.array([com[0], y, 0.0])
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jacp, None, p, fb)
            tau -= jacp.T @ np.array([0, 0, share * W])
        return tau[self.dadr]


def default_grasp_low_pose(npz="/Users/remi/microduck/notes/ground-grasp/film/default_open_s2.npz"):
    """The default grasp (shipped alpha_ground_pick, beak schedule) at its lowest beak point, from the film log
    (same model, same qpos layout). Returns (q14, theta, t)."""
    z = np.load(npz, allow_pickle=True)
    names = list(z["joint_names"])
    sm = StaticModel()
    m, d = sm.m, sm.d
    best = (9.0, None)
    for t, qp in zip(z["t"], z["qpos"]):
        if t < float(z["pick_start"]):
            continue
        d.qpos[:] = qp
        mujoco.mj_kinematics(m, d)
        low = min(d.site_xpos[sm.s_lip_up][2], d.site_xpos[sm.s_jaw_tip][2])
        if low < best[0]:
            best = (low, (qp.copy(), t))
    qp, t = best[1]
    q = np.array([qp[sm.qadr[i]] for i in range(14)])
    return q, float(qp[sm.mouth_q]), float(t), qp
