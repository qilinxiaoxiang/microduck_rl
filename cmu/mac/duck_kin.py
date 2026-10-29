"""Head-camera floor projection for the MicroDuck in numpy only (no MuJoCo): the A7Z side of target_ground.py.

Same frames and conventions as target_ground.DuckGeometry (read its docstring): foot frame on the floor below the
trunk origin, x forward, y left, trunk roll and pitch from gb, the lowest sole vertex on z = 0; camera frame x the
optical axis, y image left, z image up; continuous pixel coordinates (the centre of pixel column j, row i is
(j + 0.5, i + 0.5)). The kinematic tree comes from duck_kin.json (duck_kin_export.py, as-built model);
test_duck_kin.py checks the two agree.

Camera (default): the Innomaker U20CAM-720P on the face (D-023, mounted a quarter turn round with the plug edge up
since 2026-10-26), as a7z_cam.py publishes it: 1280 x 720 shrunk to 640 x 360 and turned 180 deg to upright.
- Intrinsics from u20cam_intrinsics.json (cam_calib.py --intrinsics). Until a real calibration exists that file holds a
  datasheet default (`"source": "datasheet"`): equidistant fisheye from the manual's 102 deg horizontal FOV over 640 px,
  f = 320 / rad(51) = 359.5 px, centre (320, 180). Equidistant gives 117 deg on the diagonal against the manual's 120;
  a pinhole with the same 102 deg would give 110, so the lens is closer to a fisheye.
- Mount from u20cam_mount.json: the lens centre and axes in the guide frame (CAD, build_face_u20.py) on the head body
  jaw_soft, plus fitted residual pitch / yaw / roll and an optional lens shift (cam_calib.py --mount). The guide frame
  is the MJCF trunk_base frame at qpos0 in mm (dynamics/asbuilt-20261015/build_asbuilt_manifest.py: a guide point p maps
  to body b as R_b^T (p - x_b)); DuckKin does that mapping with duck_kin.json at load time.
The Radxa Camera 13M 214 (RADXA_13M_UPRIGHT, the model's `head_camera` site) stays for old logs and the sim:
DuckKin(RADXA_13M_UPRIGHT) projects exactly as before. Radxa numbers: full-sensor 4208 x 3120 scaled to 640 x 480 and
turned a quarter clockwise, upright 480 wide by 640 tall; HFOV 66.2 deg along the sensor's long side, VFOV 51.6 deg
(docs.radxa.com/en/accessories/camera_13m_214), so fx = 240 / tan(25.8 deg) = 497 px, fy = 320 / tan(33.1 deg) = 491 px.

Lens models (OpenCV's, so a cv2 calibration loads unchanged; normalised coordinates a = X/Z right, b = Y/Z down):
  pinhole   no distortion
  radtan    cv2.calibrateCamera: dist (k1, k2, p1, p2, k3)
  fisheye   cv2.fisheye: dist (k1, k2, k3, k4), theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPRIGHT = (0.0, 0.0, -1.0)
U20CAM_INTRINSICS = os.path.join(HERE, "u20cam_intrinsics.json")
U20CAM_MOUNT = os.path.join(HERE, "u20cam_mount.json")


class Intrinsics:
    """Camera model in continuous pixel coordinates of a width x height image (pinhole, radtan or fisheye)."""

    def __init__(self, fx, fy, cx, cy, width, height, model="pinhole", dist=(), source=None, meta=None):
        self.fx, self.fy, self.cx, self.cy, self.width, self.height = fx, fy, cx, cy, width, height
        self.model = model
        self.dist = tuple(float(d) for d in dist)
        self.source, self.meta = source, dict(meta or {})
        if model == "radtan":
            self.dist = (self.dist + (0.0,) * 5)[:5]
        elif model == "fisheye":
            self.dist = (self.dist + (0.0,) * 4)[:4]
        elif model != "pinhole":
            raise ValueError(f"unknown lens model {model}")
        self.r_max = self._fold_radius()

    def _fold_radius(self):
        """Undistorted radius (tan of the off-axis angle) where the radial distortion stops growing: a fitted
        polynomial folds back beyond it and maps far-off rays into the picture, so project() and ray() refuse there."""
        if self.model == "pinhole":
            return math.inf
        if self.model == "radtan":
            k1, k2, _, _, k3 = self.dist
            rs = np.linspace(1e-3, 6.0, 6000)
            rd = rs * (1 + rs ** 2 * (k1 + rs ** 2 * (k2 + rs ** 2 * k3)))
        else:
            k1, k2, k3, k4 = self.dist
            rs = np.tan(np.linspace(1e-3, math.pi / 2 - 1e-3, 6000))
            th = np.arctan(rs)
            t2 = th * th
            rd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        down = np.nonzero(np.diff(rd) <= 0)[0]
        return float(rs[down[0]]) if down.size else math.inf

    @classmethod
    def from_fov(cls, hfov_deg, vfov_deg, width, height):
        return cls((width / 2) / math.tan(math.radians(hfov_deg) / 2), (height / 2) / math.tan(math.radians(vfov_deg) / 2),
                   width / 2, height / 2, width, height)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        meta = {k: v for k, v in d.items() if k not in ("fx", "fy", "cx", "cy", "width", "height", "model", "dist", "source")}
        return cls(d["fx"], d["fy"], d["cx"], d["cy"], int(d["width"]), int(d["height"]), d.get("model", "pinhole"),
                   d.get("dist", ()), d.get("source"), meta)

    @property
    def vfov_deg(self):
        """Vertical field of view of the pinhole part (the distortion widens a real wide lens)."""
        return math.degrees(2 * math.atan((self.height / 2) / self.fy))

    def scaled(self, width, height):
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height, self.model, self.dist,
                          self.source, self.meta)

    # --- lens distortion in normalised coordinates (OpenCV's camera frame: a right, b down) ----------------------
    def distort(self, a, b):
        if self.model == "pinhole":
            return a, b
        if self.model == "radtan":
            k1, k2, p1, p2, k3 = self.dist
            r2 = a * a + b * b
            rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            return (a * rad + 2 * p1 * a * b + p2 * (r2 + 2 * a * a),
                    b * rad + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b)
        k1, k2, k3, k4 = self.dist
        r = math.hypot(a, b)
        if r < 1e-12:
            return a, b
        th = math.atan(r)
        t2 = th * th
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        return a * thd / r, b * thd / r

    def undistort(self, ad, bd):
        """Normalised (a, b) whose distorted image is (ad, bd); None where the model does not invert (behind the lens)."""
        if self.model == "pinhole":
            return ad, bd
        if self.model == "fisheye":
            k1, k2, k3, k4 = self.dist
            rd = math.hypot(ad, bd)
            if rd < 1e-12:
                return ad, bd
            th = min(rd, 1.5)
            for _ in range(30):
                t2 = th * th
                f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - rd
                df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
                if df <= 1e-9:
                    return None
                step = f / df
                th -= step
                if abs(step) < 1e-13:
                    break
            if not 0 <= th < math.pi / 2 - 1e-6 or abs(f) > 1e-9:
                return None
            s = math.tan(th) / rd
            return ad * s, bd * s
        a, b = ad, bd   # radtan: Newton with a numerical Jacobian
        for _ in range(30):
            ea, eb = self.distort(a, b)
            ra, rb = ea - ad, eb - bd
            if abs(ra) < 1e-13 and abs(rb) < 1e-13:
                return a, b
            h = 1e-7
            j11, j21 = ((self.distort(a + h, b)[0] - ea) / h, (self.distort(a + h, b)[1] - eb) / h)
            j12, j22 = ((self.distort(a, b + h)[0] - ea) / h, (self.distort(a, b + h)[1] - eb) / h)
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-12:
                return None
            a -= (j22 * ra - j12 * rb) / det
            b -= (-j21 * ra + j11 * rb) / det
        ea, eb = self.distort(a, b)
        return (a, b) if abs(ea - ad) < 1e-9 and abs(eb - bd) < 1e-9 else None

    # --- the camera frame of duck_kin (x optical axis, y image left, z image up) <-> pixels --------------------
    def project(self, p):
        """Pixel (u, v) of a point in the camera frame; None if it is not in front of the lens."""
        if p[0] <= 1e-9:
            return None
        a, b = -p[1] / p[0], -p[2] / p[0]
        if a * a + b * b > self.r_max ** 2:
            return None
        ad, bd = self.distort(a, b)
        return self.cx + self.fx * ad, self.cy + self.fy * bd

    def ray(self, u, v):
        """Unit direction in the camera frame of pixel (u, v); None if the model does not invert there."""
        ab = self.undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy)
        if ab is None or ab[0] ** 2 + ab[1] ** 2 > self.r_max ** 2:
            return None
        d = np.array([1.0, -ab[0], -ab[1]])
        return d / np.linalg.norm(d)


RADXA_13M_UPRIGHT = Intrinsics.from_fov(51.6, 66.2, 480, 640)
U20CAM_DATASHEET = Intrinsics(320 / math.radians(51.0), 320 / math.radians(51.0), 320.0, 180.0, 640, 360, "fisheye",
                              (0.0, 0.0, 0.0, 0.0), "datasheet")


def load_intrinsics(path=U20CAM_INTRINSICS):
    return Intrinsics.from_json(path)


def load_mount(path=U20CAM_MOUNT):
    with open(path) as f:
        return json.load(f)


def quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def axis_angle(axis, a):
    x, y, z = axis / np.linalg.norm(axis)
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def trunk_rotation(gb):
    """Trunk-to-foot rotation Ry(pitch) @ Rx(roll) from the trunk-frame gravity direction, zero yaw."""
    g = np.asarray(gb, float)
    g = g / np.linalg.norm(g)
    p = math.asin(float(np.clip(g[0], -1, 1)))
    r = math.atan2(-g[1], -g[2])
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    return np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]) @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])


def _rx(a):
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def _rz(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def offset_rotation(pitch_deg, yaw_deg, roll_deg):
    """Residual camera rotation in the camera frame: yaw (left +) about z, then pitch (nose down +) about y, then roll
    (image turned clockwise as seen from behind, i.e. right side down, +) about the optical axis x."""
    return _rz(math.radians(yaw_deg)) @ _ry(math.radians(pitch_deg)) @ _rx(math.radians(roll_deg))


class DuckKin:
    """Camera and mouth-tip poses in the foot frame and the floor projection, from duck_kin.json.

    DuckKin()                       U20CAM: u20cam_intrinsics.json and u20cam_mount.json (the default)
    DuckKin(intrinsics)             that camera at the model's head_camera site (Radxa 13M logs, the sim's camera)
    DuckKin(intrinsics, mount=m)    that camera at mount m (dict in the u20cam_mount.json format, or its path)
    pitch_offset_deg adds a nose-down offset on top of the mount's own (the old single-point calibration)."""

    def __init__(self, intrinsics=None, pitch_offset_deg=0.0, path=os.path.join(HERE, "duck_kin.json"), mount=None):
        with open(path) as f:
            k = json.load(f)
        if intrinsics is None:
            intrinsics = load_intrinsics()
            if mount is None:
                mount = U20CAM_MOUNT
        if isinstance(mount, str):
            mount = load_mount(mount)
        self.K = intrinsics
        self.names = tuple(k["joints"])
        self.body_names = [b["name"] for b in k["bodies"]]
        self.bodies = [dict(b, R0=quat_mat(b["quat"]), p0=np.array(b["pos"]),
                            joints=[dict(j, axis=np.array(j["axis"]), pos=np.array(j["pos"])) for j in b["joints"]])
                       for b in k["bodies"]]
        self.sites = {n: dict(s, R=quat_mat(s["quat"]), p=np.array(s["pos"])) for n, s in k["sites"].items()}
        self.feet = [dict(g, R=quat_mat(g["quat"]), p=np.array(g["pos"]), v=np.array(g["verts"])) for g in k["feet"].values()]
        self.mount = mount
        self.cam_site = self.sites["head_camera"]
        self.offsets = (0.0, 0.0, 0.0)            # pitch, yaw, roll in deg (from the mount fit)
        self.shift = np.zeros(3)                  # lens shift in the camera frame, m
        if mount is not None:
            self.cam_site = self.mount_site(mount)
            o = mount.get("offsets_deg") or {}
            self.offsets = (float(o.get("pitch", 0.0)), float(o.get("yaw", 0.0)), float(o.get("roll", 0.0)))
            self.shift = np.array(mount.get("shift_m") or (0.0, 0.0, 0.0), float)
        self.pitch_offset = math.radians(pitch_offset_deg)
        self._R_off = None
        self._key, self._pose = None, None

    # --- mount --------------------------------------------------------------------------------------------------
    def qpos0_pose(self):
        """Rotation and position of every body in the trunk_base frame at qpos0 (= the guide frame, in m)."""
        Rs, ps = [], []
        for b in self.bodies:
            if b["parent"] < 0:
                R, p = np.eye(3), np.zeros(3)
            else:
                R, p = Rs[b["parent"]] @ b["R0"], ps[b["parent"]] + Rs[b["parent"]] @ b["p0"]
            Rs.append(R)
            ps.append(p)
        return Rs, ps

    def mount_site(self, mount):
        """The camera site on its body from the guide-frame centre (mm) and axes (columns: optical axis, image left,
        image up) of the mount: R_b^T (p - x_b) at qpos0."""
        cad = mount["cad"]
        b = self.body_names.index(mount.get("body", "jaw_soft"))
        Rs, ps = self.qpos0_pose()
        pg = np.array(cad["centre_guide_mm"], float) / 1000.0
        Ag = np.array(cad.get("axes_guide", np.eye(3).tolist()), float)
        return dict(body=b, p=Rs[b].T @ (pg - ps[b]), R=Rs[b].T @ Ag)

    def set_mount_offsets(self, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0, shift_m=(0.0, 0.0, 0.0)):
        self.offsets = (float(pitch_deg), float(yaw_deg), float(roll_deg))
        self.shift = np.array(shift_m, float)
        self._R_off = None

    def set_pitch_offset(self, deg):
        self.pitch_offset = math.radians(deg)
        self._R_off = None

    def _offset(self):
        if self._R_off is None:
            p, y, r = self.offsets
            self._R_off = offset_rotation(p + math.degrees(self.pitch_offset), y, r)
        return self._R_off

    # --- kinematics ---------------------------------------------------------------------------------------------
    def place(self, q, gb=UPRIGHT):
        """World (= foot frame) rotation and position of every body for joints q (dict) and trunk gravity gb."""
        key = (tuple(round(float(q[n]), 6) for n in self.names), tuple(round(float(x), 6) for x in gb))
        if key == self._key:
            return self._pose
        Rs, ps = [], []
        for b in self.bodies:
            if b["parent"] < 0:
                R, p = trunk_rotation(gb), np.zeros(3)
            else:
                Rp, pp = Rs[b["parent"]], ps[b["parent"]]
                R, p = Rp @ b["R0"], pp + Rp @ b["p0"]
                for j in b["joints"]:  # hinge about axis through pos (body frame), angle relative to qpos0
                    Rj = axis_angle(j["axis"], float(q[j["name"]]) - j["qpos0"])
                    p = p + R @ (j["pos"] - Rj @ j["pos"])
                    R = R @ Rj
            Rs.append(R)
            ps.append(p)
        # the soles depend on the legs and the tilt only: a head-only change (head aiming) reuses the last height
        leg_key = (tuple(round(float(q[n]), 6) for n in self.names if "hip" in n or "knee" in n or "ankle" in n), key[1])
        if leg_key != getattr(self, "_leg_key", None):
            self._leg_key = leg_key
            self._zmin = min(float(((g["v"] @ g["R"].T + g["p"]) @ Rs[g["body"]].T + ps[g["body"]])[:, 2].min())
                             for g in self.feet)
        zmin = self._zmin
        ps = [p - np.array([0.0, 0.0, zmin]) for p in ps]
        self._key, self._pose = key, (Rs, ps)
        return Rs, ps

    def site_pose(self, name, q, gb=UPRIGHT):
        Rs, ps = self.place(q, gb)
        s = self.sites[name]
        return ps[s["body"]] + Rs[s["body"]] @ s["p"], Rs[s["body"]] @ s["R"]

    def camera_body_pose(self, q, gb=UPRIGHT):
        """(R, p) of the body that carries the camera; with camera_from_body this is camera_pose split in two (the
        mount fit keeps the body poses fixed while it varies the offsets)."""
        Rs, ps = self.place(q, gb)
        b = self.cam_site["body"]
        return Rs[b], ps[b]

    def camera_from_body(self, Rb, pb):
        s = self.cam_site
        Rc = Rb @ s["R"]
        return pb + Rb @ s["p"] + Rc @ self.shift, Rc @ self._offset()

    def camera_pose(self, q, gb=UPRIGHT):
        """(pos, R) of the camera in the foot frame; R columns = optical axis, image left, image up."""
        return self.camera_from_body(*self.camera_body_pose(q, gb))

    def mouth_tip(self, q, gb=UPRIGHT):
        return self.site_pose("mouth_tip", q, gb)[0]

    # --- projection ---------------------------------------------------------------------------------------------
    def _K(self, img_w, img_h):
        return self.K.scaled(img_w or self.K.width, img_h or self.K.height)

    def ray(self, u, v, q, gb=UPRIGHT, img_w=None, img_h=None):
        """(origin, unit direction) in the foot frame of pixel (u, v); direction None where the lens model does not
        invert."""
        pos, R = self.camera_pose(q, gb)
        d = self._K(img_w, img_h).ray(u, v)
        return pos, (None if d is None else R @ d)

    def pixel_to_ground_pose(self, u, v, pos, R, img_w=None, img_h=None, z=0.0):
        d = self._K(img_w, img_h).ray(u, v)
        if d is None:
            return None
        d = R @ d
        if d[2] >= -1e-9:
            return None
        t = (z - pos[2]) / d[2]
        if t <= 0:
            return None
        return float(pos[0] + t * d[0]), float(pos[1] + t * d[1])

    def pixel_to_ground(self, u, v, q, gb=UPRIGHT, img_w=None, img_h=None, z=0.0):
        """Foot-frame (x, y) where the pixel's ray meets the plane at height z; None if it does not hit ahead."""
        pos, R = self.camera_pose(q, gb)
        return self.pixel_to_ground_pose(u, v, pos, R, img_w, img_h, z)

    def horizon_v(self, u, q, gb=UPRIGHT, img_w=None, img_h=None):
        """Image row of the horizon at column u: pixels below it (larger v) look down at the floor. None when the
        camera looks straight up or down, or the horizon does not reach column u."""
        K = self._K(img_w, img_h)
        R = self.camera_pose(q, gb)[1]
        if abs(R[2, 2]) < 1e-6:
            return None
        if K.model == "pinhole":
            return float(K.cy + K.fy * (R[2, 0] - R[2, 1] * (u - K.cx) / K.fx) / R[2, 2])
        # distorted lens: bisection down the column (rows further down look further down); a column that is all
        # floor returns a row above the picture, one that is all above the horizon a row below it
        def dz(v):
            d = K.ray(u, v)
            return None if d is None else float((R @ d)[2])
        lo, hi = -0.25 * K.height, 1.25 * K.height
        flo, fhi = dz(lo), dz(hi)
        if flo is None:
            lo, flo = 0.0, dz(0.0)
        if fhi is None:
            hi, fhi = float(K.height), dz(float(K.height))
        if flo is None or fhi is None:
            return None
        if flo < 0 and fhi < 0:
            return float(lo)
        if flo >= 0 and fhi >= 0:
            return float(hi)
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            fm = dz(mid)
            if fm is None:
                return None
            if fm >= 0:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    def ground_to_pixel(self, x, y, q, gb=UPRIGHT, img_w=None, img_h=None, z=0.0):
        """Pixel (u, v) of the foot-frame point; None if it is behind the camera (it may lie outside the image)."""
        pos, R = self.camera_pose(q, gb)
        return self._K(img_w, img_h).project(R.T @ (np.array([x, y, z]) - pos))
