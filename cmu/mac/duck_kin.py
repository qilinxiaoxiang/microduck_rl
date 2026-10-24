"""Head-camera floor projection for the MicroDuck in numpy only (no MuJoCo): the A7Z side of target_ground.py.

Same frames and conventions as target_ground.DuckGeometry (read its docstring): foot frame on the floor below the
trunk origin, x forward, y left, trunk roll and pitch from gb, the lowest sole vertex on z = 0; camera frame = the
`head_camera` site, x the optical axis, y image left, z image up; continuous pixel coordinates. The kinematic tree
comes from duck_kin.json (duck_kin_export.py, as-built model); test_duck_kin.py checks the two agree.

Intrinsics: the Radxa Camera 13M 214 (IMX214) as the A7Z streams it: full-sensor mode 4208 x 3120 scaled to 640 x 480
and turned a quarter clockwise, so the upright picture is 480 wide and 640 tall (a7z-bridge.mjs CAMERA_PIPELINE).
Radxa gives HFOV 66.2 deg along the sensor's long side and VFOV 51.6 deg along its short side
(docs.radxa.com/en/accessories/camera_13m_214). Upright, the 480 px width spans 51.6 deg and the 640 px height 66.2 deg:
fx = 240 / tan(25.8 deg) = 497 px, fy = 320 / tan(33.1 deg) = 491 px. Datasheet values, not a calibration: the pitch
offset (approach.py --calibrate) absorbs the mounting error, which dominates.
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UPRIGHT = (0.0, 0.0, -1.0)


class Intrinsics:
    """Pinhole in continuous pixel coordinates of a width x height image."""

    def __init__(self, fx, fy, cx, cy, width, height):
        self.fx, self.fy, self.cx, self.cy, self.width, self.height = fx, fy, cx, cy, width, height

    @classmethod
    def from_fov(cls, hfov_deg, vfov_deg, width, height):
        return cls((width / 2) / math.tan(math.radians(hfov_deg) / 2), (height / 2) / math.tan(math.radians(vfov_deg) / 2),
                   width / 2, height / 2, width, height)

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan((self.height / 2) / self.fy))

    def scaled(self, width, height):
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


RADXA_13M_UPRIGHT = Intrinsics.from_fov(51.6, 66.2, 480, 640)


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


def _ry(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


class DuckKin:
    """Camera and mouth-tip poses in the foot frame and the floor projection, from duck_kin.json."""

    def __init__(self, intrinsics=RADXA_13M_UPRIGHT, pitch_offset_deg=0.0, path=os.path.join(HERE, "duck_kin.json")):
        with open(path) as f:
            k = json.load(f)
        self.K = intrinsics
        self.names = tuple(k["joints"])
        self.bodies = [dict(b, R0=quat_mat(b["quat"]), p0=np.array(b["pos"]),
                            joints=[dict(j, axis=np.array(j["axis"]), pos=np.array(j["pos"])) for j in b["joints"]])
                       for b in k["bodies"]]
        self.sites = {n: dict(s, R=quat_mat(s["quat"]), p=np.array(s["pos"])) for n, s in k["sites"].items()}
        self.feet = [dict(g, R=quat_mat(g["quat"]), p=np.array(g["pos"]), v=np.array(g["verts"])) for g in k["feet"].values()]
        self.pitch_offset = math.radians(pitch_offset_deg)
        self._key, self._pose = None, None

    def set_pitch_offset(self, deg):
        self.pitch_offset = math.radians(deg)

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

    def camera_pose(self, q, gb=UPRIGHT):
        """(pos, R) of the camera in the foot frame; R columns = optical axis, image left, image up."""
        pos, R = self.site_pose("head_camera", q, gb)
        return pos, (R @ _ry(self.pitch_offset) if self.pitch_offset else R)

    def mouth_tip(self, q, gb=UPRIGHT):
        return self.site_pose("mouth_tip", q, gb)[0]

    def ray(self, u, v, q, gb=UPRIGHT, img_w=None, img_h=None):
        K = self.K.scaled(img_w or self.K.width, img_h or self.K.height)
        pos, R = self.camera_pose(q, gb)
        d = R @ np.array([1.0, -(u - K.cx) / K.fx, -(v - K.cy) / K.fy])
        return pos, d / np.linalg.norm(d)

    def pixel_to_ground(self, u, v, q, gb=UPRIGHT, img_w=None, img_h=None, z=0.0):
        """Foot-frame (x, y) where the pixel's ray meets the plane at height z; None if it does not hit ahead."""
        o, d = self.ray(u, v, q, gb, img_w, img_h)
        if d[2] >= -1e-9:
            return None
        t = (z - o[2]) / d[2]
        if t <= 0:
            return None
        return float(o[0] + t * d[0]), float(o[1] + t * d[1])

    def horizon_v(self, u, q, gb=UPRIGHT, img_w=None, img_h=None):
        """Image row of the horizon at column u: pixels below it (larger v) look down at the floor."""
        K = self.K.scaled(img_w or self.K.width, img_h or self.K.height)
        R = self.camera_pose(q, gb)[1]
        if abs(R[2, 2]) < 1e-6:
            return None
        return float(K.cy + K.fy * (R[2, 0] - R[2, 1] * (u - K.cx) / K.fx) / R[2, 2])

    def ground_to_pixel(self, x, y, q, gb=UPRIGHT, img_w=None, img_h=None, z=0.0):
        """Pixel (u, v) of the foot-frame point; None if it is behind the camera (it may lie outside the image)."""
        K = self.K.scaled(img_w or self.K.width, img_h or self.K.height)
        pos, R = self.camera_pose(q, gb)
        p = R.T @ (np.array([x, y, z]) - pos)
        if p[0] <= 1e-9:
            return None
        return float(K.cx - K.fx * p[1] / p[0]), float(K.cy - K.fy * p[2] / p[0])
