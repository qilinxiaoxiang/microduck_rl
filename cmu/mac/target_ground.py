"""Pixel of the MicroDuck head camera -> ground position relative to the robot (beak-pickup geometry).

Pure offline geometry on the as-built MuJoCo model (the one brain_sim_check.py uses: evaluate_delay.load with
static_gap.cfg, mass model 'asbuilt'). Masses do not matter here, only the kinematic chain trunk -> neck ->
head and the camera / mouth_tip sites exported from the community CAD. No hardware is touched.

Frames
------
foot frame   Origin on the floor vertically below the trunk_base body origin (between the feet), x forward,
             y left, z up. Yaw is defined zero: the trunk x axis projected on the floor is the foot-frame x axis.
             The trunk's roll and pitch come from gb, the gravity direction in the trunk frame that the runtime
             observes from the IMU (upright = [0, 0, -1]; gb = [sin p, -cos p sin r, -cos p cos r] with p the
             nose-down pitch and r the left-down roll, so trunk_to_foot = Ry(p) @ Rx(r)). The trunk height is
             the forward kinematics of the legs: the lowest sole-mesh vertex of either foot is put on z = 0,
             exactly as brain_sim_check.py's setup does for the simulation check.
camera frame The `head_camera` site of the model (body jaw_soft, the community camera position): x = optical
             axis (forward), y = image left, z = image up. camera_pose() returns (pos, R) with R's columns
             these three axes in the foot frame. The MJCF `camera` element shares the site position but its
             quat "0 0 -1 0" (180 deg about y) makes MuJoCo's camera, which looks along its own -z with +y up,
             look along the site's -x, i.e. backward into the head, with the image turned 90 deg. DuckGeometry
             therefore rewrites that camera's quat at load time so cam -z = site +x, cam +y = site +z,
             cam +x = site -y, and sets its fovy to the intrinsics' vertical FOV; the renderer then sees what
             the pinhole below describes. The ray for pixel (u, v) in the site frame is
             (1, -(u - cx) / fx, -(v - cy) / fy): u grows to the image right (-y), v grows downward (-z).
pixels       Continuous coordinates: the centre of column j, row i is (j + 0.5, i + 0.5); the image centre is
             (w / 2, h / 2). A centroid from a boolean mask is mean(cols) + 0.5, mean(rows) + 0.5.

Intrinsics (Raspberry Pi Camera Module 3 Wide, IMX708)
--------------------------------------------------------
The stream is 640x480 taken from the 2304x1296 full-sensor 2x2-binned mode: a centre 4:3 crop of 1728x1296
binned pixels scaled by 640 / 1728 = 480 / 1296 = 0.3704. Binning and the crop keep the angular scale per
sensor pixel, so the vertical FOV of the stream equals the full-sensor vertical FOV, 67 deg, and the pinhole
focal length in output pixels is

    f = (480 / 2) / tan(67 deg / 2) = 240 / tan(33.5 deg) = 362.6 px        (fx = fy, square pixels)

which is the datasheet lens (2.75 mm) over the binned pixel pitch (2 x 1.4 um) scaled by 0.3704:
2.75 / 0.0028 x 0.3704 = 363.8 px, within 0.3 %. The horizontal FOV of the 4:3 stream is then
2 atan(320 / 362.6) = 82.8 deg, not the 102 deg of the full 16:9 sensor (the crop discards 288 binned
columns on each side; the datasheet 102 deg also exceeds the pinhole 99 deg of the full width because the
wide lens barrel-distorts outward at the edge). Principal point: image centre (320, 240).
Cost of ignoring distortion: at the horizontal edge of the full sensor the real ray is about 1.5 deg outside
the pinhole ray; at the 4:3 crop edge (41 deg half angle) roughly 1 deg, at the vertical edge (33.5 deg)
less. One degree of ray error moves a floor hit by h / sin^2(theta) per radian (theta the depression angle):
with the camera 0.2 m up looking 40 deg down at a point 0.24 m away that is about 8 mm per degree, and
about 60 mm per degree for a point 0.8 m away seen 14 deg below horizontal. Near the image centre the
distortion is a few mm; the far, flat-angle part of the view is where it costs.
`Intrinsics` is a parameter of DuckGeometry so a measured calibration (fx, fy, cx, cy) can replace it.

Known gaps
----------
- Camera pose is the nominal CAD (community) `head_camera` site: the real module sits in the printed face with
  its own mounting error of a few degrees, which is the dominant error at range (1 deg of pitch ~ 10-20 mm
  at 0.5 m). No real calibration has been done yet.
- No lens distortion model; no measured intrinsics (the values above are from the datasheet FOV).
- Trunk height from feet-flat forward kinematics, so real servo sag (knee ~0.05 rad under load) and an IMU gb
  that is noisy or biased shift the estimate; with gb upright the model camera is 245.3 mm up at STAND and looks level (0.0 deg).
- Ground plane assumed flat and level under the feet.
One real-world calibration step that fixes the scale: put an orange object on the floor at a measured distance
on the robot's centreline (say 0.30 m from the point below the trunk), read the estimate, and absorb the ratio
into a camera pitch offset (`pitch_offset_deg`) or the focal length; two distances separate the two.

    ~/.venvs/microduck-dyn/bin/python target_ground.py            # prints STAND and head-pose variations
"""
import json
import math
import os
import sys
from dataclasses import dataclass

SIM_PYTHON = os.path.expanduser("~/.venvs/microduck-dyn/bin/python")
try:
    import mujoco
except ImportError:
    if __name__ == "__main__" and os.path.exists(SIM_PYTHON):
        os.execv(SIM_PYTHON, [SIM_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
    raise ImportError(f"target_ground needs the dyn venv: run under {SIM_PYTHON}")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WHOLE = os.path.join(HERE, "..", "whole")
for p in (os.path.join(HERE, "..", "pi"), os.path.join(WHOLE, "dynamics", "realgap-20261014"),
          os.path.join(WHOLE, "delay-repair-20260912")):
    if p not in sys.path:
        sys.path.insert(0, p)
import evaluate_delay as ev  # noqa: E402
from static_gap import cfg, setup, sole_points  # noqa: E402
from stand_pose import STAND_POSE  # noqa: E402

JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch",
          "head_pitch", "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
          "right_knee", "right_ankle")
UPRIGHT = (0.0, 0.0, -1.0)
# camera axes (MuJoCo convention: looks along -z, +y up) expressed in the head_camera site frame (x forward,
# y left, z up): cam x = -site y, cam y = site z, cam z = -site x
_CAM_IN_SITE = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole in continuous pixel coordinates of a width x height image."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480

    @classmethod
    def from_vfov(cls, vfov_deg=67.0, width=640, height=480):
        f = (height / 2) / math.tan(math.radians(vfov_deg) / 2)
        return cls(f, f, width / 2, height / 2, width, height)

    @property
    def vfov_deg(self):
        return math.degrees(2 * math.atan((self.height / 2) / self.fy))

    @property
    def hfov_deg(self):
        return math.degrees(2 * math.atan((self.width / 2) / self.fx))

    def scaled(self, width, height):
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Intrinsics(self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy, width, height)


PI_CAM3_WIDE_640x480 = Intrinsics.from_vfov(67.0, 640, 480)


def trunk_rotation(gb):
    """Trunk-to-foot rotation Ry(pitch) @ Rx(roll) from the trunk-frame gravity direction, zero yaw."""
    g = np.asarray(gb, float)
    g = g / np.linalg.norm(g)
    p = math.asin(float(np.clip(g[0], -1, 1)))
    r = math.atan2(-g[1], -g[2])
    cp, sp, cr, sr = math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Ry @ Rx


def gb_from_rotation(R):
    """Gravity in the trunk frame for a trunk-to-world rotation (what the IMU reports)."""
    return tuple(float(x) for x in np.asarray(R).reshape(3, 3).T @ np.array(UPRIGHT))


class DuckGeometry:
    """Camera and mouth-tip poses of the as-built MicroDuck model in the foot frame, and the floor projection."""

    def __init__(self, intrinsics=PI_CAM3_WIDE_640x480, model="asbuilt", pitch_offset_deg=0.0):
        self.K = intrinsics
        self.pitch_offset = math.radians(pitch_offset_deg)  # extra nose-down of the real camera vs CAD
        self.m, self.d, _, self.names, _ = ev.load(cfg(0.09, model=model, kp=200, voltage=5.0))
        assert tuple(self.names) == JOINTS, self.names
        m = self.m
        self.ji = np.array([m.actuator(n).trnid[0] for n in self.names])
        self.qa = m.jnt_qposadr[self.ji]
        self.trunk = m.body("trunk_base").id
        self.cid = m.camera("head_camera").id
        self.site = m.site("head_camera").id
        self.mouth = m.site("mouth_tip").id
        assert m.cam_bodyid[self.cid] == m.site_bodyid[self.site]
        assert np.allclose(m.cam_pos[self.cid], m.site_pos[self.site])
        m.cam_fovy[self.cid] = self.K.vfov_deg
        self._renderer = None
        self.set_pitch_offset(pitch_offset_deg)

    def set_pitch_offset(self, deg):
        """Re-aim the model camera: the site's x axis with the site's z up (module docstring), turned nose-down by deg
        (the calibration knob; pickup.py --calibrate fits it)."""
        self.pitch_offset = math.radians(deg)
        Rs = np.zeros(9)
        mujoco.mju_quat2Mat(Rs, self.m.site_quat[self.site])
        Rc = Rs.reshape(3, 3) @ _CAM_IN_SITE
        if self.pitch_offset:
            a = self.pitch_offset
            Rc = Rs.reshape(3, 3) @ np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0],
                                              [-math.sin(a), 0, math.cos(a)]]) @ _CAM_IN_SITE
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rc.ravel())
        self.m.cam_quat[self.cid] = q

    # ---- placement -------------------------------------------------------------------------------------
    def place(self, q, gb=UPRIGHT, xy=(0.0, 0.0)):
        """Put the model at the pose: trunk rotation from gb, joints from q, lowest sole vertex on z = 0.

        With xy = (0, 0) the world frame is the foot frame. Returns the trunk height."""
        pose = {n: float(q[n]) for n in self.names}
        R = trunk_rotation(gb)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.ravel())
        setup(self.m, self.d, self.names, pose, quat)
        zmin = np.vstack(sole_points(self.m, self.d))[:, 2].min()
        self.d.qpos[0:2] = xy
        self.d.qpos[2] -= zmin
        mujoco.mj_forward(self.m, self.d)
        return float(self.d.qpos[2])

    def camera_pose(self, q, gb=UPRIGHT):
        """(pos, R) of the camera in the foot frame; R columns = optical axis, image left, image up."""
        self.place(q, gb)
        Rs = self.d.site_xmat[self.site].reshape(3, 3).copy()
        if self.pitch_offset:
            a = self.pitch_offset
            Rs = Rs @ np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        return self.d.site_xpos[self.site].copy(), Rs

    def camera_depression_deg(self, q, gb=UPRIGHT):
        """Angle of the optical axis below horizontal."""
        _, R = self.camera_pose(q, gb)
        return math.degrees(math.asin(float(np.clip(-R[2, 0], -1, 1))))

    def mouth_tip(self, q, gb=UPRIGHT):
        self.place(q, gb)
        return self.d.site_xpos[self.mouth].copy()

    # ---- projection ------------------------------------------------------------------------------------
    def ray(self, u, v, q, gb=UPRIGHT, img_w=640, img_h=480):
        """Unit ray of pixel (u, v) and its origin, both in the foot frame."""
        K = self.K.scaled(img_w, img_h)
        pos, R = self.camera_pose(q, gb)
        d_site = np.array([1.0, -(u - K.cx) / K.fx, -(v - K.cy) / K.fy])
        d = R @ d_site
        return pos, d / np.linalg.norm(d)

    def pixel_to_ground(self, u, v, q, gb=UPRIGHT, img_w=640, img_h=480, z=0.0):
        """Foot-frame (x, y) where the pixel's ray meets the plane at height z; None if it does not hit ahead."""
        o, d = self.ray(u, v, q, gb, img_w, img_h)
        if d[2] >= -1e-9:
            return None
        t = (z - o[2]) / d[2]
        if t <= 0:
            return None
        p = o + t * d
        return float(p[0]), float(p[1])

    def ground_to_pixel(self, x, y, q, gb=UPRIGHT, img_w=640, img_h=480, z=0.0):
        """Pixel (u, v) of the foot-frame point (x, y, z); None if it is not in front of the camera.

        The pixel may lie outside the image; the caller checks 0 <= u < img_w, 0 <= v < img_h."""
        K = self.K.scaled(img_w, img_h)
        pos, R = self.camera_pose(q, gb)
        p = R.T @ (np.array([x, y, z]) - pos)  # site frame: x forward, y left, z up
        if p[0] <= 1e-9:
            return None
        return float(K.cx - K.fx * p[1] / p[0]), float(K.cy - K.fy * p[2] / p[0])

    def in_image(self, uv, img_w=640, img_h=480, margin=0):
        return uv is not None and margin <= uv[0] < img_w - margin and margin <= uv[1] < img_h - margin

    # ---- rendering (tests and offline pipeline checks) -------------------------------------------------
    def render(self, q, gb=UPRIGHT, spheres=(), trunk_xy=(0.0, 0.0), img_w=640, img_h=480):
        """RGB frame of the model camera. spheres: (x, y, z, radius, rgba) in world coordinates; the trunk
        stands at world trunk_xy, so world = foot frame shifted by trunk_xy."""
        self.place(q, gb, trunk_xy)
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (img_w, img_h):
            self._renderer = mujoco.Renderer(self.m, height=img_h, width=img_w)
        r = self._renderer
        r.update_scene(self.d, camera=self.cid)
        scn = r.scene
        for (x, y, z, radius, rgba) in spheres:
            g = scn.geoms[scn.ngeom]
            scn.ngeom += 1
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                                np.array([x, y, z], float), np.eye(3).ravel(), np.array(rgba, np.float32))
        return r.render().copy()


# ---- orange detection (same recipe as omx-local/locate.py orange_mask, in numpy: no cv2 in the dyn venv) ----
def rgb_to_hsv_cv(img):
    """OpenCV-scaled HSV (H 0..180, S 0..255, V 0..255) of an RGB uint8 image."""
    f = img.astype(np.float32)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx = f.max(-1)
    mn = f.min(-1)
    delta = mx - mn
    h = np.zeros_like(mx)
    nz = delta > 0
    rm, gm, bm = (mx == r) & nz, (mx == g) & nz & (mx != r), (mx == b) & nz & (mx != r) & (mx != g)
    h[rm] = (60 * (g - b)[rm] / delta[rm]) % 360
    h[gm] = 120 + 60 * (b - r)[gm] / delta[gm]
    h[bm] = 240 + 60 * (r - g)[bm] / delta[bm]
    s = np.where(mx > 0, 255 * delta / np.maximum(mx, 1e-6), 0)
    return np.stack([h / 2, s, mx], -1)


def orange_mask(img, lo=(5, 120, 120), hi=(22, 255, 255)):
    """Largest orange blob as a boolean mask (HSV in-range, 5x5 opening, largest component), or None."""
    from scipy import ndimage
    hsv = rgb_to_hsv_cv(img)
    mk = np.all((hsv >= np.array(lo)) & (hsv <= np.array(hi)), -1)
    mk = ndimage.binary_opening(mk, structure=np.ones((5, 5), bool))
    lab, n = ndimage.label(mk)
    if n < 1:
        return None
    sizes = ndimage.sum(mk, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def orange_centroid(img):
    """Continuous-pixel centroid (u, v) of the largest orange blob, or None."""
    mk = orange_mask(img)
    if mk is None:
        return None
    ys, xs = np.nonzero(mk)
    return float(xs.mean() + 0.5), float(ys.mean() + 0.5), mk


# ---- CLI ------------------------------------------------------------------------------------------------
def describe(geom, q, gb=UPRIGHT, img_w=640, img_h=480):
    pos, R = geom.camera_pose(q, gb)
    pix = {"centre": (img_w / 2, img_h / 2), "top_left": (0.5, 0.5), "top_right": (img_w - 0.5, 0.5),
           "bottom_left": (0.5, img_h - 0.5), "bottom_right": (img_w - 0.5, img_h - 0.5)}
    foot = {k: geom.pixel_to_ground(u, v, q, gb, img_w, img_h) for k, (u, v) in pix.items()}
    mouth = geom.mouth_tip(q, gb)
    return dict(neck_pitch=round(q["neck_pitch"], 3), head_pitch=round(q["head_pitch"], 3),
                gb=[round(float(x), 3) for x in gb],
                camera_xyz_mm=[round(float(x) * 1000, 1) for x in pos],
                camera_depression_deg=round(geom.camera_depression_deg(q, gb), 2),
                footprint_m={k: None if p is None else [round(p[0], 3), round(p[1], 3)] for k, p in foot.items()},
                mouth_tip_xyz_mm=[round(float(x) * 1000, 1) for x in mouth],
                mouth_tip_pixel=geom.ground_to_pixel(mouth[0], mouth[1], q, gb, img_w, img_h, z=mouth[2]))


def main():
    geom = DuckGeometry()
    K = geom.K
    print(json.dumps(dict(intrinsics=dict(fx=round(K.fx, 2), fy=round(K.fy, 2), cx=K.cx, cy=K.cy,
                                          vfov_deg=round(K.vfov_deg, 2), hfov_deg=round(K.hfov_deg, 2)))))
    print(json.dumps(dict(STAND=describe(geom, STAND_POSE))))
    for neck, head in [(0.349, 0.6), (0.349, 0.8), (0.349, 1.0), (0.0, 0.8), (0.6, 0.6), (-0.3, 1.0)]:
        q = dict(STAND_POSE, neck_pitch=neck, head_pitch=head)
        print(json.dumps(describe(geom, q)))
    tilt = trunk_rotation(UPRIGHT) @ np.array([[math.cos(0.1745), 0, math.sin(0.1745)], [0, 1, 0],
                                               [-math.sin(0.1745), 0, math.cos(0.1745)]])
    print(json.dumps(dict(trunk_pitched_10deg=describe(geom, dict(STAND_POSE, head_pitch=1.0), gb_from_rotation(tilt)))))


if __name__ == "__main__":
    main()
