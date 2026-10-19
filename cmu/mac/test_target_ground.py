"""Offline checks of target_ground.DuckGeometry on the as-built MuJoCo model.

  ~/.venvs/microduck-dyn/bin/python -m pytest test_target_ground.py -v -s

(a) ground_to_pixel -> pixel_to_ground round trip, (b) rendered orange spheres located through the same
pinhole, (c) the trunk pitched 10 deg with gb read from the model. One annotated frame is written to
RENDER_OUT (the scratchpad path below, or $TARGET_GROUND_RENDER).
"""
import math
import os

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
from PIL import Image, ImageDraw  # noqa: E402

from target_ground import UPRIGHT, DuckGeometry, gb_from_rotation, orange_centroid, trunk_rotation  # noqa: E402
from stand_pose import STAND_POSE  # noqa: E402  (target_ground puts ../pi on sys.path)

RENDER_OUT = os.environ.get("TARGET_GROUND_RENDER",
                            "/private/tmp/claude-501/-Users-xiangzheng-code-robotics/"
                            "8aa5eea5-9e56-4bdd-a766-365e15deebea/scratchpad/ground_render.png")
RADIUS = 0.014
ORANGE = (1.0, 0.4, 0.0, 1.0)
LOOK_FAR = dict(STAND_POSE, head_pitch=1.0)             # 37.3 deg down: floor from 0.15 m to 3 m
LOOK_NEAR = dict(STAND_POSE, neck_pitch=-0.3, head_pitch=1.0)  # 74.5 deg down: floor from 0.05 m to 0.28 m
# (x, y) on the floor in the foot frame, with the head pose that sees it
POSITIONS = [(0.10, 0.00, LOOK_NEAR), (0.20, 0.08, LOOK_NEAR), (0.30, -0.15, LOOK_FAR),
             (0.50, 0.30, LOOK_FAR), (0.65, -0.30, LOOK_FAR), (0.80, 0.00, LOOK_FAR)]


@pytest.fixture(scope="module")
def geom():
    return DuckGeometry()


def tolerance(x, y):
    return 0.010 + 0.05 * math.hypot(x, y)


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def test_camera_at_stand(geom):
    pos, R = geom.camera_pose(STAND_POSE)
    assert 0.24 < pos[2] < 0.25, pos
    assert abs(geom.camera_depression_deg(STAND_POSE)) < 0.5
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    # the model camera element now looks along the site x axis with the site z axis up
    geom.place(STAND_POSE)
    Rc = geom.d.cam_xmat[geom.cid].reshape(3, 3)
    assert np.allclose(-Rc[:, 2], R[:, 0], atol=1e-9) and np.allclose(Rc[:, 1], R[:, 2], atol=1e-9)
    mouth = geom.mouth_tip(STAND_POSE)
    assert mouth[0] > pos[0] and mouth[2] < pos[2]  # beak ahead of and below the lens
    assert geom.ground_to_pixel(*mouth[:2], STAND_POSE, z=mouth[2])[1] > 480  # never in the frame


def test_round_trip(geom):
    rng = np.random.default_rng(0)
    worst = 0.0
    for q, vlo in ((LOOK_FAR, 0.0), (STAND_POSE, 260.0)):  # at STAND only the rows below the horizon hit
        for _ in range(20):
            u, v = rng.uniform(0, 640), rng.uniform(vlo, 480)
            p = geom.pixel_to_ground(u, v, q)
            assert p is not None, (q["head_pitch"], u, v)
            uv = geom.ground_to_pixel(p[0], p[1], q)
            assert uv is not None and abs(uv[0] - u) < 0.01 and abs(uv[1] - v) < 0.01, (u, v, uv)
            p2 = geom.pixel_to_ground(*uv, q)
            err = math.hypot(p2[0] - p[0], p2[1] - p[1])
            worst = max(worst, err)
            assert err < 0.001, (p, p2)
    print(f"\nround trip worst error {worst * 1000:.4f} mm")
    assert geom.pixel_to_ground(320, 240, STAND_POSE) is None  # level optical axis never meets the floor
    assert geom.pixel_to_ground(320, 10, STAND_POSE) is None


def locate(geom, q, gb, x, y, trunk_xy=(0.0, 0.0), draw=None):
    img = geom.render(q, gb, spheres=[(x + trunk_xy[0], y + trunk_xy[1], RADIUS, RADIUS, ORANGE)], trunk_xy=trunk_xy)
    found = orange_centroid(img)
    assert found is not None, (x, y)
    u, v, mask = found
    est = geom.pixel_to_ground(u, v, q, gb, z=RADIUS)       # the blob centroid is the ball centre's image
    est0 = geom.pixel_to_ground(u, v, q, gb, z=0.0)         # the naive floor intersection, for the record
    pred = geom.ground_to_pixel(x, y, q, gb, z=RADIUS)
    assert est is not None and pred is not None
    err = math.hypot(est[0] - x, est[1] - y)
    err0 = math.hypot(est0[0] - x, est0[1] - y)
    print(f"  truth ({x:.3f}, {y:.3f})  pixel ({u:6.1f}, {v:6.1f}) predicted ({pred[0]:6.1f}, {pred[1]:6.1f})"
          f"  est ({est[0]:.3f}, {est[1]:.3f}) err {err * 1000:5.1f} mm (floor-plane {err0 * 1000:5.1f} mm)"
          f"  tol {tolerance(x, y) * 1000:.0f} mm  blob {int(mask.sum())} px")
    if draw is not None:
        ys, xs = np.nonzero(mask)
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        dr.rectangle([xs.min() - 3, ys.min() - 3, xs.max() + 3, ys.max() + 3], outline=(0, 255, 0), width=2)
        dr.line([u - 12, v, u + 12, v], fill=(0, 255, 0), width=1)
        dr.line([u, v - 12, u, v + 12], fill=(0, 255, 0), width=1)
        dr.ellipse([pred[0] - 4, pred[1] - 4, pred[0] + 4, pred[1] + 4], outline=(255, 0, 255), width=2)
        dr.text((8, 8), f"sphere r=14mm at foot-frame ({x:.2f}, {y:.2f}) m   centroid ({u:.1f}, {v:.1f}) px"
                        f"   estimate ({est[0]:.3f}, {est[1]:.3f}) m   error {err * 1000:.1f} mm", fill=(255, 255, 255))
        dr.text((8, 24), f"head_pitch {q['head_pitch']:.3f} neck_pitch {q['neck_pitch']:.3f}  gb {np.round(gb, 3)}"
                         f"  green = detection, magenta = predicted ball centre", fill=(255, 255, 255))
        os.makedirs(os.path.dirname(draw), exist_ok=True)
        im.save(draw)
    return err, (u, v), pred


def test_rendered_spheres(geom):
    print()
    for i, (x, y, q) in enumerate(POSITIONS):
        pred = geom.ground_to_pixel(x, y, q, z=RADIUS)
        r_px = RADIUS * geom.K.fx / np.linalg.norm(np.array([x, y, RADIUS]) - geom.camera_pose(q)[0])
        assert geom.in_image(pred, margin=r_px + 2), (x, y, pred)
        err, uv, pred = locate(geom, q, UPRIGHT, x, y, draw=RENDER_OUT if i == 3 else None)
        # silhouette centroid vs projected centre: shading and the highlight shift it by a fraction of the radius
        assert abs(uv[0] - pred[0]) < 2 + 0.1 * r_px and abs(uv[1] - pred[1]) < 2 + 0.1 * r_px, (uv, pred, r_px)
        assert err < tolerance(x, y), (x, y, err)
    assert os.path.exists(RENDER_OUT)


def test_tilted_trunk(geom):
    # pitch the trunk 10 deg nose-down in the model and read gb back from its orientation
    q = LOOK_FAR
    geom.place(q)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_y(10.0).ravel())
    geom.d.qpos[3:7] = quat
    mujoco.mj_forward(geom.m, geom.d)
    gb = gb_from_rotation(geom.d.xmat[geom.trunk].reshape(3, 3))
    assert abs(gb[0] - math.sin(math.radians(10))) < 1e-9 and abs(gb[1]) < 1e-9
    assert np.allclose(trunk_rotation(gb), rot_y(10.0), atol=1e-9)
    d0 = geom.camera_depression_deg(q)
    d1 = geom.camera_depression_deg(q, gb)
    print(f"\ncamera depression {d0:.2f} deg upright -> {d1:.2f} deg with the trunk pitched 10 deg;"
          f" camera height {geom.camera_pose(q)[0][2] * 1000:.1f} -> {geom.camera_pose(q, gb)[0][2] * 1000:.1f} mm")
    assert abs(d1 - d0 - 10.0) < 0.3
    trunk_xy = (0.12, -0.05)  # the robot stands away from the world origin: estimates are relative to it
    for x, y in [(0.25, -0.10), (0.40, 0.20), (0.60, 0.00)]:
        pred = geom.ground_to_pixel(x, y, q, gb, z=RADIUS)
        assert geom.in_image(pred, margin=10), (x, y, pred)
        err, uv, pred = locate(geom, q, gb, x, y, trunk_xy=trunk_xy)
        assert err < tolerance(x, y), (x, y, err)
    # the same pixel read with an upright gb lands elsewhere: the tilt matters and is used
    uv = geom.ground_to_pixel(0.40, 0.0, q, gb, z=0.0)
    wrong = geom.pixel_to_ground(*uv, q, UPRIGHT)
    assert abs(wrong[0] - 0.40) > 0.05, wrong
