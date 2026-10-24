"""duck_kin.DuckKin (numpy, A7Z) agrees with target_ground.DuckGeometry (MuJoCo) on random poses and tilts.

    ~/.venvs/microduck-dyn/bin/python -m pytest -q test_duck_kin.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import duck_kin  # noqa: E402
import target_ground as tg  # noqa: E402
from stand_pose import STAND_POSE  # noqa: E402

K = tg.Intrinsics(497.0, 491.0, 240.0, 320.0, 480, 640)
GEOM = tg.DuckGeometry(intrinsics=K, pitch_offset_deg=3.0)
KIN = duck_kin.DuckKin(duck_kin.Intrinsics(497.0, 491.0, 240.0, 320.0, 480, 640), pitch_offset_deg=3.0)


def poses(n=25, seed=1):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        q = {k: float(v) + float(rng.uniform(-0.25, 0.25)) for k, v in STAND_POSE.items()}
        p, r = rng.uniform(-0.25, 0.25), rng.uniform(-0.15, 0.15)
        gb = (math.sin(p), -math.cos(p) * math.sin(r), -math.cos(p) * math.cos(r))
        yield q, gb


def test_camera_and_mouth_match():
    for q, gb in poses():
        p1, R1 = GEOM.camera_pose(q, gb)
        p2, R2 = KIN.camera_pose(q, gb)
        assert np.allclose(p1, p2, atol=2e-5), (p1, p2)
        assert np.allclose(R1, R2, atol=1e-5)
        assert np.allclose(GEOM.mouth_tip(q, gb), KIN.mouth_tip(q, gb), atol=2e-5)


def test_projection_round_trip_matches():
    for q, gb in poses(10, seed=2):
        for u, v in ((240, 500), (100, 600), (400, 450)):
            a, b = GEOM.pixel_to_ground(u, v, q, gb, 480, 640), KIN.pixel_to_ground(u, v, q, gb)
            assert (a is None) == (b is None)
            if a is not None:
                assert abs(a[0] - b[0]) < 1e-4 and abs(a[1] - b[1]) < 1e-4, (a, b)
                uv = KIN.ground_to_pixel(b[0], b[1], q, gb)
                assert abs(uv[0] - u) < 1e-6 and abs(uv[1] - v) < 1e-6
