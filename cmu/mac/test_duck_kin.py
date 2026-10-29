"""duck_kin.DuckKin (numpy, A7Z) agrees with target_ground.DuckGeometry (MuJoCo) on random poses and tilts; the U20CAM
lens models (fisheye, radtan) invert exactly and match OpenCV; floor points round-trip through a distorted camera at an
offset mount within 2 mm; cam_calib's mount fit recovers injected offsets from noisy synthetic detections and its
intrinsics fit recovers a known camera from rendered chessboards.

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


# --- U20CAM: lens model, mount, calibration (numpy; the cv2 checks skip without OpenCV) ------------------------------
import copy  # noqa: E402
import json  # noqa: E402

import pytest  # noqa: E402

import approach  # noqa: E402
import cam_calib  # noqa: E402

FISHEYE = duck_kin.Intrinsics(352.0, 350.0, 318.4, 183.1, 640, 360, "fisheye", (0.06, -0.03, 0.012, -0.003))
RADTAN = duck_kin.Intrinsics(300.0, 301.0, 322.0, 178.0, 640, 360, "radtan", (-0.28, 0.09, 0.001, -0.0007, -0.012))


def u20_kin(K, pitch=0.0, yaw=0.0, roll=0.0, shift=(0.0, 0.0, 0.0)):
    m = copy.deepcopy(duck_kin.load_mount())
    m["offsets_deg"] = {"pitch": pitch, "yaw": yaw, "roll": roll}
    m["shift_m"] = list(shift)
    return duck_kin.DuckKin(K, mount=m)


def test_u20cam_default_loads_datasheet_and_cad_mount():
    kin = duck_kin.DuckKin()
    assert kin.K.source == "datasheet" and kin.K.model == "fisheye" and (kin.K.width, kin.K.height) == (640, 360)
    assert abs(kin.K.fx - 320 / math.radians(51)) < 0.01
    assert kin.mount["source"] == "cad" and kin.mount["fit"] is None
    # the camera centre at qpos0 is the CAD point in the guide frame (trunk_base frame at qpos0), axes level
    Rs, ps = kin.qpos0_pose()
    s = kin.cam_site
    assert np.allclose((ps[s["body"]] + Rs[s["body"]] @ s["p"]) * 1000, [86.0, -0.09, 131.114], atol=1e-6)
    assert np.allclose(Rs[s["body"]] @ s["R"], np.eye(3), atol=1e-9)
    why = approach.camera_ready(kin, 640, 360)
    assert why and "--intrinsics" in why
    # the old camera still projects exactly as before at the model's head_camera site
    assert np.allclose(duck_kin.DuckKin(duck_kin.RADXA_13M_UPRIGHT).cam_site["p"], KIN.sites["head_camera"]["p"])


def test_camera_ready_needs_fit_and_matching_intrinsics():
    kin = u20_kin(FISHEYE)
    assert "--mount" in approach.camera_ready(kin, 640, 360)
    kin.mount["fit"] = {"intrinsics": cam_calib.intrinsics_fingerprint(FISHEYE), "captures": "x.json"}
    assert approach.camera_ready(kin, 640, 360) is None
    assert "refit" in approach.camera_ready(duck_kin.DuckKin(RADTAN, mount=kin.mount), 640, 360)
    assert "480" in approach.camera_ready(kin, 480, 640)


@pytest.mark.parametrize("K", [FISHEYE, RADTAN], ids=["fisheye", "radtan"])
def test_lens_model_round_trip(K):
    rng = np.random.default_rng(3)
    for _ in range(300):
        p = np.array([1.0, rng.uniform(-1.1, 1.1), rng.uniform(-0.6, 0.6)])   # up to about 50 deg off axis
        uv = K.project(p)
        d = K.ray(*uv)
        assert np.allclose(d, p / np.linalg.norm(p), atol=1e-9), (p, uv, d)


@pytest.mark.parametrize("K", [FISHEYE, RADTAN], ids=["fisheye", "radtan"])
def test_lens_model_matches_opencv(K):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(4)
    P = np.stack([rng.uniform(-0.8, 0.8, 200), rng.uniform(-0.45, 0.45, 200), np.ones(200)], 1)   # OpenCV camera frame
    Kc = np.array([[K.fx, 0, K.cx - 0.5], [0, K.fy, K.cy - 0.5], [0, 0, 1]])
    if K.model == "fisheye":
        ref = cv2.fisheye.projectPoints(P.reshape(1, -1, 3), np.zeros(3), np.zeros(3), Kc, np.array(K.dist))[0].reshape(-1, 2)
    else:
        ref = cv2.projectPoints(P, np.zeros(3), np.zeros(3), Kc, np.array(K.dist))[0].reshape(-1, 2)
    ours = np.array([K.project(np.array([z, -x, -y])) for x, y, z in P]) - 0.5   # continuous -> OpenCV pixels
    assert np.abs(ours - ref).max() < 1e-6


@pytest.mark.parametrize("K", [FISHEYE, RADTAN], ids=["fisheye", "radtan"])
def test_floor_round_trip_with_distortion(K):
    """Known floor points through a distorted camera at a known (offset, shifted) mount and back: < 2 mm."""
    kin = u20_kin(K, pitch=2.0, yaw=-1.2, roll=0.8, shift=(0.002, -0.001, 0.0015))
    worst, n = 0.0, 0
    for q, gb in poses(15, seed=5):
        q = dict(q, head_pitch=float(np.clip(q["head_pitch"] + 0.6, 0.3, 1.3)))
        for x in np.linspace(0.12, 0.6, 7):
            for y in np.linspace(-0.2, 0.2, 5):
                uv = kin.ground_to_pixel(x, y, q, gb)
                if uv is None or not (0 <= uv[0] < 640 and 0 <= uv[1] < 360):
                    continue
                g = kin.pixel_to_ground(*uv, q, gb)
                worst = max(worst, math.hypot(g[0] - x, g[1] - y))
                n += 1
                h = kin.horizon_v(uv[0], q, gb)
                assert h is None or h < uv[1]   # a floor point lies below the horizon at its column
    assert n > 300 and worst < 0.002, (n, worst)


def test_horizon_row_looks_level():
    kin = u20_kin(FISHEYE, pitch=3.0, roll=2.0)
    for q, gb in poses(8, seed=6):
        for u in (40.0, 320.0, 600.0):
            v = kin.horizon_v(u, q, gb)
            if v is None or not 0 <= v <= 360:
                continue
            o, d = kin.ray(u, v, q, gb)
            assert abs(d[2]) < 1e-6, (u, v, d)


def synthetic_observations(kin_true, points, rng, radius=0.02, frames=12, px_noise=0.5, holdout=False):
    obs = []
    for label, (x, y) in points.items():
        r = math.hypot(x, y)
        bx, by = x * (r - radius) / r, y * (r - radius) / r      # the blob's bottom edge meets the floor here
        for _ in range(frames):
            q = dict(STAND_POSE, head_pitch=1.1 + float(rng.normal(0, 0.003)), neck_pitch=STAND_POSE["neck_pitch"] + float(rng.normal(0, 0.003)))
            p = float(rng.normal(0, 0.005))
            gb = (math.sin(p), 0.0, -math.cos(p))
            uv = kin_true.ground_to_pixel(bx, by, q, gb)
            assert uv is not None and 0 < uv[0] < 640 and 0 < uv[1] < 360, (label, uv)
            obs.append(dict(point=label, x=x, y=y, u=uv[0] + float(rng.normal(0, px_noise)),
                            v=uv[1] + float(rng.normal(0, px_noise)), w=640, h=360, q=q, gb=gb, holdout=holdout))
    return obs


def test_mount_fit_recovers_injected_offsets():
    rng = np.random.default_rng(7)
    true = dict(pitch=4.0, yaw=-2.0, roll=1.5)
    kin_true = u20_kin(FISHEYE, **true)
    pts = {f"p{i}": p for i, p in enumerate(cam_calib.MOUNT_POINTS)}
    obs = synthetic_observations(kin_true, pts, rng)
    hold = synthetic_observations(kin_true, {"hold": cam_calib.HOLDOUT}, rng, holdout=True)
    kin = u20_kin(FISHEYE)                                     # the CAD mount: zero offsets
    before = cam_calib.check_points(kin, hold, dict(pitch=0, yaw=0, roll=0, shift_m=[0, 0, 0]), 0.02)["hold"]
    fit = cam_calib.fit_mount(kin, obs, 0.02)
    assert abs(fit["pitch"] - true["pitch"]) < 0.3 and abs(fit["yaw"] - true["yaw"]) < 0.3 and abs(fit["roll"] - true["roll"]) < 0.5, fit
    assert fit["rms_mm"] < 3.0
    after = cam_calib.check_points(kin, hold, fit, 0.02)["hold"]
    assert math.hypot(*before) > 20 and math.hypot(*after) < 5, (before, after)
    # the fit leaves the kin's own offsets untouched; the shift is dropped when it is not determined
    assert kin.offsets == (0.0, 0.0, 0.0)
    fs = cam_calib.fit_mount(kin, obs, 0.02, fit_shift=True)
    assert abs(fs["pitch"] - true["pitch"]) < 0.5 and (not fs["shift_kept"] or max(map(abs, fs["shift_m"])) < 0.004)


def test_mount_refit_from_captures_writes_only_on_pass(tmp_path, monkeypatch):
    """cam_calib.py --mount --from: the fit from saved captures, the held-out gate and the intrinsics gate."""
    rng = np.random.default_rng(8)
    kin_true = u20_kin(FISHEYE, pitch=-3.0, yaw=1.0, roll=-1.0)
    obs = synthetic_observations(kin_true, {f"p{i}": p for i, p in enumerate(cam_calib.MOUNT_POINTS)}, rng)
    obs += synthetic_observations(kin_true, {"hold": cam_calib.HOLDOUT}, rng, holdout=True)
    cap = tmp_path / "cap.json"
    cap.write_text(json.dumps(dict(obs=obs, head=cam_calib.MOUNT_HEAD, obj_radius_m=0.02)))
    intr = tmp_path / "intr.json"
    intr.write_text(json.dumps(dict(source="calibration", model="fisheye", width=640, height=360, fx=FISHEYE.fx,
                                    fy=FISHEYE.fy, cx=FISHEYE.cx, cy=FISHEYE.cy, dist=list(FISHEYE.dist), date="test")))
    mount_src = tmp_path / "mount.json"
    mount_src.write_text(json.dumps(duck_kin.load_mount()))
    monkeypatch.setattr(duck_kin, "U20CAM_INTRINSICS", str(intr))
    monkeypatch.setattr(duck_kin, "U20CAM_MOUNT", str(mount_src))
    monkeypatch.setattr(duck_kin.load_intrinsics, "__defaults__", (str(intr),))
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["cam_calib.py", "--mount", "--from", str(cap), "--out", str(out)])
    cam_calib.main()
    m = json.loads(out.read_text())
    assert m["source"] == "fit" and m["fit"]["passed"] and m["fit"]["holdout_error_mm"] < 5
    assert abs(m["offsets_deg"]["pitch"] + 3.0) < 0.3
    kin = duck_kin.DuckKin(duck_kin.load_intrinsics(str(intr)), mount=str(out))
    assert approach.camera_ready(kin, 640, 360) is None


def test_intrinsics_from_synthetic_chessboards(tmp_path):
    """cam_calib --intrinsics on rendered chessboard frames of a known fisheye camera recovers it: floor points
    projected through the recovered model land within 2 mm of the truth over the picture."""
    pytest.importorskip("cv2")
    rng = np.random.default_rng(9)
    sets = []
    for rv, tv in cam_calib.synth_poses(18, rng):
        c = cam_calib.find_corners(cam_calib.render_board(FISHEYE, rv, tv, rng=rng))
        if c is not None:
            sets.append(c)
    assert len(sets) >= 12
    res = cam_calib.calibrate(sets, (640, 360))
    rec = cam_calib.intrinsics_record(res, (640, 360), len(sets), cam_calib.coverage(sets, (640, 360))[0])
    assert rec["rms_px"] < 0.3 and rec["model"] in ("fisheye", "radtan")
    path = tmp_path / "i.json"
    path.write_text(json.dumps(rec))
    K = duck_kin.load_intrinsics(str(path))
    kt, kr = u20_kin(FISHEYE), u20_kin(K)
    q = dict(STAND_POSE, head_pitch=1.1)
    worst = 0.0
    for u in np.linspace(20, 620, 9):
        for v in np.linspace(150, 350, 6):
            a, b = kt.pixel_to_ground(u, v, q), kr.pixel_to_ground(u, v, q)
            if a is not None and math.hypot(*a) < 0.6:
                worst = max(worst, math.hypot(a[0] - b[0], a[1] - b[1]))
    assert worst < 0.002, worst
