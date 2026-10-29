"""Calibrate the MicroDuck head camera (Innomaker U20CAM, D-023): lens intrinsics, then the mount on the head.

Writes the two files duck_kin.py projects with (procedure with the exact commands: CAMERA.md, "U20CAM calibration"):
  u20cam_intrinsics.json   --intrinsics: chessboard frames (shown on a laptop or iPad screen; the square size does not
                           matter) from the shared-memory camera on the A7Z (a7z_cam.py) or from image files, fitted
                           with cv2.calibrateCamera (radtan: k1 k2 p1 p2 k3) and cv2.fisheye.calibrate (k1..k4); the model
                           with the lower reprojection RMS is kept. Written only when the RMS is under RMS_OK_PX.
  u20cam_mount.json        --mount: robot standing on stand61 (walk_wired, Shawn guards), head at a fixed look pose,
                           the orange object on marked floor points; each detection is projected with the joints and
                           gravity logged at its capture time and the residual pitch / yaw / roll of the camera (and with
                           --fit-shift a lens shift, kept only when well determined) are fitted by least squares. One
                           held-out point checks the fit; written only when its error is under HOLDOUT_OK_MM.
Results go next to this file. On the A7Z that is ~/rt/microduck-sim/guide/firmware/mac/: copy them back into the repo
(CAMERA.md), since install_teleop.sh copies the repo's files to the A7Z.

  python cam_calib.py --board-png board.png                         chessboard picture to show full screen
  python cam_calib.py --intrinsics                                  A7Z: frames from the camera daemon, then the fit
  python cam_calib.py --intrinsics --images DIR/*.png               anywhere: the fit from saved frames
  python cam_calib.py --synth DIR                                   synthetic chessboard frames (dry run, tests)
  python cam_calib.py --mount                                       A7Z, walk_wired on stand61: place, capture, fit
  python cam_calib.py --mount --from ~/duck-logs/u20cam-mount-*.json   anywhere: refit saved captures
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import duck_kin  # noqa: E402

RMS_OK_PX = 0.5
HOLDOUT_OK_MM = 10.0
PATTERN = (9, 6)            # inner corners (columns, rows): a 10 x 7 board
N_FRAMES = 25
MIN_MOVE_PX = 25.0          # a new frame is kept only when the board moved this much (mean corner shift)
# mount fit: points in the foot frame (m), x ahead of the point between the feet, y left; the last one is held out
MOUNT_POINTS = ((0.17, 0.0), (0.25, 0.0), (0.35, 0.0), (0.25, 0.08), (0.25, -0.08))
HOLDOUT = (0.24, -0.06)
MOUNT_HEAD = {"head_pitch": 1.1, "head_yaw": 0.0}   # U20CAM datasheet model on STAND: the floor from 0.13 to 0.60 m
CAPTURE_N, CAPTURE_S = 15, 12.0
SHIFT_STD_MAX_MM = 3.0      # a fitted lens shift is kept only when every component is this well determined
EDGE_PX = 2                 # a blob within this of the picture edge is cut: not a capture


# --- intrinsics ----------------------------------------------------------------------------------------------------
def board_image(pattern=PATTERN, square_px=100, margin_px=60):
    cols, rows = pattern[0] + 1, pattern[1] + 1
    img = np.full((rows * square_px + 2 * margin_px, cols * square_px + 2 * margin_px), 255, np.uint8)
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                y, x = margin_px + j * square_px, margin_px + i * square_px
                img[y:y + square_px, x:x + square_px] = 0
    return img


def object_points(pattern=PATTERN):
    g = np.zeros((pattern[0] * pattern[1], 3), np.float64)
    g[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    return g


def find_corners(img, pattern=PATTERN):
    """Inner corners (N x 2, OpenCV pixel coordinates) or None."""
    import cv2
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, c = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
                                            | cv2.CALIB_CB_ACCURACY)
        if ok:
            return c.reshape(-1, 2).astype(np.float64)
    ok, c = cv2.findChessboardCorners(gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    c = cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4))
    return c.reshape(-1, 2).astype(np.float64)


def coverage(corner_sets, size, cells=4):
    """Fraction of a cells x cells grid of the picture that holds at least one corner (the lens edge needs corners)."""
    w, h = size
    hit = np.zeros((cells, cells), bool)
    for c in corner_sets:
        i = np.clip((c[:, 1] / h * cells).astype(int), 0, cells - 1)
        j = np.clip((c[:, 0] / w * cells).astype(int), 0, cells - 1)
        hit[i, j] = True
    return float(hit.mean()), hit


def calibrate(corner_sets, size, pattern=PATTERN):
    """Both lens models on the same corners: {"radtan": {...}, "fisheye": {...}}, each with rms_px (sqrt of the mean
    squared corner error over all corners), K, dist and per-frame rms; a model that fails to converge has "error"."""
    import cv2
    obj = object_points(pattern)
    objs = [obj.astype(np.float32) for _ in corner_sets]
    imgs = [c.astype(np.float32) for c in corner_sets]
    out = {}
    try:
        _, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, size, None, None)
        errs = [np.linalg.norm(cv2.projectPoints(obj, r, t, K, D)[0].reshape(-1, 2) - c, axis=1)
                for r, t, c in zip(rv, tv, corner_sets)]
        out["radtan"] = dict(K=K, dist=D.ravel()[:5].tolist(), rms_px=float(np.sqrt(np.mean(np.concatenate(errs) ** 2))),
                             per_frame=[float(np.sqrt(np.mean(e ** 2))) for e in errs])
    except cv2.error as e:
        out["radtan"] = dict(error=str(e).strip().splitlines()[-1])
    try:
        fo = [obj.reshape(1, -1, 3) for _ in corner_sets]   # (1, N, 3): the (N, 1, 3) layout fails in OpenCV 5
        fi = [c.reshape(1, -1, 2) for c in corner_sets]
        K0 = np.array([[size[0] / 2 / math.radians(51.0), 0, size[0] / 2], [0, size[0] / 2 / math.radians(51.0), size[1] / 2],
                       [0, 0, 1]])
        fl = lambda n: getattr(cv2.fisheye, "CALIB_" + n, None) or getattr(cv2, "CALIB_" + n)  # noqa: E731  OpenCV 4 / 5
        flags = fl("RECOMPUTE_EXTRINSIC") | fl("FIX_SKEW") | fl("USE_INTRINSIC_GUESS")
        _, K, D, rv, tv = cv2.fisheye.calibrate(fo, fi, size, K0, np.zeros((4, 1)), flags=flags,
                                                criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-10))
        errs = [np.linalg.norm(cv2.fisheye.projectPoints(obj.reshape(1, -1, 3), r, t, K, D)[0].reshape(-1, 2) - c, axis=1)
                for r, t, c in zip(rv, tv, corner_sets)]
        out["fisheye"] = dict(K=K, dist=D.ravel()[:4].tolist(), rms_px=float(np.sqrt(np.mean(np.concatenate(errs) ** 2))),
                              per_frame=[float(np.sqrt(np.mean(e ** 2))) for e in errs])
    except cv2.error as e:
        out["fisheye"] = dict(error=str(e).strip().splitlines()[-1])
    return out


def intrinsics_record(res, size, n_frames, cover, pattern=PATTERN, frames_from=None):
    """The u20cam_intrinsics.json dict for the better model (lower RMS), in duck_kin's continuous pixel convention."""
    ok = {k: v for k, v in res.items() if "rms_px" in v}
    for k, v in ok.items():   # a polynomial that folds back inside the picture cannot be inverted there
        K = v["K"]
        Ki = duck_kin.Intrinsics(K[0, 0], K[1, 1], K[0, 2] + 0.5, K[1, 2] + 0.5, size[0], size[1], k, v["dist"])
        v["folds_in_picture"] = any(Ki.ray(u, w) is None for u in (0.0, size[0] / 2, float(size[0]))
                                    for w in (0.0, size[1] / 2, float(size[1])))
    if any(not v["folds_in_picture"] for v in ok.values()):
        ok = {k: v for k, v in ok.items() if not v["folds_in_picture"]}
    if not ok:
        return None
    best = min(ok, key=lambda k: ok[k]["rms_px"])
    K = ok[best]["K"]
    return {
        "camera": "Innomaker U20CAM-720P (D-023), as a7z_cam.py publishes it: 1280 x 720 MJPEG shrunk to 640 x 360 and "
                  "turned 180 deg to upright",
        "source": "calibration",
        "model": best,
        "width": int(size[0]), "height": int(size[1]),
        "fx": round(float(K[0, 0]), 4), "fy": round(float(K[1, 1]), 4),
        "cx": round(float(K[0, 2]) + 0.5, 4), "cy": round(float(K[1, 2]) + 0.5, 4),
        "dist": [round(float(d), 8) for d in ok[best]["dist"]],
        "rms_px": round(ok[best]["rms_px"], 4),
        "frames": int(n_frames),
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pattern_inner_corners": list(pattern),
        "coverage_4x4": round(cover, 3),
        "compared": {k: ({"rms_px": round(v["rms_px"], 4), "folds_in_picture": v.get("folds_in_picture")} if "rms_px" in v
                         else {"error": v["error"]}) for k, v in res.items()},
        "frames_from": frames_from,
        "pixel_convention": "continuous: the centre of pixel column j, row i is (j + 0.5, i + 0.5); OpenCV's cx, cy plus 0.5",
    }


def collect_shm(pattern, n, timeout_s, save_dir):
    """Frames from the camera daemon with the board found and moved since the last kept one; each kept frame is saved."""
    import cv2
    from approach import ShmCamera
    cam = ShmCamera(require_calibration=False)
    kept, sets, last_idx, size = [], [], None, None
    os.makedirs(save_dir, exist_ok=True)
    t_end = time.time() + timeout_s
    t_hint = 0.0
    try:
        while len(sets) < n and time.time() < t_end:
            fr = cam.latest()
            if fr is None or fr[2] == last_idx:
                time.sleep(0.05)
                continue
            last_idx = fr[2]
            img = fr[0].copy()
            size = (img.shape[1], img.shape[0])
            c = find_corners(img, pattern)
            if c is None:
                if time.time() - t_hint > 3:
                    print(f"  no board in view ({len(sets)}/{n} kept)", flush=True)
                    t_hint = time.time()
                continue
            if sets and min(np.mean(np.linalg.norm(c - s, axis=1)) for s in sets) < MIN_MOVE_PX:
                continue
            sets.append(c)
            path = os.path.join(save_dir, f"board_{len(sets):02d}.png")
            cv2.imwrite(path, img)
            kept.append(path)
            cover, hit = coverage(sets, size)
            empty = [f"{'top' if i == 0 else 'bottom' if i == 3 else 'middle'}-{'left' if j == 0 else 'right' if j == 3 else 'centre'}"
                     for i in range(4) for j in range(4) if not hit[i, j] and (i in (0, 3) or j in (0, 3))]
            print(f"  kept {len(sets)}/{n} ({path}); picture covered {cover:.0%}"
                  + (f"; still empty near: {', '.join(sorted(set(empty))[:4])}" if empty else ""), flush=True)
    finally:
        cam.close()
    return sets, size, kept


def run_intrinsics(args):
    import cv2
    pattern = tuple(int(v) for v in args.pattern.lower().split("x"))
    if args.images:
        files = sorted(f for g in args.images for f in (glob.glob(os.path.expanduser(g)) or [g]))
        sets, size, used = [], None, []
        for f in files:
            img = cv2.imread(f)
            if img is None:
                print(f"  {f}: not an image")
                continue
            if size is not None and (img.shape[1], img.shape[0]) != size:
                sys.exit(f"{f}: {img.shape[1]} x {img.shape[0]}, the others are {size[0]} x {size[1]}")
            size = (img.shape[1], img.shape[0])
            c = find_corners(img, pattern)
            print(f"  {f}: {'board found' if c is not None else 'no board'}")
            if c is not None:
                sets.append(c)
                used.append(f)
        frames_from = os.path.dirname(os.path.abspath(used[0])) if used else None
    else:
        save = os.path.expanduser(args.save_dir or f"~/duck-logs/u20cam-calib-{time.strftime('%m%d-%H%M%S')}")
        print(f"Show the {pattern[0]} x {pattern[1]} board full screen and move it slowly: near and far, tilted, into "
              f"every corner of the picture. Frames go to {save}", flush=True)
        sets, size, used = collect_shm(pattern, args.n, args.timeout, save)
        frames_from = save
    if len(sets) < 8:
        sys.exit(f"only {len(sets)} frames with the board found: need at least 8 (aim for {N_FRAMES})")
    cover, _ = coverage(sets, size)
    res = calibrate(sets, size, pattern)
    for k, v in res.items():
        print(f"  {k:8s} " + (f"rms {v['rms_px']:.3f} px, fx {v['K'][0, 0]:.1f} fy {v['K'][1, 1]:.1f} "
                              f"cx {v['K'][0, 2] + 0.5:.1f} cy {v['K'][1, 2] + 0.5:.1f}, dist "
                              + " ".join(f"{d:+.4f}" for d in v["dist"]) if "rms_px" in v else f"failed: {v['error']}"))
    rec = intrinsics_record(res, size, len(sets), cover, pattern, frames_from)
    if rec is None:
        sys.exit("neither lens model converged")
    worst = max(res[rec["model"]]["per_frame"])
    print(f"kept model: {rec['model']}, rms {rec['rms_px']:.3f} px over {len(sets)} frames (worst frame {worst:.2f} px), "
          f"picture covered {cover:.0%}")
    out = args.out or duck_kin.U20CAM_INTRINSICS
    if rec["rms_px"] >= RMS_OK_PX and not args.force:
        cand = out.replace(".json", ".candidate.json")
        json.dump(rec, open(cand, "w"), indent=1)
        sys.exit(f"FAIL: rms {rec['rms_px']:.3f} px is not under {RMS_OK_PX}: not written to {out} (candidate in {cand}). "
                 f"Retake with sharper, steadier frames (drop blurred ones), or --force")
    json.dump(rec, open(out, "w"), indent=1)
    print(f"PASS: wrote {out}" + ("" if rec["rms_px"] < RMS_OK_PX else " (forced)"))
    if size != (rec["width"], rec["height"]) or size != (640, 360):
        print(f"note: frames are {size[0]} x {size[1]}; a7z_cam.py publishes 640 x 360")
    return rec


# --- synthetic chessboard frames (dry run and tests) ---------------------------------------------------------------
def render_board(K, rvec, tvec, pattern=PATTERN, square=1.0, ss=2, rng=None):
    """Greyscale picture of the board seen by camera K (duck_kin.Intrinsics, OpenCV pose rvec/tvec of the board in
    the camera frame, board units = squares): every pixel's ray from the lens model, cut with the board plane."""
    import cv2
    w, h = K.width, K.height
    off = (np.arange(ss) + 0.5) / ss
    jj, ii = np.meshgrid(np.arange(w), np.arange(h))
    acc = np.zeros((h, w))
    Kc = np.array([[K.fx, 0, K.cx - 0.5], [0, K.fy, K.cy - 0.5], [0, 0, 1]])
    R = cv2.Rodrigues(np.asarray(rvec, float))[0]
    t = np.asarray(tvec, float).ravel()
    n = R[:, 2]
    for dy in off:
        for dx in off:
            px = np.stack([jj + dx - 0.5, ii + dy - 0.5], -1).reshape(-1, 1, 2).astype(np.float64)
            if K.model == "fisheye":
                und = cv2.fisheye.undistortPoints(px, Kc, np.array(K.dist).reshape(4, 1)).reshape(-1, 2)
            elif K.model == "radtan":
                und = cv2.undistortPointsIter(px, Kc, np.array(K.dist), None, None,
                                              (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)).reshape(-1, 2)
            else:
                und = np.stack([(px[:, 0, 0] - Kc[0, 2]) / K.fx, (px[:, 0, 1] - Kc[1, 2]) / K.fy], -1)
            d = np.concatenate([und, np.ones((len(und), 1))], 1)
            lam = (t @ n) / (d @ n)
            P = d * lam[:, None]
            b = (P - t) @ R              # board coordinates (columns of R are the board axes)
            s, r = b[:, 0] + 1.0, b[:, 1] + 1.0   # inner corner (0, 0) is one square in from the outer corner
            cols, rows = pattern[0] + 1, pattern[1] + 1
            inside = (lam > 0) & (s >= -0.6) & (s < cols + 0.6) & (r >= -0.6) & (r < rows + 0.6)
            sq = (s >= 0) & (s < cols) & (r >= 0) & (r < rows)
            black = sq & ((np.floor(s).astype(int) + np.floor(r).astype(int)) % 2 == 0)
            val = np.where(inside, np.where(black, 20.0, 235.0), 128.0)
            acc += val.reshape(h, w)
    img = acc / (ss * ss)
    if rng is not None:
        img = img + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def synth_poses(n, rng, pattern=PATTERN):
    """Board poses that fill the picture: distance 7-14 squares, tilts up to 40 deg, offsets out to the edges."""
    poses = []
    for i in range(n):
        z = rng.uniform(8.0, 14.0)
        ax = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(-0.25, 0.25)])
        ang = rng.uniform(0.15, 0.7)
        rvec = ax / np.linalg.norm(ax) * ang
        cx, cy = rng.uniform(-0.9, 0.9) * z * 0.9, rng.uniform(-0.8, 0.8) * z * 0.45
        import cv2
        R = cv2.Rodrigues(rvec)[0]
        centre = R @ np.array([(pattern[0] - 1) / 2, (pattern[1] - 1) / 2, 0.0])
        poses.append((rvec, np.array([cx, cy, z]) - centre))
    return poses


def run_synth(args):
    import cv2
    pattern = tuple(int(v) for v in args.pattern.lower().split("x"))
    K = duck_kin.Intrinsics(330.0, 331.0, 322.0, 178.5, 640, 360, "fisheye", (0.05, -0.02, 0.01, -0.004), "synthetic")
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.synth, exist_ok=True)
    n_ok = 0
    for i, (rv, tv) in enumerate(synth_poses(args.n, rng, pattern)):
        img = render_board(K, rv, tv, pattern, rng=rng)
        cv2.imwrite(os.path.join(args.synth, f"synth_{i:02d}.png"), img)
        n_ok += 1
    truth = dict(fx=K.fx, fy=K.fy, cx=K.cx, cy=K.cy, model=K.model, dist=list(K.dist), width=640, height=360,
                 pixel_convention="continuous")
    json.dump(truth, open(os.path.join(args.synth, "truth.json"), "w"), indent=1)
    print(f"wrote {n_ok} synthetic frames and truth.json to {args.synth}")


# --- mount ---------------------------------------------------------------------------------------------------------
def object_centre(kin, u, v, pos, R, w, h, radius):
    """approach.Approach.project's rule: the blob's bottom edge meets the floor; the object's centre lies `radius`
    further out along the ground direction from the foot-frame origin."""
    xy = kin.pixel_to_ground_pose(u, v, pos, R, w, h)
    if xy is None:
        return None
    r = math.hypot(*xy)
    if r < 1e-6:
        return None
    return xy[0] * (r + radius) / r, xy[1] * (r + radius) / r


def toe_x(kin, q, gb=duck_kin.UPRIGHT):
    """Foot-frame x of the front of the soles (the lowest 2 mm of the sole meshes) at this pose."""
    Rs, ps = kin.place(q, gb)
    xs = []
    for g in kin.feet:
        v = (g["v"] @ g["R"].T + g["p"]) @ Rs[g["body"]].T + ps[g["body"]]
        xs.append(float(v[v[:, 2] < v[:, 2].min() + 0.002][:, 0].max()))
    return min(xs)


def fit_mount(kin, obs, radius=0.02, fit_shift=False, iters=60):
    """Least-squares camera offsets from detections of the object at known floor points.

    obs: dicts with u, v (blob bottom-centre pixel), w, h (picture size), q, gb (state at capture), x, y (object centre,
    foot frame, m), point (label). Each point weighs the same however many frames it has. Returns a dict: pitch, yaw,
    roll (deg), shift_m, std (same units: deg, mm), residuals per point (mm), rms_mm, shift_kept."""
    for o in obs:
        if "Rb" not in o:
            o["Rb"], o["pb"] = kin.camera_body_pose(o["q"], o["gb"])
    count = {}
    for o in obs:
        count[o["point"]] = count.get(o["point"], 0) + 1
    wts = np.array([1.0 / math.sqrt(count[o["point"]]) for o in obs])
    saved = (kin.offsets, kin.shift.copy())

    def predict(p):
        kin.set_mount_offsets(p[0], p[1], p[2], (np.array(p[3:6]) / 1000.0) if len(p) > 3 else (0.0, 0.0, 0.0))
        out = []
        for o in obs:
            pos, R = kin.camera_from_body(o["Rb"], o["pb"])
            out.append(object_centre(kin, o["u"], o["v"], pos, R, o["w"], o["h"], radius))
        return out

    def resid(p):
        r = []
        for o, c, wt in zip(obs, predict(p), wts):
            r += [0.5, 0.5] if c is None else [wt * (c[0] - o["x"]), wt * (c[1] - o["y"])]
        return np.array(r)

    def solve(n):
        p = np.zeros(n)
        r = resid(p)
        lam = 1e-3
        for _ in range(iters):
            J = np.empty((len(r), n))
            for k in range(n):
                h = 0.01 if k < 3 else 0.1   # deg / mm
                dp = np.zeros(n)
                dp[k] = h
                J[:, k] = (resid(p + dp) - r) / h
            A, g = J.T @ J, J.T @ r
            while True:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
                rn = resid(p + step)
                if rn @ rn < r @ r:
                    p, r, lam = p + step, rn, max(lam / 3, 1e-9)
                    break
                lam *= 4
                if lam > 1e9:
                    break
            if lam > 1e9 or np.max(np.abs(step)) < 1e-6:
                break
        # covariance from the per-point mean residuals (frames of one point share most of their error)
        npt = len(count)
        dof = max(1, 2 * npt - n)
        means = point_residuals(p)
        s2 = sum(m[0] ** 2 + m[1] ** 2 for m in means.values()) / 1e6 / dof
        Jp = J / np.repeat(wts, 2)[:, None]  # unweighted Jacobian per frame -> per point by averaging rows
        rows = {}
        for i, o in enumerate(obs):
            rows.setdefault(o["point"], []).append(Jp[2 * i:2 * i + 2])
        Jm = np.concatenate([np.mean(v, axis=0) for v in rows.values()])
        try:
            cov = np.linalg.inv(Jm.T @ Jm) * s2
        except np.linalg.LinAlgError:
            cov = np.full((n, n), np.inf)
        return p, cov, means

    def point_residuals(p):
        res = {}
        for o, c in zip(obs, predict(p)):
            res.setdefault(o["point"], []).append((np.nan, np.nan) if c is None else (c[0] - o["x"], c[1] - o["y"]))
        return {k: (float(np.nanmean([e[0] for e in v]) * 1000), float(np.nanmean([e[1] for e in v]) * 1000))
                for k, v in res.items()}

    p, cov, means = solve(6 if fit_shift else 3)
    std = np.sqrt(np.abs(np.diag(cov)))
    shift_kept = False
    if fit_shift:
        if np.all(std[3:] < SHIFT_STD_MAX_MM):
            shift_kept = True
        else:
            p, cov, means = solve(3)
            std = np.sqrt(np.abs(np.diag(cov)))
    kin.set_mount_offsets(*saved[0], saved[1])
    errs = [math.hypot(*m) for m in means.values()]
    return dict(pitch=float(p[0]), yaw=float(p[1]), roll=float(p[2]),
                shift_m=[float(v) / 1000 for v in p[3:6]] if shift_kept else [0.0, 0.0, 0.0],
                std=[float(s) for s in std], shift_kept=shift_kept, shift_tried=fit_shift,
                residual_mm={k: [round(m[0], 1), round(m[1], 1)] for k, m in means.items()},
                rms_mm=float(math.sqrt(np.mean(np.square(errs)))) if errs else float("nan"))


def check_points(kin, obs, fit, radius):
    """Mean error (mm) of the object centre at each point with the fitted offsets applied."""
    saved = (kin.offsets, kin.shift.copy())
    kin.set_mount_offsets(fit["pitch"], fit["yaw"], fit["roll"], fit["shift_m"])
    res = {}
    for o in obs:
        Rb, pb = o.get("Rb"), o.get("pb")
        if Rb is None:
            Rb, pb = kin.camera_body_pose(o["q"], o["gb"])
        pos, R = kin.camera_from_body(Rb, pb)
        c = object_centre(kin, o["u"], o["v"], pos, R, o["w"], o["h"], radius)
        if c is not None:
            res.setdefault(o["point"], []).append((c[0] - o["x"], c[1] - o["y"]))
    kin.set_mount_offsets(*saved[0], saved[1])
    return {k: (float(np.mean([e[0] for e in v]) * 1000), float(np.mean([e[1] for e in v]) * 1000)) for k, v in res.items()}


def intrinsics_fingerprint(K):
    return dict(source=K.source, model=K.model, fx=round(K.fx, 3), fy=round(K.fy, 3), cx=round(K.cx, 3),
                cy=round(K.cy, 3), dist=[round(d, 6) for d in K.dist], date=K.meta.get("date"))


def capture_point(cam, tail, kin, detect, label, x, y, n=CAPTURE_N, timeout_s=CAPTURE_S):
    from approach import below_horizon
    rows, last, cut = [], None, 0
    end = time.time() + timeout_s
    while len(rows) < n and time.time() < end:
        fr, st = cam.latest(), tail.latest()
        if fr is None or st is None or fr[2] == last:
            time.sleep(0.03)
            continue
        last = fr[2]
        s = tail.at(fr[1])
        if s is None or abs(s["t"] - fr[1]) > 0.1 or not s.get("ok", True):
            continue
        h, w = fr[0].shape[:2]
        det = detect(fr[0], below_horizon(kin, s, w, h))
        if det is None:
            continue
        x0, y0, bw, bh = det["bbox"]
        if y0 + bh >= h - EDGE_PX or x0 <= EDGE_PX or x0 + bw >= w - EDGE_PX or y0 <= EDGE_PX:
            cut += 1
            continue
        rows.append(dict(point=label, x=x, y=y, u=float(det.get("base_cx", det["cx"])), v=float(y0 + bh), w=w, h=h,
                         area=int(det["area"]), t=fr[1], q=dict(s["q"]), gb=list(s["gb"])))
    return rows, cut


def write_head(path, head):
    from approach import send_file
    send_file(path)({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": dict(head)})


def report_and_write(kin, obs, args, captures_path):
    fit_obs = [o for o in obs if not o.get("holdout")]
    hold = [o for o in obs if o.get("holdout")]
    if len({o["point"] for o in fit_obs}) < 3:
        sys.exit("need at least 3 fit points with captures")
    print("before the fit (CAD mount, current intrinsics):")
    zero = dict(pitch=0.0, yaw=0.0, roll=0.0, shift_m=[0.0, 0.0, 0.0])
    for k, m in check_points(kin, obs, zero, args.obj_radius).items():
        print(f"  {k:28s} error x {m[0]:+6.1f} y {m[1]:+6.1f} mm")
    fit = fit_mount(kin, fit_obs, args.obj_radius, fit_shift=args.fit_shift)
    sd = fit["std"]
    print(f"fit: pitch {fit['pitch']:+.2f} (+-{sd[0]:.2f}) yaw {fit['yaw']:+.2f} (+-{sd[1]:.2f}) roll {fit['roll']:+.2f} "
          f"(+-{sd[2]:.2f}) deg" + (f", lens shift {[round(v * 1000, 1) for v in fit['shift_m']]} mm" if fit["shift_kept"]
                                     else ", lens shift not fitted" if not fit["shift_tried"]
                                     else f", lens shift dropped (std {[round(s, 1) for s in sd[3:6]]} mm > {SHIFT_STD_MAX_MM})"))
    for k, m in fit["residual_mm"].items():
        print(f"  fit point {k:18s} residual x {m[0]:+6.1f} y {m[1]:+6.1f} mm ({math.hypot(*m):.1f})")
    print(f"  fit rms {fit['rms_mm']:.1f} mm")
    hold_err = None
    if hold:
        hm = check_points(kin, hold, fit, args.obj_radius)
        for k, m in hm.items():
            hold_err = math.hypot(*m)
            print(f"  HELD-OUT  {k:18s} error x {m[0]:+6.1f} y {m[1]:+6.1f} mm ({hold_err:.1f})")
    passed = hold_err is not None and hold_err < HOLDOUT_OK_MM
    with open(duck_kin.U20CAM_MOUNT) as f:
        mount = json.load(f)
    mount["offsets_deg"] = {"pitch": round(fit["pitch"], 3), "yaw": round(fit["yaw"], 3), "roll": round(fit["roll"], 3)}
    mount["shift_m"] = [round(v, 5) for v in fit["shift_m"]]
    mount["source"] = "fit"
    mount["fit"] = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "points": sorted({(o["point"], o["x"], o["y"]) for o in fit_obs}),
        "holdout": sorted({(o["point"], o["x"], o["y"]) for o in hold}),
        "frames": len(obs),
        "std": {"pitch_deg": round(sd[0], 3), "yaw_deg": round(sd[1], 3), "roll_deg": round(sd[2], 3)},
        "residual_mm": fit["residual_mm"], "rms_mm": round(fit["rms_mm"], 2),
        "holdout_error_mm": None if hold_err is None else round(hold_err, 1),
        "holdout_ok_mm": HOLDOUT_OK_MM, "passed": passed,
        "obj_radius_m": args.obj_radius,
        "head": args.head_used,
        "captures": captures_path,
        "intrinsics": intrinsics_fingerprint(kin.K),
    }
    out = args.out or duck_kin.U20CAM_MOUNT
    if kin.K.source == "datasheet" and not args.force:
        cand = out.replace(".json", ".candidate.json")
        json.dump(mount, open(cand, "w"), indent=1)
        sys.exit(f"FAIL: the intrinsics are the datasheet default; calibrate them first (cam_calib.py --intrinsics), "
                 f"then refit these captures with --from {captures_path}. Fit kept in {cand}")
    if not passed and not args.force:
        cand = out.replace(".json", ".candidate.json")
        json.dump(mount, open(cand, "w"), indent=1)
        sys.exit(f"FAIL: held-out error {'missing' if hold_err is None else f'{hold_err:.1f} mm'} is not under "
                 f"{HOLDOUT_OK_MM} mm: not written to {out} (candidate in {cand})")
    json.dump(mount, open(out, "w"), indent=1)
    print(f"{'PASS' if passed else 'FORCED'}: wrote {out}")


def run_mount(args):
    kin = duck_kin.DuckKin()
    if args.from_:
        with open(os.path.expanduser(args.from_)) as f:
            cap = json.load(f)
        args.obj_radius = cap.get("obj_radius_m", args.obj_radius) if args.obj_radius is None else args.obj_radius
        args.head_used = cap.get("head")
        report_and_write(kin, cap["obs"], args, os.path.abspath(os.path.expanduser(args.from_)))
        return
    import orange_target
    from approach import DEFAULT_HEAD, LogTail, ShmCamera, find_log
    args.obj_radius = 0.02 if args.obj_radius is None else args.obj_radius
    pts = [tuple(float(v) for v in p.split(",")) for p in args.points.split(";")] if args.points else list(MOUNT_POINTS)
    hold = tuple(float(v) for v in args.holdout.split(",")) if args.holdout else HOLDOUT
    log_path = args.log or find_log()
    if not log_path:
        sys.exit("no walk_wired log found: start the brain and stand61 from the app first")
    tail = LogTail(log_path)
    t0 = time.time()
    while tail.latest() is None and time.time() - t0 < 3:
        time.sleep(0.1)
    st = tail.latest()
    if st is None or st["mode"] != "walk" or not st["torque"]:
        sys.exit(f"walk_wired is not standing on stand61 (mode {st['mode'] if st else '-'}): stand it from the app, "
                 "then run this again")
    head = dict(MOUNT_HEAD, head_pitch=args.head_pitch)
    args.head_used = head
    lo, hi = (tuple(int(v) for v in a.split(",")) for a in (args.hsv_lo, args.hsv_hi))
    detect = lambda f, accept=None: orange_target.detect(f, lo, hi, 60, accept=accept)  # noqa: E731
    cam = ShmCamera(require_calibration=False)
    obs = []
    try:
        print(f"log {log_path}; head to {head} (Shawn: hand near the robot)", flush=True)
        write_head(args.cmd, head)
        time.sleep(2.5)
        st = tail.latest()
        tx = toe_x(kin, st["q"], st["gb"])
        print(f"head now pitch {st['q']['head_pitch']:.2f} yaw {st['q']['head_yaw']:.2f}. Reference: the foot-frame origin "
              f"is midway between the feet, {tx * 100:.1f} cm behind the front edge of the soles. Put tape along the front "
              f"edge of both soles and mark the centre line between the feet.", flush=True)
        plan = [(f"x{x:.2f}_y{y:+.2f}", x, y, False) for x, y in pts] + [(f"x{hold[0]:.2f}_y{hold[1]:+.2f}", hold[0], hold[1], True)]
        for label, x, y, is_hold in plan:
            while True:
                side = "left" if y > 0 else "right" if y < 0 else ""
                ans = input(f"\n{'HELD-OUT ' if is_hold else ''}point {label}: object centre {(x - tx) * 100:.1f} cm ahead of "
                            f"the toe tape, {abs(y) * 100:.1f} cm {side + ' of' if side else 'on'} the centre line. "
                            f"Enter when placed (s skip, q stop): ").strip().lower()
                if ans == "q":
                    raise KeyboardInterrupt
                if ans == "s":
                    break
                rows, cut = capture_point(cam, tail, kin, detect, label, x, y)
                if len(rows) < 5:
                    print(f"  only {len(rows)} usable frames ({cut} with the object cut at the picture edge): "
                          f"check it is in view and not at the edge, then Enter again")
                    continue
                for r in rows:
                    r["holdout"] = is_hold
                u, v = np.median([r["u"] for r in rows]), np.median([r["v"] for r in rows])
                c = object_centre(kin, u, v, *kin.camera_pose(rows[0]["q"], rows[0]["gb"]), rows[0]["w"], rows[0]["h"],
                                  args.obj_radius)
                print(f"  {len(rows)} frames, pixel ({u:.1f}, {v:.1f}); with the current mount it reads "
                      + (f"x {c[0]:.3f} y {c[1]:+.3f} (error {math.hypot(c[0] - x, c[1] - y) * 1000:.0f} mm)" if c else "off the floor"))
                obs += rows
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cam.close()
        write_head(args.cmd, {k: DEFAULT_HEAD[k] for k in ("head_pitch", "head_yaw")})
    path = os.path.expanduser(f"~/duck-logs/u20cam-mount-{time.strftime('%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(obs=[{k: v for k, v in o.items() if k not in ("Rb", "pb")} for o in obs], head=head,
                       obj_radius_m=args.obj_radius, log=log_path, intrinsics=intrinsics_fingerprint(kin.K)), f)
    print(f"captures saved to {path}")
    report_and_write(kin, obs, args, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--intrinsics", action="store_true")
    g.add_argument("--mount", action="store_true")
    g.add_argument("--board-png", metavar="PATH", help="write the chessboard picture and exit")
    g.add_argument("--synth", metavar="DIR", help="write synthetic chessboard frames (known fisheye camera) and exit")
    ap.add_argument("--images", nargs="+", help="--intrinsics from these image files (globs) instead of the camera")
    ap.add_argument("--pattern", default=f"{PATTERN[0]}x{PATTERN[1]}", help="inner corners, columns x rows")
    ap.add_argument("--n", type=int, default=N_FRAMES, help="frames to keep (--intrinsics) or to render (--synth)")
    ap.add_argument("--timeout", type=float, default=240.0, help="s to collect frames")
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output json (default: the file duck_kin reads)")
    ap.add_argument("--force", action="store_true", help="write even when the acceptance number is missed")
    ap.add_argument("--from", dest="from_", default=None, help="--mount: refit saved captures, no robot")
    ap.add_argument("--points", default=None, help='--mount fit points "x,y;x,y;..." in m (default %s)' % (MOUNT_POINTS,))
    ap.add_argument("--holdout", default=None, help='--mount held-out point "x,y" (default %s,%s)' % HOLDOUT)
    ap.add_argument("--head-pitch", type=float, default=MOUNT_HEAD["head_pitch"])
    ap.add_argument("--obj-radius", type=float, default=None, help="m, half the object's width (default 0.02)")
    ap.add_argument("--fit-shift", action="store_true", help="also fit a lens shift (kept only when well determined)")
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--log", default=None)
    ap.add_argument("--hsv-lo", default="5,120,120")
    ap.add_argument("--hsv-hi", default="22,255,255")
    args = ap.parse_args()
    if args.board_png:
        import cv2
        cv2.imwrite(args.board_png, board_image(tuple(int(v) for v in args.pattern.lower().split("x"))))
        print(f"wrote {args.board_png}: show it full screen, brightness up")
    elif args.synth:
        run_synth(args)
    elif args.intrinsics:
        run_intrinsics(args)
    else:
        run_mount(args)


if __name__ == "__main__":
    main()
