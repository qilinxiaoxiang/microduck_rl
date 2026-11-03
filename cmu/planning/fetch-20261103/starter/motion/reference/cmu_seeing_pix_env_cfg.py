"""T2, the seeing pick that watches the picture (cmu/grasp/PLAN-continuous-vision.md section 6, "T2, bite").

`Mjlab-SeeingPickPix-FromPick-MicroDuck-CMU-Plugged` is the seeing pick (`cmu_seeing_pick_env_cfg.py`) with its object
input moved from the floor to the picture. On the robot the floor projection is 3-6 cm wrong in the bow, so camera-only
picks missed; here the policy sees where the ball is in the U20CAM picture, and the projection error never sits between
the eye and the bite. Notes, sources and later commands: `cmu/grasp/T2-PIXEL.md`.

1. Actor body slots 55-60 (61-D contract kept): [u, v, width, age, holding, visible].
   u, v: centre of the last delivered detection, normalised to the picture, -1 (left / top edge) .. +1 (right / bottom
   edge), in the 640 x 360 frame a7z_cam.py publishes upright. width: the detection's width over the picture width.
   age: seconds since that detection was captured (as the seeing pick). holding: the delayed pinch bit (as the seeing
   pick). visible: 1 when the last delivered frame contained the object, 0 when it did not (dropped, out of the picture,
   behind the beak); u, v, width then keep the last detection and the age grows.
2. Sim camera: the object centre projected through the calibrated U20CAM lens (equidistant fisheye with OpenCV's
   k1..k4, u20cam_intrinsics.json 2026-10-30) at the fitted mount (u20cam_mount.json: CAD lens centre on jaw_soft, fitted
   pitch / yaw / roll, head_pitch_gain, bow_height_correction), i.e. guide/firmware/mac/duck_kin.py in torch. Per episode
   the mount is drawn within the measured errors (CAM_DR); frames at 8-10 fps, 0.1-0.3 s delay, drops and stalls.
3. Start states: the ball 8-25 cm from the foot-frame origin within +-45 deg bearing, reached through the placement
   curriculum (PIX_WINDOW_*); the head starts aimed at it (the gaze.py hand-over), the body arrives with a residual
   forward speed of 0-0.15 m/s. Objects, friction, kicks and distractors are the seeing pick's.
Rewards, the critic (true object pose, shape and size), timeline and terminations are the seeing pick's.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply

from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp
from mjlab_microduck.tasks import mdp as microduck_mdp

# ── Lens: guide/firmware/mac/u20cam_intrinsics.json (cam_calib.py --intrinsics, 2026-10-30T01:38, rms 0.40 px) ──
# Continuous pixel convention (centre of column j is j + 0.5), OpenCV fisheye: theta_d = theta (1 + k1 t^2 + ... + k4 t^8)
IMG_W, IMG_H = 640, 360
FX, FY, CX, CY = 534.1751, 534.502, 311.0997, 168.3507
DIST = (-0.1413307, -0.24242853, 0.70203388, -0.67228765)

# ── Mount: guide/firmware/mac/u20cam_mount.json as DuckKin reads it (fit 2026-10-30T00:30, bow correction 07:40) ──
MOUNT_BODY = "jaw_soft"
MOUNT_P = (0.0154994, -0.00009, -0.0779)       # lens projection centre in the jaw_soft frame, m (DuckKin.mount_site)
MOUNT_R = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))  # columns: optical axis, image left, image up
MOUNT_OFFSETS_DEG = (11.975, 1.879, -0.686)    # fitted pitch (nose down +), yaw (left +), roll (right side down +)
HEAD_PITCH_GAIN = 0.92                         # the head turns this times its reading (ruler check 2026-10-30)
BOW_DZ = -0.013                                # camera height correction at neck_pitch <= BOW_NECK_FULL, zero at >= 0
BOW_NECK_FULL, BOW_NECK_NONE = -1.2, 0.0
NECK_FIT = 0.349                               # neck_pitch of the standing mount fit (stand61 default pose)

# ── Per-episode mount randomisation within the measured errors (plan section 6 and N1 status, 2026-10-30) ──
CAM_DR = dict(
    pitch_deg=2.0,            # +- on the fitted pitch
    shift_m=0.01,             # +- per axis, lens position in the camera frame (DuckKin shift_m)
    head_gain=(0.90, 1.00),   # head_pitch gain
    pivot_m=0.01,             # +- per axis, neck pivot offset (moves the camera by (I - R_neck) d)
    bow_dz=(-0.015, -0.008),  # bow height correction in full; per-pose fits -7.8 .. -14.9 mm
)

# ── Frames and detections (A2 tracker on the Mac mini, tracker/ACCEPTANCE.md: 9.85 fps, age p50 0.21 s) ──
PIX_PERIOD_STEPS = (5, 6)      # 10 or 8.3 fps at 50 Hz, per episode
PIX_DELAY_S = (0.10, 0.30)
PIX_DROP = (0.0, 0.3)          # per-episode probability that a frame has no detection
PIX_NOISE_PX = 2.0             # per frame, on u and v
PIX_BIAS_PX = 3.0              # per episode, the box centre against the object centre
PIX_WIDTH_NOISE = 0.1          # per frame, relative
PIX_MIN_DEPTH = 0.015
BEAK_HIDE_M = (0.02, 0.03)     # per episode: no detection while the mouth tip is this close to the object
                               # (cmu/grasp/visibility: the ball leaves the picture under the beak in the last cm)
PIX_INIT_AGE_S = (0.1, 0.4)    # the hand-over frame's age when the policy starts
PIX_INIT_AGE_HIDDEN_S = (0.3, 1.0)

# ── Start states ──
R_MIN, R_MAX = 0.08, 0.25      # m from the foot-frame origin (below the trunk), plan section 6
BEARING_MAX = math.radians(45.0)
PIX_WINDOW_START, PIX_WINDOW_STEP = 0.01, 0.01
PIX_WINDOW_MAX = R_MAX - sp.OBJECT_NOMINAL_XY[0]   # 0.163: the full start set
PIX_WINDOW_OPERATING = PIX_WINDOW_MAX
START_SPEED = (0.0, 0.15)      # m/s forward at reset (the hand-over happens while moving)
LOOK_JITTER = 0.10             # rad on the aimed head pitch and yaw (gaze.py keeps the ball near one spot)
HEAD_PITCH_LIMITS = (0.349, 1.449)   # default + the head-command range walk61/stand61 were trained on (+1.10)
HEAD_YAW_LIMIT = 0.8

# Runtime normaliser for the body slots. age and holding mean the same as in seepick61 and keep its statistics and
# first-layer columns; u, v, width and visible are new, so their columns start at zero (SeeingPickPixFromPickRunner).
PIX_SLOT_MEAN = (0.0, 0.0, 0.05, None, None, 0.7)
PIX_SLOT_STD = (0.5, 0.5, 0.05, None, None, 0.46)
PIX_NEW_SLOTS = (0, 1, 2, 5)


def _fold_radius(dist=DIST) -> float:
    """tan(off-axis angle) where the fitted fisheye polynomial stops growing (duck_kin.Intrinsics._fold_radius)."""
    k1, k2, k3, k4 = dist
    th = np.linspace(1e-3, math.pi / 2 - 1e-3, 6000)
    t2 = th * th
    rd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    down = np.nonzero(np.diff(rd) <= 0)[0]
    return float(np.tan(th[down[0]])) if down.size else math.inf


R_FOLD = _fold_radius()


# ── Projection (torch, batched) ─────────────────────────────────────────────────
def project_fisheye(p: torch.Tensor):
    """Camera-frame points (..., 3) (x optical axis, y image left, z image up) -> pixel u, v (continuous) and a mask
    of points in front of the lens and inside the fold radius. duck_kin.Intrinsics.project in torch."""
    x = p[..., 0]
    front = x > 1e-9
    xs = torch.where(front, x, torch.ones_like(x))
    a, b = -p[..., 1] / xs, -p[..., 2] / xs
    r = torch.sqrt(a * a + b * b)
    ok = front & (r < R_FOLD)
    th = torch.atan(r)
    t2 = th * th
    k1, k2, k3, k4 = DIST
    thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    s = torch.where(r > 1e-12, thd / r.clamp(min=1e-12), torch.ones_like(r))
    return CX + FX * a * s, CY + FY * b * s, ok


def _rot(axis: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (n, 3) unit axes, (n,) angles -> (n, 3, 3)."""
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = torch.cos(ang), torch.sin(ang)
    C = 1 - c
    return torch.stack([
        torch.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], -1),
        torch.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], -1),
        torch.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], -1)], -2)


def _offset_rotation(pitch: torch.Tensor, yaw: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """duck_kin.offset_rotation (radians, batched): Rz(yaw) Ry(pitch) Rx(roll)."""
    n = pitch.shape[0]
    ex = torch.tensor([1.0, 0.0, 0.0], device=pitch.device).expand(n, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=pitch.device).expand(n, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], device=pitch.device).expand(n, 3)
    return _rot(ez, yaw) @ _rot(ey, pitch) @ _rot(ex, roll)


def _quat_mat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


# ── Per-env state ───────────────────────────────────────────────────────────────
H = 24  # frames in flight ring buffer (> max delay 15 steps)


def _px(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_seeing_pix", None)
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        robot: Entity = env.scene["robot"]
        m = env.sim.mj_model
        body_ids, _ = robot.find_bodies([MOUNT_BODY])
        jids, _ = robot.find_joints(["neck_pitch", "head_pitch", "head_yaw"], preserve_order=True)
        gj = {}
        for name in ("neck_pitch", "head_pitch"):
            gj[name] = next(i for i in range(m.njnt) if m.joint(i).name.split("/")[-1] == name)
        tip_ids, _ = robot.find_sites(["mouth_tip"])
        st = env._seeing_pix = {
            "body": int(body_ids[0]), "neck_j": int(jids[0]), "head_j": int(jids[1]), "yaw_j": int(jids[2]),
            "neck_g": gj["neck_pitch"], "head_g": gj["head_pitch"],
            "neck_parent": int(m.body_parentid[m.jnt_bodyid[gj["neck_pitch"]]]), "tip_site": int(tip_ids[0]),
            "mount_p": torch.tensor(MOUNT_P, device=dev),
            "mount_R": torch.tensor(MOUNT_R, device=dev),
            # per-episode camera draw (nominal until the first reset)
            "dpitch": z(), "shift": z(3), "gain": torch.full((n,), HEAD_PITCH_GAIN, device=dev), "pivot": z(3),
            "bow": torch.full((n,), BOW_DZ, device=dev), "hide": torch.full((n,), 0.025, device=dev),
            "bias": z(2), "period": torch.full((n,), 5, dtype=torch.long, device=dev),
            "delay": torch.full((n,), 10, dtype=torch.long, device=dev), "drop": z(),
            "cam_offset": torch.zeros(n, dtype=torch.long, device=dev),
            "stall_until": torch.full((n,), -1, dtype=torch.long, device=dev),
            "buf": z(H, 3), "buf_det": torch.zeros(n, H, dtype=torch.bool, device=dev),
            "buf_frame": torch.zeros(n, H, dtype=torch.bool, device=dev),
            "buf_step": torch.full((n, H), -10 ** 6, dtype=torch.long, device=dev),
            "last": z(3), "last_step": torch.zeros(n, dtype=torch.long, device=dev), "visible": z(),
            "init_pending": torch.ones(n, dtype=torch.bool, device=dev),
            "init_age": torch.full((n,), 0.2, device=dev),
            "obs_key": None, "obs": z(6), "resets": 0,
        }
    return st


def camera_pose_w(env: ManagerBasedRlEnv):
    """World position (n, 3) and rotation (n, 3, 3) of the U20CAM (columns: optical axis, image left, image up), as
    DuckKin.camera_pose on the sim's joint readings with this episode's mount draw."""
    st = _px(env)
    robot: Entity = env.scene["robot"]
    d = env.sim.data
    n = env.num_envs
    Rb = _quat_mat(robot.data.body_link_quat_w[:, st["body"]])
    pb = robot.data.body_link_pos_w[:, st["body"]]
    Rc = Rb @ st["mount_R"]
    p = pb + (Rb @ st["mount_p"].unsqueeze(-1)).squeeze(-1) + (Rc @ st["shift"].unsqueeze(-1)).squeeze(-1)
    off = _offset_rotation(torch.deg2rad(MOUNT_OFFSETS_DEG[0] + st["dpitch"]),
                           torch.full((n,), math.radians(MOUNT_OFFSETS_DEG[1]), device=env.device),
                           torch.full((n,), math.radians(MOUNT_OFFSETS_DEG[2]), device=env.device))
    R = Rc @ off
    q = robot.data.joint_pos
    # head_pitch gain: the subtree below the head joint turns gain x the reading (DuckKin.place), so the camera turns
    # (gain - 1) x the reading about the joint's world axis through its anchor
    a_h, ax_h = d.xanchor[:, st["head_g"]], d.xaxis[:, st["head_g"]]
    Rg = _rot(ax_h, (st["gain"] - 1.0) * q[:, st["head_j"]])
    p = a_h + (Rg @ (p - a_h).unsqueeze(-1)).squeeze(-1)
    R = Rg @ R
    # neck pivot error d (neck parent frame): rotating about a + d instead of a adds (I - R_neck(theta)) d, zero at the
    # neck angle of the standing fit
    ax_n = d.xaxis[:, st["neck_g"]]
    Rn = _rot(ax_n, q[:, st["neck_j"]] - NECK_FIT)
    Rp = _quat_mat(d.xquat[:, st["neck_parent"]])
    dw = (Rp @ st["pivot"].unsqueeze(-1)).squeeze(-1)
    p = p + dw - (Rn @ dw.unsqueeze(-1)).squeeze(-1)
    # measured bow height correction (DuckKin.bow_dz)
    w = ((BOW_NECK_NONE - q[:, st["neck_j"]]) / (BOW_NECK_NONE - BOW_NECK_FULL)).clamp(0.0, 1.0)
    p = p + torch.stack([torch.zeros_like(w), torch.zeros_like(w), w * st["bow"]], -1)
    return p, R


def object_pixels(env: ManagerBasedRlEnv, p_obj: torch.Tensor, half: torch.Tensor):
    """Pixel centre (u, v), pixel width and an in-picture mask of objects at p_obj (n, 3) with horizontal half extent
    half (n,), without noise, drops or delay."""
    cam_p, R = camera_pose_w(env)
    pc = (R.transpose(-1, -2) @ (p_obj - cam_p).unsqueeze(-1)).squeeze(-1)
    u, v, ok = project_fisheye(pc)
    left = R[:, :, 1]
    ul, _, okl = project_fisheye((R.transpose(-1, -2) @ (p_obj + half.unsqueeze(-1) * left - cam_p).unsqueeze(-1)).squeeze(-1))
    ur, _, okr = project_fisheye((R.transpose(-1, -2) @ (p_obj - half.unsqueeze(-1) * left - cam_p).unsqueeze(-1)).squeeze(-1))
    width = torch.where(okl & okr, (ul - ur).abs(), 2.0 * FX * half / pc[:, 0].clamp(min=1e-3))
    inside = ok & (pc[:, 0] > PIX_MIN_DEPTH) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
    return u, v, width, inside


def _norm(u, v, width):
    return torch.stack([(u - IMG_W / 2) / (IMG_W / 2), (v - IMG_H / 2) / (IMG_H / 2), width / IMG_W], -1)


# ── Events ──────────────────────────────────────────────────────────────────────
def _uniform(shape, lo, hi, dev):
    return torch.rand(shape, device=dev) * (hi - lo) + lo


def reset_pix_object(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    window: float = PIX_WINDOW_START,
    center_frac: float = sp.CENTER_FRAC,
    kick_prob: float = sp.KICK_PROB,
    distractor_prob: float = sp.DISTRACTOR_PROB,
    start_speed: tuple = START_SPEED,
    look_jitter: float = LOOK_JITTER,
) -> None:
    """The seeing pick's reset (object, distractor, kick, holding), then: the target moved to range r and bearing b in
    the window (r in [max(R_MIN, r0 - window), min(R_MAX, r0 + window)], |b| <= min(45 deg, window / r0) where r0 is
    the static pose's landing spot), this episode's camera, the head aimed at the object and the residual speed."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    sp.reset_seeing_object(env, env_ids, window_xy=0.0, center_frac=1.0, kick_prob=kick_prob,
                           distractor_prob=distractor_prob)
    n, dev = len(env_ids), env.device
    st = _px(env)
    st["resets"] += 1
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene["object"]
    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    yaw = sp._yaw(root[:, 3:7])
    c, s = torch.cos(yaw), torch.sin(yaw)
    r0 = sp.OBJECT_NOMINAL_XY[0]
    r = _uniform(n, max(R_MIN, r0 - window), min(R_MAX, r0 + window), dev)
    bmax = min(BEARING_MAX, window / r0)
    b = _uniform(n, -bmax, bmax, dev)
    centre = torch.rand(n, device=dev) < center_frac
    r = torch.where(centre, torch.full_like(r, r0), r)
    b = torch.where(centre, torch.zeros_like(b), b)
    off = torch.stack([r * torch.cos(b), r * torch.sin(b)], -1)
    pose = env.sim.data.qpos[env_ids][:, obj.indexing.free_joint_q_adr].clone()   # written by the reset above
    pose[:, 0] = root[:, 0] + c * off[:, 0] - s * off[:, 1]
    pose[:, 1] = root[:, 1] + s * off[:, 0] + c * off[:, 1]
    obj.write_root_link_pose_to_sim(pose, env_ids)
    obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids)
    microduck_mdp._gg_buffer(env, "object_spawn", 3)[env_ids] = pose[:, :3]
    # a distractor the seeing pick put near the nominal spot goes near the moved target instead
    dis: Entity = env.scene["distractor"]
    dq = env.sim.data.qpos[env_ids][:, dis.indexing.free_joint_q_adr]
    dp = dq[:, :3]
    dx = c * (dp[:, 0] - root[:, 0]) + s * (dp[:, 1] - root[:, 1])
    near = dx > -0.3
    if near.any():
        ang = (torch.rand(n, device=dev) * 2 - 1) * math.pi
        doff = off + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * _uniform(n, *sp.DISTRACTOR_DIST, dev).unsqueeze(-1)
        doff[:, 0] = doff[:, 0].clamp(min=sp.OBJECT_X_MIN)
        dpose = dq.clone()
        dpose[:, 0] = torch.where(near, root[:, 0] + c * doff[:, 0] - s * doff[:, 1], dpose[:, 0])
        dpose[:, 1] = torch.where(near, root[:, 1] + s * doff[:, 0] + c * doff[:, 1], dpose[:, 1])
        dis.write_root_link_pose_to_sim(dpose, env_ids)

    # head aimed at the object (the gaze.py hand-over): pitch from the range (standing picture centre row meets the
    # floor at 0.39 / 0.25 / 0.16 m for head_pitch 0.8 / 1.1 / 1.4, DuckKin), yaw = bearing, plus jitter
    hp = (1.1 + (0.245 - r) * (0.3 / 0.086)).clamp(*HEAD_PITCH_LIMITS)
    hp = hp + (torch.rand(n, device=dev) * 2 - 1) * look_jitter
    hy = (b + (torch.rand(n, device=dev) * 2 - 1) * look_jitter).clamp(-HEAD_YAW_LIMIT, HEAD_YAW_LIMIT)
    jids = torch.tensor([st["head_j"], st["yaw_j"]], device=dev)
    pos = torch.stack([hp.clamp(*HEAD_PITCH_LIMITS), hy], -1)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), jids, env_ids)

    # residual forward speed along the heading
    # (qvel of the free joint: linear in the world frame, angular in the body frame, as reset_base wrote it)
    qv = env.sim.data.qvel[env_ids][:, robot.indexing.free_joint_v_adr]
    vel = torch.cat([qv[:, :3], quat_apply(root[:, 3:7], qv[:, 3:])], -1)
    v = _uniform(n, *start_speed, dev)
    vel[:, 0] += c * v
    vel[:, 1] += s * v
    robot.write_root_link_velocity_to_sim(vel, env_ids)

    # this episode's camera
    st["dpitch"][env_ids] = _uniform(n, -CAM_DR["pitch_deg"], CAM_DR["pitch_deg"], dev)
    st["shift"][env_ids] = _uniform((n, 3), -CAM_DR["shift_m"], CAM_DR["shift_m"], dev)
    st["gain"][env_ids] = _uniform(n, *CAM_DR["head_gain"], dev)
    st["pivot"][env_ids] = _uniform((n, 3), -CAM_DR["pivot_m"], CAM_DR["pivot_m"], dev)
    st["bow"][env_ids] = _uniform(n, *CAM_DR["bow_dz"], dev)
    st["hide"][env_ids] = _uniform(n, *BEAK_HIDE_M, dev)
    st["bias"][env_ids] = _uniform((n, 2), -PIX_BIAS_PX, PIX_BIAS_PX, dev)
    st["period"][env_ids] = torch.randint(PIX_PERIOD_STEPS[0], PIX_PERIOD_STEPS[1] + 1, (n,), device=dev)
    dt = env.step_dt
    st["delay"][env_ids] = torch.round(_uniform(n, *PIX_DELAY_S, dev) / dt).long().clamp(1, H - 1)
    st["drop"][env_ids] = _uniform(n, *PIX_DROP, dev)
    st["cam_offset"][env_ids] = torch.randint(0, 6, (n,), device=dev)
    st["stall_until"][env_ids] = -1
    st["buf_det"][env_ids] = False
    st["buf_frame"][env_ids] = False
    st["buf_step"][env_ids] = -10 ** 6
    st["init_pending"][env_ids] = True


def pix_window_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor, event_name: str = "reset_object",
                          window_step: float = PIX_WINDOW_STEP, window_max: float = PIX_WINDOW_MAX,
                          advance_success: float = sp.WINDOW_ADVANCE_SUCCESS,
                          min_iters: int = sp.CURRICULUM_MIN_ITERS) -> dict:
    """The seeing pick's window curriculum (success EMA > advance_success widens by window_step at most every
    min_iters iterations) on the polar start window, up to the full 8-25 cm, +-45 deg start set."""
    st = sp._st(env)
    ev = env.event_manager.get_term_cfg(event_name)
    if env_ids is not None and len(env_ids) > 0:
        gs = microduck_mdp._gg_cache(env).get("grasp_state")
        if gs is not None:
            rate = float(gs["held"][env_ids].float().mean())
            st["success"] = (1 - sp.SUCCESS_EMA_ALPHA) * st["success"] + sp.SUCCESS_EMA_ALPHA * rate
    it = env.common_step_counter // sp.STEPS_PER_ITER
    if (st["success"] > advance_success and ev.params["window"] < window_max - 1e-9
            and it - st["window_it"] >= min_iters):
        ev.params["window"] = min(ev.params["window"] + window_step, window_max)
        st["window_it"] = it
    return {"window": torch.tensor(ev.params["window"]), "success_ema": torch.tensor(st["success"])}


# ── Observation ─────────────────────────────────────────────────────────────────
def _pix_update(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Advance the simulated U20CAM one policy step (once per step, cached) and return the six body slots."""
    st = _px(env)
    key = (env.common_step_counter, st["resets"], sp._st(env)["resets"])
    if st["obs_key"] == key:
        return st["obs"]
    dev, n = env.device, env.num_envs
    t = env.episode_length_buf.long()
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene["object"]
    half = sp._st(env)["half"][:, :2].max(-1).values
    p_obj = obj.data.root_link_pos_w
    u, v, width, inside = object_pixels(env, p_obj, half)
    tip = robot.data.site_pos_w[:, st["tip_site"]]
    hidden = (tip - p_obj).norm(dim=-1) < st["hide"]
    seen = inside & ~hidden
    meas = _norm(u + st["bias"][:, 0] + torch.randn(n, device=dev) * PIX_NOISE_PX,
                 v + st["bias"][:, 1] + torch.randn(n, device=dev) * PIX_NOISE_PX,
                 width * (1 + torch.randn(n, device=dev) * PIX_WIDTH_NOISE).clamp(min=0.2))

    # the hand-over frame: what the camera saw at the start pose; out of the picture -> its last position at the edge
    init = st["init_pending"]
    if init.any():
        dt = env.step_dt
        edge = _norm(u.clamp(0, IMG_W - 1), v.clamp(0, IMG_H - 1), width)
        st["last"] = torch.where(init.unsqueeze(-1), torch.where(seen.unsqueeze(-1), meas, edge), st["last"])
        st["visible"] = torch.where(init, seen.float(), st["visible"])
        age = torch.where(seen, _uniform(n, *PIX_INIT_AGE_S, dev), _uniform(n, *PIX_INIT_AGE_HIDDEN_S, dev))
        st["last_step"] = torch.where(init, t - torch.round(age / dt).long(), st["last_step"])
        st["init_pending"] = torch.zeros_like(init)

    tick = ((t + st["cam_offset"]) % st["period"]) == 0
    stall_start = tick & (torch.rand(n, device=dev) < sp.CAM_STALL_PROB) & (t >= st["stall_until"])
    st["stall_until"] = torch.where(
        stall_start, t + torch.round(_uniform(n, *sp.CAM_STALL_S, dev) / env.step_dt).long(), st["stall_until"])
    frame = tick & (t >= st["stall_until"])
    detected = frame & seen & (torch.rand(n, device=dev) >= st["drop"])
    slot = t % H
    ar = torch.arange(n, device=dev)
    st["buf"][ar, slot] = meas
    st["buf_det"][ar, slot] = detected
    st["buf_frame"][ar, slot] = frame
    st["buf_step"][ar, slot] = t
    src = (t - st["delay"]) % H
    fresh = st["buf_step"][ar, src] == t - st["delay"]
    arrived = fresh & st["buf_frame"][ar, src]
    hit = arrived & st["buf_det"][ar, src]
    st["last"] = torch.where(hit.unsqueeze(-1), st["buf"][ar, src], st["last"])
    st["last_step"] = torch.where(hit, t - st["delay"], st["last_step"])
    st["visible"] = torch.where(arrived, hit.float(), st["visible"])
    age = ((t - st["last_step"]).float() * env.step_dt).clamp(0.0, sp.AGE_MAX_S)
    # holding bit exactly as the seeing pick computes it (its state is advanced here too, once per step)
    pinched = microduck_mdp.grasp_state(env, "object")["pinched"]
    ss = sp._st(env)
    ss["pinch_hist"] = torch.cat([ss["pinch_hist"][:, 1:], pinched.unsqueeze(-1)], dim=-1)
    out = torch.zeros(n, 6, device=dev)
    out[:, :3] = st["last"]
    out[:, 3] = age
    out[:, 4] = ss["pinch_hist"][:, 0].float()
    out[:, 5] = st["visible"]
    st["obs"], st["obs_key"] = out, key
    return out


def pix_object_command(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """Actor and critic body slots: [u, v, width, age, holding, visible]."""
    return _pix_update(env)[:, :dim]


def true_object_pixels(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: the object's noise-free, undelayed picture position and width, and whether it is in the picture."""
    half = sp._st(env)["half"][:, :2].max(-1).values
    u, v, width, inside = object_pixels(env, env.scene["object"].data.root_link_pos_w, half)
    out = _norm(u.clamp(-IMG_W, 2 * IMG_W), v.clamp(-IMG_H, 2 * IMG_H), width.clamp(max=IMG_W))
    return torch.cat([out, inside.float().unsqueeze(-1)], -1)


# ── Env cfg ─────────────────────────────────────────────────────────────────────
def make_seeing_pix_env_cfg(variant: str = "plugged", play: bool = False):
    cfg = sp.make_seeing_pick_env_cfg(variant, play=play)
    cfg.events["reset_object"] = EventTermCfg(
        func=reset_pix_object, mode="reset",
        params={"window": PIX_WINDOW_OPERATING if play else PIX_WINDOW_START},
    )
    cfg.curriculum.pop("seeing_window", None)
    if not play:
        cfg.curriculum["pix_window"] = CurriculumTermCfg(func=pix_window_curriculum,
                                                         params={"event_name": "reset_object"})
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=pix_object_command,
                                                                           params={"dim": 6})
    cfg.observations["critic"].terms["object_pixels"] = ObservationTermCfg(func=true_object_pixels)
    return cfg


def seeing_pix_rl_cfg(base):
    rl = copy.deepcopy(base)
    rl.experiment_name = "seeing_pix"
    rl.run_name = "seeing_pix"
    rl.max_iterations = 3000
    return rl
