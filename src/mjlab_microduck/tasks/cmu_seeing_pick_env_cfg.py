"""CMU seeing pick: Pollen's static hover grasp on the CMU masses, with the object seen by the head camera.

MicroDuck final-goal step 4 (D-019), plan `cmu/grasp/PLAN-seeing-pickup.md` (Shawn, 2026-10-26). The walk stops the
robot with the object somewhere near the spot where the beak lands; this policy looks at it, aims while going down,
goes through the last few centimetres from memory, bites and stands up. Search, walking and retries stay in the
coordinator script outside the policy.

Changes to `Mjlab-StaticGrasp-Hover-FromPick-MicroDuck-CMU-Plugged`:

1. Actor observation, body-command slots 55-60 (the 61-D contract and its order are kept):
   [object x, object y, object size, seconds since the last detection, holding bit, 0]. x, y are on the floor in the
   runtime's foot frame (`cmu/mac/target_ground.py`: origin below trunk_base, x along the trunk heading); size is the
   object's largest horizontal extent. The value comes from a simulated head camera, updated while the robot moves:
   a frame every CAM_PERIOD_STEPS steps (10 fps), delivered CAM_DELAY_S later, dropped with a per-episode
   probability, frozen through stalls, with a per-episode bias and per-frame noise, and only when the object is in
   the U20CAM's field of view from the `head_camera` site. The ball leaves the picture for the last 4-5 cm of the
   descent (`cmu/grasp/visibility/`), so the policy sees its last detection there and the age grows. At reset the
   observation holds the coordinator's standing measurement. The holding bit is the pinch contact (both pads on the
   object) delayed by HOLD_DELAY_STEPS, the stand-in for the jaw servo's current and position check.
2. Objects: one target body with three geoms (lumpy ellipsoid ball, box, upright cylinder); per episode one is
   active and the other two shrink to 0.5 mm inside it. Sizes, mass and friction per episode (OBJECT_*). One
   distractor ball is placed near the target in half of the episodes; it is not observed.
3. Placement: the target spawns in a window around the static pose's landing spot that widens from +-1 cm to
   +-5 cm on the episode success rate (WINDOW_*), the stop error a walk leaves.
4. Recovery: in KICK_PROB of the episodes the object is rolled once during the descent (VBC, arXiv 2403.16967 App.
   C.2, 10 %). Pollen's approach target already follows the live object, so aiming at the moved object is what pays;
   object_pushed stops charging for the kick's own roll (its spawn reference follows the object for KICK_FREE_S).
5. The three stand-up terms (ground_pick_return_pose_legs / _neck, return_upright) pay only once the lips have been
   below DESCENT_GATE_M in this episode. The reward audit (cmu/grasp/seeing_audit.py, 2026-10-26) found the do-nothing
   policy (stand61) earning 1.3x the warm-start bow, mostly from these terms collected without bowing: a policy that
   stops trying would win.
Other rewards, timeline, jaw schedule and terminations are Pollen's (static grasp, hover variant). The critic also sees
the true object pose (inherited) and the active shape and size.
"""
from __future__ import annotations

import copy
import math

import mujoco
import torch
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cmu_grasp_env_cfg import make_cmu_grasp_env_cfg
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import (
    OBJECT_NOMINAL_XY,
    TIMELINE,
    make_microduck_static_grasp_env_cfg,
)

# ── Objects ───────────────────────────────────────────────────────────────────
SHAPES = ("ball", "box", "cylinder")
SHAPE_PROBS = (0.5, 0.25, 0.25)
GEOM_NAMES = ("object_geom", "object_box", "object_cyl")
BALL_RADIUS = (0.0053, 0.0072)         # per axis, independent: 1.06-1.44 cm, the printed ball Shawn tests
BOX_HALF_XY = (0.005, 0.010)           # 1-2 cm cubes and bricks
BOX_HALF_Z = (0.005, 0.009)
CYL_RADIUS = (0.005, 0.009)
CYL_HALF_H = (0.004, 0.008)
HIDDEN = 0.0005                        # inactive geoms, inside the active one
OBJECT_MASS = 0.0015                   # body mass at compile time; scaled per episode
OBJECT_MASS_RANGE = (0.0005, 0.010)    # 0.5-10 g
OBJECT_FRICTION = (0.35, 1.1)          # PLA on the PLA lips .. a glue- or rubber-lined beak
DISTRACTOR_RADIUS = 0.007
DISTRACTOR_PROB = 0.5
DISTRACTOR_DIST = (0.04, 0.10)         # from the target, in front of the toes
OBJECT_X_MIN = 0.05

# ── Placement window (m, half-width around OBJECT_NOMINAL_XY) ─────────────────
WINDOW_START, WINDOW_STEP, WINDOW_MAX = 0.01, 0.01, 0.05
WINDOW_OPERATING = 0.05                # play / evaluation
WINDOW_ADVANCE_SUCCESS = 0.5           # widen when the success EMA is above this
CENTER_FRAC = 0.2
CURRICULUM_MIN_ITERS = 50
SUCCESS_EMA_ALPHA = 0.02               # per finished episode batch
STEPS_PER_ITER = 24

# ── Camera (U20CAM-720P published at 640 x 360, about 10 fps; cmu/mac/a7z_cam.py) ──
CAM_PERIOD_STEPS = 5                   # 10 fps at 50 Hz
CAM_DELAY_S = (0.10, 0.30)             # capture -> policy, per episode
CAM_DROP = (0.0, 0.3)                  # per-episode probability that a frame has no detection
CAM_STALL_PROB = 0.02                  # per frame: a stall of CAM_STALL_S starts
CAM_STALL_S = (0.5, 1.5)
CAM_BIAS_XY = 0.01                     # per-episode projection bias (uncalibrated mount), m
CAM_NOISE_XY = 0.004                   # per frame, m
CAM_NOISE_SIZE = 0.001
CAM_HALF_FOV = (math.radians(51.0), math.radians(35.0))   # horizontal, vertical: 120 deg diagonal M12 lens, 16:9
CAM_MIN_DEPTH = 0.015
CAM_HISTORY = 16                       # ring buffer of frames in flight (> max delay in steps)
INIT_AGE_S = (0.1, 0.4)                # the standing measurement's age when the policy starts
AGE_MAX_S = 2.0
HOLD_DELAY_STEPS = 2

# ── Stand-up pay only after a real descent ────────────────────────────────────
DESCENT_GATE_M = 0.08                  # mean lip height; standing it is about 0.2 m, the bite 1.5 mm
GATED_RETURN_TERMS = ("ground_pick_return_pose_legs", "ground_pick_return_pose_neck", "return_upright")

# ── Kick (recovery) ───────────────────────────────────────────────────────────
KICK_PROB = 0.10
KICK_PHASE = (0.04, TIMELINE["down_at"] - 0.02)  # before the beak arrives
KICK_SHIFT = (0.01, 0.03)              # m: the object jumps this far (VBC changes the object position) ...
KICK_SPEED = (0.05, 0.15)              # ... and keeps rolling at this speed (m/s) in the same direction
KICK_FREE_S = 0.6

# Runtime normaliser for the body slots: the warm-start network saw zeros there (std 0.01), so its statistics and
# first-layer columns are replaced (SeeingPickRunner).
BODY_SLOT_MEAN = (OBJECT_NOMINAL_XY[0], OBJECT_NOMINAL_XY[1], 0.013, 0.4, 0.2, 0.0)
BODY_SLOT_STD = (0.03, 0.03, 0.004, 0.5, 0.4, 1.0)
BODY_SLOT_START = 55


# ── Entities ──────────────────────────────────────────────────────────────────
def get_seeing_object_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object", pos=[0.15, 0.0, 0.007])
    body.add_freejoint(name="object_free")
    kw = dict(rgba=[1.0, 0.45, 0.05, 1.0], condim=4, friction=[0.7, 0.02, 0.0001])
    # Masses split so the body totals OBJECT_MASS; per-episode mass comes from dr.pseudo_inertia.
    body.add_geom(name=GEOM_NAMES[0], type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.00625] * 3,
                  mass=OBJECT_MASS * 0.6, **kw)
    body.add_geom(name=GEOM_NAMES[1], type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0075, 0.0075, 0.007],
                  mass=OBJECT_MASS * 0.2, **kw)
    body.add_geom(name=GEOM_NAMES[2], type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.007, 0.006, 0.0],
                  mass=OBJECT_MASS * 0.2, **kw)
    return spec


def get_distractor_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="distractor", pos=[-0.5, 0.0, DISTRACTOR_RADIUS])
    body.add_freejoint(name="distractor_free")
    body.add_geom(name="distractor_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[DISTRACTOR_RADIUS],
                  mass=0.003, rgba=[0.1, 0.7, 0.2, 1.0], condim=4, friction=[0.7, 0.02, 0.0001])
    return spec


SEEING_OBJECT_CFG = EntityCfg(spec_fn=get_seeing_object_spec,
                              init_state=EntityCfg.InitialStateCfg(pos=(0.15, 0.0, 0.007)))
DISTRACTOR_CFG = EntityCfg(spec_fn=get_distractor_spec,
                           init_state=EntityCfg.InitialStateCfg(pos=(-0.5, 0.0, DISTRACTOR_RADIUS)))


# ── Per-env state ─────────────────────────────────────────────────────────────
def _st(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_seeing_pick", None)
    if st is None:
        n, dev = env.num_envs, env.device
        z = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        st = env._seeing_pick = {
            "shape": torch.zeros(n, dtype=torch.long, device=dev),
            "half": z(3),                     # active shape's half extents (x, y, z)
            "bias": z(2), "delay": torch.full((n,), 10, dtype=torch.long, device=dev),
            "drop": z(), "cam_offset": torch.zeros(n, dtype=torch.long, device=dev),
            "stall_until": torch.full((n,), -1, dtype=torch.long, device=dev),
            "buf": z(CAM_HISTORY, 3), "buf_valid": torch.zeros(n, CAM_HISTORY, dtype=torch.bool, device=dev),
            "buf_step": torch.full((n, CAM_HISTORY), -10 ** 6, dtype=torch.long, device=dev),
            "last": z(3), "last_step": torch.zeros(n, dtype=torch.long, device=dev),
            "pinch_hist": torch.zeros(n, HOLD_DELAY_STEPS + 1, dtype=torch.bool, device=dev),
            "kick_step": torch.full((n,), -1, dtype=torch.long, device=dev),
            "obs_key": None, "obs": z(6), "resets": 0,
            "lip_min": torch.ones(n, device=dev),
            "success": 0.0, "window_it": 0,
        }
    return st


def _yaw(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def _to_foot_frame(env: ManagerBasedRlEnv, p_w: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor):
    """World points (n, 3) -> (x, y) in the foot frame: origin below the root, x along the root's heading."""
    yaw = _yaw(root_quat)
    c, s = torch.cos(yaw), torch.sin(yaw)
    d = p_w[:, :2] - root_pos[:, :2]
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)


def _uniform(n: int, lo: float, hi: float, dev) -> torch.Tensor:
    return torch.rand(n, device=dev) * (hi - lo) + lo


# ── Events ────────────────────────────────────────────────────────────────────
@requires_model_fields("geom_size", "geom_rbound", "geom_aabb")
def seeing_object_shape(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("object", geom_names=GEOM_NAMES)) -> None:
    """Pick the active shape per env and its size; shrink the other two geoms inside it."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(env.device)
    n, dev = len(env_ids), env.device
    if n == 0:
        return
    st = _st(env)
    obj: Entity = env.scene["object"]
    gids = [int(obj.indexing.geom_ids[i]) for i in range(len(GEOM_NAMES))]
    shape = torch.multinomial(torch.tensor(SHAPE_PROBS, device=dev), n, replacement=True)
    half = torch.zeros(n, 3, device=dev)
    sizes = torch.full((n, 3, 3), HIDDEN, device=dev)
    ball = torch.stack([_uniform(n, *BALL_RADIUS, dev) for _ in range(3)], -1)
    box = torch.stack([_uniform(n, *BOX_HALF_XY, dev), _uniform(n, *BOX_HALF_XY, dev),
                       _uniform(n, *BOX_HALF_Z, dev)], -1)
    cyl_r, cyl_h = _uniform(n, *CYL_RADIUS, dev), _uniform(n, *CYL_HALF_H, dev)
    sizes[shape == 0, 0] = ball[shape == 0]
    sizes[shape == 1, 1] = box[shape == 1]
    sizes[shape == 2, 2, 0] = cyl_r[shape == 2]
    sizes[shape == 2, 2, 1] = cyl_h[shape == 2]
    sizes[:, 2, 2] = 0.0
    half[shape == 0] = ball[shape == 0]
    half[shape == 1] = box[shape == 1]
    half[shape == 2] = torch.stack([cyl_r, cyl_r, cyl_h], -1)[shape == 2]
    for k, gid in enumerate(gids):
        env.sim.model.geom_size[env_ids, gid] = sizes[:, k]
    _recompute_geom_bounds(env, env_ids.to(dtype=torch.int), asset_cfg)
    st["shape"][env_ids] = shape
    st["half"][env_ids] = half


def reset_seeing_object(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    nominal_xy: tuple = OBJECT_NOMINAL_XY,
    window_xy: float = WINDOW_START,
    center_frac: float = CENTER_FRAC,
    kick_prob: float = KICK_PROB,
    distractor_prob: float = DISTRACTOR_PROB,
) -> None:
    """Place the target (window around nominal_xy, robot yaw frame) and the distractor, store the object_spawn
    buffer Pollen's rewards read, draw this episode's camera, kick and initial standing measurement. Runs after the
    robot reset and seeing_object_shape (events run in insertion order)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    n, dev = len(env_ids), env.device
    st = _st(env)
    st["resets"] += 1                  # invalidates the per-step observation cache
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene["object"]
    dis: Entity = env.scene["distractor"]
    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    yaw = _yaw(root[:, 3:7])
    c, s = torch.cos(yaw), torch.sin(yaw)
    off = torch.tensor(nominal_xy, device=dev, dtype=torch.float).repeat(n, 1)
    spread = (torch.rand(n, 1, device=dev) >= center_frac).float()
    off += (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * float(window_xy) * spread
    off[:, 0] = off[:, 0].clamp(min=OBJECT_X_MIN)
    floor = env.scene.terrain.env_origins[env_ids, 2]
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = root[:, 0] + c * off[:, 0] - s * off[:, 1]
    pose[:, 1] = root[:, 1] + s * off[:, 0] + c * off[:, 1]
    pose[:, 2] = floor + st["half"][env_ids, 2] + 0.0005
    oyaw = (torch.rand(n, device=dev) * 2 - 1) * math.pi
    pose[:, 3], pose[:, 6] = torch.cos(oyaw / 2), torch.sin(oyaw / 2)
    obj.write_root_link_pose_to_sim(pose, env_ids)
    obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids)
    microduck_mdp._gg_buffer(env, "object_spawn", 3)[env_ids] = pose[:, :3]

    # Distractor: near the target in front of the toes, or parked behind the robot.
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.pi
    dist = _uniform(n, *DISTRACTOR_DIST, dev)
    doff = off + torch.stack([torch.cos(ang), torch.sin(ang)], -1) * dist.unsqueeze(-1)
    doff[:, 0] = doff[:, 0].clamp(min=OBJECT_X_MIN)
    near = torch.rand(n, device=dev) < distractor_prob
    doff[~near] = torch.tensor([-0.5, 0.0], device=dev)
    dpose = torch.zeros(n, 7, device=dev)
    dpose[:, 0] = root[:, 0] + c * doff[:, 0] - s * doff[:, 1]
    dpose[:, 1] = root[:, 1] + s * doff[:, 0] + c * doff[:, 1]
    dpose[:, 2] = floor + DISTRACTOR_RADIUS + 0.0005
    dpose[:, 3] = 1.0
    dis.write_root_link_pose_to_sim(dpose, env_ids)
    dis.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids)

    # Camera for this episode, and the coordinator's standing measurement as the first observation.
    dt = env.step_dt
    st["bias"][env_ids] = (torch.rand(n, 2, device=dev) * 2 - 1) * CAM_BIAS_XY
    st["delay"][env_ids] = torch.round(_uniform(n, *CAM_DELAY_S, dev) / dt).long().clamp(1, CAM_HISTORY - 1)
    st["drop"][env_ids] = _uniform(n, *CAM_DROP, dev)
    st["cam_offset"][env_ids] = torch.randint(0, CAM_PERIOD_STEPS, (n,), device=dev)
    st["stall_until"][env_ids] = -1
    st["buf_valid"][env_ids] = False
    st["buf_step"][env_ids] = -10 ** 6
    size = 2.0 * st["half"][env_ids, :2].max(-1).values
    st["last"][env_ids, :2] = off + st["bias"][env_ids] + torch.randn(n, 2, device=dev) * CAM_NOISE_XY
    st["last"][env_ids, 2] = size + torch.randn(n, device=dev) * CAM_NOISE_SIZE
    st["last_step"][env_ids] = -torch.round(_uniform(n, *INIT_AGE_S, dev) / dt).long()
    st["pinch_hist"][env_ids] = False
    st["lip_min"][env_ids] = 1.0
    kick = torch.rand(n, device=dev) < kick_prob
    period = env.command_manager.get_term("twist")._period if hasattr(
        env.command_manager.get_term("twist"), "_period") else 4.0
    kstep = torch.round(_uniform(n, *KICK_PHASE, dev) * period / dt).long()
    st["kick_step"][env_ids] = torch.where(kick, kstep, torch.full_like(kstep, -1))


def seeing_kick(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Every step: move and roll the object of the envs whose kick step is now; while it rolls free after a kick, move the
    object_spawn reference with it so object_pushed only charges what the robot does."""
    st = _st(env)
    st["lip_min"] = torch.minimum(st["lip_min"], _lip_height(env))
    t = env.episode_length_buf
    obj: Entity = env.scene["object"]
    dev = env.device
    now = torch.nonzero(t == st["kick_step"]).squeeze(-1)
    if len(now) > 0:
        n = len(now)
        ang = (torch.rand(n, device=dev) * 2 - 1) * math.pi
        u = torch.stack([torch.cos(ang), torch.sin(ang)], -1)
        pose = torch.cat([obj.data.root_link_pos_w[now], obj.data.root_link_quat_w[now]], -1).clone()
        pose[:, :2] += u * _uniform(n, *KICK_SHIFT, dev).unsqueeze(-1)
        vel = torch.zeros(n, 6, device=dev)
        vel[:, :2] = u * _uniform(n, *KICK_SPEED, dev).unsqueeze(-1)
        obj.write_root_link_pose_to_sim(pose, now)
        obj.write_root_link_velocity_to_sim(vel, now)
    free_steps = int(round(KICK_FREE_S / env.step_dt))
    free = (st["kick_step"] >= 0) & (t >= st["kick_step"]) & (t < st["kick_step"] + free_steps)
    if free.any():
        spawn = microduck_mdp._gg_buffer(env, "object_spawn", 3)
        spawn[free] = obj.data.root_link_pos_w[free]


def _lip_height(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Mean height of the two lips (sites lip_up, jaw_tip_bottom) above the env's floor, as lip_height_track reads them."""
    robot: Entity = env.scene["robot"]
    c = microduck_mdp._gg_cache(env)
    if "lip_site_ids" not in c:
        c["lip_site_ids"], _ = robot.find_sites(["lip_up", "jaw_tip_bottom"], preserve_order=True)
    ids = c["lip_site_ids"]
    z = 0.5 * (robot.data.site_pos_w[:, ids[0], 2] + robot.data.site_pos_w[:, ids[1], 2])
    return z - env.scene.terrain.env_origins[:, 2]


def _descended(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    return (_st(env)["lip_min"] < threshold).float()


def return_pose_after_descent(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                              std: float = 0.3, command_name: str = "twist", joint_indices: list | None = None,
                              hold_end: float = 0.35, rise_end: float = 0.60,
                              descent_threshold: float = DESCENT_GATE_M) -> torch.Tensor:
    """ground_pick_return_pose_phased, paid only after the lips went below descent_threshold this episode."""
    return _descended(env, descent_threshold) * microduck_mdp.ground_pick_return_pose_phased(
        env, asset_cfg=asset_cfg, std=std, command_name=command_name, joint_indices=joint_indices,
        hold_end=hold_end, rise_end=rise_end)


def return_upright_after_descent(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                 std: float = 0.4, command_name: str = "twist", hold_end: float = 0.35,
                                 rise_end: float = 0.60, descent_threshold: float = DESCENT_GATE_M) -> torch.Tensor:
    """ground_pick_return_upright_phased, paid only after the lips went below descent_threshold this episode."""
    return _descended(env, descent_threshold) * microduck_mdp.ground_pick_return_upright_phased(
        env, asset_cfg=asset_cfg, std=std, command_name=command_name, hold_end=hold_end, rise_end=rise_end)


# ── Observations ──────────────────────────────────────────────────────────────
def _camera_update(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Advance the simulated head camera one policy step (once per step, cached) and return the 6 body slots."""
    st = _st(env)
    key = (env.common_step_counter, st["resets"])
    if st["obs_key"] == key:
        return st["obs"]
    dev, n = env.device, env.num_envs
    t = env.episode_length_buf.long()
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene["object"]
    c = microduck_mdp._gg_cache(env)
    if "cam_site" not in c:
        ids, _ = robot.find_sites(["head_camera"])
        c["cam_site"] = ids[0]
    cam_p = robot.data.site_pos_w[:, c["cam_site"]]
    cam_q = robot.data.site_quat_w[:, c["cam_site"]]
    p = quat_apply_inverse(cam_q, obj.data.root_link_pos_w - cam_p)   # site frame: x optical axis, y left, z up
    depth = p[:, 0]
    visible = (depth > CAM_MIN_DEPTH) & (p[:, 1].abs() < depth * math.tan(CAM_HALF_FOV[0])) & \
              (p[:, 2].abs() < depth * math.tan(CAM_HALF_FOV[1]))
    tick = ((t + st["cam_offset"]) % CAM_PERIOD_STEPS) == 0
    stall_start = tick & (torch.rand(n, device=dev) < CAM_STALL_PROB) & (t >= st["stall_until"])
    st["stall_until"] = torch.where(
        stall_start, t + torch.round(_uniform(n, *CAM_STALL_S, dev) / env.step_dt).long(), st["stall_until"])
    stalled = t < st["stall_until"]
    detected = tick & visible & ~stalled & (torch.rand(n, device=dev) >= st["drop"])
    xy = _to_foot_frame(env, obj.data.root_link_pos_w, robot.data.root_link_pos_w, robot.data.root_link_quat_w)
    meas = torch.zeros(n, 3, device=dev)
    meas[:, :2] = xy + st["bias"] + torch.randn(n, 2, device=dev) * CAM_NOISE_XY
    meas[:, 2] = 2.0 * st["half"][:, :2].max(-1).values + torch.randn(n, device=dev) * CAM_NOISE_SIZE
    slot = t % CAM_HISTORY
    ar = torch.arange(n, device=dev)
    st["buf"][ar, slot] = meas
    st["buf_valid"][ar, slot] = detected
    st["buf_step"][ar, slot] = t
    # deliver the frame captured `delay` steps ago
    src = (t - st["delay"]) % CAM_HISTORY
    arrived = st["buf_valid"][ar, src] & (st["buf_step"][ar, src] == t - st["delay"])
    st["last"] = torch.where(arrived.unsqueeze(-1), st["buf"][ar, src], st["last"])
    st["last_step"] = torch.where(arrived, t - st["delay"], st["last_step"])
    age = ((t - st["last_step"]).float() * env.step_dt).clamp(0.0, AGE_MAX_S)
    pinched = microduck_mdp.grasp_state(env, "object")["pinched"]
    st["pinch_hist"] = torch.cat([st["pinch_hist"][:, 1:], pinched.unsqueeze(-1)], dim=-1)
    out = torch.zeros(n, 6, device=dev)
    out[:, :3] = st["last"]
    out[:, 3] = age
    out[:, 4] = st["pinch_hist"][:, 0].float()
    st["obs"], st["obs_key"] = out, key
    return out


def seeing_object_command(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """Actor and critic body slots: [x, y, size, age, holding, 0] as the camera and the jaw report them."""
    return _camera_update(env)[:, :dim]


def seeing_object_shape_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: active shape one-hot and its half extents."""
    st = _st(env)
    return torch.cat([torch.nn.functional.one_hot(st["shape"], len(SHAPES)).float(), st["half"]], dim=-1)


# ── Curriculum ────────────────────────────────────────────────────────────────
def seeing_window_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str = "reset_object",
    window_step: float = WINDOW_STEP,
    window_max: float = WINDOW_MAX,
    advance_success: float = WINDOW_ADVANCE_SUCCESS,
    min_iters: int = CURRICULUM_MIN_ITERS,
) -> dict:
    """Success = the object held at the end of the episode (grasp_state "held" before the reset). Widen the
    placement window by window_step when its EMA is above advance_success, at most every min_iters iterations."""
    st = _st(env)
    ev = env.event_manager.get_term_cfg(event_name)
    if env_ids is not None and len(env_ids) > 0:
        gs = microduck_mdp._gg_cache(env).get("grasp_state")
        if gs is not None:
            rate = float(gs["held"][env_ids].float().mean())
            st["success"] = (1 - SUCCESS_EMA_ALPHA) * st["success"] + SUCCESS_EMA_ALPHA * rate
    it = env.common_step_counter // STEPS_PER_ITER
    if (st["success"] > advance_success and ev.params["window_xy"] < window_max - 1e-9
            and it - st["window_it"] >= min_iters):
        ev.params["window_xy"] = min(ev.params["window_xy"] + window_step, window_max)
        st["window_it"] = it
    return {"window": torch.tensor(ev.params["window_xy"]), "success_ema": torch.tensor(st["success"])}


# ── Env cfg ───────────────────────────────────────────────────────────────────
def make_seeing_pick_env_cfg(variant: str = "plugged", play: bool = False):
    cfg = make_cmu_grasp_env_cfg(make_microduck_static_grasp_env_cfg, variant, ball=False, play=play, hover=True)
    cfg.scene.entities = {**cfg.scene.entities, "object": SEEING_OBJECT_CFG, "distractor": DISTRACTOR_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 80)

    geoms = SceneEntityCfg("object", geom_names=GEOM_NAMES)
    cfg.events["object_size"] = EventTermCfg(func=seeing_object_shape, mode="reset", params={"asset_cfg": geoms})
    cfg.events["object_mass"].params["alpha_range"] = (0.5 * math.log(OBJECT_MASS_RANGE[0] / OBJECT_MASS),
                                                       0.5 * math.log(OBJECT_MASS_RANGE[1] / OBJECT_MASS))
    cfg.events["object_friction"] = EventTermCfg(
        func=dr.geom_friction, mode="reset",
        params={"asset_cfg": geoms, "operation": "abs", "axes": [0], "ranges": OBJECT_FRICTION, "shared_random": True},
    )
    cfg.events["reset_object"] = EventTermCfg(
        func=reset_seeing_object, mode="reset",
        params={"window_xy": WINDOW_OPERATING if play else WINDOW_START},
    )
    cfg.events["seeing_kick"] = EventTermCfg(func=seeing_kick, mode="interval", interval_range_s=(0.0, 0.0))
    for name in GATED_RETURN_TERMS:
        cfg.rewards[name].func = (return_upright_after_descent if name == "return_upright"
                                  else return_pose_after_descent)
    cfg.curriculum.pop("object_noise", None)
    if not play:
        cfg.curriculum["seeing_window"] = CurriculumTermCfg(func=seeing_window_curriculum,
                                                            params={"event_name": "reset_object"})

    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=seeing_object_command,
                                                                           params={"dim": 6})
    critic = cfg.observations["critic"].terms
    critic["object_size"] = ObservationTermCfg(func=seeing_object_shape_obs)
    return cfg


def seeing_pick_rl_cfg(base):
    rl = copy.deepcopy(base)
    rl.experiment_name = "seeing_pick"
    rl.run_name = "seeing_pick"
    rl.max_iterations = 3000
    return rl
