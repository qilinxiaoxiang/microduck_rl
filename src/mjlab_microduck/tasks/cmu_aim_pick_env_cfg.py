"""CMU aiming crouch ("aim pick"): Pollen's ground-pick crouch with an object the beak has to reach.

MicroDuck final goal (D-019) steps 3 and 4: crouch while keeping balance and bring the beak to the object so the
scripted jaw (`cmu/mac/jaw_ctl.py`, current mode, outside the policy) can grip it. Pollen's ground pick
(`microduck_ground_pick_env_cfg.py`, deployed as groundpick61) bows blind: no object in the scene, the mouth tip
pulled toward any floor point. This env is that cfg with five changes:

1. Scene: one free box (the orange printed part: 3 x 3 cm footprint, 4-9 cm tall per episode, 25 g), placed at
   reset in the robot's yaw frame around the spot where groundpick61's beak lands (measured on this model,
   OBJECT_NOMINAL_XY), with a placement window that widens from +-1 cm to +-6 cm (the +-5 cm a walk leaves, plus
   a margin) each time the hold-phase beak error falls below WINDOW_ADVANCE_ERR (error-gated, as Pedipulate and
   ULC widen their goal ranges). The box never spawns under or against the toes (OBJECT_X_MIN).
2. Observation: the object's (x, y) on the floor and its height go into body_command slots 0-2 (body_x, body_y,
   body_z), which are constant zero in the ground pick and in the runtime (`walk_brain.py` concatenates
   `np.zeros(6)`). The actor still reads 61 numbers in the same order; only the runtime's six body numbers change
   for this policy. Frame = the runtime's foot frame (`cmu/mac/target_ground.py`): origin on the floor below
   trunk_base, x forward along the trunk heading, y left. The value is measured once at reset (the camera measures
   at stand, before the crouch hides the object) with a per-episode bias of +-OBS_NOISE_XY / +-OBS_NOISE_H (camera
   projection error) and held for the episode. The critic also gets the true grasp point, live.
3. Reward: mouth_ground_proximity (beak to any floor point) is replaced by three terms on the beak tip's distance d
   to the grasp point, GRIP_DEPTH below the centre of the box's top face (an open beak has to straddle the top to
   grip it). aim_reach = exp(-d / AIM_REACH_SCALE) under the ground pick's descent blend, descent and hold only;
   aim_hold = exp(-d / scale) in the low hold only, where the jaw grips, with the scale tightened from 3 cm to 1 cm
   as the hold error drops (UMI on Legs' error-keyed sigma curriculum); aim_progress pays each new closest approach
   in the descent and hold (Whole-Body End-Effector Pose Tracking's progress reward, against oscillation; bounded
   by the start distance, so not farmable). object_displacement charges pushing or kicking the box away. Sources
   and the published recipes: cmu/aim_pick/README.md (literature pass of 2026-10-21), which also holds the reward
   audit (cmu/aim_pick/audit.py) that any change to these terms must pass again.
4. Head-object contact is off (HEAD_OBJECT_CONTACT): the model has no jaw joint, so the closed beak would land on
   the box's top instead of around it; the head geoms get their own contype bit, the box still collides with the
   floor, legs and trunk, and the head still collides with the floor (head_impact_penalty) and the legs.
5. Command: every episode starts standing at phase 0 (randomize_phase False, like the runtime button), and the
   phase freezes at HOLD_PHASE for a random 0-HOLD_EXTRA_S s once per episode, as the runtime does while the jaw
   grips (`walk_brain.py` holds the wanted phase).

The jaw itself, the object in the mouth during the rise (only Pollen's 10-40 g payload force) and the camera are not
simulated. Policy contract: 61 inputs, 14 outputs, action scale 1, the ground pick's 4 s period, so the runtime's
ground-pick mode runs it with the body slots filled.
"""
from __future__ import annotations

import dataclasses
import math

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import (
    DESCENT_END,
    GP_PERIOD,
    HOLD_END,
    RISE_END,
    MicroduckGroundPickRlCfg,
    make_microduck_ground_pick_env_cfg,
)

# ── Object ────────────────────────────────────────────────────────────────────
OBJECT_NAME = "aim_object"
OBJECT_HALF_XY = 0.015                 # 3 x 3 cm footprint
OBJECT_NOMINAL_H = 0.065               # height scaled per episode by OBJECT_H_SCALE
OBJECT_H_SCALE = (0.04 / OBJECT_NOMINAL_H, 0.09 / OBJECT_NOMINAL_H)   # 4-9 cm tall
OBJECT_MASS = 0.025                    # kg (at most 40 g; it only matters when pushed)
OBJECT_FRICTION = 0.6                  # PLA on the floor
# groundpick61's beak over the low hold [DESCENT_END, HOLD_END) in this env (cmu/aim_pick/audit.py "measure", 256
# envs, 2026-10-21; reset yaw frame of trunk_base): Pollen's masses x 0.070, y 0.005, z 0.031 m (sd 6 mm), CMU
# plugged x 0.067, y 0.0075, z 0.027 m (sd 11 mm). At stand it sits at x 0.078, z 0.217. The nominal object centre
# is the mean landing spot of the two.
OBJECT_NOMINAL_XY = (0.069, 0.006)
# Soles at HOME span x -0.020..0.034 m and |y| 0.021..0.062 m (robot_groundcontact.xml FK): the box (half diagonal
# 0.021 m at any yaw) spawns at least 4 mm clear of the toes.
OBJECT_X_MIN = 0.060
OBJECT_YAW_RANGE = (-math.pi / 4, math.pi / 4)   # relative to the robot heading
OBS_NOISE_XY = 0.01                    # per-episode camera projection bias (m)
OBS_NOISE_H = 0.005                    # per-episode object-height error (m)
# Placement window half-width (m): WINDOW_START, + WINDOW_STEP whenever the hold error EMA is below
# WINDOW_ADVANCE_ERR, at most every CURRICULUM_MIN_ITERS iterations, up to WINDOW_MAX. CENTER_FRAC of the episodes
# put the box exactly on the nominal spot (centre-weighted sampling, as Whole-Body EE Pose Tracking bins its goals).
WINDOW_START, WINDOW_STEP, WINDOW_MAX = 0.01, 0.01, 0.06
WINDOW_OPERATING = 0.05                # play / evaluation window: the error a walk leaves
WINDOW_ADVANCE_ERR = 0.015
CENTER_FRAC = 0.2
CURRICULUM_MIN_ITERS = 50
STEPS_PER_ITER = 24                    # MicroduckGroundPickRlCfg.num_steps_per_env
HEAD_OBJECT_CONTACT = False
HEAD_CONTYPE = 8                       # head geoms' own contype bit (conaffinity stays 1)
OBJECT_CONTYPE = 4                     # box contype; conaffinity 1 -> floor, legs, trunk, not the head

# ── Aim reward ────────────────────────────────────────────────────────────────
GRIP_DEPTH = 0.015                     # grasp point below the top face centre (m)
GRASP_Z_MIN = 0.02                     # never below 2 cm over the floor
AIM_REACH_SCALE = 0.05                 # exp(-d / 0.05): 0.37 at 5 cm, 0.02 from the standing beak (~19 cm)
AIM_REACH_WEIGHT = 3.0                 # replaces mouth_ground_proximity (3.0)
AIM_HOLD_SCALES = (0.03, 0.02, 0.015, 0.01)   # next one when the hold error EMA < 0.8 x the current scale
# Sized by the reward audit (D-016 gate): the ground pick's own terms pay ~7.9 per second to any crouch that stands
# back up, aimed or not (feet_grounded 3.0, return poses 2.9, return_upright 1.0, mouth_perpendicular 0.9). At weight
# 3 a blind bow still collected 0.94 of an aimed one. aim_hold is the term that tells them apart per unit of reward
# mass (paid only in the commanded hold, so arriving early earns nothing); at 40 the aimed crouch earns ~4.3 per
# second from the aim terms (~35 % of its reward) and the blind bow ~0.76 of the aimed one.
AIM_HOLD_WEIGHT = 40.0
AIM_PROGRESS_WEIGHT = 5.0              # per metre of new closest approach (at most ~1 per crouch)
HOLD_ERR_EMA_ALPHA = 0.01              # per env step, over the envs in the hold (~4 iterations of memory)
OBJECT_DISPLACEMENT_MAX = 0.05         # displacement cost saturates at 5 cm
OBJECT_DISPLACEMENT_WEIGHT = -2.0

# ── Command and episode ──────────────────────────────────────────────────────
HOLD_PHASE = 0.40                      # inside the low hold [DESCENT_END, HOLD_END)
HOLD_EXTRA_S = (0.0, 1.5)
EPISODE_LENGTH_S = 6.0                 # one crouch (4-5.5 s) plus the start of the next


# ── Object entity ─────────────────────────────────────────────────────────────

def get_aim_object_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name=OBJECT_NAME, pos=[0.3, 0.0, OBJECT_NOMINAL_H / 2])
    body.add_freejoint(name=f"{OBJECT_NAME}_free")
    body.add_geom(
        name=f"{OBJECT_NAME}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[OBJECT_HALF_XY, OBJECT_HALF_XY, OBJECT_NOMINAL_H / 2],
        mass=OBJECT_MASS,
        friction=[OBJECT_FRICTION, 0.005, 0.0001],
        rgba=[0.2, 0.75, 0.1, 1.0],   # not the duck's print orange (PICKUP.md: the detector must reject the duck)
        contype=1 if HEAD_OBJECT_CONTACT else OBJECT_CONTYPE,
        conaffinity=1,
    )
    return spec


AIM_OBJECT_CFG = EntityCfg(
    spec_fn=get_aim_object_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.0, OBJECT_NOMINAL_H / 2)),
)


def head_without_object_contact(spec_fn):
    """Wrap a robot spec_fn: the head (jaw_soft) collision geoms get contype HEAD_CONTYPE, so the box (conaffinity 1,
    contype OBJECT_CONTYPE) ignores them while floor and leg contacts stay (their contype 1 meets the head's
    conaffinity 1)."""
    def fn() -> mujoco.MjSpec:
        spec = spec_fn()
        for g in spec.body("jaw_soft").geoms:
            if g.contype == 1:
                g.contype = HEAD_CONTYPE
        return spec
    return fn


# ── Per-env state ─────────────────────────────────────────────────────────────

def _aim(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_aim_pick", None)
    if st is None:
        n, dev = env.num_envs, env.device
        st = env._aim_pick = {
            "cmd": torch.zeros(n, 3, device=dev),       # observed (x, y, height), foot frame, with bias
            "spawn_xy": torch.zeros(n, 2, device=dev),  # world xy of the box at reset
            "d_min": torch.zeros(n, device=dev),        # closest beak-grasp distance this cycle (aim_progress)
            "gid": None,
            "hold_err": 0.1,                            # EMA of the hold-phase beak-grasp distance (m)
            "window_it": 0, "scale_it": 0,              # iteration of the last curriculum change
        }
    return st


def _object_geom_id(env: ManagerBasedRlEnv, object_name: str) -> int:
    st = _aim(env)
    if st["gid"] is None:
        obj: Entity = env.scene[object_name]
        st["gid"] = int(obj.indexing.geom_ids[0])
    return st["gid"]


def _object_half_height(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
    gs = env.sim.model.geom_size[:, _object_geom_id(env, object_name), 2]
    return gs.expand(env.num_envs) if gs.numel() == 1 else gs


def aim_grasp_point_w(env: ManagerBasedRlEnv, object_name: str = OBJECT_NAME, grip_depth: float = GRIP_DEPTH,
                      grasp_z_min: float = GRASP_Z_MIN) -> torch.Tensor:
    """World position GRIP_DEPTH below the centre of the box's top face (live pose), never below GRASP_Z_MIN."""
    obj: Entity = env.scene[object_name]
    h = _object_half_height(env, object_name)
    up = torch.zeros(env.num_envs, 3, device=env.device)
    up[:, 2] = h
    top = obj.data.root_link_pos_w + quat_apply(obj.data.root_link_quat_w, up)
    floor = env.scene.terrain.env_origins[:, 2]
    grasp = top.clone()
    grasp[:, 2] = torch.maximum(top[:, 2] - grip_depth, floor + grasp_z_min)
    return grasp


# ── Command ───────────────────────────────────────────────────────────────────

class AimPickPhaseCommand(microduck_mdp.GroundPickPhaseCommand):
    """GroundPickPhaseCommand that freezes the phase at hold_phase for a random duration once per episode."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._hold_left = torch.zeros(self.num_envs, device=self.device)
        self._hold_phase = float(cfg.hold_phase)
        self._hold_range = tuple(cfg.hold_extra_s)

    def compute(self, dt: float) -> None:
        holding = self._hold_left > 0
        at_hold = holding & (self._gp_phase >= self._hold_phase) & (self._gp_phase < self._hold_phase + 0.05)
        nxt = self._gp_phase + dt / self._period
        crossing = holding & (self._gp_phase < self._hold_phase) & (nxt >= self._hold_phase)
        nxt = torch.where(crossing, torch.full_like(nxt, self._hold_phase), nxt)
        nxt = torch.where(at_hold, self._gp_phase, nxt)
        self._hold_left = torch.where(at_hold, (self._hold_left - dt).clamp(min=0.0), self._hold_left)
        self._gp_phase = nxt % 1.0
        self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * self._gp_phase)
        self.vel_command_b[:, 2] = 0.0

    def reset(self, env_ids: torch.Tensor | None) -> dict:
        out = super().reset(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            lo, hi = self._hold_range
            self._hold_left[env_ids] = torch.rand(len(env_ids), device=self.device) * (hi - lo) + lo
        return out


@dataclasses.dataclass(kw_only=True)
class AimPickPhaseCommandCfg(microduck_mdp.GroundPickPhaseCommandCfg):
    class_type: type = AimPickPhaseCommand
    hold_phase: float = HOLD_PHASE
    hold_extra_s: tuple = HOLD_EXTRA_S

    def build(self, env: ManagerBasedRlEnv) -> "AimPickPhaseCommand":
        return AimPickPhaseCommand(self, env)


# ── Events and curriculum ────────────────────────────────────────────────────

def reset_aim_object(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    nominal_xy: tuple = OBJECT_NOMINAL_XY,
    window_xy: float = WINDOW_START,
    center_frac: float = CENTER_FRAC,
    x_min: float = OBJECT_X_MIN,
    yaw_range: tuple = OBJECT_YAW_RANGE,
    obs_noise_xy: float = OBS_NOISE_XY,
    obs_noise_h: float = OBS_NOISE_H,
    object_name: str = OBJECT_NAME,
) -> None:
    """Place the box in the robot's yaw frame (nominal +- window, x >= x_min; center_frac of the envs exactly on the
    nominal spot) and store the observed command.

    Reads the robot root from qpos (root_link_pos_w lags until the next forward()), so it must run after
    reset_base and after randomize_object_size (events run in insertion order)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    robot: Entity = env.scene["robot"]
    obj: Entity = env.scene[object_name]
    st = _aim(env)
    n = len(env_ids)

    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    qw, qx, qy, qz = root[:, 3], root[:, 4], root[:, 5], root[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    c, s = torch.cos(yaw), torch.sin(yaw)

    off = torch.tensor(nominal_xy, device=env.device, dtype=torch.float).repeat(n, 1)
    spread = (torch.rand(n, 1, device=env.device) >= center_frac).float()
    off += (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * float(window_xy) * spread
    off[:, 0] = off[:, 0].clamp(min=x_min)
    h = 2.0 * _object_half_height(env, object_name)[env_ids]

    pose = torch.zeros(n, 7, device=env.device)
    pose[:, 0] = root[:, 0] + c * off[:, 0] - s * off[:, 1]
    pose[:, 1] = root[:, 1] + s * off[:, 0] + c * off[:, 1]
    pose[:, 2] = env.scene.terrain.env_origins[env_ids, 2] + h / 2
    oyaw = yaw + torch.empty(n, device=env.device).uniform_(*yaw_range)
    z = torch.zeros(n, 3, device=env.device)
    z[:, 2] = 1.0
    pose[:, 3:7] = quat_from_angle_axis(oyaw, z)
    obj.write_root_link_pose_to_sim(pose, env_ids)
    obj.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids)

    st["spawn_xy"][env_ids] = pose[:, :2]
    st["cmd"][env_ids, :2] = off + (torch.rand(n, 2, device=env.device) * 2.0 - 1.0) * obs_noise_xy
    st["cmd"][env_ids, 2] = h + (torch.rand(n, device=env.device) * 2.0 - 1.0) * obs_noise_h


def aim_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str = "reset_aim_object",
    hold_term: str = "aim_hold",
    window_step: float = WINDOW_STEP,
    window_max: float = WINDOW_MAX,
    window_advance_err: float = WINDOW_ADVANCE_ERR,
    hold_scales: tuple = AIM_HOLD_SCALES,
    min_iters: int = CURRICULUM_MIN_ITERS,
    steps_per_iter: int = STEPS_PER_ITER,
) -> dict:
    """Error-gated curricula on the hold error EMA: widen the placement window by window_step when the error is below
    window_advance_err, tighten aim_hold's scale to the next of hold_scales when the error is below 0.8 x the current
    scale; each at most every min_iters iterations. Mutates the live manager term cfgs (managers deepcopy env.cfg)."""
    del env_ids
    st = _aim(env)
    it = env.common_step_counter // steps_per_iter
    ev = env.event_manager.get_term_cfg(event_name)
    rw = env.reward_manager.get_term_cfg(hold_term)
    err = st["hold_err"]
    if err < window_advance_err and ev.params["window_xy"] < window_max - 1e-9 and it - st["window_it"] >= min_iters:
        ev.params["window_xy"] = min(ev.params["window_xy"] + window_step, window_max)
        st["window_it"] = it
    scale = rw.params["scale"]
    tighter = [s for s in hold_scales if s < scale - 1e-9]
    if tighter and err < 0.8 * scale and it - st["scale_it"] >= min_iters:
        rw.params["scale"] = tighter[0]
        st["scale_it"] = it
    return {"window": torch.tensor(ev.params["window_xy"]), "hold_scale": torch.tensor(rw.params["scale"]),
            "hold_err": torch.tensor(err)}


# ── Observations ──────────────────────────────────────────────────────────────

def aim_object_command(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """body_command slots: [object x, object y, object height, 0, 0, 0] (foot frame, m), held from reset."""
    out = torch.zeros(env.num_envs, dim, device=env.device)
    out[:, :3] = _aim(env)["cmd"]
    return out


def aim_grasp_point_rel(env: ManagerBasedRlEnv, object_name: str = OBJECT_NAME) -> torch.Tensor:
    """Critic: the true grasp point relative to trunk_base, in the trunk's yaw frame, live."""
    robot: Entity = env.scene["robot"]
    rel = aim_grasp_point_w(env, object_name) - robot.data.root_link_pos_w
    q = robot.data.root_link_quat_w
    yaw = torch.atan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], dim=-1)


# ── Rewards ───────────────────────────────────────────────────────────────────

def tip_to_grasp(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
                 object_name: str = OBJECT_NAME) -> torch.Tensor:
    """Distance (m) from the beak tip to the grasp point."""
    tip = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids[0], :]
    return (tip - aim_grasp_point_w(env, object_name)).norm(dim=-1)


def aim_reach_phased(
    env: ManagerBasedRlEnv,
    scale: float = AIM_REACH_SCALE,
    command_name: str = "twist",
    descent_end: float = DESCENT_END,
    hold_end: float = HOLD_END,
    rise_end: float = RISE_END,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    object_name: str = OBJECT_NAME,
) -> torch.Tensor:
    """exp(-d / scale) under the ground pick's descent blend (0 standing -> 1 low), in the descent and hold only."""
    phase = microduck_mdp._gp_phase(env, command_name)
    gate = microduck_mdp.phase_pose_blend(phase, descent_end, hold_end, rise_end) * (phase < hold_end).float()
    return gate * torch.exp(-tip_to_grasp(env, asset_cfg, object_name) / scale)


def aim_hold_precision(
    env: ManagerBasedRlEnv,
    scale: float = AIM_HOLD_SCALES[0],
    command_name: str = "twist",
    descent_end: float = DESCENT_END,
    hold_end: float = HOLD_END,
    ema_alpha: float = HOLD_ERR_EMA_ALPHA,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    object_name: str = OBJECT_NAME,
) -> torch.Tensor:
    """exp(-d / scale) paid only in the low hold [descent_end, hold_end), where the jaw grips. Also keeps the hold
    error EMA that aim_curriculum reads."""
    phase = microduck_mdp._gp_phase(env, command_name)
    in_hold = (phase >= descent_end) & (phase < hold_end)
    d = tip_to_grasp(env, asset_cfg, object_name)
    if in_hold.any():
        st = _aim(env)
        st["hold_err"] = (1.0 - ema_alpha) * st["hold_err"] + ema_alpha * float(d[in_hold].mean())
    return in_hold.float() * torch.exp(-d / scale)


def aim_progress(
    env: ManagerBasedRlEnv,
    command_name: str = "twist",
    hold_end: float = HOLD_END,
    restart_phase: float = 0.02,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
    object_name: str = OBJECT_NAME,
) -> torch.Tensor:
    """Rate (m/s) at which the beak beats its closest approach to the grasp point in this cycle's descent and hold;
    the record restarts in the first restart_phase of every cycle. Sums to (start - closest) distance per crouch."""
    st = _aim(env)
    phase = microduck_mdp._gp_phase(env, command_name)
    d = tip_to_grasp(env, asset_cfg, object_name)
    restart = phase < restart_phase
    active = (~restart) & (phase < hold_end)
    gain = torch.where(active, (st["d_min"] - d).clamp(min=0.0), torch.zeros_like(d))
    st["d_min"] = torch.where(restart, d, torch.where(active, torch.minimum(st["d_min"], d), st["d_min"]))
    return gain / env.step_dt


def object_displacement_cost(env: ManagerBasedRlEnv, max_dist: float = OBJECT_DISPLACEMENT_MAX,
                             object_name: str = OBJECT_NAME) -> torch.Tensor:
    """Horizontal distance the box moved since reset, saturated at max_dist, as a fraction in [0, 1] (>= 0: use a
    negative weight)."""
    obj: Entity = env.scene[object_name]
    d = (obj.data.root_link_pos_w[:, :2] - _aim(env)["spawn_xy"]).norm(dim=-1)
    return d.clamp(max=max_dist) / max_dist


# ── Env cfg ───────────────────────────────────────────────────────────────────

def make_aim_pick_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_ground_pick_env_cfg(play=play, rough=rough)

    robot = cfg.scene.entities["robot"]
    if not HEAD_OBJECT_CONTACT:
        robot = dataclasses.replace(robot, spec_fn=head_without_object_contact(robot.spec_fn))
    # Robot stays the first entity (reset events write its root at qpos[:, 0:7]).
    cfg.scene.entities = {"robot": robot, OBJECT_NAME: AIM_OBJECT_CFG}
    cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 50)
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Rewards ──
    del cfg.rewards["mouth_ground_proximity"]
    gp = dict(descent_end=DESCENT_END, hold_end=HOLD_END, command_name="twist")
    tip = dict(asset_cfg=SceneEntityCfg("robot", site_names=["mouth_tip"]), object_name=OBJECT_NAME)
    cfg.rewards["aim_reach"] = RewardTermCfg(
        func=aim_reach_phased, weight=AIM_REACH_WEIGHT,
        params={**gp, **tip, "rise_end": RISE_END, "scale": AIM_REACH_SCALE},
    )
    cfg.rewards["aim_hold"] = RewardTermCfg(
        func=aim_hold_precision, weight=AIM_HOLD_WEIGHT,
        params={**gp, **tip, "scale": AIM_HOLD_SCALES[-1 if play else 0], "ema_alpha": HOLD_ERR_EMA_ALPHA},
    )
    cfg.rewards["aim_progress"] = RewardTermCfg(
        func=aim_progress, weight=AIM_PROGRESS_WEIGHT,
        params={"command_name": "twist", "hold_end": HOLD_END, **tip},
    )
    cfg.rewards["object_displacement"] = RewardTermCfg(
        func=object_displacement_cost, weight=OBJECT_DISPLACEMENT_WEIGHT,
        params={"max_dist": OBJECT_DISPLACEMENT_MAX, "object_name": OBJECT_NAME},
    )

    # ── Observations: object in body_command slots 0-2; critic also sees the true grasp point ──
    for group in ("actor", "critic"):
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(func=aim_object_command, params={"dim": 6})
    cfg.observations["critic"].terms["aim_grasp_point"] = ObservationTermCfg(
        func=aim_grasp_point_rel, params={"object_name": OBJECT_NAME},
    )

    # ── Command: start standing, freeze once in the low hold ──
    cmd = cfg.commands["twist"]
    cfg.commands["twist"] = AimPickPhaseCommandCfg(
        **{**vars(cmd), "class_type": AimPickPhaseCommand, "period": GP_PERIOD, "randomize_phase": False,
           "hold_phase": HOLD_PHASE, "hold_extra_s": HOLD_EXTRA_S}
    )

    # ── Events: object size, then placement (after reset_base: insertion order) ──
    cfg.events["randomize_object_size"] = EventTermCfg(
        func=dr.geom_size, mode="reset",
        params={"asset_cfg": SceneEntityCfg(OBJECT_NAME, geom_names=(f"{OBJECT_NAME}_geom",)),
                "operation": "scale", "axes": [2], "ranges": OBJECT_H_SCALE},
    )
    cfg.events["reset_aim_object"] = EventTermCfg(
        func=reset_aim_object, mode="reset",
        params={"nominal_xy": OBJECT_NOMINAL_XY, "window_xy": WINDOW_OPERATING if play else WINDOW_START,
                "center_frac": 0.0 if play else CENTER_FRAC, "x_min": OBJECT_X_MIN, "yaw_range": OBJECT_YAW_RANGE,
                "obs_noise_xy": OBS_NOISE_XY, "obs_noise_h": OBS_NOISE_H, "object_name": OBJECT_NAME},
    )
    if not play:
        cfg.curriculum["aim"] = CurriculumTermCfg(
            func=aim_curriculum,
            params={"event_name": "reset_aim_object", "hold_term": "aim_hold", "window_step": WINDOW_STEP,
                    "window_max": WINDOW_MAX, "window_advance_err": WINDOW_ADVANCE_ERR, "hold_scales": AIM_HOLD_SCALES,
                    "min_iters": CURRICULUM_MIN_ITERS, "steps_per_iter": STEPS_PER_ITER},
        )
    return cfg


MicroduckAimPickRlCfg = dataclasses.replace(
    MicroduckGroundPickRlCfg, experiment_name="aim_pick", run_name="aim_pick",
)

# (task id, CMU mass variant or None for Pollen's masses)
AIM_PICK_TASKS = (
    ("Mjlab-AimPick-Flat-MicroDuck", None),
    ("Mjlab-AimPick-Flat-MicroDuck-CMU-Plugged", "plugged"),
    ("Mjlab-AimPick-Flat-MicroDuck-CMU-Battery", "battery"),
)


def make_cmu_aim_pick_env_cfg(variant: str | None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """The aim-pick env on Pollen's masses (variant None) or on a CMU as-built mass variant (cmu/model)."""
    cfg = make_aim_pick_env_cfg(play=play)
    if variant is None:
        return cfg
    from .cmu_asbuilt import load_apply_masses

    apply_masses = load_apply_masses()
    parts = apply_masses.variant_parts(variant)
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn

    def spec_fn() -> mujoco.MjSpec:
        spec = base_spec_fn()
        report = apply_masses.apply_parts_to_spec(spec, parts)
        print(f"[cmu:{variant}] total mass {report['total_before_g']:.1f} g -> {report['total_after_g']:.1f} g")
        return spec

    cfg.scene.entities = {**cfg.scene.entities, "robot": dataclasses.replace(robot, spec_fn=spec_fn)}
    return cfg


def aim_pick_rl_cfg(variant: str | None):
    suffix = f"_cmu_{variant}" if variant else ""
    return dataclasses.replace(MicroduckAimPickRlCfg, experiment_name=f"aim_pick{suffix}", run_name=f"aim_pick{suffix}")


def register_aim_pick_tasks(runner_cls) -> None:
    for task_id, variant in AIM_PICK_TASKS:
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_cmu_aim_pick_env_cfg(variant),
            play_env_cfg=make_cmu_aim_pick_env_cfg(variant, play=True),
            rl_cfg=aim_pick_rl_cfg(variant),
            runner_cls=runner_cls,
        )
