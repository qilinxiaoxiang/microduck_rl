"""Ball dock: the closed-loop half of the split pick (2026-11-18 scope split, `dock-kneel-draft.md`).

A teammate's approach leaves the robot near the ball. This policy brings it to the one stance the open-loop kneel-rise
(`cmu_kneel_rise_env_cfg.py`) is trained from: standing still with the ball at the static pose's landing spot
(OBJECT_NOMINAL_XY = 8.7 cm ahead of the foot-frame origin, centred), seen only through the head camera.

Structure: a COMMANDER over the frozen deployed walk61. The learned policy outputs 6 numbers per 50 Hz step:
walk velocity (vx, vy, wz) and head aim (head_pitch, head_yaw, neck_pitch deltas from HOME). The action term writes them into
walk61's command slots and runs walk61 (the runtime's ONNX, normaliser baked in) on its own 61-D observation group
`walk`, computed as walk61 was trained (noise, delays). walk61 is exactly the gait that has walked on the robot, and
the runtime already drives it with (vx, vy, wz, head) from walk_brain.py, so the commander plugs in at that boundary.
The commander is NOT a 61-D joint policy and is not hot-swapped with them.

Commander observation (`actor`, 20-D, all available on the robot): the camera block [u, v, width, age, visible]
(the pixel pick's U20CAM model: fisheye, mount DR, 8-10 fps, 0.1-0.3 s delay, drops, beak occlusion), the measured
neck / head pitch / head yaw, projected gravity, gyro, and its own last action.

Rewards: potential-based progress on the ball's distance to the landing spot (pays getting closer, holding pays 0,
unfarmable), a multiplicative docked term (position x stillness) paid per step while docked (fixed-length episodes,
so docking sooner pays more steps), ball-kick and fall penalties, ball-in-picture to keep the gaze on it, command
smoothness. Starts: the ball 10-35 cm ahead within +-45 deg (the approach hand-over; widened by curriculum from a
near set), the head aimed at it, walking at 0-0.15 m/s.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import onnx
import torch
import torch.nn.functional as F
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.velocity import mdp
from onnx import numpy_helper

from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as pix
from mjlab_microduck.tasks.cmu_asbuilt import CURRENT, make_cmu_velocity_env_cfg
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import OBJECT_NOMINAL_XY
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg

REPO = Path(__file__).resolve().parents[3]
WALK_ONNX = REPO / "cmu" / "whole" / "runtime-overlay" / "models" / "walk61.onnx"   # deployed (robot-condition.json)
# The nudge (cmu_walk_nudge_env_cfg.py, wn-b @ 750) as the low level instead of walk61 (cmu/kneel/RESULTS.md round 5):
# bd-a2's failures are not spawn-dependent but end-of-dock limit cycles (12 % still walking at 8 s, half of them
# docked earlier) from walk61's zero-command drift and dead zone; under a scripted P-command on the true error the
# nudge docks 99.8 % kneel-ready against walk61's 71 %. Its twist contract is 8 x the remaining error.
NUDGE_LOW_ONNX = REPO / "cmu" / "kneel" / "policies" / "wn-b_750.onnx"
NUDGE_SN_ONNX = REPO / "cmu" / "kneel" / "policies" / "wn-sn1.onnx"    # wn-b retrained under servo noise level 1
# bd-c: bd-b fine-tuned to stop with the ball this much beyond the landing spot. Replaying bd-b@2000's end errors on
# the ball, ks-f@750 holds 0.667 as docked, 0.738 / 0.765 / 0.753 shifted +0.5 / +0.75 / +1.0 cm: the kneel's
# ball curve peaks at dx 0..+1 cm and falls off a cliff below -1 cm, where bd-b's near tail sits (round 6).
DOCK_TARGET_DX = 0.0075
# bd-wide: the retry dock. A missed bite leaves the robot standing in ~20 % of episodes with the ball mostly outside
# the dock's spawn range (ks-i @ 500: 42 % closer than 10 cm, 32 % beyond 35 cm, 73 % beyond +-45 deg, median 68 deg),
# where bd-sn1 ends near the spot 0.09-0.54 (< 10 cm) and 0.35-0.62 (60-90 deg). Half the resets draw from there.
WIDE_FRAC = 0.5
WIDE_R = (0.04, 0.50)

# ── Commander action: tanh-free, clipped to [-1, 1], then scaled ──
# walk61's trained ranges. It has a dead zone: in this env it does not step below ~0.2 m/s and moves ~0.1 m/s at a
# 0.25 command (cmu/kneel/RESULTS.md), so fine positioning has to come from short pulses, not small commands.
VX_MAX, VY_MAX, WZ_MAX = 0.40, 0.30, 1.0        # m/s, m/s, rad/s
HEAD_PITCH_RANGE = (0.0, 1.10)                  # delta from HOME; walk61's head-command range
HEAD_YAW_MAX = 0.8
NECK_RANGE = (-0.8, 0.0)                        # delta from HOME (negative bows): keeps the ball in the picture at the
                                                # landing spot, which leaves the bottom edge at head pitch alone
ACTION_DIM = 6

# ── Ball ──
BALL_RADIUS = 0.0065                            # the 1.06-1.44 cm printed ball
BALL_MASS = 0.003

# ── Starts (robot yaw frame, from the foot-frame origin) ──
R_RANGE = (0.10, 0.35)
BEARING_MAX = math.radians(45.0)
START_SPEED = (0.0, 0.15)
LOOK_JITTER = 0.10
# Curriculum on the start distance: the far edge of the range grows from R_START_MAX on the docked EMA.
R_START_MAX, R_STEP, R_ADVANCE = 0.15, 0.025, 0.5

# ── Reward knobs ──
DOCK_STD = 0.01                                 # m: the docked Gaussian (the kneel-rise curriculum reaches +-4 cm)
STILL_STD = 0.03                                # m/s trunk planar speed
PROGRESS_WEIGHT = 50.0                          # per m of distance gained (10 cm -> 5)
DOCKED_WEIGHT = 2.0                             # per step at the spot and still
BALL_KICK_WEIGHT = -2.0                         # per m/s of ball speed above BALL_FREE_SPEED
BALL_FREE_SPEED = 0.02
IN_VIEW_WEIGHT = 0.2
ACTION_RATE_WEIGHT = -0.05
FALL_WEIGHT = -20.0
EPISODE_S = 8.0
DOCKED_TOL = 0.015                              # m, success for logging and the curriculum
DOCKED_STILL = 0.02                             # m/s


# ── Ball entity ─────────────────────────────────────────────────────────────────
def get_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="ball", pos=[0.2, 0.0, BALL_RADIUS])
    body.add_freejoint(name="ball_free")
    body.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS], mass=BALL_MASS,
                  rgba=[1.0, 0.45, 0.05, 1.0], condim=4, friction=[0.7, 0.02, 0.0001])
    return spec


BALL_CFG = EntityCfg(spec_fn=get_ball_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.2, 0.0, BALL_RADIUS)))


# ── External command: a buffer the commander writes, never resampled ─────────────
class ExternalCommand(CommandTerm):
    def __init__(self, cfg: "ExternalCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.value = torch.zeros(self.num_envs, cfg.dim, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.value

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        pass   # the dock's reset event writes the start values (the command reset may run after it)

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class ExternalCommandCfg(CommandTermCfg):
    dim: int = 3
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: ManagerBasedRlEnv) -> ExternalCommand:
        return ExternalCommand(self, env)


# ── Frozen walk61 ───────────────────────────────────────────────────────────────
class FrozenMlp:
    """The exported actor (normaliser baked in) as a batched torch function, as seeing_audit.OnnxMlp."""

    def __init__(self, path: Path, device):
        m = onnx.load(str(path))
        assert [n.op_type for n in m.graph.node] == ["Sub", "Div"] + ["Gemm", "Elu"] * 3 + ["Gemm"], path
        w = {i.name: torch.tensor(numpy_helper.to_array(i)).float().to(device) for i in m.graph.initializer}
        self.mean = w["obs_normalizer._mean"]
        self.div = next(v for k, v in w.items() if k.startswith("onnx::Div"))
        self.layers = [(w[f"mlp.{i}.weight"], w[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]

    @torch.no_grad()
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.mean) / self.div
        for j, (wt, b) in enumerate(self.layers):
            x = x @ wt.T + b
            if j < 3:
                x = F.elu(x)
        return x


class DockCommanderAction(ActionTerm):
    """6-D commander action -> walk61's twist and head commands -> walk61 -> joint position targets."""

    def __init__(self, cfg: "DockCommanderActionCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._joint = cfg.joint_cfg.build(env)
        self._walk = FrozenMlp(Path(cfg.walk_onnx), env.device)
        self._raw = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self.walk_action = torch.zeros(self.num_envs, self._joint.action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        cm = self._env.command_manager
        twist = cm.get_term("twist").value
        twist[:, 0], twist[:, 1], twist[:, 2] = a[:, 0] * VX_MAX, a[:, 1] * VY_MAX, a[:, 2] * WZ_MAX
        head = cm.get_term("head_pose").value
        lo, hi = HEAD_PITCH_RANGE
        head[:, 1] = lo + (a[:, 3] + 1.0) * 0.5 * (hi - lo)
        head[:, 2] = a[:, 4] * HEAD_YAW_MAX
        head[:, 0] = -NECK_RANGE[0] * a[:, 5].clamp(max=0.0)   # a >= 0: HOME (the init mean), a = -1: full bow
        # walk61 reads its observation of the previous step (the runtime's one-tick command latency); its slots
        # carry the commands of the previous step too.
        walk_obs = self._env.obs_buf.get("walk")
        if walk_obs is None:   # before the first reset
            walk_obs = torch.zeros(self.num_envs, 61, device=self.device)
        self.walk_action[:] = self._walk(walk_obs)
        self._joint.process_actions(self.walk_action)

    def apply_actions(self) -> None:
        self._joint.apply_actions()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw[env_ids] = 0.0
        self.walk_action[env_ids] = 0.0
        self._joint.reset(env_ids)


@dataclass(kw_only=True)
class DockCommanderActionCfg(ActionTermCfg):
    joint_cfg: JointPositionActionCfg
    walk_onnx: str = str(WALK_ONNX)

    def build(self, env: ManagerBasedRlEnv) -> DockCommanderAction:
        return DockCommanderAction(self, env)


def walk_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    """walk61's own previous output (its `actions` obs slot), not the commander's."""
    return env.action_manager.get_term("dock").walk_action


# ── State ───────────────────────────────────────────────────────────────────────
def _ds(env: ManagerBasedRlEnv) -> dict:
    st = getattr(env, "_ball_dock", None)
    if st is None:
        n, dev = env.num_envs, env.device
        st = env._ball_dock = {
            "prev_dist": torch.zeros(n, device=dev), "r_max": R_START_MAX, "success": 0.0, "r_it": -10 ** 9,
            "cam": torch.zeros(n, 5, device=dev), "cam_key": None,
            "buf": torch.zeros(n, pix.H, 3, device=dev), "buf_det": torch.zeros(n, pix.H, dtype=torch.bool, device=dev),
            "buf_frame": torch.zeros(n, pix.H, dtype=torch.bool, device=dev),
            "buf_step": torch.full((n, pix.H), -10 ** 6, dtype=torch.long, device=dev),
            "last": torch.zeros(n, 3, device=dev), "last_step": torch.zeros(n, dtype=torch.long, device=dev),
            "visible": torch.zeros(n, device=dev), "init_pending": torch.ones(n, dtype=torch.bool, device=dev),
        }
    return st


def _yaw(q: torch.Tensor) -> torch.Tensor:
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def foot_origin(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """The runtime's foot frame: the midpoint of the two foot sites on the floor, heading = trunk yaw."""
    robot: Entity = env.scene["robot"]
    ids = getattr(env, "_ball_dock_feet", None)
    if ids is None:
        ids, _ = robot.find_sites(["left_foot", "right_foot"], preserve_order=True)
        env._ball_dock_feet = ids
    p = robot.data.site_pos_w[:, ids, :2].mean(1)
    return p, _yaw(robot.data.root_link_quat_w)


def ball_in_foot_frame(env: ManagerBasedRlEnv) -> torch.Tensor:
    o, yaw = foot_origin(env)
    d = env.scene["ball"].data.root_link_pos_w[:, :2] - o
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)


def dock_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    return ball_in_foot_frame(env) - torch.tensor(OBJECT_NOMINAL_XY, device=env.device)


def trunk_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_link_lin_vel_w[:, :2].norm(dim=-1)


def docked_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
    return (dock_error(env).norm(dim=-1) < DOCKED_TOL) & (trunk_speed(env) < DOCKED_STILL)


# ── Events ──────────────────────────────────────────────────────────────────────
SPAWN_OVERRIDE: torch.Tensor | None = None


def reset_ball_dock(env: ManagerBasedRlEnv, env_ids: torch.Tensor, r_max: float | None = None,
                    start_speed: tuple = START_SPEED, look_jitter: float = LOOK_JITTER, wide_frac: float = 0.0,
                    wide_r: tuple = (0.04, 0.50), wide_bearing: float = math.pi / 2) -> None:
    """Ball at range r in [R_RANGE[0], r_max] (curriculum) and bearing within +-45 deg of the foot frame, robot
    head aimed at it (pitch from the range, as the pixel pick's hand-over), residual forward speed, this episode's
    camera draw. wide_frac of the envs instead draw from wide_r x +-wide_bearing: where a missed bite leaves the
    ball (a retry)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device)
    n, dev = len(env_ids), env.device
    st = _ds(env)
    r_hi = st["r_max"] if r_max is None else r_max
    robot: Entity = env.scene["robot"]
    ball: Entity = env.scene["ball"]
    root = env.sim.data.qpos[env_ids][:, robot.indexing.free_joint_q_adr]
    yaw = _yaw(root[:, 3:7])
    c, s = torch.cos(yaw), torch.sin(yaw)
    r = pix._uniform(n, R_RANGE[0], r_hi, dev)
    b = pix._uniform(n, -BEARING_MAX, BEARING_MAX, dev)
    if SPAWN_OVERRIDE is not None:   # eval hook (dock_check --spawns): (k, 2) range, bearing drawn with replacement
        pick = SPAWN_OVERRIDE[torch.randint(len(SPAWN_OVERRIDE), (n,), device=dev)]
        r, b = pick[:, 0], pick[:, 1]
    elif wide_frac > 0.0:
        w = torch.rand(n, device=dev) < wide_frac
        r = torch.where(w, pix._uniform(n, *wide_r, dev), r)
        b = torch.where(w, pix._uniform(n, -wide_bearing, wide_bearing, dev), b)
    pose = torch.zeros(n, 7, device=dev)
    pose[:, 0] = root[:, 0] + c * r * torch.cos(b) - s * r * torch.sin(b)
    pose[:, 1] = root[:, 1] + s * r * torch.cos(b) + c * r * torch.sin(b)
    pose[:, 2] = BALL_RADIUS
    pose[:, 3] = 1.0
    ball.write_root_link_pose_to_sim(pose, env_ids)
    ball.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids)

    ps = pix._px(env)
    hp = (1.1 + (0.245 - r) * (0.3 / 0.086)).clamp(*pix.HEAD_PITCH_LIMITS) + pix._uniform(n, -look_jitter,
                                                                                            look_jitter, dev)
    hy = (b + pix._uniform(n, -look_jitter, look_jitter, dev)).clamp(-pix.HEAD_YAW_LIMIT, pix.HEAD_YAW_LIMIT)
    jids = torch.tensor([ps["head_j"], ps["yaw_j"]], device=dev)
    pos = torch.stack([hp.clamp(*pix.HEAD_PITCH_LIMITS), hy], -1)
    robot.write_joint_state_to_sim(pos, torch.zeros_like(pos), jids, env_ids)
    # the commander's head command starts where the head is (no step at t = 0); walk starts at zero
    env.command_manager.get_term("twist").value[env_ids] = 0.0
    head = env.command_manager.get_term("head_pose").value
    head[env_ids] = 0.0
    default = robot.data.default_joint_pos[env_ids][:, [ps["head_j"], ps["yaw_j"]]]
    head[env_ids, 1] = pos[:, 0] - default[:, 0]
    head[env_ids, 2] = pos[:, 1] - default[:, 1]

    qv = env.sim.data.qvel[env_ids][:, robot.indexing.free_joint_v_adr]
    v = pix._uniform(n, *start_speed, dev)
    vel = qv.clone()
    vel[:, 0] += c * v
    vel[:, 1] += s * v
    robot.write_root_link_velocity_to_sim(vel, env_ids)

    # this episode's camera (the pixel pick's draw)
    ps["dpitch"][env_ids] = pix._uniform(n, -pix.CAM_DR["pitch_deg"], pix.CAM_DR["pitch_deg"], dev)
    ps["shift"][env_ids] = pix._uniform((n, 3), -pix.CAM_DR["shift_m"], pix.CAM_DR["shift_m"], dev)
    ps["gain"][env_ids] = pix._uniform(n, *pix.CAM_DR["head_gain"], dev)
    ps["pivot"][env_ids] = pix._uniform((n, 3), -pix.CAM_DR["pivot_m"], pix.CAM_DR["pivot_m"], dev)
    ps["bow"][env_ids] = pix._uniform(n, *pix.CAM_DR["bow_dz"], dev)
    ps["hide"][env_ids] = pix._uniform(n, *pix.BEAK_HIDE_M, dev)
    ps["bias"][env_ids] = pix._uniform((n, 2), -pix.PIX_BIAS_PX, pix.PIX_BIAS_PX, dev)
    ps["period"][env_ids] = torch.randint(pix.PIX_PERIOD_STEPS[0], pix.PIX_PERIOD_STEPS[1] + 1, (n,), device=dev)
    ps["delay"][env_ids] = torch.round(pix._uniform(n, *pix.PIX_DELAY_S, dev) / env.step_dt).long().clamp(1, pix.H - 1)
    ps["drop"][env_ids] = pix._uniform(n, *pix.PIX_DROP, dev)
    ps["cam_offset"][env_ids] = torch.randint(0, 6, (n,), device=dev)
    ps["stall_until"][env_ids] = -1
    st["buf_det"][env_ids] = False
    st["buf_frame"][env_ids] = False
    st["buf_step"][env_ids] = -10 ** 6
    st["init_pending"][env_ids] = True
    st["prev_dist"][env_ids] = -1.0      # set on the first reward step


def dock_range_curriculum(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict:
    """Widen the start range's far edge by R_STEP while the docked-at-episode-end EMA is above R_ADVANCE."""
    st = _ds(env)
    if env_ids is not None and len(env_ids) > 0:
        rate = float(st.get("docked_end", torch.zeros(env.num_envs, device=env.device))[env_ids].float().mean())
        st["success"] = 0.98 * st["success"] + 0.02 * rate
    it = env.common_step_counter // 24
    if st["success"] > R_ADVANCE and st["r_max"] < R_RANGE[1] - 1e-9 and it - st["r_it"] >= 50:
        st["r_max"] = min(st["r_max"] + R_STEP, R_RANGE[1])
        st["r_it"] = it
    return {"r_max": torch.tensor(st["r_max"]), "docked_ema": torch.tensor(st["success"])}


def track_docked(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None) -> None:
    """Every step: remember whether each env is docked (read by the curriculum at its episode end)."""
    _ds(env)["docked_end"] = docked_mask(env)


# ── Camera observation ──────────────────────────────────────────────────────────
def dock_camera(env: ManagerBasedRlEnv) -> torch.Tensor:
    """[u, v, width, age, visible] of the ball, through the pixel pick's U20CAM model (frames, delay, drops, stalls,
    beak occlusion, bias, noise). Cached per step."""
    st, ps = _ds(env), pix._px(env)
    key = env.common_step_counter
    if st["cam_key"] == key:
        return st["cam"]
    dev, n = env.device, env.num_envs
    t = env.episode_length_buf.long()
    robot: Entity = env.scene["robot"]
    p = env.scene["ball"].data.root_link_pos_w
    u, v, width, inside = pix.object_pixels(env, p, torch.full((n,), BALL_RADIUS, device=dev))
    tip = robot.data.site_pos_w[:, ps["tip_site"]]
    seen = inside & ((tip - p).norm(dim=-1) >= ps["hide"])
    meas = pix._norm(u + ps["bias"][:, 0] + torch.randn(n, device=dev) * pix.PIX_NOISE_PX,
                     v + ps["bias"][:, 1] + torch.randn(n, device=dev) * pix.PIX_NOISE_PX,
                     width * (1 + torch.randn(n, device=dev) * pix.PIX_WIDTH_NOISE).clamp(min=0.2))
    init = st["init_pending"]
    if init.any():
        edge = pix._norm(u.clamp(0, pix.IMG_W - 1), v.clamp(0, pix.IMG_H - 1), width)
        st["last"] = torch.where(init.unsqueeze(-1), torch.where(seen.unsqueeze(-1), meas, edge), st["last"])
        st["visible"] = torch.where(init, seen.float(), st["visible"])
        age = torch.where(seen, pix._uniform(n, *pix.PIX_INIT_AGE_S, dev), pix._uniform(n, *pix.PIX_INIT_AGE_HIDDEN_S, dev))
        st["last_step"] = torch.where(init, t - torch.round(age / env.step_dt).long(), st["last_step"])
        st["init_pending"] = torch.zeros_like(init)
    tick = ((t + ps["cam_offset"]) % ps["period"]) == 0
    from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp
    stall_start = tick & (torch.rand(n, device=dev) < sp.CAM_STALL_PROB) & (t >= ps["stall_until"])
    ps["stall_until"] = torch.where(
        stall_start, t + torch.round(pix._uniform(n, *sp.CAM_STALL_S, dev) / env.step_dt).long(), ps["stall_until"])
    frame = tick & (t >= ps["stall_until"])
    detected = frame & seen & (torch.rand(n, device=dev) >= ps["drop"])
    slot = t % pix.H
    ar = torch.arange(n, device=dev)
    st["buf"][ar, slot] = meas
    st["buf_det"][ar, slot] = detected
    st["buf_frame"][ar, slot] = frame
    st["buf_step"][ar, slot] = t
    src = (t - ps["delay"]) % pix.H
    fresh = st["buf_step"][ar, src] == t - ps["delay"]
    arrived = fresh & st["buf_frame"][ar, src]
    hit = arrived & st["buf_det"][ar, src]
    st["last"] = torch.where(hit.unsqueeze(-1), st["buf"][ar, src], st["last"])
    st["last_step"] = torch.where(hit, t - ps["delay"], st["last_step"])
    st["visible"] = torch.where(arrived, hit.float(), st["visible"])
    age = ((t - st["last_step"]).float() * env.step_dt).clamp(0.0, sp.AGE_MAX_S)
    out = torch.cat([st["last"], age.unsqueeze(-1), st["visible"].unsqueeze(-1)], -1)
    st["cam"], st["cam_key"] = out, key
    return out


def head_joints(env: ManagerBasedRlEnv, biased: bool = False) -> torch.Tensor:
    """Neck pitch, head pitch, head yaw; `biased` reads what the encoders report (servo_noise level >= 1)."""
    ps = pix._px(env)
    d = env.scene["robot"].data
    q = d.joint_pos_biased if biased else d.joint_pos
    return q[:, [ps["neck_j"], ps["head_j"], ps["yaw_j"]]]


def commander_last_action(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.action_manager.get_term("dock").raw_action.clamp(-1.0, 1.0)


def target_error(env: ManagerBasedRlEnv, target_dx: float = 0.0) -> torch.Tensor:
    """The ball's error to the dock's target: the landing spot moved target_dx along the heading (the ball that far
    beyond the landing spot). dock_error stays the kneel's frame (evals, chain replay)."""
    e = dock_error(env)
    return torch.stack([e[:, 0] - target_dx, e[:, 1]], -1)


def ball_state_obs(env: ManagerBasedRlEnv, target_dx: float = 0.0) -> torch.Tensor:
    """Critic: the ball in the foot frame, its error to the dock target, its speed, and whether it is in the
    picture."""
    half = torch.full((env.num_envs,), BALL_RADIUS, device=env.device)
    _, _, _, inside = pix.object_pixels(env, env.scene["ball"].data.root_link_pos_w, half)
    v = env.scene["ball"].data.root_link_lin_vel_w[:, :2]
    return torch.cat([ball_in_foot_frame(env), target_error(env, target_dx), v, inside.float().unsqueeze(-1)], -1)


# ── Rewards ─────────────────────────────────────────────────────────────────────
def dock_progress(env: ManagerBasedRlEnv, target_dx: float = 0.0) -> torch.Tensor:
    """Distance gained toward the dock target this step (m). Potential-based: holding pays 0, backing off costs."""
    st = _ds(env)
    d = target_error(env, target_dx).norm(dim=-1)
    prev = st["prev_dist"]
    out = torch.where(prev < 0, torch.zeros_like(d), prev - d)
    st["prev_dist"] = d.clone()
    return out


def docked(env: ManagerBasedRlEnv, std: float = DOCK_STD, still_std: float = STILL_STD,
           target_dx: float = 0.0) -> torch.Tensor:
    """In [0, 1]: at the dock target AND still (product of Gaussians)."""
    e = target_error(env, target_dx).norm(dim=-1)
    return torch.exp(-(e / std) ** 2) * torch.exp(-(trunk_speed(env) / still_std) ** 2)


def ball_kick_cost(env: ManagerBasedRlEnv, free: float = BALL_FREE_SPEED) -> torch.Tensor:
    """>= 0: ball speed above `free` (it was kicked or pushed). Negative weight."""
    return torch.clamp(env.scene["ball"].data.root_link_lin_vel_w[:, :2].norm(dim=-1) - free, min=0.0)


def ball_in_view(env: ManagerBasedRlEnv) -> torch.Tensor:
    half = torch.full((env.num_envs,), BALL_RADIUS, device=env.device)
    _, _, _, inside = pix.object_pixels(env, env.scene["ball"].data.root_link_pos_w, half)
    return inside.float()


def commander_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """>= 0: squared change of the clipped commander action. Negative weight."""
    a = env.action_manager.action.clamp(-1.0, 1.0)
    p = env.action_manager.prev_action.clamp(-1.0, 1.0)
    return (a - p).pow(2).sum(-1)


def ball_lost(env: ManagerBasedRlEnv, max_dist: float = 0.6) -> torch.Tensor:
    return ball_in_foot_frame(env).norm(dim=-1) > max_dist


# ── Env cfg ─────────────────────────────────────────────────────────────────────
def make_ball_dock_env_cfg(variant: str = CURRENT, play: bool = False, walk_onnx: Path = WALK_ONNX,
                           target_dx: float = 0.0, wide: bool = False):
    cfg = make_cmu_velocity_env_cfg(variant, play=play)
    cfg.scene.entities = {**cfg.scene.entities, "ball": BALL_CFG}
    cfg.episode_length_s = EPISODE_S

    # commands: buffers the commander writes (twist, head_pose), body_pose stays zero
    cfg.commands = {"twist": ExternalCommandCfg(dim=3), "head_pose": ExternalCommandCfg(dim=4),
                    "body_pose": ExternalCommandCfg(dim=6)}

    # actions: the commander over walk61's joint action
    jp = cfg.actions["joint_pos"]
    assert isinstance(jp, JointPositionActionCfg)
    cfg.actions = {"dock": DockCommanderActionCfg(entity_name="robot", joint_cfg=copy.deepcopy(jp),
                                                  walk_onnx=str(walk_onnx))}

    # observations: walk61's 61-D group as trained (its own last action), the commander's, the critic's
    walk = copy.deepcopy(cfg.observations["actor"])
    walk.terms["actions"] = ObservationTermCfg(func=walk_last_action)
    for name in ("command", "head_command", "body_command"):
        assert name in walk.terms, name
    critic = copy.deepcopy(cfg.observations["critic"])
    gname = "projected_gravity" if "projected_gravity" in walk.terms else "raw_accelerometer"
    actor_terms = {
        "camera": ObservationTermCfg(func=dock_camera),
        "head_joints": ObservationTermCfg(func=head_joints),
        "gravity": copy.deepcopy(walk.terms[gname]),
        "gyro": copy.deepcopy(walk.terms["base_ang_vel"]),
        "last_action": ObservationTermCfg(func=commander_last_action),
    }
    actor = ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=not play)
    critic.terms["actions"] = ObservationTermCfg(func=walk_last_action)
    critic.terms["ball_state"] = ObservationTermCfg(func=ball_state_obs)
    critic.terms["commander"] = ObservationTermCfg(func=commander_last_action)
    critic.terms["camera"] = ObservationTermCfg(func=dock_camera)
    cfg.observations = {"actor": actor, "critic": critic, "walk": walk}

    # rewards: the dock's own
    cfg.rewards = {
        "dock_progress": RewardTermCfg(func=dock_progress, weight=PROGRESS_WEIGHT),
        "docked": RewardTermCfg(func=docked, weight=DOCKED_WEIGHT),
        "ball_kick": RewardTermCfg(func=ball_kick_cost, weight=BALL_KICK_WEIGHT),
        "ball_in_view": RewardTermCfg(func=ball_in_view, weight=IN_VIEW_WEIGHT),
        "action_rate": RewardTermCfg(func=commander_action_rate, weight=ACTION_RATE_WEIGHT),
        "fell": RewardTermCfg(func=mdp.is_terminated, weight=FALL_WEIGHT),
    }
    # terminations: velocity's (fall, NaN, time-out) + the ball lost
    cfg.terminations["ball_lost"] = TerminationTermCfg(func=ball_lost, time_out=False)
    cfg.terminations.pop("out_of_terrain_bounds", None)

    # events: velocity's DR; the ball and camera reset after the robot's
    cfg.events.pop("push_robot", None)
    cfg.events["reset_ball_dock"] = EventTermCfg(func=reset_ball_dock, mode="reset",
                                                 params={"r_max": R_RANGE[1] if play else None})
    cfg.events["track_docked"] = EventTermCfg(func=track_docked, mode="interval", interval_range_s=(0.0, 0.0))

    # curricula: velocity's DR ramps stay; command / reward-weight ones go (their terms are gone)
    keep = {k: v for k, v in cfg.curriculum.items() if k in ("com_range", "head_com_range")}
    cfg.curriculum = keep
    if not play:
        cfg.curriculum["dock_range"] = CurriculumTermCfg(func=dock_range_curriculum)

    # a moved target (the fine-tune of a docked policy: full range from the start, no range curriculum)
    if target_dx:
        for name in ("dock_progress", "docked"):
            cfg.rewards[name].params["target_dx"] = target_dx
        cfg.observations["critic"].terms["ball_state"].params = {"target_dx": target_dx}
        cfg.curriculum.pop("dock_range", None)
        cfg.events["reset_ball_dock"].params["r_max"] = R_RANGE[1]
    if wide:
        cfg.events["reset_ball_dock"].params.update(wide_frac=WIDE_FRAC, wide_r=WIDE_R, wide_bearing=math.pi / 2)
        cfg.terminations["ball_lost"].params = {"max_dist": 0.75}   # the wide draw reaches 0.5 m
    return cfg


BallDockRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.6, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
        num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0, symmetry_cfg=None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="ball_dock",
    run_name="ball_dock",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=1500,
)
