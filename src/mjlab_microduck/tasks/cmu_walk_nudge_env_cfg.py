"""Nudge: walk61 fine-tuned to step to a target 1-6 cm away and stop there (cmu/kneel/RESULTS.md next step 2).

walk61 does not step below ~0.2 m/s, and a brief pulse above it is not a usable substitute (`cmu/kneel/pulse_check.py`:
lateral pulses never step, forward pulses give 0 or 2-5 cm, and 10-20 % of envs creep backward several cm at zero
command). The dock's error tail (stopping 2-3 cm short) and the blind kneel's +-1 cm range window both need steps of
that size.

The 61-D layout is kept, and so is the meaning of "zero twist = stand". The twist slots carry the REMAINING position
error to the target, in the trunk's heading frame, times NUDGE_GAIN (clipped to walk61's command ranges). At
iteration 0 the policy is therefore walk61 under a P-controller (which steps once the error is above ~3 cm), and
RL has to learn the small steps and the stop. On the robot the dock writes NUDGE_GAIN x (ball error from the camera,
foot frame) into the slots, so the target is wherever the ball says the landing spot is.

Built on the CMU velocity recipe (DR, obs noise, delays, BAM, head/body commands), with these changes:
- the twist command is a NudgeCommand (target offset re-drawn every NUDGE_RESAMPLE_S, a ZERO_PROB share zero)
- velocity tracking is replaced by position (Gaussian, NUDGE_STD) + potential-based progress + yaw terms
- the stepping terms only switch on when the target is more than STEP_TOL away (|cmd| > NUDGE_GAIN x STEP_TOL)
- foot slip is penalised whenever the target is not reached (sliding a foot is not a repeatable correction)
- the inherited curricula are collapsed to their final stage (warm start from walk61, AGENTS.md)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from mjlab_microduck.tasks.cmu_asbuilt import CURRENT, make_cmu_velocity_env_cfg
from mjlab_microduck.tasks.cmu_ball_dock_env_cfg import WALK_ONNX, foot_origin
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final

NUDGE_GAIN = 8.0                     # twist slot = gain x remaining error (m -> m/s); 3 cm -> 0.24, above the dead zone
YAW_GAIN = 2.0                       # rad -> rad/s
CMD_MAX = (0.4, 0.3, 1.0)            # walk61's command ranges
TARGET_X = (-0.06, 0.06)             # m, target offset drawn in the trunk heading frame
TARGET_Y = (-0.04, 0.04)
TARGET_YAW = (-0.25, 0.25)           # rad
ZERO_PROB = 0.2                      # share of zero targets: hold still where you are
NUDGE_RESAMPLE_S = (2.5, 4.0)

NUDGE_STD = 0.015                    # m, the error still worth correcting (the kneel window is +-1 cm in range)
NUDGE_WEIGHT = 3.0
STILL_STD = 0.03                     # m/s of trunk speed, the position term's stillness factor
PROGRESS_WEIGHT = 20.0               # per m of error closed (potential-based: holding pays zero)
YAW_STD = 0.1
YAW_WEIGHT = 1.0
STEP_TOL = 0.015                     # m: closer than this, stepping is not rewarded
STAND_TOL = 0.008                    # m: closer than this, the pose term uses its tight standing std

INIT_ONNX = WALK_ONNX


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


class NudgeCommand(CommandTerm):
    """Target position (world xy) and heading; the command is the remaining error in the trunk heading frame x gain."""

    cfg: "NudgeCommandCfg"

    def __init__(self, cfg: "NudgeCommandCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.target_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.err = torch.zeros(self.num_envs, 3, device=self.device)   # body-frame dx, dy, dyaw
        # a draw is an offset from where the robot stands; it is anchored to the world on the next update, because
        # the command reset can run before the reset events have placed the robot
        self.pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["err_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["within_1cm"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        g = torch.tensor((self.cfg.gain, self.cfg.gain, self.cfg.yaw_gain), device=self.device)
        lim = torch.tensor(self.cfg.cmd_max, device=self.device)
        return torch.maximum(torch.minimum(self.err * g, lim), -lim)

    def _refresh(self) -> None:
        p, yaw = foot_origin(self._env)
        if self.pending.any():
            ids = self.pending.nonzero(as_tuple=True)[0]
            c, s = torch.cos(yaw[ids]), torch.sin(yaw[ids])
            dx, dy, dyaw = self.offset[ids].unbind(-1)
            self.target_xy[ids] = p[ids] + torch.stack([c * dx - s * dy, s * dx + c * dy], -1)
            self.target_yaw[ids] = yaw[ids] + dyaw
            self.pending[ids] = False
        d = self.target_xy - p
        c, s = torch.cos(yaw), torch.sin(yaw)
        self.err = torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1],
                                _wrap(self.target_yaw - yaw)], -1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n, dev = len(env_ids), self.device

        def u(lo_hi):
            return torch.empty(n, device=dev).uniform_(*lo_hi)

        off = torch.stack([u(self.cfg.target_x), u(self.cfg.target_y), u(self.cfg.target_yaw)], -1)
        off[torch.rand(n, device=dev) < self.cfg.zero_prob] = 0.0
        self.offset[env_ids] = off
        self.err[env_ids] = off                 # until anchored, the error is the drawn offset
        self.pending[env_ids] = True

    def _update_command(self) -> None:
        self._refresh()

    def _update_metrics(self) -> None:
        e = self.err[:, :2].norm(dim=-1)
        self.metrics["err_xy"] = e
        self.metrics["within_1cm"] = (e < 0.01).float()


@dataclass(kw_only=True)
class NudgeCommandCfg(CommandTermCfg):
    gain: float = NUDGE_GAIN
    yaw_gain: float = YAW_GAIN
    cmd_max: tuple[float, float, float] = CMD_MAX
    target_x: tuple[float, float] = TARGET_X
    target_y: tuple[float, float] = TARGET_Y
    target_yaw: tuple[float, float] = TARGET_YAW
    zero_prob: float = ZERO_PROB
    resampling_time_range: tuple[float, float] = NUDGE_RESAMPLE_S

    def build(self, env: ManagerBasedRlEnv) -> NudgeCommand:
        return NudgeCommand(self, env)


# ── Rewards ─────────────────────────────────────────────────────────────────────
def _nudge(env: ManagerBasedRlEnv, command_name: str) -> NudgeCommand:
    return env.command_manager.get_term(command_name)


def nudge_position(env: ManagerBasedRlEnv, command_name: str = "twist", std: float = NUDGE_STD,
                   still_std: float = STILL_STD) -> torch.Tensor:
    """exp(-|e_xy|^2 / std^2) x exp(-|v_trunk_xy|^2 / still_std^2): standing STILL at the target pays 1 per second.
    Without the stillness factor wn-a learned to march in place at the target (iteration 1000: within 1 cm 1.00,
    trunk under 2 cm/s 0.05; iteration 250: 0.95)."""
    e = _nudge(env, command_name).err[:, :2]
    v = env.scene["robot"].data.root_link_lin_vel_w[:, :2]
    return torch.exp(-e.square().sum(-1) / std ** 2) * torch.exp(-v.square().sum(-1) / still_std ** 2)


def nudge_yaw(env: ManagerBasedRlEnv, command_name: str = "twist", std: float = YAW_STD) -> torch.Tensor:
    e = _nudge(env, command_name).err[:, 2]
    return torch.exp(-e.square() / std ** 2)


def nudge_progress(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """Potential-based: the rate of error closed (m/s). Moving away costs as much as approaching pays; holding pays
    zero. Zeroed on the step a new target is drawn (the jump is not the policy's doing)."""
    cmd = _nudge(env, command_name)
    e = cmd.err[:, :2].norm(dim=-1)
    prev = getattr(env, "_nudge_prev_err", None)
    prev_target = getattr(env, "_nudge_prev_target", None)
    if prev is None:
        prev, prev_target = e.clone(), cmd.target_xy.clone()
    fresh = (env.episode_length_buf <= 1) | (cmd.target_xy - prev_target).abs().sum(-1).gt(1e-6)
    r = torch.where(fresh, torch.zeros_like(e), (prev - e) / env.step_dt)
    env._nudge_prev_err, env._nudge_prev_target = e.clone(), cmd.target_xy.clone()
    return r


def make_walk_nudge_env_cfg(variant: str = CURRENT, play: bool = False):
    cfg = make_cmu_velocity_env_cfg(variant, play=play)
    _collapse_curricula_to_final(cfg)
    cfg.curriculum.pop("standing_envs", None)        # it edits the velocity command's standing share
    cfg.commands["twist"] = NudgeCommandCfg()

    r = cfg.rewards
    r.pop("track_linear_velocity")
    r.pop("track_angular_velocity")
    r["nudge_position"] = RewardTermCfg(func=nudge_position, weight=NUDGE_WEIGHT, params={"command_name": "twist"})
    r["nudge_progress"] = RewardTermCfg(func=nudge_progress, weight=PROGRESS_WEIGHT, params={"command_name": "twist"})
    r["nudge_yaw"] = RewardTermCfg(func=nudge_yaw, weight=YAW_WEIGHT, params={"command_name": "twist"})
    step_thr = NUDGE_GAIN * STEP_TOL
    for k in ("air_time", "foot_clearance", "foot_swing_height"):
        r[k].params["command_threshold"] = step_thr
    r["foot_slip"].params["command_threshold"] = 0.0
    r["pose"].params["walking_threshold"] = NUDGE_GAIN * STAND_TOL
    return cfg


def walk_nudge_rl_cfg(base):
    rl = copy.deepcopy(base)
    rl.experiment_name = "walk_nudge"
    rl.run_name = "walk_nudge"
    rl.max_iterations = 2000
    return rl
