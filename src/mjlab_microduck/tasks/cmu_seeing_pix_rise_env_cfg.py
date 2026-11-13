"""Pixel seeing pick through a loaded rise into stand admission (task D, cmu/planning/fetch-20261103).

Why: on the robot (pick-handoff-real.jsonl) the pixel pick spx-b_1800 still holds the neck at -1.14 rad at the
runtime's hand-over (phase 0.90, walk_wired.SP_DONE_PHASE), 1.49 rad from stand61's nominal, and stand61 falls 0.9 s
later. transition_eval.py reproduces that fall only from the bowed exit (stand61 from its own nominal pose survives
the same 55 ms latency). The base task pays grasp_held (15 per step) for a ball lifted 1 cm in ANY posture, against
6 + 6 + 4 for the return terms, so a bowed hold earns the full main term and rising only risks it.

The one change (NEXT-EXPERIMENTS.md, task D: compare the current task with the extended terminal objective before
adding history or reward terms): from the end of the trained rise (TIMELINE return_end, phase 0.66) to the end of
the episode, grasp_held pays held x stand_admission, a product of Gaussians (AGENTS: multiplicative at goal states)
on stand61's admission state: joints at its nominal pose, trunk upright, trunk still. Before 0.66 it is unchanged.
Sensing, jaw timeline, object/start distribution, solver, PPO settings and every other term are the base task's.

Both the base and the rise task log the same hand-over numbers at phase 0.90 (track_handoff + handoff_metrics,
logging only): held, admitted (held and inside the admission thresholds), joint rms error to nominal, tilt.
"""
from __future__ import annotations

import math

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg, EventTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cmu_seeing_pix_env_cfg import make_seeing_pix_env_cfg
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import TIMELINE

HANDOFF_PHASE = 0.90                  # walk_wired.SP_DONE_PHASE: the runtime switches to stand61 here
ADMIT_FROM = TIMELINE["return_end"]   # 0.66: the trained rise is over, standing still with the ball from here
# Admission Gaussians, wide enough that today's bowed exit scores visibly (rms 0.40 rad -> 0.53) and the nominal
# stand scores 1: joint rms to stand61's nominal, trunk tilt (as return_upright, std 0.4 on 2(qx^2+qy^2)), trunk rate.
ADMIT_STD_Q = 0.5                     # rad, rms over the 14 servos
ADMIT_STD_TILT = 0.4
ADMIT_STD_W = 1.0                     # rad/s
# Logged "admitted": held and inside these (the transition_eval control stands from rms 0, tilt 0, rate 0).
ADMIT_MAX_RMS = 0.15                  # rad
ADMIT_MAX_TILT_DEG = 10.0
ADMIT_MAX_W = 0.5                     # rad/s


def _admission_state(env: ManagerBasedRlEnv) -> dict:
    """Joint rms to nominal, tilt and trunk rate, and their Gaussian product; cached per step."""
    c = microduck_mdp._gg_cache(env)
    key = env.common_step_counter
    if c.get("admission_key") == key:
        return c["admission"]
    robot: Entity = env.scene["robot"]
    err = microduck_mdp._servo_joint_pos(env, robot) - microduck_mdp._servo_default_joint_pos(env, robot)
    rms = err.pow(2).mean(-1).sqrt()
    q = robot.data.root_link_quat_w
    tilt_sq = 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    w = robot.data.root_link_ang_vel_b.norm(dim=-1)
    score = (torch.exp(-(rms / ADMIT_STD_Q) ** 2) * torch.exp(-tilt_sq / ADMIT_STD_TILT ** 2)
             * torch.exp(-(w / ADMIT_STD_W) ** 2))
    tilt_deg = torch.rad2deg(torch.acos(torch.clamp(1.0 - tilt_sq, -1.0, 1.0)))
    st = dict(rms=rms, tilt_deg=tilt_deg, w=w, score=score)
    c["admission_key"], c["admission"] = key, st
    return st


def stand_admission(env: ManagerBasedRlEnv) -> torch.Tensor:
    """In [0, 1]: how close the robot is to the state stand61 is trained to hold (1 at its nominal, still, upright)."""
    return _admission_state(env)["score"]


def grasp_held_admitted(env: ManagerBasedRlEnv, command_name: str = "twist", phase_start: float = 0.44,
                        admit_from: float = ADMIT_FROM, asset_name: str = "object") -> torch.Tensor:
    """grasp_held before admit_from; held x stand_admission from admit_from on. A bowed hold after the rise pays
    what its posture is worth to stand61, not the full main term. >= 0: positive weight."""
    held = microduck_mdp.grasp_held(env, command_name, phase_start, asset_name)
    phase = microduck_mdp._gp_phase(env, command_name)
    return torch.where(phase >= admit_from, held * stand_admission(env), held)


def track_handoff(env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, command_name: str = "twist") -> None:
    """Every step: snapshot the hand-over state of the envs whose phase crosses HANDOFF_PHASE this step."""
    c = microduck_mdp._gg_cache(env)
    phase = microduck_mdp._gp_phase(env, command_name)
    prev = c.get("handoff_prev_phase")
    if prev is None or prev.shape != phase.shape:
        prev = phase.clone()
    crossed = (prev < HANDOFF_PHASE) & (phase >= HANDOFF_PHASE)
    c["handoff_prev_phase"] = phase.clone()
    snap = microduck_mdp._gg_buffer(env, "handoff_snapshot", 5)  # valid, held, admitted, rms, tilt_deg
    if not crossed.any():
        return
    held = microduck_mdp.grasp_state(env, "object")["held"]
    a = _admission_state(env)
    admitted = held & (a["rms"] < ADMIT_MAX_RMS) & (a["tilt_deg"] < ADMIT_MAX_TILT_DEG) & (a["w"] < ADMIT_MAX_W)
    now = torch.stack([torch.ones_like(a["rms"]), held.float(), admitted.float(), a["rms"], a["tilt_deg"]], -1)
    snap[crossed] = now[crossed]


def handoff_metrics(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> dict:
    """Logging only: the hand-over snapshot of the episodes ending now, then cleared for the next episode."""
    snap = microduck_mdp._gg_buffer(env, "handoff_snapshot", 5)
    out = {}
    if env_ids is not None and len(env_ids) > 0:
        s = snap[env_ids]
        valid = s[:, 0] > 0
        if valid.any():
            v = s[valid]
            out = {"held": v[:, 1].mean(), "admitted": v[:, 2].mean(), "joint_rms_rad": v[:, 3].mean(),
                   "tilt_deg": v[:, 4].mean(), "episodes": torch.tensor(float(valid.sum()))}
        snap[env_ids] = 0.0
    return out


def make_seeing_pix_handoff_env_cfg(variant: str, play: bool = False, rise: bool = False):
    """The pixel seeing pick with hand-over logging; rise=True also swaps in the admitted main term."""
    cfg = make_seeing_pix_env_cfg(variant, play=play)
    cfg.events["track_handoff"] = EventTermCfg(func=track_handoff, mode="interval", interval_range_s=(0.0, 0.0))
    cfg.curriculum["handoff_metrics"] = CurriculumTermCfg(func=handoff_metrics)
    if rise:
        held = cfg.rewards["grasp_held"]
        held.func = grasp_held_admitted
        held.params = {**held.params, "admit_from": ADMIT_FROM}
    return cfg


def seeing_pix_rise_rl_cfg(base):
    from mjlab_microduck.tasks.cmu_seeing_pix_env_cfg import seeing_pix_rl_cfg
    rl = seeing_pix_rl_cfg(base)
    rl.experiment_name = "seeing_pix_rise"
    rl.run_name = "seeing_pix_rise"
    return rl


assert math.isclose(ADMIT_FROM, 0.66)  # TIMELINE drift would silently move the gate
