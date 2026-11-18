"""Kneel and rise with the ball, blind, from a docked stance (2026-11-18 scope split, `dock-kneel-draft.md`).

The fetch pick is split in two. A closed-loop dock (`cmu_ball_dock_env_cfg.py`) puts the robot standing still with the
ball at the static pose's landing spot (OBJECT_NOMINAL_XY, 8.7 cm ahead of the feet). This task is the open-loop part
that follows: kneel, bite, rise and keep the ball, with no camera input.

`Mjlab-KneelRise-FromRise-MicroDuck-CMU-<variant>` is the pixel seeing-pick rise task
(`cmu_seeing_pix_rise_env_cfg.py`, rise=True) with:

1. Blind actor: body slots 55-60 keep the 61-D layout, but only the holding bit (the delayed pinch, the jaw servo's
   current check on the robot) is non-zero. u, v, width, age and visible read 0. The critic still sees the camera
   slots and the true object state.
2. Docked starts: the ball inside DOCK_WINDOW of the landing spot (curriculum from DOCK_WINDOW_START on the held
   EMA), the robot almost still (START_SPEED), no mid-descent kicks (a blind policy cannot react to them). The head
   is aimed at the ball as the dock leaves it.
3. Training past the hand-over: the phase stops at HOLD_PHASE (0.90, where the runtime used to switch to stand61)
   and the episode runs HOLD_S longer. The rise objective (held x stand admission, from phase 0.66) therefore pays
   the hold-on that the issue-5 eval found keeps 96-99 % of the balls, and it is now inside the training data.
4. Neck-rate cost while holding: issue 5 found most stand61 drops thrown out of the beak at 5-8 rad/s of neck swing.
   NECK_RATE_FREE rad/s is free; each rad/s above it costs, but only while the ball is held, so the descent is not taxed.

Warm start: rise-s1 @ 999 (seeing_pix_rise, 2026-11-06, the best held-exit rms of issue 5), with the first-layer
columns of the blinded slots zeroed: at iteration 0 the policy moves as rise-s1 does for a ball where the hand-over
picture puts it.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import CurriculumTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as pix
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cmu_seeing_pix_rise_env_cfg import make_seeing_pix_handoff_env_cfg
from mjlab_microduck.tasks.microduck_ground_grasp_env_cfg import GG_PERIOD

HOLD_PHASE = 0.90                 # walk_wired.SP_DONE_PHASE; the phase freezes here (jaw shut)
HOLD_S = 2.5                      # seconds of hold-on after it
EPISODE_S = HOLD_PHASE * GG_PERIOD + HOLD_S   # 6.1 s

BLIND_KEEP = (4,)                 # body slots the actor still sees: holding
BLIND_ZERO = (0, 1, 2, 3, 5)      # u, v, width, age, visible

DOCK_WINDOW_START = 0.01          # m around the landing spot (range and bearing x r0), widened on the held EMA
DOCK_WINDOW_STEP = 0.005
DOCK_WINDOW_MAX = 0.04            # the docking target the two halves have to agree on; measured by the eval sweep
DOCK_ADVANCE = 0.3                # held-at-episode-end EMA that widens the window (the pixel pick's 0.5 was never
                                  # reached by a blind policy in kr-a)
START_SPEED = (0.0, 0.03)         # m/s: the dock ends standing, not walking
DOCK_CENTER_FRAC = 0.1

NECK_RATE_FREE = 2.0              # rad/s, per joint, free
NECK_RATE_WEIGHT = -0.5           # per rad/s above it, summed over the neck / head joints, while held

INIT_ONNX = Path(__file__).parent / "init_policies" / "rise-s1_999.onnx"


def blind_body_command(env: ManagerBasedRlEnv, dim: int = 6) -> torch.Tensor:
    """Actor body slots: the pixel pick's six with everything but the holding bit zeroed (camera state still advances,
    so the critic and the holding history keep their timing)."""
    out = pix.pix_object_command(env, dim).clone()
    out[:, list(BLIND_ZERO)] = 0.0
    return out


def held_neck_rate_cost(env: ManagerBasedRlEnv,
                        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=(r"neck_pitch", r"head_.*")),
                        free: float = NECK_RATE_FREE) -> torch.Tensor:
    """>= 0: sum over the neck / head joints of the joint speed above `free`, only while the ball is held. Negative
    weight."""
    robot: Entity = env.scene[asset_cfg.name]
    qd = robot.data.joint_vel[:, asset_cfg.joint_ids].abs()
    held = microduck_mdp.grasp_state(env, "object")["held"].float()
    return held * torch.clamp(qd - free, min=0.0).sum(-1)


SIGHTED_WINDOW_START = 0.01       # as kr-b: ks-a (+-2 cm) and ks-b (+-4 cm) starts unlearned the bite (grasp_held fell
                                  # while the bite's impact / overrun taxes shrank); start where most bites succeed
SIGHTED_WINDOW_MAX = 0.05         # covers the dock's measured end spread (bd-a2 dx p10/p90 -1.8 / +2.6 cm)


def make_kneel_rise_env_cfg(variant: str, play: bool = False, sighted: bool = False):
    """sighted=True keeps the camera slots during the descent (RESULTS.md next step 1): the blind kneel's success
    collapses for dx >= +2 cm, and a policy that sees the ball can correct for it."""
    cfg = make_seeing_pix_handoff_env_cfg(variant, play=play, rise=True)

    # 3. hold-on past the hand-over
    cfg.commands["twist"].hold_phase = HOLD_PHASE
    cfg.episode_length_s = EPISODE_S

    # 1. blind actor
    if not sighted:
        cfg.observations["actor"].terms["body_command"] = ObservationTermCfg(func=blind_body_command,
                                                                              params={"dim": 6})

    # 2. docked starts
    w0, wmax = (SIGHTED_WINDOW_START, SIGHTED_WINDOW_MAX) if sighted else (DOCK_WINDOW_START, DOCK_WINDOW_MAX)
    ev = cfg.events["reset_object"]
    ev.params.update(window=wmax if play else w0, start_speed=START_SPEED,
                     kick_prob=0.0, center_frac=DOCK_CENTER_FRAC)
    cfg.curriculum.pop("pix_window", None)
    if not play:
        cfg.curriculum["dock_window"] = CurriculumTermCfg(
            func=pix.pix_window_curriculum,
            params={"event_name": "reset_object", "window_step": DOCK_WINDOW_STEP, "window_max": wmax,
                    "advance_success": DOCK_ADVANCE})

    # 4. neck rate while holding
    cfg.rewards["held_neck_rate"] = RewardTermCfg(func=held_neck_rate_cost, weight=NECK_RATE_WEIGHT,
                                                  params={"free": NECK_RATE_FREE})
    return cfg


def kneel_rise_rl_cfg(base, sighted: bool = False):
    rl = copy.deepcopy(base)
    rl.experiment_name = "kneel_rise_sighted" if sighted else "kneel_rise"
    rl.run_name = rl.experiment_name
    rl.max_iterations = 1000
    return rl
