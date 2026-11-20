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
SOURCE_STEP = 24000               # rise-s1's common_step_counter at model_999 (its checkpoint's env_state)


def pin_curricula_at(cfg, step: int) -> None:
    """Warm start: hold every inherited stage curriculum at the stage in force at the SOURCE run's `step` (AGENTS.md).
    The sighted kneel fine-tunes (ks-a/b/c) restarted them at step 0, so the first ~250 iterations ran action_rate at
    -0.8 against rise-s1's -2.0 (and CoM DR at 3 mm against 10 mm)."""
    for term in cfg.curriculum.values():
        for key, val in list(term.params.items()):
            if isinstance(val, list) and val and all(isinstance(v, dict) and "step" in v for v in val):
                live = [v for v in val if v["step"] <= step] or val[:1]
                term.params[key] = [{**live[-1], "step": 0}]
                if key == "weight_stages" and term.params.get("reward_name") in cfg.rewards:
                    cfg.rewards[term.params["reward_name"]].weight = live[-1]["weight"]


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

# dockfit: the sighted kneel on the dock's actual end states. The window reset never puts the ball closer than
# R_MIN = 8 cm (dx >= -0.7 cm on axis), but bd-b ends with dx p10 -1.9 cm, and ks-f@750 holds only 0.33 / 0.03 of
# the dx [-2, -1) / [-3, -2) cm bins (round 6). Half the starts come from bd-b@2000's end errors (robust fit:
# dx p10/p50/p90 -1.9 / -0.2 / +1.4 cm, |dy| p90 1.1 cm; stds widened ~25 %), the rest from the full window with
# the near edge opened. The window starts at its max (ks-f already holds it), so no window curriculum.
DOCKFIT_FRAC = 0.5
DOCKFIT_MU = (-0.002, 0.0)
DOCKFIT_STD = (0.016, 0.010)
DOCKFIT_CLIP = (-0.04, 0.04, 0.03)
DOCKFIT_R_MIN = 0.05              # dx >= -3.7 cm on axis
# The fetch target is the ball, but the seeing pick's reset also draws boxes and cylinders (0.5 / 0.25 / 0.25).
# From bd-b's ends ks-f holds the ball 0.67 (misses: never bitten) and the box 0.42 (drops): the mixed number hid
# both. ball_only=True trains the ball alone (its per-axis radius DR, 1.06-1.44 cm lumpy ellipsoid, stays).
BALL_ONLY_PROBS = (1.0, 0.0, 0.0)
# ... and on bd-c's ends (the dock with its target 0.75 cm forward): dx p10 / p50 / p90 -1.2 / +0.6 / +2.3 cm,
# |dy| p90 1.1 cm (stds widened ~25 %); a larger dock-shaped share, since this kneel serves that dock.
DOCKFIT_C = dict(mu=(0.006, 0.0), std=(0.017, 0.010), frac=0.7)
# A/B (round 7): the static grasp pose pull (one fixed 14-joint target, weight 6, std 0.25 rad, phase 0.22-0.36) is
# the same whatever the ball's offset; ks-f / ks-h / ks-i trade the near side against the far side. LOW_POSE runs it
# at 2 against a control continued at 6.
LOW_POSE_WEIGHT = 2.0


def make_kneel_rise_env_cfg(variant: str, play: bool = False, sighted: bool = False, dockfit: bool = False,
                            ball_only: bool = False, dock_fit: dict | None = None, low_pose: bool = False):
    """sighted=True keeps the camera slots during the descent (RESULTS.md next step 1): the blind kneel's success
    collapses for dx >= +2 cm, and a policy that sees the ball can correct for it. dockfit=True (implies sighted)
    trains on the dock's end states, near side included."""
    sighted = sighted or dockfit
    cfg = make_seeing_pix_handoff_env_cfg(variant, play=play, rise=True)

    if sighted:
        pin_curricula_at(cfg, SOURCE_STEP)

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
    if dockfit:
        fit = dock_fit or dict(mu=DOCKFIT_MU, std=DOCKFIT_STD, frac=DOCKFIT_FRAC)
        ev.params.update(window=wmax, r_min=DOCKFIT_R_MIN, dock_frac=fit["frac"], dock_mu=fit["mu"],
                         dock_std=fit["std"], dock_clip=DOCKFIT_CLIP)
    elif not play:
        cfg.curriculum["dock_window"] = CurriculumTermCfg(
            func=pix.pix_window_curriculum,
            params={"event_name": "reset_object", "window_step": DOCK_WINDOW_STEP, "window_max": wmax,
                    "advance_success": DOCK_ADVANCE})

    if ball_only:
        cfg.events["object_size"].params["probs"] = BALL_ONLY_PROBS
    if low_pose:
        cfg.rewards["static_pose"].weight = LOW_POSE_WEIGHT

    # 4. neck rate while holding
    cfg.rewards["held_neck_rate"] = RewardTermCfg(func=held_neck_rate_cost, weight=NECK_RATE_WEIGHT,
                                                  params={"free": NECK_RATE_FREE})
    return cfg


def kneel_rise_rl_cfg(base, sighted: bool = False, dockfit: bool = False, ball_only: bool = False):
    rl = copy.deepcopy(base)
    rl.experiment_name = ("kneel_rise_ball" if ball_only else "kneel_rise_dockfit" if dockfit else
                          "kneel_rise_sighted" if sighted else "kneel_rise")
    rl.run_name = rl.experiment_name
    rl.max_iterations = 1000
    return rl
