"""Pollen's two skill tasks (ground pick, sit-stand) on a CMU body variant, default the robot as it is now.

`make_cmu_ground_pick_env_cfg(variant)` and `make_cmu_sitstand_env_cfg(variant)` are Pollen's
`make_microduck_ground_pick_env_cfg` / `make_microduck_sitstand_env_cfg` with the robot's `spec_fn` wrapped by
`cmu_asbuilt.cmu_robot_spec_fn` (the variant's signed mass parts, the ankle heel stop and the current visual), the same
mechanism `make_cmu_velocity_env_cfg` and `make_cmu_aim_pick_env_cfg` use. With the defaults (recipe "pollen",
conditions "official") rewards, commands, observations, events, curricula and terminations are Pollen's.

conditions="robot": the robot's measured sensing and actuation conditions, the same dict scratch_worker.py applies with
`--conditions robot` (`robot_route_conditions()`: ROBOT_CONDITIONS without the constant gyro factor, plus the per-env
gyro factor range, the hidden zero shift and the 0-2 tick IMU age), through `robot_conditions.apply_robot_conditions`.
Pollen's ground pick draws the IMU age 0-3 ticks (delay_max_lag 3) and the adapter only widens a lag, so the ground
pick's base_ang_vel and projected_gravity lags are set to the adapter's IMU_MAX_LAG (2) first. Sit-stand draws 0-1.

recipe="bend4" (ground pick only): the with-pack bend effort's last audited reward and command
(extensions/flight-20261110/train/BEND-PLAN.md, NEXT-PLAN.md bend4, RUNLOG.md 2026-11-12; cmu/finetune/flight/
flight_worker.py DELTAS["bend4"]) ported to this body, without the flight pack's payload event (the body IS the load):
  - torque hinge: sum over actuators of max(0, |force| - 0.45 N m), weight -12 (fq2's, flight_worker TQ_LIMIT);
  - mouth_high_at_hold / mouth_low_at_rest (bend_terms.py), -240 each (bend2);
  - the runtime's held crouch: in half the cycles the phase stops at 0.33 for 4 s (bend3);
  - trunk_pitch_off_band -150 (25-45 deg in the hold), knee_near_limit -20 (|knee| > 1.25 rad), stance_off_at_rest -20;
  - the in-episode hand-back: in a quarter of the cycles the phase stays at 0 for 2 s after the rise and a frozen
    walker (default cmu/whole/runtime-overlay/models-bw-s2c/bw-s2c-b2.onnx, zero command) drives the joints;
  - bend_guard (weight 1e-9, watches only): stops training when the hold shows the squat or the fall rate doubles on
    two windows in a row. It raises `finetune_worker.StopTraining`; cmu/finetune/kaggle/skill_worker.py provides it.
The bend_terms functions are loaded from cmu/finetune/flight/bend_terms.py by file path (cmu/ is not a package).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from copy import deepcopy
from pathlib import Path

import torch
from mjlab.managers import EventTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task

from .microduck_ground_pick_env_cfg import MicroduckGroundPickRlCfg, make_microduck_ground_pick_env_cfg
from .microduck_sitstand_env_cfg import MicroduckSitStandRlCfg, make_microduck_sitstand_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
BEND_TERMS = _REPO_ROOT / "cmu" / "finetune" / "flight" / "bend_terms.py"
DEFAULT_HANDBACK_WALKER = _REPO_ROOT / "cmu" / "whole" / "runtime-overlay" / "models-bw-s2c" / "bw-s2c-b2.onnx"
RECIPES = ("pollen", "bend4")
CONDITIONS = ("official", "robot")

TQ_LIMIT = 0.45
BEND4_REWARDS = {  # term -> weight (flight_worker.py: "bend" torque hinge, BEND_MODS_CHOSEN, BEND4_MODS, bend_guard)
    "torque_margin": -12.0,
    "mouth_high_at_hold": -240.0,
    "mouth_low_at_rest": -240.0,
    "trunk_pitch_off_band": -150.0,
    "knee_near_limit": -20.0,
    "stance_off_at_rest": -20.0,
    "bend_guard": 1e-9,
}
BEND4_HOLD = {"phase": 0.33, "hold_s": 4.0, "frac": 0.5, "hb_frac": 0.25, "hb_s": 2.0}
BEND4_GUARD = {"window_steps": 6000, "skip_windows": 1, "knee_squat": 1.30, "pitch_min_deg": 20.0, "fall_factor": 2.0}


def load_bend_terms():
    if "bend_terms" in sys.modules and Path(sys.modules["bend_terms"].__file__).resolve() == BEND_TERMS.resolve():
        return sys.modules["bend_terms"]
    if not BEND_TERMS.exists():
        raise FileNotFoundError(f"{BEND_TERMS} not found: the bend4 recipe needs the repository checkout")
    spec = importlib.util.spec_from_file_location("bend_terms", BEND_TERMS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bend_terms"] = mod
    spec.loader.exec_module(mod)
    return mod


def actuator_force_over(env, limit: float):
    """cost (N m, >= 0): sum over actuators of max(0, |actuator force| - limit) (flight_worker.actuator_force_over)."""
    f = env.scene["robot"].data.actuator_force
    return torch.sum(torch.clamp(f.abs() - limit, min=0.0), dim=1)


def apply_bend4(cfg, walker: str | Path = DEFAULT_HANDBACK_WALKER):
    bt = load_bend_terms()
    tip = SceneEntityCfg("robot", site_names=["mouth_tip"])
    w = BEND4_REWARDS
    cfg.rewards["torque_margin"] = RewardTermCfg(func=actuator_force_over, weight=w["torque_margin"],
                                                 params={"limit": TQ_LIMIT})
    cfg.rewards["mouth_high_at_hold"] = RewardTermCfg(func=bt.mouth_high_at_hold, weight=w["mouth_high_at_hold"],
                                                      params={"asset_cfg": tip})
    cfg.rewards["mouth_low_at_rest"] = RewardTermCfg(func=bt.mouth_low_at_rest, weight=w["mouth_low_at_rest"],
                                                     params={"asset_cfg": tip})
    for name in ("trunk_pitch_off_band", "knee_near_limit", "stance_off_at_rest"):
        cfg.rewards[name] = RewardTermCfg(func=getattr(bt, name), weight=w[name], params={})
    cfg.rewards["bend_guard"] = RewardTermCfg(func=bt.bend_guard, weight=w["bend_guard"], params=dict(BEND4_GUARD))
    cfg.commands["twist"], _ = bt.held_phase_cfg(cfg.commands["twist"], dict(BEND4_HOLD))
    cfg.events["handback"] = EventTermCfg(func=bt.install_handback, mode="startup",
                                          params={"stander_path": str(Path(walker).resolve())})
    cfg.__dict__["_bend4"] = dict(rewards=dict(w), hold=dict(BEND4_HOLD), guard=dict(BEND4_GUARD), walker=str(walker))
    return cfg


def robot_route_conditions() -> dict:
    """scratch_worker.route_conditions("robot"): ROBOT_CONDITIONS without the constant gyro factor, plus the per-env
    gyro factor range, the hidden zero shift range and the IMU age (tests/ check the two stay equal)."""
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE, IMU_MAX_LAG, ROBOT_CONDITIONS, ZERO_SHIFT_RANGE
    c = {k: v for k, v in ROBOT_CONDITIONS.items() if k != "gyro_noise_legspeed"}
    return dict(c, gyro_noise_legspeed_range=GYRO_LEGSPEED_RANGE, zero_shift_range=dict(ZERO_SHIFT_RANGE),
                imu_max_lag=IMU_MAX_LAG)


def apply_skill_conditions(cfg, conditions: str):
    if conditions == "official":
        return cfg
    if conditions != "robot":
        raise ValueError(f"conditions {conditions!r}: choose from {CONDITIONS}")
    from mjlab_microduck.robot_conditions import apply_robot_conditions
    cond = robot_route_conditions()
    actor = cfg.observations["actor"].terms
    lag = int(cond["imu_max_lag"])
    for t in ("base_ang_vel", "projected_gravity"):
        if actor[t].delay_max_lag > lag:  # ground pick: 3 -> 2 (the adapter only widens)
            actor[t] = deepcopy(actor[t])
            actor[t].delay_max_lag = lag
    apply_robot_conditions(cfg, cond)
    cfg.__dict__["_robot_conditions"] = cond
    return cfg


def _swap_body(cfg, variant: str, tag: str):
    from .cmu_asbuilt import cmu_robot_spec_fn, load_apply_masses
    am = load_apply_masses()
    if am.variant_foot_contact(am.resolve_variant(variant)):
        # apply_foot_contact_to_cfg is written for the velocity cfg's sensors and events; not wired for the skills
        raise ValueError(f"variant {variant} names a foot-contact layer; the skill cfgs do not support one")
    robot = cfg.scene.entities["robot"]
    spec_fn = cmu_robot_spec_fn(robot.spec_fn, variant, tag=tag)
    cfg.scene.entities = {**cfg.scene.entities, "robot": dataclasses.replace(robot, spec_fn=spec_fn)}
    return cfg


def make_cmu_ground_pick_env_cfg(variant: str = "current", recipe: str = "pollen", conditions: str = "official",
                                 play: bool = False, walker: str | Path = DEFAULT_HANDBACK_WALKER):
    if recipe not in RECIPES:
        raise ValueError(f"recipe {recipe!r}: choose from {RECIPES}")
    cfg = _swap_body(make_microduck_ground_pick_env_cfg(play=play), variant, "cmu-groundpick")
    if recipe == "bend4":
        apply_bend4(cfg, walker)
    return apply_skill_conditions(cfg, conditions)


# ── Sit-stand target override (default off) ──────────────────────────────────────────────────────────────────────
# Pollen's targets were measured on his 737 g body (microduck_sitstand_env_cfg.py 101-119: SIT keyframe knee +-1.35,
# hip pitch -+0.4079, STAND = HOME = default_joint_pos, STAND_Z 0.115, SIT_Z 0.060). apply_sit_targets replaces them
# for the sit-stand task only, without touching the action offset, default_joint_pos or the observation convention
# (both stay HOME, so the deployed runtime needs no change):
#   sit_pose     {servo index: rad} merged over Pollen's SIT keyframe -> sit_overrides of posture_pose_legs,
#                posture_pose_l1, posture_composite (and the seated spawn, unless spawn_pose is given)
#   stand_pose   {servo index: rad} merged over HOME -> the STAND target of the same three terms. Pollen's
#                mdp._posture_targets takes the STAND target from default_joint_pos (mdp.py _posture_targets,
#                `stand_target = _servo_default_joint_pos(env, asset)`), so these three terms are swapped for the
#                *_ov copies below, which differ only in that STAND target. Standing spawns stay at HOME
#                (reset_robot_joints offsets default_joint_pos).
#   stand_z / sit_z   posture_height, posture_height_sharp, posture_height_l1, posture_stillness, posture_composite
#                and the SitStandCommand (its blend initialisation from the spawn height)
#   spawn_pose   {servo index: rad} merged over Pollen's SIT keyframe for the seated spawn only
#   spawn_z      (lo, hi) of the seated spawn height (Pollen 0.06 to 0.075)
# Unchanged on purpose: upright_while_tall's gate heights (0.075 / 0.10), rise_bootstrap's 0.125 ceiling, the standing
# spawn heights (0.11 to 0.12). With no override the cfg is Pollen's unchanged (same objects, same functions).
SERVO_NAMES = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch",
               "head_pitch", "head_yaw", "head_roll", "right_hip_yaw", "right_hip_roll", "right_hip_pitch",
               "right_knee", "right_ankle")  # the 14-servo layout of microduck_sitstand_env_cfg.py
SIT_TARGET_KEYS = ("sit_pose", "stand_pose", "stand_z", "sit_z", "spawn_pose", "spawn_z", "upright_pitch_deg")
# upright_pitch_deg X: the upright terms active while seated measure tilt against a reference pitch of blend * X deg
# nose-down (roll 0), blend = the same slewed posture target the pose terms use (0 stand, 1 sit). Changed:
# upright_linear (body_upright_linear_ref), posture_composite's upright factor (posture_composite_ov), posture_stillness's
# tilt gate (posture_stillness_ref). Left: upright_while_tall (a standing term, zero below z 0.075),
# neck_strain_leaned (a neck-overload cost on a backward lean against gravity: physics, not a posture target).


def parse_pose(text: str | None) -> dict | None:
    """'left_knee=1.30,right_knee=-1.30' (servo names or indices, rad) -> {servo index: rad}; None for empty."""
    if not text:
        return None
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        k = k.strip()
        idx = int(k) if k.isdigit() else SERVO_NAMES.index(k)
        if not 0 <= idx < len(SERVO_NAMES):
            raise ValueError(f"servo index {idx} out of range")
        out[idx] = float(v)
    return out


def _stand_target_ov(env, asset, stand_overrides: dict):
    from . import mdp as M
    t = M._servo_default_joint_pos(env, asset).clone()
    for idx, val in stand_overrides.items():
        t[:, idx] = val
    return t


def _posture_targets_ov(env, asset, command_name, sit_overrides, stand_overrides):
    """mdp._posture_targets with the STAND target = HOME with stand_overrides applied (SIT as before: HOME with
    sit_overrides)."""
    from . import mdp as M
    blend = M._posture_blend(env, command_name)
    stand_target = _stand_target_ov(env, asset, stand_overrides)
    sit_target = M._servo_default_joint_pos(env, asset).clone()
    for idx, val in sit_overrides.items():
        sit_target[:, idx] = val
    return blend, stand_target + blend.unsqueeze(-1) * (sit_target - stand_target)


def posture_pose_match_ov(env, command_name, sit_overrides, joint_indices, stand_overrides, std=0.5,
                          asset_cfg=SceneEntityCfg("robot")):
    """mdp.posture_pose_match against _posture_targets_ov."""
    from . import mdp as M
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets_ov(env, asset, command_name, sit_overrides, stand_overrides)
    jp = M._servo_joint_pos(env, asset)[:, joint_indices]
    return torch.exp(-((jp - target[:, joint_indices]) / std) ** 2).mean(dim=-1)


def posture_pose_l1_ov(env, command_name, sit_overrides, joint_indices, stand_overrides,
                       asset_cfg=SceneEntityCfg("robot")):
    """mdp.posture_pose_l1 against _posture_targets_ov."""
    from . import mdp as M
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets_ov(env, asset, command_name, sit_overrides, stand_overrides)
    jp = M._servo_joint_pos(env, asset)[:, joint_indices]
    return -torch.abs(jp - target[:, joint_indices]).mean(dim=-1)


def _cos_tilt_ref(env, asset, command_name, upright_pitch_deg):
    """cos of the trunk tilt against the reference attitude: pitch blend * X deg nose-down (forward positive), roll 0,
    with blend = the SitStandCommand's slewed target (0 = stand, 1 = sit, mdp._posture_blend, the one the pose
    target uses). g_b = projected gravity in the body frame; the reference gravity is (sin th, 0, -cos th), so the
    result is g_b . g_ref. At th = 0 it equals mdp's 1 - 2 (qx^2 + qy^2) (= -g_b[2])."""
    from . import mdp as M
    th = M._posture_blend(env, command_name) * math.radians(upright_pitch_deg)
    g = torch.nan_to_num(asset.data.projected_gravity_b, nan=0.0)
    return g[:, 0] * torch.sin(th) - g[:, 2] * torch.cos(th)


def body_upright_linear_ref(env, command_name, upright_pitch_deg, asset_cfg=SceneEntityCfg("robot")):
    """mdp.body_upright_linear (no gate, as the sit env uses it) against the posture-following reference."""
    return _cos_tilt_ref(env, env.scene[asset_cfg.name], command_name, upright_pitch_deg)


def posture_stillness_ref(env, command_name, sit_z, stand_z, upright_pitch_deg, band_full=0.012, band_zero=0.03,
                          vel_std=0.05, tilt_full_deg=25.0, tilt_zero_deg=60.0, asset_cfg=SceneEntityCfg("robot")):
    """mdp.posture_stillness with its tilt gate measured against the posture-following reference."""
    from . import mdp as M
    asset = env.scene[asset_cfg.name]
    target_z, z = M._posture_height(env, command_name, sit_z, stand_z)
    v = torch.nan_to_num(asset.data.root_link_lin_vel_w, nan=0.0).norm(dim=-1)
    flag = env.command_manager.get_command(command_name)[:, 0]
    blend = M._posture_blend(env, command_name)
    ramp_done = ((flag - blend).abs() < 0.02).float()
    err = torch.abs(z - target_z)
    t = torch.clamp((band_zero - err) / max(band_zero - band_full, 1e-6), 0.0, 1.0)
    z_gate = t * t * (3.0 - 2.0 * t)
    cos_tilt = _cos_tilt_ref(env, asset, command_name, upright_pitch_deg)
    cos_full = math.cos(math.radians(tilt_full_deg))
    cos_zero = math.cos(math.radians(tilt_zero_deg))
    u = torch.clamp((cos_tilt - cos_zero) / max(cos_full - cos_zero, 1e-6), 0.0, 1.0)
    tilt_gate = u * u * (3.0 - 2.0 * u)
    return torch.exp(-((v / vel_std) ** 2)) * z_gate * tilt_gate * ramp_done


def posture_composite_ov(env, command_name, sit_overrides, joint_indices, sit_z, stand_z, stand_overrides,
                         height_std=0.03, upright_std=0.40, pose_std=0.40, head_std=None,
                         head_command_name="head_pose", upright_pitch_deg=None, asset_cfg=SceneEntityCfg("robot")):
    """mdp.posture_composite against _posture_targets_ov (height, upright and head factors as in mdp.py); with
    upright_pitch_deg the upright factor measures tilt against the posture-following reference."""
    from . import mdp as M
    asset = env.scene[asset_cfg.name]
    _, target = _posture_targets_ov(env, asset, command_name, sit_overrides, stand_overrides)
    target_z, z = M._posture_height(env, command_name, sit_z, stand_z)
    score = torch.exp(-((z - target_z) / height_std) ** 2)
    if upright_pitch_deg is None:
        quat = asset.data.root_link_quat_w
        tilt_sq = 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    else:  # 2 (qx^2 + qy^2) = 1 - cos(tilt)
        tilt_sq = 1.0 - _cos_tilt_ref(env, asset, command_name, upright_pitch_deg)
    score = score * torch.exp(-tilt_sq / (upright_std * upright_std))
    jp = M._servo_joint_pos(env, asset)[:, joint_indices]
    score = score * torch.exp(-((jp - target[:, joint_indices]) ** 2).mean(dim=-1) / (pose_std * pose_std))
    if head_std is not None:
        if not hasattr(env, "_head_pose_neck_ids"):
            ids, _ = asset.find_joints_by_actuator_names(M._NECK_JOINT_PATTERNS)
            env._head_pose_neck_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
        neck_ids = env._head_pose_neck_ids
        head_cmd = env.command_manager.get_command(head_command_name)
        actual = asset.data.joint_pos[:, neck_ids] - asset.data.default_joint_pos[:, neck_ids]
        score = score * torch.exp(-((actual - head_cmd) ** 2).mean(dim=-1) / (head_std * head_std))
    return score


_POSE_TERMS = {"posture_pose_legs": posture_pose_match_ov, "posture_pose_l1": posture_pose_l1_ov,
               "posture_composite": posture_composite_ov}
_Z_TERMS = ("posture_height", "posture_height_sharp", "posture_height_l1", "posture_stillness", "posture_composite")


def apply_sit_targets(cfg, sit_pose=None, stand_pose=None, stand_z=None, sit_z=None, spawn_pose=None, spawn_z=None,
                      upright_pitch_deg=None):
    """Override the sit-stand targets (see the block comment above). Every argument None leaves cfg untouched."""
    if all(v is None for v in (sit_pose, stand_pose, stand_z, sit_z, spawn_pose, spawn_z, upright_pitch_deg)):
        return cfg
    from . import microduck_sitstand_env_cfg as ss
    rec = {}
    sit = dict(ss.SITTING_TARGET_OVERRIDES)
    if sit_pose:
        sit.update(sit_pose)
        rec["sit_pose"] = {SERVO_NAMES[k]: v for k, v in sorted(sit.items())}
    if sit_pose or stand_pose:
        for name, fn in _POSE_TERMS.items():
            term = cfg.rewards[name]
            params = dict(term.params, sit_overrides=dict(sit))
            if stand_pose:
                params["stand_overrides"] = dict(stand_pose)
                cfg.rewards[name] = dataclasses.replace(term, func=fn, params=params)
            else:
                cfg.rewards[name] = dataclasses.replace(term, params=params)
        if stand_pose:
            rec["stand_pose"] = {SERVO_NAMES[k]: v for k, v in sorted(stand_pose.items())}
    if stand_z is not None or sit_z is not None:
        for name in _Z_TERMS:
            term = cfg.rewards[name]
            params = dict(term.params)
            if stand_z is not None:
                params["stand_z"] = float(stand_z)
            if sit_z is not None:
                params["sit_z"] = float(sit_z)
            cfg.rewards[name] = dataclasses.replace(term, params=params)
        cmd = cfg.commands["twist"]
        cfg.commands["twist"] = dataclasses.replace(
            cmd, sit_z=float(sit_z) if sit_z is not None else cmd.sit_z,
            stand_z=float(stand_z) if stand_z is not None else cmd.stand_z)
        rec.update(stand_z=cfg.commands["twist"].stand_z, sit_z=cfg.commands["twist"].sit_z)
    if sit_pose or spawn_pose or spawn_z is not None:
        ev = cfg.events["set_ground_state"]
        params = dict(ev.params)
        sp = dict(sit)
        if spawn_pose:
            sp.update(spawn_pose)
        params["sitting_joint_overrides"] = sp
        if spawn_z is not None:
            params["sitting_z_min"], params["sitting_z_max"] = float(spawn_z[0]), float(spawn_z[1])
        cfg.events["set_ground_state"] = dataclasses.replace(ev, params=params)
        rec["spawn"] = dict(pose={SERVO_NAMES[k]: v for k, v in sorted(sp.items())},
                            z=[params["sitting_z_min"], params["sitting_z_max"]])
    if upright_pitch_deg is not None:
        x = float(upright_pitch_deg)
        cmd_name = cfg.rewards["posture_composite"].params["command_name"]
        up = cfg.rewards["upright_linear"]
        if up.params.get("gate_z_below") is not None:
            raise ValueError("upright_linear carries a recovery gate; the reference copy is written for the ungated term")
        cfg.rewards["upright_linear"] = dataclasses.replace(
            up, func=body_upright_linear_ref,
            params=dict(up.params, command_name=cmd_name, upright_pitch_deg=x))
        st = cfg.rewards["posture_stillness"]
        cfg.rewards["posture_stillness"] = dataclasses.replace(
            st, func=posture_stillness_ref, params=dict(st.params, upright_pitch_deg=x))
        co = cfg.rewards["posture_composite"]
        params = dict(co.params, upright_pitch_deg=x)
        params.setdefault("stand_overrides", {})  # no --stand-pose: _ov with an empty override is Pollen's target
        cfg.rewards["posture_composite"] = dataclasses.replace(co, func=posture_composite_ov, params=params)
        rec["upright_pitch_deg"] = dict(sit=x, stand=0.0, follows="posture blend (SitStandCommand alpha)",
                                        terms=["upright_linear", "posture_composite", "posture_stillness"],
                                        unchanged=["upright_while_tall", "neck_strain_leaned"])
    cfg.__dict__["_sit_targets"] = rec
    return cfg


def make_cmu_sitstand_env_cfg(variant: str = "current", conditions: str = "official", play: bool = False,
                              sit_targets: dict | None = None):
    cfg = _swap_body(make_microduck_sitstand_env_cfg(play=play), variant, "cmu-sitstand")
    if sit_targets:
        bad = set(sit_targets) - set(SIT_TARGET_KEYS)
        if bad:
            raise ValueError(f"sit_targets keys {sorted(bad)}: choose from {SIT_TARGET_KEYS}")
        apply_sit_targets(cfg, **sit_targets)
    return apply_skill_conditions(cfg, conditions)


def _rl_cfg(base, name: str, variant: str):
    from .cmu_asbuilt import run_label
    return dataclasses.replace(base, experiment_name=f"{name}_cmu_{variant}", run_name=f"{name}_cmu_{run_label(variant)}")


# (task id, builder, kwargs, rl cfg base, experiment name); conditions official (the worker builds robot ones itself)
CMU_SKILL_TASKS = (
    ("Mjlab-GroundPick-Flat-MicroDuck-CMU-Current", make_cmu_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg,
     "ground_pick"),
    ("Mjlab-GroundPick-Flat-MicroDuck-CMU-Current-Bend4", make_cmu_ground_pick_env_cfg, {"recipe": "bend4"},
     MicroduckGroundPickRlCfg, "ground_pick_bend4"),
    ("Mjlab-SitStand-Flat-MicroDuck-CMU-Current", make_cmu_sitstand_env_cfg, {}, MicroduckSitStandRlCfg,
     "microduck_sitstand"),
)


def register_cmu_skill_tasks(runner_cls) -> None:
    for task_id, make, kw, rl, name in CMU_SKILL_TASKS:
        if kw.get("recipe") == "bend4" and not BEND_TERMS.exists():
            continue  # an installed package without the cmu/ checkout: the bend4 id needs cmu/finetune/flight
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make("current", **kw),
            play_env_cfg=make("current", play=True, **kw),
            rl_cfg=_rl_cfg(rl, name, "current"),
            runner_cls=runner_cls,
        )
