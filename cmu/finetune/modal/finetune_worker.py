"""Fine-tune the deployed walk61 policy (Mjlab-Velocity-Flat-MicroDuck) inside the Modal container.

Runs under the project venv (/root/venv). Everything that must survive a preemption or a laptop lid
lives on the Modal Volume at /vol/runs/<run_name>/:
  model_<it>.pt   rsl_rl checkpoints (actor, critic, optimizer, iteration, common_step_counter)
  model_best.pt   checkpoint with the highest mean episode reward after the critic warm-up
  progress.json   iteration, elapsed seconds, dollars, reward statistics (read on resume for the cap)
  progress.csv    one row per save
  <run_name>.onnx exported policy with the mjlab metadata (same path as scripts/export.py)
  run_record.json configuration, delta, throughput, cost

Cost and time caps are enforced here, in the job itself: the save hook (every --save-interval iterations)
compares elapsed wall time (this container plus previous attempts) against --max-hours and
--max-dollars and stops, saves and exports when either is reached.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mjlab_microduck.robot_conditions import (LEG_JOINTS_RE, ROBOT_CONDITIONS, apply_robot_conditions,  # noqa: F401
                                              make_drop_action_cfg, make_legspeed_gyro)

TASK = "Mjlab-Velocity-Flat-MicroDuck"
CURRICULA_DONE_STEP = 2000 * 24 + 24  # every velocity curriculum reaches its final stage by iter 2000
# --cmu-config: which mass manifest describes the robot. Use "current": the robot as it is now, the variant named in
# cmu/model/current.json (read from the repository copy the job runs from; its manifest sha256 is checked).
# "asbuilt" is --manifest (the 2026-10-15 base manifest by default, or any manifest passed, e.g. the flight pack's);
# "plugged", "battery" and "battery-20261104" are the full part lists cmu/model/manifests/<variant>.json, shipped by
# finetune.py and the Kaggle pack as /root/assets/manifests/<variant>.json. plugged and battery are historical.
CMU_CONFIGS = ("asbuilt", "plugged", "battery", "battery-20261104", "current")
MANIFESTS_DIR = "/root/assets/manifests"
CURRENT_FILE = Path(__file__).resolve().parents[3] / "cmu" / "model" / "current.json"  # repo root/cmu/model/current.json


class StopTraining(Exception):
    pass


# --------------------------------------------------------------------------------------------- deltas
ANKLE_HEEL_STOP_RAD = 1.36  # same as cmu/model/apply_masses.py, which this image-independent worker does not import


def apply_asbuilt_masses(robot_cfg, manifest_path: str, audit_path: str, remove_battery_kg: float):
    """Return a copy of the robot EntityCfg whose spec_fn applies the as-built mass manifest.

    Same arithmetic as cmu/whole/dynamics/evaluate.py add_signed_inertias (parallel-axis theorem on signed parts),
    applied to the MjSpec bodies as explicit inertials before mjlab compiles the scene.
    """
    import mujoco

    parts = json.loads(Path(manifest_path).read_text())["parts"]
    if remove_battery_kg:
        audit = json.loads(Path(audit_path).read_text())
        old = next(x for x in audit["models"]["official_current"]["parts"] if x["id"] == "np_f970")
        assert 0 <= remove_battery_kg < old["assumption_dependent_removal_upper_bound_g"] / 1000
        parts = parts + [dict(body="trunk_base", mass_kg=-remove_battery_kg,
                              position_m=old["uniform_mesh_com_in_body_m"],
                              inertia_about_com_kg_m2=(-remove_battery_kg * np.array(old["uniform_mesh_inertia_per_kg_m2"])).tolist())]
    base_spec_fn = robot_cfg.spec_fn

    def parallel(p):
        return np.dot(p, p) * np.eye(3) - np.outer(p, p)

    def patched_spec_fn():
        spec = base_spec_fn()
        for name in ("left_ankle", "right_ankle"):
            spec.joint(name).range = [-ANKLE_HEEL_STOP_RAD, ANKLE_HEEL_STOP_RAD]
        m = spec.copy().compile() if hasattr(spec, "copy") else spec.compile()
        report = {}
        for body in sorted({p["body"] for p in parts}):
            i = m.body(body).id
            mass = float(m.body_mass[i]); com = m.body_ipos[i].copy()
            R = np.zeros(9); mujoco.mju_quat2Mat(R, m.body_iquat[i]); R = R.reshape(3, 3)
            origin_i = R @ np.diag(m.body_inertia[i]) @ R.T + mass * parallel(com)
            first = mass * com
            for p in [p for p in parts if p["body"] == body]:
                dm = p["mass_kg"]; position = np.array(p["position_m"])
                if "inertia_about_com_kg_m2" in p:
                    di = np.array(p["inertia_about_com_kg_m2"])
                else:
                    size = np.array(p.get("dimensions_m", [.02, .02, .02]))
                    di = np.diag(dm / 12 * (np.sum(size ** 2) - size ** 2))
                    Q = np.zeros(9); mujoco.mju_quat2Mat(Q, np.array(p.get("rotation_wxyz", [1, 0, 0, 0])))
                    Q = Q.reshape(3, 3); di = Q @ di @ Q.T
                mass += dm; first += dm * position; origin_i += di + dm * parallel(position)
            assert mass > 0, body
            com = first / mass; inertia = origin_i - mass * parallel(com)
            values, vectors = np.linalg.eigh(inertia)
            assert np.all(values > 0) and values.max() <= values.sum() - values.max() + 1e-10, (body, values)
            if np.linalg.det(vectors) < 0:
                vectors[:, 0] *= -1
            b = spec.body(body)
            b.explicitinertial = True
            # the official MJCF gives fullinertia, so write the full matrix (MuJoCo derives the principal frame)
            b.mass = mass; b.ipos = com; b.iquat = [1, 0, 0, 0]; b.inertia = [0, 0, 0]
            b.fullinertia = [inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]]
            report[body] = dict(before_g=round(float(m.body_mass[i]) * 1000, 1), after_g=round(mass * 1000, 1))
        total_before = float(sum(m.body_mass)); 
        m2 = spec.copy().compile() if hasattr(spec, "copy") else spec.compile()
        print(f"[asbuilt] body masses patched: {report}")
        print(f"[asbuilt] total mass {total_before*1000:.1f} g -> {float(sum(m2.body_mass))*1000:.1f} g")
        return spec

    return dataclasses.replace(robot_cfg, spec_fn=patched_spec_fn)


DELTA_DEFAULT = dict(
    yaw_push=1.0,              # rad/s added to the root yaw rate at each push (was linear x/y only)
    joint_friction=(0.8, 2.0),  # BAM friction-budget scale per episode (was (0.9, 1.1))
    vin_range=(6.0, 8.2),       # per-env supply voltage (was (6.5, 8.2))
    vin_drop_gain=(0.0, 0.4),   # V per Nm of summed |tau| (was (0.0, 0.2))
    vin_min=5.0,                # floor on the sagged voltage (was 6.0)
    gyro_noise=0.15,            # uniform +-rad/s on base_ang_vel (was 0.03)
    joint_vel_noise=0.75,       # uniform +-rad/s on joint_vel (was 0.25)
    action_rate=-2.0,           # final action_rate_l2 weight (was -1.0)
)


DELTA_STAGED = dict(
    yaw_push=0.6,              # half-way to the measured 1.0 rad/s yaw kicks
    joint_friction=(0.8, 1.5),  # ft1 (0.8, 2.0) made walking fall-prone enough that standing scored higher
    vin_range=(6.0, 8.2),
    vin_drop_gain=(0.0, 0.3),
    vin_min=5.5,
    gyro_noise=0.10,
    joint_vel_noise=0.50,
    action_rate=-1.0,          # Pollen's final weight; ft1's -2.0 taxed stepping and the policy stood still (env_cfg note, run 5yay13u4)
)
# ft3: ft1 and ft2 drifted to standing under Pollen's reward because its track_linear_velocity (Gaussian on the
# INSTANTANEOUS base velocity, std^2 0.1) is indifferent between standing and walk61's gait: the lateral/vertical
# velocity oscillation of a stepping 25 cm biped (as-built CPU model at 0.25 m/s: vy std 0.17, vz std 0.08 m/s)
# costs the walking policy as much as its forward progress earns, so the audit paid standing 1.35/s and walking
# 1.32/s. Same structure as the head-tracking tax Pollen fixed with head_pose_bias (price the escapable DC part, let
# the oscillation cancel): the term is replaced by the same Gaussian on an EMA of the base velocity (tau 0.5 s,
# about one stride) with the std tightened to sqrt(0.05). Measured on the walk61 traces (FINETUNE-DELTA.md, ft3):
# at the 0.25 m/s command the term pays walking 1.37/s vs standing 0.57/s (raw: 1.23 vs 1.07).
DELTA_LOWCMD = dict(DELTA_STAGED, track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)))
# slip (2026-10-21): walk61 on a slippery floor. The robot tripped forward at 1.2-1.3 s on a smooth floor with bare
# printed feet and walked clean on a silicone mat; the as-built sim with --floor-mu reproduces it (mu 0.2 at 0.3 m/s:
# forward fall 0.8-0.9 s after the walk command in 2 of 3 seeds, actions up to 3.4 rad), while walk61 walks at mu 1.0.
# walk61 was trained with foot friction DR 0.7-1.3; Pollen's earlier range was (0.3, 1.2) ("narrowed" note in
# microduck_velocity_env_cfg.py). Only two changes from Pollen's recipe: the foot friction range, and ft3's EMA velocity
# tracking, without which a fine-tune of walk61 drifts to standing (ft1/ft2). Actuator, voltage and noise stay Pollen's.
DELTA_SLIP = dict(foot_friction=(0.2, 1.3), track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)))
# tether (2026-10-30): walk61 with the charger lead (cmu/mac/tether.py: hanging weight, floor drag growing with the
# distance walked, snags, a taut pull capped by the bank sliding, twist with slip). On the robot in-place turns
# stalled after 40-90 deg (cmu/mac/turn-inplace-20261030/README.md): the hip yaws use 1-2 % of the XL330 torque
# because walk61 never met a yaw disturbance in training. Changes from Pollen's recipe: the lead on 80 % of the
# envs, and ft3's EMA velocity tracking, without which a fine-tune of walk61 drifts to standing (ft1/ft2).
DELTA_TETHER = dict(tether=dict(fraction=0.8), track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)))
# play (2026-11-12): walk61 with slack between servo and link (horn, screws, brackets) that the encoder does not see.
# The effect-by-effect replay sweep (cmu/SIM2REAL.md, "Effect-by-effect replay sweep") reproduced robot falls with
# head/neck play 8 deg total, leg play 4 deg and ankle-to-foot play 6 deg, each without felling the walks the robot
# completed; walk61 never met such slack. Each servo joint gets a passive hinge right after it on the same axis
# (Pollen's add_backlash.py geometry, but the encoder and the firmware loop stay on the servo side, unlike the
# -Backlash- tasks whose encoder reads through gear play). Its half-width is drawn per episode, uniform in
# [0, PLAY_HALF_DEG], and none_fraction of the episodes have none. Obs, actions and the runtime are unchanged.
PLAY_HALF_DEG = {r"^(neck_pitch|head_pitch|head_yaw|head_roll)$": 4.0,
                 r"^(left|right)_ankle$": 3.0,
                 r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee)$": 2.0}
DELTA_PLAY = dict(play=dict(half_deg=PLAY_HALF_DEG, none_fraction=0.2),
                  track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)))
# batshift (2026-11-22, battery-walk-20261122 TRAIN-CONTRACT.json stage C1): walk61 under the robot's condition with the
# 3000 mAh battery backpack. Only the robot model changes (Shawn 2026-11-22: rewards and the exam stay as they are).
# Run with --cmu-config current (battery-v2m5-20261121 masses). Changes from Pollen's recipe:
#   zero_shift_deg  the brain's zero shift outside the policy (A7Z ~/duck-zero-offset.json, pi/joint_signs.py
#                   MOTOR_OFFSET): motor = policy goal + shift, reading = true angle - shift. In this env that is the
#                   default joint position + shift, used by the action offset, the joint observation (actor and
#                   critic), the reset pose and the pose terms alike; the policy's frame is unchanged. The exported
#                   ONNX metadata keeps the unshifted default (walk_brain.py reads it and applies the shift itself).
#   foot_friction   the existing foot friction DR widened from (0.7, 1.3) to (0.4, 1.3): the foot geoms have contact
#                   priority 1 over the floor, so the old range never reached the mu 0.4 floor of stage Y1.
# Every reward term and weight, actuator, voltage, noise and push stays Pollen's.
ZERO_SHIFT_DEG = {"left_hip_roll": 5.5, "right_hip_roll": -3.5, "head_roll": -3.5, "head_yaw": -5.0}
DELTA_BATSHIFT = dict(zero_shift_deg=ZERO_SHIFT_DEG, foot_friction=(0.4, 1.3))
# batreal (2026-11-22, battery-walk-20261122 TRAIN-CONTRACT.json stage C2): batshift plus the robot's measured sensor and
# actuator conditions (extensions/battery-walk-20261122/GAP-AUDIT.md, gaps 1-6). One change against C1: what the policy
# reads and how the servos answer; masses, shift, foot friction, rewards and commands stay C1's.
#   gyro_noise_legspeed  gyro (actor base_ang_vel) noise, Gaussian per axis per tick with sd = K x rms true leg joint
#                        speed (the 10 leg servos): robot gyro diff sd / leg rms 0.36-0.46, i.e. noise sd 0.25-0.33 x leg
#                        rms; jitter_sim.py --gyro-noise-dq is the same model. Added on top of Pollen's +-0.03.
#   joint_friction       BAM friction budget scale per episode (1.0, 1.5) (was (0.9, 1.1)): robot fits x1.25-1.4.
#   joint_vel_noise_sd   actor joint_vel noise Gaussian sd 0.5 rad/s (was uniform +-0.25, sd 0.14): robot 0.63-0.65.
#   servo_drop           fraction of policy ticks whose servo read is dropped: the brain skips the policy and holds the
#                        last goals and its last action (walk_brain.py tick: no policy step without a fresh sample;
#                        brain_sim_check.py --drop). Here the applied action and the last-action history keep the
#                        previous tick's values on those ticks. Robot 2.2-4.4 % of walk ticks.
#   gravity_noise        actor projected_gravity uniform +-0.02 (was +-0.01): robot diff sd 0.016-0.018.
#   joint_pos_lag        actor joint_pos one policy tick old (was 0): robot joint reading 17 ms old, tick 21 ms.
# Since 2026-11-22 these six values and their implementation (leg-speed gyro, servo-drop action term) live in
# src/mjlab_microduck/robot_conditions.py (ROBOT_CONDITIONS), which the -CMU-Current tasks apply by default; batreal
# reuses them so the fine-tune and the default training env never drift apart (docs/SIM-TO-REAL-GAPS.md).
DELTA_BATREAL = dict(DELTA_BATSHIFT, **ROBOT_CONDITIONS)
DELTAS = {"default": DELTA_DEFAULT, "staged": DELTA_STAGED, "lowcmd": DELTA_LOWCMD, "slip": DELTA_SLIP,
          "tether": DELTA_TETHER, "play": DELTA_PLAY,
          "playctl": dict(track_lin_vel=DELTA_PLAY["track_lin_vel"]),  # play's control arm: the same run without play
          "batshift": DELTA_BATSHIFT, "batreal": DELTA_BATREAL}


ZERO_SHIFT_ORIG: dict = {}  # the unshifted default joint position patterns of the last apply_zero_shift


def apply_zero_shift(robot_cfg, shift_deg: dict):
    """Return a copy of the robot EntityCfg whose default joint position is the original one + shift (deg) on the
    named joints: exact-name patterns placed first, since mjlab's resolve_expr takes the first matching pattern."""
    from mjlab.utils.string import resolve_expr
    orig = dict(robot_cfg.init_state.joint_pos)
    new = {}
    for name, deg in shift_deg.items():
        base = resolve_expr(orig, (name,), None)[0]
        assert base is not None, f"zero shift: no default joint position pattern matches {name}"
        new[f"^{name}$"] = float(base) + math.radians(deg)
    new.update(orig)
    init = dataclasses.replace(robot_cfg.init_state, joint_pos=new)
    ZERO_SHIFT_ORIG.clear()
    ZERO_SHIFT_ORIG.update(orig)  # for check_zero_shift (later robot cfg copies, e.g. the mass patch, drop attributes)
    return dataclasses.replace(robot_cfg, init_state=init)


def zero_shift_vector(joint_names, shift_deg: dict):
    """shift in rad per joint, in the robot's joint order (0 for joints without a shift)"""
    missing = sorted(set(shift_deg) - set(joint_names))
    assert not missing, f"zero shift joints not in the robot: {missing}"
    return [math.radians(shift_deg.get(n, 0.0)) for n in joint_names]


def check_zero_shift(env, shift_deg: dict, orig_joint_pos: dict):
    """after the env is built: default_joint_pos = original default + shift on every joint, and the action offset
    of joint_pos is that default (motor goal = policy goal + shift)"""
    from mjlab.utils.string import resolve_expr
    robot = env.scene["robot"]
    names = tuple(robot.joint_names)
    want = torch.tensor([b + s for b, s in zip(resolve_expr(orig_joint_pos, names, 0.0), zero_shift_vector(names, shift_deg))],
                        device=robot.data.default_joint_pos.device)
    got = robot.data.default_joint_pos[0]
    err = float((got - want).abs().max())
    assert err < 1e-6, f"zero shift: default_joint_pos differs from original + shift by {err}"
    term = env.action_manager.get_term("joint_pos")
    off = term._offset[0] if isinstance(term._offset, torch.Tensor) else None  # noqa: SLF001
    if off is not None:
        exp = got[term._target_ids] if hasattr(term, "_target_ids") else got  # noqa: SLF001
        err2 = float((off - exp).abs().max())
        assert err2 < 1e-6, f"zero shift: action offset differs from the shifted default by {err2}"
    print(f"[zero-shift] default + shift on {sorted(shift_deg)} ({shift_deg} deg); action offset follows", flush=True)


def unshift_metadata(meta: dict, shift_deg: dict) -> dict:
    """the exported default_joint_pos without the shift: the runtime reads it from the ONNX and adds the shift itself"""
    names = meta["joint_names"]
    sv = zero_shift_vector(names, shift_deg)
    meta = dict(meta)
    # rounded to 1e-9 and + 0.0: a float32 0 + shift - shift prints "-0.000", which walk_brain.py's string compare of
    # default_joint_pos across policy slots would reject
    meta["default_joint_pos"] = [round(float(v) - s, 9) + 0.0 for v, s in zip(meta["default_joint_pos"], sv)]
    return meta


def install_runtime_delta(env, delta):
    """checks on a built env (ManagerBasedRlEnv, unwrapped) for the parts of a delta that need one"""
    if delta and delta.get("zero_shift_deg"):
        check_zero_shift(env, delta["zero_shift_deg"], ZERO_SHIFT_ORIG)


def export_metadata(env, run_path: str, delta):
    from mjlab.rl.exporter_utils import get_base_metadata
    meta = get_base_metadata(env, run_path=run_path)
    if delta and delta.get("zero_shift_deg"):
        meta = unshift_metadata(meta, delta["zero_shift_deg"])
    return meta


PLAY_SUFFIX = "_play"
PLAY_EPS = 1e-4  # rad; the half-width of a "no play" hinge (a zero range is degenerate)


def add_play_joints(robot_cfg, half_deg: dict):
    """Return a copy of the robot EntityCfg with a passive_<servo>_play hinge after each matched servo joint, built at
    the largest half-width; randomize_play narrows it per episode."""
    import re
    import mujoco

    base_spec_fn = robot_cfg.spec_fn

    def patched_spec_fn():
        spec = base_spec_fn()
        added = []
        for j in list(spec.joints):
            if j.name.startswith("passive_") or j.type != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            h = next((math.radians(d) for pat, d in half_deg.items() if re.search(pat, j.name)), None)
            if not h:
                continue
            pj = j.parent.add_joint(name=f"passive_{j.name}{PLAY_SUFFIX}", type=mujoco.mjtJoint.mjJNT_HINGE,
                                    axis=list(j.axis), pos=list(j.pos), range=[-h, h], limited=True, damping=0.01)
            pj.armature = 0.001
            pj.solref_limit = [0.01, 1]  # stiff tooth contact, as add_backlash.py
            pj.solimp_limit = [0.95, 0.999, 0.0001, 0.5, 2]
            added.append(j.name)
        print(f"[play] hinges after {added}", flush=True)
        return spec

    return dataclasses.replace(robot_cfg, spec_fn=patched_spec_fn)


def _make_randomize_play():
    from mjlab.managers.event_manager import requires_model_fields
    import torch

    @requires_model_fields("jnt_range")
    def randomize_play(env, env_ids, half_deg: dict, none_fraction: float, asset_cfg):
        """Reset event: each play hinge's half-width uniform in [0, its maximum] per episode, none_fraction of the
        episodes without play; the hinge starts centred. Writes absolute values, so nothing accumulates."""
        import re
        asset = env.scene[asset_cfg.name]
        cache = env.__dict__.setdefault("_play_cache", {})
        if "ids" not in cache:
            ids, names = asset.find_joints(r"^passive_.*" + PLAY_SUFFIX + "$")
            hmax = [math.radians(next(d for pat, d in half_deg.items()
                                      if re.search(pat, n[len("passive_"):-len(PLAY_SUFFIX)]))) for n in names]
            cache.update(ids=torch.as_tensor(ids, device=env.device),
                         hmax=torch.as_tensor(hmax, device=env.device, dtype=torch.float32))
        ids, hmax = cache["ids"], cache["hmax"]
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = env_ids.to(env.device, dtype=torch.long)
        n = len(env_ids)
        h = torch.rand(n, len(ids), device=env.device) * hmax
        h = torch.where((torch.rand(n, 1, device=env.device) < none_fraction).expand_as(h), torch.zeros_like(h), h)
        h = h.clamp(min=PLAY_EPS)
        jid = asset.indexing.joint_ids[ids].to(torch.long)
        rng = env.sim.model.jnt_range
        eg, jg = torch.meshgrid(env_ids, jid, indexing="ij")
        rng[eg, jg] = torch.stack([-h, h], dim=-1).to(rng.dtype)
        qa, va = asset.indexing.joint_q_adr[ids].to(torch.long), asset.indexing.joint_v_adr[ids].to(torch.long)
        env.sim.data.qpos[env_ids[:, None], qa[None, :]] = 0.0
        env.sim.data.qvel[env_ids[:, None], va[None, :]] = 0.0

    return randomize_play


def apply_play(env_cfg, play: dict):
    """Play hinges on the robot, their per-episode randomisation, and the servo-only selections the extra passive
    joints need (the same three fixes backlash.make_backlash_variant makes)."""
    from mjlab.managers import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from copy import deepcopy

    servo = (r"^(?!passive_).*",)
    env_cfg.scene.entities["robot"] = add_play_joints(env_cfg.scene.entities["robot"], play["half_deg"])
    env_cfg.events["randomize_play"] = EventTermCfg(
        func=_make_randomize_play(), mode="reset",
        params=dict(half_deg=play["half_deg"], none_fraction=play["none_fraction"], asset_cfg=SceneEntityCfg("robot")))
    for group in ("actor", "critic"):
        for name in ("joint_pos", "joint_vel"):
            term = env_cfg.observations[group].terms.get(name)
            if term is not None and "asset_cfg" not in term.params:
                term.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=servo)
    lim = env_cfg.rewards.get("dof_pos_limits")
    if lim is not None and "asset_cfg" not in lim.params:
        lim.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=servo)
    pose = env_cfg.rewards.get("pose")
    if pose is not None and "asset_cfg" in pose.params:
        ac = deepcopy(pose.params["asset_cfg"])
        ac.joint_names = tuple(r"^(?!passive_.*" + PLAY_SUFFIX + ")" + p.lstrip("^") for p in ac.joint_names)
        pose.params["asset_cfg"] = ac


class tether_lead:
    """Step event: the charger lead's wrench on trunk_base every env step (cmu/mac/tether.py), per-env parameters
    drawn at each episode start; a `fraction` of the envs carry it, the rest walk free."""

    def __init__(self, cfg, env):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mac"))
        import tether as T
        self.T, self.env = T, env
        self.robot = env.scene[cfg.params["asset_cfg"].name]
        self.body = self.robot.body_names.index("trunk_base")
        self.n, self.dev = env.num_envs, env.device
        self.U = lambda lo, hi, shape: torch.empty(shape, device=self.dev).uniform_(lo, hi)
        self.prm = T.sample(torch, self.n, self.U)
        self.on = torch.zeros(self.n, dtype=torch.bool, device=self.dev)
        self.fresh = torch.ones(self.n, dtype=torch.bool, device=self.dev)
        self.state, self.p_prev, self.yaw_prev, self.yaw = None, None, None, None

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.fresh[env_ids] = True

    def _exit(self):
        from mjlab.utils.lab_api.math import quat_apply
        q = self.robot.data.root_link_quat_w
        return self.robot.data.root_link_pos_w + quat_apply(q, self.prm["exit_b"]), q

    def __call__(self, env, env_ids, fraction: float, asset_cfg):
        del env_ids, asset_cfg
        T = self.T
        p, q = self._exit()
        raw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        if self.state is None:
            self.yaw = raw.clone()
            self.state = T.start(torch, self.prm, self.robot.data.root_link_pos_w[:, :2], raw, p[:, :2])
            self.p_prev = p.clone()
        self.yaw = self.yaw + torch.remainder(raw - self.yaw + math.pi, 2 * math.pi) - math.pi
        f = self.fresh
        if f.any():  # new episode: new lead, new bank place, untwisted
            new = T.sample(torch, self.n, self.U)
            for k in self.prm:
                self.prm[k][f] = new[k][f]
            self.on[f] = torch.rand(int(f.sum()), device=self.dev) < fraction
            p, q = self._exit()
            self.yaw[f] = raw[f]
            st = T.start(torch, self.prm, self.robot.data.root_link_pos_w[:, :2], self.yaw, p[:, :2])
            for k in self.state:
                self.state[k][f] = st[k][f]
            self.p_prev[f] = p[f]
            self.fresh[:] = False
        dt = float(env.step_dt)
        v = (p - self.p_prev) / dt
        self.p_prev = p.clone()
        T.step_snags(torch, self.state, dt, self.U)
        force, torque = T.wrench(torch, self.prm, self.state, p, v, self.robot.data.root_com_pos_w, self.yaw)
        m = self.on.float()[:, None]
        self.robot.write_external_wrench_to_sim((force * m)[:, None], (torque * m)[:, None], body_ids=[self.body])


def track_linear_velocity_ema(env, std: float, command_name: str, tau_s: float = 0.5, asset_cfg=None):
    """mjlab's track_linear_velocity on an EMA of the base linear velocity (body frame) instead of the instantaneous one.

    exp(-(|cmd_xy - ema(v)_xy|^2 + ema(v)_z^2) / std^2). The EMA (time constant tau_s, reset at episode start like
    head_pose_bias_penalty) lets the stride oscillation cancel and prices only the mean velocity error, so walking at
    the command pays the full term and standing under a non-zero command pays exp(-|cmd|^2/std^2).
    """
    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)
    v = asset.data.root_link_lin_vel_b
    if not hasattr(env, "_track_vel_ema"):
        env._track_vel_ema = torch.zeros_like(v)
    fresh = env.episode_length_buf <= 1
    env._track_vel_ema[fresh] = 0.0
    alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-6))
    env._track_vel_ema = (1.0 - alpha) * env._track_vel_ema + alpha * v
    f = env._track_vel_ema
    err = torch.sum(torch.square(cmd[:, :2] - f[:, :2]), dim=1) + torch.square(f[:, 2])
    return torch.exp(-err / std ** 2)


def apply_delta(env_cfg, delta: dict):
    from mjlab.utils.noise import UniformNoiseCfg as Unoise

    # a key left out of the delta keeps Pollen's value (DELTA_SLIP sets only two)
    if "vin_range" in delta:
        robot = env_cfg.scene.entities["robot"]
        act = robot.articulation.actuators[0]
        act2 = dataclasses.replace(act, vin_range=tuple(delta["vin_range"]), vin_drop_gain_range=tuple(delta["vin_drop_gain"]), vin_min=delta["vin_min"])
        env_cfg.scene.entities["robot"] = dataclasses.replace(robot, articulation=dataclasses.replace(robot.articulation, actuators=(act2,)))
    if delta.get("yaw_push"):
        env_cfg.events["push_robot"].params["velocity_range"]["yaw"] = (-delta["yaw_push"], delta["yaw_push"])
    if "joint_friction" in delta:
        env_cfg.events["randomize_joint_friction"].params["scale_range"] = tuple(delta["joint_friction"])
    if "foot_friction" in delta:
        env_cfg.events["foot_friction"].params["ranges"] = tuple(delta["foot_friction"])
    if "gyro_noise" in delta:
        g, jv = delta["gyro_noise"], delta["joint_vel_noise"]
        env_cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-g, n_max=g)
        env_cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-jv, n_max=jv)
    if "action_rate" in delta:
        env_cfg.rewards["action_rate_l2"].weight = delta["action_rate"]
        stages = env_cfg.curriculum["action_rate_weight"].params["weight_stages"]
        stages[-1]["weight"] = delta["action_rate"]
    if delta.get("tether"):
        from mjlab.managers import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        env_cfg.events["tether_lead"] = EventTermCfg(func=tether_lead, mode="step",
                                                     params=dict(fraction=delta["tether"]["fraction"],
                                                                 asset_cfg=SceneEntityCfg("robot")))
    if delta.get("play"):
        apply_play(env_cfg, delta["play"])
    if delta.get("zero_shift_deg"):
        env_cfg.scene.entities["robot"] = apply_zero_shift(env_cfg.scene.entities["robot"], delta["zero_shift_deg"])
    # the robot's sensor and actuator conditions (batreal): the shared implementation the -CMU-Current tasks use;
    # joint_friction was applied above with the other deltas' friction ranges
    apply_robot_conditions(env_cfg, {k: delta[k] for k in ROBOT_CONDITIONS if k != "joint_friction" and delta.get(k)})
    if delta.get("track_lin_vel"):
        term = env_cfg.rewards["track_linear_velocity"]  # weight 2.0 unchanged: same reward mass, filtered velocity
        term.func = track_linear_velocity_ema
        term.params = dict(command_name=term.params["command_name"], std=delta["track_lin_vel"]["std"],
                           tau_s=delta["track_lin_vel"]["tau_s"])
    return env_cfg


def resolve_manifest(cmu_config: str, manifest: str, manifests_dir: str = MANIFESTS_DIR,
                     current_file: Path = CURRENT_FILE) -> str:
    """Path of the mass manifest for --cmu-config; exits with the repo file to add when a variant is not shipped.
    "current" resolves through cmu/model/current.json and checks the shipped manifest against its sha256."""
    if cmu_config == "asbuilt":
        return manifest
    expect_sha = None
    if cmu_config == "current":
        if not Path(current_file).exists():
            raise SystemExit(f"--cmu-config current: {current_file} is missing from the job's repository copy")
        cur = json.loads(Path(current_file).read_text())
        cmu_config, expect_sha = cur["variant"], cur["mass_manifest_sha256"]
        print(f"[asbuilt] cmu-config current = {cmu_config} ({current_file})", flush=True)
    path = Path(manifests_dir) / f"{cmu_config}.json"
    if not path.exists():
        raise SystemExit(f"--cmu-config {cmu_config}: {path} is missing. The repo file cmu/model/manifests/{cmu_config}.json must exist "
                         f"(finetune.py and the Kaggle pack copy cmu/model/manifests/*.json to {manifests_dir}/); "
                         f"it is the full part list for the {cmu_config} robot in the base manifest's schema.")
    if expect_sha is not None:
        import hashlib
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expect_sha:
            raise SystemExit(f"--cmu-config current: {path} has sha256 {got}, cmu/model/current.json records {expect_sha}")
    return str(path)


# --------------------------------------------------------------------------------------------- init from ONNX
def load_walk61_into_actor(actor, onnx_path: str, init_std: float):
    import onnx
    from onnx import numpy_helper

    m = onnx.load(onnx_path)
    W = {i.name: torch.tensor(numpy_helper.to_array(i)).clone() for i in m.graph.initializer}
    mean = W["obs_normalizer._mean"]
    # the exported graph divides by (std + eps); store std itself so the re-export reproduces the same divisor
    eps = float(getattr(actor.obs_normalizer, "eps", 0.0))
    std = next(v for k, v in W.items() if k.startswith("onnx::Div")) - eps
    assert float(std.min()) > 0, float(std.min())
    sd = actor.state_dict()
    new = {}
    for k in sd:
        if k == "obs_normalizer._mean":
            new[k] = mean
        elif k == "obs_normalizer._var":
            new[k] = std ** 2
        elif k == "obs_normalizer._std":
            new[k] = std
        elif k == "obs_normalizer.count":
            new[k] = torch.tensor(10 ** 9, dtype=sd[k].dtype)
        elif k.startswith("mlp."):
            new[k] = W[k]
        elif k == "distribution.std_param":
            new[k] = torch.full_like(sd[k], init_std)
        else:
            raise KeyError(f"unexpected actor key {k}")
    assert new["mlp.0.weight"].shape == sd["mlp.0.weight"].shape, (new["mlp.0.weight"].shape, sd["mlp.0.weight"].shape)
    actor.load_state_dict({k: v.to(sd[k].device, sd[k].dtype) for k, v in new.items()}, strict=True)
    return mean, std


def compare_onnx(a: str, b: str, n: int = 256, seed: int = 0) -> float:
    import onnxruntime as ort

    sa, sb = ort.InferenceSession(a), ort.InferenceSession(b)
    rng = np.random.default_rng(seed)
    obs = rng.normal(0, 0.5, size=(n, 61)).astype(np.float32)
    d = 0.0
    for o in obs:
        ya = sa.run(None, {sa.get_inputs()[0].name: o[None]})[0]
        yb = sb.run(None, {sb.get_inputs()[0].name: o[None]})[0]
        d = max(d, float(np.abs(ya - yb).max()))
    return d


# --------------------------------------------------------------------------------------------- eval
def eval_policy(env, actor, steps: int, device: str, noise_std: float = 0.0) -> dict:
    """Rollout of the current actor: falls and episode lengths per variant. Deterministic (distribution mean) by
    default; noise_std > 0 adds Gaussian action noise of that sd, the sampling PPO's start policy sees in training."""
    obs = env.get_observations().to(device)
    n = env.num_envs
    age = torch.zeros(n, device=device)
    lengths, falls, timeouts = [], 0, 0
    robot, cm, rm = env.unwrapped.scene["robot"], env.unwrapped.command_manager, env.unwrapped.reward_manager
    err_sum, along_sum, mov_n, low_sum, low_n = 0.0, 0.0, 0, 0.0, 0
    term_sum = torch.zeros(len(rm.active_terms), device=env.device)
    rew_sum = 0.0
    with torch.inference_mode():
        for _ in range(steps):
            act = actor(obs)  # deterministic output
            if noise_std:
                act = act + noise_std * torch.randn_like(act)
            obs, rew, dones, extras = env.step(act.to(env.device))
            obs = obs.to(device)
            age += 1
            term_sum += rm._step_reward.sum(0)  # per-step values before the dt scale, i.e. reward per simulated second
            rew_sum += float(rew.sum())
            cmd, v = cm.get_command("twist"), robot.data.root_link_lin_vel_b
            moving = cmd[:, 0].abs() > 0.15  # envs asked to walk forward/back at a clear speed
            if moving.any():
                err_sum += float((cmd[:, :2] - v[:, :2]).norm(dim=-1)[moving].sum())
                along_sum += float((v[:, 0] * torch.sign(cmd[:, 0]))[moving].sum())
                mov_n += int(moving.sum())
            low = moving & (cmd[:, 0].abs() < 0.3)
            if low.any():
                low_sum += float((v[:, 0] * torch.sign(cmd[:, 0]))[low].sum())
                low_n += int(low.sum())
            d = dones.to(device).bool()
            if d.any():
                to = extras.get("time_outs", torch.zeros_like(d)).to(device).bool()
                lengths += age[d].tolist()
                timeouts += int((d & to).sum()); falls += int((d & ~to).sum())
                age[d] = 0
    env_s = steps * n * 0.02
    return dict(steps=steps, envs=n, episodes=len(lengths), falls=falls, timeouts=timeouts,
                falls_per_env_minute=round(falls / env_s * 60, 3), mean_ep_len_s=round(float(np.mean(lengths)) * 0.02, 2) if lengths else None,
                survived_to_end=int((age >= steps).sum()),
                track_err_xy=round(err_sum / max(mov_n, 1), 3), speed_along_cmd=round(along_sum / max(mov_n, 1), 3),
                moving_samples=mov_n, low_cmd_speed=round(low_sum / max(low_n, 1), 3),
                reward_per_s=round(rew_sum / (n * steps) / env.unwrapped.step_dt, 3),
                terms_per_s={name: round(float(x) / (n * steps), 4) for name, x in zip(rm.active_terms, term_sum)})


# --------------------------------------------------------------------------------------------- stand-still guards
# Fine-tunes of walk61 have learned "stand still when told to walk" four times: ft1 (2026-10-16, total reward tripled
# while the robot stood), ft2, and battery-walk C2 (2026-11-22, delta batreal: the deploy gate's vx 0.25 walk moved
# -0.002 m against walk61's 0.577 m). Two guards, each judged on the task metric, never on total reward:
#   1. preflight (before training): under the run's own env, walk61 and two stand-still policies (stand61, and the zero
#      action = hold the default pose) are rolled out with the training start's action noise; the run is refused when a
#      stand-still policy earns >= --preflight-max-ratio of walk61's reward per second (PPO would be paid to stand).
#      The same preflight reports walk61's speed along the command in the reference env (no delta), for reading only.
#   2. tracking guard (during training, ON by default, factor 1.3, the value ft3-ft4 ran with): a save whose speed along
#      the command, overall or at the robot's everyday 0.15-0.3 m/s commands, falls below the warm-up baseline (frozen
#      walk61 in the same env) / factor or under --low-speed-floor, or whose tracking error rises above factor x
#      baseline, is not a best candidate; two such saves in a row stop the run. C1 and C2 replayed through it pass
#      (C2 walked on the robot 2026-11-22 although the sim deploy gate read dist_x -0.002 at vx 0.25).
PREFLIGHT_MAX_RATIO = 0.9


class ZeroActor:
    """Stand-still policy: action 0 = the default joint pose (plus the delta's zero shift) on every joint."""

    def __init__(self, num_actions: int):
        self.num_actions = num_actions

    def __call__(self, obs):
        return torch.zeros(obs.shape[0], self.num_actions, device=obs.device)


def degenerate_verdict(walk: dict, stills: dict, max_ratio: float = PREFLIGHT_MAX_RATIO) -> dict:
    """FAIL when the best stand-still policy earns >= max_ratio of walk61's reward per second. A walk61 reward at or
    below zero is compared on the same margin: stand-still fails when it is within (1 - max_ratio) x |walk| of it."""
    w = float(walk["reward_per_s"])
    name, best = max(stills.items(), key=lambda kv: kv[1]["reward_per_s"])
    s = float(best["reward_per_s"])
    limit = w - (1.0 - max_ratio) * abs(w)
    return dict(walk61_reward_per_s=w, stand_still=name, stand_still_reward_per_s=s,
                ratio=round(s / w, 3) if w > 0 else None, max_ratio=max_ratio, limit=round(limit, 4),
                verdict="FAIL" if s >= limit else "PASS")


def track_guard_check(factor, track, base, speed, sbase, low, lbase, floor, in_grace=False, rebasing=False):
    """(ok, reasons) for one save, judged on the training env's own commanded-speed tracking against the warm-up
    baseline (frozen walk61 in the same env). factor 0 = guard off."""
    # no samples this save, or no baseline yet (first save, no warm-up): nothing to judge
    if not factor or rebasing or track != track or base != base:
        return True, []
    reasons = []
    if base == base and track == track and track > factor * base:
        reasons.append(f"tracking error {track:.3f} > {factor} x {base:.3f}")
    if not in_grace and speed == speed:
        if sbase == sbase and sbase > 0:
            if speed < sbase / factor:
                reasons.append(f"speed along command {speed:.3f} < {sbase:.3f} / {factor}")
        elif speed < floor:
            reasons.append(f"speed along command {speed:.3f} < floor {floor}")
    if not in_grace and low == low and lbase == lbase and (low < lbase / factor or low < floor):
        reasons.append(f"low-command speed {low:.3f} < {lbase:.3f} / {factor} (floor {floor})")
    return not reasons, reasons


def build_eval_env(args, manifest, delta, device, num_envs):
    """The worker's env (delta + as-built masses) at num_envs, wrapped, curricula at their final stage."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    cfg = load_env_cfg(TASK, play=False)
    if delta is not None:
        apply_delta(cfg, dict(delta))
    if not args.no_asbuilt:
        cfg.scene.entities["robot"] = apply_asbuilt_masses(cfg.scene.entities["robot"], manifest, args.audit, args.remove_battery_kg)
    cfg.scene.num_envs = num_envs
    cfg.seed = args.seed
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    install_runtime_delta(env, delta)
    env = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(TASK).clip_actions)
    env.unwrapped.common_step_counter = CURRICULA_DONE_STEP
    return env


def preflight(args, manifest, delta, device, run_dir, stand_onnx=None) -> dict:
    """Guard 1 (see above): degenerate-policy audit in the run's env plus walk61's tracking in the reference env."""
    from mjlab.rl import MjlabOnPolicyRunner
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    stand_onnx = stand_onnx or str(Path(args.onnx).with_name("stand61.onnx"))
    steps, n, noise = args.preflight_steps, args.preflight_envs, args.init_std
    out = dict(key=preflight_key(args), delta=delta, steps=steps, envs=n, action_noise_std=noise,
               max_ratio=args.preflight_max_ratio, time=time.strftime("%Y-%m-%dT%H:%M:%S"))

    def rollouts(d, with_stills):
        env = build_eval_env(args, manifest, d, device, n)
        runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env, dataclasses.asdict(load_rl_cfg(TASK)), str(run_dir / "preflight-tmp"), device)
        res = {}
        pols = [("walk61", args.onnx)] + ([("stand61", stand_onnx)] if with_stills and Path(stand_onnx).exists() else [])
        for name, path in pols:
            load_walk61_into_actor(runner.alg.actor, path, args.init_std)
            runner.alg.eval_mode()
            with torch.inference_mode():  # the env's buffers became inference tensors in the previous rollout
                env.reset()
            res[name] = eval_policy(env, runner.alg.actor, steps, device, noise_std=noise)
        if with_stills:
            with torch.inference_mode():
                env.reset()
            res["zero_action"] = eval_policy(env, ZeroActor(env.num_actions), steps, device, noise_std=noise)
        env.close()
        return res

    res = rollouts(delta, True)
    out["delta_env"] = res
    out["audit"] = degenerate_verdict(res["walk61"], {k: v for k, v in res.items() if k != "walk61"}, args.preflight_max_ratio)
    ref = rollouts(None, False)["walk61"] if delta is not None else res["walk61"]
    out["reference_env"] = dict(walk61=ref)
    for name, r in res.items():
        track = {k: v for k, v in r["terms_per_s"].items() if "track" in k}
        print(f"[preflight] delta env {name}: reward/s {r['reward_per_s']} tracking terms/s {track} speed along cmd "
              f"{r['speed_along_cmd']} low-cmd speed {r['low_cmd_speed']} falls/env-min {r['falls_per_env_minute']}", flush=True)
    print(f"[preflight] reference env (no delta) walk61: speed along cmd {ref['speed_along_cmd']} low-cmd speed "
          f"{ref['low_cmd_speed']} track err {ref['track_err_xy']} (report only)", flush=True)
    a = out["audit"]
    print(f"[preflight] audit: best stand-still {a['stand_still']} {a['stand_still_reward_per_s']} vs walk61 "
          f"{a['walk61_reward_per_s']} reward/s (ratio {a['ratio']}, max {a['max_ratio']}) -> {a['verdict']}", flush=True)
    (run_dir / "preflight.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def preflight_key(args) -> dict:
    """What a preflight record must match for kaggle_job.py to accept it for a run."""
    return dict(delta=args.delta, cmu_config=args.cmu_config, no_asbuilt=bool(args.no_asbuilt),
                onnx=Path(args.onnx).name, remove_battery_kg=args.remove_battery_kg)


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--root", default="/vol/runs")
    ap.add_argument("--onnx", default="/root/assets/walk61.onnx", help="init policy; for --eval-variants a comma list of policies to roll out")
    ap.add_argument("--manifest", default="/root/assets/asbuilt-mass-manifest.json", help="mass manifest for --cmu-config asbuilt")
    ap.add_argument("--cmu-config", choices=CMU_CONFIGS, default="asbuilt",
                    help=f"robot mass configuration: current = the robot as it is now (cmu/model/current.json; use this); "
                         f"asbuilt = --manifest; plugged/battery/battery-20261104 = {MANIFESTS_DIR}/<variant>.json "
                         f"(from cmu/model/manifests/; plugged and battery are historical)")
    ap.add_argument("--audit", default="/root/assets/existing-model-mass-audit.json")
    ap.add_argument("--remove-battery-kg", type=float, default=0.09)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iters", type=int, default=1000)
    ap.add_argument("--warmup-iters", type=int, default=100, help="critic-only iterations with the actor frozen")
    ap.add_argument("--init-std", type=float, default=0.25)
    ap.add_argument("--save-interval", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--max-dollars", type=float, required=True)
    ap.add_argument("--rate-per-hour", type=float, required=True)
    ap.add_argument("--gpu", default="?")
    ap.add_argument("--delta", default="default", help="'default', 'staged', 'none' or a JSON dict overriding DELTA_DEFAULT keys")
    ap.add_argument("--max-lr", type=float, default=0.0, help="clamp the adaptive PPO learning rate (0 = no clamp)")
    ap.add_argument("--low-speed-floor", type=float, default=0.03, help="m/s along a 0.15-0.3 m/s command below which the policy counts as not stepping")
    ap.add_argument("--rebaseline-saves", type=int, default=0, help="on a resume: discard the stored guard baseline and take the next N saves as the new one (guard off meanwhile)")
    ap.add_argument("--track-guard", type=float, default=1.3, help="tracking guard factor (ON by default): a save whose speed along the command falls below baseline / factor or whose tracking error exceeds factor x baseline is not a best candidate, two in a row stop the run; baselines = max(warm-up walk61, preflight anchor). 0 = off, needs --track-guard-off-reason")
    ap.add_argument("--track-guard-off-reason", default="", help="required with --track-guard 0: why this walking fine-tune may run without the stand-still guard")
    ap.add_argument("--preflight-envs", type=int, default=256, help="envs of the preflight rollouts (degenerate-policy audit, tracking anchor)")
    ap.add_argument("--preflight-steps", type=int, default=500, help="policy steps per preflight rollout (500 = 10 s)")
    ap.add_argument("--preflight-max-ratio", type=float, default=PREFLIGHT_MAX_RATIO, help="refuse the run when a stand-still policy earns at least this fraction of walk61's reward per second in the run's env")
    ap.add_argument("--allow-degenerate", default="", help="reason string: train although the preflight audit FAILs (the record and the log keep the FAIL)")
    ap.add_argument("--preflight-only", action="store_true", help="run the preflight, write preflight.json and exit 0 (the kaggle_job.py audit preset)")
    ap.add_argument("--track-grace", type=int, default=0, help="iterations before the speed checks of --track-guard apply (start policies that cannot walk in the env yet)")
    ap.add_argument("--no-asbuilt", action="store_true")
    ap.add_argument("--guard-fraction", type=float, default=0.5, help="stop when mean reward < fraction * peak for two saves")
    ap.add_argument("--guard-after", type=int, default=100, help="iterations after the warm-up before the reward guard may stop the run")
    ap.add_argument("--eval-variants", default="", help="comma list of baseline,asbuilt,delta,delta_asbuilt,staged,staged_asbuilt: evaluate --onnx deterministically in each env and exit")
    ap.add_argument("--eval-steps", type=int, default=750)
    ap.add_argument("--audit-max-ratio", type=float, default=0.8, help="reward audit: a standing policy may collect at most this fraction of the walking policy's reward per second")
    args = ap.parse_args()
    if not args.eval_variants and args.track_guard <= 0 and len(args.track_guard_off_reason.strip()) < 10:
        raise SystemExit("--track-guard 0 turns off the stand-still guard of a walking fine-tune (ft1, ft2 and battery-walk C2 "
                         "learned to stand when told to walk); give --track-guard-off-reason '<why, at least 10 characters>'")
    manifest = resolve_manifest(args.cmu_config, args.manifest)
    print(f"[asbuilt] cmu-config {args.cmu_config}: manifest {manifest}", flush=True)

    t_start = time.time()
    run_dir = Path(args.root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    prev = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    elapsed_prev = float(prev.get("elapsed_s", 0.0))

    import mjlab.tasks  # noqa: F401  (registers Mjlab-*-MicroDuck through the plugin entry point)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = "cuda:0"

    if args.eval_variants:
        results = {}
        for variant in args.eval_variants.split(","):
            cfg_v = load_env_cfg(TASK, play=False)
            # variant = "<delta>[_asbuilt]" with <delta> a DELTAS key, "delta" (= default), "baseline" or "asbuilt"
            dname = variant.split("_")[0]
            dv = None
            if dname == "delta":
                dv = dict(DELTA_DEFAULT)
                apply_delta(cfg_v, dv)
            elif dname in DELTAS:
                dv = dict(DELTAS[dname])
                apply_delta(cfg_v, dv)
            elif variant not in ("baseline", "asbuilt"):
                raise SystemExit(f"unknown eval variant {variant!r}: use baseline, asbuilt, delta, or a DELTAS key, each optionally _asbuilt")
            if "asbuilt" in variant:
                cfg_v.scene.entities["robot"] = apply_asbuilt_masses(cfg_v.scene.entities["robot"], manifest, args.audit, args.remove_battery_kg)
            cfg_v.scene.num_envs = args.num_envs
            cfg_v.seed = args.seed
            env_v = ManagerBasedRlEnv(cfg=cfg_v, device=device)
            install_runtime_delta(env_v, dv)
            env_v = RslRlVecEnvWrapper(env_v, clip_actions=load_rl_cfg(TASK).clip_actions)
            env_v.unwrapped.common_step_counter = CURRICULA_DONE_STEP
            runner_v = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(env_v, dataclasses.asdict(load_rl_cfg(TASK)), str(run_dir), device)
            results[variant] = {}
            for onnx_path in args.onnx.split(","):
                pname = Path(onnx_path).stem
                load_walk61_into_actor(runner_v.alg.actor, onnx_path, args.init_std)
                runner_v.alg.eval_mode()
                results[variant][pname] = eval_policy(env_v, runner_v.alg.actor, args.eval_steps, device)
                print(f"[eval] {variant} {pname}: {results[variant][pname]}", flush=True)
            if "walk61" in results[variant] and "stand61" in results[variant]:
                w, st = results[variant]["walk61"], results[variant]["stand61"]
                ratio = st["reward_per_s"] / w["reward_per_s"] if w["reward_per_s"] > 0 else float("inf")
                results[variant]["audit"] = dict(stand_over_walk=round(ratio, 3), max_ratio=args.audit_max_ratio,
                                                 verdict="PASS" if ratio <= args.audit_max_ratio else "FAIL",
                                                 walk_speed=w["speed_along_cmd"], stand_speed=st["speed_along_cmd"])
                print(f"[audit] {variant}: stand/walk reward {ratio:.3f} (max {args.audit_max_ratio}) -> {results[variant]['audit']['verdict']}; "
                      f"low-command (0.15-0.3 m/s) speed walk61 {w['low_cmd_speed']} stand61 {st['low_cmd_speed']}", flush=True)
            env_v.close()
        (run_dir / "eval.json").write_text(json.dumps(dict(onnx=args.onnx, cmu_config=args.cmu_config, manifest=manifest,
                                                           time=time.strftime("%Y-%m-%dT%H:%M:%S"), results=results), indent=1))
        return

    env_cfg = load_env_cfg(TASK, play=False)
    agent_cfg = load_rl_cfg(TASK)
    if args.delta == "none":
        delta = None
    else:
        delta = dict(DELTAS.get(args.delta, DELTA_DEFAULT))
        if args.delta not in DELTAS:
            delta.update(json.loads(args.delta))
        apply_delta(env_cfg, delta)
    if not args.no_asbuilt:
        env_cfg.scene.entities["robot"] = apply_asbuilt_masses(env_cfg.scene.entities["robot"], manifest, args.audit, args.remove_battery_kg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = args.save_interval
    agent_cfg.run_name = args.run_name
    agent_cfg.max_iterations = args.max_iters

    # guard 1: degenerate-policy audit + tracking anchor, once per run (a resume reuses the run's preflight.json)
    pf_path = run_dir / "preflight.json"
    pf = json.loads(pf_path.read_text()) if pf_path.exists() and not args.preflight_only else preflight(args, manifest, delta, device, run_dir)
    if args.preflight_only:
        print(f"[preflight] --preflight-only: {pf_path} written, verdict {pf['audit']['verdict']}", flush=True)
        return
    if pf["audit"]["verdict"] != "PASS":
        if not args.allow_degenerate.strip():
            raise SystemExit(f"[preflight] REFUSED: standing still pays in this env ({pf['audit']}); fix the reward or the "
                             f"delta, or pass --allow-degenerate '<reason>'")
        print(f"[preflight] audit FAIL overridden: --allow-degenerate {args.allow_degenerate!r}", flush=True)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    install_runtime_delta(env, delta)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_base = load_runner_cls(TASK) or MjlabOnPolicyRunner
    worker = {"stop": None}

    class CappedRunner(runner_base):
        peak = -math.inf
        below = 0
        warm = args.warmup_iters > 0
        last_save_t = time.time()
        last_it = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.track_sum, self.track_n, self.track_bad = 0.0, 0, 0
            self.track_base = list(prev.get("track_base_values", []))  # survives a resume
            self.speed_base = list(prev.get("speed_base_values", []))
            self.rebase_left = args.rebaseline_saves
            self.saves_seen = 0
            if self.rebase_left:
                self.track_base, self.speed_base, self.low_base = [], [], []
            orig_step = self.logger.process_env_step

            self.speed_sum, self.low_sum, self.low_n = 0.0, 0.0, 0
            self.low_base = list(prev.get("low_base_values", []))
            robot_h, cm_h = self.env.unwrapped.scene["robot"], self.env.unwrapped.command_manager

            def hooked_step(rewards, dones, extras, intrinsic_rewards=None):
                # per-step velocity tracking from the env state (the command manager's episode metric grows with
                # episode length, which hid ft1's drift behind longer survival): mean |cmd_xy - v_xy| and the speed
                # along the command over envs asked to move
                cmd, v = cm_h.get_command("twist"), robot_h.data.root_link_lin_vel_b
                along = v[:, 0] * torch.sign(cmd[:, 0])
                moving = cmd[:, 0].abs() > 0.15
                low = moving & (cmd[:, 0].abs() < 0.3)  # the robot's everyday 0.2-0.3 m/s commands: the first band a standing drift abandons
                if moving.any():
                    self.track_sum += float((cmd[:, :2] - v[:, :2]).norm(dim=-1)[moving].mean())
                    self.speed_sum += float(along[moving].mean())
                    self.track_n += 1
                if low.any():
                    self.low_sum += float(along[low].mean())
                    self.low_n += 1
                return orig_step(rewards, dones, extras, intrinsic_rewards)
            self.logger.process_env_step = hooked_step
            orig_update = self.alg.update

            def clamped_update():
                out = orig_update()
                if args.max_lr and self.alg.learning_rate > args.max_lr:
                    self.alg.learning_rate = args.max_lr
                    for g in self.alg.optimizer.param_groups:
                        g["lr"] = args.max_lr
                return out
            self.alg.update = clamped_update

        def save(self, path, infos=None):
            super().save(path, infos)
            it = self.current_learning_iteration
            now = time.time()
            elapsed = elapsed_prev + (now - t_start)
            dollars = elapsed / 3600 * args.rate_per_hour
            rew = float(np.mean(self.logger.rewbuffer)) if len(self.logger.rewbuffer) else float("nan")
            length = float(np.mean(self.logger.lenbuffer)) if len(self.logger.lenbuffer) else float("nan")
            its = (it - self.last_it) / (now - self.last_save_t) if self.last_it is not None and now > self.last_save_t else float("nan")
            self.last_it, self.last_save_t = it, now
            std = float(self.alg.get_policy().output_std.mean()) if hasattr(self.alg.get_policy(), "output_std") else float("nan")
            track = self.track_sum / self.track_n if self.track_n else float("nan")
            speed = self.speed_sum / self.track_n if self.track_n else float("nan")
            low = self.low_sum / self.low_n if self.low_n else float("nan")
            self.track_sum, self.speed_sum, self.track_n, self.low_sum, self.low_n = 0.0, 0.0, 0, 0.0, 0
            first_save = self.saves_seen == 0  # the first save after a start or resume covers the random-length init rollout
            self.saves_seen += 1
            if (self.warm or self.rebase_left) and track == track and not first_save:
                self.track_base.append(track)
                self.speed_base.append(speed)
                self.low_base.append(low)
                self.rebase_left = max(0, self.rebase_left - 1)
            base = float(np.mean(self.track_base)) if self.track_base else float("nan")
            sbase = float(np.mean(self.speed_base)) if self.speed_base else float("nan")
            lbase = float(np.nanmean(self.low_base)) if self.low_base else float("nan")
            # --track-grace N: a start policy that cannot walk in this env (walk61 under the flight pack: speed along
            # the command -0.04 in the warm-up) fails both speed checks from the first save; until iteration N only
            # the tracking-error check applies, from N on the speed checks apply as usual (2026-11-19 ts3m-w61long
            # was stopped at iteration 250, 150 actor iterations in). Guard off (0) never divides (2026-11-22
            # bw-c1-smoke ZeroDivisionError).
            in_grace = it < args.track_grace
            track_ok, guard_reasons = track_guard_check(
                args.track_guard, track, base, speed, sbase, low, lbase, args.low_speed_floor,
                in_grace=in_grace, rebasing=bool(self.rebase_left))
            if guard_reasons and not self.warm:
                print(f"[cap] tracking guard: save at iter {it} rejected as a best candidate: {'; '.join(guard_reasons)}", flush=True)
            row = dict(iter=it, elapsed_s=round(elapsed, 1), dollars=round(dollars, 3), mean_reward=rew, mean_ep_len=length,
                       it_per_s=round(its, 3) if its == its else None, lr=self.alg.learning_rate, action_std=std, warmup=self.warm,
                       track_err=round(track, 4) if track == track else None, track_base=round(base, 4) if base == base else None,
                       speed_along_cmd=round(speed, 4) if speed == speed else None, speed_base=round(sbase, 4) if sbase == sbase else None,
                       low_cmd_speed=round(low, 4) if low == low else None, low_cmd_base=round(lbase, 4) if lbase == lbase else None,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            progress_path.write_text(json.dumps({**row, "peak_reward": self.peak if self.peak > -math.inf else None,
                                                 "track_base_values": self.track_base, "speed_base_values": self.speed_base,
                                                 "low_base_values": self.low_base}, indent=1))
            with open(run_dir / "progress.csv", "a") as f:
                f.write(",".join(str(row[k]) for k in row) + "\n")
            print(f"[cap] it {it} elapsed {elapsed/60:.1f} min ${dollars:.2f} reward {rew:.3f} len {length:.0f} it/s {its:.2f} std {std:.3f} warm {self.warm} track {track:.3f}/{base:.3f} speed {speed:.3f}/{sbase:.3f} low {low:.3f}/{lbase:.3f}", flush=True)
            if self.warm and it >= args.warmup_iters:
                self.warm = False
                for p in self.alg.actor.parameters():
                    p.requires_grad_(True)
                self.alg.schedule = "adaptive"
                print(f"[cap] critic warm-up done at iter {it}; actor unfrozen, adaptive lr on", flush=True)
            if not self.warm and rew == rew:
                self.track_bad = self.track_bad + 1 if not track_ok else 0
                if rew > self.peak and track_ok:
                    self.peak = rew
                    torch.save(torch.load(path, map_location="cpu", weights_only=False), run_dir / "model_best.pt")
                self.below = self.below + 1 if rew < args.guard_fraction * self.peak else 0
            if hasattr(self, "volume_commit"):
                self.volume_commit()
            if elapsed >= args.max_hours * 3600:
                worker["stop"] = f"time cap {args.max_hours} h reached at iter {it}"
            elif dollars >= args.max_dollars:
                worker["stop"] = f"dollar cap ${args.max_dollars} reached at iter {it} (${dollars:.2f})"
            elif self.below >= 2 and it >= args.warmup_iters + args.guard_after:
                worker["stop"] = f"reward guard: {rew:.3f} < {args.guard_fraction} x peak {self.peak:.3f} on two saves at iter {it}"
            elif self.track_bad >= 2:
                worker["stop"] = (f"tracking guard: {'; '.join(guard_reasons)} (warm-up error {base:.3f}, speed {sbase:.3f}, low-command speed {lbase:.3f}; "
                                  f"factor {args.track_guard}, floor {args.low_speed_floor}) on two saves at iter {it}")
            if worker["stop"]:
                print("[cap] STOP:", worker["stop"], flush=True)
                raise StopTraining(worker["stop"])

    runner = CappedRunner(env, dataclasses.asdict(agent_cfg), str(run_dir), device)
    try:
        import modal
        vol = modal.Volume.from_name("microduck-rl")
        runner.volume_commit = vol.commit
    except Exception as e:  # noqa: BLE001
        print("[cap] no volume commit from inside the worker:", e)

    ckpts = sorted(run_dir.glob("model_[0-9]*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        runner.load(str(ckpts[-1]))
        print(f"[init] resumed from {ckpts[-1].name} at iteration {runner.current_learning_iteration}, elapsed before {elapsed_prev/60:.1f} min", flush=True)
        if runner.current_learning_iteration >= args.warmup_iters:
            runner.warm = False
        if prev.get("peak_reward") is not None:
            runner.peak = float(prev["peak_reward"])
    else:
        mean, std = load_walk61_into_actor(runner.alg.actor, args.onnx, args.init_std)
        env.unwrapped.common_step_counter = CURRICULA_DONE_STEP
        runner.current_learning_iteration = 0
        # plain torch.save: the runner's save() needs the logger writer that learn() creates later
        torch.save(runner.alg.save() | {"iter": 0, "infos": {"env_state": {"common_step_counter": CURRICULA_DONE_STEP}}}, run_dir / "model_0.pt")
        print(f"[init] actor initialised from {args.onnx}; curricula collapsed (common_step_counter={CURRICULA_DONE_STEP})", flush=True)
        # the exported model_0 must reproduce walk61 exactly
        runner.export_policy_to_onnx(str(run_dir), "model_0.onnx")
        d0 = compare_onnx(args.onnx, str(run_dir / "model_0.onnx"))
        print(f"[init] model_0.onnx vs walk61.onnx max abs diff on 256 random obs: {d0:.2e}", flush=True)
        assert d0 < 1e-4, d0
    if runner.warm:
        for p in runner.alg.actor.parameters():
            p.requires_grad_(False)
        runner.alg.schedule = "fixed"
        print(f"[init] critic warm-up: actor frozen for the first {args.warmup_iters} iterations", flush=True)

    record = dict(run_name=args.run_name, task=TASK, gpu=args.gpu, num_envs=args.num_envs, max_iters=args.max_iters,
                  warmup_iters=args.warmup_iters, init_std=args.init_std, seed=args.seed, delta=delta, asbuilt=not args.no_asbuilt,
                  cmu_config=args.cmu_config, manifest=manifest, remove_battery_kg=args.remove_battery_kg, caps=dict(max_hours=args.max_hours, max_dollars=args.max_dollars, rate_per_hour=args.rate_per_hour),
                  source_onnx=args.onnx, started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))

    remaining = args.max_iters - runner.current_learning_iteration
    stop_reason = "max_iters reached"
    try:
        if remaining > 0:
            runner.learn(num_learning_iterations=remaining, init_at_random_ep_len=True)
    except StopTraining as e:
        stop_reason = str(e)
    final_it = runner.current_learning_iteration
    final_ckpt = run_dir / f"model_{final_it}.pt"
    if not final_ckpt.exists():
        torch.save(runner.alg.save() | {"iter": final_it, "infos": {"env_state": {"common_step_counter": env.unwrapped.common_step_counter}}}, final_ckpt)
    runner.export_policy_to_onnx(str(run_dir), f"{args.run_name}.onnx")
    meta = export_metadata(runner.env.unwrapped, str(final_ckpt), delta)
    attach_metadata_to_onnx(str(run_dir / f"{args.run_name}.onnx"), meta)
    if (run_dir / "model_best.pt").exists():
        # the normalizer buffers became inference tensors during the rollouts; clone them so load_state_dict can copy in place
        for mod in (runner.alg.actor, runner.alg.critic):
            norm = getattr(mod, "obs_normalizer", None)
            for name in ("_mean", "_var", "_std"):
                if norm is not None and hasattr(norm, name):
                    setattr(norm, name, getattr(norm, name).detach().clone())
        runner.load(str(run_dir / "model_best.pt"))
        runner.export_policy_to_onnx(str(run_dir), f"{args.run_name}_best.onnx")
        attach_metadata_to_onnx(str(run_dir / f"{args.run_name}_best.onnx"), export_metadata(runner.env.unwrapped, str(run_dir / "model_best.pt"), delta))
    elapsed = elapsed_prev + (time.time() - t_start)
    record.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), final_iter=final_it, stop_reason=stop_reason,
                  elapsed_s=round(elapsed, 1), dollars=round(elapsed / 3600 * args.rate_per_hour, 3),
                  onnx=str(run_dir / f"{args.run_name}.onnx"))
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"[done] {stop_reason}; final iter {final_it}; {elapsed/60:.1f} min; ${record['dollars']:.2f}; onnx {record['onnx']}", flush=True)
    if hasattr(runner, "volume_commit"):
        runner.volume_commit()
    env.close()


if __name__ == "__main__":
    main()
