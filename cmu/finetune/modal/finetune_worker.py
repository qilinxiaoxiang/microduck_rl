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

TASK = "Mjlab-Velocity-Flat-MicroDuck"
CURRICULA_DONE_STEP = 2000 * 24 + 24  # every velocity curriculum reaches its final stage by iter 2000
# --cmu-config: which mass manifest describes the robot. "asbuilt" is --manifest (the 2026-10-15 base manifest);
# "plugged" and "battery" are the full part lists cmu/model/manifests/<variant>.json, shipped by finetune.py and the
# Kaggle pack as /root/assets/manifests/<variant>.json when they exist in the repo.
CMU_CONFIGS = ("asbuilt", "plugged", "battery")
MANIFESTS_DIR = "/root/assets/manifests"


class StopTraining(Exception):
    pass


# --------------------------------------------------------------------------------------------- deltas
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
# t1 (2026-10-30, cmu/grasp/PLAN-continuous-vision.md section 5): the body follows the head. walk61 on the robot holds
# the neck near HOME whatever is commanded (-0.09 rad absolute for -0.4..-0.8 asked, PICKUP.md) and stops stepping while a
# head command is active; the as-built CPU sim (brain_sim_check.py --neck-pitch, plugged manifest, 2026-10-30) shows the
# same policy behaviour (standing: -0.4 commanded -> -0.125 reached, -0.8 -> -0.26; with vx 0.25 and any neck command the
# duck does not step at all, path 0.007 m in 6 s), so this is the policy, not the sim-to-real gap, and the sim measures it.
# Changes to Pollen's recipe, nothing else (actuator, masses, noise, pushes stay as walk61 was trained):
#   twist: forward only, vx uniform 0..0.30 (the robot-tested 0.25 inside the range), small vy, yaw as trained; 15 % of
#          envs get the exact zero command (the deployment idle), was 25 % at the end of Pollen's curriculum;
#   head_pose: fixed final ranges biased to looking down (neck delta -1.10..+0.40, head_pitch -0.30..+1.00 = the gaze
#          loop's 0.35..1.35 absolute, yaw +-0.90, roll +-0.05), held 3-8 s per sample (a tracked target moves slowly);
#   rewards: ft3's EMA velocity tracking (std^2 0.05, tau 0.5 s) so stepping at 0.15-0.30 m/s beats standing by
#          0.7-1.5/s; head_pose_bias (L1 on the EMA head error) weight 3 -> 5 and tau 1.0 -> 0.5 s (the done criterion is
#          the DC error after 0.5 s); head_pose_tracking stays at std 0.5 (a tight instantaneous std killed walking,
#          Pollen's run 5yay13u4). tests/test_t1_delta.py scores the degenerate policies from these constants.
DELTA_T1 = dict(
    track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)),
    twist_ranges=dict(lin_vel_x=(0.0, 0.30), lin_vel_y=(-0.10, 0.10), ang_vel_z=(-1.0, 1.0)),
    rel_standing_envs=0.15,
    head_ranges=((-1.10, 0.40), (-0.30, 1.00), (-0.90, 0.90), (-0.05, 0.05)),   # neck_pitch, head_pitch, head_yaw, head_roll deltas from HOME
    head_resample_s=(3.0, 8.0),
    head_bias=dict(weight=5.0, tau_s=0.5),
)
# t2 (2026-10-30, FINETUNE-DELTA.md "Run t2 plan"): T1 made the neck follow on the robot but rewrote walk61's gait in
# states walk61 already handled (0.71 rad action distance on walk61's own robot walks, head at default) and fell in 2 of
# 5 walks. t2 = T1 + a behaviour-cloning anchor to walk61 (Wolczyk et al., ICML 2024): during the critic warm-up the
# actor IS walk61, so its mean actions on default-head states with vx 0 or 0.25-0.30 (walk61's robot-proven regime) are
# kept, and after every PPO update one supervised step pulls the actor back toward them (anchor_loss). 20 % of head
# resamples give the exact default head (zero_command_prob, as rel_standing_envs does for the twist), so the buffer
# fills and the deployment's most common head state stays trained. The actor's observation normalizer is frozen at
# walk61's for anchored runs: rsl_rl keeps updating it during a fine-tune (T1's drifted by up to 0.025 in mean, 3 % in
# std), which would move every input the anchor compares. Rewards are T1's.
DELTA_T2A = dict(DELTA_T1, default_head_envs=0.2, anchor=dict(beta=0.5, states=200_000, batch=4096, vx_band=(0.25, 0.30)))
DELTA_T2B = dict(DELTA_T2A, anchor=dict(DELTA_T2A["anchor"], beta=2.0))
# t3 (2026-10-30, FINETUNE-DELTA.md "Run t3 plan"): t2b walked like walk61 with the default head and shook like T1 as
# soon as the neck was commanded (robot 14 jumps/s), which is where the t2 anchor did not reach. Three changes: (1) the
# sim gap the jitter study identified goes into the training DR: BAM friction 1.0-1.4 (the study's calibrated value is
# the top; at 1.0-1.5 the audit-t3 run of 2026-10-30 had walk61 falling 0.527 per env-minute, over the 0.5 limit, and
# the 10/17 study already found x1.5 stops stepping) and gyro vibration proportional to leg speed (k 0.15-0.35, study
# value 0.25), actor observation only; (2) a second-difference action term on every state (weight: -1.0 placeholder for
# the audit, then set so its mass on walk61's own rollouts matches action_rate_l2's, recorded in the delta before the
# run; audit-t3b gave -0.58); (3) the anchor also covers head-command states with a
# counterfactual target: walk61's own leg actions on the same state with the head command zeroed, leg outputs only, at
# beta_legs (walk61 does not step with a head command, so walk61(s) itself would teach standing).
DELTA_T3A = dict(DELTA_T2B, joint_friction=(1.0, 1.4), gyro_vibration=dict(k_range=(0.15, 0.35)),
                 action_acc=dict(weight=-0.58),  # audit-t3b: walk61 masses/s action_rate -0.199, action_acc at -1.0 -0.342
                 anchor=dict(DELTA_T2B["anchor"], beta_legs=0.0))
DELTA_T3B = dict(DELTA_T3A, anchor=dict(DELTA_T3A["anchor"], beta_legs=1.0))
# audit-only diagnostics (no AUDIT_KEY, never trained): t2b plus one t3 DR change each, to attribute walk61's fall rate
DELTA_T3_FRIC = dict(DELTA_T2B, joint_friction=(1.0, 1.5))
DELTA_T3_VIB = dict(DELTA_T2B, gyro_vibration=dict(k_range=(0.15, 0.35)))
DELTAS = {"default": DELTA_DEFAULT, "staged": DELTA_STAGED, "lowcmd": DELTA_LOWCMD, "slip": DELTA_SLIP, "t1": DELTA_T1,
          "t2a": DELTA_T2A, "t2b": DELTA_T2B, "t3a": DELTA_T3A, "t3b": DELTA_T3B,
          "t3fric": DELTA_T3_FRIC, "t3vib": DELTA_T3_VIB}

# columns of the 61-input actor vector: 48 proprioception, then twist(3), head_pose(4), body_pose(6)
ACTOR_TWIST, ACTOR_HEAD = slice(48, 51), slice(51, 55)
# the 14 action outputs: 0-4 left leg, 5-8 neck/head, 9-13 right leg (ctrl index = joint index on the walk model)
LEG_OUT = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


def anchor_mask(x: torch.Tensor, vx_band, heads: bool = False) -> torch.Tensor:
    """Rows of actor inputs x (N, 61) the anchor keeps: forward command 0 or in vx_band, and (unless heads) the exact
    default head command."""
    vx = x[:, ACTOR_TWIST.start]
    keep = (vx == 0) | ((vx >= vx_band[0]) & (vx <= vx_band[1]))
    if not heads:
        keep &= (x[:, ACTOR_HEAD] == 0).all(dim=1)
    return keep


def actor_mean(actor, x: torch.Tensor) -> torch.Tensor:
    """Deterministic action of an rsl_rl MLPModel actor on concatenated raw actor inputs (the Gaussian mean)."""
    return actor.mlp(actor.obs_normalizer(x))


def anchor_loss(actor, x: torch.Tensor, mu0: torch.Tensor, beta: float) -> torch.Tensor:
    """beta x mean over states of the squared action distance to the anchor policy's mean actions mu0."""
    return beta * torch.square(actor_mean(actor, x) - mu0).sum(dim=1).mean()


def anchor_targets(actor, x: torch.Tensor, anchor: dict):
    """Anchor rows, targets and per-output weights for a batch of actor inputs x under the frozen walk61 actor.

    Default-head rows: target walk61(x), weight beta on all 14 outputs. Head-command rows (only when beta_legs > 0):
    target walk61(x with the head command zeroed), weight beta_legs on the leg outputs and 0 on the head outputs.
    Rows with no forward command in the band, and head rows when beta_legs is 0, are dropped.
    """
    beta_legs = float(anchor.get("beta_legs", 0.0))
    keep = anchor_mask(x, anchor["vx_band"], heads=beta_legs > 0)
    x = x[keep]
    head_default = (x[:, ACTOR_HEAD] == 0).all(dim=1)
    w = torch.zeros(len(x), 14, device=x.device, dtype=x.dtype)
    w[head_default] = anchor["beta"]
    w[torch.nonzero(~head_default).flatten()[:, None], torch.tensor(LEG_OUT, device=x.device)[None, :]] = beta_legs
    x0 = x.clone()
    x0[:, ACTOR_HEAD] = 0.0
    with torch.no_grad():
        mu0 = actor_mean(actor, x0)  # equals walk61(x) on default-head rows
    return x, mu0, w


def anchor_loss_w(actor, x: torch.Tensor, mu0: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mean over states of sum_j w_j (mu_j(s) - mu0_j)^2; with w = beta on every output it is anchor_loss."""
    return (w * torch.square(actor_mean(actor, x) - mu0)).sum(dim=1).mean()


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
    if delta.get("track_lin_vel"):
        term = env_cfg.rewards["track_linear_velocity"]  # weight 2.0 unchanged: same reward mass, filtered velocity
        term.func = track_linear_velocity_ema
        term.params = dict(command_name=term.params["command_name"], std=delta["track_lin_vel"]["std"],
                           tau_s=delta["track_lin_vel"]["tau_s"])
    # t1 keys. Curricula are collapsed to their final stage in a fine-tune (CURRICULA_DONE_STEP), and the curriculum
    # functions write the final stage back into the live cfg on every call, so the final stage is edited as well.
    if "twist_ranges" in delta:
        for k, v in delta["twist_ranges"].items():
            setattr(env_cfg.commands["twist"].ranges, k, tuple(v))
    if "rel_standing_envs" in delta:
        env_cfg.commands["twist"].rel_standing_envs = delta["rel_standing_envs"]
        env_cfg.curriculum["standing_envs"].params["standing_stages"][-1]["rel_standing_envs"] = delta["rel_standing_envs"]
    if "head_ranges" in delta:
        ranges = tuple(tuple(r) for r in delta["head_ranges"])
        env_cfg.commands["head_pose"].ranges = ranges
        env_cfg.curriculum["head_pose_range"].params["range_stages"][-1]["ranges"] = ranges
    if "head_resample_s" in delta:
        env_cfg.commands["head_pose"].resampling_time_range = tuple(delta["head_resample_s"])
    if "head_bias" in delta:
        hb = delta["head_bias"]
        env_cfg.rewards["head_pose_bias"].weight = hb["weight"]
        env_cfg.rewards["head_pose_bias"].params["tau_s"] = hb["tau_s"]
        env_cfg.curriculum["head_pose_bias_weight"].params["weight_stages"][-1]["weight"] = hb["weight"]
    if "default_head_envs" in delta:
        env_cfg.commands["head_pose"].zero_command_prob = delta["default_head_envs"]
    # t3 keys
    if "gyro_vibration" in delta:
        import mjlab_microduck.tasks.mdp as microduck_mdp

        term = env_cfg.observations["actor"].terms["base_ang_vel"]
        assert term.func is microduck_mdp.base_ang_vel_imu_misaligned, term.func  # the recipe's actor gyro (IMU misalignment DR)
        term.func = microduck_mdp.base_ang_vel_imu_misaligned_vibration
        term.params = dict(term.params, k_range=tuple(delta["gyro_vibration"]["k_range"]))
    if "action_acc" in delta:
        import mjlab_microduck.tasks.mdp as microduck_mdp
        from mjlab.managers import RewardTermCfg

        env_cfg.rewards["action_acc_l2"] = RewardTermCfg(func=microduck_mdp.action_acc_l2, weight=float(delta["action_acc"]["weight"]))
    return env_cfg


def resolve_manifest(cmu_config: str, manifest: str, manifests_dir: str = MANIFESTS_DIR) -> str:
    """Path of the mass manifest for --cmu-config; exits with the repo file to add when a variant is not shipped."""
    if cmu_config == "asbuilt":
        return manifest
    path = Path(manifests_dir) / f"{cmu_config}.json"
    if not path.exists():
        raise SystemExit(f"--cmu-config {cmu_config}: {path} is missing. The repo file cmu/model/manifests/{cmu_config}.json must exist "
                         f"(finetune.py and the Kaggle pack copy cmu/model/manifests/*.json to {manifests_dir}/); "
                         f"it is the full part list for the {cmu_config} robot in the base manifest's schema.")
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
def eval_policy(env, actor, steps: int, device: str) -> dict:
    """Deterministic rollout (distribution mean) of the current actor: falls and episode lengths per variant."""
    obs = env.get_observations().to(device)
    n = env.num_envs
    age = torch.zeros(n, device=device)
    lengths, falls, timeouts = [], 0, 0
    robot, cm, rm = env.unwrapped.scene["robot"], env.unwrapped.command_manager, env.unwrapped.reward_manager
    err_sum, along_sum, mov_n, low_sum, low_n = 0.0, 0.0, 0, 0.0, 0
    term_sum = torch.zeros(len(rm.active_terms), device=env.device)
    rew_sum = 0.0
    # head tracking (t1): |measured - HOME - command| per head joint, averaged over envs and steps (plain walk model:
    # the measured angle is the servo joint; the backlash twin is not used here)
    head_ids = [robot.joint_names.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
    head_err = torch.zeros(4, device=env.device)
    with torch.inference_mode():
        for _ in range(steps):
            act = actor(obs)  # deterministic output
            obs, rew, dones, extras = env.step(act.to(env.device))
            obs = obs.to(device)
            age += 1
            term_sum += rm._step_reward.sum(0)  # per-step values before the dt scale, i.e. reward per simulated second
            rew_sum += float(rew.sum())
            hcmd = cm.get_command("head_pose")
            head_err += ((robot.data.joint_pos[:, head_ids] - robot.data.default_joint_pos[:, head_ids]) - hcmd).abs().sum(0)
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
                head_err_rad={name: round(float(x) / (n * steps), 4) for name, x in
                              zip(("neck_pitch", "head_pitch", "head_yaw", "head_roll"), head_err)},
                reward_per_s=round(rew_sum / (n * steps) / env.unwrapped.step_dt, 3),
                terms_per_s={name: round(float(x) / (n * steps), 4) for name, x in zip(rm.active_terms, term_sum)})


# --------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--root", default="/vol/runs")
    ap.add_argument("--onnx", default="/root/assets/walk61.onnx", help="init policy; for --eval-variants a comma list of policies to roll out")
    ap.add_argument("--manifest", default="/root/assets/asbuilt-mass-manifest.json", help="mass manifest for --cmu-config asbuilt")
    ap.add_argument("--cmu-config", choices=CMU_CONFIGS, default="asbuilt",
                    help=f"robot mass configuration: asbuilt = --manifest; plugged/battery = {MANIFESTS_DIR}/<variant>.json (from cmu/model/manifests/)")
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
    ap.add_argument("--track-guard", type=float, default=0.0, help="stop when the episode velocity-tracking error exceeds this factor x the warm-up (frozen walk61) value on two saves; 0 = off")
    ap.add_argument("--no-asbuilt", action="store_true")
    ap.add_argument("--guard-fraction", type=float, default=0.5, help="stop when mean reward < fraction * peak for two saves")
    ap.add_argument("--guard-after", type=int, default=100, help="iterations after the warm-up before the reward guard may stop the run")
    ap.add_argument("--eval-variants", default="", help="comma list of baseline,asbuilt,delta,delta_asbuilt,staged,staged_asbuilt: evaluate --onnx deterministically in each env and exit")
    ap.add_argument("--eval-steps", type=int, default=750)
    ap.add_argument("--audit-max-ratio", type=float, default=0.8, help="reward audit: a standing policy may collect at most this fraction of the walking policy's reward per second")
    args = ap.parse_args()
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
            if dname == "delta":
                apply_delta(cfg_v, dict(DELTA_DEFAULT))
            elif dname in DELTAS:
                apply_delta(cfg_v, dict(DELTAS[dname]))
            elif variant not in ("baseline", "asbuilt"):
                raise SystemExit(f"unknown eval variant {variant!r}: use baseline, asbuilt, delta, or a DELTAS key, each optionally _asbuilt")
            if "asbuilt" in variant:
                cfg_v.scene.entities["robot"] = apply_asbuilt_masses(cfg_v.scene.entities["robot"], manifest, args.audit, args.remove_battery_kg)
            cfg_v.scene.num_envs = args.num_envs
            cfg_v.seed = args.seed
            env_v = ManagerBasedRlEnv(cfg=cfg_v, device=device)
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
    anchor = (delta or {}).get("anchor")
    if anchor and args.warmup_iters <= 0:
        raise SystemExit("the anchor collects walk61's states during the critic warm-up: --warmup-iters must be > 0")
    if not args.no_asbuilt:
        env_cfg.scene.entities["robot"] = apply_asbuilt_masses(env_cfg.scene.entities["robot"], manifest, args.audit, args.remove_battery_kg)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg.seed = args.seed
    agent_cfg.logger = "tensorboard"
    agent_cfg.save_interval = args.save_interval
    agent_cfg.run_name = args.run_name
    agent_cfg.max_iterations = args.max_iters

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
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
            self.anchor_x, self.anchor_mu, self.anchor_w, self.anchor_parts = None, None, None, []
            self.anchor_dist_sum, self.anchor_n = 0.0, 0

            def clamped_update():
                if anchor and self.warm:
                    self.collect_anchor_states()  # before update(): it clears the rollout storage
                out = orig_update()
                if args.max_lr and self.alg.learning_rate > args.max_lr:
                    self.alg.learning_rate = args.max_lr
                    for g in self.alg.optimizer.param_groups:
                        g["lr"] = args.max_lr
                if anchor and not self.warm and self.anchor_x is not None:
                    out["anchor"] = self.anchor_step()
                return out
            self.alg.update = clamped_update

        def collect_anchor_states(self):
            """Keep vx 0 / vx_band actor inputs of this rollout (default head; head-command rows too when beta_legs > 0)
            with the frozen walk61 actor's targets and per-output weights (anchor_targets)."""
            actor = self.alg.actor
            obs = self.alg.storage.observations
            x = torch.cat([obs[g] for g in actor.obs_groups], dim=-1).flatten(0, 1)
            x, mu0, w = anchor_targets(actor, x, anchor)
            per_iter = math.ceil(anchor["states"] / max(1, args.warmup_iters))
            if len(x) > per_iter:
                sel = torch.randperm(len(x), device=x.device)[:per_iter]
                x, mu0, w = x[sel], mu0[sel], w[sel]
            self.anchor_parts.append((x.clone(), mu0, w))

        def finish_anchor_states(self):
            if not self.anchor_parts:
                raise RuntimeError("anchor: the warm-up kept no state with vx 0 or in the band")
            self.anchor_x = torch.cat([p[0] for p in self.anchor_parts])[: anchor["states"]]
            self.anchor_mu = torch.cat([p[1] for p in self.anchor_parts])[: anchor["states"]]
            self.anchor_w = torch.cat([p[2] for p in self.anchor_parts])[: anchor["states"]]
            self.anchor_parts = []
            torch.save(dict(x=self.anchor_x.cpu(), mu=self.anchor_mu.cpu(), w=self.anchor_w.cpu()), run_dir / "anchor_states.pt")
            standing = int((self.anchor_x[:, ACTOR_TWIST.start] == 0).sum())
            heads = int((~(self.anchor_x[:, ACTOR_HEAD] == 0).all(dim=1)).sum())
            print(f"[anchor] {len(self.anchor_x)} walk61 states kept ({standing} at vx 0, {heads} with a head command), "
                  f"beta {anchor['beta']}, beta_legs {anchor.get('beta_legs', 0.0)}", flush=True)

        def anchor_step(self):
            """One supervised step toward walk61's targets on a minibatch of the warm-up states."""
            idx = torch.randint(len(self.anchor_x), (anchor["batch"],), device=self.anchor_x.device)
            x, mu0, w = self.anchor_x[idx], self.anchor_mu[idx], self.anchor_w[idx]
            loss = anchor_loss_w(self.alg.actor, x, mu0, w)
            self.alg.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.alg.actor.parameters(), self.alg.max_grad_norm)
            self.alg.optimizer.step()
            with torch.no_grad():  # distance on the anchored outputs only (w > 0)
                dist = float(((actor_mean(self.alg.actor, x) - mu0) * (w > 0)).norm(dim=1).mean())
            self.anchor_dist_sum += dist
            self.anchor_n += 1
            return float(loss)

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
            low_bad = bool(args.track_guard) and low == low and lbase == lbase and (low < lbase / args.track_guard or low < args.low_speed_floor)
            track_ok = not (args.track_guard and not self.rebase_left and base == base and track == track and
                            (track > args.track_guard * base or speed < sbase / args.track_guard or low_bad))
            row = dict(iter=it, elapsed_s=round(elapsed, 1), dollars=round(dollars, 3), mean_reward=rew, mean_ep_len=length,
                       it_per_s=round(its, 3) if its == its else None, lr=self.alg.learning_rate, action_std=std, warmup=self.warm,
                       track_err=round(track, 4) if track == track else None, track_base=round(base, 4) if base == base else None,
                       speed_along_cmd=round(speed, 4) if speed == speed else None, speed_base=round(sbase, 4) if sbase == sbase else None,
                       low_cmd_speed=round(low, 4) if low == low else None, low_cmd_base=round(lbase, 4) if lbase == lbase else None,
                       time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       anchor_dist=round(self.anchor_dist_sum / self.anchor_n, 4) if self.anchor_n else None)
            self.anchor_dist_sum, self.anchor_n = 0.0, 0
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
                if anchor:
                    self.finish_anchor_states()
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
                worker["stop"] = (f"tracking guard: error {track:.3f} vs warm-up {base:.3f}, speed along command {speed:.3f} vs {sbase:.3f}, "
                                  f"low-command speed {low:.3f} vs {lbase:.3f} (factor {args.track_guard}, floor {args.low_speed_floor}) on two saves at iter {it}")
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
        if anchor and not runner.warm:
            st = torch.load(run_dir / "anchor_states.pt")
            runner.anchor_x, runner.anchor_mu = st["x"].to(device), st["mu"].to(device)
            runner.anchor_w = st["w"].to(device) if "w" in st else torch.full_like(runner.anchor_mu, anchor["beta"])
            print(f"[anchor] resumed with {len(runner.anchor_x)} stored walk61 states", flush=True)
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
    if anchor:
        # walk61's normalizer stays fixed (rsl_rl would keep updating it), so the anchor targets and the export use it
        runner.alg.actor.obs_normalizer.until = 1
        o = env.get_observations()
        x = torch.cat([o[g] for g in runner.alg.actor.obs_groups], dim=-1)
        cm = env.unwrapped.command_manager
        assert x.shape[1] == 61, x.shape
        assert torch.allclose(x[:, ACTOR_HEAD], cm.get_command("head_pose")), "actor input columns 51-54 are not the head command"
        assert torch.allclose(x[:, ACTOR_TWIST], cm.get_command("twist")[:, :3]), "actor input columns 48-50 are not the twist command"
        print(f"[anchor] beta {anchor['beta']}, beta_legs {anchor.get('beta_legs', 0.0)}, {anchor['states']} states, minibatch {anchor['batch']}, "
              f"vx 0 or {anchor['vx_band']}; actor normalizer frozen; input columns checked against the commands", flush=True)
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
    meta = get_base_metadata(runner.env.unwrapped, run_path=str(final_ckpt))
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
        attach_metadata_to_onnx(str(run_dir / f"{args.run_name}_best.onnx"), get_base_metadata(runner.env.unwrapped, run_path=str(run_dir / "model_best.pt")))
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
