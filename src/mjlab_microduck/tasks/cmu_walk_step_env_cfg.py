"""Walk step: a capped correction on frozen walk61 that makes a short burst a short, repeatable step.

For step-look-step: burst, stop, look again with the camera, repeat. The camera closes the loop BETWEEN steps, so
unlike the nudge (cmu_walk_nudge_env_cfg.py, fell on the robot 2026-11-21 while chasing its leg odometry) the policy
gets no target and no odometry. It reads what walk61 reads in teleop: the twist of a for_s burst after walk_wired's
command low-pass, then zero. So the baseline arm (plain walk61) and this one are driven by the same runtime.

Why a capped correction and not a fine-tune: on 645 logged real walk61 states the fine-tunes' action distance from
walk61 was t2b 0.22 rad (walked like walk61 on the robot), ft3 0.43 (lost balance 3 of 3), T1 0.71 (fell 2 of 5), nudge
1.05 (fell). residual_model.py bounds the distance per joint by construction.

Round 2 (2026-11-22 afternoon): the current body became battery-v2m5-20261121 (1034.8 g, CoM 11.2 mm behind the feet
against 7.6). On it stand61 and walk61 creep about 20 cm backward at zero command and fall, so the policy fills BOTH
slots (it also learns to just stand), and the control loop is timed as the robot's brain. With the recipe's 15-30 ms
dead time round 1 trained in an easier loop than it ran in; with RUNTIME_* the training env reproduces the brain sim on
this body (walk61 at rest: -24 cm and 20 % tipped against -23 cm and 22 %).

Built on the CMU velocity recipe on the CURRENT body (cmu/model/current.json; the nudge, dock and kneel were trained
on the retired `plugged` body), with:
- twist = StepBurstCommand: vx 0.25 pulses of 0.2-1.0 s, small arcs, rests of 1.5-4 s (exact zero, the idle state)
- velocity tracking replaced by a reference that moves REF_SPEED x T and then holds (position, heading, stillness)
- head near neutral through a burst and its stop; in the rest it looks down as the runtime does between steps
  (capped at 1.1 rad absolute: 1.3 tipped the stand on 2026-11-21)
- servo noise level 1, and BAM friction widened towards the 1.4 the 2026-10-30 jitter study measured (walk61 itself
  falls too often in training at 1.5, FINETUNE-DELTA.md t3)
- curricula collapsed to their final stage (the base was trained there)
"""
from __future__ import annotations

import copy
import dataclasses

from mjlab.envs import mdp as envs_mdp
from mjlab.managers import RewardTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.cmu_asbuilt import CURRENT, make_cmu_velocity_env_cfg
from mjlab_microduck.tasks.cmu_ball_dock_env_cfg import WALK_ONNX
from mjlab_microduck.tasks.microduck_velstand_env_cfg import _collapse_curricula_to_final
from mjlab_microduck.tasks.residual_model import ResidualModelCfg
from mjlab_microduck.tasks.servo_noise import apply_servo_noise, apply_servo_noise_range

BURST_VX = (0.24, 0.26)          # walk61 steps at 0.25 and not below
LONG_VX = (0.18, 0.32)           # teleop walks: the deploy gate walks 0.2, 0.25 and 0.3 m/s
BURST_VY = (-0.01, 0.01)         # only keeps the input alive
BURST_WZ = (-0.3, 0.3)           # trim + heading hold, or an arc (approach.VTH_ARC)
STRAIGHT_PROB = 0.3
PULSE_S = (0.2, 1.0)
REST_S = (1.5, 4.0)
FIRST_REST_S = (0.8, 2.0)
LAND_S = 0.5
REF_SPEED = 0.10                 # m per second of pulse: 2-10 cm (real walk61 covers about 10 cm in a 0.7-1.1 s burst)
CMD_ALPHA = 0.2                  # walk_wired.CMD_ALPHA

POS_STD = 0.025                  # m
POS_WEIGHT = 6.0
POS_L1_WEIGHT = -6.0            # per m: the gradient where the Gaussian has none
POS_L1_MAX = 0.15                # m
HEADING_STD = 0.2                # rad
HEADING_WEIGHT = 1.0
SETTLE_S = 1.0
SETTLE_SPEED_WEIGHT = -8.0       # per m/s
PROGRESS_WEIGHT = 8.0            # only while a burst runs: what separates stepping from standing (audit 2)
SETTLE_SPEED_MAX = 0.2

# Reward audit 2026-11-22 before launch: with the velocity recipe's upright 2 and head 2 + 3 (and position 3) stand61
# earned 0.90 of walk61's reward (limit 0.8: ft1 / ft2 drifted to standing at that ratio). Those terms pay standing in
# full, so they are halved and the position term carries the task. The recipe's +-0.3 m/s pushes displace a robot
# that has no position input by tens of cm (inescapable error), so they are halved too.
UPRIGHT_WEIGHT = 1.0
HEAD_TRACK_WEIGHT = 0.5
HEAD_BIAS_WEIGHT = 1.0
PUSH_V = 0.15                    # m/s

HEAD_RANGES = ((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015))
LOOK_RANGES = ((-0.05, 0.05), (0.0, 0.75), (-0.5, 0.5), (-0.015, 0.015))   # deltas: head_pitch 0.35..1.10 absolute
LOOK_PROB = 0.7
LOOK_AFTER_S = 1.0               # of standing after a burst
LOOK_LEAD_S = 0.6                # head back at neutral this long before the next burst
HEAD_ALPHA = 0.2                 # walk_brain.HEAD_ALPHA
JOINT_FRICTION = (0.9, 1.2)
SERVO_NOISE_LEVEL = 1

DROP_RANGE = (0.0, 0.08)         # dropped servo reads per tick (robot 3.6 %, gate stress 8 %)
IDLE_PROB = 0.2                  # episodes that only stand (stand61's slot)
LONG_PROB = 0.15                 # pulses that are teleop walks (the deploy gate walks 10 s)
LONG_PULSE_S = (1.5, 5.0)

# The loop as the brain runs it (cmu/finetune/flight/flight_worker.py LOOP_DELAY): readings one 20 ms tick old, goals
# 20 ms late. Command lag in 5 ms physics steps, one draw held per RUNTIME_ACT_PERIOD steps.
RUNTIME_ACT_LAG = (3, 5)        # 35-45 ms in all (4-8 tripled walk61 falls on the OLD body)
RUNTIME_ACT_PERIOD = 400
RUNTIME_OBS_LAG = {"joint_pos": 1, "joint_vel": 2, "base_ang_vel": 1, "projected_gravity": 1}

# CoM: the recipe's +-25 mm (startup) and +-15 mm (reset) on the trunk move the whole robot further than the new pack
# did. These give about +-6 mm whole-robot, around the estimated (unweighed) pack.
BASE_COM = {0: (-0.008, 0.008), 1: (-0.005, 0.005), 2: (-0.008, 0.008)}
RESET_COM = 0.005
HEAD_COM = 0.005
FOOT_FRICTION = (0.6, 1.3)       # the bare printed foot is an estimated 0.3, the sole 0.7

# IMU misalignment: the recipe's 6 deg is more than the whole pitch-trim range that decides whether a back-heavy duck
# stands. The robot's sensor is trimmed and reads right within about 1 deg (walk_brain IMU_ROLL_TRIM note).
IMU_MISALIGN_DEG = 2.0
# With walk61's reward near zero on this body ending the episode would pay; a fall costs about 2 s of a good policy.
FALL_WEIGHT = -500.0             # x dt = -10 per fall

# Jitter decided the deploy gate in round 1 (cap 0.10 failed on action jumps)
ACTION_ACC_WEIGHT = -0.3
STEP_OVER_LIMIT = 0.3            # rad per tick
STEP_OVER_WEIGHT = -10.0

CAPS = (0.05, 0.10, 0.15, 0.20, 0.30)              # rad per joint
# round 4 (2026-11-22 night): servo noise for cautious steps, as task variants (cap and L1 come from the CLI)
NOISE_VARIANTS = {"SN3": (3, 3), "SNR13": (1, 3)}
# round 6: main's robot conditions (robot_conditions.py), alone and with servo position read noise drawn in [1, 3]
RC_VARIANTS = {"RC": None, "RCSNR13": (1, 3)}
L1 = 0.1
INIT_STD = 0.05
LEARNING_RATE = 3e-4
MAX_ITERATIONS = 1000


def make_walk_step_env_cfg(variant: str = CURRENT, play: bool = False, noise=(SERVO_NOISE_LEVEL, SERVO_NOISE_LEVEL),
                           robot_conditions: bool = False):
    """The step task on `variant`, timed as the robot's brain. noise (lo, hi): servo noise level, fixed when lo == hi,
    else drawn per env at every reset (servo_noise.apply_servo_noise_range).
    robot_conditions=False: rounds 1-5 (cmu/step/RESULTS.md) trained without main's default -CMU-Current robot
    conditions (robot_conditions.py, 2026-11-22); this task models dropped reads itself (DROP_RANGE) and servo read
    noise through servo_noise.py. Adopting them is a new round, not a silent change to these tasks.
    robot_conditions=True (round 6, the -RC- tasks): main's conditions as they are (leg-speed gyro noise, joint_vel
    Gaussian 0.5, gravity noise, joint_pos one tick old, 4 % dropped reads, BAM friction x1.0-1.5); this task's own
    drop range, joint friction and joint_vel noise are then not applied, and `noise` (None for none) adds servo position
    read noise only."""
    cfg = make_cmu_velocity_env_cfg(variant, play=play, robot_conditions=robot_conditions)
    _collapse_curricula_to_final(cfg)
    cfg.curriculum.pop("standing_envs", None)      # edits the velocity command's standing share
    cfg.curriculum.pop("head_pose_range", None)    # would widen the head to +-1.1
    cfg.commands["twist"] = microduck_mdp.StepBurstCommandCfg(
        vx=BURST_VX, vy=BURST_VY, wz=BURST_WZ, straight_prob=STRAIGHT_PROB, pulse_s=PULSE_S, rest_s=REST_S,
        first_rest_s=FIRST_REST_S, land_s=LAND_S, ref_speed=REF_SPEED, cmd_alpha=CMD_ALPHA,
        idle_prob=IDLE_PROB, long_prob=LONG_PROB, long_pulse_s=LONG_PULSE_S, long_vx=LONG_VX,
    )
    if not robot_conditions:   # robot_conditions.py brings its own 4 % drop action
        old = cfg.actions["joint_pos"]
        cfg.actions["joint_pos"] = microduck_mdp.DroppedReadJointPositionActionCfg(
            **{f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}, drop_range=DROP_RANGE)
    cfg.commands["head_pose"] = microduck_mdp.StepLookHeadCommandCfg(
        ranges=HEAD_RANGES, zero_command_prob=0.5, look_ranges=LOOK_RANGES, look_prob=LOOK_PROB,
        look_after_s=LOOK_AFTER_S, lead_s=LOOK_LEAD_S, alpha=HEAD_ALPHA,
    )
    cfg.commands["body_pose"].zero_command_prob = 0.5

    r, m = cfg.rewards, microduck_mdp
    r.pop("track_linear_velocity")
    r.pop("track_angular_velocity")
    r["step_position"] = RewardTermCfg(func=m.step_ref_position, weight=POS_WEIGHT,
                                       params={"command_name": "twist", "std": POS_STD})
    r["step_position_l1"] = RewardTermCfg(func=m.step_ref_position_l1, weight=POS_L1_WEIGHT,
                                          params={"command_name": "twist", "max_err": POS_L1_MAX})
    r["step_heading"] = RewardTermCfg(func=m.step_ref_heading, weight=HEADING_WEIGHT,
                                      params={"command_name": "twist", "std": HEADING_STD})
    r["step_settled_speed"] = RewardTermCfg(func=m.step_settled_speed, weight=SETTLE_SPEED_WEIGHT,
                                            params={"command_name": "twist", "settle_s": SETTLE_S,
                                                    "max_speed": SETTLE_SPEED_MAX})
    r["upright"].weight = UPRIGHT_WEIGHT
    r["head_pose_tracking"].weight = HEAD_TRACK_WEIGHT
    r["head_pose_bias"].weight = HEAD_BIAS_WEIGHT
    cfg.curriculum.pop("head_pose_bias_weight", None)    # would write the recipe's weight back
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-PUSH_V, PUSH_V), "y": (-PUSH_V, PUSH_V)}
    r["step_progress"] = RewardTermCfg(func=m.step_burst_progress, weight=PROGRESS_WEIGHT,
                                       params={"command_name": "twist"})
    r["action_acc_l2"] = RewardTermCfg(func=m.action_acc_l2, weight=ACTION_ACC_WEIGHT)
    r["leg_step_over"] = RewardTermCfg(func=m.leg_action_step_over_l2, weight=STEP_OVER_WEIGHT,
                                       params={"limit": STEP_OVER_LIMIT})
    r["fall"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=FALL_WEIGHT)
    _runtime_loop(cfg)
    _narrow_randomisation(cfg)
    if robot_conditions:
        if noise is not None:
            apply_servo_noise_range(cfg, noise[0], int(noise[1]), vel=False)
        return cfg
    if noise[0] == noise[1]:
        apply_servo_noise(cfg, int(noise[1]))
    else:
        apply_servo_noise_range(cfg, noise[0], int(noise[1]))
    if "randomize_joint_friction" in cfg.events:
        cfg.events["randomize_joint_friction"].params["scale_range"] = JOINT_FRICTION
    return cfg


def _runtime_loop(cfg) -> None:
    robot = copy.deepcopy(cfg.scene.entities["robot"])
    for act in robot.articulation.actuators:
        act.delay_min_lag, act.delay_max_lag = RUNTIME_ACT_LAG
        act.delay_update_period = RUNTIME_ACT_PERIOD
    cfg.scene.entities["robot"] = robot
    terms = cfg.observations["actor"].terms
    for name, lag in RUNTIME_OBS_LAG.items():
        terms[name] = copy.deepcopy(terms[name])
        terms[name].delay_min_lag = terms[name].delay_max_lag = lag


def _narrow_randomisation(cfg) -> None:
    ev = cfg.events
    if "base_com" in ev:
        ev["base_com"].params["ranges"] = dict(BASE_COM)
    for event, curriculum, span in (("randomize_com", "com_range", RESET_COM),
                                    ("randomize_head_com", "head_com_range", HEAD_COM)):
        cfg.curriculum.pop(curriculum, None)     # would write the recipe's range back
        if event in ev:
            ev[event].params["ranges"] = (-span, span)
    for name in ("base_ang_vel", "projected_gravity"):
        term = cfg.observations["actor"].terms[name]
        if "max_angle_deg" in term.params:
            term.params = {**term.params, "max_angle_deg": IMU_MISALIGN_DEG}
    if "foot_friction" in ev:
        ev["foot_friction"].params["ranges"] = FOOT_FRICTION


def walk_step_rl_cfg(base, cap: float):
    rl = copy.deepcopy(base)
    rl.actor = ResidualModelCfg(
        hidden_dims=(256, 128, 64), activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": INIT_STD, "std_type": "log"},
        base_onnx=str(WALK_ONNX), cap=cap, l1=L1,
    )
    a = rl.algorithm
    # adaptive would run the lr to its ceiling during the critic warm-up (frozen actor = zero KL)
    a.schedule, a.learning_rate, a.entropy_coef = "fixed", LEARNING_RATE, 0.0
    a.symmetry_cfg = None
    rl.experiment_name = f"walk_step_cap{round(cap * 100):03d}"
    rl.run_name = rl.experiment_name
    rl.max_iterations = MAX_ITERATIONS
    return rl
