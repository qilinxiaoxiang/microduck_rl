"""DELTA_T1 (cmu/finetune/modal/finetune_worker.py): the walk61 fine-tune in which the body follows the head
(cmu/grasp/PLAN-continuous-vision.md section 5, T1).

CPU only: builds the velocity cfg, applies the delta, checks the command ranges, the collapsed curricula, the reward
terms and the 61-input layout, and scores the two degenerate policies (stand still; ignore the neck) against the
intended one from the cfg's own reward constants (the ft1 lesson: standing collected 80 % of walking's reward).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cmu" / "finetune" / "modal"))
import finetune_worker as fw  # noqa: E402

HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")


@pytest.fixture(scope="module")
def cfg():
    c = load_env_cfg(fw.TASK, play=False)
    fw.apply_delta(c, dict(fw.DELTA_T1))
    return c


def test_t1_is_registered_for_launch_and_audit():
    assert fw.DELTAS["t1"] is fw.DELTA_T1
    sys.path.insert(0, str(REPO / "cmu" / "finetune" / "modal"))
    src = (REPO / "cmu" / "finetune" / "modal" / "finetune.py").read_text()
    assert '"t1": "t1_asbuilt"' in src  # finetune.py::run refuses an unaudited delta


def test_twist_commands_forward_only_with_exact_zero_envs(cfg):
    tw = cfg.commands["twist"]
    d = fw.DELTA_T1["twist_ranges"]
    assert tuple(tw.ranges.lin_vel_x) == tuple(d["lin_vel_x"]) and tw.ranges.lin_vel_x[0] == 0.0
    assert 0.25 < tw.ranges.lin_vel_x[1] <= 0.3  # the robot-tested 0.25 m/s lies inside, not on the edge
    assert tuple(tw.ranges.lin_vel_y) == tuple(d["lin_vel_y"])
    assert tuple(tw.ranges.ang_vel_z) == tuple(d["ang_vel_z"])
    assert tw.rel_standing_envs == fw.DELTA_T1["rel_standing_envs"] > 0  # exact-zero commands: the deployment idle
    stages = cfg.curriculum["standing_envs"].params["standing_stages"]
    assert stages[-1]["rel_standing_envs"] == fw.DELTA_T1["rel_standing_envs"]  # the collapsed curriculum agrees
    assert tw.rel_turn_in_place_envs > 0


def test_head_commands_cover_the_looking_down_range(cfg):
    hp = cfg.commands["head_pose"]
    ranges = tuple(tuple(r) for r in fw.DELTA_T1["head_ranges"])
    assert tuple(tuple(r) for r in hp.ranges) == ranges
    assert tuple(tuple(r) for r in cfg.curriculum["head_pose_range"].params["range_stages"][-1]["ranges"]) == ranges
    neck, head, yaw, roll = ranges
    assert neck[0] <= -1.0 and neck[1] >= 0.3        # neck down to the trained cap, a little up
    assert head[1] >= 1.0 and head[0] <= -0.2         # head pitch: the gaze loop's 0.35..1.35 absolute = 0..1.0 delta
    assert yaw[1] >= 0.8 and yaw[0] == -yaw[1]
    assert abs(roll[1]) <= 0.1                        # roll stays small: not a tracked quantity
    assert tuple(hp.resampling_time_range) == tuple(fw.DELTA_T1["head_resample_s"])
    for j, (lo, hi) in zip(HEAD_JOINTS, ranges):     # inside the mechanical caps walk61 was trained with
        cap = {"neck_pitch": 1.10, "head_pitch": 1.10, "head_yaw": 1.40, "head_roll": 0.31}[j]
        assert -cap <= lo < hi <= cap, j


def test_reward_terms(cfg):
    r = cfg.rewards
    assert r["track_linear_velocity"].func is fw.track_linear_velocity_ema and r["track_linear_velocity"].weight == 2.0
    assert r["track_linear_velocity"].params["tau_s"] == 0.5 and abs(r["track_linear_velocity"].params["std"] ** 2 - 0.05) < 1e-9
    hb = fw.DELTA_T1["head_bias"]
    assert r["head_pose_bias"].weight == hb["weight"] and r["head_pose_bias"].params["tau_s"] == hb["tau_s"]
    assert cfg.curriculum["head_pose_bias_weight"].params["weight_stages"][-1]["weight"] == hb["weight"]
    assert r["head_pose_tracking"].weight == 2.0 and r["head_pose_tracking"].params["std"] == 0.5  # Pollen's, unchanged
    assert r["air_time"].weight == 3.0                                                              # stepping term unchanged
    # action_rate_l2 ramps -0.1 -> -1.0 by curriculum; a fine-tune runs at the collapsed final stage, Pollen's -1.0
    assert cfg.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] == -1.0


def test_obs_layout_keeps_the_13d_command_block(cfg):
    for group in ("actor", "critic"):
        names = list(cfg.observations[group].terms)
        assert names[-2:] == ["head_command", "body_command"], names
    assert "head_command" in cfg.observations["actor"].terms


def _per_second(cfg, neck_err, other_err):
    """Head reward per simulated second for a policy with DC error |neck_err| on the neck and |other_err| on the
    other three head joints: head_pose_tracking (Gaussian, mean over joints) plus head_pose_bias (L1 on the EMA)."""
    std = cfg.rewards["head_pose_tracking"].params["std"]
    w_t, w_b = cfg.rewards["head_pose_tracking"].weight, cfg.rewards["head_pose_bias"].weight
    errs = np.array([neck_err, other_err, other_err, other_err])
    return w_t * np.mean(np.exp(-(errs / std) ** 2)) - w_b * np.mean(np.abs(errs))


def test_degenerate_audit_ignoring_the_neck_loses(cfg):
    """walk61 holds the neck near HOME whatever is asked (as-built sim 2026-10-30: -0.75 commanded -> -0.125 reached
    standing; the robot: -0.09 absolute for -0.4..-0.8 asked). Over the T1 neck range that policy must score clearly
    under one that tracks, and the margin must be of the order of the stepping reward (air_time, about 1/s)."""
    lo, hi = fw.DELTA_T1["head_ranges"][0]
    cmds = np.linspace(lo, hi, 2001)
    other = 0.08  # walk61's DC error on the joints it does track (audit-1016-1940: head_pose_bias -0.226/s at weight 3)
    tracking = _per_second(cfg, other, other)
    ignoring = float(np.mean([_per_second(cfg, abs(c), other) for c in cmds]))
    margin = tracking - ignoring
    assert margin >= 0.6, (tracking, ignoring, margin)
    # a 0.1 rad DC neck error (the sim done criterion) costs little: the term prices the robot's 0.3-0.7 rad, not jitter
    assert tracking - _per_second(cfg, 0.1, other) < 0.1


def test_degenerate_audit_standing_still_loses_at_low_commands(cfg):
    """track_linear_velocity_ema at the T1 std: standing under a 0.15..0.30 m/s command against walking at it."""
    std = cfg.rewards["track_linear_velocity"].params["std"]
    w = cfg.rewards["track_linear_velocity"].weight
    for v in (0.15, 0.20, 0.25, 0.30):
        standing = w * math.exp(-(v ** 2) / std ** 2)
        assert w - standing >= 0.6, (v, standing)   # plus air_time about 1/s for stepping (audit-1016-1940)
    # below 0.1 m/s the term cannot separate creeping from standing: the plan's lowest trained speed is 0.10
    assert w - w * math.exp(-(0.05 ** 2) / std ** 2) < 0.15
