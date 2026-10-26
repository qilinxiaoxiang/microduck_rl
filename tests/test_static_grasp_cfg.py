"""Static grasp: the ground-grasp env pulled into the static pose (reward signs, pose in range, registration)."""

import numpy as np

import mjlab_microduck.tasks  # noqa: F401  (registry first)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot.ground_grasp import ground_grasp_joint_limits
from mjlab_microduck.tasks import mdp as M
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import (
    OBJECT_NOMINAL_XY,
    STATIC_JAW_RAD,
    STATIC_POSE,
    make_microduck_static_grasp_env_cfg,
)


def test_registered():
    cfg = load_env_cfg("Mjlab-StaticGrasp-FromPick-MicroDuck")
    assert "static_pose" in cfg.rewards


def test_reward_signs_and_terms():
    r = make_microduck_static_grasp_env_cfg().rewards
    assert "beak_bisector" not in r
    # positive costs take negative weights, self-negating penalties positive ones (AGENTS.md sign rule)
    assert r["beak_floor_touch"].func is M.beak_floor_touch and r["beak_floor_touch"].weight < 0
    for name in ("head_floor_impact", "jaw_floor_impact"):
        assert r[name].weight < 0 and r[name].params["threshold"] == 0.0
    assert r["hold_still"].func is M.hold_stillness and r["hold_still"].weight > 0
    for name in ("static_pose", "lip_height", "com_support", "grasp_held"):
        assert r[name].weight > 0


def test_mouth_and_object_follow_the_static_pose():
    cfg = make_microduck_static_grasp_env_cfg()
    assert cfg.actions["mouth"].open_rad == STATIC_JAW_RAD
    assert tuple(cfg.events["reset_object"].params["nominal_xy"]) == OBJECT_NOMINAL_XY


def test_static_pose_inside_the_joint_ranges():
    lim = np.array(list(ground_grasp_joint_limits().values()))
    q = np.array(STATIC_POSE)
    assert lim.shape == (14, 2)
    assert np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1])


def test_strict_variant_prices_beak_support_harder():
    base = make_microduck_static_grasp_env_cfg().rewards
    strict = load_env_cfg("Mjlab-StaticGrasp-Strict-FromPick-MicroDuck").rewards
    assert strict["beak_floor_touch"].weight < base["beak_floor_touch"].weight < 0
    assert strict["head_floor_impact"].weight < base["head_floor_impact"].weight < 0
    assert strict["com_support"].params["outside_floor"] < -1.0 and strict["com_support"].weight > 0
    assert strict["lip_height_wide"].weight > 0 and strict["lip_height_wide"].params["std"] > base["lip_height"].params["std"]


def test_retimed_variant_holds_before_the_bite():
    cfg = load_env_cfg("Mjlab-StaticGrasp-Retimed-FromPick-MicroDuck")
    m, r = cfg.actions["mouth"], cfg.rewards
    at = r["approach_track"].params
    assert at["down_at"] <= r["hold_still"].params["phase_start"] < r["hold_still"].params["phase_end"] <= m.close_start
    assert m.close_start < m.close_end <= r["grasp_held"].params["phase_start"]
    assert r["return_upright"].params["rise_end"] <= 0.70          # robotd hands back to the stand policy at 0.70
    assert r["lip_height"].params["phase_end"] == m.close_start


def test_hover_variant_levels_the_head_and_ends_on_beak_contact():
    cfg = load_env_cfg("Mjlab-StaticGrasp-Hover-FromPick-MicroDuck")
    assert cfg.rewards["head_level"].func is M.head_level_penalty and cfg.rewards["head_level"].weight > 0
    t = cfg.terminations["beak_on_floor"]
    assert t.func is M.beak_floor_contact_in_window and not t.time_out
    assert t.params["phase_end"] <= cfg.actions["mouth"].close_start     # the bite itself is not terminated
