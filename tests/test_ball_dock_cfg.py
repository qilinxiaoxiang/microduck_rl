"""Invariants of the ball-dock commander task (tasks/cmu_ball_dock_env_cfg.py). Cfg-only, no env."""
import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import OBJECT_NOMINAL_XY

BODIES = ("Plugged", "Battery-20261104")


def test_tasks_registered_and_walk61_present():
    for b in BODIES:
        t = f"Mjlab-BallDock-Walk61-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "ball_dock"
    assert bd.WALK_ONNX.exists()


def test_commands_inside_walk61_training_ranges():
    # microduck_velocity_env_cfg: lin x +-0.4, lin y +-0.3, ang z +-1.0, head deltas +-1.10 (yaw +-1.40)
    assert bd.VX_MAX <= 0.4 and bd.VY_MAX <= 0.3 and bd.WZ_MAX <= 1.0
    assert 0.0 <= bd.HEAD_PITCH_RANGE[0] < bd.HEAD_PITCH_RANGE[1] <= 1.10
    assert bd.HEAD_YAW_MAX <= 1.40
    assert -1.10 <= bd.NECK_RANGE[0] < bd.NECK_RANGE[1] <= 0.0


def test_walk_group_is_walk61s_61d_layout():
    cfg = bd.make_ball_dock_env_cfg("plugged")
    walk = cfg.observations["walk"].terms
    assert list(walk)[-4:] == ["actions", "command", "head_command", "body_command"]
    assert walk["actions"].func is bd.walk_last_action        # walk61's own last action, not the commander's
    assert set(cfg.commands) == {"twist", "head_pose", "body_pose"}
    dims = {"twist": 3, "head_pose": 4, "body_pose": 6}
    for k, d in dims.items():
        assert cfg.commands[k].dim == d


def test_actor_sees_only_what_the_robot_has():
    cfg = bd.make_ball_dock_env_cfg("plugged")
    actor = cfg.observations["actor"].terms
    assert set(actor) == {"camera", "head_joints", "gravity", "gyro", "last_action"}
    assert "ball_state" in cfg.observations["critic"].terms


def test_reward_signs():
    cfg = bd.make_ball_dock_env_cfg("plugged")
    r = cfg.rewards
    assert r["dock_progress"].weight > 0 and r["docked"].weight > 0 and r["ball_in_view"].weight > 0
    for k in ("ball_kick", "action_rate", "fell"):   # >= 0 costs (or the 0/1 termination flag): negative weight
        assert r[k].weight < 0, k


def test_target_is_the_kneel_rise_landing_spot():
    assert OBJECT_NOMINAL_XY == (0.087, 0.0)
    assert bd.R_RANGE[0] > OBJECT_NOMINAL_XY[0]   # the robot always has to move to dock
    assert bd.DOCKED_TOL <= 0.02


def test_nudge_low_level_variant():
    for b in BODIES:
        t = f"Mjlab-BallDock-Nudge-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "ball_dock_nudge"
    assert bd.NUDGE_LOW_ONNX.exists()
    cfg = bd.make_ball_dock_env_cfg("plugged", walk_onnx=bd.NUDGE_LOW_ONNX)
    assert cfg.actions["dock"].walk_onnx == str(bd.NUDGE_LOW_ONNX)


def test_forward_dock_moves_only_the_target():
    import torch
    from mjlab.tasks.registry import list_tasks
    from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd
    assert "Mjlab-BallDock-Nudge-Fwd-MicroDuck-CMU-Plugged" in list_tasks()
    cfg = bd.make_ball_dock_env_cfg("plugged", walk_onnx=bd.NUDGE_LOW_ONNX, target_dx=bd.DOCK_TARGET_DX)
    assert cfg.rewards["docked"].params["target_dx"] == bd.DOCK_TARGET_DX > 0
    assert cfg.rewards["dock_progress"].params["target_dx"] == bd.DOCK_TARGET_DX
    assert "dock_range" not in cfg.curriculum and cfg.events["reset_ball_dock"].params["r_max"] == bd.R_RANGE[1]
    # the plain dock is untouched
    plain = bd.make_ball_dock_env_cfg("plugged", walk_onnx=bd.NUDGE_LOW_ONNX)
    assert "target_dx" not in plain.rewards["docked"].params and "dock_range" in plain.curriculum

    class _E:   # target_error shifts dx only
        pass
    e = _E()
    orig = bd.dock_error
    bd.dock_error = lambda env: torch.tensor([[0.0075, 0.002]])
    try:
        assert torch.allclose(bd.target_error(e, 0.0075), torch.tensor([[0.0, 0.002]]))
    finally:
        bd.dock_error = orig


def test_wide_retry_dock_draws_half_from_the_miss_region():
    from mjlab.tasks.registry import list_tasks, load_env_cfg
    from mjlab_microduck.tasks import cmu_ball_dock_env_cfg as bd
    t = "Mjlab-BallDock-Nudge-Fwd-Wide-SN1-MicroDuck-CMU-Plugged"
    assert t in list_tasks()
    cfg = load_env_cfg(t, play=False)
    p = cfg.events["reset_ball_dock"].params
    assert p["wide_frac"] == bd.WIDE_FRAC and p["wide_r"][0] < bd.R_RANGE[0] and p["wide_r"][1] > bd.R_RANGE[1]
    assert p["wide_r"][1] < cfg.terminations["ball_lost"].params["max_dist"]
    assert cfg.rewards["docked"].params["target_dx"] == bd.DOCK_TARGET_DX
    assert cfg.actions["dock"].walk_onnx.endswith("wn-sn1.onnx")
