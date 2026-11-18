"""Invariants of the walk-nudge task (tasks/cmu_walk_nudge_env_cfg.py). Cfg-only, no env."""
import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.tasks import cmu_walk_nudge_env_cfg as wn

BODIES = ("Plugged", "Battery-20261104")


def test_tasks_registered_and_walk61_present():
    for b in BODIES:
        t = f"Mjlab-WalkNudge-FromWalk61-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "walk_nudge"
    assert wn.INIT_ONNX.exists()


def test_61d_command_block_kept():
    cfg = wn.make_walk_nudge_env_cfg("plugged")
    assert list(cfg.observations["actor"].terms)[-3:] == ["command", "head_command", "body_command"]
    assert isinstance(cfg.commands["twist"], wn.NudgeCommandCfg)
    assert set(cfg.commands) == {"twist", "head_pose", "body_pose"}


def test_command_stays_inside_walk61_ranges_and_steps_above_dead_zone():
    assert wn.CMD_MAX == (0.4, 0.3, 1.0)
    # a 3 cm error reads above walk61's ~0.2 m/s dead zone at iteration 0
    assert wn.NUDGE_GAIN * 0.03 > 0.2
    assert wn.ZERO_PROB > 0   # zero-command (hold still) is trained explicitly


def test_velocity_tracking_replaced_and_signs():
    cfg = wn.make_walk_nudge_env_cfg("plugged")
    r = cfg.rewards
    assert "track_linear_velocity" not in r and "track_angular_velocity" not in r
    for k in ("nudge_position", "nudge_progress", "nudge_yaw", "air_time", "upright"):
        assert r[k].weight > 0, k
    for k in ("action_rate_l2", "foot_slip", "foot_clearance", "dof_pos_limits"):
        assert r[k].weight < 0, k
    assert r["air_time"].params["command_threshold"] == wn.NUDGE_GAIN * wn.STEP_TOL
    assert r["foot_slip"].params["command_threshold"] == 0.0


def test_inherited_curricula_collapsed():
    cfg = wn.make_walk_nudge_env_cfg("plugged")
    assert "standing_envs" not in cfg.curriculum
    for term in cfg.curriculum.values():
        for v in term.params.values():
            if isinstance(v, list) and v and all(isinstance(s, dict) and "step" in s for s in v):
                assert len(v) == 1 and v[0]["step"] == 0


def test_camera_variant_registered_and_shaped_like_the_detector():
    for b in BODIES:
        t = f"Mjlab-WalkNudge-Cam-FromWalk61-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "walk_nudge_cam"
    cfg = wn.make_walk_nudge_env_cfg("plugged", camera=True)
    assert cfg.commands["twist"].camera and not wn.make_walk_nudge_env_cfg("plugged").commands["twist"].camera
    from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as pix
    assert wn.CAM_PERIOD_STEPS == pix.PIX_PERIOD_STEPS
    dt = 0.02
    assert (wn.CAM_DELAY_STEPS[0] * dt, wn.CAM_DELAY_STEPS[1] * dt) == pix.PIX_DELAY_S
    assert wn.CAM_DROP[1] >= pix.PIX_DROP[1]
