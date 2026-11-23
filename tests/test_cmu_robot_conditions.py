"""The -CMU-Current velocity tasks train under the real robot's measured sensor and actuator conditions by default
(src/mjlab_microduck/robot_conditions.py, docs/SIM-TO-REAL-GAPS.md); the dated tasks stay as they were; the fine-tune
delta batreal uses the same values. CPU only. The last test builds a small env and steps it.

  uv run --with pytest pytest tests/test_cmu_robot_conditions.py
"""
import math
import sys
from pathlib import Path

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.robot_conditions import ROBOT_CONDITIONS
from mjlab_microduck.tasks.cmu_asbuilt import make_cmu_velocity_env_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT_IDS = ("Mjlab-Velocity-Flat-MicroDuck-CMU-Current", "Mjlab-Velocity-Rough-MicroDuck-CMU-Current")
DATED_IDS = ("Mjlab-Velocity-Flat-MicroDuck-CMU-Battery-v2m5-20261121", "Mjlab-Velocity-Flat-MicroDuck-CMU-Battery-20261104")

# measured on the robot 2026-11-22 (GAP-AUDIT.md); friction is a sim fit, not a measurement
EXPECTED = dict(gyro_noise_legspeed=0.3, joint_friction=(1.0, 1.5), joint_vel_noise_sd=0.5, servo_drop=0.04,
                gravity_noise=0.02, joint_pos_lag=1)


def _conditions(cfg):
    actor = cfg.observations["actor"].terms
    av, jv, pg, jp = actor["base_ang_vel"], actor["joint_vel"], actor["projected_gravity"], actor["joint_pos"]
    return dict(
        gyro_k=av.params.get("k"),
        gyro_floor=(av.noise.n_min, av.noise.n_max),
        joint_friction=tuple(cfg.events["randomize_joint_friction"].params["scale_range"]),
        joint_vel_sd=getattr(jv.noise, "std", None),
        gravity=(pg.noise.n_min, pg.noise.n_max),
        joint_pos_lag=(jp.delay_min_lag, jp.delay_max_lag),
        drop=getattr(cfg.actions["joint_pos"], "drop", None),
    )


def test_values_are_the_measured_ones():
    assert ROBOT_CONDITIONS == EXPECTED


@pytest.mark.parametrize("task_id", CURRENT_IDS)
@pytest.mark.parametrize("play", (False, True))
def test_current_tasks_carry_the_robot_conditions(task_id, play):
    c = _conditions(load_env_cfg(task_id, play=play))
    assert c["gyro_k"] == 0.3
    assert c["gyro_floor"] == (-0.03, 0.03)  # Pollen's floor stays under the leg-speed noise
    assert c["joint_friction"] == (1.0, 1.5)
    assert c["joint_vel_sd"] == 0.5
    assert c["gravity"] == (-0.02, 0.02)
    assert c["joint_pos_lag"] == (1, 1)
    assert c["drop"] == 0.04


@pytest.mark.parametrize("task_id", DATED_IDS)
def test_dated_tasks_are_unchanged(task_id):
    c = _conditions(load_env_cfg(task_id))
    assert c["gyro_k"] is None
    assert c["joint_friction"] == (0.9, 1.1)
    assert c["joint_vel_sd"] is None
    assert c["gravity"] == (-0.01, 0.01)
    assert c["joint_pos_lag"] == (0, 0)
    assert c["drop"] is None


def test_opt_out_and_no_leak_between_cfgs():
    off = _conditions(make_cmu_velocity_env_cfg("current", robot_conditions=False))
    assert off["gyro_k"] is None and off["drop"] is None and off["joint_friction"] == (0.9, 1.1)
    on = make_cmu_velocity_env_cfg("current")
    again = _conditions(make_cmu_velocity_env_cfg("battery-v2m5-20261121"))
    assert again["gyro_k"] is None and again["gravity"] == (-0.01, 0.01)
    assert _conditions(on)["drop"] == 0.04
    # obs layout unchanged (61-D contract): same terms, same order
    assert list(on.observations["actor"].terms) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command", "body_command",
    ]


def test_finetune_delta_batreal_reuses_the_defaults():
    sys.path.insert(0, str(REPO_ROOT / "cmu" / "finetune" / "modal"))
    import finetune_worker as fw
    batreal = fw.DELTAS["batreal"]
    assert {k: batreal[k] for k in ROBOT_CONDITIONS} == ROBOT_CONDITIONS
    assert set(batreal) - set(ROBOT_CONDITIONS) == set(fw.DELTAS["batshift"])


def test_current_env_builds_and_steps_on_cpu():
    from mjlab.envs import ManagerBasedRlEnv
    cfg = load_env_cfg(CURRENT_IDS[0])
    cfg.scene.num_envs = 16
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (16, 61)
        term = env.action_manager.get_term("joint_pos")
        torch.manual_seed(0)
        for _ in range(60):
            a = (torch.rand(16, term.action_dim) - 0.5) * 0.4
            obs, rew, *_ = env.step(a)
            assert torch.isfinite(obs["actor"]).all() and torch.isfinite(rew).all()
        frac = term.drop_ticks / term.ticks
        assert 0.01 < frac < 0.08, frac
        from mjlab_microduck.actuator.friction_dr_bam import FrictionDRBamActuator
        fs = torch.cat([a.friction_scale.flatten() for a in env.scene["robot"].actuators
                        if isinstance(a, FrictionDRBamActuator)])
        assert float(fs.min()) >= 1.0 - 1e-6 and float(fs.max()) <= 1.5 + 1e-6
        assert not math.isnan(float(rew.mean()))
    finally:
        env.close()
