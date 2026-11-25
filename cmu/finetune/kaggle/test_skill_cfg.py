"""CPU checks of the current-body skill cfgs (tasks/cmu_skill_env_cfg.py) that skill_worker.py trains.

Builds small CPU envs (2 envs each) of Pollen's ground pick and sit-stand and of the CMU current-body versions, with
the robot conditions off and on, and checks: total body mass = the current manifest's mass (1 g); 14 actions; the
actor observation has Pollen's size and term layout; the phase (ground pick, slots 48-50) and the sit flag (slot 48)
sit where Pollen's cfg puts them and carry the live command; recipe pollen = Pollen's reward terms and weights, bend4
adds exactly the bend4 terms; conditions robot = scratch_worker.route_conditions("robot").
Run on the Mac mini: `.venv/bin/python -m pytest -q cmu/finetune/kaggle/test_skill_cfg.py` from the repository root.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scratch_worker as sw  # noqa: E402
import skill_worker as skw  # noqa: E402

import mjlab_microduck.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab_microduck.tasks import cmu_skill_env_cfg as sk  # noqa: E402
from mjlab_microduck.tasks.cmu_asbuilt import load_apply_masses  # noqa: E402
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg  # noqa: E402
from mjlab_microduck.tasks.microduck_sitstand_env_cfg import make_microduck_sitstand_env_cfg  # noqa: E402

skw._install_stop_shim()

BUILDERS = {
    "gp_pollen": lambda: make_microduck_ground_pick_env_cfg(),
    "gp_cur": lambda: sk.make_cmu_ground_pick_env_cfg("current"),
    "gp_cur_robot": lambda: sk.make_cmu_ground_pick_env_cfg("current", conditions="robot"),
    "gp_bend4_robot": lambda: sk.make_cmu_ground_pick_env_cfg("current", recipe="bend4", conditions="robot"),
    "ss_pollen": lambda: make_microduck_sitstand_env_cfg(),
    "ss_cur": lambda: sk.make_cmu_sitstand_env_cfg("current"),
    "ss_cur_robot": lambda: sk.make_cmu_sitstand_env_cfg("current", conditions="robot"),
}
POLLEN = {"gp": "gp_pollen", "ss": "ss_pollen"}
CMU = [k for k in BUILDERS if "pollen" not in k]


@pytest.fixture(scope="module")
def envs():
    out = {}
    for k, b in BUILDERS.items():
        cfg = b()
        cfg.scene.num_envs = 2
        cfg.seed = 3
        out[k] = (cfg, ManagerBasedRlEnv(cfg=cfg, device="cpu"))
    yield out
    for _, e in out.values():
        e.close()


def manifest_mass_g():
    am = load_apply_masses()
    _, report = am.build_scene("current")
    return report["total_after_g"]


def twist_slot(cfg, env):
    names = env.observation_manager.active_terms["actor"]
    dims = env.observation_manager.group_obs_term_dim["actor"]
    off = 0
    for n, d in zip(names, dims):
        p = cfg.observations["actor"].terms[n].params or {}
        if p.get("command_name") == "twist":
            return off, int(d[0])
        off += int(d[0])
    raise AssertionError("no twist command term in the actor group")


@pytest.mark.parametrize("key", CMU)
def test_body_mass_is_current_manifest(envs, key):
    _, env = envs[key]
    got = float(env.sim.mj_model.body_mass.sum()) * 1000
    want = manifest_mass_g()
    assert abs(got - want) <= 1.0, (key, got, want)
    assert abs(got - 918.0) <= 1.0, (key, got)


@pytest.mark.parametrize("key", list(BUILDERS))
def test_action_size_14(envs, key):
    assert envs[key][1].action_manager.total_action_dim == 14


@pytest.mark.parametrize("key", CMU)
def test_obs_layout_matches_pollen(envs, key):
    cfg, env = envs[key]
    pcfg, penv = envs[POLLEN[key[:2]]]
    om, pom = env.observation_manager, penv.observation_manager
    assert om.group_obs_dim["actor"] == pom.group_obs_dim["actor"]
    assert om.group_obs_dim["actor"][0] == 61
    assert om.active_terms["actor"] == pom.active_terms["actor"]
    assert om.group_obs_term_dim["actor"] == pom.group_obs_term_dim["actor"]
    assert twist_slot(cfg, env) == twist_slot(pcfg, penv) == (48, 3)


@pytest.mark.parametrize("key", CMU)
def test_command_slots_carry_live_command(envs, key):
    _, env = envs[key]
    obs, _ = env.reset()
    for _ in range(3):
        obs, *_ = env.step(torch.zeros(2, 14))
    a = obs["actor"][:, 48:51]
    cmd = env.command_manager.get_command("twist")
    assert torch.allclose(a, cmd, atol=1e-5), (a, cmd)
    if key.startswith("gp"):
        ph = env.command_manager.get_term("twist")._gp_phase
        assert torch.allclose(a[:, 0], torch.cos(2 * math.pi * ph), atol=1e-5)
        assert torch.allclose(a[:, 1], torch.sin(2 * math.pi * ph), atol=1e-5)
    else:
        assert set(a[:, 0].tolist()) <= {0.0, 1.0} and torch.all(a[:, 1:] == 0)


def _weights(cfg):
    return {k: v.weight for k, v in cfg.rewards.items()}


def test_pollen_recipe_reward_is_pollens():
    assert _weights(sk.make_cmu_ground_pick_env_cfg("current")) == _weights(make_microduck_ground_pick_env_cfg())
    assert _weights(sk.make_cmu_sitstand_env_cfg("current")) == _weights(make_microduck_sitstand_env_cfg())
    assert _weights(sk.make_cmu_ground_pick_env_cfg("current", conditions="robot")) == _weights(
        make_microduck_ground_pick_env_cfg())


def test_bend4_adds_exactly_its_terms():
    p = _weights(make_microduck_ground_pick_env_cfg())
    b = _weights(sk.make_cmu_ground_pick_env_cfg("current", recipe="bend4"))
    assert set(b) - set(p) == set(sk.BEND4_REWARDS)
    assert {k: b[k] for k in p} == p
    assert {k: b[k] for k in sk.BEND4_REWARDS} == sk.BEND4_REWARDS
    for k, w in sk.BEND4_REWARDS.items():
        assert w < 0 or k == "bend_guard", k  # costs (>= 0) on negative weights; the guard only watches


def test_bend4_env_has_held_phase_and_handback(envs):
    cfg, env = envs["gp_bend4_robot"]
    cmd = env.command_manager.get_term("twist")
    assert cmd.HOLD["phase"] == 0.33 and cmd.HOLD["hb_frac"] == 0.25 and cmd.HOLD["hb_s"] == 2.0
    assert hasattr(env, "_hb_stats")  # install_handback ran at startup with the walker
    assert Path(cfg.events["handback"].params["stander_path"]).name == "bw-s2c-b2.onnx"


def test_robot_conditions_equal_scratch_route():
    assert sk.robot_route_conditions() == sw.route_conditions("robot")


@pytest.mark.parametrize("make", [lambda c: sk.make_cmu_ground_pick_env_cfg("current", conditions=c),
                                  lambda c: sk.make_cmu_sitstand_env_cfg("current", conditions=c)], ids=["gp", "ss"])
def test_conditions_on_and_off(make):
    off, on = make("official"), make("robot")
    a_off, a_on = off.observations["actor"].terms, on.observations["actor"].terms
    assert "zero_shift_bias" not in off.events and "zero_shift_bias" in on.events
    assert "k_range" not in (a_off["base_ang_vel"].params or {}) and "k_range" in a_on["base_ang_vel"].params
    assert a_on["joint_pos"].delay_min_lag == a_on["joint_pos"].delay_max_lag == 1
    for t in ("base_ang_vel", "projected_gravity"):
        assert a_on[t].delay_max_lag == 2, (t, a_on[t].delay_max_lag)
    assert on.events["randomize_joint_friction"].params["scale_range"] == (1.0, 1.5)
    assert type(on.actions["joint_pos"]) is not type(off.actions["joint_pos"])  # servo read drop action
    assert on.__dict__["_robot_conditions"] == sw.route_conditions("robot")


def test_ground_pick_official_imu_lag_is_pollens():
    a = sk.make_cmu_ground_pick_env_cfg("current").observations["actor"].terms
    assert a["base_ang_vel"].delay_max_lag == 3 and a["projected_gravity"].delay_max_lag == 3
