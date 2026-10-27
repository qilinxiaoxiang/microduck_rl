"""Invariants of the CMU seeing pick (tasks/cmu_seeing_pick_env_cfg.py).

The cfg tests only build cfgs; the env tests build one 8-env CPU env (module fixture) and step it.
"""
import math

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp
from mjlab_microduck.tasks import mdp as microduck_mdp

TASK = "Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged"
BODY = slice(55, 61)


# ── cfg ──────────────────────────────────────────────────────────────────────

def test_task_registered():
    assert TASK in list_tasks()


def test_events_in_order():
    names = list(sp.make_seeing_pick_env_cfg().events)
    assert names.index("reset_base") < names.index("object_size") < names.index("reset_object")
    assert names.index("record_grasp_start") > names.index("reset_base")


def test_pollen_rewards_kept():
    from mjlab_microduck.tasks.cmu_grasp_env_cfg import make_cmu_grasp_env_cfg
    from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import make_microduck_static_grasp_env_cfg
    base = make_cmu_grasp_env_cfg(make_microduck_static_grasp_env_cfg, "plugged", ball=False, hover=True).rewards
    r = sp.make_seeing_pick_env_cfg().rewards
    assert set(r) == set(base)
    for k in r:
        assert r[k].weight == base[k].weight, k
        assert r[k].params == base[k].params, k
    for k in sp.GATED_RETURN_TERMS:
        assert r[k].func in (sp.return_pose_after_descent, sp.return_upright_after_descent), k


def test_play_uses_operating_window():
    cfg = sp.make_seeing_pick_env_cfg(play=True)
    assert cfg.events["reset_object"].params["window_xy"] == sp.WINDOW_OPERATING
    assert "seeing_window" not in cfg.curriculum


# ── env ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = 8
    e = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    e.reset()
    yield e
    e.close()


def _zero(env):
    return torch.zeros(env.num_envs, env.action_manager.total_action_dim)


def _obs(env):
    return env.observation_manager.compute()["actor"]


def _foot_xy(env):
    robot, obj = env.scene["robot"], env.scene["object"]
    return sp._to_foot_frame(env, obj.data.root_link_pos_w, robot.data.root_link_pos_w, robot.data.root_link_quat_w)


def test_actor_obs_61d_and_start_measurement(env):
    o = _obs(env)
    assert o.shape == (env.num_envs, 61)
    body = o[:, BODY]
    err = (body[:, :2] - _foot_xy(env)).abs()
    assert float(err.max()) < sp.CAM_BIAS_XY + 4 * sp.CAM_NOISE_XY + 0.003
    assert torch.all(body[:, 3] >= 0) and torch.all(body[:, 3] <= sp.AGE_MAX_S)
    assert torch.all(body[:, 5] == 0)


def test_one_active_shape_on_the_floor(env):
    st = sp._st(env)
    obj = env.scene["object"]
    gids = [int(obj.indexing.geom_ids[i]) for i in range(3)]
    sizes = env.sim.model.geom_size[:, gids]  # (n, 3, 3)
    big = (sizes[..., :2].max(-1).values > 2 * sp.HIDDEN)
    assert torch.all(big.sum(-1) == 1)
    assert torch.all(big.float().argmax(-1) == st["shape"])
    z = obj.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]
    assert torch.allclose(z, st["half"][:, 2], atol=0.003)


def test_camera_sees_object_in_view_and_loses_one_behind(env, monkeypatch):
    """An object on the floor 20 deg below the camera's optical axis is detected (age falls back to about the delay);
    one behind the robot never is (age keeps growing)."""
    from mjlab.utils.lab_api.math import quat_apply
    env.reset()
    robot, obj = env.scene["robot"], env.scene["object"]
    st = sp._st(env)
    st["drop"][:] = 0.0
    st["stall_until"][:] = -1
    monkeypatch.setattr(sp, "CAM_STALL_PROB", 0.0)
    sid = robot.find_sites(["head_camera"])[0][0]
    cam_p, cam_q = robot.data.site_pos_w[:, sid], robot.data.site_quat_w[:, sid]
    d = quat_apply(cam_q, torch.tensor([[1.0, 0.0, -math.tan(math.radians(20))]]).repeat(env.num_envs, 1))
    floor = env.scene.terrain.env_origins[:, 2] + st["half"][:, 2] + 0.0005
    assert torch.all(d[:, 2] < 0)
    hit = cam_p + d * ((floor - cam_p[:, 2]) / d[:, 2]).unsqueeze(-1)
    ids = torch.arange(env.num_envs)
    front = ids < env.num_envs // 2
    root = robot.data.root_link_pos_w
    yaw = sp._yaw(robot.data.root_link_quat_w)
    pose = torch.zeros(env.num_envs, 7)
    pose[:, :3] = hit
    pose[~front, 0] = root[~front, 0] - 0.4 * torch.cos(yaw[~front])
    pose[~front, 1] = root[~front, 1] - 0.4 * torch.sin(yaw[~front])
    pose[:, 2] = floor
    pose[:, 3] = 1.0
    obj.write_root_link_pose_to_sim(pose, ids)
    obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6), ids)
    st["kick_step"][:] = -1
    for _ in range(25):
        env.step(_zero(env))
    body = _obs(env)[:, BODY]
    max_seen_age = sp.CAM_DELAY_S[1] + sp.CAM_PERIOD_STEPS * env.step_dt + 1e-6
    assert torch.all(body[front, 3] <= max_seen_age), body[front, 3]
    assert torch.all(body[~front, 3] > 0.45), body[~front, 3]
    true = _foot_xy(env)
    assert float((body[front, :2] - true[front]).abs().max()) < 0.04


def test_kick_rolls_object_and_frees_spawn(env):
    env.reset()
    st = sp._st(env)
    t = env.episode_length_buf.clone()
    st["kick_step"][:] = -1
    st["kick_step"][0] = t[0] + 1
    obj = env.scene["object"]
    before = obj.data.root_link_pos_w[0, :2].clone()
    for _ in range(3):
        env.step(_zero(env))
    assert float((obj.data.root_link_pos_w[0, :2] - before).norm()) > 0.008
    spawn = microduck_mdp._gg_buffer(env, "object_spawn", 3)
    assert torch.allclose(spawn[0, :2], obj.data.root_link_pos_w[0, :2], atol=0.005)


def test_return_terms_wait_for_descent(env):
    env.reset()
    st = sp._st(env)
    rm = env.reward_manager
    cmd = env.command_manager.get_term("twist")
    cmd._gp_phase[:] = 0.8                       # in the stand-up window
    cmd.vel_command_b[:, 0] = math.cos(2 * math.pi * 0.8)
    cmd.vel_command_b[:, 1] = math.sin(2 * math.pi * 0.8)
    st["lip_min"][:] = 1.0
    st["lip_min"][0] = 0.01
    up = sp.return_upright_after_descent(env, **{k: v for k, v in rm.get_term_cfg("return_upright").params.items()})
    assert float(up[1:].abs().max()) == 0.0 and float(up[0]) > 0.0


def test_rewards_finite(env):
    env.reset()
    for _ in range(10):
        _, rew, *_ = env.step(_zero(env))
        assert torch.isfinite(rew).all()
