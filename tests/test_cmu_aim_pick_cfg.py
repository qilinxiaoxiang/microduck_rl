"""Invariants of the CMU aiming crouch (tasks/cmu_aim_pick_env_cfg.py).

The cfg tests only build cfgs; the env tests build one 8-env CPU env (module fixture, about 30 s with cold warp
kernels) and step it a few times.
"""
import json
import math
from pathlib import Path

import mujoco
import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.tasks import cmu_aim_pick_env_cfg as ap
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg

TASK = "Mjlab-AimPick-Flat-MicroDuck"
AUDIT_FILE = Path(__file__).resolve().parents[1] / "cmu" / "aim_pick" / "reward-audit.json"
BODY = slice(55, 61)  # body_command slots in the 61-D actor obs: 48 proprio + twist(3) + head(4) + body(6)


# ── cfg ──────────────────────────────────────────────────────────────────────

def test_tasks_registered():
    assert {t for t, _ in ap.AIM_PICK_TASKS} <= set(list_tasks())


def test_rewards_replace_ground_proximity_with_aim_terms():
    r = ap.make_aim_pick_env_cfg().rewards
    assert "mouth_ground_proximity" not in r
    assert r["aim_reach"].weight > 0 and r["aim_hold"].weight > 0 and r["aim_progress"].weight > 0
    # object_displacement_cost returns >= 0: mjlab-base style, negative weight
    assert r["object_displacement"].weight < 0
    # the ground pick's balance and return terms are kept unchanged
    gp = make_microduck_ground_pick_env_cfg().rewards
    for name in ("mouth_perpendicular_to_ground", "ground_pick_return_pose_legs", "ground_pick_return_pose_neck",
                 "return_upright", "feet_grounded", "feet_flat", "head_impact_penalty", "mouth_payload_force"):
        assert r[name].weight == gp[name].weight, name


def test_command_starts_standing_and_holds():
    cmd = ap.make_aim_pick_env_cfg().commands["twist"]
    assert cmd.class_type is ap.AimPickPhaseCommand
    assert cmd.randomize_phase is False
    assert cmd.period == 4.0  # the runtime's GP_PERIOD
    from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import DESCENT_END, HOLD_END
    assert DESCENT_END <= cmd.hold_phase < HOLD_END


def test_object_events_after_reset_base():
    names = list(ap.make_aim_pick_env_cfg().events)
    assert names.index("reset_base") < names.index("randomize_object_size") < names.index("reset_aim_object")


def test_play_uses_operating_window_and_final_scale():
    cfg = ap.make_aim_pick_env_cfg(play=True)
    assert cfg.events["reset_aim_object"].params["window_xy"] == ap.WINDOW_OPERATING
    assert cfg.rewards["aim_hold"].params["scale"] == ap.AIM_HOLD_SCALES[-1]
    assert "aim" not in cfg.curriculum


def test_reward_audit_passed_on_the_current_reward():
    """cmu/aim_pick/audit.py must be rerun (and pass) after any change to the aim terms, their weights or the set of
    reward terms: the D-016 gate, as finetune.py::require_audit does for the walk fine-tune."""
    audit = json.loads(AUDIT_FILE.read_text())
    assert {"Mjlab-AimPick-Flat-MicroDuck", "Mjlab-AimPick-Flat-MicroDuck-CMU-Plugged"} <= set(audit)
    for task, a in audit.items():
        cfg = load_env_cfg(task)
        assert a["audit"]["verdict"] == "PASS", task
        assert a["weights"] == {k: cfg.rewards[k].weight for k in a["weights"]}, f"{task}: rerun cmu/aim_pick/audit.py"
        assert set(a["scenarios"]["reach"]["terms_per_s"]) == set(cfg.rewards), f"{task}: rerun cmu/aim_pick/audit.py"
        assert a["hold_scale"] == ap.AIM_HOLD_SCALES[-1] and a["window_m"] == ap.WINDOW_OPERATING
        assert all(a["audit"]["over_reach"][k] <= a["audit"]["max_ratio"] for k in a["audit"]["gated"])


def test_window_reaches_operating_range_and_clears_toes():
    assert ap.WINDOW_START == 0.01 and ap.WINDOW_MAX >= 0.05
    # soles end at x 0.034 m (HOME FK); the box's half diagonal at any yaw is OBJECT_HALF_XY * sqrt(2)
    assert ap.OBJECT_X_MIN - ap.OBJECT_HALF_XY * math.sqrt(2) >= 0.034


# ── env (CPU) ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def env():
    cfg = load_env_cfg(TASK, play=False)
    cfg.scene.num_envs = 8
    e = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    e.reset()
    yield e
    e.close()


def _yaw(q):
    return torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def test_actor_obs_is_61d_with_object_in_body_slots(env):
    assert env.observation_manager.active_terms["actor"] == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command",
        "body_command"]
    obs = env.observation_manager.compute()["actor"]
    assert obs.shape == (8, 61)
    body = obs[:, BODY]
    assert torch.allclose(body[:, :3], env._aim_pick["cmd"])
    assert (body[:, 3:] == 0).all()
    # x ahead of the toes, y inside the window plus the observation bias, height 4-9 cm plus its bias
    assert (body[:, 0] >= ap.OBJECT_X_MIN - ap.OBS_NOISE_XY - 1e-6).all()
    assert (body[:, 1].abs() <= ap.WINDOW_MAX + ap.OBS_NOISE_XY).all()
    assert (body[:, 2] >= 0.04 - ap.OBS_NOISE_H - 1e-6).all() and (body[:, 2] <= 0.09 + ap.OBS_NOISE_H + 1e-6).all()


def test_box_spawns_where_the_command_says(env):
    env.reset()
    robot, obj = env.scene["robot"], env.scene[ap.OBJECT_NAME]
    rel = (obj.data.root_link_pos_w - robot.data.root_link_pos_w)[:, :2]
    yaw = _yaw(robot.data.root_link_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    local = torch.stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], -1)
    cmd = env._aim_pick["cmd"][:, :2]
    assert (local - cmd).abs().max() <= ap.OBS_NOISE_XY + 1e-4
    assert (local[:, 0] >= ap.OBJECT_X_MIN - 1e-4).all()
    h = 2 * env.sim.model.geom_size[:, env._aim_pick["gid"], 2]
    assert torch.allclose(obj.data.root_link_pos_w[:, 2], h / 2, atol=1e-4)
    assert ((h >= 0.04 - 1e-6) & (h <= 0.09 + 1e-6)).all()


def test_head_ignores_box_but_floor_and_legs_do_not(env):
    m = env.sim.mj_model
    head = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/jaw_soft")
    box = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{ap.OBJECT_NAME}/{ap.OBJECT_NAME}_geom")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    foot = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "robot/left_foot_collision")

    def collide(a, b):
        return bool(m.geom_contype[a] & m.geom_conaffinity[b]) or bool(m.geom_contype[b] & m.geom_conaffinity[a])

    head_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == head and (m.geom_contype[g] or m.geom_conaffinity[g])]
    assert len(head_geoms) == 3
    for g in head_geoms:
        assert not collide(g, box)
        assert collide(g, floor)          # head_impact_penalty still sees the floor
        assert collide(g, foot)           # self-collision still sees the legs
    assert collide(box, floor) and collide(box, foot)


def test_rewards_finite_and_penalties_signed(env):
    env.reset()
    torch.manual_seed(0)
    for _ in range(40):
        _, rew, _, _, _ = env.step(0.2 * torch.randn(8, 14))
        assert torch.isfinite(rew).all()
        step = env.reward_manager._step_reward
        assert torch.isfinite(step).all()
    names = env.reward_manager.active_terms
    for name in ("aim_reach", "aim_hold", "aim_progress"):
        assert (step[:, names.index(name)] >= 0).all()
    assert (step[:, names.index("object_displacement")] <= 0).all()


def _set_phase(env, phase):
    cmd = env.command_manager.get_term("twist")
    cmd._gp_phase[:] = phase
    cmd.vel_command_b[:, 0] = math.cos(2 * math.pi * phase)
    cmd.vel_command_b[:, 1] = math.sin(2 * math.pi * phase)


def test_aim_terms_peak_at_the_grasp_point(env):
    env.reset()
    env.step(torch.zeros(8, 14))
    robot, obj = env.scene["robot"], env.scene[ap.OBJECT_NAME]
    sid = robot.find_sites(["mouth_tip"])[0][0]
    tip = robot.data.site_pos_w[:, sid, :].clone()
    # move every box so its grasp point sits on the beak tip, upright
    h = 2 * env.sim.model.geom_size[:, env._aim_pick["gid"], 2]
    pose = torch.zeros(8, 7)
    pose[:, :2] = tip[:, :2]
    pose[:, 2] = tip[:, 2] + ap.GRIP_DEPTH - h / 2
    pose[:, 3] = 1.0
    obj.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    tip_cfg = env.reward_manager.get_term_cfg("aim_hold").params["asset_cfg"]
    d = ap.tip_to_grasp(env, tip_cfg)
    assert d.max() < 1e-3
    _set_phase(env, ap.HOLD_PHASE)
    hold = ap.aim_hold_precision(env, scale=0.01, asset_cfg=tip_cfg)
    reach = ap.aim_reach_phased(env, asset_cfg=tip_cfg)
    assert torch.allclose(hold, torch.ones(8), atol=0.1) and torch.allclose(reach, torch.ones(8), atol=0.05)
    _set_phase(env, 0.9)  # standing rest: no aim pay
    assert ap.aim_hold_precision(env, asset_cfg=tip_cfg).abs().max() == 0
    assert ap.aim_reach_phased(env, asset_cfg=tip_cfg).abs().max() == 0


def test_phase_freezes_once_in_the_hold(env):
    cmd = env.command_manager.get_term("twist")
    env.reset()
    cmd._hold_left[:] = 0.5
    dt = env.step_dt
    seen = []
    for _ in range(int(2.6 / dt)):
        cmd.compute(dt)
        seen.append(float(cmd._gp_phase[0]))
    at_hold = sum(abs(p - ap.HOLD_PHASE) < 1e-6 for p in seen)
    # held 0.5 s at HOLD_PHASE, plus the crossing step that lands on it and one step of float32 countdown slop
    assert abs(at_hold * dt - 0.5) <= 2 * dt + 1e-6
    assert seen[-1] > ap.HOLD_PHASE            # then moved on
    assert float(cmd._hold_left[0]) == 0.0


def test_displacement_cost_saturates(env):
    env.reset()
    env.step(torch.zeros(8, 14))
    obj = env.scene[ap.OBJECT_NAME]
    assert ap.object_displacement_cost(env).max() < 0.05
    pose = torch.cat([obj.data.root_link_pos_w.clone(), obj.data.root_link_quat_w.clone()], -1)
    pose[:, 0] += 0.10
    obj.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    env.scene.update(dt=env.physics_dt)
    assert torch.allclose(ap.object_displacement_cost(env), torch.ones(8))


def test_curriculum_widens_and_tightens_on_low_error(env):
    st = env._aim_pick
    ev = env.event_manager.get_term_cfg("reset_aim_object")
    rw = env.reward_manager.get_term_cfg("aim_hold")
    w0, s0 = ev.params["window_xy"], rw.params["scale"]
    st["hold_err"] = 0.5   # large error: nothing moves
    env.common_step_counter = 10_000 * ap.STEPS_PER_ITER
    ap.aim_curriculum(env, None)
    assert ev.params["window_xy"] == w0 and rw.params["scale"] == s0
    st["hold_err"] = 0.005  # small error: one step each
    out = ap.aim_curriculum(env, None)
    assert ev.params["window_xy"] == pytest.approx(w0 + ap.WINDOW_STEP)
    assert rw.params["scale"] < s0
    assert set(out) == {"window", "hold_scale", "hold_err"}
    ap.aim_curriculum(env, None)  # within min_iters: no further step
    assert ev.params["window_xy"] == pytest.approx(w0 + ap.WINDOW_STEP)
