"""DELTA_T3A / DELTA_T3B (cmu/finetune/modal/finetune_worker.py; FINETUNE-DELTA.md "Run t3 plan"): t2b plus the
identified sim gap in the DR (BAM friction 1.0-1.5, gyro vibration proportional to leg speed), a second-difference
action term, and a counterfactual leg anchor on head-command states.

CPU only: registration, the cfg edits, the two new mdp functions on a fake env, and the anchor targets and weights on
walk61's own weights (rebuilt from walk61.onnx).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
import mjlab_microduck.tasks.mdp as mdp
from mjlab.tasks.registry import load_env_cfg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "cmu" / "finetune" / "modal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_worker as fw  # noqa: E402
from test_t2_anchor import WALK61, _Actor  # noqa: E402


@pytest.fixture(scope="module")
def walk61():
    if not WALK61.exists():
        pytest.skip(f"{WALK61} missing")
    return _Actor(WALK61)


def test_t3_is_t2b_plus_three_changes_and_registered():
    for name, beta_legs in (("t3a", 0.0), ("t3b", 1.0)):
        d = fw.DELTAS[name]
        base = {k: v for k, v in d.items() if k not in ("joint_friction", "gyro_vibration", "action_acc", "anchor")}
        assert base == {k: v for k, v in fw.DELTA_T2B.items() if k != "anchor"}
        assert tuple(d["joint_friction"]) == (1.0, 1.4)
        assert tuple(d["gyro_vibration"]["k_range"]) == (0.15, 0.35)
        assert d["action_acc"]["weight"] < 0
        assert d["anchor"]["beta"] == 2.0 and d["anchor"]["beta_legs"] == beta_legs
    src = (REPO / "cmu" / "finetune" / "modal" / "finetune.py").read_text()
    assert '"t3a": "t3a_asbuilt"' in src and '"t3b": "t3a_asbuilt"' in src


def test_cfg_edits_reach_the_actor_only():
    c = load_env_cfg(fw.TASK, play=False)
    fw.apply_delta(c, dict(fw.DELTA_T3A))
    assert tuple(c.events["randomize_joint_friction"].params["scale_range"]) == (1.0, 1.4)
    av = c.observations["actor"].terms["base_ang_vel"]
    assert av.func is mdp.base_ang_vel_imu_misaligned_vibration
    assert tuple(av.params["k_range"]) == (0.15, 0.35) and "max_angle_deg" in av.params
    assert c.observations["critic"].terms["base_ang_vel"].func is not mdp.base_ang_vel_imu_misaligned_vibration
    acc = c.rewards["action_acc_l2"]
    assert acc.func is mdp.action_acc_l2 and acc.weight == fw.DELTA_T3A["action_acc"]["weight"]
    assert c.rewards["action_rate_l2"].weight < 0  # the first-difference term stays
    # t2b's settings survive
    assert c.commands["head_pose"].zero_command_prob == 0.2


def test_action_acc_l2_zero_on_constant_positive_on_alternating():
    a = torch.ones(5, 14)
    env = SimpleNamespace(action_manager=SimpleNamespace(action=a, prev_action=a.clone(), prev_prev_action=a.clone()))
    assert torch.all(mdp.action_acc_l2(env) == 0)
    # a smooth ramp (constant first difference) costs nothing; an alternating pattern costs (2 x amplitude)^2 per joint
    env.action_manager.action, env.action_manager.prev_action, env.action_manager.prev_prev_action = 3 * a, 2 * a, a
    assert torch.all(mdp.action_acc_l2(env) == 0)
    env.action_manager.action, env.action_manager.prev_action, env.action_manager.prev_prev_action = a, -a, a
    assert torch.allclose(mdp.action_acc_l2(env), torch.full((5,), 14 * 16.0))


def test_gyro_vibration_scales_with_leg_speed(monkeypatch):
    n = 4000
    env = SimpleNamespace(num_envs=n, device="cpu", episode_length_buf=torch.full((n,), 10), scene={"robot": object()})
    monkeypatch.setattr(mdp, "base_ang_vel_imu_misaligned", lambda env, max_angle_deg, asset_cfg: torch.zeros(env.num_envs, 3))
    speeds = {}

    def fake_vel(env, asset):
        v = torch.zeros(env.num_envs, 14)
        v[:, mdp._LEG_SERVO_IDX] = speeds["legs"]
        v[:, 5:9] = 50.0  # head joints never count
        return v
    monkeypatch.setattr(mdp, "_servo_joint_vel", fake_vel)
    torch.manual_seed(0)
    speeds["legs"] = 0.0
    assert torch.all(mdp.base_ang_vel_imu_misaligned_vibration(env, k_range=(0.25, 0.25)) == 0)
    speeds["legs"] = 2.0  # rms leg speed 2 rad/s -> std 0.5 at k 0.25
    out = mdp.base_ang_vel_imu_misaligned_vibration(env, k_range=(0.25, 0.25))
    assert abs(float(out.std()) - 0.5) < 0.03
    speeds["legs"] = 4.0
    out2 = mdp.base_ang_vel_imu_misaligned_vibration(env, k_range=(0.25, 0.25))
    assert abs(float(out2.std()) - 1.0) < 0.05
    # k is per env and resampled at episode start
    env.episode_length_buf[:] = 0
    mdp.base_ang_vel_imu_misaligned_vibration(env, k_range=(0.15, 0.35))
    k = env._gyro_vib_k
    assert k.shape == (n, 1) and 0.15 <= float(k.min()) and float(k.max()) <= 0.35 and float(k.std()) > 0.03


def _rows(vx_head):
    x = torch.randn(len(vx_head), 61) * 0.5
    x[:, fw.ACTOR_HEAD] = 0.0
    for i, (vx, h) in enumerate(vx_head):
        x[i, fw.ACTOR_TWIST.start] = vx
        x[i, fw.ACTOR_HEAD.start] = h
    return x


def test_anchor_targets_default_head_rows_only_when_beta_legs_is_zero(walk61):
    x = _rows([(0.27, 0.0), (0.27, -0.4), (0.0, 0.0), (0.10, 0.0), (0.0, -0.8)])
    xk, mu0, w = fw.anchor_targets(walk61, x, dict(fw.DELTA_T3A["anchor"]))
    assert xk.shape[0] == 2 and torch.all(w == 2.0)
    with torch.no_grad():
        assert torch.allclose(mu0, fw.actor_mean(walk61, xk))
    assert float(fw.anchor_loss_w(walk61, xk, mu0, w)) < 1e-10


def test_anchor_targets_counterfactual_legs_on_head_rows(walk61):
    x = _rows([(0.27, 0.0), (0.27, -0.4), (0.0, -0.8), (0.10, -0.4)])
    xk, mu0, w = fw.anchor_targets(walk61, x, dict(fw.DELTA_T3B["anchor"]))
    assert xk.shape[0] == 3  # the 0.10 row is outside the band
    head_rows = ~(xk[:, fw.ACTOR_HEAD] == 0).all(dim=1)
    assert int(head_rows.sum()) == 2
    assert torch.all(w[~head_rows] == 2.0)
    assert torch.all(w[head_rows][:, fw.LEG_OUT] == 1.0) and torch.all(w[head_rows][:, 5:9] == 0.0)
    x0 = xk.clone()
    x0[:, fw.ACTOR_HEAD] = 0.0
    with torch.no_grad():
        assert torch.allclose(mu0, fw.actor_mean(walk61, x0))
        # walk61 itself is not at the counterfactual target on head rows: its loss there is beta_legs x the leg-output
        # distance between walk61(x) and walk61(x with the head zeroed), and zero on the default-head row
        own = fw.actor_mean(walk61, xk)
        expect = (w * torch.square(own - mu0)).sum(dim=1)
        assert float(expect[~head_rows].sum()) < 1e-10 and float(expect[head_rows].min()) > 0
        assert abs(float(fw.anchor_loss_w(walk61, xk, mu0, w)) - float(expect.mean())) < 1e-6
    # a gradient step on a perturbed copy lowers the weighted loss
    import copy
    pert = copy.deepcopy(walk61)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in pert.mlp.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=g))
    before = float(fw.anchor_loss_w(pert, xk, mu0, w))
    opt = torch.optim.Adam(pert.mlp.parameters(), lr=3e-4)
    opt.zero_grad()
    fw.anchor_loss_w(pert, xk, mu0, w).backward()
    opt.step()
    assert float(fw.anchor_loss_w(pert, xk, mu0, w)) < before
