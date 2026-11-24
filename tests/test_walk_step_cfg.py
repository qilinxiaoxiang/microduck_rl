"""Invariants of the walk-step task and its capped residual actor. CPU, no env."""
import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.tasks import cmu_walk_step_env_cfg as ws
from mjlab_microduck.tasks.residual_model import ResidualMLPModel


def _model(cap=0.05, l1=0.1, trim=0.0):
    obs = TensorDict({"actor": torch.zeros(2, 61)}, batch_size=[2])
    dist = {"class_name": "GaussianDistribution", "init_std": ws.INIT_STD, "std_type": "log"}
    return ResidualMLPModel(obs, {"actor": ["actor"]}, "actor", 14, distribution_cfg=dist,
                            base_onnx=str(ws.WALK_ONNX), cap=cap, l1=l1, base_trim_deg=trim)


def _walk61(x):
    s = ort.InferenceSession(str(ws.WALK_ONNX))
    return np.concatenate([s.run(None, {s.get_inputs()[0].name: r[None]})[0] for r in x.numpy()])


def _batch(n=32, scale=1.0):
    x = torch.randn(n, 61, generator=torch.Generator().manual_seed(0)) * scale
    return x, TensorDict({"actor": x}, batch_size=[n])


def test_iteration_zero_is_walk61():
    x, td = _batch(scale=0.3)
    np.testing.assert_allclose(_model()(td).detach().numpy(), _walk61(x), atol=1e-4)


def test_correction_never_exceeds_the_cap():
    for cap in ws.CAPS:
        m = _model(cap)
        with torch.no_grad():
            for p in m.mlp.parameters():
                p.normal_(0, 5.0)
        x, td = _batch(scale=0.3)
        assert np.abs(m(td).detach().numpy() - _walk61(x)).max() <= cap + 1e-4


def test_base_is_not_trainable():
    m = _model()
    assert not any(k.startswith("base.") for k, _ in m.named_parameters())
    assert any(k.startswith("base.") for k in m.state_dict())


def test_l1_hook_equals_an_l1_loss():
    a, b = _model(l1=0.3), _model(l1=0.0)
    with torch.no_grad():
        for p in a.mlp.parameters():
            p.normal_(0, 0.1)
    b.load_state_dict(a.state_dict())
    x, td = _batch()
    a(td).square().mean().backward()
    d = b.correction(b.base.normalize(x))
    (b(td).square().mean() + 0.3 * d.abs().sum(1).mean()).backward()
    for p, q in zip(a.mlp.parameters(), b.mlp.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-3)


def test_onnx_matches_the_model(tmp_path):
    m = _model()
    with torch.no_grad():
        for p in m.mlp.parameters():
            p.normal_(0, 0.1)
    f, e = str(tmp_path / "step.onnx"), m.as_onnx(False)
    torch.onnx.export(e, e.get_dummy_inputs(), f, input_names=e.input_names, output_names=e.output_names,
                      opset_version=18, dynamo=False)
    s = ort.InferenceSession(f)
    x, td = _batch(8, 0.3)
    got = np.concatenate([s.run(None, {"obs": r[None]})[0] for r in x.numpy()])
    np.testing.assert_allclose(got, m(td).detach().numpy(), atol=1e-4)


def test_cfg_contract():
    cfg = ws.make_walk_step_env_cfg()
    assert list(cfg.observations["actor"].terms)[-3:] == ["command", "head_command", "body_command"]
    t = cfg.commands["twist"]
    assert t.cmd_alpha == 0.2               # walk_wired.CMD_ALPHA
    assert t.resampling_time_range[0] > cfg.episode_length_s
    assert t.vy[0] < 0 < t.vy[1]            # the slot stays alive
    assert "head_pose_range" not in cfg.curriculum and "standing_envs" not in cfg.curriculum
    assert "expand_bam_friction_fields" in cfg.events


def test_reward_signs():
    r = ws.make_walk_step_env_cfg().rewards
    assert r["step_position"].weight > 0 and r["step_heading"].weight > 0
    assert r["step_position_l1"].weight < 0 and r["step_settled_speed"].weight < 0
    assert "track_linear_velocity" not in r


def test_tasks_registered():
    for cap in ws.CAPS:
        t = f"Mjlab-WalkStep-Cap{round(cap * 100):03d}-FromWalk61-MicroDuck-CMU-Current"
        assert t in list_tasks()
        rl = load_rl_cfg(t)
        assert rl.actor.cap == cap and rl.algorithm.schedule == "fixed" and rl.algorithm.entropy_coef == 0


def test_head_looks_down_only_between_steps():
    cfg = ws.make_walk_step_env_cfg()
    h = cfg.commands["head_pose"]
    assert h.alpha == 0.2                   # walk_brain.HEAD_ALPHA
    assert list(cfg.commands).index("twist") < list(cfg.commands).index("head_pose")
    assert 0.349 + h.look_ranges[1][1] <= 1.11   # 1.3 tipped the stand on the robot
    assert ws.REST_S[1] > h.look_after_s + h.lead_s


def test_base_trim_is_walk_brains_pitch_trim():
    """Positive trim shows the base a trunk leaning back (gravity x < 0), and touches only gyro and gravity."""
    from mjlab_microduck.tasks.residual_model import pitch_trim_matrix

    x = torch.randn(61, generator=torch.Generator().manual_seed(1))
    x[3:6] = torch.tensor([0.0, 0.0, -1.0])
    y = x @ pitch_trim_matrix(6.0).T
    assert -0.11 < float(y[3]) < -0.10 and float(y[5]) < -0.99
    torch.testing.assert_close(y[6:], x[6:])
    torch.testing.assert_close(y[:3].norm(), x[:3].norm())


def test_trimmed_base_exports_and_stays_within_the_cap(tmp_path):
    from mjlab_microduck.tasks.residual_model import pitch_trim_matrix

    m = _model(cap=0.1, trim=6.0)
    with torch.no_grad():
        for p in m.mlp.parameters():
            p.normal_(0, 0.5)
    f, e = str(tmp_path / "trim.onnx"), m.as_onnx(False)
    torch.onnx.export(e, e.get_dummy_inputs(), f, opset_version=18, dynamo=False, input_names=["obs"],
                      output_names=["actions"])
    x, td = _batch(8, 0.3)
    s = ort.InferenceSession(f)
    got = np.concatenate([s.run(None, {"obs": r[None]})[0] for r in x.numpy()])
    np.testing.assert_allclose(got, m(td).detach().numpy(), atol=1e-4)
    assert np.abs(got - _walk61(x @ pitch_trim_matrix(6.0).T)).max() <= 0.1 + 1e-4


def test_round2_loop_and_smoothness():
    cfg = ws.make_walk_step_env_cfg()
    for a in cfg.scene.entities["robot"].articulation.actuators:
        assert (a.delay_min_lag, a.delay_max_lag) == ws.RUNTIME_ACT_LAG
    for name, lag in ws.RUNTIME_OBS_LAG.items():
        assert cfg.observations["actor"].terms[name].delay_max_lag == lag
    assert cfg.rewards["action_acc_l2"].weight < 0 and cfg.rewards["leg_step_over"].weight < 0
    assert cfg.commands["twist"].idle_prob > 0      # it stands in stand61's slot
    assert "com_range" not in cfg.curriculum


def test_dropped_reads_and_teleop_speeds_are_trained():
    from mjlab_microduck.tasks import mdp
    cfg = ws.make_walk_step_env_cfg()
    a = cfg.actions["joint_pos"]
    assert isinstance(a, mdp.DroppedReadJointPositionActionCfg) and a.drop_range[1] >= 0.08
    assert a.use_default_offset and a.scale == 1.0
    lo, hi = cfg.commands["twist"].long_vx
    assert lo <= 0.2 and hi >= 0.3


def test_noise_variants_configure_servo_noise():
    from types import SimpleNamespace
    from mjlab_microduck.tasks import servo_noise as sn
    assert sn.SERVO_NOISE[3] == dict(pos=0.016, vel=2.0, bias=0.06)
    fixed = ws.make_walk_step_env_cfg(noise=(3, 3))
    assert fixed.observations["actor"].terms["joint_pos"].noise.n_max == 0.016
    assert fixed.events["encoder_bias"].params["bias_range"] == (-0.06, 0.06)
    assert "servo_noise_level" not in fixed.events
    rng = ws.make_walk_step_env_cfg(noise=(1, 3))
    jp = rng.observations["actor"].terms["joint_pos"].noise
    assert isinstance(jp, sn.EpisodeScaledUniformNoiseCfg) and jp.n_max == 0.016
    assert rng.observations["actor"].terms["joint_vel"].noise.n_max == 2.0
    assert rng.events["servo_noise_level"].mode == "reset"
    assert not isinstance(rng.observations["critic"].terms["joint_pos"].noise, sn.EpisodeScaledUniformNoiseCfg)
    # the drawn level sets each env's amplitude: level 1 = a quarter of level 3's, level 3 = all of it
    env = SimpleNamespace(num_envs=2000, device="cpu")
    sn.draw_noise_level(env, None, key="t", lo=1.0, hi=3.0)
    s = sn.NOISE_SCALE["t"]
    assert 0.25 <= float(s.min()) < 0.27 and 0.95 < float(s.max()) <= 1.0
    noisy = sn.EpisodeScaledUniformNoiseCfg(n_min=-1.0, n_max=1.0, key="t").apply(torch.zeros(2000, 14))
    assert torch.all(noisy.abs() <= s[:, None] + 1e-6)
    assert "Mjlab-WalkStep-SN3-FromWalk61-MicroDuck-CMU-Current" in list_tasks()
    assert "Mjlab-WalkStep-SNR13-FromWalk61-MicroDuck-CMU-Current" in list_tasks()


def test_walk_step_keeps_its_own_conditions():
    """rounds 1-5 trained without main's default robot conditions: the step task opts out (its own drop + servo noise)"""
    cfg = ws.make_walk_step_env_cfg()
    assert type(cfg.actions["joint_pos"]).__name__ == "DroppedReadJointPositionActionCfg"
    assert "legspeed" not in cfg.observations["actor"].terms["base_ang_vel"].func.__name__


def test_rc_variants_use_main_robot_conditions():
    """round 6: the -RC- step tasks take robot_conditions.py as it is; RCSNR13 adds servo position noise only"""
    from mjlab_microduck.robot_conditions import ROBOT_CONDITIONS
    from mjlab_microduck.tasks.servo_noise import EpisodeScaledUniformNoiseCfg
    for tag, noise in ws.RC_VARIANTS.items():
        cfg = ws.make_walk_step_env_cfg(noise=noise, robot_conditions=True)
        t = cfg.observations["actor"].terms
        assert "legspeed" in t["base_ang_vel"].func.__name__
        assert type(t["joint_vel"].noise).__name__ == "GaussianNoiseCfg" and t["joint_vel"].noise.std == 0.5
        assert t["joint_pos"].delay_min_lag == t["joint_pos"].delay_max_lag == 1
        assert cfg.actions["joint_pos"].drop == ROBOT_CONDITIONS["servo_drop"]
        assert cfg.events["randomize_joint_friction"].params["scale_range"] == ROBOT_CONDITIONS["joint_friction"]
        assert isinstance(t["joint_pos"].noise, EpisodeScaledUniformNoiseCfg) == (noise is not None)
    for tag in ws.RC_VARIANTS:
        assert f"Mjlab-WalkStep-{tag}-FromWalk61-MicroDuck-CMU-Current" in list_tasks()
