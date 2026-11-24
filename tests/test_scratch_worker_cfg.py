"""The from-scratch route (cmu/finetune/kaggle/scratch_worker.py) trains Pollen's recipe unchanged.

CPU only: builds cfgs and compiles the robot models, never steps an env. For --model stock the env cfg must equal
make_microduck_velocity_env_cfg() field by field; for --model current only the robot's spec_fn may differ, and the
compiled models may differ only in body masses/inertias and the ankle joint range (the heel stop). The runner cfg must
equal MicroduckRlCfg apart from the bookkeeping fields. Run with -s to see the diff printed.
"""
import dataclasses
import enum
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmu/finetune/kaggle"))
import scratch_worker as sw  # noqa: E402

ROBOT_SPEC_FN = "scene.entities['robot'].spec_fn"


def _fn_id(f):
    """Stable identity of a callable: module.qualname, plus its closure values for factory-made functions."""
    name = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}"
    cells = getattr(f, "__closure__", None) or ()
    return (name, tuple(repr(c.cell_contents) for c in cells if not callable(c.cell_contents)))


def _type_id(t):
    """A class made inside a factory (robot_conditions.make_drop_action_cfg) is a new class object per call; two such
    classes count as the same type when module and qualname match, as factory-made functions do in _fn_id. Their
    fields are still compared one by one."""
    return (t.__module__, t.__qualname__) if "<locals>" in t.__qualname__ else t


def cfg_diff(a, b, path="", out=None, seen=None):
    """Every path where two cfg trees differ (dataclasses, dicts, sequences, callables, arrays, scalars)."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    # cycle guard for containers and objects only: Python shares one object per small int, string, True, None, so
    # tracking scalars made every later comparison of an already-seen value (say a delay lag of 1 against 2) a skip
    if not isinstance(a, (int, float, str, bytes, bool, type(None), tuple, frozenset)):
        if id(a) in seen and id(b) in seen:
            return out
        seen.update((id(a), id(b)))
    if type(a) is not type(b) and _type_id(type(a)) != _type_id(type(b)):
        out.append((path, f"type {type(a).__name__} != {type(b).__name__}"))
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        for f in dataclasses.fields(a):
            cfg_diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}" if path else f.name, out, seen)
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((f"{path}[{k!r}]", "only in " + ("route" if k in a else "upstream")))
            else:
                cfg_diff(a[k], b[k], f"{path}[{k!r}]", out, seen)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} != {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                cfg_diff(x, y, f"{path}[{i}]", out, seen)
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b):
            out.append((path, "array differs"))
    elif isinstance(a, type) or isinstance(a, enum.Enum):
        if a is not b:
            out.append((path, f"{a} != {b}"))
    elif callable(a):
        if _fn_id(a) != _fn_id(b):
            out.append((path, f"{_fn_id(a)} != {_fn_id(b)}"))
    elif hasattr(a, "__dict__") and not isinstance(a, (str, bytes, int, float, bool)):
        cfg_diff(vars(a), vars(b), path, out, seen)
    elif a != b:
        out.append((path, f"{a!r} != {b!r}"))
    return out


def _report(label, diffs):
    print(f"\n[{label}] {len(diffs)} difference(s) from upstream")
    for p, d in diffs:
        print(f"  {p}: {d}")


@pytest.fixture(scope="module")
def upstream():
    return make_microduck_velocity_env_cfg(play=False)


def test_stock_env_cfg_equals_upstream(upstream):
    diffs = cfg_diff(sw.make_env_cfg("stock"), upstream)
    _report("stock env cfg", diffs)
    assert diffs == []


def test_current_env_cfg_differs_only_in_robot_model(upstream):
    diffs = cfg_diff(sw.make_env_cfg("current"), upstream)
    _report("current env cfg", diffs)
    assert [p for p, _ in diffs] == [ROBOT_SPEC_FN]


def test_the_diff_sees_recipe_changes(upstream):
    """Negative control: the comparison catches the changes earlier routes made (weights, DR ranges, noise, curricula,
    command ranges, the fitted robot conditions)."""
    from mjlab_microduck.robot_conditions import apply_robot_conditions
    c = sw.make_env_cfg("stock")
    c.rewards["track_linear_velocity"].weight *= 1.01
    c.events["foot_friction"].params["ranges"] = (0.4, 1.3)
    c.observations["actor"].terms["base_ang_vel"].noise.n_max *= 2
    c.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] = -0.123
    c.commands["twist"].ranges.lin_vel_x = (-0.3, 0.3)
    found = cfg_diff(c, upstream)
    assert len(found) >= 5, found
    rc = sw.make_env_cfg("stock")
    apply_robot_conditions(rc)
    assert len(cfg_diff(rc, upstream)) >= 3


def test_two_builds_of_the_route_are_independent(upstream):
    """No shared mutable state: building `current` does not change a later `stock` build or the upstream cfg."""
    sw.make_env_cfg("current")
    assert cfg_diff(sw.make_env_cfg("stock"), make_microduck_velocity_env_cfg(play=False)) == []


GYRO_TERM = "observations['actor'].terms['base_ang_vel']"
GRAV_TERM = "observations['actor'].terms['projected_gravity']"


def test_official_conditions_are_the_default():
    assert cfg_diff(sw.make_env_cfg("current", conditions="official"), sw.make_env_cfg("current")) == []
    assert cfg_diff(sw.make_env_cfg("stock", conditions="official"), sw.make_env_cfg("stock")) == []


def test_gyro_conditions_change_only_the_legspeed_gyro_term():
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE
    off, gyro = sw.make_env_cfg("current"), sw.make_env_cfg("current", conditions="gyro")
    diffs = cfg_diff(gyro, off)
    _report("gyro vs official", diffs)
    assert {p for p, _ in diffs} == {f"{GYRO_TERM}.func", f"{GYRO_TERM}.params['k_range']"}
    av, av0 = gyro.observations["actor"].terms["base_ang_vel"], off.observations["actor"].terms["base_ang_vel"]
    assert av.func.__qualname__ == "make_legspeed_gyro.<locals>.base_ang_vel_legspeed_noise"
    assert av.params["k_range"] == GYRO_LEGSPEED_RANGE == (0.15, 0.30) and "k" not in av.params
    assert (av.noise.n_min, av.noise.n_max) == (av0.noise.n_min, av0.noise.n_max) == (-0.03, 0.03)  # stock floor kept
    # the critic sees the true rate, as in robot conditions
    assert cfg_diff(gyro.observations["critic"], off.observations["critic"]) == []


def test_robot_conditions_equal_the_builder_apart_from_the_gyro_factor():
    """robot = the builder's six conditions; only the gyro factor changes from the constant 0.3 to the per-env range."""
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE
    from mjlab_microduck.tasks.cmu_asbuilt import make_cmu_velocity_env_cfg
    rc = sw.make_env_cfg("current", conditions="robot")
    built = make_cmu_velocity_env_cfg("current", robot_conditions=True)
    diffs = cfg_diff(rc, built)
    _report("robot vs builder", diffs)
    assert {p for p, _ in diffs} == {f"{GYRO_TERM}.params['k']", f"{GYRO_TERM}.params['k_range']",
                                     f"{GYRO_TERM}.delay_max_lag", f"{GRAV_TERM}.delay_max_lag",
                                     "events['zero_shift_bias']"}
    assert rc.observations["actor"].terms["base_ang_vel"].params["k_range"] == GYRO_LEGSPEED_RANGE
    assert built.observations["actor"].terms["base_ang_vel"].params["k"] == 0.3  # the -CMU-Current tasks keep 0.3
    assert len(cfg_diff(rc, sw.make_env_cfg("current"))) >= 6  # and it is not the official cfg


def test_route_conditions_are_what_the_record_holds():
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE, ROBOT_CONDITIONS
    assert sw.route_conditions("official") == {}
    assert sw.route_conditions("gyro") == {"gyro_noise_legspeed_range": GYRO_LEGSPEED_RANGE}
    r = sw.route_conditions("robot")
    assert "gyro_noise_legspeed" not in r and r["gyro_noise_legspeed_range"] == GYRO_LEGSPEED_RANGE
    extra = ("gyro_noise_legspeed_range", "zero_shift_range", "imu_max_lag")
    assert {k: v for k, v in r.items() if k not in extra} == {
        k: v for k, v in ROBOT_CONDITIONS.items() if k != "gyro_noise_legspeed"}
    assert r["imu_max_lag"] == 2
    zs = r["zero_shift_range"]  # d = true - believed, between 0 and the robot's shift (rad)
    assert set(zs) == {"left_hip_roll", "right_hip_roll", "head_roll", "head_yaw"}
    for j, (lo, hi) in {"left_hip_roll": (0.0, 0.0960), "right_hip_roll": (-0.0611, 0.0),
                        "head_roll": (-0.0611, 0.0), "head_yaw": (-0.0873, 0.0)}.items():
        assert abs(zs[j][0] - lo) < 1e-4 and abs(zs[j][1] - hi) < 1e-4, (j, zs[j])
    # the record is JSON-serialisable
    import json
    json.loads(json.dumps(r))
    assert ROBOT_CONDITIONS["gyro_noise_legspeed"] == 0.3  # old scalar key kept for finetune_worker and the tasks


def test_zero_shift_and_imu_age_only_in_robot():
    off, gyro, rob = (sw.make_env_cfg("current", conditions=c) for c in ("official", "gyro", "robot"))
    for cfg in (off, gyro):
        assert "zero_shift_bias" not in cfg.events
        a = cfg.observations["actor"].terms
        assert (a["base_ang_vel"].delay_min_lag, a["base_ang_vel"].delay_max_lag) == (0, 1)
        assert (a["projected_gravity"].delay_min_lag, a["projected_gravity"].delay_max_lag) == (0, 1)
    a = rob.observations["actor"].terms
    assert (a["base_ang_vel"].delay_min_lag, a["base_ang_vel"].delay_max_lag) == (0, 2)
    assert (a["projected_gravity"].delay_min_lag, a["projected_gravity"].delay_max_lag) == (0, 2)
    assert a["base_ang_vel"].delay_update_period == off.observations["actor"].terms["base_ang_vel"].delay_update_period
    assert cfg_diff(rob.observations["critic"], off.observations["critic"]) == []  # critic unchanged
    ev = list(rob.events)
    assert rob.events["zero_shift_bias"].mode == "startup" and ev.index("zero_shift_bias") > ev.index("encoder_bias")
    assert rob.events["encoder_bias"].params["bias_range"] == off.events["encoder_bias"].params["bias_range"]


def test_encoder_bias_sign_convention_with_a_known_offset():
    """mjlab's encoder bias b: the actor's joint_pos reads q_true + b and the action target is action - b. So with
    b = -d the policy commanding angle a puts the real target at a + d and reading the real angle q gives q - d: the
    real joint sits d from where the policy believes it (d = true - believed). Official cfg, 2 envs, CPU."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.envs.mdp.observations import joint_pos_rel
    cfg = sw.make_env_cfg("current")
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        robot = env.scene["robot"]
        j = robot.find_joints("^left_hip_roll$")[0][0]
        d = 0.05
        robot.data.encoder_bias[:] = 0.0
        robot.data.encoder_bias[:, j] = -d
        believed = joint_pos_rel(env, biased=True)  # what the actor reads
        true = joint_pos_rel(env, biased=False)  # what the critic reads
        assert torch.allclose(true[:, j] - believed[:, j], torch.full((2,), d), atol=1e-6)  # true - believed = d
        others = [i for i in range(true.shape[1]) if i != j]
        assert torch.allclose(true[:, others], believed[:, others])
        term = env.action_manager.get_term("joint_pos")
        term.process_actions(torch.zeros(2, term.action_dim))
        term.apply_actions()
        tid = list(term._target_ids.tolist()).index(j) if isinstance(term._target_ids, torch.Tensor) else j  # noqa: SLF001
        believed_target = term._processed_actions[:, tid]  # noqa: SLF001
        real_target = robot.data.joint_pos_target[:, j]
        assert torch.allclose(real_target - believed_target, torch.full((2,), d), atol=1e-6)  # real = believed + d
    finally:
        env.close()


def test_robot_env_draws_zero_shift_per_env_in_range():
    """robot set, 64 envs, CPU: d per env and joint lies in [0, shift], differs across envs, and the encoder bias on
    those joints is the stock +-0.015 draw minus d; the other joints keep the stock draw only."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.robot_conditions import ZERO_SHIFT_RANGE
    cfg = sw.make_env_cfg("current", conditions="robot")
    cfg.scene.num_envs = 64
    cfg.seed = 5
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        d, names = env.__dict__["_zero_shift_d"], env.__dict__["_zero_shift_joints"]
        robot = env.scene["robot"]
        assert d.shape == (64, 4) and names == list(ZERO_SHIFT_RANGE)
        for c, n in enumerate(names):
            lo, hi = ZERO_SHIFT_RANGE[n]
            assert float(d[:, c].min()) >= lo - 1e-7 and float(d[:, c].max()) <= hi + 1e-7, n
            assert len(torch.unique(d[:, c])) == 64
            assert float(d[:, c].abs().max()) > 0.8 * max(abs(lo), abs(hi))  # spans most of the range
            j = robot.find_joints(f"^{n}$")[0][0]
            stock = robot.data.encoder_bias[:, j] + d[:, c]
            assert float(stock.abs().max()) <= 0.015 + 1e-6, n
        other = [i for i in range(robot.num_joints)
                 if robot.joint_names[i] not in names and not robot.joint_names[i].startswith("passive_")]
        assert float(robot.data.encoder_bias[:, other].abs().max()) <= 0.015 + 1e-6
        d0 = d.clone()
        env.reset()
        term = env.action_manager.get_term("joint_pos")
        for _ in range(5):
            obs, *_ = env.step(torch.zeros(64, term.action_dim))
            assert torch.isfinite(obs["actor"]).all()
        assert torch.equal(env.__dict__["_zero_shift_d"], d0)  # once per env, not per reset
    finally:
        env.close()


def test_both_gyro_keys_are_refused():
    from mjlab_microduck.robot_conditions import apply_robot_conditions
    with pytest.raises(AssertionError):
        apply_robot_conditions(sw.make_env_cfg("current"), {"gyro_noise_legspeed": 0.3,
                                                            "gyro_noise_legspeed_range": (0.15, 0.3)})


class _FakeRobot:
    def __init__(self, qd):
        import types
        self.data = types.SimpleNamespace(joint_vel=qd)

    def find_joints(self, _re):
        return (list(range(10)), None)


def _fake_env(qd):
    import types
    return types.SimpleNamespace(num_envs=qd.shape[0], device="cpu", scene={"robot": _FakeRobot(qd)})


def test_per_env_factors_lie_in_the_range_and_differ():
    import torch
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE, make_legspeed_gyro
    torch.manual_seed(0)
    f = make_legspeed_gyro(lambda env: torch.zeros(env.num_envs, 3))
    env = _fake_env(torch.randn(512, 10))
    f(env, k_range=GYRO_LEGSPEED_RANGE)
    k = env._legspeed_gyro_k
    lo, hi = GYRO_LEGSPEED_RANGE
    assert k.shape == (512, 1) and float(k.min()) >= lo and float(k.max()) <= hi
    assert len(torch.unique(k)) == 512  # every env its own value
    assert float(k.min()) < lo + 0.01 and float(k.max()) > hi - 0.01  # spans the range (512 uniform draws)
    k0 = k.clone()
    for _ in range(5):
        f(env, k_range=GYRO_LEGSPEED_RANGE)
    assert torch.equal(env._legspeed_gyro_k, k0)  # drawn once, never redrawn per tick


def test_noise_sd_is_factor_times_rms_leg_speed():
    """Synthetic leg speeds with a known rms per env; the measured per-env noise sd over many ticks matches K x rms."""
    import torch
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE, make_legspeed_gyro
    torch.manual_seed(1)
    n, ticks = 256, 400
    rms = torch.linspace(0.5, 8.0, n)[:, None]  # rad/s, roughly walking leg speeds
    qd = rms * torch.ones(n, 10)  # constant per env: rms over the 10 joints is exactly `rms`
    qd[:, ::2] *= -1  # signs do not matter
    f = make_legspeed_gyro(lambda env: torch.zeros(env.num_envs, 3))
    env = _fake_env(qd)
    samples = torch.stack([f(env, k_range=GYRO_LEGSPEED_RANGE) for _ in range(ticks)])  # (ticks, n, 3)
    k = env._legspeed_gyro_k
    ratio = samples.std(dim=0) / (k * rms)  # per env, per axis
    se = 1.0 / (2 * (ticks - 1)) ** 0.5  # relative standard error of a sample sd
    assert abs(float(ratio.mean()) - 1.0) < 4 * se / (n * 3) ** 0.5 + 1e-3, float(ratio.mean())
    assert float((ratio - 1).abs().max()) < 5 * se, float((ratio - 1).abs().max())
    assert abs(float(samples.mean())) < 0.01  # zero-centred
    # the scalar k path is unchanged: one K for every env
    env2 = _fake_env(qd)
    s2 = torch.stack([f(env2, k=0.3) for _ in range(ticks)]).std(dim=0) / (0.3 * rms)
    assert float((s2 - 1).abs().max()) < 5 * se and "_legspeed_gyro_k" not in vars(env2)


def test_gyro_env_draws_factors_at_creation_on_cpu():
    """The real env: the factors exist once the env is built, lie in the range, differ across envs and stay fixed."""
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE
    cfg = sw.make_env_cfg("current", conditions="gyro")
    cfg.scene.num_envs = 8
    cfg.seed = 3
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        k = env.__dict__["_legspeed_gyro_k"]
        lo, hi = GYRO_LEGSPEED_RANGE
        assert k.shape == (8, 1) and float(k.min()) >= lo and float(k.max()) <= hi and len(torch.unique(k)) == 8
        k0 = k.clone()
        obs, _ = env.reset()
        term = env.action_manager.get_term("joint_pos")
        for _ in range(10):
            obs, *_ = env.step(torch.zeros(8, term.action_dim))
            assert torch.isfinite(obs["actor"]).all()
        assert torch.equal(env.__dict__["_legspeed_gyro_k"], k0)
    finally:
        env.close()


@pytest.mark.parametrize("cond", ["gyro", "robot"])
def test_stock_model_refuses_robot_conditions(cond):
    with pytest.raises(SystemExit):
        sw.make_env_cfg("stock", conditions=cond)
    with pytest.raises(SystemExit):
        sw.main(["--run-name", "x", "--model", "stock", "--conditions", cond, "--max-hours", "0.1"])
    import kaggle_job as kj
    with pytest.raises(SystemExit):
        kj.worker_args("scratch", "x", ["--model", "stock", "--conditions", cond])


def test_kaggle_preset_carries_the_conditions():
    import kaggle_job as kj
    args = kj.worker_args("scratch", "bw-s2-a", ["--model", "current", "--seed", "1", "--conditions", "gyro"])
    assert kj.flag_value(args, "--conditions") == "gyro"
    with pytest.raises(SystemExit):
        kj.worker_args("scratch", "x", ["--model", "current", "--conditions", "noisy"])


def _model_facts(cfg):
    m = cfg.scene.entities["robot"].spec_fn().compile()
    bodies = {m.body(i).name: (float(m.body_mass[i]), m.body_ipos[i].copy(), m.body_inertia[i].copy()) for i in range(m.nbody)}
    joints = {m.joint(i).name: m.jnt_range[i].copy() for i in range(m.njnt)}
    coll = sorted((m.geom(i).name, int(m.geom_contype[i]), int(m.geom_conaffinity[i])) for i in range(m.ngeom)
                  if m.geom_contype[i] or m.geom_conaffinity[i])
    return m, bodies, joints, coll


def test_current_model_differs_only_in_mass_and_ankle_stop(upstream):
    m0, b0, j0, c0 = _model_facts(upstream)
    m1, b1, j1, c1 = _model_facts(sw.make_env_cfg("current"))
    print(f"\n[current model] total mass {m0.body_mass.sum() * 1000:.1f} g (stock) -> {m1.body_mass.sum() * 1000:.1f} g")
    assert set(b0) == set(b1) and set(j0) == set(j1)
    changed_bodies = sorted(n for n in b0 if not (np.isclose(b0[n][0], b1[n][0]) and np.allclose(b0[n][1], b1[n][1])
                                                  and np.allclose(b0[n][2], b1[n][2])))
    for n in changed_bodies:
        print(f"  body {n}: {b0[n][0] * 1000:.1f} g -> {b1[n][0] * 1000:.1f} g")
    changed_joints = sorted(n for n in j0 if not np.allclose(j0[n], j1[n]))
    for n in changed_joints:
        print(f"  joint {n} range {np.round(j0[n], 3).tolist()} -> {np.round(j1[n], 3).tolist()}")
    assert changed_joints == ["left_ankle", "right_ankle"]
    assert np.allclose(j1["left_ankle"], [-1.36, 1.36])
    # collision set, actuators, joints, DOFs are the stock model's: the mass parts and the visual layer add no contact
    assert c0 == c1, "collision geoms differ"
    assert (m0.nu, m0.nq, m0.nv, m0.nbody) == (m1.nu, m1.nq, m1.nv, m1.nbody)
    assert abs(m1.body_mass.sum() * 1000 - 918.0) < 0.01  # 918 g weighed robot (cmu/model/current.json since 2026-11-23)


def test_agent_cfg_is_the_official_runner_cfg():
    cfg = sw.make_agent_cfg("bw-s1-cur-s1", seed=2, max_iters=4000, save_interval=250)
    diffs = cfg_diff(cfg, MicroduckRlCfg)
    _report("runner cfg", diffs)
    assert {p for p, _ in diffs} <= set(sw.AGENT_BOOKKEEPING)
    a = cfg.actor
    assert a.distribution_cfg["init_std"] == 1.0 and cfg.algorithm.learning_rate == 1e-3
    assert cfg.algorithm.entropy_coef == 0.01 and a.hidden_dims == (512, 256, 128)
    assert cfg.save_interval == MicroduckRlCfg.save_interval == 250 and cfg.num_steps_per_env == 24


def test_kaggle_preset_flags_exist_in_the_worker():
    import kaggle_job as kj
    args = kj.worker_args("scratch", "bw-s1-cur-s1", ["--model", "current", "--seed", "1"])
    assert kj.flag_value(args, "--max-iters") == "4000" and kj.flag_value(args, "--max-hours") == "5.8"
    with pytest.raises(SystemExit):
        kj.worker_args("scratch", "x", ["--seed", "1"])  # no --model
    with pytest.raises(SystemExit):
        kj.worker_args("scratch", "x", ["--model", "current", "--delta", "batreal"])  # a fine-tune flag
