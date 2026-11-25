"""CPU tests of --pack-levels (per-env pack levels, mdp.pack_mass_ramp.update_levels) and --cmd-progress-weight
(mdp.cmd_progress) of scratch_worker.py.

Run: python -m pytest cmu/finetune/kaggle/test_pack_levels.py -q   (the env test builds a 4-env CPU env)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent
PACK = "flightpack-v1-w918-20261124"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sw = _load("scratch_worker_under_test_levels", "scratch_worker.py")
tp = _load("test_pack_thrust_canon", "test_pack_thrust.py")


def bare_levels(levels, n_levels=11, gate=0.04, min_fwd_s=2.0):
    from mjlab_microduck.tasks import mdp
    r = object.__new__(mdp.pack_mass_ramp)
    n = len(levels)
    r.n_levels, r.gate_speed, r.min_fwd_s = n_levels, gate, min_fwd_s
    r.levels = torch.tensor(levels, dtype=torch.long)
    r.fwd_ticks = torch.zeros(n)
    r.fwd_speed = torch.zeros(n)
    r._gen = torch.Generator().manual_seed(0)
    r.reset_window()
    return r


def test_level_update_rule_hand_made_batch():
    dt = 0.02
    #        walk    fall+fwd  stand   slow-ok  no-fwd+fall  short-fwd  top-walk  bottom-stand
    lv = [4, 4, 4, 4, 4, 4, 10, 0]
    r = bare_levels(lv)
    secs = torch.tensor([5.0, 5.0, 5.0, 5.0, 0.0, 1.5, 5.0, 5.0])
    speed = torch.tensor([0.05, 0.08, 0.01, 0.03, 0.0, 0.10, 0.06, 0.0])
    ticks = secs / dt
    r.fwd_ticks[:] = ticks
    r.fwd_speed[:] = speed * ticks
    fell = torch.tensor([False, True, False, False, True, False, False, False])
    new = r.update_levels(torch.arange(8), fell, dt).tolist()
    assert new[0] == 5          # walked without falling at >= gate: up
    assert new[1] == 3          # fell under a forward command: down
    assert new[2] == 3          # stood (< gate / 2): down
    assert new[3] == 4          # between gate/2 and gate, no fall: keep
    assert new[4] == 4          # no forward command: keep, even after a fall
    assert new[5] == 4          # under 2 s of forward command: keep
    assert 0 <= new[6] <= 10    # promoted past the top: redrawn over all levels
    assert new[7] == 0          # demoted at the bottom: stays at 0
    assert float(r.fwd_ticks.abs().sum()) == 0.0 and float(r.fwd_speed.abs().sum()) == 0.0
    m = r.metrics()
    assert m["pack_eps_l0_3"] == 1 and m["pack_eps_l4_7"] == 6 and m["pack_eps_l8_10"] == 1
    assert m["pack_falls_per_ep_l4_7"] == pytest.approx(2 / 6, abs=1e-3)
    assert m["pack_speed_l4_7"] == pytest.approx((0.05 + 0.08 + 0.01 + 0.03) / 4, abs=1e-4)  # 4 with >= 2 s
    assert m["pack_speed_l8_10"] == pytest.approx(0.06, abs=1e-4)


def _episode(r, env_lv, secs, speed, fell, dt=0.02):
    """one hand-made episode end for one env per entry; returns the new levels"""
    ticks = torch.tensor(secs) / dt
    r.fwd_ticks[:] = ticks
    r.fwd_speed[:] = torch.tensor(speed) * ticks
    r.levels[:] = torch.tensor(env_lv)
    return r.update_levels(torch.arange(len(env_lv)), torch.tensor(fell), dt).tolist()


def test_exploit_lunge_and_fall_before_min_fwd_is_not_demoted():
    """What the rule does now (review point, no fix yet): a fall with 1.9 s of forward command, lunging at 0.3 m/s,
    keeps its level, so falling early costs no level."""
    r = bare_levels([6, 6])
    new = _episode(r, [6, 6], [1.9, 2.0], [0.30, 0.30], [True, True])
    assert new == [6, 5]    # 1.9 s: kept; at exactly 2.0 s the fall demotes


def test_exploit_creep_at_gate_is_promoted():
    """A creep at exactly the gate (0.04 m/s, 16 % of the 0.25 press) without a fall is promoted; 0.0399 keeps."""
    r = bare_levels([3, 3, 3])
    new = _episode(r, [3, 3, 3], [5.0, 5.0, 5.0], [0.04, 0.0399, 0.0199], [False, False, False])
    assert new == [4, 3, 2]


def test_top_cycle_always_walking_env():
    """An env that always walks without a fall climbs one level per episode to 10 and is then redrawn uniformly over
    0..10, so it never stays at the top: over a long run its time at levels 9-10 is about 2 / (expected climb length)."""
    r = bare_levels([0])
    lv, at_top, eps = 0, 0, 20000
    for _ in range(eps):
        lv = _episode(r, [lv], [5.0], [0.06], [False])[0]
        at_top += lv >= 9
    share = at_top / eps
    # renewal: from a uniform redraw u, the climb back to 10 takes 11 - u episodes (mean 6); levels 9 and 10 take 2
    assert 0.25 < share < 0.42, share


def test_top_cycle_fall_at_top_then_walk():
    """At 10: a fall (>= 2 s forward) demotes to 9, the next walk promotes to 10, the walk after that redraws."""
    r = bare_levels([10])
    a = _episode(r, [10], [5.0], [0.06], [True])[0]
    b = _episode(r, [a], [5.0], [0.06], [False])[0]
    c = _episode(r, [b], [5.0], [0.06], [False])[0]
    assert (a, b) == (9, 10) and 0 <= c <= 10


def test_metric_keys_top_share_is_levels_9_10():
    r = bare_levels([8, 9, 10, 10])
    m = r.metrics()
    assert m["pack_level_share_top"] == pytest.approx(0.75)   # levels >= n_levels - 2 = 9
    assert m["pack_level_mean"] == pytest.approx(9.25)
    assert {"pack_speed_l8_10", "pack_falls_per_ep_l8_10", "pack_eps_l8_10"} <= set(m)


def test_top_redraw_covers_levels():
    r = bare_levels([10] * 2000)
    r.fwd_ticks[:] = 250.0
    r.fwd_speed[:] = 250.0 * 0.06
    new = r.update_levels(torch.arange(2000), torch.zeros(2000, dtype=torch.bool), 0.02)
    assert set(new.tolist()) == set(range(11))


def test_track_counts_forward_ticks_only():
    r = bare_levels([0, 0, 0])
    cmd = torch.tensor([[0.25, 0.0], [0.05, 0.0], [0.0, 0.3]])
    vel = torch.tensor([[0.10, 0.05], [0.10, 0.0], [0.0, 0.3]])
    r.track(None, cmd, vel)
    assert r.fwd_ticks.tolist() == [1.0, 0.0, 0.0]
    assert r.fwd_speed[0] == pytest.approx(0.10)


def fake_env(cmd, vel):
    data = SimpleNamespace(root_link_lin_vel_b=torch.tensor(vel, dtype=torch.float32))
    return SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda name: torch.tensor(cmd, dtype=torch.float32)),
                           scene={"robot": SimpleNamespace(data=data)})


def test_cmd_progress_values():
    from mjlab_microduck.tasks import mdp
    cmd = [[0.25, 0.0, 0.0]] * 5 + [[0.05, 0.0, 0.0], [0.0, 0.2, 0.5]]
    vel = [[0.0, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [-0.1, 0.0, 0.0],
           [0.05, 0.0, 0.0], [0.0, 0.1, 0.0]]
    v = mdp.cmd_progress(fake_env(cmd, vel)).tolist()
    assert v[0] == pytest.approx(0.0)   # standing
    assert v[1] == pytest.approx(0.5)   # half the commanded speed
    assert v[2] == pytest.approx(1.0)   # full
    assert v[3] == pytest.approx(1.0)   # faster: clipped
    assert v[4] == pytest.approx(0.0)   # backwards: clipped
    assert v[5] == pytest.approx(0.0)   # command <= 0.1 m/s: off
    assert v[6] == pytest.approx(0.5)   # sideways command, half speed along it


def _gate(n_envs, window_s=1.0, dt=0.02):
    from mjlab_microduck.tasks import mdp
    cfg = SimpleNamespace(params=dict(gate_window_s=window_s))
    env = SimpleNamespace(step_dt=dt, num_envs=n_envs, device="cpu")
    return mdp.upright_progress_gate(cfg, env)


def test_upright_gate_values_over_a_window():
    """50 ticks (1.0 s at 0.02 s) of: walk 0.2, creep 0.04, stand, zero command, walk but negative inner value,
    sideways command at 0.05 along it."""
    g = _gate(6)
    assert g.n == 50
    cmd = [[0.25, 0, 0]] * 5 + [[0.0, 0.2, 0]]
    cmd[3] = [0.0, 0.0, 0.0]
    vel = [[0.2, 0, 0], [0.04, 0, 0], [0.0, 0, 0], [0.0, 0, 0], [0.2, 0, 0], [0.0, 0.05, 0]]
    env = fake_env(cmd, vel)
    inner_vals = torch.tensor([2.0, 2.0, 2.0, 2.0, -1.0, 2.0])
    for _ in range(50):
        out = g(env, gate_inner=lambda e: inner_vals.clone(), gate_window_s=1.0, gate_v_ref=0.10)
    assert out.tolist() == pytest.approx([2.0, 0.8, 0.0, 2.0, -1.0, 1.0])  # 0.04/0.10 = 0.4; 0.05/0.10 = 0.5


def test_upright_gate_partial_window_and_reset():
    g = _gate(2)
    env = fake_env([[0.25, 0, 0]] * 2, [[0.2, 0, 0]] * 2)
    one = lambda e: torch.ones(2)
    for _ in range(10):  # 10 of 50 ticks at 0.2: mean 0.04 -> g 0.4
        out = g(env, gate_inner=one, gate_v_ref=0.10)
    assert out.tolist() == pytest.approx([0.4, 0.4])
    g.reset(torch.tensor([1]))
    out = g(env, gate_inner=one, gate_v_ref=0.10)  # env 1: one tick of the new episode -> 0.2 / 50 / 0.1 = 0.04
    assert out.tolist() == pytest.approx([0.44, 0.04])


def test_upright_gate_off_by_default_and_wraps_when_on():
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
    base = sw.make_env_cfg(PACK, conditions="robot")
    assert base.rewards["upright"].func is not mdp.upright_progress_gate
    old_w, old_std = base.rewards["upright"].weight, base.rewards["upright"].params["std"]
    rec = apply_upright_progress_gate(base, window_s=1.0, v_ref=0.10)
    t = base.rewards["upright"]
    assert t.func is mdp.upright_progress_gate and t.weight == old_w and t.params["std"] == old_std
    assert t.params["gate_v_ref"] == 0.10 and t.params["gate_window_s"] == 1.0 and rec["terms"][0]["term"] == "upright"


def test_upright_gate_flags_validate():
    with pytest.raises(SystemExit):
        sw.main(["--run-name", "x", "--model", PACK, "--max-hours", "0.1", "--upright-gate-vref", "0.2"])


def test_both_off_leave_cfg_identical():
    from mjlab_microduck.tasks.cmu_asbuilt import apply_cmd_progress, apply_pack_options
    base = sw.make_env_cfg(PACK, conditions="robot")
    other = sw.make_env_cfg(PACK, conditions="robot")
    assert apply_pack_options(other, PACK, levels=0) == {}
    assert apply_cmd_progress(other, 0.0) == {}
    assert json.dumps(tp.canon(base), sort_keys=True, default=str) == json.dumps(tp.canon(other), sort_keys=True,
                                                                                 default=str)


def test_on_adds_terms():
    from mjlab_microduck.tasks.cmu_asbuilt import apply_cmd_progress, apply_pack_options
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    rec = apply_pack_options(cfg, PACK, levels=11)
    assert rec["mass_levels"]["fractions"][5] == 0.5 and rec["mass_levels"]["level_init"] == [0, 5]
    assert cfg.events["pack_mass_ramp"].params["levels"] == 11 and cfg.events["pack_level_track"].mode == "step"
    apply_cmd_progress(cfg, 2.0)
    assert cfg.rewards["cmd_progress"].weight == 2.0
    with pytest.raises(AssertionError):
        apply_pack_options(sw.make_env_cfg(PACK, conditions="robot"), PACK, levels=11, ramp_steps=24)


def test_levels_env_masses():
    """4-env CPU env: level 0 has the base robot's mass, level 10 the full pack, level 5 half way."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.cmu_asbuilt import apply_cmd_progress, apply_pack_options
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    apply_pack_options(cfg, PACK, levels=11)
    apply_cmd_progress(cfg, 2.0)
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 4
    cfg.seed = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    ramp = env.event_manager.get_term_cfg("pack_mass_ramp").func
    ramp.levels[:] = torch.tensor([0, 5, 10, 10])
    env.reset()
    mass = (env.sim.model.body_mass.sum(-1) * 1000).flatten().tolist()  # grams per env
    # startup mass DR scales each env; compare against the 918 g base and 1823 g pack within the DR spread
    assert mass[0] < 1100 and mass[2] > 1600 and mass[3] > 1600
    assert mass[0] < mass[1] < mass[2]
    for _ in range(5):
        env.step(torch.zeros(4, 14))
    assert float(ramp.fwd_ticks.sum()) >= 0.0
    assert "cmd_progress" in env.reward_manager.active_terms
    env.close()


def test_upright_gate_env_steps():
    """4-env CPU env, gate on, fixed full pack (levels and ramp off): the wrapped upright term builds (mjlab's class
    term inside the gate), steps finite, and under a walk command a standing robot's upright reward is gated to 0."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    apply_upright_progress_gate(cfg)
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 4
    cfg.seed = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    term = env.reward_manager.get_term_cfg("upright")
    assert isinstance(term.func, mdp.upright_progress_gate) and term.func.inner is not None
    env.reset()
    tw = env.command_manager.get_term("twist")
    i = env.reward_manager.active_terms.index("upright")
    for _ in range(5):
        tw.vel_command_b[:] = 0.0
        tw.vel_command_b[:2, 0] = 0.25  # envs 0, 1: walk command; envs 2, 3: zero command
        env.step(torch.zeros(4, 14))
        r = env.reward_manager._step_reward[:, i]
        assert torch.isfinite(r).all()
    g = term.func.last_g
    assert float(g[2]) == 1.0 and float(g[3]) == 1.0
    assert float(g[0]) < 0.5 and float(g[1]) < 0.5  # 5 ticks of a 50-tick window, zero action: barely moving
    env.close()


ON4 = ("upright", "track_angular_velocity", "pose", "head_pose_tracking")


def test_on4_gates_four_terms_only():
    """--progress-gate-terms with the four terms: those four are wrapped with the upright gate's exact params; every
    other term (track_linear_velocity and air_time included) is identical to the ungated cfg."""
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
    base = sw.make_env_cfg(PACK, conditions="robot")
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    rec = apply_upright_progress_gate(cfg, window_s=1.0, v_ref=0.10, terms=ON4)
    assert [t["term"] for t in rec["terms"]] == list(ON4) and rec["window_s"] == 1.0 and rec["v_ref"] == 0.10
    for name in ON4:
        t, b = cfg.rewards[name], base.rewards[name]
        assert t.func is mdp.upright_progress_gate and t.params["gate_inner"] is b.func and t.weight == b.weight
        assert {k: v for k, v in t.params.items() if not k.startswith("gate_")} == b.params
        assert (t.params["gate_window_s"], t.params["gate_v_ref"], t.params["gate_min_speed"],
                t.params["gate_command"]) == (1.0, 0.10, 0.1, "twist")
    for name in set(base.rewards) - set(ON4):
        assert cfg.rewards[name].func is base.rewards[name].func, name
        assert json.dumps(tp.canon(cfg.rewards[name]), sort_keys=True, default=str) == \
            json.dumps(tp.canon(base.rewards[name]), sort_keys=True, default=str), name
    assert "track_linear_velocity" not in ON4 and "air_time" not in ON4


def test_on4_zero_speed_zero_and_zero_command_unchanged():
    """Each of the four terms' gate params on a 1 s window: forward command 0.25 with zero speed -> positive value 0;
    zero command -> unchanged; negative values unchanged."""
    from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    apply_upright_progress_gate(cfg, terms=ON4)
    env = fake_env([[0.25, 0, 0], [0.0, 0, 0], [0.25, 0, 0]], [[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]])
    vals = torch.tensor([1.5, 1.5, -0.5])
    for name in ON4:
        p = {k: v for k, v in cfg.rewards[name].params.items() if k.startswith("gate_") and k != "gate_inner"}
        g = _gate(3, window_s=p["gate_window_s"])
        for _ in range(60):
            out = g(env, gate_inner=lambda e: vals.clone(), **p)
        assert out.tolist() == pytest.approx([0.0, 1.5, -0.5]), name


def test_on4_flags():
    with pytest.raises(SystemExit):  # one gate flag at a time
        sw.main(["--run-name", "x", "--model", PACK, "--max-hours", "0.1", "--upright-progress-gate",
                 "--progress-gate-terms", "upright"])
    with pytest.raises(SystemExit):  # window / v_ref still need a gate flag
        sw.main(["--run-name", "x", "--model", PACK, "--max-hours", "0.1", "--upright-gate-window", "2.0"])
    with pytest.raises(SystemExit):  # empty list
        sw.main(["--run-name", "x", "--model", PACK, "--max-hours", "0.1", "--progress-gate-terms", ","])


def test_no_gate_flag_cfg_unchanged():
    from mjlab_microduck.tasks import mdp
    base = sw.make_env_cfg(PACK, conditions="robot")
    assert not any(t.func is mdp.upright_progress_gate for t in base.rewards.values())
    assert json.dumps(tp.canon(base), sort_keys=True, default=str) == json.dumps(
        tp.canon(sw.make_env_cfg(PACK, conditions="robot")), sort_keys=True, default=str)


def test_on4_env_steps_gate_equals_inner_times_g():
    """4-env CPU env with the four terms gated: each builds (pose and the trackers may be class terms), steps finite;
    on every tick each gated term's step reward = weight x (inner x g where inner > 0, else inner), g = 1 at zero
    command (envs 2, 3), g < 0.5 under a walk command after 5 ticks of zero action (envs 0, 1)."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    apply_upright_progress_gate(cfg, terms=ON4)
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 4
    cfg.seed = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    tw = env.command_manager.get_term("twist")
    rm = env.reward_manager
    for _ in range(5):
        tw.vel_command_b[:] = 0.0
        tw.vel_command_b[:2, 0] = 0.25
        env.step(torch.zeros(4, 14))
        for name in ON4:
            term = rm.get_term_cfg(name)
            assert isinstance(term.func, mdp.upright_progress_gate)
            inner, g = term.func.last_inner, term.func.last_g
            want = term.weight * torch.where(inner > 0, inner * g, inner)
            got = rm._step_reward[:, rm.active_terms.index(name)]
            assert torch.isfinite(got).all() and torch.allclose(got, want, atol=1e-5), name
            assert float(g[2]) == 1.0 and float(g[3]) == 1.0
    for name in ON4:
        g = rm.get_term_cfg(name).func.last_g
        assert float(g[0]) < 0.5 and float(g[1]) < 0.5, name
    env.close()


def test_fall_cost_off_leaves_cfg_identical_and_flag_validates():
    base = sw.make_env_cfg(PACK, conditions="robot")
    other = sw.make_env_cfg(PACK, conditions="robot")
    assert sw.apply_fall_cost(other, 0.0) == {} and "fall_cost" not in other.rewards
    assert json.dumps(tp.canon(base), sort_keys=True, default=str) == json.dumps(tp.canon(other), sort_keys=True,
                                                                                 default=str)
    rec = sw.apply_fall_cost(other, 10.0)
    assert other.rewards["fall_cost"].weight == -10.0 and rec["term"] == "fell_over"
    for name in base.rewards:  # every other term unchanged
        assert json.dumps(tp.canon(other.rewards[name]), sort_keys=True, default=str) == \
            json.dumps(tp.canon(base.rewards[name]), sort_keys=True, default=str), name
    with pytest.raises(SystemExit):
        sw.main(["--run-name", "x", "--model", PACK, "--max-hours", "0.1", "--fall-cost", "-1"])


def test_fall_cost_reaches_ppo_as_minus_c_once_per_fall():
    """4-env CPU env with --fall-cost 7 and the four-term gate: the fell_over termination is forced for envs 0 and 1 on
    one tick, env 1 also times out on that tick. The reward the env returns to PPO (env.step's reward, the same tensor
    RslRlVecEnvWrapper passes to the runner) carries exactly -7 from fall_cost for env 0 and 0 for envs 1 to 3; on the
    other ticks fall_cost is 0 everywhere."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
    C = 7.0
    cfg = sw.make_env_cfg(PACK, conditions="robot")
    apply_upright_progress_gate(cfg, terms=ON4)
    sw.apply_fall_cost(cfg, C)
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = 4
    cfg.seed = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    rm, tm = env.reward_manager, env.termination_manager
    i = rm.active_terms.index("fall_cost")
    fo, to = tm.get_term_cfg("fell_over"), tm.get_term_cfg("time_out")
    real_fo, real_to = fo.func, to.func
    for k in range(4):
        force = k == 2
        fo.func = (lambda e, **kw: torch.tensor([True, True, False, False])) if force else real_fo
        to.func = (lambda e, **kw: torch.tensor([False, True, False, False])) if force else real_to
        _, rew, *_ = env.step(torch.zeros(4, 14))
        contrib = rm._step_reward[:, i] * env.step_dt  # what fall_cost added to this tick's reward
        assert torch.allclose(rew, rm._step_reward.sum(1) * env.step_dt, atol=1e-5)  # returned reward = sum x dt
        want = torch.tensor([-C, 0.0, 0.0, 0.0]) if force else torch.zeros(4)
        if not force:
            want = torch.where(tm.get_term("fell_over") & ~tm.time_outs, torch.full((4,), -C), want)
        assert torch.allclose(contrib, want, atol=1e-4), (k, contrib)
    env.close()
