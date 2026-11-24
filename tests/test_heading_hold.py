"""scratch_worker --heading-hold (src/mjlab_microduck/heading_hold.py): the brain's straight-walk heading hold in training.

--heading-hold 0 leaves the cfg exactly as before; --heading-hold 0.5 makes heading-hold envs whose yaw command equals
clip(K * wrap(target - heading), -1, 1) of the measured heading at every control step, with the target the env's own
heading at the resample; the default gain is the fixed K = 3.0, a --heading-hold-gain range draws K per env. CPU only.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.heading_hold import HeadingHoldVelocityCommand, HeadingHoldVelocityCommandCfg
from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmu/finetune/kaggle"))
import scratch_worker as sw  # noqa: E402
from test_scratch_worker_cfg import cfg_diff  # noqa: E402

TWIST = "commands['twist']"


def test_share_zero_is_the_recipe():
    for cond in ("official", "robot"):
        a, b = sw.make_env_cfg("current", conditions=cond, heading_hold=0.0), sw.make_env_cfg("current", conditions=cond)
        assert cfg_diff(a, b) == [] and type(a.commands["twist"]) is VelocityCommandCommandOnlyCfg
        assert "_heading_hold_record" not in a.__dict__


def test_share_changes_only_the_twist_command():
    a, b = sw.make_env_cfg("current", conditions="robot", heading_hold=0.5), sw.make_env_cfg("current", conditions="robot")
    diffs = {p for p, _ in cfg_diff(a, b)}
    assert diffs and all(p.startswith(TWIST) for p in diffs), diffs
    c = a.commands["twist"]
    assert type(c) is HeadingHoldVelocityCommandCfg and c.rel_heading_hold_envs == 0.5
    assert c.heading_hold_gain == (3.0, 3.0) and c.heading_hold_max == 1.0
    old = b.commands["twist"]
    for f in ("ranges", "rel_forward_envs", "rel_standing_envs", "rel_turn_in_place_envs", "resampling_time_range",
              "heading_command", "rel_heading_envs"):
        assert getattr(c, f) == getattr(old, f), f
    rec = a.__dict__["_heading_hold_record"]
    assert rec["heading_hold"]["expected_share_of_all_envs"] == pytest.approx(0.5 * old.rel_forward_envs)
    assert rec["press_speed"]["share_of_bucket"] == 0.0


def test_worker_flag_checks():
    for bad in (["--heading-hold", "1.5"], ["--heading-hold-gain", "4", "3", "--heading-hold", "0.5"],
                ["--heading-hold-gain", "3", "6"]):
        with pytest.raises(SystemExit):
            sw.main(["--run-name", "x", "--model", "current", "--max-hours", "0.1"] + bad)


def _env(share, gain, n=48, seed=7):
    from mjlab.envs import ManagerBasedRlEnv
    cfg = sw.make_env_cfg("current", conditions="official", heading_hold=share, heading_hold_gain=gain)
    cfg.scene.num_envs = n
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def _force_some_hold_envs(term, n, want=4, tries=200):
    """Resample every env until a few are heading-hold envs (a forward zero-turn bucket draw is 0.2 x share)."""
    ids = torch.arange(n)
    for _ in range(tries):
        term._resample_command(ids)  # noqa: SLF001
        if int(term.is_heading_hold_env.sum()) >= want:
            return
    raise AssertionError("no heading-hold envs after resampling")


def _run(share, gain, steps=40):
    env = _env(share, gain)
    try:
        term = env.command_manager.get_term("twist")
        n = env.num_envs
        env.reset()
        rows = []
        if share > 0:
            _force_some_hold_envs(term, n)
            hold0 = term.is_heading_hold_env.clone()
            # the error starts at zero: target = own heading at the resample
            err0 = torch.atan2(torch.sin(term.heading_hold_target - term.robot.data.heading_w),
                               torch.cos(term.heading_hold_target - term.robot.data.heading_w))
            assert float(err0[hold0].abs().max()) < 1e-6
            assert (term.vel_command_b[hold0, 2] == 0).all() and (term.vel_command_b[hold0, 0] > 0).all()
            assert (term.vel_command_b[hold0, 1] == 0).all()
        act = env.action_manager.get_term("joint_pos").action_dim
        for t in range(steps):
            g = torch.Generator().manual_seed(t)
            env.step(0.3 * torch.randn(n, act, generator=g))  # random actions make the heading drift
            rows.append(dict(cmd=term.vel_command_b.clone(), heading=term.robot.data.heading_w.clone(),
                             hold=term.is_heading_hold_env.clone() if share > 0 else None,
                             target=term.heading_hold_target.clone() if share > 0 else None,
                             gain=term.heading_hold_gain.clone() if share > 0 else None))
        return term, rows
    finally:
        env.close()


@pytest.mark.parametrize("gain", [(3.0, 3.0), (3.0, 6.0)])
def test_heading_hold_command_is_the_brain_law(gain):
    term, rows = _run(0.5, gain)
    assert isinstance(term, HeadingHoldVelocityCommand)
    worst, moved, saw_clip = 0.0, 0.0, False
    for r in rows:
        h = r["hold"]
        if not h.any():
            continue
        err = torch.atan2(torch.sin(r["target"] - r["heading"]), torch.cos(r["target"] - r["heading"]))[h]
        k = r["gain"][h]
        if gain[0] == gain[1]:
            assert torch.equal(k, torch.full_like(k, 3.0))  # the default is the fixed brain gain, exactly
            want = torch.clip(3.0 * err, -1.0, 1.0)
        else:
            assert float(k.min()) >= gain[0] and float(k.max()) <= gain[1]
            want = torch.clip(k * err, -1.0, 1.0)
        worst = max(worst, float((r["cmd"][h, 2] - want).abs().max()))
        moved = max(moved, float(err.abs().max()))
        saw_clip = saw_clip or bool((r["cmd"][h, 2].abs() == 1.0).any())
        assert float(r["cmd"][h, 2].abs().max()) <= 1.0
    assert worst < 1e-5, worst  # float32: mjlab's wrap_to_pi against the atan2 form here
    assert moved > 0.01, f"heading never drifted ({moved} rad): the test did not exercise the feedback"
    print(f"\n[heading hold {gain}] max |cmd - clip(K err)| {worst:.2e}, max |err| {moved:.3f} rad, clip seen {saw_clip}")


def test_gain_range_draws_per_env():
    env = _env(0.5, (3.0, 6.0), n=256)
    try:
        term = env.command_manager.get_term("twist")
        term._resample_command(torch.arange(256))  # noqa: SLF001
        k = term.heading_hold_gain
        assert float(k.min()) >= 3.0 and float(k.max()) <= 6.0 and len(torch.unique(k)) > 200
        k0 = k.clone()
        term._resample_command(torch.arange(256))  # noqa: SLF001
        assert not torch.equal(term.heading_hold_gain, k0)  # redrawn at each resample
    finally:
        env.close()


def test_all_defaults_leave_the_cfg_untouched():
    base = sw.make_env_cfg("current", conditions="robot")
    for kw in (dict(), dict(heading_hold=0.0, press_speed_share=0.0, heading_hold_law="p", heading_hold_trim=0.0,
                         press_speed=0.25, heading_hold_gain=(3.0, 3.0))):
        c = sw.make_env_cfg("current", conditions="robot", **kw)
        assert cfg_diff(c, base) == [] and type(c.commands["twist"]) is VelocityCommandCommandOnlyCfg


# ---- law brain against a pure-Python transcription of walk_brain.py (Brain.step, straight walk) ----

def _brain_reference(headings, h0, age, kp, trim, dt=0.02):
    """walk_brain.py lines 1113-1139 for vx > 0, vy = vtheta = 0: the held heading is set at the resample, the IMU yaw
    the brain reads is `age` ticks old. Returns the turn input per tick."""
    ki, i_max, turn_max = kp * 2.0 / 3.0, 1.5, 1.0
    hist = [h0] * 3
    target, i, out = hist[age], 0.0, []
    for h in headings:
        hist = [h] + hist[:2]
        yaw = hist[age]
        err = math.atan2(math.sin(target - yaw), math.cos(target - yaw))
        i_new = min(i_max, max(-i_max, i + err * dt))
        u_new = trim + kp * err + ki * i_new
        if abs(u_new) < turn_max or abs(i_new) < abs(i):
            i = i_new
        out.append(min(turn_max, max(-turn_max, trim + ki * i + kp * err)))
    return out


class _FakeData:
    def __init__(self, n):
        self.heading_w = torch.zeros(n)
        self.root_link_lin_vel_b = torch.zeros(n, 3)
        self.root_link_ang_vel_b = torch.zeros(n, 3)
        self.root_link_quat_w = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)


def _recorded_trace(steps=120):
    """Heading of one env of a real CPU env under random actions (it drifts and turns)."""
    env = _env(0.0, (3.0, 3.0), n=4, seed=11)
    try:
        env.reset()
        act = env.action_manager.get_term("joint_pos").action_dim
        tr = []
        for t in range(steps):
            env.step(0.4 * torch.randn(4, act, generator=torch.Generator().manual_seed(100 + t)))
            tr.append(float(env.scene["robot"].data.heading_w[0]))
        return tr
    finally:
        env.close()


def test_brain_law_matches_the_brain_transcription():
    rec = _recorded_trace()
    # recorded trace, then a held 0.9 rad offset (saturates: 3 x 0.9 > 1), then a slow return through the band
    tail = [rec[-1] + 0.9] * 60 + [rec[-1] + 0.9 * (1 - k / 80) for k in range(80)]
    trace = rec + tail
    assert len(trace) >= 200
    n = 32
    from mjlab.envs import ManagerBasedRlEnv
    cfg = sw.make_env_cfg("current", heading_hold=1.0, heading_hold_gain=(3.0, 4.5), heading_hold_law="brain",
                          heading_hold_trim=0.35)
    cfg.scene.num_envs = n
    cfg.seed = 5
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        term = env.command_manager.get_term("twist")
        fake = _FakeData(n)
        h0 = trace[0] - 0.01
        term.robot = type("R", (), {"data": fake})()  # the command reads this heading; nothing else is stepped
        fake.heading_w[:] = h0
        _force_some_hold_envs(term, n, want=4)  # resamples on the fake heading: target = h0 for every hold env
        ids = term.is_heading_hold_env.nonzero().flatten()
        assert len(ids) >= 3
        term.time_left[:] = 1e9  # no further resample during the trace
        got = []
        for h in trace:
            fake.heading_w[:] = h
            term.compute(0.02)
            got.append(term.vel_command_b[ids, 2].clone())
        got = torch.stack(got)  # (T, len(ids))
        worst, sat = 0.0, 0
        for c, e in enumerate(ids.tolist()):
            ref = _brain_reference(trace, h0, int(term.heading_hold_age[e]), float(term.heading_hold_gain[e]),
                                   float(term.heading_hold_trim[e]))
            d = (got[:, c].double() - torch.tensor(ref, dtype=torch.float64)).abs().max()
            worst = max(worst, float(d))
            sat += sum(abs(x) >= 1.0 for x in ref)
        assert worst < 1e-5, worst
        assert sat > 50, "the trace never saturated the turn input"
        ages = {int(term.heading_hold_age[e]) for e in ids.tolist()}
        trims = term.heading_hold_trim[ids]
        assert float(trims.abs().max()) <= 0.35 and len(torch.unique(trims)) == len(ids)
        print(f"\n[brain law] {len(ids)} envs, {len(trace)} ticks, max |sim - transcription| {worst:.2e}, "
              f"saturated ticks {sat}, ages {sorted(ages)}")
    finally:
        env.close()


def test_law_p_unchanged_by_the_new_fields():
    """Law p (and press share 0) draws the same random numbers and gives the same commands as before the new options:
    a cfg built with every new field at its default equals the old two-field cfg."""
    a = sw.make_env_cfg("current", heading_hold=0.5)
    b = sw.make_env_cfg("current", heading_hold=0.5, heading_hold_law="p", heading_hold_trim=0.0,
                        press_speed_share=0.0, press_speed=0.25)
    assert cfg_diff(a, b) == []
    c = a.commands["twist"]
    assert (c.heading_hold_law, c.heading_hold_trim, c.rel_press_speed_envs) == ("p", 0.0, 0.0)


@pytest.mark.parametrize("hh", [0.0, 1.0])
def test_press_speed_share(hh):
    from mjlab.envs import ManagerBasedRlEnv
    n, q = 4096, 0.5
    cfg = sw.make_env_cfg("current", heading_hold=hh, press_speed_share=q,
                          **(dict(heading_hold_law="brain", heading_hold_trim=0.35) if hh else {}))
    cfg.scene.num_envs = n
    cfg.seed = 9
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        term = env.command_manager.get_term("twist")
        term._resample_command(torch.arange(n))  # noqa: SLF001
        cmd = term.vel_command_b
        bucket = term.is_forward_env & ~term.is_standing_env & (cmd[:, 0] > 0) & (cmd[:, 1] == 0)
        if not hh:
            bucket &= cmd[:, 2] == 0
        press = term.is_press_speed_env
        nb = int(bucket.sum())
        frac = float(press[bucket].float().mean())
        sd = math.sqrt(q * (1 - q) / nb)
        assert abs(frac - q) < 3 * sd, (frac, nb)
        assert not press[~bucket].any()
        assert torch.all(cmd[press, 0] == 0.25) and torch.all(cmd[press, 1] == 0)
        assert torch.all(cmd[bucket & ~press, 0] >= 0.3)  # the rest keep mjlab's clamp(min=0.3)
        if not hh:
            assert torch.all(cmd[press, 2] == 0)
        else:  # independent draws: heading hold appears on press and non-press envs alike
            h = term.is_heading_hold_env
            assert h[press].any() and h[bucket & ~press].any()
            both = float((h & press)[bucket].float().mean())
            assert abs(both - q * 1.0) < 3 * sd
        print(f"\n[press speed hh={hh}] bucket {nb} of {n}, at 0.25 m/s {frac:.3f} (3 sd {3 * sd:.3f})")
    finally:
        env.close()


def test_share_zero_env_uses_the_recipe_command():
    term, rows = _run(0.0, (3.0, 3.0), steps=5)
    assert type(term) is VelocityCommandCommandOnly
    assert not hasattr(term, "is_heading_hold_env")


def test_target_is_the_new_heading_after_an_episode_reset():
    """mjlab resamples the command inside _reset_idx, before sim.forward(): xquat (heading_w) still holds the heading
    of the episode that just ended, while the reset event has written a new random yaw. The target and the heading
    history must be taken from the heading after the reset, so a fresh hold env starts with error 0 and wz = trim.
    Found 2026-11-23: the arm B CPU smoke showed 77 % of hold ticks at the clamp."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import wrap_to_pi
    n = 256
    for law, trim in (("brain", 0.35), ("p", 0.0)):
        cfg = sw.make_env_cfg("current", heading_hold=1.0, heading_hold_gain=(3.0, 4.5), heading_hold_law=law,
                              heading_hold_trim=trim)
        cfg.scene.num_envs = n
        cfg.seed = 3
        env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
        try:
            term = env.command_manager.get_term("twist")
            act = env.action_manager.get_term("joint_pos").action_dim
            env.reset()
            seen, spread = 0, 0.0
            for r in range(3):
                for t in range(3):  # a few steps so the old episode's heading differs from the next reset's
                    env.step(torch.zeros(n, act))
                before = env.scene["robot"].data.heading_w.clone()
                env.reset()
                h = env.scene["robot"].data.heading_w
                spread = max(spread, float(wrap_to_pi(h - before).abs().max()))
                hold = term.is_heading_hold_env
                seen += int(hold.sum())
                err = wrap_to_pi(term.heading_hold_target - h)[hold]
                assert float(err.abs().max()) < 1e-4, (law, r, float(err.abs().max()))
                want = term.heading_hold_trim[hold] if law == "brain" else torch.zeros(int(hold.sum()))
                assert torch.allclose(term.vel_command_b[hold, 2], want, atol=1e-3), (law, r)
                # and the first real step starts from that target: no clamp from a stale history
                env.step(torch.zeros(n, act))
                hold2 = term.is_heading_hold_env & hold & (env.episode_length_buf == 1)
                assert float(term.heading_hold_error[hold2].abs().max()) < 0.3, law
            assert seen >= 30 and spread > 1.0, (seen, spread)  # the resets did move the heading
        finally:
            env.close()
