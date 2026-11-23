"""Tests for the suite's floor-friction cases and brain_sim_check.py's --floor-mu and fall_dir (CPU only).

  uv run --no-sync --with pytest pytest cmu/simcheck -q
"""
import json
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_suite  # noqa: E402


def condition(tmp_path, **facts):
    """The fork's robot-condition.json with extra facts."""
    c = json.load(open(sim_suite.DEFAULT_CONDITION))
    c["facts"].update({k: {"value": v} for k, v in facts.items()})
    p = tmp_path / "condition.json"
    p.write_text(json.dumps(c))
    return str(p)


def brain_sim_check():
    pytest.importorskip("mujoco")
    sys.path.insert(0, os.path.join(sim_suite.CMU, "mac"))
    try:
        import brain_sim_check as b
    except (ImportError, SystemExit, OSError):
        pytest.skip("as-built model not loadable here")
    return b


def test_floor_mu_cases_run_on_the_estimates_as_stress(tmp_path):
    cases = sim_suite.floor_mu_cases(condition(tmp_path))
    assert cases == [("stress", "floor_mu_bare", ["--vx", "0.25", "--floor-mu", "0.3"]),
                     ("stress", "floor_mu_sole", ["--vx", "0.25", "--floor-mu", "0.7"])]


def test_a_measured_floor_mu_makes_its_case_real(tmp_path):
    cases = dict((name, (cls, flags)) for cls, name, flags in sim_suite.floor_mu_cases(condition(tmp_path, floor_mu_bare=0.42)))
    assert cases["floor_mu_bare"] == ("real", ["--vx", "0.25", "--floor-mu", "0.42"])
    assert cases["floor_mu_sole"][0] == "stress"


def test_the_first_eight_cases_are_unchanged():
    assert [(cls, name) for cls, name, _ in sim_suite.CASES] == [
        ("real", "nominal"), ("real", "vx0.25"), ("real", "vx0.2"), ("stress", "com+5"), ("real", "imu_pitch4"),
        ("stress", "volt6.3"), ("stress", "drop8_imu35"), ("stress", "drop8_imu35_s1")]


def test_floor_mu_sets_the_foot_floor_contact(monkeypatch):
    """--floor-mu reaches the floor and both foot geoms; with equal priorities MuJoCo uses the larger of the two
    frictions, so the foot-floor contact gets exactly the flag's value (1.0 by default, the model's own)."""
    b = brain_sim_check()
    real_load, seen = b.ev.load, {}

    class Loaded(Exception):
        pass

    def load(config):
        m = real_load(config)[0]
        geoms = [m.geom(g).id for g in ("floor", "left_foot_collision", "right_foot_collision")]
        seen.update(mu=[float(m.geom_friction[g, 0]) for g in geoms], priority={int(m.geom_priority[g]) for g in geoms})
        raise Loaded

    monkeypatch.setattr(b.ev, "load", load)
    # --foot-play moves the foot geoms to child bodies; the friction must follow them there
    for flags, mu in (([], 1.0), (["--floor-mu", "0.35"], 0.35), (["--floor-mu", "0.35", "--foot-play", "8"], 0.35)):
        monkeypatch.setattr(sys, "argv", ["brain_sim_check.py", "--model", "asbuilt", "--manifest", "current", "--volt", "7.0"] + flags)
        with pytest.raises(Loaded):
            b.main()
        assert seen["mu"] == pytest.approx([mu] * 3) and len(seen["priority"]) == 1


def test_current_is_the_default_and_old_variants_say_why():
    assert sim_suite.variant_flags("current") == ["--manifest", "current"]
    assert sim_suite.variant_flags(sim_suite.resolve_variant("current")) == ["--manifest", "current"]
    plugged = sim_suite.variant_flags("plugged")
    assert plugged[plugged.index("--historical-model") + 1] == "sim_suite --variant plugged"
    assert sim_suite.variant_flags("asbuilt") == ["--historical-model", "sim_suite --variant asbuilt"]
    src = open(sim_suite.__file__).read()
    assert 'choices=VARIANTS, default="current"' in src


def test_fall_dir_follows_the_trunk_lean():
    b = brain_sim_check()
    a = math.radians(30)
    roll = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])  # about +x
    assert b._lean_dir(b.rot_y(30)) == "forward"  # nose down: the up axis leans toward +x
    assert b._lean_dir(b.rot_y(-30)) == "backward"
    assert b._lean_dir(roll) == "right"
    assert b._lean_dir(roll.T) == "left"


# --- the two starts of every real walk case (2026-11-22) ---

def walk_row(case, dist, fell_at=None, steady=True, seeds=3, start=None):
    """Seed rows of one walk case; steady sets the stand61 stretch before the command (gapcheck STEADY)."""
    stand = dict(stand_dur=3.0, stand_jumps_s=0.0, stand_leg_rms=0.05, stand_still_frac=1.0, stand_settle_s=0.0) if steady \
        else dict(stand_dur=3.0, stand_jumps_s=4.0, stand_leg_rms=0.9, stand_still_frac=0.1, stand_settle_s=None)
    rows = []
    for s in range(seeds):
        r = {"case": case, "class": "real", "seed": s, "kind": "walk", "dist_x": dist, "fell_at": fell_at,
             "max_tilt_cmd": 8.0, "action_jumps_cmd": 20, "max_action": 1.0, **stand}
        r["start"] = start or sim_suite.start_of(case, True)
        r["outcome"] = sim_suite.outcome(r, True)
        rows.append(r)
    return rows


def test_every_real_walk_case_gets_a_direct_start_twin(tmp_path):
    base = sim_suite.CASES[:5] + sim_suite.STAND_CASES + sim_suite.CASES[5:] + sim_suite.floor_mu_cases(condition(tmp_path))
    cases = sim_suite.with_starts(base)
    names = [n for _, n, _ in cases]
    for cls, name, extra in base:
        twin = name + sim_suite.DIRECT in names
        assert twin == (cls == "real" and name != "stand"), name
    twins = [(c, n) for c, n, _ in cases if n.endswith(sim_suite.DIRECT)]
    assert twins == [("real", "nominal/direct"), ("real", "vx0.25/direct"), ("real", "vx0.2/direct"),
                     ("stress", "imu_pitch4/direct")]  # walk61 on the 650 mAh model falls 3/3 there
    flags = dict((n, f) for _, n, f in cases)
    assert flags["vx0.25/direct"] == ["--vx", "0.25", "--stand-s", "6.0", "--walk-at", "6.0"]
    assert flags["vx0.25"] == ["--vx", "0.25"]  # the paused start keeps brain_sim_check's 1 s stand mode, command at 4 s
    # a measured floor makes its case real, and real walk cases get both starts
    real_floor = sim_suite.with_starts(sim_suite.floor_mu_cases(condition(tmp_path, floor_mu_bare=0.42)))
    assert [n for _, n, _ in real_floor] == ["floor_mu_bare", "floor_mu_bare/direct", "floor_mu_sole"]


def test_payload_direct_start_commands_at_the_first_tick():
    """With a payload the policy runs from 0 s (no open-loop stand), so the direct start commands at 0 s."""
    assert sim_suite.direct_flags("flightpack.json") == ["--stand-s", "0.0", "--walk-at", "0.0"]
    assert sim_suite.start_of("vx0.25/direct", True) == "direct" and sim_suite.start_of("vx0.25", True) == "paused"
    assert sim_suite.start_of("stand", False) is None


def test_a_paused_freeze_after_a_steady_stand_has_its_own_label():
    assert sim_suite.aggregate(walk_row("vx0.25", 0.0))["outcome"] == sim_suite.FREEZE
    # bw-c2-s1b seed 2 on the Mac mini, 2026-11-22: settled at 0.9 s, the 3 s stretch with its open-loop switch is not
    # STEADY as a whole (leg rms 0.563), and it is still the freeze
    c2 = {"start": "paused", "dist_x": -0.003, "fell_at": None, "stand_dur": 3.0, "stand_jumps_s": 1.33,
          "stand_leg_rms": 0.563, "stand_still_frac": 0.571, "stand_settle_s": 0.9}
    assert sim_suite.outcome(c2, True) == sim_suite.FREEZE
    # an unsteady stand before the command, or the direct start, is a plain "did not start walking"
    assert sim_suite.aggregate(walk_row("vx0.25", 0.0, steady=False))["outcome"] == "did not start walking"
    assert sim_suite.aggregate(walk_row("vx0.25/direct", 0.0))["outcome"] == "did not start walking"


def test_the_direct_start_must_walk_whatever_the_reference_did():
    ok, why = sim_suite.judge(walk_row("vx0.25/direct", 0.0), walk_row("vx0.25/direct", 0.0))
    assert not ok and "direct start" in why[0]
    ok, _ = sim_suite.judge(walk_row("vx0.25/direct", 0.0), walk_row("vx0.25/direct", None, fell_at=1.2))
    assert not ok  # the reference fell: standing still is still not walking in the robot's main start
    assert sim_suite.judge(walk_row("vx0.25/direct", 1.3), walk_row("vx0.25/direct", 0.8))[0]
    assert sim_suite.judge(walk_row("vx0.25/direct", 0.5), walk_row("vx0.25/direct", None, fell_at=1.2))[0]
    # walking worse than the reference still fails
    assert not sim_suite.judge(walk_row("vx0.25/direct", 0.3), walk_row("vx0.25/direct", 0.8))[0]


def test_a_paused_freeze_is_not_a_failure_but_a_paused_fall_is():
    assert sim_suite.judge(walk_row("vx0.25", 0.0), walk_row("vx0.25", 0.7)) == (True, [])
    assert not sim_suite.judge(walk_row("vx0.25", None, fell_at=2.0), walk_row("vx0.25", 0.7))[0]
    # a freeze after an unsteady stand is "did not start walking", which fails where the reference walked
    assert not sim_suite.judge(walk_row("vx0.25", 0.0, steady=False), walk_row("vx0.25", 0.7))[0]


def test_old_rows_without_a_start_field_judge_as_before():
    """Reference files written before the two starts carry no "start": plain names are the paused start."""
    ref = walk_row("vx0.25", 0.7)
    for r in ref:
        del r["start"]
    assert sim_suite.aggregate(ref)["start"] == "paused"
    assert sim_suite.judge(walk_row("vx0.25", 0.75), ref)[0]


def test_verdict_line_keeps_the_wording_deploy_policy_matches():
    src = open(sim_suite.__file__).read()
    assert '"candidate matches or beats the reference on every real case"' in src
    deploy = open(os.path.join(sim_suite.CMU, "deploy", "deploy_policy.py")).read()
    assert 'PASS = "verdict: candidate matches or beats the reference on every real case"' in deploy
    assert 'l.startswith("known defect: ")' in deploy  # the freeze label reaches the deploy output
