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
        ("real", "nominal"), ("real", "vx0.25"), ("real", "vx0.2"), ("real", "com+5"), ("real", "imu_pitch4"),
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
    for flags, mu in (([], 1.0), (["--floor-mu", "0.35"], 0.35)):
        monkeypatch.setattr(sys, "argv", ["brain_sim_check.py", "--model", "asbuilt", "--volt", "7.0"] + flags)
        with pytest.raises(Loaded):
            b.main()
        assert seen["mu"] == pytest.approx([mu] * 3) and len(seen["priority"]) == 1


def test_fall_dir_follows_the_trunk_lean():
    b = brain_sim_check()
    a = math.radians(30)
    roll = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])  # about +x
    assert b._lean_dir(b.rot_y(30)) == "forward"  # nose down: the up axis leans toward +x
    assert b._lean_dir(b.rot_y(-30)) == "backward"
    assert b._lean_dir(roll) == "right"
    assert b._lean_dir(roll.T) == "left"
