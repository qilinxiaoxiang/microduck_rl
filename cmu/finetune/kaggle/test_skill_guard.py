"""skill_worker.py --bend-guard: observe (default) keeps bend4's guard logging but it can never stop a run; stop keeps
bend_terms.bend_guard's own thresholds. Builds env cfgs only (no env, no simulation).

  PYTHONPATH=src .venv/bin/python -m pytest -q cmu/finetune/kaggle/test_skill_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mjlab")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill_worker as skw  # noqa: E402


def _bend4():
    return skw.make_env_cfg("groundpick", "current", recipe="bend4", conditions="robot")


def test_default_is_observe():
    assert skw.main.__code__  # the flag lives in main's parser
    src = (Path(__file__).resolve().parent / "skill_worker.py").read_text()
    assert '"--bend-guard", choices=("observe", "stop"), default="observe"' in src
    assert "guard_mode = apply_guard_mode(env_cfg, args.bend_guard)" in src


def test_observe_can_never_flag_a_window():
    cfg = _bend4()
    rec = skw.apply_guard_mode(cfg, "observe")
    p = cfg.rewards["bend_guard"].params
    assert rec["mode"] == "observe"
    # bend_guard: squat = knee > knee_squat or pitch < pitch_min_deg; falling = fall > fall_factor * base
    for knee, pitch_deg in ((1.57, -90.0), (3.14, -180.0), (0.0, 0.0)):
        assert not (knee > p["knee_squat"] or pitch_deg < p["pitch_min_deg"])
    assert 1e6 <= p["fall_factor"] * 1e-6  # even a fall rate 1e6 times a 1e-6 first window is not "falling"
    assert p["window_steps"] == skw_guard_window()


def skw_guard_window():
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    return sk.BEND4_GUARD["window_steps"]


def test_stop_keeps_the_guard_thresholds():
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    cfg = _bend4()
    rec = skw.apply_guard_mode(cfg, "stop")
    assert rec["mode"] == "stop"
    for k, v in sk.BEND4_GUARD.items():
        assert cfg.rewards["bend_guard"].params[k] == v


def test_no_guard_without_bend4():
    assert skw.apply_guard_mode(skw.make_env_cfg("groundpick", "current", recipe="pollen"), "observe") is None
    assert skw.apply_guard_mode(skw.make_env_cfg("sitstand", "current"), "observe") is None
