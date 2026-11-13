"""Invariants of the task D loaded-rise variant (tasks/cmu_seeing_pix_rise_env_cfg.py). Cfg-only, no env."""
import math

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.tasks import cmu_seeing_pix_rise_env_cfg as rise
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import TIMELINE

BODIES = ("Plugged", "Battery-20261104")


def test_tasks_registered_with_distinct_experiments():
    for b in BODIES:
        base = f"Mjlab-SeeingPickPix-FromSpx-MicroDuck-CMU-{b}"
        ris = f"Mjlab-SeeingPickPixRise-FromSpx-MicroDuck-CMU-{b}"
        assert base in list_tasks() and ris in list_tasks()
        assert load_rl_cfg(base).experiment_name != load_rl_cfg(ris).experiment_name


def test_only_the_main_term_differs():
    base = rise.make_seeing_pix_handoff_env_cfg("plugged", rise=False)
    r = rise.make_seeing_pix_handoff_env_cfg("plugged", rise=True)
    assert set(base.rewards) == set(r.rewards)
    for k in base.rewards:
        assert r.rewards[k].weight == base.rewards[k].weight, k
        if k != "grasp_held":
            assert r.rewards[k].func is base.rewards[k].func, k
            assert r.rewards[k].params == base.rewards[k].params, k
    assert base.rewards["grasp_held"].func is microduck_mdp.grasp_held
    assert r.rewards["grasp_held"].func is rise.grasp_held_admitted
    assert r.rewards["grasp_held"].weight > 0  # >= 0 function: positive weight (AGENTS sign convention)
    assert r.rewards["grasp_held"].params["phase_start"] == base.rewards["grasp_held"].params["phase_start"]
    assert set(base.observations["actor"].terms) == set(r.observations["actor"].terms)
    assert base.episode_length_s == r.episode_length_s


def test_gate_and_handoff_match_the_trained_timeline_and_runtime():
    assert rise.ADMIT_FROM == TIMELINE["return_end"]
    assert TIMELINE["return_end"] < rise.HANDOFF_PHASE < 1.0
    assert rise.HANDOFF_PHASE == 0.90  # walk_wired.SP_DONE_PHASE


def test_admission_scores_todays_exit_visibly():
    bowed_rms = 1.48 / math.sqrt(14)  # replay.py: mean L2 offset of the held late-pick targets from stand nominal
    score = math.exp(-(bowed_rms / rise.ADMIT_STD_Q) ** 2)
    assert 0.3 < score < 0.8          # visible gradient, well below the nominal's 1
    assert rise.ADMIT_MAX_RMS < bowed_rms


def test_handoff_logging_on_both_tasks():
    for flag in (False, True):
        cfg = rise.make_seeing_pix_handoff_env_cfg("plugged", rise=flag)
        assert cfg.events["track_handoff"].func is rise.track_handoff
        assert cfg.curriculum["handoff_metrics"].func is rise.handoff_metrics
