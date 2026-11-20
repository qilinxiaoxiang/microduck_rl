"""Invariants of the blind kneel-rise task (tasks/cmu_kneel_rise_env_cfg.py). Cfg-only, no env."""
import math

import torch

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_rl_cfg
from mjlab_microduck.tasks import cmu_kneel_rise_env_cfg as kr
from mjlab_microduck.tasks import cmu_seeing_pix_rise_env_cfg as rise
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_grasp_env_cfg import GG_PERIOD
from mjlab_microduck.tasks.microduck_static_grasp_env_cfg import TIMELINE

BODIES = ("Plugged", "Battery-20261104")


def test_tasks_registered_with_own_experiment():
    for b in BODIES:
        t = f"Mjlab-KneelRise-FromRise-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "kneel_rise"
    assert kr.INIT_ONNX.exists()


def test_rise_objective_kept_and_hold_on_inside_the_episode():
    cfg = kr.make_kneel_rise_env_cfg("plugged")
    assert cfg.rewards["grasp_held"].func is rise.grasp_held_admitted
    assert cfg.rewards["grasp_held"].weight > 0
    assert TIMELINE["return_end"] < kr.HOLD_PHASE < 1.0
    assert cfg.commands["twist"].hold_phase == kr.HOLD_PHASE
    # the episode reaches the hold phase and then runs HOLD_S more
    assert math.isclose(cfg.episode_length_s, kr.HOLD_PHASE * GG_PERIOD + kr.HOLD_S)
    assert cfg.episode_length_s > GG_PERIOD * kr.HOLD_PHASE + 2.0


def test_phase_freezes_instead_of_wrapping():
    class Cfg:
        hold_phase = 0.9
    cmd = microduck_mdp.GroundPickPhaseCommand.__new__(microduck_mdp.GroundPickPhaseCommand)
    cmd.cfg = Cfg()
    cmd._period = 4.0
    cmd._gp_phase = torch.tensor([0.0, 0.85, 0.899])
    cmd.vel_command_b = torch.zeros(3, 3)
    for _ in range(400):
        cmd.compute(0.02)
    assert torch.allclose(cmd._gp_phase, torch.full((3,), 0.9))
    # jaw schedule: closed for good after close_end, so a frozen phase keeps it shut
    frac = microduck_mdp.mouth_schedule_fraction(cmd._gp_phase, 0.02, 0.10, TIMELINE["close_start"],
                                                 TIMELINE["close_end"])
    assert torch.all(frac == 0)


def test_actor_is_blind_but_keeps_holding_and_61d():
    cfg = kr.make_kneel_rise_env_cfg("plugged")
    assert cfg.observations["actor"].terms["body_command"].func is kr.blind_body_command
    assert sorted(kr.BLIND_ZERO + kr.BLIND_KEEP) == list(range(6))
    assert kr.BLIND_KEEP == (4,)   # holding bit
    # the critic still sees the camera block
    assert cfg.observations["critic"].terms["body_command"].func is not kr.blind_body_command


def test_docked_starts_and_no_kicks():
    cfg = kr.make_kneel_rise_env_cfg("plugged")
    p = cfg.events["reset_object"].params
    assert p["kick_prob"] == 0.0
    assert p["window"] == kr.DOCK_WINDOW_START
    assert p["start_speed"][1] <= 0.05
    assert "dock_window" in cfg.curriculum and "pix_window" not in cfg.curriculum
    assert kr.make_kneel_rise_env_cfg("plugged", play=True).events["reset_object"].params["window"] == kr.DOCK_WINDOW_MAX


def test_neck_rate_cost_sign_and_gate():
    cfg = kr.make_kneel_rise_env_cfg("plugged")
    t = cfg.rewards["held_neck_rate"]
    assert t.func is kr.held_neck_rate_cost and t.weight < 0   # >= 0 cost: negative weight
    assert 0 < kr.NECK_RATE_FREE < 5.0                          # the issue-5 drops were at 5-8 rad/s


def test_sighted_variant_keeps_camera_and_own_experiment():
    for b in BODIES:
        t = f"Mjlab-KneelRise-Sighted-FromRise-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "kneel_rise_sighted"
    cfg = kr.make_kneel_rise_env_cfg("plugged", sighted=True)
    assert cfg.observations["actor"].terms["body_command"].func is not kr.blind_body_command
    assert cfg.events["reset_object"].params["window"] == kr.SIGHTED_WINDOW_START
    assert cfg.curriculum["dock_window"].params["window_max"] == kr.SIGHTED_WINDOW_MAX
    assert cfg.commands["twist"].hold_phase == kr.HOLD_PHASE


def test_sighted_pins_inherited_curricula_at_rise_s1_step():
    cfg = kr.make_kneel_rise_env_cfg("plugged", sighted=True)
    ar = cfg.curriculum["action_rate_weight"].params["weight_stages"]
    assert ar == [{"step": 0, "weight": -2.0}] and cfg.rewards["action_rate_l2"].weight == -2.0
    assert cfg.curriculum["com_range"].params["range_stages"] == [{"step": 0, "range": 0.01}]


def test_dockfit_variant_opens_near_side_and_draws_dock_ends():
    from mjlab_microduck.tasks import cmu_seeing_pix_env_cfg as pix
    for b in BODIES:
        t = f"Mjlab-KneelRise-DockFit-MicroDuck-CMU-{b}"
        assert t in list_tasks()
        assert load_rl_cfg(t).experiment_name == "kneel_rise_dockfit"
    cfg = kr.make_kneel_rise_env_cfg("plugged", dockfit=True)
    p = cfg.events["reset_object"].params
    assert cfg.observations["actor"].terms["body_command"].func is not kr.blind_body_command   # sighted
    assert "dock_window" not in cfg.curriculum and p["window"] == kr.SIGHTED_WINDOW_MAX
    # the near edge reaches past the dock's p10 (-1.9 cm) where the plain window stops at R_MIN
    r0 = kr.pix.sp.OBJECT_NOMINAL_XY[0]
    assert pix.R_MIN - r0 > -0.01 and p["r_min"] - r0 < -0.03
    assert 0.0 < p["dock_frac"] < 1.0 and p["dock_clip"][0] < -0.03
    assert cfg.curriculum["com_range"].params["range_stages"] == [{"step": 0, "range": 0.01}]


def test_ball_only_variant_draws_only_the_ball():
    t = "Mjlab-KneelRise-DockFit-Ball-MicroDuck-CMU-Plugged"
    assert t in list_tasks() and load_rl_cfg(t).experiment_name == "kneel_rise_ball"
    cfg = kr.make_kneel_rise_env_cfg("plugged", dockfit=True, ball_only=True)
    assert cfg.events["object_size"].params["probs"] == (1.0, 0.0, 0.0)
    assert cfg.events["reset_object"].params["dock_frac"] == kr.DOCKFIT_FRAC
    # the plain variants keep the seeing pick's mix
    assert "probs" not in kr.make_kneel_rise_env_cfg("plugged", dockfit=True).events["object_size"].params


def test_low_pose_variant_only_lowers_static_pose():
    from mjlab.tasks.registry import load_env_cfg
    lo = load_env_cfg("Mjlab-KneelRise-DockFitC-Ball-LowPose-MicroDuck-CMU-Plugged", play=False)
    ctl = load_env_cfg("Mjlab-KneelRise-DockFitC-Ball-MicroDuck-CMU-Plugged", play=False)
    assert lo.rewards["static_pose"].weight == kr.LOW_POSE_WEIGHT < ctl.rewards["static_pose"].weight
    assert {k: r.weight for k, r in lo.rewards.items() if k != "static_pose"} == \
        {k: r.weight for k, r in ctl.rewards.items() if k != "static_pose"}
    assert ctl.events["reset_object"].params["dock_mu"] == kr.DOCKFIT_C["mu"]
