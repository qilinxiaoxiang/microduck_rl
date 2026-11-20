"""Servo measurement noise twins: every corrupted joint reading at the level, critics untouched, bias widened."""
import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from mjlab_microduck.tasks.servo_noise import SERVO_NOISE, noisy_task_id

BASES = ("Mjlab-WalkNudge-FromWalk61-MicroDuck-CMU-Plugged", "Mjlab-BallDock-Nudge-MicroDuck-CMU-Plugged",
         "Mjlab-BallDock-Nudge-Fwd-MicroDuck-CMU-Plugged", "Mjlab-KneelRise-Sighted-FromRise-MicroDuck-CMU-Plugged",
         "Mjlab-KneelRise-DockFitC-Ball-MicroDuck-CMU-Plugged")


def test_noisy_task_id():
    assert noisy_task_id(BASES[0], 1) == "Mjlab-WalkNudge-FromWalk61-SN1-MicroDuck-CMU-Plugged"


def test_twins_set_every_joint_reading():
    for base in BASES:
        for lvl in (1, 2):
            t = noisy_task_id(base, lvl)
            assert t in list_tasks(), t
            assert load_rl_cfg(t).experiment_name.endswith(f"_sn{lvl}")
            cfg, lv = load_env_cfg(t, play=False), SERVO_NOISE[lvl]
            assert cfg.events["encoder_bias"].params["bias_range"] == (-lv["bias"], lv["bias"])
            seen = 0
            for grp in cfg.observations.values():
                for name, amp in (("joint_pos", lv["pos"]), ("joint_vel", lv["vel"]), ("head_joints", lv["pos"])):
                    term = grp.terms.get(name)
                    if term is None:
                        continue
                    if grp.enable_corruption:
                        assert term.noise.n_max == amp and term.noise.n_min == -amp, (t, name)
                        seen += 1
                    else:   # critics keep the recipe's level (corruption off anyway)
                        assert term.noise is None or term.noise.n_max < amp
            assert seen >= 2, t
            if "BallDock" in base:
                assert cfg.observations["actor"].terms["head_joints"].params["biased"] is True


def test_base_tasks_untouched():
    cfg = load_env_cfg(BASES[0], play=False)
    assert cfg.observations["actor"].terms["joint_pos"].noise.n_max == 0.001
    assert cfg.events["encoder_bias"].params["bias_range"] == (-0.015, 0.015)
    dock = load_env_cfg(BASES[2], play=False)
    assert "biased" not in dock.observations["actor"].terms["head_joints"].params
