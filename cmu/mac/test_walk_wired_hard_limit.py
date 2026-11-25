"""walk_wired.py hard_limit: --hard-max-action-sit widens the per-tick action stop in sit mode only (2026-11-24: the
sit-and-rise policy completes in sim only at 4.5 rad while it sits). No hardware.

  ~/.venvs/microduck-walk/bin/python -m unittest test_walk_wired_hard_limit -v
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import walk_wired as ww  # noqa: E402


def guard(hard, actions):
    """The main loop's use of the limit: ticks counted over it, and the applied action after the clip."""
    over = sum(a > hard for a in actions)
    return over, [max(-hard, min(hard, a)) for a in actions]


class HardLimit(unittest.TestCase):
    def test_unset_sit_limit_equals_hard_max_action(self):
        for mode in ("sit", "walk", "stand", "groundpick"):
            self.assertEqual(ww.hard_limit(3.0, None, mode, None), 3.0)
        self.assertEqual(ww.hard_limit(3.2, None, "sit", None), 3.2)

    def test_set_sit_limit_lets_3_5_rad_through_in_sit(self):
        hard = ww.hard_limit(3.0, 4.5, "sit", None)
        self.assertEqual(hard, 4.5)
        over, applied = guard(hard, [3.5] * ww.OVER_LIMIT_TICKS)
        self.assertEqual(over, 0)
        self.assertEqual(applied, [3.5] * ww.OVER_LIMIT_TICKS)

    def test_walk_still_stops_and_clips_at_3_0(self):
        for mode in ("walk", "stand", "kick", "groundpick", "seepick"):
            hard = ww.hard_limit(3.0, 4.5, mode, None)
            self.assertEqual(hard, 3.0)
            over, applied = guard(hard, [3.5] * ww.OVER_LIMIT_TICKS)
            self.assertEqual(over, ww.OVER_LIMIT_TICKS)
            self.assertEqual(applied, [3.0] * ww.OVER_LIMIT_TICKS)

    def test_getup_case_unchanged(self):
        self.assertEqual(ww.hard_limit(3.0, None, "walk", object()), max(3.0, ww.GETUP_MAX_ACTION))
        self.assertEqual(ww.hard_limit(3.0, 4.5, "walk", object()), max(3.0, ww.GETUP_MAX_ACTION))
        self.assertEqual(ww.hard_limit(7.0, None, "walk", object()), 7.0)

    def test_main_loop_uses_hard_limit(self):
        with open(os.path.join(HERE, "walk_wired.py")) as f:
            src = f.read()
        self.assertIn("hard = hard_limit(args.hard_max_action, args.hard_max_action_sit, brain.mode, brain.getup)", src)
        self.assertIn('"--hard-max-action-sit", type=float, default=None', src)


if __name__ == "__main__":
    unittest.main()
