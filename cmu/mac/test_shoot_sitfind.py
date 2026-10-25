"""shoot.py sitfind helpers: keyframe interpolation and the visual-servo step's signs and limits."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shoot  # noqa: E402

AIM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sitlook.json")))["aim"]
W, H, FX, FY = 480, 640, 497.0, 491.0


class SitFind(unittest.TestCase):
    def test_keyframes_ramp_linearly_and_hold_after_the_end(self):
        keys = [{"t": 0.0, "head": {"head_yaw": 0.0}}, {"t": 2.0, "head": {"head_yaw": 0.6}}]
        self.assertAlmostEqual(shoot.keys_at(keys, 1.0)["head_yaw"], 0.3)
        self.assertAlmostEqual(shoot.keys_at(keys, 5.0)["head_yaw"], 0.6)

    def test_a_part_right_of_centre_turns_the_head_right_and_below_centre_looks_down(self):
        d = shoot.aim_step({}, {"cx": W / 2 + 100, "cy": H / 2 + 100}, AIM, FX, FY, W, H)
        self.assertLess(d["head_yaw"], 0)
        self.assertGreater(d["head_pitch"], 0)
        d = shoot.aim_step({}, {"cx": W / 2 - 100, "cy": H / 2 - 100}, AIM, FX, FY, W, H)
        self.assertGreater(d["head_yaw"], 0)
        self.assertLess(d["head_pitch"], 0)

    def test_a_step_is_rate_limited_and_the_head_stays_in_its_limits(self):
        d = shoot.aim_step({}, {"cx": 0, "cy": H}, AIM, FX, FY, W, H)
        self.assertLessEqual(abs(d["head_yaw"]), AIM["max_rate"] * AIM["period_s"] + 1e-9)
        d = {"head_yaw": AIM["yaw_max"], "head_pitch": AIM["pitch_max"]}
        d = shoot.aim_step(d, {"cx": 0, "cy": H}, AIM, FX, FY, W, H)
        self.assertEqual(d["head_yaw"], AIM["yaw_max"])
        self.assertEqual(d["head_pitch"], AIM["pitch_max"])

    def test_centred_part_leaves_the_head_where_it_is(self):
        d = shoot.aim_step({"head_yaw": 0.2, "head_pitch": 0.1}, {"cx": W / 2, "cy": H / 2}, AIM, FX, FY, W, H)
        self.assertAlmostEqual(d["head_yaw"], 0.2)
        self.assertAlmostEqual(d["head_pitch"], 0.1)

    def test_the_servo_converges_on_a_fixed_part(self):
        # a part at head yaw -0.4, pitch +0.3 from the start: the picture offset follows the head error
        import math
        d = {"head_yaw": 0.0, "head_pitch": 0.0}
        for _ in range(60):
            ex, ey = -0.4 - d["head_yaw"], 0.3 - d["head_pitch"]
            det = {"cx": W / 2 - FX * math.tan(ex), "cy": H / 2 + FY * math.tan(ey)}
            d = shoot.aim_step(d, det, AIM, FX, FY, W, H)
        self.assertAlmostEqual(d["head_yaw"], -0.4, places=2)
        self.assertAlmostEqual(d["head_pitch"], 0.3, places=2)


if __name__ == "__main__":
    unittest.main()
