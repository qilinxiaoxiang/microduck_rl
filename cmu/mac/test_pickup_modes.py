"""walk_wired.py pickup additions end to end on the fake link: ground-pick mode, its guards and hand-back, the head
command, and the jaw (calibration marks, mode 5 setup, limits, grip, release). No hardware, no physics: the fake
world turns goals into present positions at once, so the policies' actions are not meaningful, only the plumbing.

  ~/.venvs/microduck-walk/bin/python -m unittest test_pickup_modes -v
"""
import json
import math
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import jaw_ctl  # noqa: E402
import walk_brain as wb  # noqa: E402
import walk_wired as ww  # noqa: E402
from test_battery import World  # noqa: E402
from test_walk_wired_battery import WiredBattery, make_link as make_link_for  # noqa: E402

JAW = jaw_ctl.JAW_ID
TICKS_PER_DEG = 4096 / 360


def write_cmd(path, cmd):
    def hook():
        with open(path, "w") as fh:
            json.dump(cmd, fh)
    return hook


class PickupModes(WiredBattery):
    def test_groundpick_runs_the_policy_and_hands_back_to_stand61(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, cmd_path: [(2.0, write_cmd(cmd_path, {"mode": "groundpick", "phase": 1.0}))]  # noqa: E731
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "groundpick", "phase": 0.33},
                                                              end=7.0, supply="bench", hooks=hooks)
        self.assertIn("torque on at present positions", text)
        self.assertIn("mode -> groundpick", text)
        gp = [ln for ln in self.log_lines if ln.get("policy") == "groundpick"]
        self.assertTrue(gp, "the ground-pick policy never ran")
        self.assertEqual(gp[0]["kp"], wb.KP_RL)
        phases = [ln["gp_phase"] for ln in gp]
        # 0 -> 0.33 at 1/4 per s (1.32 s), held until the 2 s command, then up to the rest segment
        self.assertAlmostEqual(max(p for ln, p in zip(gp, phases) if ln["t"] - 1e9 < 2.0), 0.33, places=2)
        self.assertTrue(all(abs(p - 0.33) < 1e-6 for ln, p in zip(gp, phases) if 1.5 < ln["t"] - 1e9 < 2.0))
        self.assertIn("STOP -> zero command: ground pick complete, stand61 takes over", text)
        self.assertEqual(last_log["mode"], "walk")
        self.assertEqual(last_log["policy"], "stand")
        # the hand-back happens inside the rest segment, after the rise
        t_done = self.first("mode", "walk")
        self.assertGreater(t_done, 2.0 + (ww.GP_DONE_PHASE - 0.33) * wb.GP_PERIOD - 0.05)

    def test_groundpick_tilt_guard_holds_instead_of_ramping_to_stand(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "groundpick", "phase": 0.33},
                                                              end=3.0, supply="bench", extra_args=("--max-tilt-gp", "-1"))
        self.assertIn("STOP -> hold the last goals: ground pick: body tilt over -1.0 deg", text)
        self.assertEqual(last_log["mode"], "monitor")
        self.assertTrue(last_log["torque"])
        self.assertNotIn("mode -> stand", text)

    def test_head_command_reaches_the_policy(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        cmd = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0, "head": {"head_yaw": 0.6, "head_pitch": 0.5}}
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, cmd, end=2.0, supply="bench")
        self.assertEqual(last_log["policy"], "stand")
        hc = last_log["head_cmd"]  # deltas from the default pose, EMA-smoothed: yaw 0.6 - 0, pitch 0.5 - 0.349
        self.assertAlmostEqual(hc[2], 0.6, places=2)
        self.assertAlmostEqual(hc[1], 0.5 - 0.349, places=2)
        self.assertEqual(hc[0], 0.0)

    def run_jaw(self, cmd, hooks, end, cal=None):
        with tempfile.TemporaryDirectory() as d:
            cal_path = os.path.join(d, "jaw_calibration.json")
            if cal:
                with open(cal_path, "w") as fh:
                    json.dump(cal, fh)
            w = World(shutdown=52, hw_err=0, volts_dv=74)
            w.regs[JAW][132:136] = (1500).to_bytes(4, "little")  # the jaw sits at tick 1500
            with mock.patch.object(ww, "JawCtl", lambda **kw: jaw_ctl.JawCtl(cal_path, **kw)):
                text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, cmd, end=end, supply="bench",
                                                                      hooks=hooks)
            saved = json.load(open(cal_path)) if os.path.exists(cal_path) else None
        return text, last_log, w, saved

    def test_jaw_marks_write_the_calibration(self):
        def hooks(d, cmd_path):
            return [(0.5, write_cmd(cmd_path, {"mode": "monitor", "jaw_mark": "closed"})),
                    (1.0, lambda: None),
                    (1.5, write_cmd(cmd_path, {"mode": "monitor", "jaw_mark": "open"}))]
        text, last_log, w, saved = self.run_jaw({"mode": "monitor"}, hooks, end=2.0)
        self.assertIn("jaw closed marked at tick 1500", text)
        self.assertIn("(mark the other end too)", text)
        self.assertIn("jaw mark failed: Jaw open and closed references are too close", text)  # not moved by hand
        self.assertEqual(saved["closed_tick"], 1500)
        self.assertEqual(saved["open_tick"], 1500)  # both marks are kept; the next open mark fixes it
        self.assertEqual(last_log["jaw_tick"], 1500)
        self.assertNotIn("jaw torque on", text)  # no jaw goal was sent

    def test_jaw_open_and_grip_under_the_limits(self):
        cal = {"closed_tick": 1500, "open_tick": 1700}  # opening raises the tick
        def hooks(d, cmd_path):
            return [(1.0, write_cmd(cmd_path, {"mode": "monitor", "jaw": "grip"})),
                    (2.0, write_cmd(cmd_path, {"mode": "monitor", "jaw": 40})),
                    (3.0, write_cmd(cmd_path, {"mode": "monitor", "jaw_torque": False}))]
        text, last_log, w, saved = self.run_jaw({"mode": "monitor", "jaw": "open"}, hooks, end=4.0, cal=cal)
        self.assertIn("jaw torque on at tick 1500 (0.0 deg open), mode 5, goal current 100 mA", text)
        self.assertEqual(w.read(JAW, jaw_ctl.ADDR_OPERATING_MODE, 1), 5)
        self.assertEqual(w.read(JAW, jaw_ctl.ADDR_GOAL_CURRENT, 2), jaw_ctl.GOAL_CURRENT_MA)
        self.assertEqual(w.read(JAW, jaw_ctl.ADDR_CURRENT_LIMIT, 2), jaw_ctl.CURRENT_LIMIT_MA)
        goals = [v for addr, i, v in w.log if i == JAW and addr == jaw_ctl.ADDR_GOAL]
        # "open" = 20 deg; the head at STAND (neck/head pitch 20 deg) allows the full 25 deg cap
        self.assertEqual(goals[1], 1500 + round(20 * TICKS_PER_DEG))
        # "grip" aims 3 deg past closed
        self.assertIn(1500 + round(-3 * TICKS_PER_DEG), goals)
        # 40 deg is capped at 25 deg and logged as a clamp
        self.assertIn(1500 + int(25 * TICKS_PER_DEG), goals)
        clamped = [ln for ln in self.log_lines if "jaw" in (ln.get("jaw_clamps") or [])]
        self.assertTrue(clamped)
        self.assertIn("jaw torque off", text)
        self.assertEqual(w.read(JAW, jaw_ctl.ADDR_TORQUE, 1), 0)
        self.assertIsNone(last_log["jaw_goal"])

    def test_jaw_holding_is_reported_when_the_grip_stops_short(self):
        cal = {"closed_tick": 1500, "open_tick": 1700}
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        w.regs[JAW][132:136] = (1600).to_bytes(4, "little")
        with tempfile.TemporaryDirectory() as d:
            cal_path = os.path.join(d, "jaw_calibration.json")
            with open(cal_path, "w") as fh:
                json.dump(cal, fh)
            # the goal never reaches the servo: the jaw stays at 1600 as if something were between the bills
            with mock.patch.object(ww, "JawCtl", lambda **kw: jaw_ctl.JawCtl(cal_path, **kw)), \
                    mock.patch.object(jaw_ctl.JawCtl, "write", lambda self, link, ch, mh: None):
                text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "monitor", "jaw": "grip"},
                                                                      end=1.0, supply="bench")
        self.assertIn("jaw torque on", text)
        self.assertTrue(last_log["jaw_holding"])
        self.assertAlmostEqual(last_log["jaw_deg"], 100 / TICKS_PER_DEG, places=1)

    def test_uncalibrated_jaw_command_is_refused_once(self):
        text, last_log, w, saved = self.run_jaw({"mode": "monitor", "jaw": "open"}, lambda d, p: [], end=1.0)
        self.assertEqual(text.count("jaw command ignored: not calibrated"), 1)
        self.assertEqual(w.read(JAW, jaw_ctl.ADDR_TORQUE, 1), 0)


if __name__ == "__main__":
    unittest.main()


class StartWithStand(WiredBattery):
    """2026-10-16 restart: the first sync read after a start misses every servo (the port has just opened). With the
    control file at stand, the Brain entered stand on that tick and ramped from the zero ticks (+-pi) to STAND; the
    first goals after torque-on threw every joint half a turn and overloaded the head and neck. Now the Brain waits
    for a fresh sample before entering any torque mode, so the first goals written are the present positions."""

    def test_stand_entry_waits_for_the_first_fresh_read(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        stand_ticks = {i: ww.to_tick(n, wb.STAND[n]) for i, n in zip(ww.IDS, ww.NAMES)}
        for i, t in stand_ticks.items():
            w.regs[i][132:136] = int(t).to_bytes(4, "little", signed=True)
        reads_seen = []
        def missing_first(world, clock, reads):
            base = make_link_for(world, clock, reads)

            class FirstMiss(base):
                def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
                    got = super().sync_read(addr, size, ids, timeout, allow_alert)
                    if addr == ww.ADDR_VEL and not reads_seen:
                        reads_seen.append(addr)
                        return {}  # the first position read misses every servo
                    return got
            return FirstMiss
        with mock.patch.object(sys.modules[WiredBattery.__module__], "make_link", missing_first):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "stand"}, end=3.0,
                                                                  supply="bench")
        self.assertIn("mode stand waits for a fresh joint and IMU sample", text)
        self.assertIn("mode -> stand", text)
        self.assertIn("torque on at present positions", text)
        goals = [(i, v) for addr, i, v in w.log if addr == ww.ADDR_GOAL]
        self.assertTrue(goals)
        first_goals = {i: v for i, v in goals[:14]}
        for i, t in stand_ticks.items():  # the first goals hold the present (STAND) positions, nothing near +-pi
            self.assertLess(abs(first_goals[i] - t), 20, (i, first_goals[i], t))
        self.assertTrue(all(abs(v - stand_ticks[i]) < 20 for i, v in goals), "a goal left the present pose")
        self.assertEqual(last_log["mode"], "stand")
        self.assertTrue(last_log["torque"])

    def test_without_the_wait_the_first_goals_are_half_a_turn_away(self):
        """The 2026-10-16 slam reproduced: with the wait disabled the ramp starts from the zero ticks."""
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        stand_ticks = {i: ww.to_tick(n, wb.STAND[n]) for i, n in zip(ww.IDS, ww.NAMES)}
        for i, t in stand_ticks.items():
            w.regs[i][132:136] = int(t).to_bytes(4, "little", signed=True)
        reads_seen = []

        def missing_first(world, clock, reads):
            base = make_link_for(world, clock, reads)

            class FirstMiss(base):
                def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
                    got = super().sync_read(addr, size, ids, timeout, allow_alert)
                    if addr == ww.ADDR_VEL and not reads_seen:
                        reads_seen.append(addr)
                        return {}
                    return got
            return FirstMiss
        with mock.patch.object(sys.modules[WiredBattery.__module__], "make_link", missing_first), \
                mock.patch.object(wb, "WAIT_FOR_FRESH", False):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "stand"}, end=0.3,
                                                                  supply="bench")
        self.assertIn("torque on at present positions", text)
        goals = [(i, v) for addr, i, v in w.log if addr == ww.ADDR_GOAL]
        ramp_goals = goals[14:28]  # the hold-where-they-are write, then the first ramp tick
        self.assertTrue(ramp_goals)
        worst = max(abs(v - stand_ticks[i]) for i, v in ramp_goals)
        self.assertGreater(worst, 1500, "the ramp should have started near the zero ticks (half a turn away)")
