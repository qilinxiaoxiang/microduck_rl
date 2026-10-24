"""hold.py (graded hold estimator) and its use in walk_wired.py: the grip and the unloaded feet scale stand61's actions
and the gains gradually, the hand leaving gives them back at once, a fall without a hand changes nothing, and the hold
file switches it at run time. Real Brain and policies, the fake servo link and clock of test_walk_wired_touch.

  ~/.venvs/microduck-walk/bin/python -m unittest test_hold -v
"""
import json
import os
import sys
import tempfile
import unittest
import unittest.mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import hold  # noqa: E402
import walk_wired as ww  # noqa: E402
from test_battery import World  # noqa: E402
from test_walk_wired_touch import WALK0, FakePads, WiredRun, hook, release  # noqa: E402

REST = [22000.0, 23000.0, 24000.0, 25000.0, 26000.0]
DT = 0.02


def counts(pressed=(), rise=2000.0):
    return [c + (rise if p in pressed else 0.0) for p, c in zip(hold.PADS, REST)]


def run(est, seconds, pressed=(), legs_ma=None, load=None, age=0.01):
    """Feed one constant input for `seconds`; returns the per-tick (s, gains) history."""
    out = []
    for _ in range(int(round(seconds / DT))):
        est.update(DT, counts(pressed), age, legs_ma, load)
        out.append((est.s, est.gains()))
    return out


GRIP = ("L1", "R1", "M")


class Estimator(unittest.TestCase):
    def setUp(self):
        self.est = hold.HoldEstimator({"legs_stand_ma": 1000, "legs_air_ma": 200}, kp_rl=200)
        run(self.est, 1.0, legs_ma=1000)  # rest baseline, standing

    def test_untouched_standing_changes_nothing(self):
        hist = run(self.est, 2.0, legs_ma=1000)
        self.assertTrue(all(s == 1.0 for s, g in hist))
        self.assertTrue(all(set(g.values()) == {200} for s, g in hist))

    def test_a_longer_grip_lowers_the_authority_further_and_never_below_the_touch_floor(self):
        hist = run(self.est, 3.0, pressed=GRIP, legs_ma=1000)
        s = [h[0] for h in hist]
        self.assertTrue(all(b <= a + 1e-12 for a, b in zip(s, s[1:])))  # gradual, never back up while held
        self.assertGreater(s[4], 0.8)    # 0.1 s into the grip: still most of the authority (no freeze)
        self.assertLess(s[49], s[14])    # 1 s lower than 0.3 s: a longer grip counts more
        self.assertGreater(s[-1], 1 - hold.A_TOUCH - 0.01)  # the hand alone keeps at least 1 - A_TOUCH
        self.assertLess(s[-1], 1 - hold.A_TOUCH + 0.05)
        self.assertTrue(all(set(g.values()) == {200} for _, g in hist))  # feet loaded: no gain change

    def test_a_light_brush_moves_almost_nothing(self):
        hist = run(self.est, 0.2, pressed=("M",), legs_ma=1000)
        self.assertGreater(min(h[0] for h in hist), 0.9)

    def test_lifted_by_the_head_softens_the_legs_slowly_and_stiffens_the_neck_at_once(self):
        run(self.est, 1.0, pressed=GRIP, legs_ma=1000)
        hist = run(self.est, 1.0, pressed=GRIP, legs_ma=200)  # feet in the air
        self.assertLess(hist[-1][0], 0.02)
        g0, g_end = hist[1][1], hist[-1][1]
        self.assertGreater(g0["neck_pitch"], 300)  # within two ticks of the lift
        self.assertGreater(g0["left_knee"], 150)   # the legs have not dropped yet
        self.assertEqual(g_end["left_knee"], hold.KP_LEG_MIN)
        self.assertEqual(g_end["neck_pitch"], hold.KP_NECK_HOLD)
        self.assertEqual((g_end["head_yaw"], g_end["head_roll"]), (200, 200))
        legs = [h[1]["right_ankle"] for h in hist]
        self.assertTrue(all(b <= a for a, b in zip(legs, legs[1:])))
        self.assertGreater(sum(1 for v in legs if hold.KP_LEG_MIN < v < 200), 5)  # a ramp, not a step

    def test_set_down_stiffens_the_legs_the_same_tick_and_letting_go_returns_the_authority_fast(self):
        run(self.est, 1.0, pressed=GRIP, legs_ma=1000)
        run(self.est, 1.0, pressed=GRIP, legs_ma=200)
        down = run(self.est, 0.5, pressed=GRIP, legs_ma=1000)  # set down, hand still on
        knee = [h[1]["left_knee"] for h in down]
        self.assertGreater(knee[1], 120)        # no slow ramp back: the gain follows the (smoothed) load
        self.assertGreaterEqual(knee[3], 170)   # LOAD_TAU 0.06 s: most of the gain within 0.08 s
        self.assertEqual(knee[12], 200)         # and exactly the full gain within 0.25 s
        self.assertEqual(down[-1][0], 1.0)              # landed: the hand term is off until the hand leaves
        letgo = run(self.est, 0.4, legs_ma=1000)
        s = [h[0] for h in letgo]
        self.assertEqual(s[-1], 1.0)
        reach = next(k for k, v in enumerate(s) if v >= 0.99) * DT
        self.assertLess(reach, 0.2)

    def test_feet_unloaded_without_a_hand_is_a_fall_and_changes_nothing(self):
        hist = run(self.est, 1.0, legs_ma=200)
        self.assertTrue(all(s == 1.0 for s, g in hist))
        self.assertTrue(all(set(g.values()) == {200} for s, g in hist))

    def test_a_stale_touch_block_counts_as_no_hand(self):
        hist = run(self.est, 1.0, pressed=GRIP, legs_ma=200, age=0.5)
        self.assertTrue(all(s == 1.0 for s, g in hist))

    def test_set_down_with_the_hand_still_on_gives_the_authority_back_before_the_hand_leaves(self):
        run(self.est, 1.0, pressed=GRIP, legs_ma=1000)
        run(self.est, 1.0, pressed=GRIP, legs_ma=200)
        self.assertTrue(self.est.airborne and not self.est.landed)
        down = run(self.est, 0.5, pressed=GRIP, legs_ma=1000)
        self.assertTrue(self.est.landed)
        self.assertEqual(down[-1][0], 1.0)  # no hand term after landing: it balances on its own feet
        run(self.est, 0.3, legs_ma=1000)    # hand leaves: the latch resets
        self.assertFalse(self.est.airborne or self.est.landed)
        again = run(self.est, 3.0, pressed=GRIP, legs_ma=1000)  # a new grip lowers it again
        self.assertLess(again[-1][0], 0.45)

    def test_a_short_touch_dropout_keeps_the_hand_and_the_touchdown_latch(self):
        run(self.est, 1.0, pressed=GRIP, legs_ma=1000)
        run(self.est, 1.0, pressed=GRIP, legs_ma=200)
        run(self.est, 0.5, pressed=GRIP, legs_ma=1000)
        self.assertTrue(self.est.landed)
        hiccup = run(self.est, 0.3, pressed=GRIP, legs_ma=1000, age=0.5)  # I2C hiccup, hand still on
        self.assertTrue(self.est.landed and self.est.e > 0.9)
        self.assertTrue(all(h[0] == 1.0 for h in hiccup))
        run(self.est, 0.6, pressed=GRIP, legs_ma=1000, age=0.5)  # gone for good: no hand
        self.assertFalse(self.est.landed)
        self.assertLess(self.est.e, hold.GATE_LO)

    def test_gains_and_record_work_before_the_first_update(self):
        est = hold.HoldEstimator({}, kp_rl=200)
        self.assertEqual(set(est.gains().values()), {200})
        self.assertEqual(est.record()["s"], 1.0)

    def test_without_load_calibration_only_the_grip_acts(self):
        est = hold.HoldEstimator({}, kp_rl=200)
        run(est, 1.0)
        hist = run(est, 3.0, pressed=GRIP, legs_ma=50)
        self.assertIsNone(est.load_f)
        self.assertGreater(hist[-1][0], 1 - hold.A_TOUCH - 0.01)
        self.assertTrue(all(set(g.values()) == {200} for s, g in hist))

    def test_the_rest_baseline_follows_slow_drift_but_not_a_held_touch(self):
        est = hold.HoldEstimator({}, kp_rl=200)
        for k in range(1500):  # 30 s of a 100-count warm-up drift on every pad
            est.update(DT, [c + 100.0 * k / 1500 for c in REST], 0.01)
        self.assertEqual(est.levels, [0.0] * 5)
        base = list(est.base)
        run(est, 5.0, pressed=GRIP)
        self.assertEqual([round(b) for b in est.base[:2]], [round(b) for b in base[:2]])  # covered pads keep it
        self.assertGreater(est.e, 0.9)

    def test_hold_file_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "hold.json")
            self.assertEqual(hold.read_hold_file(p), {"on": False, "cal": {}, "error": None})
            hold.write_hold_file(True, {"legs_stand_ma": 900}, p)
            self.assertEqual(hold.read_hold_file(p), {"on": True, "cal": {"legs_stand_ma": 900}, "error": None})
            with open(p, "w") as f:
                f.write("{broken")
            self.assertEqual(hold.read_hold_file(p)["on"], False)
            bad = ({"on": "false", "cal": {}}, {"on": True, "cal": {"pad_lo": [1, 2, 3]}},
                   {"on": True, "cal": {"legs_stand_ma": "900", "legs_air_ma": 100}}, {"on": True, "cal": [1, 2]},
                   {"on": True, "cal": {"grip_full_mm2": 0}}, [1, 2])
            for d in bad:
                with open(p, "w") as f:
                    json.dump(d, f)
                r = hold.read_hold_file(p)
                self.assertFalse(r["on"], d)
                hold.HoldEstimator(r["cal"]).update(DT, counts(), 0.01, 500)  # never raises


class GainWrites(unittest.TestCase):
    def test_whole_robot_gain_exact_and_hold_gains_in_steps(self):
        names = ww.NAMES
        self.assertEqual(ww.p_gain_writes(200, None), {i: 200 for i in ww.IDS})
        now = {n: 200 for n in names}
        self.assertEqual(ww.p_gain_writes(200, now), {})
        self.assertEqual(ww.p_gain_writes(400, now), {i: 400 for i in ww.IDS})
        soft = dict(now, left_knee=195, neck_pitch=210)
        self.assertEqual(ww.p_gain_writes(soft, now), {ww.IDS[names.index("neck_pitch")]: 210})
        back = dict(now, left_knee=200)
        self.assertEqual(ww.p_gain_writes(back, dict(now, left_knee=196)), {ww.IDS[names.index("left_knee")]: 200})
        self.assertEqual(ww.p_gain_writes(dict(now, left_knee=144), dict(now, left_knee=140)), {})
        self.assertEqual(ww.p_gain_writes(dict(now, left_knee=144), dict(now, left_knee=140), settled=True),
                         {ww.IDS[names.index("left_knee")]: 144})  # a plateau reaches its exact value
        self.assertEqual(ww.kp_for_log(now), 200)
        self.assertEqual(ww.kp_for_log(soft)["neck_pitch"], 210)


def legs_world(stand_ma=800):
    """World whose ten leg servos read `stand_ma` mA in total (80 mA each) and the head servos 20 mA."""
    w = World(shutdown=52, hw_err=0, volts_dv=74)
    for i in w.regs:
        w.regs[i][126:128] = int(80 if i <= 10 else 20).to_bytes(2, "little", signed=True)
    return w


def set_legs(w, ma_each):
    def go():
        for i in range(1, 11):
            w.regs[i][126:128] = int(ma_each).to_bytes(2, "little", signed=True)
    return go


def grip(pads):
    return hook(pads, {"pressed": set(GRIP)}, lambda p, n, f: any(c != p.REST for c in p.served[n:]))


class InTheLoop(WiredRun):
    def setUp(self):
        # the fake loop runs its simulated clock far faster than the pad reader thread's wall clock, so every block
        # looks old; the age rule itself is covered by Estimator.test_a_stale_touch_block_counts_as_no_hand
        patch = unittest.mock.patch.object(hold, "TOUCH_STALE_S", 1e9)
        patch.start()
        self.addCleanup(patch.stop)

    def run_hold(self, on, hooks, end=5.0, cmd=WALK0, cal=None, link=None, cmd_hooks=()):
        pads = FakePads()
        FakePads.PRESS = 3000
        self.world = legs_world()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hold.json")
            if on is not None:
                hold.write_hold_file(on, cal if cal is not None else {"legs_stand_ma": 800, "legs_air_ma": 100}, path)
            text = self.run_loop(self.world, cmd, end=end, touch_args=("--touch-bus", "0", "--hold-file", path),
                                 pads=pads, hooks=[(t, f(pads) if f in (grip, release) else f) for t, f in hooks],
                                 link=link, cmd_hooks=cmd_hooks)
        return text

    def policy_rows(self, t0, t1):
        return [ln for ln in self.rows(t0, t1) if ln.get("action") is not None]

    def test_off_logs_the_estimate_and_changes_nothing(self):
        text = self.run_hold(None, [(1.5, grip), (2.0, set_legs(self.world_ref(), 10))])
        self.assertIn("hold: off (estimate logged only)", text)
        rows = self.policy_rows(2.5, 5.0)
        self.assertTrue(rows)
        self.assertTrue(all(not ln["hold_on"] and "action_raw" not in ln for ln in rows))
        self.assertGreater(rows[-1]["hold"]["e"], 0.9)  # the hand is seen
        self.assertEqual({ln["kp"] for ln in self.rows(1.0)}, {200})

    def world_ref(self):
        # hooks are built before run_hold creates the world: hand them a proxy that resolves at call time
        test = self

        class Proxy:
            regs = property(lambda _: test.world.regs)
        return Proxy()

    def test_on_lift_by_the_head_scales_the_actions_and_the_gains_then_hands_back(self):
        text = self.run_hold(True, [(1.5, grip), (2.5, set_legs(self.world_ref(), 10)),
                                    (3.5, set_legs(self.world_ref(), 80)), (4.0, release)], end=5.0)
        self.assertIn("hold: ON", text)
        held = self.policy_rows(3.2, 3.5)  # gripped and in the air
        self.assertTrue(held and all(ln["hold_on"] for ln in held))
        for ln in held:
            self.assertLess(max(abs(a) for a in ln["action"]), 0.05 * max(1e-6, max(abs(a) for a in ln["action_raw"])) + 1e-3)
        last = held[-1]
        self.assertLess(last["hold"]["s"], 0.05)
        self.assertEqual(last["kp"]["left_knee"], hold.KP_LEG_MIN)
        self.assertEqual(last["kp"]["neck_pitch"], hold.KP_NECK_HOLD)
        p_gain = [e for e in self.world.log if e[0] == ww.ADDR_P_GAIN]
        self.assertTrue(any(i == 4 and v == hold.KP_LEG_MIN for _, i, v in p_gain))  # right_hip_yaw
        self.assertTrue(any(i == 12 and v == hold.KP_NECK_HOLD for _, i, v in p_gain))  # neck_pitch
        down = self.policy_rows(3.6, 4.0)  # set down, hand still on: legs at full gain again
        knee = [ln["kp"] if isinstance(ln["kp"], int) else ln["kp"]["left_knee"] for ln in down]
        self.assertTrue(all(k >= 170 for k in knee[4:]))
        self.assertTrue(all(k == 200 for k in knee[13:]))
        after = self.policy_rows(4.25, 5.0)  # hand off
        self.assertTrue(after and all(ln["hold"]["s"] == 1.0 for ln in after))
        self.assertEqual({ln["kp"] for ln in after}, {200})
        self.assertNotIn("STOP ->", text)

    def test_on_without_leg_calibration_stays_off(self):
        text = self.run_hold(True, [(1.5, grip), (2.5, set_legs(self.world_ref(), 10))], cal={})
        self.assertIn("leg load not calibrated: it stays off", text)
        rows = self.policy_rows(2.0, 5.0)
        self.assertTrue(rows and all(not ln["hold_on"] for ln in rows))

    def test_switching_the_hold_off_while_held_ramps_the_authority_back(self):
        state = {}

        def switch_off():
            hold.write_hold_file(False, {"legs_stand_ma": 800, "legs_air_ma": 100}, state["path"])

        pads = FakePads()
        self.world = legs_world()
        with tempfile.TemporaryDirectory() as d:
            state["path"] = os.path.join(d, "hold.json")
            hold.write_hold_file(True, {"legs_stand_ma": 800, "legs_air_ma": 100}, state["path"])
            text = self.run_loop(self.world, WALK0, end=5.0,
                                 touch_args=("--touch-bus", "0", "--hold-file", state["path"]), pads=pads,
                                 hooks=[(1.0, grip(pads)), (3.5, switch_off)])
        self.assertIn("hold: off", text)
        s = [ln.get("hold_s", 1.0) for ln in self.policy_rows(3.0, 5.0)]
        self.assertLess(min(s), 0.5)  # held by the hand alone before the switch
        steps = [b - a for a, b in zip(s, s[1:])]
        self.assertLessEqual(max(steps), hold.UP_RATE * 0.02 * 1.6)  # a ramp of a few ticks, never one jump
        self.assertEqual(s[-1], 1.0)

    def test_gains_are_written_again_after_the_hold_has_changed_them(self):
        self.run_hold(True, [(1.5, grip), (2.0, set_legs(self.world_ref(), 10)),
                             (2.8, set_legs(self.world_ref(), 80)), (3.0, release)], end=8.0)
        knee_writes = [e for e in self.world.log if e[0] == ww.ADDR_P_GAIN and e[1] == ww.IDS[ww.NAMES.index("left_knee")]]
        late = [e for e in knee_writes if e[2] == 200]
        self.assertGreaterEqual(len(late), 4)  # the restore, then the refresh every second while the window lasts

    def test_hold_allowed_only_on_a_standing_or_rested_policy(self):
        class B:
            mode, stepped = "walk", False
        b = B()
        self.assertTrue(ww.hold_allowed(WALK0, b, 0.0))
        self.assertFalse(ww.hold_allowed(WALK0, b, None))       # a velocity command
        b.stepped = True
        self.assertFalse(ww.hold_allowed(WALK0, b, 1.0))        # walk61 still bringing its stride to rest
        self.assertTrue(ww.hold_allowed(WALK0, b, ww.QUACK_REST_S))
        b.mode = "sit"
        self.assertFalse(ww.hold_allowed(WALK0, b, 5.0))        # a sit standing up under a walk command
        self.assertFalse(ww.hold_allowed({"mode": "stand"}, b, 5.0))
        b.mode, b.stepped = "walk", True
        self.assertTrue(ww.hold_allowed(WALK0, b, None, grip_stopped=True))  # stopped by a grip: at once
        self.assertFalse(ww.hold_allowed({"mode": "stand"}, b, None, grip_stopped=True))

    def test_grip_stop_latches_on_a_grip_and_clears_only_on_a_zero_command(self):
        walk = {"mode": "walk", "vx": 0.2, "vy": 0.0, "vtheta": 0.1}
        g = ww.GripStop()
        self.assertIsNone(g.update(walk, True, ww.GRIP_STOP_E))       # a brush at the threshold: keeps walking
        self.assertIsNone(g.update(walk, False, 1.0))                  # hold off or uncalibrated: keeps walking
        self.assertEqual(g.apply(walk), walk)
        self.assertEqual(g.update(walk, True, 0.5), "stop")
        self.assertEqual(g.apply(walk), dict(walk, vx=0.0, vtheta=0.0))
        self.assertIsNone(g.update(walk, True, 0.0))                   # the hand leaves: still stopped
        self.assertTrue(g.on)
        self.assertEqual(g.update(WALK0, True, 0.0), "clear")
        self.assertIsNone(g.update(walk, True, 0.0))                   # pushed again without a hand: walks
        self.assertFalse(g.on)
        g.update(walk, True, 0.5)
        self.assertEqual(g.update({"mode": "stand"}, True, 0.5), "clear")

    def test_grip_stop_hands_the_set_down_to_stand61_only_once_fully_held(self):
        class B:
            mode, stepped = "walk", True
        b, g = B(), ww.GripStop()
        self.assertFalse(g.settle(b, 0.0))                             # no grip stop: walk61 stays
        g.update({"mode": "walk", "vx": 0.2}, True, 0.5)
        self.assertFalse(g.settle(b, 0.4))                             # the hand alone, feet loaded: walk61 stays
        self.assertTrue(b.stepped)
        self.assertTrue(g.settle(b, ww.GRIP_STOP_HELD_S))
        self.assertFalse(b.stepped)
        self.assertFalse(g.settle(b, 0.0))                             # once

    def test_a_malformed_hold_file_is_reported_and_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hold.json")
            with open(path, "w") as f:
                json.dump({"on": True, "cal": {"legs_stand_ma": "900", "legs_air_ma": 100}}, f)
            self.world = legs_world()
            text = self.run_loop(self.world, WALK0, end=2.0, touch_args=("--touch-bus", "none", "--hold-file", path))
        self.assertIn("malformed hold file", text)
        self.assertTrue(all(not ln["hold_on"] for ln in self.log_lines))

    def test_on_a_grip_while_walking_stops_and_holds_at_once(self):
        walk = {"mode": "walk", "vx": 0.2, "vy": 0.0, "vtheta": 0.0}
        text = self.run_hold(True, [(1.5, grip), (2.5, set_legs(self.world_ref(), 10)),
                                    (3.5, set_legs(self.world_ref(), 80)), (4.0, release)], end=6.0, cmd=walk)
        self.assertIn("hold: head gripped while walking, stopped", text)
        before = self.policy_rows(1.0, 1.5)
        self.assertTrue(before and all(not ln["grip_stop"] and ln["cmd"]["vx"] == 0.2 for ln in before))
        stopped = self.policy_rows(2.2, 6.0)  # latched from the grip on, the hand gone or not
        self.assertTrue(stopped and all(ln["grip_stop"] and ln["cmd"]["vx"] == 0.0 for ln in stopped))
        held = self.policy_rows(3.2, 3.5)  # lifted: held without waiting for the stride to settle
        self.assertTrue(held and all(ln["hold_on"] for ln in held))
        self.assertLess(held[-1]["hold"]["s"], 0.05)
        self.assertIn("hold: fully held after the grip stop, the set-down runs stand61", text)
        self.assertEqual(self.policy_rows(4.25, 6.0)[-1]["policy"], "stand")
        after = self.policy_rows(4.25, 6.0)
        self.assertTrue(after and all(ln["hold"]["s"] == 1.0 for ln in after))
        self.assertNotIn("STOP ->", text)

    def test_a_grip_stop_clears_on_a_zero_command_and_walks_on_a_new_push(self):
        walk = {"mode": "walk", "vx": 0.2, "vy": 0.0, "vtheta": 0.0}
        text = self.run_hold(True, [(1.5, grip), (3.0, release)], end=6.0, cmd=walk,
                             cmd_hooks=[(4.0, WALK0), (5.0, walk)])
        self.assertIn("hold: grip stop cleared by a zero or non-walk command", text)
        self.assertTrue(all(ln["grip_stop"] and ln["cmd"]["vx"] == 0.0 for ln in self.policy_rows(2.2, 4.0)))
        again = self.policy_rows(5.2, 6.0)
        self.assertTrue(again and all(not ln["grip_stop"] and ln["cmd"]["vx"] == 0.2 for ln in again))

    def test_off_a_grip_while_walking_changes_nothing(self):
        walk = {"mode": "walk", "vx": 0.2, "vy": 0.0, "vtheta": 0.0}
        text = self.run_hold(False, [(1.5, grip), (2.5, set_legs(self.world_ref(), 10))], cmd=walk)
        self.assertNotIn("head gripped while walking", text)
        rows = self.policy_rows(2.0, 5.0)
        self.assertTrue(rows and all(not ln["hold_on"] and not ln["grip_stop"] and ln["cmd"]["vx"] == 0.2
                                     for ln in rows))
        self.assertEqual({ln["kp"] for ln in self.rows(1.0)}, {200})

    def test_the_hold_file_switches_it_at_run_time(self):
        state = {}

        def switch_on():
            hold.write_hold_file(True, {"legs_stand_ma": 800, "legs_air_ma": 100}, state["path"])

        pads = FakePads()
        self.world = legs_world()
        with tempfile.TemporaryDirectory() as d:
            state["path"] = os.path.join(d, "hold.json")
            text = self.run_loop(self.world, WALK0, end=4.0,
                                 touch_args=("--touch-bus", "0", "--hold-file", state["path"]), pads=pads,
                                 hooks=[(1.0, grip(pads)), (2.0, switch_on)])
        self.assertIn("hold: ON", text)
        self.assertTrue(all(not ln["hold_on"] for ln in self.policy_rows(1.0, 2.0)))
        self.assertTrue(all(ln["hold_on"] for ln in self.policy_rows(2.6, 4.0)))

    def test_carried_tilted_does_not_trip_the_tilt_stop_but_a_fall_without_a_hand_does(self):
        for gripped in (True, False):
            hooks = [(1.0, grip)] if gripped else []
            hooks.append((2.0, lambda: setattr(self.world, "tilt_deg", 45)))
            text = self.run_hold(True, hooks, end=3.5, link=tilted_link(self))
            if gripped:
                self.assertNotIn("body tilt over", text)
            else:
                self.assertIn("STOP -> stand: body tilt over 30.0 deg", text)


def tilted_link(test):
    """The fake link of test_walk_wired_battery, with the IMU quaternion tilted about x by world.tilt_deg."""
    import math
    import test_walk_wired_battery as twb
    make = twb.make_link  # the original: run_loop patches twb.make_link with this factory

    def factory(world, clock, reads):
        base = make(world, clock, reads)

        class Tilted(base):
            def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
                out = super().sync_read(addr, size, ids, timeout, allow_alert)
                deg = getattr(world, "tilt_deg", 0)
                if deg and self.imu is not None:
                    t, f = self.imu
                    h = math.radians(deg) / 2
                    q = twb.UPRIGHT_SENSOR_QUAT
                    rot = (math.cos(h), math.sin(h), 0.0, 0.0)
                    qq = (rot[0] * q[0] - rot[1] * q[1] - rot[2] * q[2] - rot[3] * q[3],
                          rot[0] * q[1] + rot[1] * q[0] + rot[2] * q[3] - rot[3] * q[2],
                          rot[0] * q[2] - rot[1] * q[3] + rot[2] * q[0] + rot[3] * q[1],
                          rot[0] * q[3] + rot[1] * q[2] - rot[2] * q[1] + rot[3] * q[0])
                    self.imu = (t, (f[0], *qq, *f[5:]))
                return out
        return Tilted
    return factory

if __name__ == "__main__":
    unittest.main()
