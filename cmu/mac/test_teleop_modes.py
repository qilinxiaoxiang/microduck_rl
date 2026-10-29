"""Remote-control additions end to end on the fake link: kick (one shot per id, hand-back to stand61, guards), sit and
stand-up flag timing, the quack key (one player per new counter value, never on the control thread, jaw opening) and
hold-to-walk with for_s. No hardware and no physics: the fake world turns goals into present positions at once, so
only the plumbing and the timing are checked; brain_sim_check.py runs the same Brain in the as-built model.

  ~/.venvs/microduck-walk/bin/python -m unittest test_teleop_modes -v
"""
import json
import math
import os
import sys
import tempfile
import threading
import time as real_time
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import duck_sound  # noqa: E402
import jaw_ctl  # noqa: E402
import walk_brain as wb  # noqa: E402
import walk_wired as ww  # noqa: E402
from test_battery import World  # noqa: E402
import test_walk_wired_battery as twb  # noqa: E402  (module import: its TestCase is not collected twice)
from test_walk_wired_battery import UPRIGHT_SENSOR_QUAT, conj, make_link as make_link_for  # noqa: E402

JAW = jaw_ctl.JAW_ID
TICKS_PER_DEG = 4096 / 360
WALK0 = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}


def write_cmd(path, cmd):
    def hook():
        with open(path, "w") as fh:
            json.dump(cmd, fh)
    return hook


# The as-built sim's seat (brain_sim_check --sit-at 3, t = 12 s), NAMES order; a seat folds both knees to about 1.3 rad.
SEAT_Q = [-0.0798, -1.31, 0.4587, -0.1703, 0.0138, 0.089, 1.3085, -0.4648, 0.1703, 0.0706, 0.3559, 0.3728, -0.0046, 0.0383]
FLAG_SLOT = 3 + 3 + 14 + 14 + 14  # gyro, gravity, q, dq, last action, then the command: posture flag in its vx slot


class ScriptedSitStand:
    """sitstand61 stand-in for the fake world (whose present positions follow the goals at once, with no physics):
    flag 1 asks for the seat, flag 0 for the policy default pose (knees straight), or for the seat again when
    `rises` is False (a rise that never comes up, as the knees of the 2026-10-29 fake_pi seats)."""

    def __init__(self, real, default, rises=True):
        self.real, self.rises = real, rises
        seat = dict(zip(wb.NAMES, SEAT_Q))
        self.sit = np.array([seat[n] - default[n] for n in wb.OBSERVATION_DOF_ORDER], dtype=np.float32)

    def get_inputs(self):
        return self.real.get_inputs()

    def get_modelmeta(self):
        return self.real.get_modelmeta()

    def run(self, outputs, feeds):
        obs = next(iter(feeds.values()))[0]
        sitting = obs[FLAG_SLOT] > 0.5 or not self.rises
        return [(self.sit if sitting else np.zeros(14, dtype=np.float32))[None]]


def brain_with_sitstand(rises=True):
    class SitBrain(wb.Brain):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.sessions["sitstand"] = ScriptedSitStand(self.sessions["sitstand"], self.default, rises)

    return SitBrain


def tilted_sensor_quat(deg):
    """Sensor quaternion of a body pitched forward by deg (same frame chain as UPRIGHT_SENSOR_QUAT)."""
    h = math.radians(deg) / 2
    q_body = (math.cos(h), 0.0, math.sin(h), 0.0)
    return wb.qmul(wb.qmul(q_body, conj(wb.IMU_MOUNT_QUAT)), conj(wb.MOUNT_FIX))


def tilting_link(at, deg):
    """make_link replacement whose IMU reports a deg pitch from fake time `at` on."""
    def factory(world, clock, reads):
        base = make_link_for(world, clock, reads)

        class Tilting(base):
            def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
                got = super().sync_read(addr, size, ids, timeout, allow_alert)
                if clock.t >= at and self.imu is not None:
                    q = tilted_sensor_quat(deg)
                    self.imu = (self.imu[0], (self.imu[1][0], *q) + tuple(self.imu[1][5:]))
                return got
        return Tilting
    return factory


class FakeProc:
    def __init__(self, argv, log):
        self.argv, self.killed, self.log = argv, False, log

    def poll(self):
        return None if not self.killed else -9

    def kill(self):
        self.killed = True
        self.log.append(("kill", self.argv))

    def wait(self, timeout=None):
        return -9


class Teleop(unittest.TestCase):
    run_wired = twb.WiredBattery.run_wired  # the harness only: WiredBattery's own tests run in their module
    first = twb.WiredBattery.first

    def kicks(self):
        return [ln for ln in self.log_lines if str(ln.get("policy", "")).startswith("kick_")]

    def test_kick_runs_25_ticks_once_and_hands_back_to_stand61(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        state = {}

        def hooks(d, cmd_path):
            state["path"] = cmd_path
            return [(2.0, write_cmd(cmd_path, {"mode": "kick", "foot": "left", "id": 1, "quack": 0})),
                    (3.5, write_cmd(cmd_path, {"mode": "kick", "foot": "left", "id": 1})),   # same id again
                    (4.5, write_cmd(cmd_path, {"mode": "kick", "foot": "right", "id": 2}))]  # a new id kicks
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=6.0, supply="bench",
                                                              hooks=hooks)
        self.assertEqual(text.count("mode -> kick"), 2, text)
        kicks = self.kicks()
        left = [ln for ln in kicks if ln["policy"] == "kick_left"]
        right = [ln for ln in kicks if ln["policy"] == "kick_right"]
        self.assertEqual(len(left), 25)  # Pollen: 0.5 s at 50 Hz
        self.assertEqual(len(right), 25)
        self.assertEqual([ln["kick_tick"] for ln in left], list(range(1, 26)))
        self.assertTrue(all(ln["kp"] == wb.KP_RL for ln in kicks))
        # the kick's 25 ticks are consecutive and start at the command
        t_left = [ln["t"] - 1e9 for ln in left]
        self.assertAlmostEqual(t_left[0], 2.0, delta=0.05)
        self.assertAlmostEqual(t_left[-1] - t_left[0], 24 * 0.02, delta=0.03)
        self.assertIn("STOP -> zero command: kick 1 done, stand61 takes over", text)
        self.assertIn("STOP -> zero command: kick 2 done, stand61 takes over", text)
        # between and after the kicks: walk mode on stand61 (no step was ever commanded)
        after = [ln for ln in self.log_lines if 2.6 < ln["t"] - 1e9 < 4.4 and ln.get("policy")]
        self.assertTrue(after and all(ln["mode"] == "walk" and ln["policy"] == "stand" for ln in after))
        self.assertEqual(last_log["mode"], "walk")
        self.assertEqual(last_log["policy"], "stand")
        self.assertEqual(last_log["cmd"], WALK0)

    def test_kick_keeps_the_action_history(self):
        brain = wb.Brain(os.path.join(wb.WHOLE, "runtime-overlay", "models"))
        seen = []

        class Spy:
            def __init__(self, name, sess):
                self.name, self.sess = name, sess

            def get_inputs(self):
                return self.sess.get_inputs()

            def run(self, out, feeds):
                obs = next(iter(feeds.values()))[0]
                seen.append((self.name, obs[34:48].copy()))
                return self.sess.run(out, feeds)
        brain.sessions = {k: Spy(k, s) for k, s in brain.sessions.items()}
        pos = [wb.to_tick(n, wb.STAND[n]) for n in wb.NAMES]
        for k in range(30):
            want = WALK0 if k < 10 else {"mode": "kick", "foot": "right", "id": "a"}
            brain.tick(want, 100 + k * 0.02, pos, [0] * 14, UPRIGHT_SENSOR_QUAT, (0.0, 0.0, 0.0), True, 0.004, 0.006)
        first_kick = next(i for i, (nm, _) in enumerate(seen) if nm == "kick_right")
        prev_name, prev_obs_action = seen[first_kick - 1]
        self.assertEqual(prev_name, "stand")
        # the kick's first observation carries stand61's last action, not zeros
        self.assertGreater(float(abs(seen[first_kick][1]).max()), 1e-3)

    def test_kick_is_refused_outside_stand_and_walk_and_never_fires_later(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(2.0, write_cmd(p, WALK0))]  # noqa: E731
        text, marker, wants, reads, last_log = self.run_wired(
            w, lambda t: 74, {"mode": "kick", "foot": "left", "id": 9}, end=3.0, supply="bench", hooks=hooks)
        self.assertIn("kick 9 refused: only from stand or walk mode (now monitor)", text)
        self.assertNotIn("mode -> kick", text)
        self.assertFalse(self.kicks())
        self.assertEqual(last_log["mode"], "walk")

    def test_tilt_guard_stops_a_kick_into_stand(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(2.0, write_cmd(p, {"mode": "kick", "foot": "left", "id": 1}))]  # noqa: E731
        with mock.patch.object(twb, "make_link", tilting_link(2.2, 40.0)):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=3.5, supply="bench",
                                                                  hooks=hooks)
        self.assertIn("mode -> kick", text)
        self.assertIn("STOP -> stand: body tilt over 30.0 deg", text)
        self.assertIn("kick 1 stopped: body tilt over 30.0 deg", text)
        self.assertLess(len(self.kicks()), wb.KICK_TICKS)
        self.assertEqual(last_log["mode"], "stand")
        self.assertTrue(last_log["torque"])

    def test_sit_flag_timing_and_stand_up(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (5.0, write_cmd(p, WALK0))]  # noqa: E731
        with mock.patch.object(ww, "Brain", brain_with_sitstand()):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=8.5, supply="bench",
                                                                  hooks=hooks)
        sit = [ln for ln in self.log_lines if ln.get("policy") == "sitstand"]
        self.assertTrue(sit)
        t_sit0 = sit[0]["t"] - 1e9
        flag1 = [ln["t"] - 1e9 for ln in sit if ln["sit_flag"] == 1.0]
        self.assertAlmostEqual(t_sit0, 1.0, delta=0.05)
        self.assertAlmostEqual(flag1[0] - t_sit0, 0.8, delta=0.03)  # flag 0 for 0.8 s first
        self.assertTrue(all(ln["sit_flag"] == 0.0 for ln in sit if ln["t"] - 1e9 < flag1[0]))
        rise = [ln["t"] - 1e9 for ln in sit if ln["sit_phase"] == "rise"]
        self.assertAlmostEqual(rise[0], 5.0, delta=0.05)  # seated by then: the stand-up starts at the command
        self.assertTrue(all(ln["sit_flag"] == 0.0 for ln in sit if ln["sit_phase"] == "rise"))
        back = next(ln["t"] - 1e9 for ln in self.log_lines if ln["t"] - 1e9 > 5.0 and ln.get("policy") == "stand")
        # flag 0 until up: the stand-in straightens the knees on the first flag-0 tick, then RISE_UP_S of evidence
        self.assertAlmostEqual(back - rise[0], wb.RISE_UP_S, delta=0.07)
        self.assertIn("sit: up after", text)
        self.assertTrue(all(ln["kp"] == wb.KP_RL for ln in sit))
        self.assertEqual(last_log["mode"], "walk")
        self.assertEqual(last_log["policy"], "stand")

    def test_stand_up_asked_during_the_glide_waits_for_the_seat(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (2.3, write_cmd(p, WALK0))]  # noqa: E731
        with mock.patch.object(ww, "Brain", brain_with_sitstand()):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=7.5, supply="bench",
                                                                  hooks=hooks)
        sit = [ln for ln in self.log_lines if ln.get("policy") == "sitstand"]
        flag1 = next(ln["t"] - 1e9 for ln in sit if ln["sit_flag"] == 1.0)
        rise = next(ln["t"] - 1e9 for ln in sit if ln["sit_phase"] == "rise")
        self.assertAlmostEqual(rise - flag1, 2.0, delta=0.05)
        self.assertEqual(last_log["policy"], "stand")

    def test_a_rise_that_never_comes_up_sits_back_down_and_waits_for_a_sit(self):
        # 2026-10-29 (sit-rise-20261029): through fake_pi the rise cycled 1.4 -> 0.6 -> 1.4 rad and never finished; the
        # 2 s timer handed stand61 a random point of the cycle and 8 of 8 seats fell
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (4.0, write_cmd(p, WALK0))]  # noqa: E731
        with mock.patch.object(ww, "Brain", brain_with_sitstand(rises=False)):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=12.0, supply="bench",
                                                                  hooks=hooks)
        sit = [ln for ln in self.log_lines if ln.get("policy") == "sitstand"]
        rise = next(ln["t"] - 1e9 for ln in sit if ln["sit_phase"] == "rise")
        back_down = next(ln["t"] - 1e9 for ln in sit if ln["t"] - 1e9 > rise and ln["sit_phase"] == "down")
        self.assertAlmostEqual(back_down - rise, wb.RISE_GIVEUP_S, delta=0.05)
        self.assertEqual(text.count("sit: RISE FAILED"), 1)
        self.assertFalse(any(ln.get("policy") == "stand" and ln["t"] - 1e9 > rise for ln in self.log_lines))
        self.assertTrue(all(ln["sit_flag"] == 1.0 for ln in sit if ln["t"] - 1e9 >= back_down))
        # the control file still says walk: the seat holds, no second rise until {"mode": "sit"} re-arms it
        self.assertEqual(sum(1 for a, b in zip(sit, sit[1:]) if a["sit_phase"] != "rise" and b["sit_phase"] == "rise"), 1)
        self.assertEqual(last_log["mode"], "sit")
        self.assertEqual(last_log["sit_phase"], "seated")

    def test_a_failed_rise_is_re_armed_by_a_sit_and_the_next_stand_up_hands_over_when_up(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        stand_ins = []

        class SitBrain(wb.Brain):  # the stand-in never comes up until the hook below lets it
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.sessions["sitstand"] = ScriptedSitStand(self.sessions["sitstand"], self.default, rises=False)
                stand_ins.append(self.sessions["sitstand"])

        def hooks(d, p):
            def lets_it_rise():
                stand_ins[-1].rises = True
            return [(1.0, write_cmd(p, {"mode": "sit"})), (4.0, write_cmd(p, WALK0)), (10.9, lets_it_rise),
                    (11.0, write_cmd(p, {"mode": "sit"})), (11.5, write_cmd(p, WALK0))]

        with mock.patch.object(ww, "Brain", SitBrain):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=14.0, supply="bench",
                                                                  hooks=hooks)
        self.assertEqual(text.count("sit: RISE FAILED"), 1)
        self.assertEqual(text.count("sit: stand-up re-armed"), 1)
        self.assertEqual(text.count("sit: up after"), 1)
        second = [ln["t"] - 1e9 for ln in self.log_lines if ln.get("sit_phase") == "rise" and ln["t"] - 1e9 > 11.0]
        back = next(ln["t"] - 1e9 for ln in self.log_lines if ln["t"] - 1e9 > 11.0 and ln.get("policy") == "stand")
        self.assertAlmostEqual(back - second[0], wb.RISE_UP_S, delta=0.07)
        self.assertEqual(last_log["mode"], "walk")

    def test_straight_knees_on_a_tilted_body_are_not_up(self):
        # lying on its back after a failed rise the knees are straight too (in-process sim, 2026-10-29): the trunk must
        # be upright before stand61 takes over
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (4.0, write_cmd(p, WALK0))]  # noqa: E731
        with mock.patch.object(ww, "Brain", brain_with_sitstand()), \
                mock.patch.object(twb, "make_link", tilting_link(4.0, 20.0)):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=9.0, supply="bench",
                                                                  hooks=hooks)
        self.assertNotIn("STOP", text)  # 20 deg: inside walk_wired's 30 deg sit guard
        self.assertNotIn("sit: up after", text)
        self.assertIn("sit: RISE FAILED", text)
        # 20 deg forward less the 4 deg IMU pitch trim, as the brain reads it
        self.assertIn("knees 0.00 rad, pitch rate 0.00 rad/s, tilt 16.0 deg", text)
        self.assertFalse(any(ln.get("policy") == "stand" and ln["t"] - 1e9 > 4.0 for ln in self.log_lines))

    def test_rise_route_hands_over_only_on_held_evidence(self):
        brain = wb.Brain.__new__(wb.Brain)
        brain.kick, brain.mode, brain.sit_next = None, "sit", None
        brain.base_want, brain._refused = {"mode": "sit"}, None
        brain.sit = {"phase": "seated", "t": 0.0}
        self.assertEqual(brain.route(WALK0, 10.0), {"mode": "sit"})  # the stand-up starts
        self.assertEqual(brain.sit["phase"], "rise")
        self.assertEqual(brain.route(WALK0, 12.5), {"mode": "sit"})  # 2.5 s, no evidence: the old timer is gone
        brain.sit["up_since"] = 12.6
        self.assertEqual(brain.route(WALK0, 12.6 + wb.RISE_UP_S - 0.02), {"mode": "sit"})
        self.assertEqual(brain.route(WALK0, 12.6 + wb.RISE_UP_S), WALK0)
        brain.sit = {"phase": "rise", "t": 10.0, "up_since": None}
        self.assertEqual(brain.route(WALK0, 10.0 + wb.RISE_GIVEUP_S), {"mode": "sit"})
        self.assertEqual((brain.sit["phase"], brain.sit["failed"]), ("down", True))
        self.assertEqual(brain.route(WALK0, 20.0), {"mode": "sit"})  # seated again; the walk command waits
        self.assertEqual(brain.sit["phase"], "seated")
        self.assertEqual(brain.route({"mode": "monitor"}, 20.1), {"mode": "monitor"})  # a hold still leaves at once

    def test_monitor_leaves_a_sit_at_once_and_sit_from_a_standing_monitor_is_refused(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (1.4, write_cmd(p, {"mode": "monitor"})),  # noqa: E731
                              (2.0, write_cmd(p, {"mode": "sit"}))]
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=3.0, supply="bench",
                                                              hooks=hooks)
        self.assertEqual(text.count("sit refused: only from stand or walk mode, or from monitor mode while seated "
                                    "(now monitor)"), 1)
        self.assertEqual(last_log["mode"], "monitor")
        self.assertTrue(last_log["torque"])

    def test_a_brain_restarted_on_a_seated_robot_sits_without_standing_up(self):
        # the seated pose of the as-built sim (brain_sim_check --sit-at 3, t = 12 s), servos still holding it
        seat = [-0.0798, -1.31, 0.4587, -0.1703, 0.0138, 0.089, 1.3085, -0.4648, 0.1703, 0.0706, 0.3559, 0.3728,
                -0.0046, 0.0383]
        w = World(shutdown=52, hw_err=0, volts_dv=74, torque=1)
        for i, n, q in zip(wb.IDS, wb.NAMES, seat):
            w.regs[i][132:136] = wb.to_tick(n, q).to_bytes(4, "little")
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"}))]  # noqa: E731
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "monitor"}, end=2.5,
                                                              supply="bench", hooks=hooks)
        self.assertNotIn("sit refused", text)
        self.assertIn("torque on at present positions", text)
        sit = [ln for ln in self.log_lines if ln.get("policy") == "sitstand"]
        self.assertTrue(sit)
        self.assertTrue(all(ln["sit_phase"] == "seated" and ln["sit_flag"] == 1.0 for ln in sit))
        self.assertFalse(sit[-1]["sit_stalled"])
        self.assertEqual(last_log["mode"], "sit")

    def test_tilt_guard_in_a_sit_holds_the_last_goals(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"}))]  # noqa: E731
        with mock.patch.object(twb, "make_link", tilting_link(2.5, 60.0)):
            text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=3.5, supply="bench",
                                                                  hooks=hooks)
        self.assertIn("STOP -> hold the last goals: sit: body tilt over 30.0 deg", text)
        self.assertEqual(last_log["mode"], "monitor")
        self.assertNotIn("mode -> stand", text)

    def run_quack(self, hooks, cmd, end, cal=None, jaw_tick=1500):
        spawned = []
        with tempfile.TemporaryDirectory() as d:
            sounds = os.path.join(d, "sounds")
            os.makedirs(sounds)
            for k in range(3):
                open(os.path.join(sounds, f"call_{k}.wav"), "wb").close()
            cal_path = os.path.join(d, "jaw_calibration.json")
            if cal:
                with open(cal_path, "w") as fh:
                    json.dump(cal, fh)
            voices = []

            def make_voice(path):
                v = duck_sound.Voice(path, popen=lambda argv, **kw: (spawned.append(argv), FakeProc(argv, spawned))[1])
                voices.append(v)
                return v
            w = World(shutdown=52, hw_err=0, volts_dv=74)
            w.regs[JAW][132:136] = jaw_tick.to_bytes(4, "little")
            with mock.patch.object(ww, "Voice", make_voice), \
                    mock.patch.object(ww, "JawCtl", lambda **kw: jaw_ctl.JawCtl(cal_path, **kw)):
                text, marker, wants, reads, last_log = self.run_wired(
                    w, lambda t: 74, cmd, end=end, supply="bench", hooks=hooks,
                    extra_args=("--sounds", sounds, "--max-action", "1e9", "--max-walk-action", "1e9",
                                "--hard-max-action", "1e9"))
            voices[0].q.join()
        return text, last_log, w, spawned, voices[0]

    def test_quack_plays_one_call_per_new_counter_value(self):
        def hooks(d, p):
            return [(1.0, write_cmd(p, dict(WALK0, quack=1))),
                    (1.3, write_cmd(p, dict(WALK0, quack=1))),   # a rewrite with the same counter: silent
                    (1.6, write_cmd(p, dict(WALK0, quack=2))),
                    (1.9, write_cmd(p, {"mode": "kick", "foot": "left", "id": 4, "quack": 3}))]
        text, last_log, w, spawned, voice = self.run_quack(hooks, WALK0, end=3.0)
        players = [x for x in spawned if isinstance(x, list)]
        self.assertEqual(len(players), 3)
        self.assertTrue(all(argv[:2] == ["aplay", "-q"] and len(argv) == 3 and argv[2].endswith(".wav")
                            for argv in players))
        self.assertFalse(any("-D" in argv or any("hw:" in a for a in argv) for argv in players))  # default device only
        kills = [x for x in spawned if isinstance(x, tuple)]
        self.assertEqual(len(kills), 2)  # each new call killed the one still playing
        self.assertEqual(voice.errors, [])
        self.assertIn("quack 3:", text)
        self.assertIn("mode -> kick", text)  # a quack rides along with any command

    def test_quack_never_blocks_the_caller(self):
        release = threading.Event()

        def slow_popen(argv, **kw):
            release.wait(2.0)  # a stuck fork/exec or device open
            return FakeProc(argv, [])
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.wav"), "wb").close()
            v = duck_sound.Voice(d, popen=slow_popen)
            t0 = real_time.perf_counter()
            for _ in range(5):
                v.quack()
            dt = real_time.perf_counter() - t0
            release.set()
            v.q.join()
        self.assertLess(dt, 0.02)
        self.assertEqual(v.started, 5)

    def test_quack_opens_a_calibrated_jaw_and_puts_it_back(self):
        cal = {"closed_tick": 1500, "open_tick": 1700}

        def hooks(d, p):
            return [(1.0, write_cmd(p, {"mode": "monitor", "quack": 1}))]
        text, last_log, w, spawned, voice = self.run_quack(hooks, {"mode": "monitor", "jaw": 5}, end=2.5, cal=cal)
        goals = [(v) for addr, i, v in w.log if i == JAW and addr == jaw_ctl.ADDR_GOAL]
        self.assertIn(1500 + round(ww.QUACK_JAW_DEG * TICKS_PER_DEG), goals)
        self.assertEqual(goals[-1], 1500 + round(5 * TICKS_PER_DEG))  # back to the opening it had
        self.assertEqual(last_log["jaw_goal"], 5)

    def test_quack_enables_an_unpowered_jaw_presses_it_shut_and_holds_it(self):
        cal = {"closed_tick": 1500, "open_tick": 1700}
        # end after the quack head move (QUACK_HEAD_S) is back: the jaw limits follow the head pose
        text, last_log, w, spawned, voice = self.run_quack(lambda d, p: [(1.0, write_cmd(p, dict(WALK0, quack=1)))],
                                                           WALK0, end=1.5 + ww.QUACK_HEAD_S, cal=cal)
        self.assertEqual(len([x for x in spawned if isinstance(x, list)]), 1)
        self.assertNotIn("jaw stays put", text)
        self.assertIn(JAW, {i for addr, i, v in w.log if addr == jaw_ctl.ADDR_TORQUE})
        goals = [v for addr, i, v in w.log if i == JAW and addr == jaw_ctl.ADDR_GOAL]
        self.assertIn(1500 + round(ww.QUACK_JAW_DEG * TICKS_PER_DEG), goals)
        self.assertIn(1500 + round(ww.QUACK_SHUT_DEG * TICKS_PER_DEG), goals)  # pressed shut
        self.assertEqual(goals[-1], 1500 + round(ww.QUACK_HOLD_DEG * TICKS_PER_DEG))  # then eased to the hold
        self.assertEqual(last_log["jaw_goal"], ww.QUACK_HOLD_DEG)

    def test_torque_off_releases_every_servo_and_reboots_latched_ones(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)

        def overload():
            w.regs[4][70] = 0x20  # right_hip_roll: overload latched while standing

        def hooks(d, p):
            return [(1.5, overload), (2.0, write_cmd(p, {"mode": "off"}))]
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "stand"}, end=3.0,
                                                              supply="bench", hooks=hooks)
        self.assertIn("torque on at present positions", text)
        self.assertIn("torque off, rebooted ['right_hip_roll']", text)
        torque = [(i, v) for a, i, v in w.log if a == ww.ADDR_TORQUE]
        self.assertEqual({i for i, v in torque if v == 0}, set(ww.IDS))
        self.assertEqual(last_log["torque"], False)
        self.assertEqual(last_log["mode"], "monitor")
        self.assertEqual(last_log["cmd"], {"mode": "monitor"})

    def test_quack_head_tilt_is_a_smooth_bump(self):
        ts = [k * 0.02 for k in range(int(ww.QUACK_HEAD_S / 0.02) + 2)]
        share = [ww.quack_head_share(t) for t in ts]
        self.assertEqual(share[0], 0.0)
        self.assertEqual(share[-1], 0.0)
        self.assertAlmostEqual(max(share), 1.0)
        # raised cosine: the steepest tick moves pi/2 * 0.02 / QUACK_HEAD_RAMP_S of the move, never a step
        self.assertLess(max(abs(b - a) for a, b in zip(share, share[1:])), math.pi / 2 * 0.02 / ww.QUACK_HEAD_RAMP_S + 0.01)

    def test_quack_head_moves_wherever_a_policy_rebalances(self):
        # 2026-10-21: quacks while seated and after a walk left the head still; only stand61 had the move
        class B:
            mode, stepped, sit = "walk", False, None
        b = B()
        self.assertIsNone(ww.quack_head_block(WALK0, b, 0.0))  # stand61
        self.assertEqual(ww.quack_head_block(dict(WALK0, vx=0.25), b, None), "walking")
        b.stepped = True  # walk61 at zero command after a walk
        self.assertEqual(ww.quack_head_block(WALK0, b, ww.QUACK_REST_S - 0.5), "the stride is still coming to rest")
        self.assertIsNone(ww.quack_head_block(WALK0, b, ww.QUACK_REST_S))
        b.mode = "sit"
        for phase, why in (("prep", "sitting down"), ("down", "sitting down"), ("seated", None)):
            b.sit = {"phase": phase}
            self.assertEqual(ww.quack_head_block({"mode": "sit"}, b, None), why)
        b.sit = {"phase": "rise"}
        self.assertEqual(ww.quack_head_block(WALK0, b, None), "standing up")
        b.mode = "stand"
        self.assertEqual(ww.quack_head_block({"mode": "stand"}, b, None), "open-loop stand mode cannot rebalance")
        b.mode = "walk"
        self.assertEqual(ww.quack_head_block(dict(WALK0, head={"head_yaw": 0.3}), b, 9.0), "a head command is active")

    def test_seated_head_command_reaches_the_sit_policy(self):
        # Brain.route used to return a bare {"mode": "sit"} while seated, which dropped want["head"]
        brain = wb.Brain.__new__(wb.Brain)
        brain.kick, brain.mode, brain.sit, brain.sit_next = None, "sit", {"phase": "seated", "t": 0.0}, None
        brain.base_want, brain._refused = {"mode": "sit"}, None
        head = {"head_yaw": 0.2}
        self.assertEqual(brain.route({"mode": "sit", "head": head}, 10.0), {"mode": "sit", "head": head})
        self.assertEqual(brain.route({"mode": "sit"}, 10.0), {"mode": "sit"})

    def test_hold_to_walk_keeps_walking_while_rewritten(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        cmd = {"mode": "walk", "vx": 0.03, "vy": 0.0, "vtheta": 0.0, "for_s": 0.6}

        def hooks(d, p):
            return [(0.3 * (k + 1), write_cmd(p, cmd)) for k in range(12)]  # last rewrite at 3.6 s
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, cmd, end=6.0, supply="bench",
                                                              hooks=hooks)
        self.assertEqual(text.count("STOP -> zero command: walk for_s 0.6 elapsed"), 1)
        stop_t = next(ln["t"] - 1e9 for ln in self.log_lines if ln["t"] - 1e9 > 1.0 and "for_s" not in ln["cmd"])
        self.assertAlmostEqual(stop_t, 3.6 + 0.6, delta=0.06)
        self.assertTrue(all(w_.get("vx", 0) > 0 for t, w_ in wants if 0.5 < t < 4.1 and w_.get("mode") == "walk"))
        self.assertEqual(last_log["mode"], "walk")


if __name__ == "__main__":
    unittest.main()
