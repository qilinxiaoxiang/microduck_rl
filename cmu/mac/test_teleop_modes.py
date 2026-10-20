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
        self.assertAlmostEqual(back - rise[0], 2.0, delta=0.05)  # flag 0 for 2 s, then stand61
        self.assertTrue(all(ln["kp"] == wb.KP_RL for ln in sit))
        self.assertEqual(last_log["mode"], "walk")
        self.assertEqual(last_log["policy"], "stand")

    def test_stand_up_asked_during_the_glide_waits_for_the_seat(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (2.3, write_cmd(p, WALK0))]  # noqa: E731
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=7.5, supply="bench",
                                                              hooks=hooks)
        sit = [ln for ln in self.log_lines if ln.get("policy") == "sitstand"]
        flag1 = next(ln["t"] - 1e9 for ln in sit if ln["sit_flag"] == 1.0)
        rise = next(ln["t"] - 1e9 for ln in sit if ln["sit_phase"] == "rise")
        self.assertAlmostEqual(rise - flag1, 2.0, delta=0.05)
        self.assertEqual(last_log["policy"], "stand")

    def test_monitor_leaves_a_sit_at_once_and_sit_from_monitor_is_refused(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        hooks = lambda d, p: [(1.0, write_cmd(p, {"mode": "sit"})), (3.0, write_cmd(p, {"mode": "monitor"})),  # noqa: E731
                              (3.5, write_cmd(p, {"mode": "sit"}))]
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, WALK0, end=4.5, supply="bench",
                                                              hooks=hooks)
        self.assertEqual(text.count("sit refused: only from stand or walk mode (now monitor)"), 1)
        self.assertEqual(last_log["mode"], "monitor")
        self.assertTrue(last_log["torque"])

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

    def test_quack_without_jaw_torque_only_plays(self):
        cal = {"closed_tick": 1500, "open_tick": 1700}
        text, last_log, w, spawned, voice = self.run_quack(lambda d, p: [(1.0, write_cmd(p, dict(WALK0, quack=1)))],
                                                           WALK0, end=2.0, cal=cal)
        self.assertEqual(len([x for x in spawned if isinstance(x, list)]), 1)
        self.assertIn("quack: jaw stays put (torque off", text)
        self.assertNotIn(JAW, {i for addr, i, v in w.log if addr == jaw_ctl.ADDR_TORQUE})

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
