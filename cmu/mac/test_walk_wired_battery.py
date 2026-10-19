"""walk_wired.py on a 2S pack, end to end with the real Brain and policies: a fake servo link (goals become present
positions, Alert bit on every packet), an upright IMU and a simulated clock. No hardware.

  ~/.venvs/microduck-walk/bin/python -m unittest test_walk_wired_battery -v
"""
import collections
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import walk_wired as ww  # noqa: E402
from test_battery import World  # noqa: E402
from walk_brain import IMU_MOUNT_QUAT, MOUNT_FIX, STAND, qmul, to_model, to_tick  # noqa: E402
from dxl_fast import _packet  # noqa: E402

IDS = ww.IDS


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


UPRIGHT_SENSOR_QUAT = qmul(qmul((1.0, 0.0, 0.0, 0.0), conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))


class Done(Exception):
    pass


class FakeClock:
    def __init__(self, world, profile, end, hooks=()):
        self.t, self.world, self.profile, self.end, self.hooks = 0.0, world, profile, end, list(hooks)

    def perf_counter(self):
        return self.t

    def time(self):
        return 1e9 + self.t

    def sleep(self, s):
        self.t += max(s, 0.0)
        self.world.set_all(144, 2, self.profile(self.t))
        while self.hooks and self.t >= self.hooks[0][0]:
            self.hooks.pop(0)[1]()
        if self.t >= self.end:
            raise Done


def make_link(world, clock, reads):
    class FakeLink:
        """sync_read / sync_write on the World; goal positions become present positions at once."""

        def __init__(self, port):
            self.ser = mock.Mock()
            self.status = collections.deque()
            self.lock = threading.Lock()
            self.imu, self.imu_count = None, 0
            self.alert_ok = False
            self.late = 0

        def sync_write(self, addr, size, values):
            for i, v in values.items():
                world.write(i, addr, size, v)
                if addr == ww.ADDR_GOAL:
                    world.regs[i][132:136] = int(v).to_bytes(4, "little", signed=True)

        def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
            # a read takes bus time; the torque-refusal path loops on reads without sleeping
            clock.t += 0.002
            if clock.t >= clock.end:
                raise Done
            reads.append(addr)
            if clock.t < getattr(world, "imu_stop_at", float("inf")):  # the Pi stream stops (power cut)
                self.imu = (clock.t, (0, *UPRIGHT_SENSOR_QUAT, 0.0, 0.0, 0.0, 0.0, 0.0, -9.8, 0.002, 1))
                self.imu_count += 1
            alert_ok = self.alert_ok if allow_alert is None else allow_alert  # same rule as walk_wired.Link
            return {i: bytes(world.regs[i][addr:addr + size]) for i in ids
                    if i in world.regs and (world.err(i) == 0 or alert_ok)}

    return FakeLink


class WiredBattery(unittest.TestCase):
    def run_wired(self, world, profile, cmd, end, supply="auto", hooks=lambda d, cmd_path: (), old_marker=False,
                  extra_args=("--max-action", "1e9", "--max-walk-action", "1e9", "--hard-max-action", "1e9")):  # the fake world has no physics: the policy runs away
        reads, wants = [], []

        class SpyBrain(ww.Brain):
            def tick(self, want, now, *a):
                wants.append((now - 1e9, dict(want)))
                return super().tick(want, now, *a)

        with tempfile.TemporaryDirectory() as d:
            cmd_path, marker = os.path.join(d, "cmd.json"), os.path.join(d, "stop.json")
            with open(cmd_path, "w") as fh:
                json.dump(cmd, fh)
            if old_marker:
                with open(marker, "w") as fh:
                    json.dump({"volts": 6.5}, fh)
            clock = FakeClock(world, profile, end, hooks(d, cmd_path))
            argv = ["walk_wired.py", "--port", "fake", "--cmd", cmd_path, "--log", os.path.join(d, "log.jsonl"),
                    "--battery-marker", marker, "--supply", supply] + list(extra_args)
            out = io.StringIO()
            with mock.patch.object(ww, "time", clock), mock.patch.object(ww, "Link", make_link(world, clock, reads)), \
                    mock.patch.object(ww, "Brain", SpyBrain), mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(out):
                with self.assertRaises(Done):
                    ww.main()
            marker_data = None
            if os.path.exists(marker):
                with open(marker) as fh:
                    marker_data = json.load(fh)
            with open(os.path.join(d, "log.jsonl")) as fh:
                self.log_lines = [json.loads(ln) for ln in fh]
            last_log = self.log_lines[-1] if self.log_lines else None
        return out.getvalue(), marker_data, wants, reads, last_log

    def first(self, key, value):
        """Wall time (s from the start) of the first log line with key == value."""
        return next(ln["t"] - 1e9 for ln in self.log_lines if ln.get(key) == value)

    def test_walk_for_s_stops_on_walk61_at_zero_command(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "walk", "vx": 0.03, "for_s": 3},
                                                              end=6.0, supply="bench")
        self.assertIn("STOP -> zero command: walk for_s 3 elapsed", text)
        self.assertEqual(last_log["mode"], "walk")
        # walk61 brings its own stride to rest, as in Pollen's simulator; no swap to stand61 mid-stride
        self.assertEqual(last_log["policy"], "walk")
        self.assertTrue(all(ln.get("policy") in (None, "walk") for ln in self.log_lines))
        # the Brain gets the command through Pollen's cmd_alpha EMA: a ramp up to 0.03, never above it
        early = [w_.get("vx") for t, w_ in wants if t < 2.5 and w_.get("mode") == "walk"]
        self.assertTrue(all(0.0 <= v <= 0.03 + 1e-9 for v in early))
        self.assertTrue(all(b_ >= a_ - 1e-12 for a_, b_ in zip(early, early[1:])))
        self.assertAlmostEqual(early[-1], 0.03, places=4)
        self.assertTrue(all(w_ == {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0} for t, w_ in wants if t > 4.0))  # the EMA reaches zero ~0.5 s after the stop
        self.assertEqual({v for a, i, v in w.log if a == ww.ADDR_TORQUE}, {1})

    def test_imu_loss_is_reported_and_freezes_the_policy(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        w.imu_stop_at = 3.0
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "walk", "vx": 0.03}, end=5.0,
                                                              supply="bench")
        self.assertEqual(text.count("IMU LOST"), 1)
        lost_t = next(ln["t"] - 1e9 for ln in self.log_lines if ln["imu_age"] > ww.IMU_LOST_S and ln["t"] - 1e9 > 2)
        self.assertLess(lost_t, 3.0 + ww.IMU_LOST_S + 0.05)
        # blind: no policy step once the sample is older than the Brain accepts
        self.assertTrue(all(ln["action"] is None for ln in self.log_lines if ln["t"] - 1e9 > 3.0 + ww.IMU_LOST_S))

    def test_runaway_action_stops_into_stand(self):
        w = World(shutdown=52, hw_err=0, volts_dv=74)
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 74, {"mode": "walk", "vx": 0.3}, end=10.0,
                                                              supply="bench", extra_args=())
        self.assertIn("STOP -> stand: policy action", text)
        self.assertEqual(last_log["mode"], "stand")
        self.assertEqual({v for a, i, v in w.log if a == ww.ADDR_TORQUE}, {1})
        # the over-limit action is never written: every goal stays within the walking limit of the policy start pose
        sent = [(i, v) for a, i, v in w.log if a == ww.ADDR_GOAL]
        self.assertTrue(sent)
        worst = max(abs(to_model(ww.NAMES[i - 1], v) - STAND[ww.NAMES[i - 1]]) for i, v in sent)
        self.assertLessEqual(worst, ww.HARD_MAX_ACTION + 0.01)

    def test_over_limit_stops_only_a_sustained_or_extreme_action(self):
        over, why = ww.over_limit(1.86, 0, 1.8)  # one slippery catch tick (real robot 2026-10-15)
        self.assertEqual((over, why), (1, None))
        over, why = ww.over_limit(1.2, over, 1.8)
        self.assertEqual((over, why), (0, None))
        for _ in range(ww.OVER_LIMIT_TICKS - 1):
            over, why = ww.over_limit(2.0, over, 1.8)
            self.assertIsNone(why)
        over, why = ww.over_limit(2.0, over, 1.8)
        self.assertIn("for 3 ticks", why)
        self.assertIn("> 3.0", ww.over_limit(3.2, 0, 1.8)[1])

    def test_action_limit_keeps_walking_limit_while_the_gait_settles(self):
        self.assertEqual(ww.action_limit(True, 0.0, 1.0, 1.8), 1.8)
        self.assertEqual(ww.action_limit(False, 0.5, 1.0, 1.8), 1.8)  # walk61 bringing the stride to rest
        self.assertEqual(ww.action_limit(False, ww.WALK_SETTLE_S + 0.01, 1.0, 1.8), 1.0)
        self.assertEqual(ww.action_limit(False, float("inf"), 1.0, 1.8), 1.0)  # never stepped

    def test_walking_stops_at_zero_command_and_keeps_torque(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=76)
        text, marker, wants, reads, last_log = self.run_wired(
            w, lambda t: 76 if t < 5 else 65, {"mode": "walk", "vx": 0.3}, end=14.0)
        self.assertIn("torque on at present positions", text)
        self.assertIn("battery guard armed", text)
        self.assertIn("BATTERY LOW", text)
        self.assertIn("BATTERY STOP: 6.5 V", text)
        self.assertEqual(marker["holding"], "policy at zero command, torque on")
        # the guard never writes torque off
        self.assertEqual({v for a, i, v in w.log if a == ww.ADDR_TORQUE}, {1})
        # after the stop every command is walk at zero velocity (walk61 at rest), whatever the control file says
        stop_t = next(t for t, want in wants if want.get("vx") == 0.0)
        after = [want for t, want in wants if t >= stop_t]
        self.assertGreater(len(after), 300)
        self.assertTrue(all(want == {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0} for want in after))
        before = [want.get("vx") for t, want in wants if t < stop_t and want.get("mode") == "walk"]
        self.assertTrue(all(0.0 < v <= 0.3 + 1e-9 for v in before))
        self.assertAlmostEqual(max(before), 0.3, places=3)
        # signalled 1.5 s after the stop, not before; never an open-loop ramp
        self.assertGreaterEqual(self.first("battery_signalled", True) - self.first("battery", "stop"), 1.5)
        self.assertLess(self.first("battery_signalled", True) - self.first("battery", "stop"), 1.6)
        stop_text = text[text.index("BATTERY STOP"):]
        self.assertNotIn("mode -> stand", stop_text)
        self.assertIn("Holding: policy at zero command, torque on", stop_text)
        self.assertEqual((last_log["battery"], last_log["battery_dv"], last_log["mode"]), ("stop", 65, "walk"))

    def test_standing_stop_returns_to_plain_stand(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=76)
        text, marker, wants, reads, _ = self.run_wired(
            w, lambda t: 76 if t < 5 else 65, {"mode": "stand", "head_pitch": 0.2}, end=12.0)
        self.assertEqual(marker["holding"], "STAND, torque on")
        self.assertEqual({v for a, i, v in w.log if a == ww.ADDR_TORQUE}, {1})
        final = {i: w.read(i, ww.ADDR_GOAL, 4) for i in IDS}
        self.assertEqual(final, {i: to_tick(n, STAND[n]) for i, n in zip(IDS, ww.NAMES)})  # override dropped

    def test_stop_during_an_override_ramp_signals_only_on_stand(self):
        # stand mode; a new head_pitch override starts a 1.5 s ramp at 6.5 s; the stop lands at about 7 s, inside it
        w = World(shutdown=52, hw_err=0x01, volts_dv=76)

        def override(d, cmd_path):
            def write():
                with open(cmd_path, "w") as fh:
                    json.dump({"mode": "stand", "head_pitch": 0.2}, fh)
                os.utime(cmd_path, (2e9, 2e9))
            return [(6.5, write)]
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 76 if t < 5 else 65, {"mode": "stand"},
                                                       end=13.0, hooks=override)
        stop_t, signal_t = self.first("battery", "stop"), self.first("battery_signalled", True)
        self.assertTrue(6.5 < stop_t < 8.0, stop_t)
        self.assertGreater(signal_t, 9.4)  # override ramp ends at 8.0, then 1.5 s back to STAND
        at_signal = next(ln for ln in self.log_lines if ln.get("battery_signalled"))
        stand_q = [STAND[n] for n in ww.NAMES]
        self.assertTrue(all(abs(q - sq) < 0.002 for q, sq in zip(at_signal["q"], stand_q)))
        self.assertEqual(marker["holding"], "STAND, torque on")

    def test_bench_drops_alert_packets_as_before(self):
        # bench supply, factory Shutdown, ID4 latched overload: its packets are dropped, so torque never comes on
        w = World(shutdown=53, volts_dv=68)
        w.regs[4][70] = 0x20
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 68, {"mode": "stand"}, end=3.0, supply="bench")
        self.assertNotIn(ww.ADDR_TORQUE, {a for a, i, v in w.log})
        self.assertNotIn("torque on", text)

    def test_hardware_fault_on_the_pack_freezes_the_policy(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=80)

        def overload(d, cmd_path):
            def fault():
                w.regs[4][70] = 0x21
            return [(5.0, fault)]
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 80, {"mode": "walk", "vx": 0.3}, end=9.0,
                                                       hooks=overload)
        self.assertIn("HARDWARE ERROR (address 70) on ['right_hip_roll']", text)
        self.assertEqual({v for a, i, v in w.log if a == ww.ADDR_TORQUE}, {1})
        before = [ln for ln in self.log_lines if 3.0 < ln["t"] - 1e9 < 5.0]
        after = [ln for ln in self.log_lines if ln["t"] - 1e9 > 6.2]
        self.assertTrue(all(ln["fresh"] and ln["action"] is not None for ln in before))
        self.assertTrue(after and all(not ln["fresh"] and ln["action"] is None for ln in after))

    def test_restart_on_a_drained_pack_is_refused(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=71)
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 71, {"mode": "walk", "vx": 0.3}, end=4.0,
                                                       supply="battery", old_marker=True)
        self.assertIn("records a battery stop", text)
        self.assertNotIn(ww.ADDR_TORQUE, {a for a, i, v in w.log})
        self.assertIsNotNone(marker)

    def test_a_charged_pack_clears_the_marker(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=81)
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 81, {"mode": "stand"}, end=3.0,
                                                       old_marker=True)
        self.assertIn("torque on at present positions", text)
        self.assertIsNone(marker)

    def test_torque_off_stays_off_after_a_stop(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=65)

        def walk_later(d, cmd_path):
            def write():
                with open(cmd_path, "w") as fh:
                    json.dump({"mode": "walk", "vx": 0.3}, fh)
                os.utime(cmd_path, (2e9, 2e9))
            return [(6.0, write)]
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 65, {"mode": "monitor"}, end=9.0, hooks=walk_later,
                                                       supply="battery")
        self.assertEqual(marker["holding"], "nothing, torque off")
        self.assertNotIn(ww.ADDR_TORQUE, {a for a, i, v in w.log})
        self.assertTrue(all(want == {"mode": "monitor"} for t, want in wants if t > 6.0))

    def test_bench_supply_reads_no_voltage(self):
        w = World(shutdown=53, volts_dv=68)
        text, marker, wants, reads, last_log = self.run_wired(w, lambda t: 64, {"mode": "stand"}, end=6.0,
                                                              supply="bench")
        self.assertNotIn(ww.ADDR_PRESENT_VOLTAGE, reads)
        self.assertIsNone(marker)
        self.assertEqual(last_log["battery"], None)

    def test_auto_on_the_bench_rail_never_stops(self):
        w = World(shutdown=53, volts_dv=68)
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 68 if t < 3 else 64, {"mode": "stand"}, end=8.0)
        self.assertIn(ww.ADDR_PRESENT_VOLTAGE, reads)
        self.assertNotIn("BATTERY", text)
        self.assertIn("battery guard not armed: the servos read 6.8 V", text)
        self.assertIsNone(marker)

    def test_torque_refused_on_a_real_hardware_error(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=80)
        w.regs[4][70] = 0x21  # overload latched as well as the voltage flag
        text, marker, wants, reads, _ = self.run_wired(w, lambda t: 80, {"mode": "stand"}, end=3.0)
        self.assertIn("torque refused", text)
        self.assertIn("right_hip_roll", text)
        self.assertNotIn(ww.ADDR_TORQUE, {a for a, i, v in w.log})


class LinkAlert(unittest.TestCase):
    """The real Link.sync_read parser: Alert passes, an instruction error does not."""

    def link_answering(self, world, alert_ok=True):
        link = ww.Link.__new__(ww.Link)
        link.status = collections.deque()
        link.alert_ok = alert_ok
        link.late = 0

        class Ser:
            def write(self, pkt):
                addr, size = struct.unpack_from("<HH", pkt, 8)
                for i in pkt[12:-2]:
                    err = world.err(i) | (0x07 if world.regs[i][64] == 9 else 0)
                    body = b"\xff\xff\xfd\x00" + bytes([i]) + struct.pack("<H", size + 4) + bytes([0x55, err]) + \
                        bytes(world.regs[i][addr:addr + size])
                    link.status.append(body + struct.pack("<H", 0))
        link.ser = Ser()
        return link

    def test_alert_packets_are_read(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=82)
        got = self.link_answering(w).sync_read(ww.ADDR_PRESENT_VOLTAGE, 2, IDS, timeout=0.01)
        self.assertEqual({i: int.from_bytes(b, "little") for i, b in got.items()}, {i: 82 for i in IDS})

    def test_bench_link_is_strict_unless_asked(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=82)
        link = self.link_answering(w, alert_ok=False)
        self.assertEqual(link.sync_read(ww.ADDR_PRESENT_VOLTAGE, 2, IDS, timeout=0.01), {})
        self.assertEqual(len(link.sync_read(ww.ADDR_PRESENT_VOLTAGE, 2, IDS, timeout=0.01, allow_alert=True)), 14)

    def test_error_packets_are_dropped(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=82)
        w.regs[3][64] = 9  # marker for the fake: answer this ID with an access error
        got = self.link_answering(w).sync_read(ww.ADDR_PRESENT_VOLTAGE, 2, IDS, timeout=0.01)
        self.assertEqual(sorted(got), [i for i in IDS if i != 3])


if __name__ == "__main__":
    unittest.main()


def test_imu_frame_ok_rejects_corrupt_frames():
    import math
    from walk_wired import imu_frame_ok
    good = (7, 1.0, 0.0, 0.0, 0.0, 0.1, -0.2, 0.0, 0.0, 0.0, 9.8, 0.008, 1)
    assert imu_frame_ok(good)
    assert not imu_frame_ok(good[:1] + (2.2e76, 0.0, 0.0, 0.0) + good[5:])   # the 10/15 garbage frame
    assert not imu_frame_ok(good[:1] + (math.nan, 0.0, 0.0, 0.0) + good[5:])
    assert not imu_frame_ok(good[:5] + (99.0, 0.0, 0.0) + good[8:])           # gyro beyond the BNO range
    assert not imu_frame_ok(good[:11] + (-1.0, 1))                            # negative age
