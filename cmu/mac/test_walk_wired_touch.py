"""walk_wired.py: Present Current in the per-tick servo read and the head touch pads in the 50 Hz log. The fake servo
link and simulated clock of test_walk_wired_battery (real Brain and policies), fake touch buses; no hardware.

  ~/.venvs/microduck-walk/bin/python -m unittest test_walk_wired_touch -v
"""
import contextlib
import io
import os
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import test_walk_wired_battery as twb  # noqa: E402
import walk_wired as ww  # noqa: E402
from test_battery import World  # noqa: E402
from test_walk_wired_battery import make_link as make_link_for  # noqa: E402

IDS = ww.IDS
WALK0 = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
NO_LIMITS = ("--max-action", "1e9", "--max-walk-action", "1e9", "--hard-max-action", "1e9")  # no physics in the fake


def state_world(drop=()):
    """Every servo with its own signed current, velocity and position; IDs in `drop` never answer."""
    w = World(shutdown=52, hw_err=0, volts_dv=74)
    for i in drop:
        del w.regs[i]
    cur = {i: (-1) ** i * (10 * i + 3) for i in w.regs}  # ID1 -13 mA, ID2 +23 mA, ...
    vel = {i: 5 - 2 * i for i in w.regs}
    pos = {i: 1500 + 37 * i for i in w.regs}
    for i in w.regs:
        struct.pack_into("<h", w.regs[i], 126, cur[i])
        struct.pack_into("<i", w.regs[i], 128, vel[i])
        struct.pack_into("<i", w.regs[i], 132, pos[i])
    return w, cur, vel, pos


def recording_link(calls):
    def factory(world, clock, reads):
        class Recording(make_link_for(world, clock, reads)):
            def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
                calls.append((addr, size, tuple(ids)))
                return super().sync_read(addr, size, ids, timeout, allow_alert)
        return Recording
    return factory


class FakePads:
    """The XIAO's I2C block: every read is a new scan (sequence number + 1), rest counts unless pressed. mode "ok" serves
    blocks, "fail" raises the Remote I/O error of a slave that does not answer, "hang" never returns."""
    REST = (22100, 23400, 24800, 25500, 26300)
    PRESS = 3000

    def __init__(self):
        self.mode, self.pressed, self.done = "ok", set(), False
        self.calls, self.seq, self.served, self.failed = 0, 0, [], 0
        self.cond = threading.Condition()
        self.park = threading.Event()  # never set

    def read(self):
        with self.cond:
            self.calls += 1
            self.cond.notify_all()
        if self.done or self.mode == "hang":
            self.park.wait()
        with self.cond:
            if self.mode == "fail":
                self.failed += 1
                self.cond.notify_all()
                raise OSError(121, "Remote I/O error")
            self.seq += 1
            counts = tuple(c + (self.PRESS if p in self.pressed else 0) for p, c in zip(ww.TOUCH_PADS, self.REST))
            self.served.append(counts)
            self.cond.notify_all()
            return struct.pack("<6I", self.seq, *counts)


def hook(pads, change, until=lambda p, n, f: True):
    """A FakeClock hook: once a first block is served, apply `change` to the pads and hold the simulated loop (wall
    clock) until `until(pads, served before, failed before)` and the reader thread has started its next read, so that
    whatever it read before is stored."""
    def go():
        with pads.cond:
            ok = pads.cond.wait_for(lambda: pads.served, 5.0)
            n, f = len(pads.served), pads.failed
            for k, v in change.items():
                setattr(pads, k, v)
            ok = ok and pads.cond.wait_for(lambda: until(pads, n, f), 5.0)
            c = pads.calls
            ok = ok and pads.cond.wait_for(lambda: pads.calls > c, 5.0)
        if not ok:
            raise AssertionError("the touch reader thread did not get there in time")
    return go


def press(pads, pad):
    return hook(pads, {"pressed": {pad}}, lambda p, n, f: any(c != p.REST for c in p.served[n:]))


def release(pads):
    return hook(pads, {"pressed": set()}, lambda p, n, f: any(c == p.REST for c in p.served[n:]))


class WiredRun(unittest.TestCase):
    run_wired = twb.WiredBattery.run_wired

    def run_loop(self, w, cmd, end, touch_args=("--touch-bus", "none"), pads=None, hooks=(), link=None, sysfs=None):
        """run_wired with the touch bus patched to `pads` (opened paths in self.opened); returns stdout."""
        self.opened, patches = [], []
        if pads:
            patches.append(mock.patch.object(ww, "open_touch_bus",
                                             lambda path: (self.opened.append(path), pads.read)[1]))
        if link:
            patches.append(mock.patch.object(twb, "make_link", link))
        if sysfs:
            patches.append(mock.patch.object(ww, "I2C_SYSFS", sysfs))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            try:
                text, *_ = self.run_wired(w, lambda t: 74, cmd, end=end, supply="bench",
                                          hooks=lambda d, cmd_path: list(hooks),
                                          extra_args=NO_LIMITS + tuple(touch_args))
            finally:
                if pads:
                    pads.done = True  # park the daemon reader thread
        return text

    def rows(self, t0=float("-inf"), t1=float("inf")):
        return [ln for ln in self.log_lines if t0 <= ln["t"] - 1e9 < t1]


class CurrentInTheStateRead(WiredRun):
    def test_current_is_read_with_velocity_and_position_in_one_block(self):
        w, cur, vel, pos = state_world()
        calls = []
        self.run_loop(w, {"mode": "monitor"}, end=1.0, link=recording_link(calls))
        state = [c for c in calls if c[0] == ww.ADDR_CUR]
        self.assertGreater(len(state), 40)
        self.assertEqual({(size, ids) for addr, size, ids in state}, {(10, tuple(IDS + [ww.JAW_ID]))})
        self.assertNotIn(ww.ADDR_VEL, {addr for addr, size, ids in calls})  # no second read for the old block
        self.assertTrue(self.log_lines)
        for ln in self.log_lines:
            self.assertEqual(ln["cur_ma"], [cur[i] for i in IDS])  # signed mA, ID order like pos_tick
            self.assertEqual(ln["vel_tick"], [vel[i] for i in IDS])
            self.assertEqual(ln["pos_tick"], [pos[i] for i in IDS])
            self.assertEqual((ln["jaw_cur_ma"], ln["jaw_tick"]), (cur[ww.JAW_ID], pos[ww.JAW_ID]))

    def spy_brain(self):
        """Patch Brain.tick to record the (position, velocity) lists the loop hands it."""
        seen, tick = [], ww.Brain.tick

        def spy(self_, want, now, p, v, *a):
            seen.append((list(p), list(v)))
            return tick(self_, want, now, p, v, *a)
        return seen, mock.patch.object(ww.Brain, "tick", spy)

    def test_brain_gets_the_position_and_velocity_the_old_8_byte_read_gave(self):
        w, cur, vel, pos = state_world()
        seen, patch = self.spy_brain()
        with patch:
            self.run_loop(w, {"mode": "monitor"}, end=0.5)
        old = [struct.unpack("<ii", bytes(w.regs[i][128:136])) for i in IDS]  # (velocity, position) at address 128
        self.assertTrue(seen)
        self.assertTrue(all(s == ([p for v, p in old], [v for v, p in old]) for s in seen))

    def test_positions_follow_the_goals_while_walking(self):
        w, cur, vel, pos = state_world()
        seen, patch = self.spy_brain()
        with patch:
            self.run_loop(w, WALK0, end=2.0)  # torque on: every goal becomes the next present position
        self.assertTrue(any(ln["torque"] for ln in self.log_lines))
        moved = 0
        for prev, ln in zip(self.log_lines, self.log_lines[1:]):
            if prev["torque"] and prev["goal_tick"]:
                self.assertEqual(ln["pos_tick"], [prev["goal_tick"][str(i)] for i in IDS])
                moved += ln["pos_tick"] != prev["pos_tick"]
            self.assertEqual(ln["vel_tick"], [vel[i] for i in IDS])
            self.assertEqual(ln["cur_ma"], [cur[i] for i in IDS])
        self.assertGreater(moved, 10, "the policy should have moved the joints between reads")
        self.assertEqual([p for p, v in seen][:len(self.log_lines)], [ln["pos_tick"] for ln in self.log_lines])

    def test_a_servo_that_never_answers_has_no_current(self):
        w, cur, vel, pos = state_world(drop=(7,))
        self.run_loop(w, {"mode": "monitor"}, end=0.5)
        k = IDS.index(7)
        for ln in self.log_lines:
            self.assertIsNone(ln["cur_ma"][k])
            self.assertIn(7, ln["missing"])
            self.assertEqual([c for j, c in enumerate(ln["cur_ma"]) if j != k], [cur[i] for i in IDS if i != 7])


class TouchInTheLoop(WiredRun):
    def test_absent_bus_is_a_warning_and_null_fields(self):
        with tempfile.TemporaryDirectory() as empty:
            text = self.run_loop(World(shutdown=52, hw_err=0, volts_dv=74), WALK0, end=2.0,
                                 touch_args=("--touch-bus", "auto"), sysfs=empty)
        self.assertEqual(text.count("touch: no TWI0 I2C bus"), 1)
        self.assertAlmostEqual(len(self.log_lines), 100, delta=3)  # 2 s at 50 Hz
        for ln in self.log_lines:
            self.assertEqual((ln["touch"], ln["touch_seq"], ln["touch_age"], ln["touch_bad"]), (None,) * 4)

    def test_a_bus_that_cannot_be_opened_is_a_warning(self):
        text = self.run_loop(World(shutdown=52, hw_err=0, volts_dv=74), WALK0, end=1.0,
                             touch_args=("--touch-bus", "/nonexistent/i2c-9"))
        self.assertIn("touch: cannot open /nonexistent/i2c-9", text)
        self.assertAlmostEqual(len(self.log_lines), 50, delta=3)
        self.assertTrue(all(ln["touch"] is None for ln in self.log_lines))

    def test_pads_are_logged_with_their_sequence_and_age(self):
        pads = FakePads()
        text = self.run_loop(World(shutdown=52, hw_err=0, volts_dv=74), WALK0, end=3.0,
                             touch_args=("--touch-bus", "0"), pads=pads,
                             hooks=[(1.0, press(pads, "L1")), (2.0, release(pads))])
        self.assertEqual(self.opened, ["/dev/i2c-0"])
        self.assertIn("touch: pads L1/R1/L2/R2/M on /dev/i2c-0", text)
        rest, l1 = list(FakePads.REST), [FakePads.REST[0] + FakePads.PRESS] + list(FakePads.REST[1:])
        self.assertTrue(all(ln["touch"] in (None, rest) for ln in self.rows(0.0, 1.0)))
        self.assertTrue(self.rows(1.0, 2.0) and all(ln["touch"] == l1 for ln in self.rows(1.0, 2.0)))
        self.assertTrue(self.rows(2.0) and all(ln["touch"] == rest for ln in self.rows(2.0)))
        logged = [ln for ln in self.log_lines if ln["touch"] is not None]
        seqs = [ln["touch_seq"] for ln in logged]
        self.assertEqual(seqs, sorted(seqs))
        self.assertGreater(seqs[-1], seqs[0])
        self.assertTrue(all(ln["touch_age"] >= 0 and ln["touch_bad"] == 0 for ln in logged))

    def test_read_errors_keep_the_last_block_and_are_reported_once(self):
        pads = FakePads()
        text = self.run_loop(World(shutdown=52, hw_err=0, volts_dv=74), WALK0, end=3.0,
                             touch_args=("--touch-bus", "0"), pads=pads,
                             hooks=[(1.0, hook(pads, {"mode": "fail"}, lambda p, n, f: p.failed > f)),
                                    (2.0, hook(pads, {"mode": "ok"}, lambda p, n, f: len(p.served) > n))])
        self.assertEqual(text.count("touch: read failed (OSError: [Errno 121] Remote I/O error)"), 1)
        self.assertEqual(text.count("touch: reads back"), 1)
        failing = self.rows(1.0, 2.0)
        self.assertTrue(failing)
        self.assertEqual({tuple(ln["touch"]) for ln in failing}, {FakePads.REST})
        self.assertEqual(len({ln["touch_seq"] for ln in failing}), 1)
        ages = [ln["touch_age"] for ln in failing]
        self.assertEqual(ages, sorted(ages))
        self.assertGreater(ages[-1], 0.9)  # the age grows while nothing new arrives
        bad = [ln["touch_bad"] for ln in failing]  # the simulated second passes in a few wall-clock reads
        self.assertTrue(bad[0] >= 1 and bad == sorted(bad))
        self.assertGreater(self.rows(2.0)[-1]["touch_seq"], failing[-1]["touch_seq"])

    def test_the_touch_reader_never_changes_the_control_loop(self):
        """Same world and command without a bus, with pads, and with a bus that hangs at 0.5 s: the same ticks, the
        same loop and read times, the same goals written."""
        runs = {}
        for name in ("none", "pads", "hung"):
            w = World(shutdown=52, hw_err=0, volts_dv=74)
            pads = FakePads() if name != "none" else None
            hooks = [(0.5, hook(pads, {"mode": "hang"}))] if name == "hung" else []
            self.run_loop(w, WALK0, end=2.0, touch_args=("--touch-bus", "0") if pads else ("--touch-bus", "none"),
                          pads=pads, hooks=hooks)
            runs[name] = (self.log_lines, [e for e in w.log if e[0] == ww.ADDR_GOAL])
        base_rows, base_goals = runs["none"]
        self.assertAlmostEqual(len(base_rows), 100, delta=3)
        for name in ("pads", "hung"):
            rows, goals = runs[name]
            for key in ("t", "loop_ms", "read_ms", "action", "goal_tick"):
                self.assertEqual([ln[key] for ln in rows], [ln[key] for ln in base_rows], (name, key))
            self.assertEqual(goals, base_goals, name)
        hung = [ln for ln in runs["hung"][0] if ln["t"] - 1e9 >= 0.5]
        self.assertEqual(len({ln["touch_seq"] for ln in hung}), 1)  # the last block before the bus hung
        self.assertGreater(hung[-1]["touch_age"], 1.3)


class TouchPadsUnit(unittest.TestCase):
    class Clock:
        def __init__(self):
            self.t = 10.0

        def perf_counter(self):
            return self.t

    def scripted(self, *replies):
        """A reader that returns (or raises) the replies in order."""
        it = iter(replies)

        def read():
            r = next(it)
            if isinstance(r, Exception):
                raise r
            return r
        return read

    def test_blocks_age_and_rejects(self):
        clock, out = self.Clock(), io.StringIO()
        block = lambda seq, *c: struct.pack("<6I", seq, *c)  # noqa: E731
        sat = ww.TOUCH_SATURATED
        reader = self.scripted(block(5, 1, 2, 3, 4, 5), block(5, 1, 2, 3, 4, 5), block(6, sat, 2, 3, 4, 5),
                               block(7, sat + 1, 2, 3, 4, 5), b"\x00" * 23, OSError(121, "Remote I/O error"),
                               block(8, 9, 9, 9, 9, 9))
        with mock.patch.object(ww, "time", clock), contextlib.redirect_stdout(out):
            pads = ww.TouchPads(reader, start=False)
            self.assertEqual(pads.latest(clock.t), (None, None, None))
            self.assertTrue(pads.poll())
            clock.t += 0.03
            counts, seq, age = pads.latest(clock.t)
            self.assertEqual((counts, seq), ([1, 2, 3, 4, 5], 5))
            self.assertAlmostEqual(age, 0.03)
            self.assertTrue(pads.poll())  # the same scan again: still the old arrival, the age keeps growing
            clock.t += 0.02
            self.assertAlmostEqual(pads.latest(clock.t)[2], 0.05)
            self.assertTrue(pads.poll())  # a saturated pad (lead on metal) is a real reading and is logged
            self.assertEqual(pads.latest(clock.t), ([sat, 2, 3, 4, 5], 6, 0.0))
            for _ in range(3):  # above saturation (misaligned), a short read, a bus error: rejected, block kept
                self.assertFalse(pads.poll())
            self.assertEqual((pads.latest(clock.t)[1], pads.bad), (6, 3))
            self.assertTrue(pads.poll())
            self.assertEqual(pads.latest(clock.t)[:2], ([9, 9, 9, 9, 9], 8))
        text = out.getvalue()
        self.assertEqual(text.count("touch: read failed"), 1)
        self.assertIn("misaligned", text)
        self.assertEqual(text.count("touch: reads back"), 1)

    def test_bus_path(self):
        with tempfile.TemporaryDirectory() as d:
            tree, sysfs = os.path.join(d, "soc@3000000"), os.path.join(d, "devices")
            for node in ("twi@2510000", "twi@2517000"):
                os.makedirs(os.path.join(tree, node))
            for bus, node in (("i2c-7", "twi@2517000"), ("i2c-0", "twi@2510000")):
                os.makedirs(os.path.join(sysfs, bus))
                os.symlink(os.path.join(tree, node), os.path.join(sysfs, bus, "of_node"))
            with mock.patch.object(ww, "I2C_SYSFS", sysfs):
                self.assertEqual(ww.touch_bus_path("auto"), "/dev/i2c-0")
                os.remove(os.path.join(sysfs, "i2c-0", "of_node"))
                self.assertIsNone(ww.touch_bus_path("auto"))  # only the IMU's TWI7
        self.assertIsNone(ww.touch_bus_path("none"))
        self.assertEqual(ww.touch_bus_path("3"), "/dev/i2c-3")
        self.assertEqual(ww.touch_bus_path("/dev/i2c-5"), "/dev/i2c-5")


class TouchPadsThread(unittest.TestCase):
    """Wall-clock checks of the reader thread itself."""

    def test_a_hung_bus_never_blocks_the_reader_of_latest(self):
        started, park = threading.Event(), threading.Event()

        def hang():
            started.set()
            park.wait()
        pads = ww.TouchPads(hang)
        self.assertTrue(started.wait(1.0))
        t0 = time.perf_counter()
        for _ in range(1000):
            pads.latest(time.perf_counter())
        self.assertLess(time.perf_counter() - t0, 0.05)  # 1000 calls while the bus read is stuck

    def test_reads_at_the_scan_rate_of_the_xiao(self):
        """50 reads a second although each read takes 5 ms: the pause is shortened by the read time."""
        n, park = [0], threading.Event()

        def slow_read():
            if n[0] >= 100:
                park.wait()
            n[0] += 1
            time.sleep(0.005)
            return struct.pack("<6I", n[0], 1, 2, 3, 4, 5)
        with contextlib.redirect_stdout(io.StringIO()):
            ww.TouchPads(slow_read)
            time.sleep(1.0)
        self.assertGreaterEqual(n[0], 45)
        self.assertLessEqual(n[0], 52)


if __name__ == "__main__":
    unittest.main()
