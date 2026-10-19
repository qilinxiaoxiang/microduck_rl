"""Unit tests for the 2S battery software: battery.py, set_shutdown.py, check_voltage_limit.py --rail battery, the
Alert-bit handling in seam_fix.py and stand.py, and stand.py's battery stop. Mocked buses only, no hardware.

Run from this folder:  python3 -m unittest test_battery -v
pyserial and dynamixel_sdk are replaced by stand-ins when they are not installed.
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    import serial  # noqa: F401
except ImportError:
    _serial = types.ModuleType("serial")
    _serial.Serial = None
    _serial.SerialException = type("SerialException", (OSError,), {})
    sys.modules["serial"] = _serial

COMM_SUCCESS, COMM_RX_TIMEOUT = 0, -3001


class World:
    """Fifteen XL330 control tables. Status error byte: 0x80 (Alert) whenever Hardware Error Status is non-zero."""

    def __init__(self, ids=range(1, 16), shutdown=53, volts_dv=74, hw_err=0, torque=0):
        self.regs = {}
        for i in ids:
            r = bytearray(256)
            struct.pack_into("<H", r, 32, 70)
            struct.pack_into("<H", r, 34, 35)
            r[11], r[63], r[64], r[70] = (4 if i == 5 else 3), shutdown, torque, hw_err
            struct.pack_into("<i", r, 132, 2048)
            struct.pack_into("<H", r, 144, volts_dv)
            self.regs[i] = r
        self.ignore_writes = set()
        self.fail_sync_reads = set()  # addresses whose Sync Read fails (a comm glitch)
        self.log = []  # (addr, id, value) for every accepted write

    def err(self, i):
        return 0x80 if self.regs[i][70] else 0

    def read(self, i, addr, size):
        return int.from_bytes(self.regs[i][addr:addr + size], "little")

    def write(self, i, addr, size, value):
        """Returns the status error byte; EEPROM (below 64) is refused with torque on, like the servo."""
        r = self.regs[i]
        if addr < 64 and r[64]:
            return 0x07 | self.err(i)  # access error
        if i not in self.ignore_writes:
            r[addr:addr + size] = (int(value) & ((1 << (8 * size)) - 1)).to_bytes(size, "little")
            self.log.append((addr, i, int(value)))
        return self.err(i)

    def set_all(self, addr, size, value):
        for i in self.regs:
            self.regs[i][addr:addr + size] = int(value).to_bytes(size, "little")


import battery as bt  # noqa: E402
import check_voltage_limit as cvl  # noqa: E402
import set_shutdown as ss  # noqa: E402


class FakeSerial:
    """Byte-level DYNAMIXEL Protocol 2.0 servos behind a serial port: answers READ (0x02) and WRITE (0x03)."""

    def __init__(self, world, echo=False):
        self.world, self.echo, self.out = world, echo, b""

    def reset_input_buffer(self):
        self.out = b""

    def read(self, n):
        chunk, self.out = self.out[:n], self.out[n:]
        return chunk

    def close(self):
        pass

    def write(self, pkt):
        if self.echo:
            self.out += pkt  # a half-duplex line that hears its own instruction
        i, inst, params = pkt[4], pkt[7], pkt[8:-2]
        assert ss.crc16(pkt[:-2]) == struct.unpack_from("<H", pkt, len(pkt) - 2)[0]
        if i not in self.world.regs:
            return
        if inst == 0x02:
            addr, size = struct.unpack_from("<HH", params)
            reply, err = bytes(self.world.regs[i][addr:addr + size]), self.world.err(i)
        elif inst == 0x03:
            addr = struct.unpack_from("<H", params)[0]
            data = params[2:]
            err, reply = self.world.write(i, addr, len(data), int.from_bytes(data, "little")), b""
        else:
            raise AssertionError(f"unexpected instruction 0x{inst:02x}")
        body = b"\xff\xff\xfd\x00" + bytes([i]) + struct.pack("<H", len(reply) + 4) + bytes([0x55, err]) + reply
        self.out += body + struct.pack("<H", ss.crc16(body))


def raw_bus(world, echo=False):
    bus = ss.Bus.__new__(ss.Bus)
    bus.ser = FakeSerial(world, echo)
    return bus


# ---------------------------------------------------------------------------------------------- battery.py
class Thresholds(unittest.TestCase):
    def test_values_from_the_plan(self):
        self.assertEqual((bt.SHUTDOWN_FACTORY, bt.SHUTDOWN_BATTERY), (53, 52))
        self.assertEqual(bt.SHUTDOWN_FACTORY & ~bt.INPUT_VOLTAGE_BIT, bt.SHUTDOWN_BATTERY)  # only bit 0 differs
        self.assertEqual((bt.WARN_BELOW_DV, bt.STOP_AT_DV), (70, 66))
        self.assertEqual(bt.BATTERY_WINDOW_DV, (64, 84))
        self.assertEqual(bt.ADDR_PRESENT_VOLTAGE, 144)
        self.assertGreater(bt.ARM_ABOVE_DV, 70)  # the bench 7 V rail (6.80 V, window up to 7.0 V) never arms

    def test_status_ok_lets_only_the_alert_bit_through(self):
        self.assertTrue(bt.status_ok(0x00))
        self.assertTrue(bt.status_ok(0x80))
        for err in (0x01, 0x07, 0x87, 0x83):
            self.assertFalse(bt.status_ok(err), hex(err))

    def test_voltage_alert_only(self):
        self.assertTrue(bt.voltage_alert_only(0x01, 52))
        self.assertFalse(bt.voltage_alert_only(0x01, 53))   # torque was cut
        self.assertFalse(bt.voltage_alert_only(0x21, 52))   # overload as well
        self.assertFalse(bt.voltage_alert_only(0x20, 52))
        self.assertFalse(bt.voltage_alert_only(0x00, 52))


def feed(guard, values, t0=0.0):
    """Feed one sample per second; each item is a volt value for 15 servos, None (no answer) or a dict."""
    events = []
    for k, v in enumerate(values):
        readings = {} if v is None else (v if isinstance(v, dict) else {i: round(v * 10) for i in range(1, 16)})
        events.append(guard.update(t0 + k, readings))
    return events


class Guard(unittest.TestCase):
    def test_warn_needs_two_samples_under_7v(self):
        g = bt.BatteryGuard("battery")
        self.assertEqual(feed(g, [8.2, 7.0, 6.9]), [None, None, None])
        self.assertEqual(g.state, "ok")
        self.assertEqual(g.update(3.0, {i: 69 for i in range(1, 16)}), "warn")
        self.assertEqual(g.state, "warn")

    def test_stop_needs_three_samples_at_or_under_6_6v_and_latches(self):
        g = bt.BatteryGuard("battery")
        events = feed(g, [6.7, 6.7, 6.6, 6.6, 6.6, 7.4, 8.0])
        self.assertEqual(events, [None, "warn", None, None, "stop", None, None])
        self.assertEqual((g.state, g.stop_t), ("stop", 4.0))

    def test_a_good_sample_restarts_the_count(self):
        g = bt.BatteryGuard("battery")
        events = feed(g, [6.6, 6.6, 7.1, 6.6, 6.6, 6.6])
        self.assertEqual(events.count("stop"), 1)
        self.assertEqual(events.index("stop"), 5)

    def test_missed_samples_change_nothing(self):
        g = bt.BatteryGuard("battery")
        events = feed(g, [6.5, None, 6.5, {1: 60, 2: 60}, 6.5])  # a miss and a two-servo answer are skipped
        self.assertEqual(events, [None, None, "warn", None, "stop"])
        self.assertEqual((g.misses, g.samples), (2, 3))

    def test_median_ignores_one_bad_servo(self):
        g = bt.BatteryGuard("battery")
        readings = {i: 80 for i in range(1, 15)}
        readings[15] = 0
        for k in range(5):
            self.assertIsNone(g.update(k, readings))
        self.assertEqual((g.dv, g.state), (80, "ok"))

    def test_auto_arms_only_on_battery_voltage(self):
        g = bt.BatteryGuard("auto")
        self.assertEqual(feed(g, [6.8, 6.5, 6.5, 6.5, 6.5]), ["unarmed"] + [None] * 4)  # the bench rail: never armed
        self.assertEqual(g.state, "ok")
        self.assertEqual(g.update(10.0, {i: 81 for i in range(1, 16)}), "armed")
        self.assertEqual(feed(g, [6.6, 6.6, 6.6], t0=11.0), [None, "warn", "stop"])

    def test_bench_has_no_guard(self):
        with self.assertRaises(ValueError):
            bt.BatteryGuard("bench")

    def test_once_a_second(self):
        g = bt.BatteryGuard("battery")
        self.assertTrue(g.due(0.0))
        g.update(5.0, {})
        self.assertFalse(g.due(5.99))
        self.assertTrue(g.due(6.0))

    def test_signal_writes_the_marker_once(self):
        g = bt.BatteryGuard("battery")
        feed(g, [6.5, 6.5, 6.5])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "stop.json")
            msg = g.signal(path, "stand pose, torque on")
            self.assertIn("BATTERY STOP", msg)
            self.assertIn("sudo shutdown -h now", msg)
            self.assertIsNone(g.signal(path, "again"))
            with open(path) as fh:
                marker = json.load(fh)
            self.assertEqual((marker["volts"], marker["stop_at_v"], marker["holding"]), (6.5, 6.6, "stand pose, torque on"))


class GuardSafety(unittest.TestCase):
    def test_signal_never_raises_when_the_marker_cannot_be_written(self):
        g = bt.BatteryGuard("battery")
        feed(g, [6.5, 6.5, 6.5])
        msg = g.signal("/nonexistent-dir/stop.json", "stand pose")
        self.assertIn("Marker NOT written", msg)
        self.assertTrue(g.signalled)

    def test_median(self):
        self.assertEqual(bt.median_dv({1: 80, 2: 70, 3: 75}), 75)
        self.assertEqual(bt.median_dv({1: 80, 2: 70, 3: 75, 4: 60}), 70)
        self.assertIsNone(bt.median_dv({1: 80, 2: 80}))

    def test_torque_block(self):
        with tempfile.TemporaryDirectory() as d:
            marker = os.path.join(d, "stop.json")
            g = bt.BatteryGuard("battery")
            self.assertIn("no Present Input Voltage", g.torque_block(marker))
            feed(g, [7.4])
            self.assertIsNone(g.torque_block(marker))
            with open(marker, "w") as fh:
                fh.write("{}")
            feed(g, [7.0, 7.5])  # a drained pack recovers at rest
            self.assertIn("records a battery stop", g.torque_block(marker))
            self.assertTrue(os.path.exists(marker))
            feed(g, [7.6])  # a charged or swapped pack clears the marker
            self.assertIsNone(g.torque_block(marker))
            self.assertFalse(os.path.exists(marker))
            empty = bt.BatteryGuard("battery")
            feed(empty, [6.6])
            self.assertIn("at or under the 6.6 V battery stop", empty.torque_block(marker))
            usb = bt.BatteryGuard("auto")  # unarmed 5 V supply: only a marker blocks
            feed(usb, [5.0])
            self.assertIsNone(usb.torque_block(marker))


class StopCommand(unittest.TestCase):
    def test_torque_off_stays_in_monitor(self):
        for mode in ("monitor", "stand", "walk"):
            self.assertEqual(bt.stop_command(False, mode), {"mode": "monitor"})

    def test_walking_stops_into_stand61_and_stays_there(self):
        self.assertEqual(bt.stop_command(True, "walk"), {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        self.assertEqual(bt.stop_command(True, "stand"), {"mode": "stand"})  # also drops pose overrides
        self.assertEqual(bt.stop_command(True, "monitor"), {"mode": "stand"})

    def test_settled(self):
        self.assertTrue(bt.stop_settled(False, "monitor", False, 0.0))
        self.assertFalse(bt.stop_settled(True, "walk", False, 1.49))
        self.assertTrue(bt.stop_settled(True, "walk", False, 1.5))
        self.assertFalse(bt.stop_settled(True, "stand", False, 9.0))  # still ramping or on an override
        self.assertTrue(bt.stop_settled(True, "stand", True, 0.0))
        self.assertFalse(bt.stop_settled(True, "monitor", True, 9.0))


# ---------------------------------------------------------------------------------------------- set_shutdown.py
class FakeBus:
    def __init__(self, world, missing=()):
        self.world, self.missing, self.writes = world, set(missing), []

    def read(self, i, addr, size):
        if i in self.missing or i not in self.world.regs:
            return None
        return self.world.read(i, addr, size)

    def write(self, i, addr, size, value):
        self.writes.append((i, addr, value))
        return None if i in self.missing else self.world.write(i, addr, size, value)


def run_quiet(bus, target, **kw):
    lines = []
    code = ss.run(bus, target, out=lines.append, **kw)
    return code, lines


def shutdown_values(world):
    return {i: r[63] for i, r in world.regs.items()}


class SetShutdown(unittest.TestCase):
    def test_writes_52_and_reads_every_servo_back(self):
        w = World(shutdown=53)
        bus = FakeBus(w)
        code, lines = run_quiet(bus, 52)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(i for i, a, v in bus.writes), list(range(1, 16)))
        self.assertTrue(all(a == 63 and v == 52 for i, a, v in bus.writes))
        self.assertEqual(set(shutdown_values(w).values()), {52})
        rows = [ln for ln in lines if ln[:3].strip().isdigit()]
        self.assertEqual(len(rows), 15)
        self.assertTrue(all(" 53 0x35   52 0x34  written" in ln for ln in rows), rows[0])
        self.assertTrue(lines[-1].startswith("PASS: 15/15"))

    def test_restore_writes_53(self):
        w = World(shutdown=52)
        code, lines = run_quiet(FakeBus(w), 53)
        self.assertEqual(code, 0)
        self.assertEqual(set(shutdown_values(w).values()), {53})
        self.assertIn("check_voltage_limit.py.", lines[-1])

    def test_already_set_writes_nothing(self):
        bus = FakeBus(World(shutdown=52))
        code, lines = run_quiet(bus, 52)
        self.assertEqual((code, bus.writes), (0, []))
        self.assertTrue(all("unchanged" in ln for ln in lines if ln[:3].strip().isdigit()))

    def test_refuses_with_torque_on_and_writes_nothing(self):
        w = World(shutdown=53)
        w.regs[3][64] = 1
        bus = FakeBus(w)
        code, lines = run_quiet(bus, 52)
        self.assertEqual((code, bus.writes), (1, []))
        self.assertIn("REFUSED", lines[-1])
        self.assertIn("ID 3", lines[-1])
        self.assertEqual(set(shutdown_values(w).values()), {53})
        self.assertTrue(any(ln.startswith("  3") and " ON " in ln for ln in lines))

    def test_refuses_when_a_servo_is_missing(self):
        bus = FakeBus(World(shutdown=53), missing=[7])
        code, lines = run_quiet(bus, 52)
        self.assertEqual((code, bus.writes), (1, []))
        self.assertIn("ID 7 did not answer", lines[-1])

    def test_readback_mismatch_fails(self):
        w = World(shutdown=53)
        w.ignore_writes.add(9)
        code, lines = run_quiet(FakeBus(w), 52)
        self.assertEqual(code, 1)
        self.assertTrue(any(ln.startswith("  9") and "MISMATCH" in ln for ln in lines))
        self.assertIn("ID 9", lines[-1])

    def test_refused_write_is_reported(self):
        w = World(shutdown=53)
        bus = FakeBus(w)
        real = w.write
        w.write = lambda i, a, s, v: 0x07 if i == 4 else real(i, a, s, v)
        code, lines = run_quiet(bus, 52)
        self.assertEqual(code, 1)
        self.assertTrue(any(ln.startswith("  4") and "write refused, error 0x07" in ln for ln in lines))

    def test_check_only_never_writes(self):
        for shutdown, expect in ((53, 1), (52, 0)):
            w = World(shutdown=shutdown, torque=1)  # torque on does not matter for a read
            bus = FakeBus(w)
            code, lines = run_quiet(bus, 52, check_only=True)
            self.assertEqual((code, bus.writes), (expect, []))

    def test_raw_protocol_with_alert_and_echo(self):
        # The real Bus class over byte-level fake servos: every status packet carries Alert (battery above 7.0 V
        # with Shutdown 52 already on ID1-14) and the line echoes each instruction.
        w = World(shutdown=52, hw_err=0x01, volts_dv=84)
        w.regs[15][63] = 53
        bus = raw_bus(w, echo=True)
        code, lines = run_quiet(bus, 52)
        self.assertEqual(code, 0, lines)
        self.assertEqual(w.log, [(63, 15, 52)])
        self.assertEqual(bus.read(1, 144, 2), 84)

    def test_raw_protocol_torque_on_eeprom_write_is_an_access_error(self):
        w = World(torque=1)
        self.assertEqual(raw_bus(w).write(2, 63, 1, 52), 0x07)
        self.assertEqual(w.regs[2][63], 53)

    def test_find_status_rejects_bad_crc_and_instruction_packets(self):
        good = FakeSerial(World())
        good.write(ss.packet(1, 0x02, struct.pack("<HH", 63, 1)))
        pkt = good.out
        self.assertEqual(ss.find_status(pkt, 1), (0, bytes([53])))
        self.assertIsNone(ss.find_status(pkt[:-1] + bytes([pkt[-1] ^ 0xFF]), 1))
        self.assertIsNone(ss.find_status(ss.packet(1, 0x02, b"\x3f\x00\x01\x00"), 1))
        self.assertIsNone(ss.find_status(pkt, 2))


# ---------------------------------------------------------------------------------------------- check_voltage_limit.py
class Assess(unittest.TestCase):
    def test_bench_rail_messages_unchanged(self):
        self.assertEqual(cvl.assess(1, 70, 53, 0, 6.8, "7v"), ([], []))
        p, n = cvl.assess(2, 70, 53, 0x01, 7.1, "7v")
        self.assertEqual(p, ["ID2 has a latched Input Voltage Error (Hardware Error Status 0x01); power-cycle after fixing the rail",
                             "ID2 reads 7.1 V, outside 6.5-7.0 V"])
        self.assertEqual(n, ["     warning: ID2 reads above its Max Voltage Limit with the input-voltage shutdown armed"])
        p, n = cvl.assess(3, 70, 53, 0, 6.95, "7v")
        self.assertEqual(n, ["     warning: ID3 reads 7.0 V, above 6.9 V: re-trim XL4015 #1 to 6.80 V"])
        p, n = cvl.assess(4, 70, 52, 0, 6.8, "7v")
        self.assertEqual(n, ["     note: ID4 Shutdown bit 0 is clear, so an input-voltage error does not switch torque off"])
        self.assertEqual(cvl.assess(5, 70, 53, 0, 5.0, "none"), ([], []))

    def test_battery_pass_with_the_voltage_flag(self):
        for volts in (6.4, 7.4, 8.2, 8.4):
            p, _ = cvl.assess(1, 70, 52, 0x01, volts, "battery")
            self.assertEqual(p, [], volts)

    def test_battery_failures(self):
        self.assertIn("expected 52", cvl.assess(1, 70, 53, 0, 7.4, "battery")[0][0])
        self.assertTrue(any("torque is cut" in x for x in cvl.assess(1, 70, 53, 0x01, 7.4, "battery")[0]))
        self.assertIn("outside 6.4-8.4 V", cvl.assess(1, 70, 52, 0, 8.5, "battery")[0][0])
        self.assertIn("outside 6.4-8.4 V", cvl.assess(1, 70, 52, 0, 6.3, "battery")[0][0])
        self.assertIn("expected the factory 7.0 V", cvl.assess(1, 65, 52, 0, 7.4, "battery")[0][0])

    def test_battery_warnings(self):
        self.assertEqual(cvl.assess(1, 70, 52, 0, 7.0, "battery"), ([], []))
        self.assertIn("charge the pack before walking", cvl.assess(1, 70, 52, 0, 6.9, "battery")[1][0])
        self.assertIn("battery stop", cvl.assess(1, 70, 52, 0, 6.6, "battery")[1][0])


def run_check(world, rail):
    out = io.StringIO()
    with mock.patch.object(cvl.serial, "Serial", lambda *a, **k: FakeSerial(world)), contextlib.redirect_stdout(out):
        code = cvl.main(["--port", "fake", "--rail", rail])
    return code, out.getvalue()


class CheckVoltageMain(unittest.TestCase):
    def test_battery_mode_passes_on_a_full_pack(self):
        code, text = run_check(World(shutdown=52, hw_err=0x01, volts_dv=84), "battery")
        self.assertEqual(code, 0, text)
        self.assertIn("PASS: 15/15 servos, Max Voltage Limit 7.0 V on all, Shutdown 52 on all, supply 6.4-8.4 V on all", text)
        self.assertIn("HWerr 0x01 is the input-voltage flag", text)
        self.assertNotIn("Shutdown bit 0 is clear", text)

    def test_battery_mode_fails_on_a_factory_servo(self):
        w = World(shutdown=52, volts_dv=76)
        w.regs[12][63] = 53
        code, text = run_check(w, "battery")
        self.assertEqual(code, 1)
        self.assertIn("ID12 Shutdown 53 (0x35), expected 52", text)

    def test_bench_mode_still_fails_on_a_battery(self):
        code, text = run_check(World(shutdown=52, hw_err=0x01, volts_dv=84), "7v")
        self.assertEqual(code, 1)
        self.assertIn("outside 6.5-7.0 V", text)

    def test_bench_mode_passes_on_the_7v_rail(self):
        code, text = run_check(World(shutdown=53, volts_dv=68), "7v")
        self.assertEqual(code, 0, text)
        self.assertTrue(text.rstrip().endswith("PASS: 15/15 servos, Max Voltage Limit 7.0 V on all, supply 6.5-7.0 V on all"))


# ---------------------------------------------------------------------------------------------- dynamixel_sdk stand-in
class FakePacketHandler:
    def __init__(self, world):
        self.world = world

    def _read(self, i, addr, size):
        if i not in self.world.regs:
            return 0, COMM_RX_TIMEOUT, 0
        return self.world.read(i, addr, size), COMM_SUCCESS, self.world.err(i)

    def _write(self, i, addr, size, value):
        if i not in self.world.regs:
            return COMM_RX_TIMEOUT, 0
        return COMM_SUCCESS, self.world.write(i, addr, size, value)

    def read1ByteTxRx(self, port, i, addr):
        return self._read(i, addr, 1)

    def read4ByteTxRx(self, port, i, addr):
        return self._read(i, addr, 4)

    def write1ByteTxRx(self, port, i, addr, v):
        return self._write(i, addr, 1, v)

    def write2ByteTxRx(self, port, i, addr, v):
        return self._write(i, addr, 2, v)

    def write4ByteTxRx(self, port, i, addr, v):
        return self._write(i, addr, 4, v)

    def reboot(self, port, i):
        return COMM_SUCCESS, 0

    def getTxRxResult(self, r):
        return f"result {r}"

    def getRxPacketError(self, err):
        return f"error 0x{err:02x}"


def install_sdk(world):
    sdk = types.ModuleType("dynamixel_sdk")
    sdk.COMM_SUCCESS = COMM_SUCCESS

    class PortHandler:
        def __init__(self, name):
            pass

        def openPort(self):
            return True

        def setBaudRate(self, baud):
            return True

        def closePort(self):
            pass

    class GroupSyncRead:
        def __init__(self, port, pk, addr, size):
            self.addr, self.size, self.ids = addr, size, []

        def addParam(self, i):
            self.ids.append(i)

        def txRxPacket(self):
            if self.addr in world.fail_sync_reads:
                return COMM_RX_TIMEOUT
            return COMM_SUCCESS if all(i in world.regs for i in self.ids) else COMM_RX_TIMEOUT

        def getData(self, i, addr, size):
            return world.read(i, addr, size)

    class GroupSyncWrite:
        def __init__(self, port, pk, addr, size):
            self.addr, self.size, self.values = addr, size, {}

        def addParam(self, i, data):
            self.values[i] = int.from_bytes(bytes(data), "little", signed=True)

        def txPacket(self):
            for i, v in self.values.items():
                world.write(i, self.addr, self.size, v)
            return COMM_SUCCESS

    sdk.PortHandler, sdk.GroupSyncRead, sdk.GroupSyncWrite = PortHandler, GroupSyncRead, GroupSyncWrite
    sdk.PacketHandler = lambda version: FakePacketHandler(world)
    return sdk


def import_with_sdk(world, *names):
    sdk = install_sdk(world)
    with mock.patch.dict(sys.modules, {"dynamixel_sdk": sdk}):
        for n in names + ("scan_bus", "seam_fix"):
            sys.modules.pop(n, None)
        mods = [__import__(n) for n in names]
        for n in names + ("scan_bus", "seam_fix"):
            sys.modules.pop(n, None)
    return mods


class AlertTolerance(unittest.TestCase):
    def test_seam_fix_runs_on_the_battery(self):
        w = World(shutdown=52, hw_err=0x01, volts_dv=82)
        (seam_fix,) = import_with_sdk(w, "seam_fix")
        raw, homing, pos = seam_fix.fix_seam(None, FakePacketHandler(w))
        self.assertEqual((raw, homing), (2048, -2048))

    def test_seam_fix_still_stops_on_a_real_hardware_error(self):
        w = World(shutdown=52, hw_err=0x21, volts_dv=82)  # overload as well as the voltage flag
        (seam_fix,) = import_with_sdk(w, "seam_fix")
        with self.assertRaises(RuntimeError):
            seam_fix.fix_seam(None, FakePacketHandler(w))

    def test_seam_fix_stops_on_the_voltage_flag_with_shutdown_53(self):
        w = World(shutdown=53, hw_err=0x01, volts_dv=82)
        (seam_fix,) = import_with_sdk(w, "seam_fix")
        with self.assertRaises(RuntimeError):
            seam_fix.fix_seam(None, FakePacketHandler(w))

    def test_stand_bus_check(self):
        w = World(shutdown=52, hw_err=0x01)
        (stand,) = import_with_sdk(w, "stand")
        bus = stand.Bus.__new__(stand.Bus)
        bus.port, bus.pk = None, FakePacketHandler(w)
        bus.check(3, COMM_SUCCESS, 0x80, "ok")  # the battery's voltage flag passes
        with self.assertRaises(RuntimeError):
            bus.check(3, COMM_SUCCESS, 0x87, "access error with alert")
        with self.assertRaises(RuntimeError):
            bus.check(3, COMM_RX_TIMEOUT, 0, "no answer")
        w.regs[3][70] = 0x04  # overheating
        with self.assertRaises(RuntimeError):
            bus.check(3, COMM_SUCCESS, 0x80, "overheating")


# ---------------------------------------------------------------------------------------------- stand.py battery stop
class FakeClock:
    """time.time / time.sleep for stand.py; sleeping advances the clock and sets the servos' supply voltage."""

    def __init__(self, world, profile, hooks=()):
        self.t, self.world, self.profile, self.hooks = 1000.0, world, profile, list(hooks)

    def time(self):
        return self.t

    def monotonic(self):
        return self.t - 900.0

    def sleep(self, s):
        self.t += s
        self.world.set_all(144, 2, self.profile(self.t - 1000.0))
        while self.hooks and self.t - 1000.0 >= self.hooks[0][0]:
            self.hooks.pop(0)[1]()


class StandBatteryStop(unittest.TestCase):
    def run_stand(self, profile, extra=lambda d: (), hold=16.0, hooks=lambda d: (), marker_name="stop.json",
                  old_marker=False, world=None):
        w = world or World(shutdown=52, hw_err=0x01, volts_dv=profile(0.0))
        (stand,) = import_with_sdk(w, "stand")
        with tempfile.TemporaryDirectory() as d:
            clock = FakeClock(w, profile, hooks(d))
            marker = os.path.join(d, marker_name)
            if old_marker:
                with open(marker, "w") as fh:
                    json.dump({"volts": 6.5}, fh)
            argv = ["stand.py", "--port", "fake", "--hold", str(hold), "--battery-marker", marker] + list(extra(d))
            out = io.StringIO()
            with mock.patch.object(stand, "time", clock), mock.patch.object(sys, "argv", argv), \
                    contextlib.redirect_stdout(out):
                self.code = stand.main()
            marker_data = None
            if os.path.exists(marker):
                with open(marker) as fh:
                    marker_data = json.load(fh)
        return w, stand, out.getvalue(), marker_data

    def test_stop_ramps_to_stand_pose_holds_torque_and_signals(self):
        # 7.6 V, then the pack collapses to 6.5 V at 5 s; a head_pitch override is live until the stop, and a new
        # override arriving at 12 s (after the stop) must not move the robot off the stand pose.
        def targets(d):
            path = os.path.join(d, "targets.json")
            with open(path, "w") as fh:
                json.dump({"head_pitch": 0.2}, fh)
            return ["--targets", path]

        def late_override(d):
            def write():
                path = os.path.join(d, "targets.json")
                with open(path, "w") as fh:
                    json.dump({"head_pitch": -0.3}, fh)
                os.utime(path, (2e9, 2e9))
            return [(12.0, write)]
        w, stand, text, marker = self.run_stand(lambda t: 76 if t < 5 else 65, targets, hooks=late_override)
        self.assertIn("battery guard armed", text)
        self.assertIn("BATTERY LOW", text)
        self.assertIn("BATTERY STOP: 6.5 V; ramping to the stand pose and holding", text)
        self.assertIn("Hold the robot", text)
        self.assertEqual(marker["volts"], 6.5)
        torque_writes = [(k, i, v) for k, (a, i, v) in enumerate(w.log) if a == 64]
        offs = [k for k, i, v in torque_writes if v == 0]
        ons = [k for k, i, v in torque_writes if v == 1]
        # torque off once before the enable, then only by the end of --hold (the existing finally), never by the guard
        last_goal = max(k for k, (a, i, v) in enumerate(w.log) if a == 116)
        self.assertTrue(all(k < min(ons) or k > last_goal for k in offs))
        stand_ticks = {stand.MOTOR_TO_ID[n]: stand.ticks(n) for n in stand.MOTOR_TO_ID}
        final = {i: w.read(i, 116, 4) for i in stand_ticks}
        self.assertEqual(final, stand_ticks)  # the override is gone: plain STAND_POSE
        between = text[text.index("BATTERY STOP"):text.index("Hold the robot")]
        self.assertIn("\nramp", between)  # the signal waits for the ramp back to the stand pose

    def test_bench_rail_never_arms(self):
        w, stand, text, marker = self.run_stand(lambda t: 68 if t < 5 else 64, hold=10.0)
        self.assertNotIn("BATTERY", text)
        self.assertIn("battery guard not armed", text)
        self.assertIsNone(marker)

    def assert_held_until_the_end(self, w):
        torque = [(k, v) for k, (a, i, v) in enumerate(w.log) if a == 64]
        first_on = min(k for k, v in torque if v == 1)
        last_goal = max(k for k, (a, i, v) in enumerate(w.log) if a == 116)
        self.assertTrue(all(k < first_on or k > last_goal for k, v in torque if v == 0))
        self.assertGreater(len([1 for a, i, v in w.log if a == 116]), 14 * 500)  # the loop ran the whole hold

    def test_missed_voltage_reads_do_not_stop_the_hold(self):
        # the pack collapses, but every voltage read in the hold fails: no stop, no release, the hold runs to its end
        w = World(shutdown=52, hw_err=0x01, volts_dv=76)
        def glitch(d):
            return [(2.0, lambda: w.fail_sync_reads.add(144))]
        _, _, text, marker = self.run_stand(lambda t: 76 if t < 3 else 60, hold=12.0, hooks=glitch, world=w)
        self.assertNotIn("BATTERY STOP", text)
        self.assertIsNone(marker)
        self.assert_held_until_the_end(w)

    def test_unwritable_marker_does_not_end_the_hold(self):
        w, _, text, marker = self.run_stand(lambda t: 76 if t < 5 else 65, marker_name="missing-dir/stop.json")
        self.assertIn("Marker NOT written", text)
        self.assertNotIn("STOP:", text.split("Marker NOT written")[1].replace("BATTERY STOP", ""))
        self.assert_held_until_the_end(w)

    def test_restart_on_a_drained_pack_is_refused(self):
        w, _, text, marker = self.run_stand(lambda t: 70, old_marker=True)
        self.assertEqual(self.code, 1)
        self.assertIn("torque refused", text)
        self.assertNotIn((64, 1, 1), w.log)
        self.assertFalse(any(a == 64 and v == 1 for a, i, v in w.log))
        self.assertIsNotNone(marker)  # still there

    def test_a_charged_pack_clears_the_marker(self):
        w, _, text, marker = self.run_stand(lambda t: 80, old_marker=True, hold=4.0)
        self.assertNotIn("torque refused", text)
        self.assertTrue(any(a == 64 and v == 1 for a, i, v in w.log))
        self.assertIsNone(marker)


if __name__ == "__main__":
    unittest.main()
