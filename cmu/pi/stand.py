"""Ramp MicroDuck to STAND_POSE (community NEUTRAL_POSE plus this build's lean-back trim) and hold it, with current and error watch. No IMU needed.

Follows the community runtime's ramp_to_neutral (Microduck-build-tutorial/microduck/src/main.py) and its
constants: hardware rad = model rad * sign + offset (signs below), knees +-45 deg, 2048 ticks = 0 rad, KP 400.
The jaw (ID15) stays torque-off. ID5 gets the encoder-seam fix (seam_fix.py) first.

  python3 stand.py --port /dev/cu.usbmodemXXXX --dry      # print present vs target per joint, no torque
  python3 stand.py --port /dev/cu.usbmodemXXXX            # torque on, ramp, hold, torque off
Holds until Ctrl-C or SIGTERM (or --hold seconds). Hold the robot when it stops: torque off drops it.

Battery guard (battery.py), --supply auto by default: once the servos read 7.2 V or more (only the 2S pack does), it
reads Present Input Voltage once a second, warns under 7.0 V and at 6.6 V ramps back to STAND_POSE without the
--targets overrides, keeps holding with torque on and writes /tmp/duck_battery_stop.json. While that marker exists
torque is not switched on until the servos read 7.6 V or more (a charged or swapped pack), which deletes it.
--supply battery applies it from the start; --supply bench switches it off. A failed voltage read is a skipped sample
and never ends the hold.
"""
import argparse
import json
import os
import math
import signal
import sys
import time

from dynamixel_sdk import COMM_SUCCESS, GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler

from battery import (ADDR_HW_ERROR as ADDR_HW_ERROR_STATUS, ADDR_PRESENT_VOLTAGE, ADDR_SHUTDOWN, ALERT_BIT,
                     STOP_MARKER, BatteryGuard, voltage_alert_only)
from scan_bus import ROLES
from seam_fix import fix_seam

MOTOR_TO_ID = {"right_hip_yaw": 5, "right_hip_roll": 4, "right_hip_pitch": 3, "right_knee": 2, "right_ankle": 1,
               "left_hip_yaw": 10, "left_hip_roll": 9, "left_hip_pitch": 8, "left_knee": 7, "left_ankle": 6,
               "neck_pitch": 12, "head_pitch": 11, "head_yaw": 13, "head_roll": 14}
from stand_pose import NEUTRAL_POSE, STAND_POSE  # noqa: E402  (the one stand pose source)
from joint_signs import MOTOR_OFFSET, MOTOR_SIGN  # noqa: E402  (the one sign source, checked joint by joint)
KP = 400

ADDR_TORQUE_ENABLE, ADDR_HW_ERROR = 64, 70
ADDR_P_GAIN, ADDR_PROFILE_ACC, ADDR_PROFILE_VEL = 84, 108, 112
ADDR_GOAL_POSITION, ADDR_PRESENT_CURRENT, ADDR_PRESENT_POSITION = 116, 126, 132
IDS = list(MOTOR_TO_ID.values())
SERVO_LIMIT_MA, TOTAL_LIMIT_MA, TOTAL_LIMIT_S = 1300, 3000, 1.0


def ticks(name, overrides=None):
    q = (overrides or {}).get(name, STAND_POSE[name])
    return round(2048 + (q * MOTOR_SIGN[name] + MOTOR_OFFSET[name]) * 4096 / (2 * math.pi))


def model_deg(name, tick):
    return math.degrees(((tick - 2048) * 2 * math.pi / 4096 - MOTOR_OFFSET[name]) * MOTOR_SIGN[name])


def load_targets(path):
    if not path or not os.path.exists(path):
        return {}, None
    return {k: float(v) for k, v in json.load(open(path)).items()}, os.path.getmtime(path)


def s(v, bits):
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def check(pk, r, err, what):
    if r != COMM_SUCCESS or err:
        raise RuntimeError(f"{what}: {pk.getTxRxResult(r)} {pk.getRxPacketError(err)}")


class Bus:
    def __init__(self, name):
        self.port = PortHandler(name)
        self.pk = PacketHandler(2.0)
        if not self.port.openPort() or not self.port.setBaudRate(1_000_000):
            sys.exit(f"Cannot open {name}")
        self.pos_read = GroupSyncRead(self.port, self.pk, ADDR_PRESENT_POSITION, 4)
        self.cur_read = GroupSyncRead(self.port, self.pk, ADDR_PRESENT_CURRENT, 2)
        self.volt_read = GroupSyncRead(self.port, self.pk, ADDR_PRESENT_VOLTAGE, 2)
        self.last = {}
        for i in IDS:
            self.pos_read.addParam(i)
            self.cur_read.addParam(i)
            self.volt_read.addParam(i)

    def read(self, group, addr, size):
        # A dropped status packet on the UART bridge must never release torque: retry, then reuse the last reading.
        for _ in range(5):
            r = group.txRxPacket()
            if r == COMM_SUCCESS:
                self.last[addr] = {i: s(group.getData(i, addr, size), 8 * size) for i in IDS}
                return self.last[addr]
        if addr in self.last:
            print(f"warning: sync read {addr} failed 5 times, reusing last reading", flush=True)
            return self.last[addr]
        raise RuntimeError(f"sync read {addr}: {self.pk.getTxRxResult(r)}")

    def positions(self):
        return self.read(self.pos_read, ADDR_PRESENT_POSITION, 4)

    def currents(self):
        return self.read(self.cur_read, ADDR_PRESENT_CURRENT, 2)

    def voltages(self):
        """Present Input Voltage (0.1 V) per ID; {} when the read fails, which the battery guard skips."""
        if self.volt_read.txRxPacket() != COMM_SUCCESS:
            return {}
        return {i: self.volt_read.getData(i, ADDR_PRESENT_VOLTAGE, 2) for i in IDS}

    def check(self, i, r, err, what):
        """check() for one servo; on the 2S pack the Alert bit for the input-voltage flag alone passes (battery.py)."""
        if r == COMM_SUCCESS and err == ALERT_BIT and self.voltage_alert_only(i):
            return
        check(self.pk, r, err, what)

    def voltage_alert_only(self, i):
        hw, r1, _ = self.pk.read1ByteTxRx(self.port, i, ADDR_HW_ERROR_STATUS)
        shutdown, r2, _ = self.pk.read1ByteTxRx(self.port, i, ADDR_SHUTDOWN)
        return r1 == COMM_SUCCESS and r2 == COMM_SUCCESS and voltage_alert_only(hw, shutdown)

    def write_goals(self, goals):
        w = GroupSyncWrite(self.port, self.pk, ADDR_GOAL_POSITION, 4)
        for i, g in goals.items():
            w.addParam(i, int(g).to_bytes(4, "little", signed=True))
        for _ in range(5):
            if w.txPacket() == COMM_SUCCESS:
                return
        print("warning: goal write failed 5 times", flush=True)

    def each(self, addr, size, value):
        for i in IDS:
            fn = {1: self.pk.write1ByteTxRx, 2: self.pk.write2ByteTxRx, 4: self.pk.write4ByteTxRx}[size]
            self.check(i, *fn(self.port, i, addr, value), f"ID{i} addr {addr}")

    def torque_off(self):
        for _ in range(3):
            for i in IDS:
                self.pk.write1ByteTxRx(self.port, i, ADDR_TORQUE_ENABLE, 0)


def _sigterm(*_):
    raise KeyboardInterrupt


def battery_event(guard, event):
    if event == "armed":
        print(f"battery guard armed: the servos read {guard.dv / 10:.1f} V", flush=True)
    elif event == "unarmed":
        print(f"battery guard not armed: the servos read {guard.dv / 10:.1f} V, a bench rail; on the 2S "
              "pack restart with --supply battery", flush=True)
    elif event == "warn":
        print(f"BATTERY LOW: {guard.dv / 10:.1f} V, under 7.0 V; finish and charge the pack", flush=True)
    elif event == "stop":
        print(f"BATTERY STOP: {guard.dv / 10:.1f} V; ramping to the stand pose and holding", flush=True)


def main():
    signal.signal(signal.SIGTERM, _sigterm)
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--ramp", type=float, default=3.0, help="seconds to reach the pose")
    ap.add_argument("--hold", type=float, default=0.0, help="seconds to hold before torque off; 0 = until Ctrl-C / SIGTERM")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--takeover", action="store_true", help="servos already hold torque: keep them on, start from present")
    ap.add_argument("--targets", default="", help="JSON {joint: model rad} overrides, re-read live while holding")
    ap.add_argument("--state-out", default="", help="write present joint angles (model deg) here every second")
    ap.add_argument("--supply", choices=("auto", "bench", "battery"), default="auto",
                    help="battery guard: auto arms at 7.2 V (2S pack), battery from the start, bench off")
    ap.add_argument("--battery-marker", default=STOP_MARKER)
    args = ap.parse_args()
    guard = None if args.supply == "bench" else BatteryGuard(args.supply)
    bus = Bus(args.port)
    if args.takeover:
        mode, _, _ = bus.pk.read1ByteTxRx(bus.port, 5, 11)
        if mode != 4:
            sys.exit("takeover needs ID5 already in Extended Position mode from fix_seam")
    else:
        fix_seam(bus.port, bus.pk)
    start = bus.positions()
    if args.takeover:  # continue from the goals the previous process left, so holding force does not dip
        for i in IDS:
            g, r, err = bus.pk.read4ByteTxRx(bus.port, i, ADDR_GOAL_POSITION)
            bus.check(i, r, err, f"ID{i} goal read")
            start[i] = s(g, 32)
    overrides, overrides_mtime = load_targets(args.targets)
    target = {MOTOR_TO_ID[n]: ticks(n, overrides) for n in MOTOR_TO_ID}
    for n, i in sorted(MOTOR_TO_ID.items(), key=lambda kv: kv[1]):
        d = (target[i] - start[i]) * 360 / 4096
        print(f"ID{i:>2} {ROLES[i]:<18} present {start[i]:>5}  target {target[i]:>5}  move {d:+6.1f} deg")
    if args.dry:
        return 0
    if guard and not args.takeover:  # never switch torque on from a pack that a battery stop left empty
        for _ in range(3):
            battery_event(guard, guard.update(time.monotonic(), bus.voltages()))
            if guard.dv is not None:
                break
        block = guard.torque_block(args.battery_marker)
        if block:
            print(f"torque refused: {block}")
            return 1
    try:
        if not args.takeover:
            bus.each(ADDR_TORQUE_ENABLE, 1, 0)
            bus.each(ADDR_P_GAIN, 2, KP)
            bus.each(ADDR_PROFILE_VEL, 4, 0)
            bus.each(ADDR_PROFILE_ACC, 4, 0)
            bus.write_goals(start)
            bus.each(ADDR_TORQUE_ENABLE, 1, 1)
        t0, over_since, peak_total = time.time(), None, 0
        end = t0 + args.ramp + args.hold if args.hold > 0 else float('inf')
        last_print = 0
        last_check = 0
        battery_stop = False
        while time.time() < end:
            a = min(1.0, (time.time() - t0) / args.ramp)
            a = a * a * (3 - 2 * a)  # smoothstep
            goal = {i: start[i] + (target[i] - start[i]) * a for i in IDS}
            bus.write_goals(goal)
            if guard and guard.due(time.monotonic()):  # monotonic: the Pi's wall clock jumps when NTP syncs
                event = guard.update(time.monotonic(), bus.voltages())
                battery_event(guard, event)
                if event == "stop":
                    # back to STAND_POSE without the --targets overrides, then keep holding with torque on
                    battery_stop, start, t0, a = True, goal, time.time(), 0.0
                    target = {MOTOR_TO_ID[n]: ticks(n) for n in MOTOR_TO_ID}
            if battery_stop and a >= 1.0:
                msg = guard.signal(args.battery_marker, "stand pose, torque on (stand.py releases torque when it stops)")
                if msg:
                    print(msg, flush=True)
            if args.targets and not battery_stop and time.time() - last_check >= 0.5:
                last_check = time.time()
                new, mt = load_targets(args.targets)
                if mt != overrides_mtime:
                    overrides_mtime = mt
                    start, t0 = goal, time.time()
                    target = {MOTOR_TO_ID[n]: ticks(n, new) for n in MOTOR_TO_ID}
                    print(f"new targets: {new}", flush=True)
            cur = bus.currents()
            total = sum(abs(c) for c in cur.values())
            peak_total = max(peak_total, total)
            worst = max(IDS, key=lambda i: abs(cur[i]))
            if abs(cur[worst]) > SERVO_LIMIT_MA:
                raise RuntimeError(f"ID{worst} {ROLES[worst]} at {cur[worst]} mA")
            over_since = (over_since or time.time()) if total > TOTAL_LIMIT_MA else None
            if over_since and time.time() - over_since >= TOTAL_LIMIT_S:
                raise RuntimeError(f"total current {total} mA for {TOTAL_LIMIT_S} s")
            if time.time() - last_print >= 1.0:
                last_print = time.time()
                if args.state_out:
                    pos = bus.positions()
                    json.dump({n: round(model_deg(n, pos[i]), 1) for n, i in MOTOR_TO_ID.items()},
                              open(args.state_out, "w"), indent=1)
                phase = "ramp" if a < 1 else ("holding" if end == float("inf") else f"hold {end - time.time():4.1f}s left")
                print(f"{phase:<18} total {total:5d} mA  max ID{worst} {cur[worst]:+5d} mA"
                      + (f"  {guard.describe()}" if guard else ""), flush=True)
            time.sleep(0.02)
        for i in IDS:
            e, _, _ = bus.pk.read1ByteTxRx(bus.port, i, ADDR_HW_ERROR)
            if e:
                print(f"ID{i} hardware error 0x{e:02x}")
        pos = bus.positions()
        print(f"peak total {peak_total} mA; final error per joint (deg): "
              + ", ".join(f"{i}:{(pos[i] - target[i]) * 360 / 4096:+.1f}" for i in IDS))
    except KeyboardInterrupt:
        print("stopped by Ctrl-C")
    except RuntimeError as e:
        print(f"STOP: {e}")
    finally:
        bus.torque_off()
        print("torque off")
    return 0


if __name__ == "__main__":
    sys.exit(main())
