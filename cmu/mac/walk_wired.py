"""Wired MicroDuck: the MacBook drives the servos over the OpenRB USB port; the Pi streams the IMU over J12.

Needs openrb_dual_host on the OpenRB and pi_imu_uart.py on the Pi (instead of pi_bridge.py). Same modes and
control file as walk_brain.py (monitor / stand / walk), same policies and compensation; only the transport changes.

Torque: nothing is written in monitor mode. The first stand or walk command writes the present positions as goals
and only then enables torque, so the servos start holding where they are. Stand mode then ramps to STAND; walk
mode runs stand61 (before the first step) and walk61 from the present pose with no open-loop ramp (walk_brain.py).
Torque goes off only on {"mode": "off"} (the remote's torque-off key): every servo, the jaw included, and any servo
with a latched hardware error (overload, overheating) gets the Reboot instruction, which clears the latch without a
power cycle; the control file then reads monitor, and the next stand or walk command enables torque as above.
Stopping this process leaves the servos holding their last goals.

Battery guard (pi/battery.py), --supply auto by default: once the servos read 7.2 V or more (only the 2S pack does),
Present Input Voltage is read once a second; under 7.0 V it warns, at 6.6 V it replaces the control file: walking gets
zero velocity, so walk61 stops its own steps and keeps the robot balanced (no open-loop ramp to STAND, which falls in
the simulation), stand mode returns to plain STAND, torque stays on, and once that pose is held (1.5 s at zero command)
/tmp/duck_battery_stop.json is written. With torque still off it stays in monitor. While that marker exists torque is
not switched on until the servos read 7.6 V or more (a charged or swapped pack), which deletes it. --supply battery
applies the guard from the start; --supply bench switches it off (the bench rail sags under 6.6 V while walking,
which the guard reads as an empty pack) but keeps the Alert-bit packets and the hardware-error watch. A failed voltage read is a skipped sample and never
releases torque. Armed, the loop also reads Hardware Error Status once a second; a fault other than the battery's
input-voltage flag freezes the policy on its last goals, as a dropped status packet does on the bench.

Pickup additions (2026-10-16, walk_brain.py and jaw_ctl.py have the details):
  {"mode": "walk", "vx": 0, "vy": 0, "vtheta": 0, "head": {"head_yaw": 0.6}}   head pose command to stand61/walk61
  {"mode": "groundpick", "phase": 0.33}   Pollen's ground-pick policy descends to the phase and holds it (0.33 is
                                          the lowest pose: as-built sim 2026-10-16, mouth tip 22 mm over the floor,
                                          69 mm ahead of the feet, trunk 35 deg); "phase": 1.0 rises; once the rise
                                          is complete this process writes walk mode at zero command (stand61). Only
                                          the hard action limit and --max-tilt-gp guard it; a trip holds the legs'
                                          last goals (monitor mode, torque on) instead of ramping to STAND from a
                                          crouch; the head joints hold their positions at the trip.
  "jaw": 20 | "open" | "grip", "jaw_torque": false, "jaw_mark": "closed" | "open"   the ID15 jaw (jaw_ctl.py)

Remote-control additions (2026-10-20, walk_brain.py has the details):
  {"mode": "kick", "foot": "left" | "right", "id": 7}   one kick per new id, from stand or walk mode: 25 ticks of
                                          Pollen's kick policy at zero command, then this process rewrites the
                                          control file to walk at zero command (stand61) if it still holds that kick.
                                          The walk guards apply; a trip stops the kick and goes to stand.
  {"mode": "sit"}                         Pollen's sitstand policy (flag 0 for 0.8 s, then 1); any stand or walk
                                          command stands it up (flag 0 for 2 s) before running. --max-tilt-sit (30)
                                          and the hard action limit guard it; a trip holds the legs' last goals
                                          (monitor) and the head holds where it is. Policy goals stay within the
                                          trained joint ranges (walk_brain.JOINT_RANGE).
  "quack": N                              any mode; a new value of the counter (start at 1) plays one random call
                                          from --sounds through `aplay -q` on the ALSA default device (duck_sound.py,
                                          off the control thread, a new call kills the old one) and, when the jaw is
                                          calibrated, opens it to QUACK_JAW_DEG for QUACK_S under the jaw limits,
                                          enabling its torque first if needed; a jaw that was not held open is then
                                          pressed shut for QUACK_CLOSE_S and held at QUACK_HOLD_DEG. A gripping jaw
                                          stays shut. The head also moves gently to a random QUACK_HEAD pose (a tilt or a
                                          look up) and back
                                          over QUACK_HEAD_S whenever a policy can rebalance under it: walk mode at zero
                                          command (stand61, or walk61 QUACK_REST_S after a stop) and seated. Never
                                          while stepping, sitting down or standing up, and never in the open-loop stand
                                          mode (it tipped over there on 2026-10-20). The quack line in stdout names the
                                          head move or why the head stayed (quack_head_block).
  Hold-to-walk: rewrite {"mode": "walk", "vx": 0.25, ..., "for_s": 0.6} every 0.3 s; every rewrite restarts for_s,
  so the walk stops 0.6 s after the last one.

Tick log additions (2026-10-21): cur_ma, the Present Current of the 14 body servos in mA (IDS order, like pos_tick;
read in the same sync read as velocity and position), and jaw_cur_ma; touch, the raw counts of the head pads L1, R1, L2,
R2, M, with touch_seq (the XIAO's scan number), touch_age (s since that scan arrived) and touch_bad (failed reads so far),
read on their own thread when the A7Z's TWI0 bus exists (--touch-bus), null otherwise.

Graded hold (hold.py, 2026-10-21): every tick estimates from the head pads and the leg current how much a person has
taken over the robot and logs it (hold, legs_ma). While --hold-file (~/.duck_hold.json) says {"on": true} with the
leg currents calibrated and walk mode runs at zero command, the estimate scales stand61/walk61's actions toward the
default stance (s), softens the leg gains and stiffens the neck as the feet unload, and hands everything back within
0.1 s of the hand leaving (hold_on true in the log). With the grip engaged (e > HOLD_TILT_E) the tilt stop does not fire: being carried tilted
is not a fall. The file is re-read when it changes; hold-20261022/hold_session.py writes it.

Walk stops: {"mode": "walk", ..., "for_s": 7} sets zero velocity 7 s after the command arrived; walk61 brings its own
stride to rest (torque on), the sequence brain_sim_check.py verifies on the as-built model. An action over 3.0 rad in 3
of 10 ticks or body tilt over --max-tilt deg (30) switches to stand mode; a single tick over 3.0 rad is clipped to
+-3.0, and smaller actions are left to the policy, as in Pollen's runtime. Each stop is written to the control file, so it holds until a new command.

  ~/.venvs/microduck-walk/bin/python walk_wired.py [--port /dev/cu.usbmodem101]
"""
import argparse
import collections
import fcntl
import glob
import json
import math
import os
import struct
import sys
import threading
import time

import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
from walk_brain import IDS, NAMES, OBSERVATION_DOF_ORDER, STAND, WHOLE, Brain, Breakaway, read_cmd, to_model, to_tick  # noqa: E402
import jaw_ctl  # noqa: E402
import walk_brain  # noqa: E402
from duck_sound import Voice  # noqa: E402
from hold import DOWN_RATE, HOLD_FILE, KP_STEP, UP_RATE, HoldEstimator, legs_current_ma, read_hold_file  # noqa: E402
from jaw_ctl import GOAL_CURRENT_MA, JAW_ID, POSE_JOINTS, JawCtl  # noqa: E402
from dxl_fast import _packet  # noqa: E402
from battery import (ADDR_HW_ERROR, ADDR_PRESENT_VOLTAGE, ADDR_SHUTDOWN, INPUT_VOLTAGE_BIT, STOP_MARKER,  # noqa: E402
                     BatteryGuard, status_ok, stop_command, stop_settled, voltage_alert_only)

ADDR_TORQUE, ADDR_P_GAIN, ADDR_GOAL, ADDR_VEL = 64, 84, 116, 128
# The per-tick state read: Present Current (126, 2 B signed, 1 mA; the XL330 measures it at its supply input), Present
# Velocity (128, 4 B) and Present Position (132, 4 B) as one 10-byte block. The current adds 2 bytes to each of the 15
# status packets: 0.3 ms on the 1 Mbaud servo bus, 0.6 ms on the A7Z's 500 kbaud J12 link, no extra transaction.
ADDR_CUR = 126
STATE_FMT = "<hii"
STATE_SIZE = struct.calcsize(STATE_FMT)
IMU_FMT = "<I11fB"


def imu_frame_ok(f):
    """Reject a frame whose numbers cannot come from the BNO085: unit quaternion, sane gyro, sane age."""
    quat, gyro, age = f[1:5], f[5:8], f[11]
    if not all(math.isfinite(x) for x in f[1:12]):
        return False
    return 0.9 < math.sqrt(sum(x * x for x in quat)) < 1.1 and all(abs(g) < 35.0 for g in gyro) and 0.0 <= age < 5.0
PERIOD = 0.02


# The robot I/O lock: walk_wired.py holds it for its whole run; duck_sense.py (the read-only standby reader behind the
# live page) holds it only while no brain process exists, and gives it up within 0.5 s of seeing one, so the two never
# share the servo bus or the BNO085.
HW_LOCK = "/tmp/duck-hw.lock"


def take_hw_lock(path=HW_LOCK):
    f = open(path, "a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("waiting for duck_sense.py to hand over the servo bus and IMU", flush=True)
        fcntl.flock(f, fcntl.LOCK_EX)
    return f


class Link:
    """USB serial to openrb_dual_host: DYNAMIXEL status packets and Pi IMU frames arrive as whole units."""

    def __init__(self, port, baud=1_000_000, imu_bus=None):
        self.ser = serial.Serial(port, baud, timeout=0.002)
        self.status = collections.deque()
        self.imu = None  # (arrival perf_counter, payload tuple)
        self.imu_count = 0
        self.imu_bad = 0  # frames that passed the 8-bit sum but failed imu_frame_ok
        self.late = 0
        self.alert_ok = False  # True once the battery guard is armed (battery.py); the bench stays strict
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()
        if imu_bus is not None:
            threading.Thread(target=self._local_imu, args=(imu_bus,), daemon=True).start()

    def _local_imu(self, bus):
        """Onboard brain (A7Z): read the BNO085 on this host's I2C bus and fill self.imu with the same tuple a J12
        frame carries (pi_imu_uart.py), so the control loop cannot tell the two paths apart."""
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pi"))
        from pi_imu_uart import frame_age_s
        from imu_reader import ThreadedIMUReader
        reader = ThreadedIMUReader(i2c_bus=bus, frequency_hz=100)
        reader.start()
        seq, last_ts = 0, None
        while True:
            snap = reader.get_latest()
            if snap.timestamp_s != last_ts:
                last_ts = snap.timestamp_s
                now = time.perf_counter()
                f = (seq, *snap.quat, *snap.gyro, *snap.acc, frame_age_s(snap, now), 1 if snap.valid else 0)
                seq += 1
                if imu_frame_ok(f):
                    with self.lock:
                        self.imu = (now, f)
                        self.imu_count += 1
                else:
                    self.imu_bad += 1
            time.sleep(0.002)

    def _reader(self):
        buf = b""
        while True:
            # read what is waiting, or block for the next byte: read(4096) waited out its whole 2 ms timeout on every
            # call and added ~2 ms to each servo read (6.7 vs 4.7 ms median, 2026-10-15)
            buf += self.ser.read(max(1, self.ser.in_waiting))
            while True:
                if len(buf) < 3:
                    break
                if buf[0] == 0xFF and buf[1:4] == b"\xff\xfd\x00" and len(buf) >= 7:
                    total = 7 + struct.unpack_from("<H", buf, 5)[0]
                    if len(buf) < total:
                        break
                    self.status.append(buf[:total])
                    buf = buf[total:]
                elif buf[0] == 0xA5 and buf[1] == 0x5A:
                    total = 3 + buf[2] + 1
                    if len(buf) < total:
                        break
                    payload = buf[3:total - 1]
                    if sum(payload) & 0xFF == buf[total - 1] and len(payload) == struct.calcsize(IMU_FMT):
                        f = struct.unpack(IMU_FMT, payload)
                        if imu_frame_ok(f):  # the 8-bit sum lets ~1/256 corrupted frames through (2e76 gravity, 10/15)
                            with self.lock:
                                self.imu = (time.perf_counter(), f)
                                self.imu_count += 1
                        else:
                            self.imu_bad += 1
                    buf = buf[total:]
                elif buf[0] == 0xFF and len(buf) < 7 and buf[1:4] == b"\xff\xfd\x00"[:len(buf) - 1]:
                    break
                else:
                    buf = buf[1:]  # resync

    def sync_write(self, addr, size, values):
        params = struct.pack("<HH", addr, size)
        for i, v in values.items():
            params += bytes([i]) + int(v).to_bytes(size, "little", signed=size == 4)
        self.ser.write(_packet(0xFE, 0x83, params))

    def sync_read(self, addr, size, ids, timeout=0.01, allow_alert=None):
        """allow_alert: accept status packets whose only error is the Alert bit; None follows self.alert_ok."""
        alert_ok = self.alert_ok if allow_alert is None else allow_alert
        self.late += len(self.status)  # replies that arrived after the previous read gave up (timing, not loss)
        self.status.clear()
        self.ser.write(_packet(0xFE, 0x82, struct.pack("<HH", addr, size) + bytes(ids)))
        out, end = {}, time.perf_counter() + timeout
        while len(out) < len(ids) and time.perf_counter() < end:
            if not self.status:
                time.sleep(0.0002)
                continue
            pkt = self.status.popleft()
            # On the bench any error, Alert included, drops the packet, so a faulted servo leaves the state stale and
            # the policy holds. On the 2S pack every servo sets Alert above 7.0 V (battery.py): once the guard is
            # armed Alert alone passes, and the loop reads Hardware Error Status once a second instead.
            if pkt[7] == 0x55 and len(pkt) == 11 + size and (pkt[8] == 0 or (alert_ok and status_ok(pkt[8]))):
                out[pkt[4]] = pkt[9:9 + size]
        return out


# Head touch pads (D-013, guide/extension-20261015-touch): a XIAO ESP32-S3 scans the five pads every 20 ms
# (firmware/touch_xiao/touch_main.cpp) and answers an I2C read at 0x2A with the newest scan, 24 bytes: uint32 LE
# sequence number, then the raw counts of L1, R1, L2, R2, M (a touch raises them). On the A7Z it sits on the kernel bus
# of TWI0, header pins 29/31 (CURRENT-STATE.json a7z_touch_bus). The Mac brain has no path to it.
TOUCH_ADDR, TOUCH_BLOCK, TOUCH_PERIOD = 0x2A, 24, 0.02
TOUCH_PADS = ("L1", "R1", "L2", "R2", "M")
TOUCH_SATURATED = 0x3FFFFF  # touchRead's ceiling, a pad lead on metal; a count above it means a misaligned block
I2C_SLAVE = 0x0703
I2C_SYSFS = "/sys/bus/i2c/devices"
_wall = time.monotonic  # the touch reader's pacing clock; the tests replace this module's `time` with the loop's


def touch_bus_path(arg):
    """--touch-bus: 'auto' is the kernel I2C bus of TWI0 (twi@2510000) or None, 'none' is None, N is /dev/i2c-N, anything
    else a device path. No bit-banged fallback here: it costs tens of ms of CPU per read (touch_live_a7z.py)."""
    if arg == "none":
        return None
    if arg == "auto":
        return next(("/dev/" + n.split("/")[-2] for n in sorted(glob.glob(os.path.join(I2C_SYSFS, "i2c-*", "of_node")))
                     if os.path.realpath(n).endswith("/twi@2510000")), None)
    return f"/dev/i2c-{arg}" if arg.isdigit() else arg


def open_touch_bus(path):
    """A function that reads one block from the XIAO."""
    fd = os.open(path, os.O_RDWR)
    fcntl.ioctl(fd, I2C_SLAVE, TOUCH_ADDR)
    return lambda: os.read(fd, TOUCH_BLOCK)


class TouchPads:
    """Reads the pads on its own thread every TOUCH_PERIOD; the control loop only takes the newest block (latest()),
    which never waits on the bus. A failed read, or a block no pad can produce, keeps the previous block, whose age then
    grows; so does a scan that has stopped (the sequence number no longer advances: a saturated pad holds the scan of
    touch_main.cpp on that pad, touch_live_a7z.py)."""

    def __init__(self, read, start=True):
        self.read = read
        self.newest = None  # (seq, counts, perf_counter time the seq arrived), replaced whole so a reader needs no lock
        self.bad = 0  # reads that failed or were rejected
        self.failing = False
        if start:
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        pause = threading.Event()  # a wall-clock wait, never the loop's (simulated) clock
        due = _wall()
        while True:
            self.poll()
            due = max(due + TOUCH_PERIOD, _wall())  # the read time comes out of the pause; a stuck read skips, no burst
            pause.wait(due - _wall())

    def poll(self):
        """One read; True when it gave a valid block."""
        try:
            raw = self.read()
            why = None if len(raw) == TOUCH_BLOCK else f"{len(raw)}-byte read"
        except OSError as e:
            why = f"{type(e).__name__}: {e}"
        if why is None:
            seq, *counts = struct.unpack("<6I", raw)
            if max(counts) > TOUCH_SATURATED:
                why = "a count above the pad ceiling, misaligned block"
        if why:
            self.bad += 1
            if not self.failing:
                print(f"touch: read failed ({why}); the pads keep their last block, retrying", flush=True)
                self.failing = True
            return False
        if self.failing:
            print("touch: reads back", flush=True)
            self.failing = False
        if self.newest is None or seq != self.newest[0]:
            self.newest = (seq, counts, time.perf_counter())
        return True

    def latest(self, now):
        """(counts L1 R1 L2 R2 M, seq, s since that block arrived) of the newest block; all None before the first."""
        b = self.newest
        return (None, None, None) if b is None else (b[1], b[0], now - b[2])


# Stop guards for walk mode: either one switches to plain stand and writes that to the control file.
# Simulation actions stay within +-0.9 rad (scratch simact.py, 2026-10-15); the 10-15 divergence reached +-5.
MAX_ACTION = 1.0  # slow walking needs no more; the 10-15 divergence crossed 1.0 within 1 s of its onset
# the stop of a gait: walk61 at zero command peaks at 0.74 rad on the as-built model (0.25 m/s, 12 seeds, friction 0.4
# and 1.0); the former swap to stand61 peaked at 1.64 in sim and at 1.59 and 1.93 on the real robot (2026-10-15)
WALK_SETTLE_S = 1.0


def action_limit(stepping, since_step_s, max_action, max_walk_action):
    """The walking limit while stepping and for WALK_SETTLE_S after; the standing limit otherwise."""
    return max_walk_action if stepping or since_step_s < WALK_SETTLE_S else max_action


# A catch on a slippery floor may pass the limit for a moment; a fall stays over it. As-built sim at friction 0.2-0.25
# (16 seeds, 2026-10-15): every recovered run had at most 1 tick over 1.8 rad (peak 1.85), every fall 9-29 consecutive
# ticks over and 3-8 rad. The real robot tripped on a single 1.86 tick and then fell from static stand. Shawn agreed to
# loosen the limit if the stop needs it ("如果收步动作确实需要超过1.8 那就放宽").
OVER_LIMIT_TICKS = 3   # consecutive ticks over the limit before the stop
HARD_WINDOW = 10  # ticks (0.2 s); a runaway alternates over and under the hard limit
HARD_MAX_ACTION = 3.0  # one tick over this stops at once (the 10-15 divergence reached +-5)


def over_limit(amax, over, limit, ticks=OVER_LIMIT_TICKS, hard=HARD_MAX_ACTION):
    """(consecutive ticks over the limit, stop reason or None) after an action of size amax."""
    over = over + 1 if amax > limit else 0
    if amax > hard:
        return over, f"policy action {amax:.2f} rad > {hard}"
    if over >= ticks:
        return over, f"policy action over {limit} rad for {over} ticks (last {amax:.2f})"
    return over, None
# Stepping needs more: the simulated walk61 gait at vx 0.25-0.3 peaks at 1.1-1.23 rad (2026-10-15), and the real
# divergence passed 1.4 within 0.1 s of 1.29.
MAX_WALK_ACTION = 1.8  # Shawn 2026-10-15: the as-built simulation peaks at 1.64 rad in a normal 0.25 m/s gait
MAX_TILT_DEG = 30.0
MAX_TILT_GP_DEG = 55.0  # the ground-pick crouch pitches the trunk 35 deg in the as-built sim (2026-10-16)
MAX_TILT_SIT_DEG = 30.0  # sit and stand-up peak at 10.3-11.4 deg in the plugged as-built sim at 7.0 V (2026-10-20)
QUACK_JAW_DEG = 15.0  # jaw opening for a quack; jaw_limits still caps it by the head pose
QUACK_S = 0.4
# After a quack a jaw that was released or resting shut is pressed to QUACK_SHUT_DEG (the 100 mA current limit sets the
# force) for QUACK_CLOSE_S, then eased to QUACK_HOLD_DEG and held with torque on. On 2026-10-23 (closed by hand it read
# the calibrated closed tick 2042): at KP 200 a 2 deg rest goal settled at 2.8 deg, a 0 deg goal at 1.6 deg, both
# visibly open; pressing and releasing sprang back to 3.3 deg (the bills flex); a 0 deg hold after the press sat at
# -0.5 deg, still open at the tip; -10 then -4 held at -1.7 deg on 22 mA, closed.
QUACK_CLOSE_S = 0.3
QUACK_SHUT_DEG = -10.0
QUACK_HOLD_DEG = -4.0
# Random head move per quack on stand61, as deltas from the policy default, model rad:
# one of the QUACK_HEAD styles, each joint's magnitude drawn from its range, with a random sign where the range is
# unsigned (a pair) and the given sign where it is (lo, hi, sign); head_pitch negative looks up (pickup.py looks down
# with +0.8). All well inside the trained +-1.10 / +-1.40 / +-0.31 (walk_brain.HEAD_RANGE). The move spans the call:
# the calls are 0.12-0.40 s long (48 simulator chirps, median 0.20 s), so the target follows a raised cosine up over
# QUACK_HEAD_RAMP_S, holds, and is back QUACK_HEAD_S after the quack; the Brain's head EMA (0.2 per tick, Pollen's
# runtime value) adds about 0.1 s. The policies were trained on head commands that step to a new value every 2-5 s
# (microduck_velocity_env_cfg.py, UniformPoseCommand holds between resamples), so one move per quack is inside the
# trained command distribution; the ramp stays for the seated case, where Pollen saw abrupt head moves tip the real
# duck backward (microduck_sitstand_env_cfg.py). Plugged as-built sim at 7.0 V and 4 % dropped reads (2026-10-21,
# ~/code/reports/20261021-microduck-quack-head-balance.html): this profile, the earlier 1.6 s / 0.5 s ramp and a
# plain 0.5 s step all keep the trunk tilt within 0.12 deg of rest on stand61, walk61 at rest and seated, no fall;
# the head passes half the commanded tilt 0.26-0.38 s after the quack (0.48-0.58 s with the 1.6 s profile). A quack
# inside a running move restarts it: the head dips 0.04-0.06 rad toward the start and rises again, no fall.
QUACK_HEAD = ({"head_roll": (0.08, 0.18), "head_yaw": (0.0, 0.25)},       # tilt
              {"head_pitch": (0.15, 0.30, -1), "head_roll": (0.0, 0.10)})  # look up
QUACK_HEAD_S = 0.6
QUACK_HEAD_RAMP_S = 0.15
QUACK_REST_S = 2.0  # walk61 after stepping: a zero command held this long has brought the stride to rest
# Seated (sitstand61): share of the move. Pollen trained the seated head on small slow commands and saw fast head
# jerks tip a seated duck backward; this move is ramped. Plugged as-built sim at 7.0 V and 4 % dropped reads, 3 seeds,
# all three styles at full size with the 1.6 s profile: head reaches 0.22-0.27 rad, seated tilt moves at most 1.7 deg,
# no fall; the 0.6 s profile moves the seated tilt 0.06 deg (2026-10-21).
QUACK_SIT_SCALE = 1.0


def quack_head_share(t):
    """0 -> 1 -> 0 over QUACK_HEAD_S: raised-cosine ramps of QUACK_HEAD_RAMP_S at each end, 1 in between."""
    edge = min(t, QUACK_HEAD_S - t) / QUACK_HEAD_RAMP_S
    return 0.0 if edge <= 0 else 1.0 if edge >= 1 else 0.5 - 0.5 * math.cos(math.pi * edge)


def quack_head_block(want, brain, zero_s):
    """Why a quack may not move the head on this tick, or None when it may. zero_s: s the velocity command (wanted and
    smoothed) has been all zero, None while it is not.

    Walk mode at zero command: stand61, and walk61 once its stride has come to rest (QUACK_REST_S after the stop).
    Seated: the sitstand policy, at QUACK_SIT_SCALE of the move. Never while stepping, sitting down or standing up,
    and never in the open-loop stand mode, which cannot rebalance (it tipped over there on 2026-10-20)."""
    if "head" in want:
        return "a head command is active"
    if brain.mode == "walk" and want.get("mode") == "walk":
        if zero_s is None:
            return "walking"
        if brain.stepped and zero_s < QUACK_REST_S:
            return "the stride is still coming to rest"
        return None
    if brain.mode == "sit" and want.get("mode") == "sit":
        return None if brain.sit is not None and brain.sit["phase"] == "seated" else "sitting down"
    if brain.mode == "sit":
        return "standing up"
    if brain.mode == "stand":
        return "open-loop stand mode cannot rebalance"
    return f"{brain.mode} mode"


def quack_head_pose(delta, t, brain):
    """Absolute head targets (model rad) t s into a quack head move of `delta` (QUACK_HEAD deltas)."""
    scale = QUACK_SIT_SCALE if brain.mode == "sit" else 1.0
    share = quack_head_share(t)
    return {n: brain.default[n] + share * scale * d for n, d in delta.items()}
GP_DONE_PHASE = 0.9  # inside the policy's standing rest segment: the rise is complete, hand back to stand61
CMD_ALPHA = 0.2  # velocity command EMA per 20 ms tick, Pollen's robotd default
# No IMU sample this long: the Brain holds its last goals (blind, so no policy step). 2026-10-15 the Pi Zero lost power
# 0.85 s into a walk, the stream stopped, the robot froze mid-stride and fell with nothing on the console.
IMU_LOST_S = 0.25
HOLD_TILT_E = 0.3  # grip engagement over which the tilt stop stays off while the hold is applied
HOLD_FILE_CHECK_S = 0.5
# A P-gain packet is never acknowledged (sync write): while the hold has changed the gains within the last
# HOLD_REFRESH_WINDOW_S, the whole gain set is written again every HOLD_REFRESH_S, so a lost restore cannot leave a
# leg at P 80 under a robot that walks again.
HOLD_REFRESH_S = 1.0
HOLD_REFRESH_WINDOW_S = 5.0


def p_gain_writes(kp, kp_now, settled=False, step=KP_STEP):
    """{servo ID: gain} to write for the Brain's kp (one int for every joint, or {joint name: int} from the hold) given
    the gains last written (kp_now, {joint name: int} or None). A whole-robot gain is written exactly; per-joint hold
    gains when they moved `step`, reached KP_RL, or stopped changing (settled: the same wish as the tick before), so a
    slow soften costs a few writes and its end value is always reached."""
    if not kp:
        return {}
    want = kp if isinstance(kp, dict) else {nm: kp for nm in NAMES}
    if kp_now is None:
        return {i: want[nm] for i, nm in zip(IDS, NAMES)}
    tol = step if isinstance(kp, dict) and not settled else 1
    return {i: want[nm] for i, nm in zip(IDS, NAMES)
            if want[nm] != kp_now[nm] and (abs(want[nm] - kp_now[nm]) >= tol or want[nm] == walk_brain.KP_RL)}


def hold_allowed(want, brain, zero_s, grip_stopped=False):
    """The hold acts on stand61, or walk61 once its stride has come to rest, never on a gait, a kick, a sit (also not
    while a sit stands up under a walk command), a crouch or the open-loop stand. zero_s as in quack_head_block.
    A walk stopped by a grip on the head (GripStop) holds at once: the hand is already on the robot, and a lift
    mid-stride must not leave walk61 stepping in the air while the command ramps down and the stride settles."""
    if want.get("mode") != "walk" or brain.mode != "walk":
        return False
    if grip_stopped:
        return True
    if zero_s is None:
        return False
    return not brain.stepped or zero_s >= QUACK_REST_S


GRIP_STOP_E = 0.3  # grip engagement that stops a walk (as HOLD_TILT_E: a palm on the head, not a brushing finger)
GRIP_STOP_HELD_S = 0.05  # applied authority at or below which a grip-stopped robot counts as fully held


class GripStop:
    """A hand gripping the head stops a walk (Shawn 2026-10-23: a held head means stop). While the hold is on and
    calibrated, a walk command meeting grip engagement e > GRIP_STOP_E is latched: its velocities read as zero, so the
    usual command ramp brings walk61 to a stop and hold_allowed lets the hold act at once. The latch stays through a
    zero command (the stick released, a for_s expiry, a script's stop) and after the hand leaves, so a lifted robot
    stays held and a robot set down stands where it is. It clears on a walk command that moves again after the
    command has been zero, with no hand on the head (release the stick, push it again), or on a non-walk command.
    2026-10-24: a latch that cleared on any zero command dropped the hold 0.24 s after the grip, and walk61 kicked in
    the air for 2.5 s while the robot was lifted (shoot.py take 2)."""

    def __init__(self):
        self.on, self.released = False, False

    def update(self, want, ready, e):
        """want: the control file's command; ready: hold on and calibrated. Returns 'stop' on the tick it latches,
        'clear' on the tick it releases, else None."""
        walk = want.get("mode") == "walk"
        moving = walk and any(float(want.get(k, 0)) for k in ("vx", "vy", "vtheta"))
        if self.on:
            if not walk:
                self.on = False
                return "clear"
            if not moving:
                self.released = True
                return None
            if self.released and e <= GRIP_STOP_E:
                self.on = False
                return "clear"
            return None
        if moving and ready and e > GRIP_STOP_E:
            self.on, self.released = True, False
            return "stop"
        return None

    def apply(self, want):
        return dict(want, vx=0.0, vy=0.0, vtheta=0.0) if self.on else want

    def settle(self, brain, s):
        """Once the hold has the whole authority after a grip stop (applied s <= GRIP_STOP_HELD_S), the legs are at the
        default stance, not mid-stride: walk61's stepped latch is cleared so the set-down runs stand61. walk61 at zero
        command set down 14-22 deg tilted from a pivoting hand fell in 2 of 4 seeds (as-built sim, 2026-10-23); the swap
        happens while the actions are scaled to nothing, so stand61 starts from the stance and the applied (zero) last
        action, unlike the mid-stride swap that threw 1.93 rad on 2026-10-15. True on the tick it clears."""
        if self.on and s <= GRIP_STOP_HELD_S and brain.mode == "walk" and brain.stepped:
            brain.stepped = False
            return True
        return False


def kp_for_log(kp_now):
    """One number when every joint has the same gain (as before the hold), else {joint name: gain}."""
    if kp_now is None:
        return None
    vals = set(kp_now.values())
    return vals.pop() if len(vals) == 1 else dict(kp_now)


def want_s(cmd):
    return cmd[1].get("for_s")


def write_cmd(path, want, why):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(want, f)
    os.replace(tmp, path)
    what = {"walk": "zero command", "stand": "stand", "monitor": "hold the last goals"}.get(want.get("mode"), want.get("mode"))
    print(f"STOP -> {what}: {why}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="default: the only /dev/cu.usbmodem* (Mac) or /dev/ttyACM* (A7Z)")
    ap.add_argument("--baud", type=int, default=1_000_000, help="500000 for the A7Z J12 port /dev/ttyAS2")
    ap.add_argument("--imu-bus", type=int, default=None,
                    help="onboard brain: read the BNO085 on this I2C bus (A7Z: 7) instead of J12 frames from the OpenRB")
    ap.add_argument("--models", default=os.path.join(WHOLE, "runtime-overlay", "models"))
    # Off by default: wired sample ages are a few ms, where the V1.9 simulation passes 9/9 without it, and the
    # predict-to-apply variant hurt on the replayed Wi-Fi link (dynamics/v19-20261014).
    ap.add_argument("--compensation", action="store_true")
    ap.add_argument("--cmd", default="/tmp/duck_cmd.json")
    ap.add_argument("--yaw-trim-file", default=os.path.expanduser("~/.duck_yaw_trim.json"),
                    help="walk_brain's learned straight-walk yaw trim, read at start and updated after each forward walk")
    ap.add_argument("--log", default="/tmp/duck_wired_log.jsonl")
    ap.add_argument("--supply", choices=("auto", "bench", "battery"), default="auto",
                    help="battery guard: auto arms at 7.2 V (2S pack), battery from the start, bench off")
    ap.add_argument("--battery-marker", default=STOP_MARKER)
    ap.add_argument("--max-action", type=float, default=MAX_ACTION, help="walk stops into stand above this |action|, rad")
    ap.add_argument("--max-walk-action", type=float, default=MAX_WALK_ACTION,
                    help="the same limit while a walking speed is commanded, rad")
    ap.add_argument("--hard-max-action", type=float, default=HARD_MAX_ACTION,
                    help="one action over this stops at once; the limits above need OVER_LIMIT_TICKS ticks in a row")
    ap.add_argument("--max-tilt", type=float, default=MAX_TILT_DEG, help="walk stops into stand above this tilt, deg")
    ap.add_argument("--jaw-kp", type=int, default=jaw_ctl.KP_JAW, help="ID15 position P gain in mode 5")
    ap.add_argument("--max-tilt-gp", type=float, default=MAX_TILT_GP_DEG,
                    help="ground pick holds its last goals above this trunk tilt, deg (the crouch itself is 35 deg in the as-built sim)")
    ap.add_argument("--max-tilt-sit", type=float, default=MAX_TILT_SIT_DEG,
                    help="sit mode holds its last goals above this trunk tilt, deg")
    ap.add_argument("--sounds", default=os.path.join(HERE, "sounds"),
                    help="folder of 16-bit WAV duck calls for the quack key (played with aplay -q, ALSA default device)")
    ap.add_argument("--breakaway", action="store_true",
                    help="Mac-side stiction compensation on the leg joints in stand mode (walk_brain.Breakaway, validated standing 2026-10-16)")
    ap.add_argument("--breakaway-walk", action="store_true",
                    help="also apply it in walk mode (2026-10-16: stand61 at kp 200 diverged in 0.4 s with it; off unless you are testing that)")
    ap.add_argument("--kp-rl", type=int, default=None,
                    help="firmware P gain in walk mode (default walk_brain.KP_RL 200; E1 on 10/15: loaded joints ignore <0.05 rad below ~800)")
    ap.add_argument("--kp-stand", type=int, default=None,
                    help="firmware P gain in stand mode (default constants.KP_DEFAULT 400; Shawn 2026-10-16: 200, as walk mode)")
    ap.add_argument("--touch-bus", default="auto",
                    help="head touch pads (XIAO at I2C 0x2A): auto = the kernel bus of the A7Z's TWI0 when it exists, "
                         "N = /dev/i2c-N, a device path, or none")
    ap.add_argument("--hold-file", default=HOLD_FILE,
                    help='graded hold switch and calibration, {"on": bool, "cal": {...}} (hold.py); missing = off, '
                         "the estimate is still logged")
    args = ap.parse_args()
    if args.kp_rl:
        import walk_brain as _wb
        _wb.KP_RL = args.kp_rl
    if args.kp_stand:
        import walk_brain as _wb
        _wb.KP_DEFAULT = args.kp_stand
    port = args.port or (glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyACM*") or [None])[0]
    if port is None:
        sys.exit("no OpenRB USB port")

    brain = Brain(args.models, compensation=args.compensation, yaw_trim_file=args.yaw_trim_file)
    print(f"yaw trim {brain.yaw_trim:+.3f} rad/s from {args.yaw_trim_file}", flush=True)
    guard = None if args.supply == "bench" else BatteryGuard(args.supply)
    hw_lock = take_hw_lock()  # noqa: F841  (held until exit)
    link = Link(port, args.baud, args.imu_bus)
    # The bench rail (7.1-7.3 V) is above the servos' 7.0 V Max Voltage Limit, and with Shutdown 52 (D-014) every status
    # packet carries the Alert bit. The bench accepts those packets and watches Hardware Error Status once a second
    # like the armed guard, so a fault other than the input-voltage flag still freezes the policy.
    link.alert_ok = guard is None or guard.armed
    log = open(args.log, "a")

    def _sync_log():  # a brownout loses only what is not on the card: fsync every 50 ms, off the control thread
        pause = threading.Event()  # a wall-clock wait, never the loop's (simulated) clock
        while True:
            pause.wait(0.05)
            try:
                os.fsync(log.fileno())
            except (OSError, ValueError):
                pass
    threading.Thread(target=_sync_log, daemon=True).start()
    print(f"wired on {port}, compensation {brain.comp_cfg['delay_compensation']}", flush=True)
    touch, touch_path = None, touch_bus_path(args.touch_bus)
    if touch_path is None:
        print("touch: off (--touch-bus none)" if args.touch_bus == "none" else
              "touch: no TWI0 I2C bus (twi@2510000); the head pads are not read, the touch fields stay null", flush=True)
    else:
        try:
            read_touch = open_touch_bus(touch_path)
        except OSError as e:
            print(f"touch: cannot open {touch_path} ({e}); the head pads are not read, the touch fields stay null",
                  flush=True)
        else:
            print(f"touch: pads {'/'.join(TOUCH_PADS)} on {touch_path} (XIAO 0x{TOUCH_ADDR:02X}), read every "
                  f"{TOUCH_PERIOD * 1000:.0f} ms off the control thread", flush=True)
            touch = TouchPads(read_touch)
    hold_cfg = read_hold_file(args.hold_file)
    est = HoldEstimator(hold_cfg["cal"], kp_rl=walk_brain.KP_RL)
    hold_mtime, next_hold_check, hold_t = None, 0.0, None
    hold_s = 1.0  # the authority actually handed to the Brain: ramps at UP_RATE / DOWN_RATE across on/off edges too
    hold_kp_prev, hold_gain_t, next_refresh = None, float("-inf"), 0.0
    print(f"hold: {'ON' if hold_cfg['on'] and est.legs_range else 'off (estimate logged only)'} ({args.hold_file}), "
          f"{'calibrated' if est.legs_range else 'leg load not calibrated: it stays off until hold_session.py calibrates'}",
          flush=True)

    pos, vel = [0] * 14, [0] * 14
    cur, jaw_cur = [None] * 14, None  # mA; a servo that has not answered yet has none
    comp = Breakaway()
    jaw = JawCtl(kp=args.jaw_kp)
    print("jaw ID15: " + (f"calibrated, closed tick {jaw.cal['closed_tick']}, open tick {jaw.cal['open_tick']}"
                         if jaw.calibrated else "not calibrated (jaw_torque false, close by hand, jaw_mark closed, "
                         "open by hand, jaw_mark open)"), flush=True)
    jaw_warned = False
    voice = Voice(args.sounds)
    print(f"quack: {len(voice.files)} calls in {args.sounds}" if voice.files else
          f"quack: no .wav calls in {args.sounds}, the quack key moves only the jaw", flush=True)
    last_quack = 0  # "quack" counter value last acted on; a new value quacks once
    quack_until, quack_prev, quack_hint, quack_close_until = None, None, False, None
    quack_head, quack_head_t0 = None, None
    zero_t = None  # perf_counter time since which the walk velocity command has been all zero
    grip_stop = GripStop()
    rng = np.random.default_rng()
    torque_on, kp_now, last_cmd = False, None, None
    write_delay = 0.002  # measured below: read-midpoint to goal write
    last_print, n, misses, t_stat = 0.0, 0, 0, time.perf_counter()
    miss_by_id = {}  # servo ID -> sync reads it did not answer (since start)
    cmd_since = time.time()
    last_goal = None  # the goals most recently written to the servos
    goal_written = None  # tick goals of the newest goal write, every mode (log only)
    last_step_t = float("-inf")  # last loop with a nonzero walk command
    diverging = None  # set when the policy output or the body tilt leaves the trained range
    imu_lost = False
    over = 0  # policy ticks over the hard limit within the last HARD_WINDOW ticks
    hard_hits = collections.deque(maxlen=HARD_WINDOW)
    cmd_s = [0.0, 0.0, 0.0]  # EMA-smoothed (vx, vy, vtheta) handed to the Brain
    next_t = time.perf_counter()
    hw_faults, next_hw, last_block = [], time.perf_counter() + 0.5, float("-inf")
    while True:
        t0 = time.perf_counter()
        got = link.sync_read(ADDR_CUR, STATE_SIZE, IDS + [JAW_ID], timeout=0.018)  # USB adds a few ms; median 6-11 ms measured
        t_read = time.perf_counter()
        t_mid = (t0 + t_read) / 2
        read_ms = (t_read - t0) * 1000
        for k, i in enumerate(IDS):
            if i in got:
                cur[k], vel[k], pos[k] = struct.unpack(STATE_FMT, got[i])
        if JAW_ID in got:  # the jaw rides along; a missing jaw never stops the body
            jaw_cur, _, jaw.last_tick = struct.unpack(STATE_FMT, got[JAW_ID])
        fresh = all(i in got for i in IDS) and not hw_faults  # a faulted servo freezes the policy, as on the bench
        misses += not fresh
        missing = [i for i in IDS if i not in got]  # a daisy-chain break drops every ID past it together
        for i in missing:
            miss_by_id[i] = miss_by_id.get(i, 0) + 1
        with link.lock:
            imu = link.imu
        touch_counts, touch_seq, touch_age = touch.latest(time.perf_counter()) if touch else (None, None, None)
        t_hold = time.perf_counter()
        if t_hold >= next_hold_check:
            next_hold_check = t_hold + HOLD_FILE_CHECK_S
            try:
                mt = os.stat(args.hold_file).st_mtime
            except OSError:
                mt = None
            if mt != hold_mtime:
                hold_mtime, was = mt, hold_cfg["on"]
                hold_cfg = read_hold_file(args.hold_file)
                if hold_cfg["error"]:
                    print(f"hold: {hold_cfg['error']}", flush=True)
                est.set_cal(hold_cfg["cal"])
                if hold_cfg["on"] != was or mt is not None:
                    print(f"hold: {'ON' if hold_cfg['on'] and est.legs_range else 'off'}, "
                          f"{'leg load calibrated' if est.legs_range else 'leg load not calibrated, so it stays off'}",
                          flush=True)
        legs_ma = legs_current_ma(cur, NAMES)
        hold_dt = 0.02 if hold_t is None else min(0.1, t_hold - hold_t)
        est.update(hold_dt, touch_counts, touch_age, legs_ma)
        hold_t = t_hold
        new_cmd = read_cmd(args.cmd, last_cmd)
        cmd_is_new = new_cmd is not last_cmd
        if cmd_is_new:
            cmd_since = time.time()  # when this command arrived (the file's mtime is wall clock, not this loop's)
        last_cmd = new_cmd
        want = last_cmd[1]
        gs = grip_stop.update(want, hold_cfg["on"] and est.legs_range is not None, est.e)
        if gs:
            print("hold: head gripped while walking, stopped (walking again needs the stick released and pushed again)"
                  if gs == "stop" else "hold: grip stop cleared by a new push without a hand on the head, or a non-walk command", flush=True)
        want = grip_stop.apply(want)
        if want.get("mode") == "off":
            link.sync_write(ADDR_TORQUE, 1, {i: 0 for i in IDS})
            if jaw.on:
                jaw.release(link)
            hw = link.sync_read(ADDR_HW_ERROR, 1, IDS, timeout=0.03, allow_alert=True)
            latched = [i for i in IDS if i in hw and hw[i][0] & ~INPUT_VOLTAGE_BIT]
            for i in latched:
                link.ser.write(_packet(i, 0x08, b""))  # Reboot: clears Hardware Error Status, torque stays off
                time.sleep(0.02)
            if latched:
                time.sleep(0.5)  # the servos boot (their RAM settings reset; the P gain is rewritten on torque on)
            link.status.clear()
            torque_on, kp_now, hw_faults = False, None, []
            want = {"mode": "monitor"}
            tmp = args.cmd + ".tmp"
            with open(tmp, "w") as f:
                json.dump(want, f)
            os.replace(tmp, args.cmd)
            print("torque off" + (f", rebooted {[NAMES[IDS.index(i)] for i in latched]} to clear their hardware errors"
                                  if latched else ""), flush=True)
        if cmd_is_new and want.get("quack") is not None and want.get("quack") != last_quack:
            last_quack = want.get("quack")
            call = voice.quack()
            if jaw.calibrated and jaw.goal_deg != "grip":
                if quack_until is None:
                    # None: the jaw was released, resting shut or still being pressed shut, and is shut again after
                    held_open = (jaw.on and quack_close_until is None and isinstance(jaw.goal_deg, (int, float))
                                 and jaw.goal_deg > 0)
                    quack_prev = jaw.goal_deg if held_open else None
                jaw.goal_deg, quack_until = QUACK_JAW_DEG, time.perf_counter() + QUACK_S  # enables the torque if off
            elif not quack_hint:
                print("quack: jaw stays put (" + ("gripping" if jaw.goal_deg == "grip" else "not calibrated") + ")",
                      flush=True)
                quack_hint = True
            why = quack_head_block(want, brain, None if zero_t is None else time.perf_counter() - zero_t)
            if why is None:
                style = QUACK_HEAD[rng.integers(len(QUACK_HEAD))]
                quack_head = {n: float((r[2] if len(r) == 3 else rng.choice((-1, 1))) * rng.uniform(r[0], r[1]))
                              for n, r in style.items()}
                quack_head_t0 = time.perf_counter()
            else:
                quack_head = quack_head_t0 = None  # starting it later would jump in mid-profile
            print(f"quack {last_quack}: {os.path.basename(call) if call else 'no sound'}, "
                  + (f"head {', '.join(f'{n} {d:+.2f}' for n, d in quack_head.items())}"
                     + (f" x{QUACK_SIT_SCALE} seated" if brain.mode == "sit" else "") if why is None
                     else f"head stays ({why})"), flush=True)
        if quack_until is not None and (time.perf_counter() >= quack_until or (not jaw.on and jaw.goal_deg is None)):
            if jaw.on and quack_prev is not None:
                jaw.goal_deg = quack_prev  # the jaw was held before the quack: back to that opening
            elif jaw.on:  # press shut, then ease to the hold (a 2 deg goal stopped at 2.8, beak open)
                jaw.goal_deg, quack_close_until = QUACK_SHUT_DEG, time.perf_counter() + QUACK_CLOSE_S
            quack_until = None
        if quack_close_until is not None and (time.perf_counter() >= quack_close_until or quack_until is not None
                                              or not jaw.on or jaw.goal_deg != QUACK_SHUT_DEG):
            if quack_until is None and jaw.on and jaw.goal_deg == QUACK_SHUT_DEG:
                jaw.goal_deg = QUACK_HOLD_DEG  # eased off but still past closed: the tip stays shut, low current
            quack_close_until = None
        if cmd_is_new:  # jaw keys act once per command; the jaw goal persists until the next jaw key
            if want.get("jaw_mark") in ("closed", "open"):
                try:
                    cal = jaw.mark(want["jaw_mark"], jaw.last_tick)
                    print(f"jaw {want['jaw_mark']} marked at tick {jaw.last_tick}: {cal}"
                          + ("" if jaw.calibrated else " (mark the other end too)"), flush=True)
                except (RuntimeError, ValueError) as e:
                    print(f"jaw mark failed: {e}", flush=True)
            if want.get("jaw_torque") is False and jaw.on:
                jaw.release(link)
                print("jaw torque off", flush=True)
            if "jaw" in want:
                quack_until = None  # an explicit jaw command wins over a quack in progress
                j = want["jaw"]
                if j in ("grip", "open") or isinstance(j, (int, float)):
                    jaw.goal_deg = j
                else:
                    print(f"jaw command ignored: {j!r}", flush=True)
        # a walk command carries "for_s": once that long has passed since it was written, the command goes to zero and
        # walk61 brings its stride to rest (closed loop, torque on), as in the simulation; an open-loop ramp to STAND
        # out of a gait can tip the robot. Written back so the stop holds until a new command.
        if want.get("mode") == "walk" and want.get("for_s") is not None \
                and time.time() - cmd_since > float(want["for_s"]):
            want = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            if last_cmd[1].get("head"):
                want["head"] = last_cmd[1]["head"]  # the head keeps looking where the walk command aimed it
            write_cmd(args.cmd, want, f"walk for_s {want_s(last_cmd)} elapsed")
            last_cmd, cmd_since = read_cmd(args.cmd, None), time.time()
        if brain.mode in ("walk", "kick") and diverging:
            if brain.mode == "kick":
                brain.end_kick(diverging)  # the stand ramp next tick starts from the last goals sent
            want = {"mode": "stand"}
            hard_hits.clear()  # a later walk starts with a clean window
            write_cmd(args.cmd, want, diverging)
            last_cmd, diverging, cmd_since = read_cmd(args.cmd, None), None, time.time()
        elif brain.mode in ("groundpick", "sit") and diverging:
            # no open-loop ramp to STAND out of a crouch or a seat: the legs hold the last goals (torque on) and wait
            # for a command; the head holds where it is instead of a runaway goal (2026-10-23 head pressed on the floor)
            brain.hold_head()
            want = {"mode": "monitor"}
            hard_hits.clear()
            write_cmd(args.cmd, want, ("ground pick: " if brain.mode == "groundpick" else "sit: ") + diverging)
            last_cmd, diverging, cmd_since = read_cmd(args.cmd, None), None, time.time()
        now = time.perf_counter()
        if guard and guard.state == "stop":
            # a kick or a sit ends through the walk policy at zero command (the Brain stands a sit up first)
            want = stop_command(torque_on, "walk" if brain.mode in ("kick", "sit") else brain.mode)
        if imu is None:
            quat, gyro, imu_valid, imu_age = (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False, 1.0
            acc, imu_seq = None, None
        else:
            arrival, f = imu
            quat, gyro, imu_valid = f[1:5], f[5:8], bool(f[12])
            acc, imu_seq = [round(a, 4) for a in f[8:11]], f[0]
            imu_age = f[11] + (now - arrival) + write_delay  # Pi-side age + time on the Mac + until the write
        joint_age = (now - t_mid) + write_delay
        block = guard.torque_block(args.battery_marker) if guard and not torque_on else None
        if block and guard.dv is not None and want.get("mode", "monitor") in Brain.TORQUE_MODES and now - last_block > 5.0:
            print(f"torque refused: {block}", flush=True)
            last_block = now
        if not torque_on and want.get("mode", "monitor") in Brain.TORQUE_MODES and fresh and not block:
            # never enable torque without every present position, every servo in position mode (3) and every
            # reading away from the 0/4095 wrap edge (ID5 included since its horn refit, pi/hip_yaw_check.py)
            modes = link.sync_read(11, 1, IDS, timeout=0.03)
            bad = [nm for i, nm, p in zip(IDS, NAMES, pos)
                   if i not in modes or modes[i][0] != 3 or not 256 <= p <= 3840]
            if bad:
                if now - last_print > 1.0:
                    print(f"torque refused: {bad} not in position mode or near the wrap edge", flush=True)
                continue
            # a latched hardware error (overload, overheating...) needs a power-cycle; only the battery's
            # input-voltage flag with Shutdown 52 is allowed through
            hw = link.sync_read(ADDR_HW_ERROR, 1, IDS, timeout=0.03, allow_alert=True)
            shutdown = link.sync_read(ADDR_SHUTDOWN, 1, IDS, timeout=0.03, allow_alert=True)
            faults = [nm for i, nm in zip(IDS, NAMES) if i not in hw or i not in shutdown or
                      (hw[i][0] and not voltage_alert_only(hw[i][0], shutdown[i][0]))]
            if faults:
                if now - last_print > 1.0:
                    print(f"torque refused: {faults} report a hardware error (address 70) or did not answer; "
                          "power-cycle the robot", flush=True)
                continue
            goal_written = dict(zip(IDS, pos))
            link.sync_write(ADDR_GOAL, 4, goal_written)  # hold where they are, then enable
            link.sync_write(ADDR_TORQUE, 1, {i: 1 for i in IDS})
            torque_on = True
            print("torque on at present positions", flush=True)
        # Pollen's runtime smooths the velocity command every tick, cmd += cmd_alpha * (target - cmd), cmd_alpha 0.2
        # (pollen-robotics/microduck deploy/robotd.toml [control], read 2026-10-16): "a stick snap becomes a ramp the
        # gait can follow". A for_s stop is then a ramp to zero, not a step.
        if want.get("mode") == "walk":
            for k, key in enumerate(("vx", "vy", "vtheta")):
                cmd_s[k] += CMD_ALPHA * (float(want.get(key, 0)) - cmd_s[k])
                if abs(cmd_s[k]) < 1e-4:
                    cmd_s[k] = 0.0
            want_b = dict(want, vx=cmd_s[0], vy=cmd_s[1], vtheta=cmd_s[2])
            if any(cmd_s) or any(float(want.get(k, 0)) for k in ("vx", "vy", "vtheta")):
                zero_t = None
            elif zero_t is None:
                zero_t = time.perf_counter()
        else:
            cmd_s = [0.0, 0.0, 0.0]
            zero_t = None
            want_b = dict(want)
        if quack_head is not None:
            tq = time.perf_counter() - quack_head_t0
            # stops the move when it may no longer run (a step, a stand-up); the Brain's head EMA eases it back
            if tq >= QUACK_HEAD_S or quack_head_block(want, brain, None if zero_t is None else
                                                      time.perf_counter() - zero_t) is not None:
                quack_head = quack_head_t0 = None
            else:
                want_b["head"] = quack_head_pose(quack_head, tq, brain)
        # only with the leg load calibrated (the hand term alone fell once in the sim, hold.py) and where hold_allowed;
        # the authority handed over ramps across every on/off edge (a switch, a new walk command), never a step
        hold_on = hold_cfg["on"] and est.legs_range is not None and \
            hold_allowed(want_b, brain, None if zero_t is None else time.perf_counter() - zero_t, grip_stop.on)
        target = est.s if hold_on else 1.0
        hold_s = max(target, hold_s - DOWN_RATE * hold_dt) if target < hold_s else min(target, hold_s + UP_RATE * hold_dt)
        if hold_on or hold_s < 1.0:
            want_b["hold"] = {"s": hold_s, "kp": est.gains() if hold_on else None}
        if grip_stop.settle(brain, hold_s):
            print("hold: fully held after the grip stop, the set-down runs stand61", flush=True)
        goal, kp, info = brain.tick(want_b, time.time(), pos, vel, quat, gyro, imu_valid and fresh,
                                    joint_age, imu_age)
        # after the tick, so the stop command has reached the Brain and any ramp to STAND has started
        holding_stand = brain.ramp is None and brain.goal is not None and \
            all(abs(brain.goal[nm] - STAND[nm]) < 1e-6 for nm in NAMES)
        if guard and guard.state == "stop" and stop_settled(torque_on, brain.mode, holding_stand, now - guard.stop_t):
            holding = {"walk": "policy at zero command, torque on", "stand": "STAND, torque on"}.get(
                brain.mode, f"{brain.mode}, torque on") if torque_on else "nothing, torque off"
            msg = guard.signal(args.battery_marker, holding)
            if msg:
                print(msg, flush=True)
        act = info.get("action")
        # only while the file still holds that kick: a command written during the kick takes over instead
        if info.get("kick_done") is not None and last_cmd[1].get("mode") == "kick" \
                and last_cmd[1].get("id") == info["kick_done"]:
            want = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            write_cmd(args.cmd, want, f"kick {info['kick_done']!r} done, stand61 takes over")
            last_cmd, cmd_since = read_cmd(args.cmd, None), time.time()
        if brain.mode == "groundpick" and info.get("gp_phase", 0.0) >= GP_DONE_PHASE \
                and float(want.get("phase", 0.0)) >= GP_DONE_PHASE:
            want = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
            write_cmd(args.cmd, want, "ground pick complete, stand61 takes over")
            last_cmd, cmd_since = read_cmd(args.cmd, None), time.time()
        if brain.mode not in Brain.POLICY_MODES:
            hard_hits.clear()
        if brain.mode in Brain.POLICY_MODES and diverging is None:
            # checked before the goals go out: an action over HARD_MAX_ACTION is never sent, and a run over the limit
            # is cut at its third tick (2026-10-15 the robot fell on the first step after a 1.49 rad knee goal
            # reached the servos and only then tripped this check)
            stepping = brain.mode == "walk" and abs(float(want.get("vx", 0))) + abs(float(want.get("vy", 0))) > 1e-3
            if stepping:
                last_step_t = now
            if act is not None:
                amax = max(abs(a) for a in act)
                hard = args.hard_max_action
                # 2026-10-16: the 1.8 rad limit stopped walks the policy was recovering (a 3-tick run to 2.03 rad at
                # 15 deg; a 4 s walk cut at 1.3 s), and a single 3.16 rad tick at 4.4 deg tilt switched a steady 6 s
                # walk to the open-loop STAND, which fell. Pollen's runtime has no action limit. An action under the
                # hard limit now goes to the servos as is; a tick over it is clipped to +-hard, and only
                # OVER_LIMIT_TICKS ticks over it within HARD_WINDOW ticks (or the tilt limit) switch to stand.
                hard_hits.append(amax > hard)
                over = sum(hard_hits)
                diverging = (f"policy action over {hard} rad in {over} of the last {len(hard_hits)} ticks "
                             f"(last {amax:.2f})" if over >= OVER_LIMIT_TICKS else None)
                if amax > hard and not diverging:
                    goal = walk_brain.clamp_goal({n: (brain.default[n] + max(-hard, min(hard, float(act[k])))
                                                      if n in brain.default else goal[n])
                                                  for k, n in enumerate(OBSERVATION_DOF_ORDER)})
                    brain.goal = dict(goal)
            max_tilt = {"groundpick": args.max_tilt_gp, "sit": args.max_tilt_sit}.get(brain.mode, args.max_tilt)
            carried = hold_on and est.e > HOLD_TILT_E  # tilted in a hand, not falling
            if not diverging and not carried and info["gb"][2] > -math.cos(math.radians(max_tilt)):
                diverging = f"body tilt over {max_tilt} deg"
            if diverging and last_goal is not None:
                goal = dict(last_goal)  # hold the last goal sent; the stand ramp next tick starts from it
                brain.goal = dict(last_goal)
        if torque_on and goal is not None:
            last_goal = dict(goal)
            writes = p_gain_writes(kp, kp_now, settled=kp == hold_kp_prev)
            hold_kp_prev = kp if isinstance(kp, dict) else None
            if isinstance(kp, dict) or (kp_now and len(set(kp_now.values())) > 1):
                hold_gain_t = now
            if now - hold_gain_t < HOLD_REFRESH_WINDOW_S and now >= next_refresh and kp_now:
                next_refresh = now + HOLD_REFRESH_S
                writes = {**{i: kp_now[nm] for i, nm in zip(IDS, NAMES)}, **writes}  # the full set again
            if writes:
                link.sync_write(ADDR_P_GAIN, 2, writes)
                kp_now = dict(kp_now or {}, **{NAMES[IDS.index(i)]: v for i, v in writes.items()})
            goal_out = goal
            # 2026-10-16 02:30: with the push on in walk mode, stand61 (kp 200) diverged in 0.4 s and the duck fell,
            # while the same policy had stood 165 s at kp 200 without it; stand mode (open loop, kp 400) is where the
            # push is validated (E3e). Walk mode gets it only on explicit request.
            if args.breakaway and kp_now and (brain.mode != "walk" or args.breakaway_walk):
                goal_out = comp.apply(goal, {nm: to_model(nm, t) for nm, t in zip(NAMES, pos)}, kp_now["left_knee"])
            # No goal low-pass here: a 2026-10-16 trial (EMA alpha 0.5 on the goals sent to the servos, policy
            # unchanged) left the walking-phase action jumps at 57-70 per 4 s, the same as unfiltered, and fell
            # once at the stop in three walks (unfiltered: 6 of 6 stood). Pollen trains the policies unfiltered.
            goal_written = {i: to_tick(nm, goal_out[nm]) for i, nm in zip(IDS, NAMES)}
            link.sync_write(ADDR_GOAL, 4, goal_written)
            write_delay = 0.8 * write_delay + 0.2 * (time.perf_counter() - t_mid)
        # the jaw: its own servo, its own torque; the body may be off (calibration, grip tests) or on
        if jaw.goal_deg is not None and not jaw.on:
            if not jaw.calibrated:
                if not jaw_warned:
                    print("jaw command ignored: not calibrated (jaw_mark closed / open first)", flush=True)
                    jaw_warned = True
                jaw.goal_deg = None
            elif jaw.last_tick is not None:
                try:
                    pos0 = jaw.setup(link)
                    print(f"jaw torque on at tick {pos0} ({jaw.opening_deg(pos0):.1f} deg open), mode 5, "
                          f"goal current {GOAL_CURRENT_MA} mA", flush=True)
                except RuntimeError as e:
                    print(f"jaw refused: {e}", flush=True)
                    jaw.goal_deg = None
        if jaw.on and jaw.goal_deg is not None:
            measured_head = {nm: to_model(nm, t) for nm, t in zip(NAMES, pos) if nm in POSE_JOINTS}
            jaw.write(link, brain.goal or measured_head, measured_head)
        if guard and guard.due(now):
            volts = link.sync_read(ADDR_PRESENT_VOLTAGE, 2, IDS, timeout=0.015, allow_alert=True)
            event = guard.update(now, {i: int.from_bytes(b, "little") for i, b in volts.items()})
            link.alert_ok = guard.armed
            if event == "armed":
                print(f"battery guard armed: the servos read {guard.dv / 10:.1f} V", flush=True)
            elif event == "unarmed":
                print(f"battery guard not armed: the servos read {guard.dv / 10:.1f} V, a bench rail; on the 2S pack "
                      "restart with --supply battery", flush=True)
            elif event == "warn":
                print(f"BATTERY LOW: {guard.dv / 10:.1f} V, under 7.0 V; stop walking soon and charge the pack",
                      flush=True)
            elif event == "stop":
                print(f"BATTERY STOP: {guard.dv / 10:.1f} V; walking stops and the robot keeps standing", flush=True)
        elif (guard is None or guard.armed) and now >= next_hw:  # half a second away from the voltage read
            next_hw = now + 1.0
            hw = link.sync_read(ADDR_HW_ERROR, 1, IDS, timeout=0.015, allow_alert=True)
            faults = [nm for i, nm in zip(IDS, NAMES) if i in hw and hw[i][0] & ~INPUT_VOLTAGE_BIT]
            if faults != hw_faults:
                print(f"HARDWARE ERROR (address 70) on {faults}: the policy holds its last goals; hold the robot and "
                      "power-cycle" if faults else "hardware errors cleared", flush=True)
                hw_faults = faults
        if (imu_age > IMU_LOST_S) != imu_lost:
            imu_lost = imu_age > IMU_LOST_S
            print(f"IMU LOST: no IMU sample for {imu_age:.2f} s, the policy holds its last goals; hold the robot. "
                  "Check the Pi: ping, systemctl status duck-imu, ~/duck/pwrlog.txt" if imu_lost else "IMU back",
                  flush=True)
        # 50 Hz servo-rail record for brownout diagnosis: one servo's Present Input Voltage per tick, round robin, after the goal write
        vid = IDS[n % len(IDS)]
        vr = link.sync_read(ADDR_PRESENT_VOLTAGE, 2, [vid], timeout=0.003, allow_alert=True)
        servo_v = (vid, struct.unpack("<H", vr[vid])[0] / 10) if vid in vr else (vid, None)
        loop_ms = (time.perf_counter() - t0) * 1000
        log.write(json.dumps(dict(info, t=time.time(), mode=brain.mode, fresh=fresh, quat=quat, gyro=gyro,
                                  imu_valid=imu_valid, imu_age=imu_age, joint_age=joint_age, loop_ms=loop_ms,
                                  torque=torque_on, kp=kp_for_log(kp_now), battery_dv=guard.dv if guard else None,
                                  battery=guard.state if guard else None,
                                  battery_signalled=guard.signalled if guard else None, hw_faults=hw_faults,
                                  missing=missing, read_ms=read_ms, late=link.late,
                                  # raw record for later replay and system ID: accel (m/s2, IMU frame), IMU sample
                                  # seq, servo present ticks and the goal ticks last written, the command in force
                                  acc=acc, imu_seq=imu_seq, pos_tick=list(pos), vel_tick=list(vel),
                                  cur_ma=list(cur), goal_tick=goal_written, cmd=want, servo_v=servo_v,
                                  touch=touch_counts, touch_seq=touch_seq, touch_age=touch_age,
                                  touch_bad=touch.bad if touch else None, hold=est.record(), hold_on=hold_on,
                                  grip_stop=grip_stop.on, legs_ma=legs_ma, jaw_cur_ma=jaw_cur,
                                  jaw_tick=jaw.last_tick, jaw_deg=None if jaw.opening_deg() is None else round(jaw.opening_deg(), 1),
                                  jaw_goal=jaw.goal_deg if jaw.on else None, jaw_clamps=jaw.clamps if jaw.on else [],
                                  jaw_holding=jaw.holding())) + "\n")
        log.flush()  # readers (pickup.py) see the newest tick; buffered writes lagged up to ~1 s
        n += 1
        if now - last_print > 1.0:
            with link.lock:
                imu_hz, link.imu_count = link.imu_count / (now - t_stat), 0
            gb = info["gb"]
            print(f"{brain.mode:<7} {n / (now - t_stat):4.1f} Hz loop {loop_ms:4.1f} ms misses {misses} imu "
                  f"{imu_hz:4.0f}/s age {imu_age * 1000:4.1f} ms joint age {joint_age * 1000:4.1f} ms | body g "
                  f"[{gb[0]:+.2f} {gb[1]:+.2f} {gb[2]:+.2f}] torque {torque_on}"
                  + (f" | gp phase {info['gp_phase']:.2f}" if info.get("gp_phase") is not None else "")
                  + (f" | jaw {jaw.opening_deg():.1f} deg -> {jaw.goal_deg}{' HOLDING' if jaw.holding() else ''}"
                     f"{' clamp ' + ','.join(jaw.clamps) if jaw.clamps else ''}" if jaw.on and jaw.opening_deg() is not None else "")
                  + (f" | hold {'ON' if hold_on else 'off'} grip {est.e:.2f} load "
                     f"{'?' if est.load_f is None else f'{est.load_f:.2f}'} s {est.s:.2f}"
                     if hold_cfg["on"] or est.e > 0.1 else "")
                  + (f" | {guard.describe()}" if guard else "")
                  + (f" | missed by ID {dict(sorted(miss_by_id.items()))} late replies {link.late}" if miss_by_id else ""),
                  flush=True)
            last_print, n, t_stat = now, 0, now
        next_t += PERIOD
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()


if __name__ == "__main__":
    main()
