"""Wired MicroDuck: the MacBook drives the servos over the OpenRB USB port; the Pi streams the IMU over J12.

Needs openrb_dual_host on the OpenRB and pi_imu_uart.py on the Pi (instead of pi_bridge.py). Same modes and
control file as walk_brain.py (monitor / stand / walk), same policies and compensation; only the transport changes.

Torque: nothing is written in monitor mode. The first stand or walk command writes the present positions as goals
and only then enables torque, so the servos start holding where they are. Stand mode then ramps to STAND; walk
mode runs stand61 (before the first step) and walk61 from the present pose with no open-loop ramp (walk_brain.py).
This process never disables torque. Stopping it leaves the servos holding their last goals.

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
                                          the hard action limit and --max-tilt-gp guard it; a trip holds the last
                                          goals (monitor mode, torque on) instead of ramping to STAND from a crouch.
  "jaw": 20 | "open" | "grip", "jaw_torque": false, "jaw_mark": "closed" | "open"   the ID15 jaw (jaw_ctl.py)

Remote-control additions (2026-10-20, walk_brain.py has the details):
  {"mode": "kick", "foot": "left" | "right", "id": 7}   one kick per new id, from stand or walk mode: 25 ticks of
                                          Pollen's kick policy at zero command, then this process rewrites the
                                          control file to walk at zero command (stand61) if it still holds that kick.
                                          The walk guards apply; a trip stops the kick and goes to stand.
  {"mode": "sit"}                         Pollen's sitstand policy (flag 0 for 0.8 s, then 1); any stand or walk
                                          command stands it up (flag 0 for 2 s) before running. --max-tilt-sit (30)
                                          and the hard action limit guard it; a trip holds the last goals (monitor).
  "quack": N                              any mode; a new value of the counter (start at 1) plays one random call
                                          from --sounds through `aplay -q` on the ALSA default device (duck_sound.py,
                                          off the control thread, a new call kills the old one) and, when the jaw is
                                          calibrated and its torque is on (any jaw command enables it), opens it to
                                          QUACK_JAW_DEG for QUACK_S, under the jaw limits. A gripping jaw stays shut.
  Hold-to-walk: rewrite {"mode": "walk", "vx": 0.25, ..., "for_s": 0.6} every 0.3 s; every rewrite restarts for_s,
  so the walk stops 0.6 s after the last one.

Walk stops: {"mode": "walk", ..., "for_s": 7} sets zero velocity 7 s after the command arrived; walk61 brings its own
stride to rest (torque on), the sequence brain_sim_check.py verifies on the as-built model. An action over 3.0 rad in 3
of 10 ticks or body tilt over --max-tilt deg (30) switches to stand mode; a single tick over 3.0 rad is clipped to
+-3.0, and smaller actions are left to the policy, as in Pollen's runtime. Each stop is written to the control file, so it holds until a new command.

  ~/.venvs/microduck-walk/bin/python walk_wired.py [--port /dev/cu.usbmodem101]
"""
import argparse
import collections
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
from duck_sound import Voice  # noqa: E402
from jaw_ctl import GOAL_CURRENT_MA, JAW_ID, POSE_JOINTS, JawCtl  # noqa: E402
from dxl_fast import _packet  # noqa: E402
from battery import (ADDR_HW_ERROR, ADDR_PRESENT_VOLTAGE, ADDR_SHUTDOWN, INPUT_VOLTAGE_BIT, STOP_MARKER,  # noqa: E402
                     BatteryGuard, status_ok, stop_command, stop_settled, voltage_alert_only)

ADDR_TORQUE, ADDR_P_GAIN, ADDR_GOAL, ADDR_VEL = 64, 84, 116, 128
IMU_FMT = "<I11fB"


def imu_frame_ok(f):
    """Reject a frame whose numbers cannot come from the BNO085: unit quaternion, sane gyro, sane age."""
    quat, gyro, age = f[1:5], f[5:8], f[11]
    if not all(math.isfinite(x) for x in f[1:12]):
        return False
    return 0.9 < math.sqrt(sum(x * x for x in quat)) < 1.1 and all(abs(g) < 35.0 for g in gyro) and 0.0 <= age < 5.0
PERIOD = 0.02


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
GP_DONE_PHASE = 0.9  # inside the policy's standing rest segment: the rise is complete, hand back to stand61
CMD_ALPHA = 0.2  # velocity command EMA per 20 ms tick, Pollen's robotd default
# No IMU sample this long: the Brain holds its last goals (blind, so no policy step). 2026-10-15 the Pi Zero lost power
# 0.85 s into a walk, the stream stopped, the robot froze mid-stride and fell with nothing on the console.
IMU_LOST_S = 0.25


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

    brain = Brain(args.models, compensation=args.compensation)
    guard = None if args.supply == "bench" else BatteryGuard(args.supply)
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

    pos, vel = [0] * 14, [0] * 14
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
    quack_until, quack_prev, quack_hint = None, None, False
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
        got = link.sync_read(ADDR_VEL, 8, IDS + [JAW_ID], timeout=0.018)  # USB adds a few ms; median 6-11 ms measured
        t_read = time.perf_counter()
        t_mid = (t0 + t_read) / 2
        read_ms = (t_read - t0) * 1000
        for k, i in enumerate(IDS):
            if i in got:
                vel[k], pos[k] = struct.unpack("<ii", got[i])
        if JAW_ID in got:  # the jaw rides along; a missing jaw never stops the body
            jaw.last_tick = struct.unpack("<ii", got[JAW_ID])[1]
        fresh = all(i in got for i in IDS) and not hw_faults  # a faulted servo freezes the policy, as on the bench
        misses += not fresh
        missing = [i for i in IDS if i not in got]  # a daisy-chain break drops every ID past it together
        for i in missing:
            miss_by_id[i] = miss_by_id.get(i, 0) + 1
        with link.lock:
            imu = link.imu
        new_cmd = read_cmd(args.cmd, last_cmd)
        cmd_is_new = new_cmd is not last_cmd
        if cmd_is_new:
            cmd_since = time.time()  # when this command arrived (the file's mtime is wall clock, not this loop's)
        last_cmd = new_cmd
        want = last_cmd[1]
        if cmd_is_new and want.get("quack") is not None and want.get("quack") != last_quack:
            last_quack = want.get("quack")
            call = voice.quack()
            if jaw.calibrated and jaw.on and jaw.goal_deg != "grip":
                if quack_until is None:
                    quack_prev = jaw.goal_deg
                jaw.goal_deg, quack_until = QUACK_JAW_DEG, time.perf_counter() + QUACK_S
            elif not quack_hint:
                print("quack: jaw stays put (" + ("gripping" if jaw.goal_deg == "grip" else
                      "not calibrated" if not jaw.calibrated else "torque off; any jaw command, e.g. \"jaw\": 2, "
                      "enables it") + ")", flush=True)
                quack_hint = True
            print(f"quack {last_quack}: {os.path.basename(call) if call else 'no sound'}", flush=True)
        if quack_until is not None and (time.perf_counter() >= quack_until or not jaw.on):
            if jaw.on:
                jaw.goal_deg = quack_prev if quack_prev is not None else 2.0  # back to the rest opening
            quack_until = None
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
            # no open-loop ramp to STAND out of a crouch or a seat: hold the last goals (torque on), wait for a command
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
        else:
            cmd_s = [0.0, 0.0, 0.0]
            want_b = want
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
                    goal = {n: (brain.default[n] + max(-hard, min(hard, float(act[k])))
                                if n in brain.default else goal[n]) for k, n in enumerate(OBSERVATION_DOF_ORDER)}
                    brain.goal = dict(goal)
            max_tilt = {"groundpick": args.max_tilt_gp, "sit": args.max_tilt_sit}.get(brain.mode, args.max_tilt)
            if not diverging and info["gb"][2] > -math.cos(math.radians(max_tilt)):
                diverging = f"body tilt over {max_tilt} deg"
            if diverging and last_goal is not None:
                goal = dict(last_goal)  # hold the last goal sent; the stand ramp next tick starts from it
                brain.goal = dict(last_goal)
        if torque_on and goal is not None:
            last_goal = dict(goal)
            if kp and kp != kp_now:
                link.sync_write(ADDR_P_GAIN, 2, {i: kp for i in IDS})
                kp_now = kp
            goal_out = goal
            # 2026-10-16 02:30: with the push on in walk mode, stand61 (kp 200) diverged in 0.4 s and the duck fell,
            # while the same policy had stood 165 s at kp 200 without it; stand mode (open loop, kp 400) is where the
            # push is validated (E3e). Walk mode gets it only on explicit request.
            if args.breakaway and kp_now and (brain.mode != "walk" or args.breakaway_walk):
                goal_out = comp.apply(goal, {nm: to_model(nm, t) for nm, t in zip(NAMES, pos)}, kp_now)
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
                                  torque=torque_on, kp=kp_now, battery_dv=guard.dv if guard else None,
                                  battery=guard.state if guard else None,
                                  battery_signalled=guard.signalled if guard else None, hw_faults=hw_faults,
                                  missing=missing, read_ms=read_ms, late=link.late,
                                  # raw record for later replay and system ID: accel (m/s2, IMU frame), IMU sample
                                  # seq, servo present ticks and the goal ticks last written, the command in force
                                  acc=acc, imu_seq=imu_seq, pos_tick=list(pos), vel_tick=list(vel),
                                  goal_tick=goal_written, cmd=want, servo_v=servo_v,
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
