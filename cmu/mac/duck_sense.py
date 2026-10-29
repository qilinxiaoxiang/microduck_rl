"""Standby reader for the live page: the robot's pose while no brain runs on this board, without moving it.

    ~/.venvs/duck-policy/bin/python duck_sense.py [--port /dev/ttyAS2 --baud 500000 --imu-bus 7]

duck-telemetry.py (piped over SSH by the Mac's live page) starts this on the A7Z when walk_wired.py is not running and
reads one JSON line per 0.1 s from it:

    {"t": ..., "q": [14 model rad, walk_brain.NAMES order], "jaw_deg": ..., "gb": [3], "imu_age": s,
     "servo_v": [min, max, count], "missing": [servo IDs that did not answer]}

Only reads: Present Position, Present Voltage and the BNO085, through walk_wired.py's own Link and walk_brain.py's
own joint and gravity conversions, so the numbers are the ones the brain would log. It never writes a register
(no torque, goal, mode or limit). It holds the robot I/O lock (walk_wired.HW_LOCK) and exits within 0.5 s when a
walk_wired.py or duck_selfcheck.py process appears (that process waits for the lock), when its own file or a brain
module it imports is replaced (a deploy), or when its parent is gone. Port, baud and IMU bus
default to what ~/duck-run.sh passes walk_wired.py.
"""
import argparse
import fcntl
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walk_wired import HW_LOCK, Link  # noqa: E402
from walk_brain import IDS, NAMES, body_gravity, to_model  # noqa: E402
from jaw_ctl import JAW_ID, JawCtl  # noqa: E402
from battery import ADDR_PRESENT_VOLTAGE  # noqa: E402

ADDR_PRESENT_POSITION = 132
PERIOD = 0.1
VOLT_EVERY = 10  # Present Voltage once a second
BRAIN_EVERY = 5  # the /proc scan for a starting brain, twice a second: at every tick it was half this reader's CPU
IMU_STALE_S = 0.5


def run_args():
    """--port/--baud/--imu-bus from ~/duck-run.sh, the launch the brain uses on this board."""
    try:
        text = open(os.path.expanduser("~/duck-run.sh")).read()
    except OSError:
        text = ""
    pick = lambda flag, dflt: (re.search(rf"{flag}\s+(\S+)", text) or [None, dflt])[1]
    return pick("--port", "/dev/ttyAS2"), int(pick("--baud", 500000)), int(pick("--imu-bus", 7))


def code_stamp():
    """Modification times of this reader and the brain modules it runs: a deploy that replaces any of them makes the
    running reader quit, so duck-telemetry.py restarts it on the new code instead of an old copy staying in memory."""
    stamp = []
    for name in ("duck_sense.py", "walk_wired.py", "walk_brain.py", "jaw_ctl.py", "battery.py"):
        try:
            stamp.append(os.stat(os.path.join(HERE, name)).st_mtime_ns)
        except OSError:
            stamp.append(None)
    return stamp


def brain_running():
    """A python process running walk_wired.py or the boot self-check (duck_selfcheck.py, which takes HW_LOCK for the IMU
    and servo checks); a shell or pgrep that only names the file does not count."""
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().split(b"\0")
        except OSError:
            continue
        if b"python" in os.path.basename(argv[0]) and any(os.path.basename(a) in (b"walk_wired.py", b"duck_selfcheck.py")
                                                          for a in argv[1:3]):
            return True
    return False


def main():
    port, baud, bus = run_args()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=port)
    ap.add_argument("--baud", type=int, default=baud)
    ap.add_argument("--imu-bus", type=int, default=bus)
    args = ap.parse_args()
    parent = os.getppid()
    lock = open(HW_LOCK, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("duck_sense: the brain holds the servo bus and IMU")
    if brain_running():
        sys.exit("duck_sense: walk_wired.py is starting")
    link = Link(args.port, args.baud, args.imu_bus)
    link.alert_ok = True  # the bench rail sets Alert on every status packet; reading needs no fault gating
    jaw = JawCtl()
    volts, n = {}, 0
    next_t = time.monotonic()
    sources = code_stamp()
    while True:
        if os.getppid() != parent or (n % BRAIN_EVERY == 0 and (brain_running() or code_stamp() != sources)):
            return  # the lock and the port close with the process; duck-telemetry.py starts the new code 2 s later
        pos = link.sync_read(ADDR_PRESENT_POSITION, 4, IDS + [JAW_ID], timeout=0.02)
        if n % VOLT_EVERY == 0:
            got = link.sync_read(ADDR_PRESENT_VOLTAGE, 2, IDS, timeout=0.02)
            volts = {i: int.from_bytes(b, "little") / 10 for i, b in got.items()}
        n += 1
        tick = {i: int.from_bytes(b, "little", signed=True) for i, b in pos.items()}
        q = [round(to_model(name, tick[i]), 4) if i in tick else None for name, i in zip(NAMES, IDS)]
        with link.lock:
            imu = link.imu
        now = time.perf_counter()
        gb = imu_age = None
        if imu is not None and now - imu[0] < IMU_STALE_S:
            f = imu[1]
            gb = [round(float(v), 3) for v in body_gravity(tuple(f[1:5]))]
            imu_age = round(now - imu[0], 3)
        vals = list(volts.values())
        print(json.dumps({"t": round(time.time(), 2), "q": q,
                          "jaw_deg": None if JAW_ID not in tick or jaw.opening_deg(tick[JAW_ID]) is None
                          else round(jaw.opening_deg(tick[JAW_ID]), 1),
                          "gb": gb, "imu_age": imu_age,
                          "servo_v": [min(vals), max(vals), len(vals)] if vals else None,
                          "missing": [i for i in IDS + [JAW_ID] if i not in tick]}, separators=(",", ":")), flush=True)
        next_t += PERIOD
        time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    main()
