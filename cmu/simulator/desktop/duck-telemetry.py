"""MicroDuck A7Z live status for the Mac: one JSON line on stdout per sample.

The launcher and the simulator's live page pipe this file over SSH (`ssh a7z nice -n 10 python3 -u - duck-telemetry
[fast] < duck-telemetry.py`), so nothing is installed on the A7Z. Health (temperatures, clock, voltages) is sampled
every 0.5 s; with `fast` the brain's pose (joints, body gravity) is sent every 0.1 s in between. It only reads: SoC thermal zones, CPU clock, load, memory and any PMIC voltage from
sysfs, plus the tail of the newest brain tick log (~/duck-logs/duck_wired_log_*.jsonl, written by walk_wired.py).
Servo angles and voltages come from that log; this script never opens the servo bus, so it cannot collide with the
brain on /dev/ttyAS2 or add bus traffic. When no brain is writing a log, there are no servo readings.
"""
import glob
import json
import os
import re
import sys
import time

HEALTH_PERIOD = 0.5
PERIOD = 0.1 if "fast" in sys.argv[1:] else HEALTH_PERIOD
FRESH_S = 3.0  # a tick log older than this belongs to a brain that has stopped
# walk_wired.py logs q in walk_brain.NAMES order: servo IDs 1..14
NAMES = ["right_ankle", "right_knee", "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
         "left_ankle", "left_knee", "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
         "head_pitch", "neck_pitch", "head_yaw", "head_roll"]
SERVO_V = re.compile(rb'"servo_v": \[(\d+), ([0-9.]+)\]')
LOGS = os.path.expanduser("~/duck-logs/duck_wired_log_*.jsonl")


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


zones = sorted(glob.glob("/sys/class/thermal/thermal_zone*"))
zone_names = [(read(z + "/type") or os.path.basename(z)).replace("_thermal_zone", "").replace("_zone", "") for z in zones]
supplies = sorted(glob.glob("/sys/class/power_supply/*/voltage_now"))
boot_id = read("/proc/sys/kernel/random/boot_id")
log_path, log_offset, next_glob = None, 0, 0.0
last_row = None
servo_v = {}  # servo ID -> (volts, time read), from the brain's round-robin voltage reads


def brain_state(now):
    """Last tick of the newest brain log, or why there is none."""
    global log_path, log_offset, next_glob, last_row
    if now >= next_glob:  # a new run starts a new log file; looking once every 2 s is enough
        next_glob = now + 2.0
        found = glob.glob(LOGS)
        newest = max(found, key=os.path.getmtime) if found else None
        if newest != log_path:
            log_path, log_offset, last_row = newest, 0, None
            servo_v.clear()
    if log_path is None:
        return {"running": False}
    try:
        st = os.stat(log_path)
    except OSError:
        log_path = None
        return {"running": False}
    age = now - st.st_mtime
    out = {"running": age < FRESH_S, "log": os.path.basename(log_path), "age": round(age, 1)}
    if age >= FRESH_S:
        return out
    # Read only what was appended since the last look (at most 64 KB, about 0.7 s of 50 Hz ticks).
    start = max(log_offset, st.st_size - 65536)
    with open(log_path, "rb") as f:
        f.seek(start)
        chunk = f.read(st.st_size - start)
    log_offset = start + len(chunk)
    for sid, volts in SERVO_V.findall(chunk):
        servo_v[int(sid)] = (float(volts), now)
    for line in reversed(chunk.rsplit(b"\n", 2)[-2:]):  # the newest complete row; the last piece may be partial
        try:
            last_row = json.loads(line)
            break
        except ValueError:
            continue
    row = last_row
    if row is None:
        return out
    recent = [v for v, t in servo_v.values() if now - t < 2.0]
    q = row.get("q") or []
    out.update(mode=row.get("mode"), torque=row.get("torque"), loop_ms=row.get("loop_ms"), gb=row.get("gb"),
               imu_age_ms=None if row.get("imu_age") is None else row["imu_age"] * 1000,
               hw_faults=row.get("hw_faults") or [], missing=row.get("missing") or [],
               joints={n: round(v, 4) for n, v in zip(NAMES, q)},  # rad, the simulator's joint convention
               jaw_deg=row.get("jaw_deg"),
               servo_v_min=min(recent) if recent else None, servo_v_max=max(recent) if recent else None,
               servo_v_count=len(recent))
    return out


cpu0 = "/sys/devices/system/cpu/cpu0/cpufreq/"
next_t = next_health = time.monotonic()
while True:
    now = time.time()
    if time.monotonic() < next_health:
        print(json.dumps({"t": round(now, 2), "brain": brain_state(now)}, separators=(",", ":")), flush=True)
        next_t += PERIOD
        time.sleep(max(0.0, next_t - time.monotonic()))
        continue
    next_health += HEALTH_PERIOD
    temps = {}
    for z, name in zip(zones, zone_names):
        raw = read(z + "/temp")
        if raw:
            temps[name] = int(raw) / 1000
    volts = {}
    for p in supplies:
        raw = read(p)
        if raw:
            volts[p.split("/")[-2]] = int(raw) / 1e6
    mem = read("/proc/meminfo") or ""
    avail = re.search(r"MemAvailable:\s+(\d+)", mem)
    cur, top, hw_max = read(cpu0 + "scaling_cur_freq"), read(cpu0 + "scaling_max_freq"), read(cpu0 + "cpuinfo_max_freq")
    record = {"t": round(now, 2), "boot": boot_id, "up": float((read("/proc/uptime") or "0").split()[0]),
              "temps": temps, "vin": volts,
              "mhz": int(cur) // 1000 if cur else None, "mhz_cap": int(top) // 1000 if top else None,
              "mhz_hw": int(hw_max) // 1000 if hw_max else None,
              "load": os.getloadavg()[0], "mem_mb": int(avail.group(1)) // 1024 if avail else None,
              "cpu_s": round(sum(os.times()[:2]), 2), "brain": brain_state(now)}
    print(json.dumps(record, separators=(",", ":")), flush=True)
    next_t += PERIOD
    time.sleep(max(0.0, next_t - time.monotonic()))
