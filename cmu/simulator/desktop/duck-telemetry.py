"""MicroDuck A7Z live status for the Mac: one JSON line on stdout per sample.

The launcher and the simulator's live page pipe this file over SSH (`ssh a7z nice -n 10 python3 -u - duck-telemetry
[fast] [bus] < duck-telemetry.py`), so nothing is installed by it. Health (temperatures, clock, voltages) is sampled
every 0.5 s; with `fast` the pose (joints, body gravity) is sent every 0.1 s in between. It reads SoC thermal zones,
CPU clock, load, memory and any PMIC voltage from sysfs, plus the tail of the newest brain tick log
(~/duck-logs/duck_wired_log_*.jsonl, written by walk_wired.py) while a brain runs.
With `fast bus` and no brain running, it starts the brain checkout's read-only standby reader (duck_sense.py, next to
walk_wired.py in the directory ~/duck-run.sh changes to) and reports its servo angles, voltages and IMU gravity as
the pose. duck_sense.py holds the brain's I/O lock and quits within 0.5 s once walk_wired.py starts, so the two never
share /dev/ttyAS2 or the IMU.
Clock per CPU cluster (`clusters`: little cpu0-5, big cpu6-7 on the A7Z): the average over each health period from
cpufreq time_in_state, the thermal cap (scaling_max_freq), the hardware top and the busy share from /proc/stat. The
instantaneous scaling_cur_freq jumps between the lowest and highest step every few ms under ondemand, so one sample
every 0.5 s says little. `mhz` is the little cluster's average (the launcher shows it), `mhz_cap`/`mhz_hw` its caps. The page leaves out `bus` while a brain on the Mac drives the servos over USB: through
the OpenRB bridge either host's request takes the bus, and the Mac would lose replies.
Touch pads (`touch`): the head's XIAO ESP32-S3 answers as I2C slave 0x2A on header pins 29 (PB2, SCL) and 31 (PB3, SDA).
A thread reads it about 8 times a second through TWI0 (/dev/i2c-N, from guide/firmware/pi/duck-twi0-touch.dts) when
that overlay is loaded, else by bit-banging gpiochip0 lines 34/35 (open-drain, no root; one read then costs tens of ms
of CPU, so it pauses while a brain runs). The newest block rides on every record: sequence number, the five counts
(L1, R1, L2, R2, M), reads, failed reads and the bus used.
"""
import ctypes
import fcntl
import struct
import subprocess
import threading
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
SENSE = "bus" in sys.argv[1:] and "fast" in sys.argv[1:]
SENSE_PY = os.path.expanduser("~/.venvs/duck-policy/bin/python")
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
# Tick times ("t" of every row) from the last TICK_WINDOW_S, for the brain's real control rate. walk61 is trained at
# 50 Hz; on 2026-10-23 a walk at 35.7 Hz (SoC 86 C, both clusters capped at 416 MHz) flailed within 0.4 s.
TICK_WINDOW_S = 2.0
TICK_T = re.compile(rb'"t": ([0-9.]+)')
ticks = []


class Standby:
    """duck_sense.py as a child process; its newest line, or why there is none."""

    def __init__(self):
        self.proc, self.row, self.error, self.next_start = None, None, None, 0.0

    def checkout(self):
        text = read(os.path.expanduser("~/duck-run.sh")) or ""
        m = re.search(r"^cd\s+(\S+)", text, re.M)
        return os.path.expanduser(m.group(1)) if m else None

    def ensure(self, now):
        if self.proc is not None and self.proc.poll() is None:
            return
        if self.proc is not None:  # it quit: a brain started, or it failed
            self.error = (self.proc.stderr.read() or b"").decode(errors="replace").strip().split("\n")[-1] or None
            self.proc, self.row = None, None
        if now < self.next_start:
            return
        self.next_start = now + 2.0
        where = self.checkout()
        if not where or not os.path.exists(os.path.join(where, "duck_sense.py")):
            self.error = "duck_sense.py is not in the brain checkout (run install_teleop.sh)"
            return
        self.proc = subprocess.Popen([SENSE_PY, "-u", "duck_sense.py"], cwd=where, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=self.pump, args=(self.proc,), daemon=True).start()

    def pump(self, proc):
        for line in proc.stdout:
            try:
                self.row = (json.loads(line), time.time())
                self.error = None
            except ValueError:
                pass

    def cpu_s(self):
        """CPU seconds of the reader running now plus the ones that have quit (this monitor's cost on the A7Z)."""
        live = 0.0
        if self.proc is not None:
            stat = read(f"/proc/{self.proc.pid}/stat")
            if stat:
                f = stat.rsplit(")", 1)[1].split()
                live = (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")
        return sum(os.times()[2:4]) + live

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=1.0)  # reaped, so its CPU time lands in os.times()
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc, self.row = None, None

    def state(self, now):
        self.ensure(now)
        if self.row is None or now - self.row[1] > 1.0:
            return {"standby": False, **({"standby_error": self.error} if self.error else {})}
        r = self.row[0]
        v = r.get("servo_v") or [None, None, 0]
        return {"standby": True, "gb": r.get("gb"), "imu_age_ms": None if r.get("imu_age") is None else r["imu_age"] * 1000,
                "joints": {n: q for n, q in zip(NAMES, r.get("q") or []) if q is not None}, "jaw_deg": r.get("jaw_deg"),
                "missing": r.get("missing") or [], "servo_v_min": v[0], "servo_v_max": v[1], "servo_v_count": v[2]}


standby = Standby()


class Touch:
    """The XIAO touch block over bit-banged I2C, read by a thread; `state()` is the newest block or why there is none."""
    CHIP, SDA, SCL, ADDR, PERIOD = "/dev/gpiochip0", 35, 34, 0x2A, 0.1
    SATURATED = 0x3FFFFF  # touchRead's ceiling: that pad cannot charge (its lead touches metal, 2026-10-20 the XIAO can)

    class Req(ctypes.Structure):
        _fields_ = [("lineoffsets", ctypes.c_uint32 * 64), ("flags", ctypes.c_uint32),
                    ("default_values", ctypes.c_uint8 * 64), ("consumer_label", ctypes.c_char * 32),
                    ("lines", ctypes.c_uint32), ("fd", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("values", ctypes.c_uint8 * 64)]

    GET_HANDLE = (3 << 30) | (ctypes.sizeof(Req) << 16) | (0xB4 << 8) | 0x03
    GET_VALUES = (3 << 30) | (ctypes.sizeof(Data) << 16) | (0xB4 << 8) | 0x08
    SET_VALUES = (3 << 30) | (ctypes.sizeof(Data) << 16) | (0xB4 << 8) | 0x09

    def __init__(self):
        self.block, self.error, self.ok, self.bad, self.same_since, self.paused, self.bus = None, None, 0, 0, None, False, None
        threading.Thread(target=self.run, daemon=True).start()

    def line(self, chip, off):
        r = self.Req(); r.lineoffsets[0] = off; r.lines = 1; r.flags = (1 << 1) | (1 << 3)  # output, open drain
        r.default_values[0] = 1; r.consumer_label = b"touch"
        fcntl.ioctl(chip, self.GET_HANDLE, r)
        return r.fd

    def set(self, fd, v):
        d = self.Data(); d.values[0] = v; fcntl.ioctl(fd, self.SET_VALUES, d)

    def get(self, fd):
        d = self.Data(); fcntl.ioctl(fd, self.GET_VALUES, d); return d.values[0]

    # No sleeps between line changes: one ioctl from Python takes longer than the 5 us half period of 100 kHz.
    def scl_high(self):
        self.set(self.scl, 1); t0 = time.monotonic()
        while not self.get(self.scl):  # the XIAO stretches the clock while it fills the block
            if time.monotonic() - t0 > 0.05:
                raise OSError("SCL held low")

    def byte_out(self, b):
        for i in range(8):
            self.set(self.sda, (b >> (7 - i)) & 1); self.scl_high(); self.set(self.scl, 0)
        self.set(self.sda, 1); self.scl_high(); ack = self.get(self.sda) == 0
        self.set(self.scl, 0)
        return ack

    def byte_in(self, ack):
        v = 0; self.set(self.sda, 1)
        for _ in range(8):
            self.scl_high(); v = (v << 1) | self.get(self.sda); self.set(self.scl, 0)
        self.set(self.sda, 0 if ack else 1); self.scl_high(); self.set(self.scl, 0)
        self.set(self.sda, 1)
        return v

    def read_block(self):
        """One bit-banged read of exactly 24 bytes: reading more leaves filler queued for later blocks."""
        sda, scl = self.sda, self.scl
        self.set(sda, 1); self.scl_high(); self.set(sda, 0); self.set(scl, 0)
        try:
            if not self.byte_out((self.ADDR << 1) | 1):
                return None
            raw = bytes(self.byte_in(i < 23) for i in range(24))
        finally:
            self.set(sda, 0); self.scl_high(); self.set(sda, 1)
        return self.decode(raw)

    def recover(self):
        """Nine clocks and a STOP end any half-read byte when the reader is out of step with the slave."""
        for _ in range(9):
            self.set(self.scl, 0); self.set(self.scl, 1)
        self.set(self.sda, 0); self.scl_high(); self.set(self.sda, 1)

    @classmethod
    def decode(cls, raw):
        """The block from 24 read bytes. Bit-banged reads get a filler byte (0x55) first and lose the block's last byte,
        the top byte of M, which is always 0. A read without the filler is the block as sent. The wrong alignment shifts
        every count by 8 bits, putting them above 0x3FFFFF, so the right one is the one whose five counts all fit."""
        for block in (raw[1:] + b"\0", raw):
            b = struct.unpack("<6I", block)
            if max(b[1:]) <= cls.SATURATED:
                return b
        return None

    @staticmethod
    def kernel_bus():
        """/dev/i2c-N of TWI0 on PB2/PB3 once zz-duck-twi0-touch.dtbo is loaded (guide/firmware/pi/duck-twi0-touch.dts)."""
        for node in glob.glob("/sys/bus/i2c/devices/i2c-*/of_node"):
            if os.path.realpath(node).endswith("/twi@2510000"):
                return "/dev/" + node.split("/")[-2]
        return None

    def open(self):
        """(read, recover, close) for the kernel bus if there is one, else for the bit-banged GPIO lines."""
        dev = self.kernel_bus()
        if dev:
            fd = os.open(dev, os.O_RDWR)
            fcntl.ioctl(fd, 0x0703, self.ADDR)  # I2C_SLAVE
            self.bus = dev
            return (lambda: self.decode(os.read(fd, 24))), (lambda: None), (lambda: os.close(fd))
        chip = os.open(self.CHIP, os.O_RDWR)
        try:
            self.sda, self.scl = self.line(chip, self.SDA), self.line(chip, self.SCL)
        finally:
            os.close(chip)
        self.bus = "gpio"
        return self.read_block, self.recover, (lambda: (os.close(self.sda), os.close(self.scl)))

    def run(self):
        while True:
            try:
                read, recover, close = self.open()
            except OSError as e:  # EBUSY: touch_live_a7z.py or another reader holds the pins
                self.error = f"pins 29/31: {e.strerror or e}"
                time.sleep(2.0)
                continue
            self.error = None
            last_seq, bad_run = None, 0
            try:
                while True:
                    if self.paused and self.bus == "gpio":  # a brain runs: a bit-banged read costs tens of ms of CPU
                        time.sleep(0.5)
                        continue
                    if bad_run >= 3:
                        recover()
                        bad_run = 0
                    try:
                        b = read()
                    except OSError as e:  # the kernel bus raises on a NACK (EREMOTEIO)
                        b, self.error = None, str(e)
                    if b is None or (last_seq is not None and not 0 < b[0] - last_seq < 500):
                        self.bad += 1
                        bad_run += 1
                        if b is None:
                            last_seq = None
                            self.error = self.error or "no answer at 0x2A"
                    else:
                        if self.block is None or b[1:] != self.block[1:]:
                            self.same_since = time.time()
                        last_seq, self.block, self.ok, self.error, bad_run = b[0], b, self.ok + 1, None, 0
                    time.sleep(self.PERIOD)
            except OSError as e:
                self.error = str(e)
                close()
                time.sleep(2.0)

    def state(self, now, brain_running):
        self.paused = brain_running
        if brain_running and self.bus == "gpio":
            return {"paused": True, "ok": self.ok, "bad": self.bad, "bus": self.bus}
        if self.block is None:
            return {"error": self.error or "no block yet", "ok": self.ok, "bad": self.bad, "bus": self.bus}
        out = {"seq": self.block[0], "v": list(self.block[1:]), "ok": self.ok, "bad": self.bad, "bus": self.bus,
               "stuck": self.SATURATED in self.block[1:] and now - self.same_since > 2.0}
        if self.error:
            out["error"] = self.error
        return out


touch = Touch() if "touch" in sys.argv[1:] else None


def brain_state(now):
    """Last tick of the newest brain log, or why there is none (with the standby reader's pose under `bus`)."""
    out = brain_log(now)
    if out.get("running"):
        standby.stop()
    elif SENSE:
        out.update(standby.state(now))
    return out


def brain_log(now):
    global log_path, log_offset, next_glob, last_row
    if now >= next_glob:  # a new run starts a new log file; looking once every 2 s is enough
        next_glob = now + 2.0
        found = glob.glob(LOGS)
        newest = max(found, key=os.path.getmtime) if found else None
        if newest != log_path:
            log_path, log_offset, last_row = newest, 0, None
            servo_v.clear()
            ticks.clear()
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
    if start > log_offset:  # rows were skipped: a rate from what is left would read low
        ticks.clear()
    with open(log_path, "rb") as f:
        f.seek(start)
        chunk = f.read(st.st_size - start)
    log_offset = start + len(chunk)
    ticks.extend(float(t) for t in TICK_T.findall(chunk))
    if ticks:
        del ticks[:next((i for i, t in enumerate(ticks) if t > ticks[-1] - TICK_WINDOW_S), len(ticks))]
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
               servo_v_count=len(recent),
               # ticks per second over the last TICK_WINDOW_S of the log; None until it spans 1 s
               tick_hz=round((len(ticks) - 1) / (ticks[-1] - ticks[0]), 1) if len(ticks) > 1 and ticks[-1] - ticks[0] >= 1.0 else None)
    return out


class Cluster:
    """One cpufreq policy: average clock and busy share since the previous sample."""

    def __init__(self, policy):
        self.dir = policy + "/"
        self.cpus = [int(c) for c in (read(self.dir + "related_cpus") or "").split()]
        self.prev_state, self.prev_stat = None, None

    def sample(self, stat):
        state = {}
        for line in (read(self.dir + "stats/time_in_state") or "").splitlines():
            f, t = line.split()
            state[int(f)] = int(t)
        busy = [stat[c] for c in self.cpus if c in stat]
        out = {"cpus": f"{self.cpus[0]}-{self.cpus[-1]}" if self.cpus else None,
               "cap_mhz": int(read(self.dir + "scaling_max_freq") or 0) // 1000 or None,
               "hw_mhz": int(read(self.dir + "cpuinfo_max_freq") or 0) // 1000 or None, "avg_mhz": None, "busy": None}
        if self.prev_state and state:
            dt = {f: state[f] - self.prev_state.get(f, 0) for f in state}
            total = sum(dt.values())
            if total > 0:
                out["avg_mhz"] = round(sum(f * d for f, d in dt.items()) / total / 1000)
            elif out["cap_mhz"]:  # no time recorded in the period: stats off or idle at one step
                out["avg_mhz"] = int(read(self.dir + "scaling_cur_freq") or 0) // 1000 or None
        if self.prev_stat and busy:
            d_all = sum(b[0] - p[0] for b, p in zip(busy, self.prev_stat))
            d_idle = sum(b[1] - p[1] for b, p in zip(busy, self.prev_stat))
            out["busy"] = round(100 * (1 - d_idle / d_all), 1) if d_all > 0 else None
        self.prev_state, self.prev_stat = state, busy
        return out


def cpu_times():
    """{cpu: (all jiffies, idle + iowait jiffies)} from /proc/stat."""
    out = {}
    for line in (read("/proc/stat") or "").splitlines():
        m = re.match(r"cpu(\d+)\s+(.*)", line)
        if m:
            v = [int(x) for x in m.group(2).split()]
            out[int(m.group(1))] = (sum(v[:8]), v[3] + v[4])
    return out


CLUSTER_NAMES = {0: "little", 6: "big"}
clusters = {}
for pol in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
    c = Cluster(pol)
    if c.cpus:
        clusters[CLUSTER_NAMES.get(c.cpus[0], f"cpu{c.cpus[0]}")] = c
next_t = next_health = time.monotonic()
while True:
    now = time.time()
    if time.monotonic() < next_health:
        fast = {"t": round(now, 2), "brain": brain_state(now)}
        if touch:
            fast["touch"] = touch.state(now, bool(fast["brain"].get("running")))
        print(json.dumps(fast, separators=(",", ":")), flush=True)
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
    stat = cpu_times()
    clock = {name: c.sample(stat) for name, c in clusters.items()}
    first = next(iter(clock.values()), {})
    record = {"t": round(now, 2), "boot": boot_id, "up": float((read("/proc/uptime") or "0").split()[0]),
              "temps": temps, "vin": volts,
              "clusters": clock, "mhz": first.get("avg_mhz"), "mhz_cap": first.get("cap_mhz"),
              "mhz_hw": first.get("hw_mhz"),
              "load": os.getloadavg()[0], "mem_mb": int(avail.group(1)) // 1024 if avail else None,
              "cpu_s": round(sum(os.times()[:2]) + standby.cpu_s(), 2), "brain": brain_state(now)}
    if touch:
        record["touch"] = touch.state(now, bool(record["brain"].get("running")))
    print(json.dumps(record, separators=(",", ":")), flush=True)
    next_t += PERIOD
    time.sleep(max(0.0, next_t - time.monotonic()))
