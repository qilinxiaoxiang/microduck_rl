# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

from __future__ import annotations  # the A7Z runs Python 3.9

import fcntl
import glob
import os
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from constants import IMU_MOUNT_QUAT

I2C_SLAVE = 0x0703
I2C_SLAVE_FORCE = 0x0706

CHAN_EXE = 1
CHAN_CONTROL = 2
CHAN_REPORTS = 3
CHAN_REPORTS_WAKE = 4
# SH-2 5.2 / 6.5.44.1: the Gyro-Integrated Rotation Vector "is reported on a separate 'gyroRV' SHTP channel";
# CEVA's sh2.c registers it as CHAN_SENSORHUB_INPUT_GIRV (5).
CHAN_GYRO_RV = 5

SET_FEATURE = 0xFD
REPORT_BASE_TIMESTAMP = 0xFB  # SH-2 7.2.1 Base Timestamp Reference, 5 bytes
REPORT_TIMESTAMP_REBASE = 0xFA  # SH-2 7.2.2 Timestamp Rebase, 5 bytes
REPORT_ACCEL = 0x01
REPORT_GYRO = 0x02
REPORT_RV = 0x05
REPORT_GAME_RV = 0x08
# SH-2 6.5.44: enabled with report ID 0x2A on the control channel like any other feature, but its input report is
# 14 bytes with NO report ID / sequence / status / delay header (Figure 119) and arrives on CHAN_GYRO_RV.
REPORT_GYRO_INTEGRATED_RV = 0x2A
REPORT_RAW_GYRO = 0x15  # SH-2 6.5.12: unfiltered MEMS ADC counts + hub timestamp, 16 bytes (diagnostic)
GYRO_RV_REPORT_LEN = 14

RV_REPORT_MODES = ("game", "gyro_integrated")

QUAT_SCALE = 1.0 / (1 << 14)
GYRO_SCALE = 1.0 / (1 << 9)
GYRO_RV_ANGVEL_SCALE = 1.0 / (1 << 10)  # SH-2 6.5.44: "angular velocity is [Q point] 10", rad/s
ACCEL_SCALE = 1.0 / (1 << 8)
TIMESTAMP_TICK_S = 100e-6  # SH-2 base delta, rebase delta and report delay are all in 100 us ticks
BASE_DELTA_OVERFLOW = 0x7FFFFFFF  # SH-2 7.2.1: reserved, base delta did not fit in 32 bits


@dataclass(frozen=True)
class IMUSnapshot:
    timestamp_s: float
    quat: tuple[float, float, float, float]
    gyro: tuple[float, float, float]
    acc: tuple[float, float, float]
    valid: bool
    error_count: int
    # MicroDuck: how old the quaternion sample (gyro's if the packet had no quaternion) already was when the
    # BNO asserted data-ready, from the SH-2 base timestamp and report delay fields. timestamp_s stays the host
    # read time, so the true sample time is timestamp_s - sensor_delay_s.
    sensor_delay_s: float = 0.0


def _normalize_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    norm = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    return (w / norm, x / norm, y / norm, z / norm)


def _quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    w, x, y, z = q
    return (w, -x, -y, -z)


def _quat_mul(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def imu_quat_to_body(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Convert a BNO08x quaternion from the IMU mount frame to the robot body frame."""
    return _normalize_quat(_quat_mul(q, IMU_MOUNT_QUAT))


def quat_apply_inverse(quat: list[float] | tuple[float, float, float, float], vec: list[float] | tuple[float, float, float]) -> list[float]:
    """Apply an inverse quaternion rotation to a vector."""
    xyz = np.array(quat[1:], dtype=float)
    v = np.array(vec, dtype=float)
    w = float(quat[0])
    t = 2.0 * np.cross(xyz, v)
    out = v - w * t + np.cross(xyz, t)
    return out.tolist()

class _BNO08x:
    def __init__(self, bus: int, addr: int) -> None:
        self.fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
        try:
            fcntl.ioctl(self.fd, I2C_SLAVE, addr)
        except OSError:
            fcntl.ioctl(self.fd, I2C_SLAVE_FORCE, addr)
        self.seq = [0] * 6

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass

    def send(self, channel: int, data: bytes | bytearray | list[int]) -> None:
        payload = bytes(data)
        packet = struct.pack("<HBB", len(payload) + 4, channel, self.seq[channel] & 0xFF) + payload
        self.seq[channel] = (self.seq[channel] + 1) & 0xFF
        os.write(self.fd, packet)

    def read_packet(self) -> tuple[int, bytes] | None:
        header = os.read(self.fd, 4)
        if len(header) < 4:
            return None
        raw_len, channel, seq = struct.unpack("<HBB", header)
        total = raw_len & 0x7FFF
        if total <= 4 or total == 0x7FFF or channel > 5:
            return None
        self.seq[channel] = seq
        payload = os.read(self.fd, total)
        if len(payload) == total - 4:
            return channel, payload
        if len(payload) >= 4:
            return channel, payload[4:]
        return None

    def enable_feature(self, report_id: int, interval_us: int) -> None:
        payload = bytearray(17)
        payload[0] = SET_FEATURE
        payload[1] = report_id
        struct.pack_into("<I", payload, 5, interval_us)
        self.send(CHAN_CONTROL, payload)


def _report_delay_ticks(payload: bytes, i: int) -> int:
    """14-bit report delay of the sensor report at payload[i], in 100 us ticks (SH-2 6.5.1 Common Fields:
    status bits 7:2 are the 6 MSBs, byte 3 the 8 LSBs). Unsigned: the sample is this long after the base."""
    return ((payload[i + 2] & 0xFC) << 6) | payload[i + 3]


def _sample_delay_s(base_ticks: int, report_delay_ticks: int) -> float:
    """Seconds between a sample and the host interrupt (HINT) that announced it, positive = sample older.

    SH-2 7.2.1: base delta 10 means the batch is timed relative to HINT - 1 ms, so the base lies base_ticks
    before HINT; the report delay then runs forwards from that base (7.2.2 example). Sample time relative to
    HINT is (report_delay - base) ticks, and its age at HINT is the negative of that."""
    return (base_ticks - report_delay_ticks) * TIMESTAMP_TICK_S


def _parse_reports(payload: bytes) -> dict[str, tuple[float, ...] | float]:
    """Sensor reports of one SHTP packet: quat (w, x, y, z), gyro, acc in SI, plus "<name>_delay_s" for each,
    the sample's age at the moment the BNO asserted data-ready (see _sample_delay_s). Without a base timestamp
    record the delay is taken from the report's own field alone, as CEVA's sh2.c does (base = HINT)."""
    out: dict[str, tuple[float, ...] | float] = {}
    base_ticks = 0
    i = 0
    while i < len(payload):
        report_id = payload[i]
        if report_id == REPORT_BASE_TIMESTAMP and i + 5 <= len(payload):
            delta = struct.unpack_from("<i", payload, i + 1)[0]
            base_ticks = 0 if delta == BASE_DELTA_OVERFLOW else delta
            i += 5
        elif report_id == REPORT_TIMESTAMP_REBASE and i + 5 <= len(payload):
            # Rebase moves the base forwards (later) by its signed delta, so the samples are younger.
            base_ticks -= struct.unpack_from("<i", payload, i + 1)[0]
            i += 5
        elif report_id in (REPORT_GAME_RV, REPORT_RV) and i + 12 <= len(payload):
            iq, jq, kq, rq = struct.unpack_from("<hhhh", payload, i + 4)
            out["quat"] = (
                rq * QUAT_SCALE,
                iq * QUAT_SCALE,
                jq * QUAT_SCALE,
                kq * QUAT_SCALE,
            )
            out["quat_delay_s"] = _sample_delay_s(base_ticks, _report_delay_ticks(payload, i))
            i += 12 if report_id == REPORT_GAME_RV else 14
        elif report_id == REPORT_GYRO and i + 10 <= len(payload):
            gx, gy, gz = struct.unpack_from("<hhh", payload, i + 4)
            out["gyro"] = (gx * GYRO_SCALE, gy * GYRO_SCALE, gz * GYRO_SCALE)
            out["gyro_delay_s"] = _sample_delay_s(base_ticks, _report_delay_ticks(payload, i))
            i += 10
        elif report_id == REPORT_RAW_GYRO and i + 16 <= len(payload):
            gx, gy, gz = struct.unpack_from("<hhh", payload, i + 4)
            out["raw_gyro"] = (float(gx), float(gy), float(gz))  # ADC counts, no filter: latency reference
            i += 16
        elif report_id == REPORT_ACCEL and i + 10 <= len(payload):
            ax, ay, az = struct.unpack_from("<hhh", payload, i + 4)
            out["acc"] = (ax * ACCEL_SCALE, ay * ACCEL_SCALE, az * ACCEL_SCALE)
            out["acc_delay_s"] = _sample_delay_s(base_ticks, _report_delay_ticks(payload, i))
            i += 10
        else:
            i += 1
    return out


def _parse_gyro_rv(payload: bytes) -> dict[str, tuple[float, ...] | float]:
    """Gyro-Integrated Rotation Vector packet from the gyroRV SHTP channel (SH-2 6.5.44.2, Figure 119).

    Each report is 14 bytes: quaternion i, j, k, real (int16 Q14) then angular velocity x, y, z (int16 Q10, rad/s,
    "calibrated gyroscope data" in the same sensor frame as the quaternion). There is no report ID, sequence,
    status or delay field, and no 0xFB base timestamp record ("This is the only type of report delivered on the
    'inputGyroRV' SHTP channel, so it is omitted to save transmission time"). CEVA's sh2.c walks the payload in
    14-byte steps; the newest (last) complete report wins here. With no delay field the sample age is unknown,
    so the delays are reported as 0.0.
    """
    out: dict[str, tuple[float, ...] | float] = {}
    n = (len(payload) // GYRO_RV_REPORT_LEN) * GYRO_RV_REPORT_LEN
    if n == 0:
        return out
    i = n - GYRO_RV_REPORT_LEN
    iq, jq, kq, rq, gx, gy, gz = struct.unpack_from("<hhhhhhh", payload, i)
    out["quat"] = (rq * QUAT_SCALE, iq * QUAT_SCALE, jq * QUAT_SCALE, kq * QUAT_SCALE)
    out["gyro"] = (gx * GYRO_RV_ANGVEL_SCALE, gy * GYRO_RV_ANGVEL_SCALE, gz * GYRO_RV_ANGVEL_SCALE)
    out["quat_delay_s"] = 0.0
    out["gyro_delay_s"] = 0.0
    return out


def _device_responds(bus: int, addr: int) -> bool:
    try:
        fd = os.open(f"/dev/i2c-{bus}", os.O_RDWR)
    except OSError:
        return False
    try:
        try:
            fcntl.ioctl(fd, I2C_SLAVE, addr)
        except OSError:
            fcntl.ioctl(fd, I2C_SLAVE_FORCE, addr)
        os.read(fd, 4)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def _find_bno08x(preferred_bus: int) -> tuple[int, int]:
    buses: list[int] = []
    if preferred_bus >= 0:
        buses.append(preferred_bus)
    for node in sorted(glob.glob("/dev/i2c-*")):
        try:
            bus = int(node.rsplit("-", 1)[1])
        except ValueError:
            continue
        if bus not in buses:
            buses.append(bus)

    for bus in buses:
        for addr in (0x4A, 0x4B):
            if _device_responds(bus, addr):
                return bus, addr
    raise RuntimeError("BNO08x not found at 0x4A/0x4B")


class ThreadedIMUReader:
    """Reads a GY-BNO080/BNO08x on a background thread.

    rv_report selects the orientation source (MicroDuck, 2026-10-15):
      "game"             Game Rotation Vector 0x08 + Gyroscope Calibrated 0x02 + Accelerometer 0x01 on the normal
                         input channel, with SH-2 timestamps. The default and the pre-2026-10-15 behaviour.
      "gyro_integrated"  Gyro-Integrated Rotation Vector 0x2A (SH-2 6.5.44: gyro integrated "at every gyroscope
                         sample and correcting to the more-accurate Rotation Vector periodically"; 4.3.24
                         "high-rate, low-latency") on the gyroRV channel, plus Accelerometer 0x01. The 0x2A report
                         carries the calibrated angular velocity, so 0x02 is not enabled and gyro comes from it.
                         Its report has no timestamp, so sensor_delay_s is 0.0 in this mode.
    """

    def __init__(self, i2c_bus: int, frequency_hz: float = 50.0, warn_interval_s: float = 1.0,
                 rv_report: str = "game", raw_gyro: bool = False) -> None:
        if frequency_hz <= 0:
            raise ValueError("frequency_hz must be > 0")
        if rv_report not in RV_REPORT_MODES:
            raise ValueError(f"rv_report must be one of {RV_REPORT_MODES}, got {rv_report!r}")
        self.rv_report = rv_report
        self.raw_gyro = raw_gyro  # diagnostic: raw gyro ADC counts ride in the acc slots instead of the accelerometer

        self._bus, self._addr = _find_bno08x(i2c_bus)
        self._imu = _BNO08x(self._bus, self._addr)
        self._period_s = 1.0 / frequency_hz
        self._interval_us = int(round(1_000_000.0 / frequency_hz))
        self._warn_interval_s = warn_interval_s

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, name="imu-reader", daemon=True)

        now = time.perf_counter()
        self._snapshot = IMUSnapshot(
            timestamp_s=now,
            quat=(1.0, 0.0, 0.0, 0.0),
            gyro=(0.0, 0.0, 0.0),
            acc=(0.0, 0.0, 0.0),
            valid=False,
            error_count=0,
        )
        self._error_count = 0
        self._last_warn_s = 0.0
        self._delay_hist: deque[tuple[float, float]] = deque()  # (host time, sensor_delay_s) of recent snapshots

    def start(self) -> None:
        if not self._thread.is_alive():
            self._configure()
            self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout_s)
        self._imu.close()

    def get_latest(self) -> IMUSnapshot:
        with self._lock:
            return self._snapshot

    def get_status(self) -> dict[str, float | int | bool]:
        now = time.perf_counter()
        with self._lock:
            snap = self._snapshot
            self._prune_delay_hist(now)
            delays = [d for _, d in self._delay_hist]
        return {
            "valid": snap.valid,
            "age_s": max(0.0, now - snap.timestamp_s),
            "sensor_delay_s": snap.sensor_delay_s,
            "sensor_delay_mean_s": sum(delays) / len(delays) if delays else 0.0,
            "sensor_delay_max_s": max(delays) if delays else 0.0,
            "error_count": snap.error_count,
            "target_frequency_hz": 1.0 / self._period_s,
            "bus": self._bus,
            "addr": self._addr,
            "rv_report": self.rv_report,
        }

    def _prune_delay_hist(self, now: float, window_s: float = 1.0) -> None:
        # caller holds self._lock
        while self._delay_hist and now - self._delay_hist[0][0] > window_s:
            self._delay_hist.popleft()

    def _configure(self) -> None:
        self._imu.send(CHAN_EXE, [1])
        time.sleep(0.8)
        deadline = time.time() + 0.4
        while time.time() < deadline:
            try:
                self._imu.read_packet()
            except OSError:
                time.sleep(0.01)
        if self.rv_report == "gyro_integrated":
            # SH-2 6.5.44.1: "controlled (via set-feature reports) with report ID 0x2A on the common control
            # channel"; the data then comes on CHAN_GYRO_RV. Angular velocity is inside that report (no 0x02).
            self._imu.enable_feature(REPORT_GYRO_INTEGRATED_RV, self._interval_us)
        else:
            self._imu.enable_feature(REPORT_GAME_RV, self._interval_us)
            self._imu.enable_feature(REPORT_GYRO, self._interval_us)
        if self.raw_gyro:
            self._imu.enable_feature(REPORT_RAW_GYRO, self._interval_us)
        else:
            self._imu.enable_feature(REPORT_ACCEL, self._interval_us)
        time.sleep(0.2)

    def _run_loop(self) -> None:
        latest_q = (1.0, 0.0, 0.0, 0.0)
        latest_g = (0.0, 0.0, 0.0)
        latest_a = (0.0, 0.0, 0.0)

        last_q = last_g = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            # MicroDuck: the BNO085 can reset itself (it then stops sending rotation and gyro reports while
            # still answering on the bus). With no fresh quaternion for 0.5 s, mark the data invalid and
            # re-enable the reports.
            last_report = min(last_q, last_g)
            if now - last_report > 0.5:
                self._resets = getattr(self, "_resets", 0) + 1
                print(f"Warning: BNO08x reports stale for {now - last_report:.1f} s, re-enabling (reset {self._resets})",
                      end="\r\n", flush=True)
                with self._lock:
                    p = self._snapshot
                    self._snapshot = IMUSnapshot(p.timestamp_s, p.quat, p.gyro, p.acc, False, self._error_count,
                                                 p.sensor_delay_s)
                try:
                    self._configure()
                except Exception as exc:  # keep trying every 0.5 s
                    print(f"Warning: BNO08x re-enable failed: {exc}", end="\r\n", flush=True)
                last_q = last_g = time.perf_counter()
                continue
            try:
                packet = self._imu.read_packet()
                if packet is None:
                    time.sleep(0.005)  # MicroDuck Pi Zero W: yield the single core when no report is waiting (reports come every 20 ms)
                if packet is not None:
                    channel, payload = packet
                    if channel in (CHAN_REPORTS, CHAN_REPORTS_WAKE):
                        got = _parse_reports(payload)
                    elif channel == CHAN_GYRO_RV:
                        # Only enabled in gyro_integrated mode; harmless to decode in game mode too (the hub does
                        # not send it unless 0x2A was enabled).
                        got = _parse_gyro_rv(payload)
                    else:
                        got = {}
                    if got:
                        # In gyro_integrated mode 0x02 is off: the 0x2A report refreshes both quat and gyro, so
                        # the stale check (min(last_q, last_g) > 0.5 s) keeps working on this channel alone.
                        if "quat" in got:
                            last_q = now
                            latest_q = _normalize_quat(got["quat"])  # type: ignore[arg-type]
                        if "gyro" in got:
                            last_g = now
                            latest_g = got["gyro"]  # type: ignore[assignment]
                        if "acc" in got and not self.raw_gyro:
                            latest_a = got["acc"]  # type: ignore[assignment]
                        if "raw_gyro" in got and self.raw_gyro:
                            latest_a = got["raw_gyro"]  # type: ignore[assignment]
                        # Sensor-side age of this packet's sample at data-ready: the quaternion's, else the gyro's.
                        delay_s = float(got.get("quat_delay_s", got.get("gyro_delay_s", 0.0)))  # type: ignore[arg-type]
                        with self._lock:
                            self._snapshot = IMUSnapshot(
                                timestamp_s=now,
                                quat=latest_q,
                                gyro=latest_g,
                                acc=latest_a,
                                valid=True,
                                error_count=self._error_count,
                                sensor_delay_s=delay_s,
                            )
                            self._delay_hist.append((now, delay_s))
                            self._prune_delay_hist(now)
            except Exception as exc:
                self._error_count += 1
                if (now - self._last_warn_s) >= self._warn_interval_s:
                    print(f"Warning: BNO08x read failed ({self._error_count}): {exc}", end="\r\n", flush=True)
                    self._last_warn_s = now
                with self._lock:
                    prev = self._snapshot
                    self._snapshot = IMUSnapshot(
                        timestamp_s=prev.timestamp_s,
                        quat=prev.quat,
                        gyro=prev.gyro,
                        acc=prev.acc,
                        valid=prev.valid,
                        error_count=self._error_count,
                        sensor_delay_s=prev.sensor_delay_s,
                    )
                time.sleep(0.02)
