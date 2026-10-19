"""Pi side of the wired MicroDuck: stream BNO085 samples to the OpenRB over the J12 UART; no servo traffic.

With openrb_dual_host on the OpenRB, the MacBook drives the servos over USB and receives these IMU frames on the
same USB link, so neither path uses Wi-Fi. Run this instead of pi_bridge.py (both use /dev/serial0).

Frame: 0xA5 0x5A, payload length, payload, 8-bit sum of the payload.
Payload "<I11fB" (49 bytes): sample seq, quat w x y z, gyro xyz rad/s, accel xyz, age_s, valid.
age_s = time since the Pi read the packet (Pi clock) + the sample's age inside the BNO when it asserted
data-ready (SH-2 base timestamp + report delay, see imu_reader.IMUSnapshot.sensor_delay_s). --raw-age sends the
Pi read age only, the pre-2026-10-15 behaviour.
--rv-report gyro_integrated switches the BNO to the SH-2 Gyro-Integrated Rotation Vector (0x2A, gyroRV channel):
quaternion plus angular velocity in one report, gyro integrated at every gyro sample with the fusion only as a
periodic correction (see imu_reader.ThreadedIMUReader). Default game = Game Rotation Vector 0x08 + gyro 0x02.

  A7Z: python3 pi_imu_uart.py --port /dev/ttyAS2 --imu-bus 7
  python3 pi_imu_uart.py [--port /dev/serial0] [--imu-bus 1] [--baud 500000] [--hz 100] [--raw-age] [--rv-report game|gyro_integrated]
"""
import argparse
import os
import signal
import struct
import sys
import time

import serial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "microduck_rt"))
from imu_reader import RV_REPORT_MODES, ThreadedIMUReader  # noqa: E402

PAYLOAD_FMT = "<I11fB"


def frame(payload):
    return bytes([0xA5, 0x5A, len(payload)]) + payload + bytes([sum(payload) & 0xFF])


def frame_age_s(snap, now, raw_age=False):
    """age_s field of a frame written at host time `now`: Pi read age, plus the BNO-side sample delay unless raw."""
    age = now - snap.timestamp_s
    return age if raw_age else age + snap.sensor_delay_s


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--imu-bus", type=int, default=1)
    ap.add_argument("--baud", type=int, default=500_000,
                    help="J12 UART rate, must match PI_BAUD in openrb_dual_host.ino (the A7Z cannot make 1 Mbaud)")
    ap.add_argument("--hz", type=float, default=100.0)
    ap.add_argument("--raw-age", action="store_true",
                    help="age_s = Pi read age only, without the BNO's reported sample delay (old behaviour)")
    ap.add_argument("--raw-gyro", action="store_true",
                    help="diagnostic: stream raw gyro ADC counts (0x15) in the accel slots instead of the accelerometer")
    ap.add_argument("--rv-report", choices=RV_REPORT_MODES, default="game",
                    help="orientation report: game = Game Rotation Vector 0x08 + Gyroscope 0x02 (default); "
                         "gyro_integrated = Gyro-Integrated Rotation Vector 0x2A (quaternion + angular velocity, "
                         "lower fusion latency, no BNO-side sample delay available)")
    return ap


def main():
    args = build_parser().parse_args()

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    ser = serial.Serial(args.port, args.baud, timeout=0)
    imu = ThreadedIMUReader(i2c_bus=args.imu_bus, frequency_hz=args.hz, rv_report=args.rv_report, raw_gyro=args.raw_gyro)
    imu.start()
    print(f"imu rv report {imu.rv_report} at {args.hz:g} Hz", flush=True)
    seq, last_ts, last_print, sent = 0, None, 0.0, 0
    while not stop["now"]:
        snap = imu.get_latest()
        if snap.timestamp_s != last_ts:
            last_ts = snap.timestamp_s
            age = frame_age_s(snap, time.perf_counter(), args.raw_age)
            ser.write(frame(struct.pack(PAYLOAD_FMT, seq, *snap.quat, *snap.gyro, *snap.acc, age,
                                        1 if snap.valid else 0)))
            seq += 1
            sent += 1
        now = time.perf_counter()
        if now - last_print > 5:
            st = imu.get_status()
            print(f"imu {st['rv_report']} frames {sent / (now - last_print if last_print else 5):.0f}/s valid {snap.valid} "
                  f"sensor delay mean/max {st['sensor_delay_mean_s'] * 1e3:.1f}/{st['sensor_delay_max_s'] * 1e3:.1f} ms"
                  f"{' (not in age_s, --raw-age)' if args.raw_age else ''}", flush=True)
            last_print, sent = now, 0
        time.sleep(0.002)
    print("imu stream stopped", flush=True)


if __name__ == "__main__":
    main()
