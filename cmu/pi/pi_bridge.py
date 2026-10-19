"""Pi side of the MacBook-driven MicroDuck: the Pi only relays.

Every 20 ms the Pi reads the 14 joint positions and velocities over the OpenRB UART bridge plus the BNO085
(game rotation vector, gyro, accel), sends them to the Mac over UDP, and writes the newest goal ticks the Mac sent.
The Pi never turns torque off. With no fresh command it keeps the last goals, so the servos keep holding.
On start it takes over from whatever was holding (stand.py killed with SIGKILL): it reads the present goals and
keeps writing them until the Mac sends new ones.

  python3 pi_bridge.py [--port /dev/serial0] [--udp 9870] [--imu-bus 1]

State packet (Pi -> Mac): b"DKS1", seq u32, pi_time f64, 14 pos i32, 14 vel i32, quat wxyz 4f, gyro 3f, acc 3f,
imu_age f32, read_ms f32, loop_ms f32, cmd_age f32, flags u8 (bit0 joints fresh, bit1 imu valid),
applied_age f32: when the newest command was written to the servos, how old the joint sample it was computed from
was (Pi clock, from the middle of that joint read; -1 if unknown). The Mac adds the IMU age to get the IMU sample age.
Command packet (Mac -> Pi): b"DKC2", seq u32, state seq u32 the goals were computed from, kp u16 (0 = leave),
14 goal ticks i32.
"""
import argparse
import os
import signal
import socket
import struct
import sys
import time


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "microduck_rt"))
from imu_reader import ThreadedIMUReader  # noqa: E402  (copy of the community runtime reader)
from dxl_fast import FastBus  # noqa: E402

IDS = list(range(1, 15))  # the jaw (ID15) is never touched here
ADDR_P_GAIN, ADDR_GOAL, ADDR_VEL = 84, 116, 128  # velocity 128 (4) and position 132 (4) are contiguous
STATE_FMT = "<4sId14i14i4f3f3f4fBf"
CMD_FMT = "<4sIIH14i"
PERIOD = 0.02


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial0")
    ap.add_argument("--udp", type=int, default=9870)
    ap.add_argument("--imu-bus", type=int, default=1)
    args = ap.parse_args()

    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    bus = FastBus(args.port)
    goals = {}
    for _ in range(20):  # take over: continue from the goals the holding process left
        got = bus.sync_read(ADDR_GOAL, 4, [i for i in IDS if i not in goals], timeout=0.03)
        goals.update({i: struct.unpack("<i", b)[0] for i, b in got.items()})
        if len(goals) == len(IDS):
            break
    else:
        sys.exit(f"cannot read goals of {sorted(set(IDS) - set(goals))}; nothing written")
    goals = {i: goals[i] for i in IDS}
    print("took over goals:", goals, flush=True)

    imu = ThreadedIMUReader(i2c_bus=args.imu_bus, frequency_hz=50.0)
    imu.start()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.udp))
    sock.setblocking(False)
    mac, seq, cmd_seq, cmd_t, kp_now = None, 0, -1, 0.0, None
    pos, vel = [0] * 14, [0] * 14
    loop_ms, last_print, misses = 0.0, 0.0, 0
    sample_t = {}  # state seq -> perf_counter at the middle of its joint read
    pending_seq, applied_age = None, -1.0

    next_t = time.perf_counter()
    while not stop["now"]:
        t0 = time.perf_counter()
        # newest command wins
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except BlockingIOError:
                break
            if len(data) == struct.calcsize(CMD_FMT) and data[:4] == b"DKC2":
                _, cs, src_seq, kp, *g = struct.unpack(CMD_FMT, data)
                mac = addr
                if cs > cmd_seq or cs < 10:  # cs < 10: the Mac side restarted
                    cmd_seq, cmd_t = cs, t0
                    goals = dict(zip(IDS, g))
                    pending_seq = src_seq
                    if kp and kp != kp_now:
                        bus.sync_write(ADDR_P_GAIN, 2, {i: kp for i in IDS})
                        kp_now = kp
            elif data[:4] == b"HELO":
                mac = addr

        bus.sync_write(ADDR_GOAL, 4, goals)  # a lost write is repeated 20 ms later; never a reason to stop
        if pending_seq is not None:
            if pending_seq in sample_t:
                applied_age = time.perf_counter() - sample_t[pending_seq]
            pending_seq = None

        tr = time.perf_counter()
        got = bus.sync_read(ADDR_VEL, 8, IDS)
        fresh = len(got) == len(IDS)
        for k, i in enumerate(IDS):
            if i in got:
                vel[k], pos[k] = struct.unpack("<ii", got[i])
        if not fresh:
            misses += 1
        t_read_end = time.perf_counter()
        read_ms = (t_read_end - tr) * 1000
        sample_t[seq] = (tr + t_read_end) / 2
        sample_t.pop(seq - 64, None)

        snap = imu.get_latest()
        if mac is not None:
            pkt = struct.pack(STATE_FMT, b"DKS1", seq, time.time(), *pos, *vel, *snap.quat, *snap.gyro, *snap.acc,
                              time.perf_counter() - snap.timestamp_s, read_ms, loop_ms,
                              (t0 - cmd_t) if cmd_seq >= 0 else -1.0, (1 if fresh else 0) | (2 if snap.valid else 0),
                              applied_age)
            try:
                sock.sendto(pkt, mac)
            except OSError:
                pass
        seq += 1

        if t0 - last_print > 5:
            last_print = t0
            print(f"seq {seq} read {read_ms:.1f} ms loop {loop_ms:.1f} ms misses {misses} imu_age "
                  f"{(time.perf_counter() - snap.timestamp_s) * 1000:.0f} ms applied_age {applied_age * 1000:.0f} ms kp {kp_now} "
                  f"mac {mac}", flush=True)

        loop_ms = (time.perf_counter() - t0) * 1000
        next_t += PERIOD
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()
    print("bridge stopped; torque left on, servos hold their last goals", flush=True)


if __name__ == "__main__":
    main()
