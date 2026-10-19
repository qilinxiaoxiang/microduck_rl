"""Simulated duck behind the pi_bridge.py UDP protocol, for dry-running walk_brain.py before the robot.

Physics is the September whole-robot rehearsal model (hardware-design-20260912/whole-robot-20260912/dynamics/
evaluate.py: BAM XL330 at 5 V, current limit 1.47 A). Every 20 ms tick it writes the newest goals, steps the model,
and sends a DKS1 state packet in raw hardware units (goal/position ticks, velocity LSB, BNO quaternion and gyro in
the sensor frame), so walk_brain runs unchanged. Packets can be delayed by intervals sampled from a real
walk_brain log, to reproduce the Wi-Fi jitter.

  python fake_pi.py [--config dynamics-config.json] [--jitter-log /tmp/duck_walk_log.jsonl] [--trace out.jsonl]
  python walk_brain.py --pi 127.0.0.1 --cmd /tmp/fake_cmd.json --log /tmp/fake_walk_log.jsonl
"""
import argparse
import collections
import heapq
import json
import math
import os
import socket
import struct
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
DYN = os.path.join(HERE, "..", "..", "..", "hardware-design-20260912", "whole-robot-20260912", "dynamics")
sys.path.insert(0, DYN)
sys.path.insert(0, HERE)
import evaluate  # noqa: E402
from walk_brain import (IDS, IMU_MOUNT_QUAT, MOUNT_FIX, NAMES, SIGN, STATE_FMT, CMD_FMT, TICK, VEL_UNIT,  # noqa: E402
                        STAND, qmul, to_tick, to_model)


def conj(q):
    return (q[0], -q[1], -q[2], -q[3])


def rotate(q, v):
    """quat_apply(q, v)"""
    w, xyz = q[0], np.array(q[1:])
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="dynamics config JSON (mass manifest etc.); default: official model at 5 V")
    ap.add_argument("--udp", type=int, default=9870)
    ap.add_argument("--jitter-log", help="walk_brain log whose receive intervals set the packet delays")
    ap.add_argument("--base-delay", type=float, default=0.006, help="one-way delay floor, s")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--trace", default="/tmp/fake_pi_trace.jsonl")
    ap.add_argument("--start", choices=("stand", "home"), default="stand",
                    help="stand: the pose the real robot holds; home: the policy default pose")
    args = ap.parse_args()

    cfg = json.load(open(args.config)) if args.config else {}
    cfg.setdefault("voltage", 5.0)
    cfg.setdefault("kp", 400)
    cfg.setdefault("current_limit", 1.47)
    m, d, c, names, bam = evaluate.load(cfg)
    ji = np.array([m.actuator(n).trnid[0] for n in names])
    qa, va = m.jnt_qposadr[ji], m.jnt_dofadr[ji]
    idx = {n: k for k, n in enumerate(names)}
    trunk = m.body("trunk_base").id
    gyro_adr = m.sensor("imu_ang_vel").adr[0]
    d.qpos[:7] = [0, 0, 0.12, 1, 0, 0, 0]
    for n in NAMES:  # start in the hardware stand pose the real robot holds, or the policy default pose
        d.qpos[qa[idx[n]]] = STAND[n] if args.start == "stand" else evaluate.HOME[idx[n]]
    mujoco.mj_forward(m, d)
    c.q_target = d.qpos[qa].copy()
    # The model cannot hold a pose passively (it relies on the stand policy), so physics is frozen until the first
    # command arrives.
    substeps = round(0.02 / m.opt.timestep)

    extra = []
    if args.jitter_log:
        rows = [json.loads(line) for line in open(args.jitter_log).readlines()[-3000:]]
        ts = [r["t"] for r in rows]
        extra = [max(0.0, (b - a) - 0.02) for a, b in zip(ts, ts[1:])]  # lateness beyond the 20 ms cadence
    rng = np.random.default_rng(0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.udp))
    sock.setblocking(False)
    mac, cmd_seq, seq, applied_age, pending = None, -1, 0, -1.0, None
    goals = {i: to_tick(n, d.qpos[qa[idx[n]]]) for i, n in zip(IDS, NAMES)}
    sample_t = {}
    outbox = []  # (due, pkt)
    trace = open(args.trace, "w")
    t0 = time.perf_counter()
    next_t = t0
    mac_last = None
    while time.perf_counter() - t0 < args.duration:
        now = time.perf_counter()
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except BlockingIOError:
                break
            mac = addr
            if len(data) == struct.calcsize(CMD_FMT) and data[:4] == b"DKC2":
                _, cs, src, kp, *g = struct.unpack(CMD_FMT, data)
                if cs > cmd_seq:
                    cmd_seq, goals, pending = cs, dict(zip(IDS, g)), src
                    if kp and kp != bam.actuator.kp:
                        bam.actuator.kp = kp
        # write goals
        c.q_target = np.array([to_model(n, goals[i]) for i, n in zip(IDS, NAMES)])[[NAMES.index(n) for n in names]]
        if pending is not None and pending in sample_t:
            applied_age = now - sample_t[pending]
            pending = None
        for _ in range(substeps if cmd_seq >= 0 else 0):
            c.update()
            mujoco.mj_step(m, d)
        # read back in hardware units
        sample_t[seq] = time.perf_counter()
        sample_t.pop(seq - 64, None)
        pos = [to_tick(n, d.qpos[qa[idx[n]]]) for n in NAMES]
        vel = [int(round(d.qvel[va[idx[n]]] * SIGN[n] / VEL_UNIT)) for n in NAMES]
        q_body = tuple(d.xquat[trunk])
        q_raw = qmul(qmul(q_body, conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))
        g_body = d.sensordata[gyro_adr:gyro_adr + 3].copy()
        g_raw = rotate(MOUNT_FIX, rotate(IMU_MOUNT_QUAT, g_body))
        tilt = math.degrees(math.acos(float(np.clip(d.xmat[trunk].reshape(3, 3)[2, 2], -1, 1))))
        trace.write(json.dumps({"t": now - t0, "x": float(d.xpos[trunk, 0]), "y": float(d.xpos[trunk, 1]),
                                "z": float(d.xpos[trunk, 2]), "tilt": tilt, "applied_age": applied_age,
                                "kp": bam.actuator.kp}) + "\n")
        if mac is not None:
            pkt = struct.pack(STATE_FMT, b"DKS1", seq, time.time(), *pos, *vel, *q_raw, *g_raw, 0.0, 0.0, -1.0,
                              0.010, 12.0, 15.0, 0.0, 3, applied_age)
            delay = args.base_delay + (float(rng.choice(extra)) if extra else 0.0)
            heapq.heappush(outbox, (time.perf_counter() + delay, seq, pkt))
        while outbox and outbox[0][0] <= time.perf_counter():
            sock.sendto(heapq.heappop(outbox)[2], mac)
        seq += 1
        next_t += 0.02
        while time.perf_counter() < next_t:  # flush delayed packets while waiting for the next tick
            while outbox and outbox[0][0] <= time.perf_counter():
                sock.sendto(heapq.heappop(outbox)[2], mac)
            time.sleep(0.0005)
    print("fake pi done", flush=True)


if __name__ == "__main__":
    main()
