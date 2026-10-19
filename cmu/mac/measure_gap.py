"""Measure the IMU pitch bias of the standing MicroDuck against the as-built model (read-only; never writes goals).

Put the robot on a flat table with both soles flat and the servos holding STAND (walk_wired.py in mode "stand",
or torque off and hold it by hand at the STAND pose), then run this for a few seconds. It reads the BNO085 through
the wired link exactly as walk_brain.Brain does (MOUNT_FIX, IMU_MOUNT_QUAT) and compares the body gravity with the
model's feet-flat value at the same pose (dynamics/realgap-20261014/static_gap.py, feet_flat_rot):

  STAND  gravity_body [-0.0853, -0.0006, -0.9964]   trunk pitch -4.89 deg (leaning back)
  HOME   gravity_body [ 0.0017, -0.0006, -1.0000]   trunk pitch +0.10 deg

The difference, robot minus model, is the IMU pitch bias to feed into brain_sim_check.py --imu-pitch and
policy_test.py --imu-pitch (positive = the sensor reads more forward lean than the model). A bias measured the same
at STAND and HOME is in the IMU mount; one that changes with the pose is a joint-zero error (see --pose).

  ~/.venvs/microduck-dyn/bin/python measure_gap.py --pose STAND [--seconds 3] [--out /tmp/duck_gap.json]
  ~/.venvs/microduck-dyn/bin/python measure_gap.py --selftest      # math check without hardware
"""
import argparse
import glob
import json
import math
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walk_brain import IMU_MOUNT_QUAT, MOUNT_FIX, NAMES, STAND, apply_inverse, qmul, to_model  # noqa: E402

MODEL_GB = {"STAND": [-0.0853, -0.0006, -0.9964], "HOME": [0.0017, -0.0006, -1.0]}


def body_gravity(quat):
    """Body gravity the policy sees for one raw BNO085 quaternion (same maths as Brain.tick)."""
    q_body = qmul(qmul(quat, MOUNT_FIX), IMU_MOUNT_QUAT)
    return apply_inverse(q_body, [0.0, 0.0, -1.0])


def pitch_deg(g):
    return math.degrees(math.atan2(g[0], -g[2]))


def compare(samples, pose):
    g = np.mean(samples, axis=0)
    ref = np.array(MODEL_GB[pose])
    out = {"pose": pose, "samples": len(samples), "gravity_body": np.round(g, 4).tolist(),
           "gravity_body_std": np.round(np.std(samples, axis=0), 4).tolist(),
           "robot_pitch_deg": round(pitch_deg(g), 2), "model_pitch_deg": round(pitch_deg(ref), 2),
           "roll_deg": round(math.degrees(math.atan2(-g[1], -g[2])), 2)}
    out["imu_pitch_bias_deg"] = round(out["robot_pitch_deg"] - out["model_pitch_deg"], 2)
    out["sim_args"] = f"--imu-pitch {out['imu_pitch_bias_deg']}"
    return out


def selftest():
    def conj(q):
        return (q[0], -q[1], -q[2], -q[3])
    # a trunk pitched +7 deg forward: build the raw sensor quaternion the way brain_sim_check does and read it back
    a = math.radians(7.0)
    q_trunk = (math.cos(a / 2), 0.0, math.sin(a / 2), 0.0)
    q_sensor = qmul(qmul(q_trunk, conj(IMU_MOUNT_QUAT)), conj(MOUNT_FIX))
    g = body_gravity(q_sensor)
    r = compare([g] * 5, "HOME")
    assert abs(r["robot_pitch_deg"] - 7.0) < 0.05, r
    assert abs(r["imu_pitch_bias_deg"] - 6.9) < 0.05, r
    print(json.dumps({"selftest": "ok", **r}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="STAND", choices=sorted(MODEL_GB))
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default="/tmp/duck_gap.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    from walk_wired import ADDR_VEL, IDS, Link  # noqa: E402  (needs pyserial)
    port = args.port or (glob.glob("/dev/cu.usbmodem*") or [None])[0]
    if port is None:
        sys.exit("no OpenRB USB port")
    link = Link(port)
    time.sleep(0.5)
    samples, joints, t0 = [], None, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        got = link.sync_read(ADDR_VEL, 8, IDS, timeout=0.02)
        if len(got) == len(IDS):
            joints = {n: round(to_model(n, struct.unpack("<ii", got[i])[1]), 4) for i, n in zip(IDS, NAMES)}
        with link.lock:
            imu = link.imu
        if imu is not None and bool(imu[1][12]):
            samples.append(body_gravity(imu[1][1:5]))
        time.sleep(0.02)
    if not samples:
        sys.exit("no valid IMU frames (is pi_imu_uart.py running on the Pi?)")
    out = compare(samples, args.pose)
    if joints is not None:
        out["joints_model_rad"] = joints
        out["joint_minus_pose_deg"] = {n: round(math.degrees(joints[n] - STAND[n]), 1) for n in NAMES} \
            if args.pose == "STAND" else None
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
