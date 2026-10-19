"""Render the MicroDuck's present pose from its joint encoders and IMU (read-only; never writes to the servos).

Reads one complete joint sample and the newest IMU frame over the wired link (walk_wired.Link), maps them into the
training model with this build's signs, offsets and IMU mount, rests the lowest point on the floor, and renders
three views next to each other.

  ~/.venvs/microduck-dyn/bin/python pose_snapshot.py --out pose.png
"""
import argparse
import glob
import math
import os
import struct
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from walk_brain import IMU_MOUNT_QUAT, MOUNT_FIX, NAMES, qmul, to_model  # noqa: E402
from walk_wired import ADDR_VEL, IDS, Link  # noqa: E402

XML = os.path.join(HERE, "..", "..", "..", "microduck_rl", "src", "mjlab_microduck", "robot", "microduck",
                   "scene_walk.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--out", default="/tmp/duck_pose.png")
    ap.add_argument("--display-fix", nargs=2, action="append", default=[], metavar=("JOINT", "TICKS"),
                    help="add TICKS to JOINT's reading for the picture only")
    args = ap.parse_args()

    link = Link(args.port or glob.glob("/dev/cu.usbmodem*")[0])
    time.sleep(0.5)
    for _ in range(50):
        got = link.sync_read(ADDR_VEL, 8, IDS, timeout=0.02)
        if len(got) == len(IDS):
            break
        time.sleep(0.02)
    else:
        sys.exit(f"incomplete joint read: {sorted(got)}")
    pos = [struct.unpack("<ii", got[i])[1] for i in IDS]
    with link.lock:
        imu = link.imu
    if imu is None:
        sys.exit("no IMU frame")
    quat = qmul(imu[1][1:5], MOUNT_FIX)
    q_body = qmul(quat, IMU_MOUNT_QUAT)
    q = {n: to_model(n, t) for n, t in zip(NAMES, pos)}
    for n, ticks in args.display_fix:  # display-only correction for a servo whose EEPROM offset is pending
        q[n] = to_model(n, pos[NAMES.index(n)] + int(ticks))

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    for n in NAMES:
        d.qpos[m.jnt_qposadr[m.joint(n).id]] = q[n]
    d.qpos[3:7] = q_body
    d.qpos[2] = 0.5
    mujoco.mj_forward(m, d)
    floor = m.geom("floor").id
    low = min(d.geom_xpos[g, 2] - m.geom_rbound[g] * 0.3 for g in range(m.ngeom) if g != floor and m.geom_rbound[g] > 0)
    d.qpos[2] -= low
    mujoco.mj_forward(m, d)

    trunk = m.body("trunk_base").id
    r = mujoco.Renderer(m, height=480, width=480)
    views = []
    for az, el, label in ((90, -10, "side (left)"), (180, -10, "front"), (135, -10, "3/4"), (180, -89.9, "top")):
        cam = mujoco.MjvCamera()
        cam.lookat = d.xpos[trunk].copy()
        cam.distance, cam.azimuth, cam.elevation = 0.55, az, el
        r.update_scene(d, camera=cam)
        views.append(r.render())
    img = np.concatenate(views, axis=1)
    import imageio.v2 as imageio
    imageio.imwrite(args.out, img)
    g = [-2 * (q_body[1] * q_body[3] - q_body[0] * q_body[2])]
    w, x, y, z = q_body
    roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = math.degrees(math.asin(max(-1, min(1, 2 * (w * y - z * x)))))
    print(f"wrote {args.out}; trunk roll {roll:+.0f} deg pitch {pitch:+.0f} deg (body frame)")
    print("joints (deg): " + ", ".join(f"{n} {math.degrees(q[n]):+.0f}" for n in NAMES))


if __name__ == "__main__":
    main()
