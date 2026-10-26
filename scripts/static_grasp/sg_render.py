#!/usr/bin/env python
"""Film an sg_motion / sg_hold qpos log in a mujoco-scenes preset (runs in the scenes kit's venv).

The robot is the ground-grasp robot exported from this worktree (notes/static-grasp/film/sg_robot.xml), the object
the log's box / capsule / cylinder. Fixed camera; --close films the beak at floor level.

  ~/mujoco_scenes/.venv/bin/python scripts/static_grasp/sg_render.py LOG.npz -o out.mp4 [--t0 2.3 --t1 6.0]
  ~/mujoco_scenes/.venv/bin/python scripts/static_grasp/sg_render.py LOG.npz -o out.png --still 5.0 --close
"""
import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path.home() / "mujoco_scenes"))
from scenekit import camera, compose, render, replay, robots as R   # noqa: E402

FILM = Path("/Users/remi/microduck/notes/static-grasp/film")
BRICK = [0.85, 0.12, 0.10, 1.0]       # a red Lego


def build(preset, size, shape, half):
    world = compose.world_spec(preset)
    world.visual.global_.offwidth = max(world.visual.global_.offwidth, size[0])
    world.visual.global_.offheight = max(world.visual.global_.offheight, size[1])
    rb = R.Robot(name="duck", xml=FILM / "sg_robot.xml", height=0.25, look_z=0.10, root="trunk_base")
    rb, prefix = compose.attach(world, rb, mount="mount", prefix="robot/")
    b = world.worldbody.add_body(name="object/object")
    b.add_freejoint(name="object/object_free")
    gt = {"box": mujoco.mjtGeom.mjGEOM_BOX, "capsule_up": mujoco.mjtGeom.mjGEOM_CAPSULE,
          "cyl_up": mujoco.mjtGeom.mjGEOM_CYLINDER, "cyl_lying": mujoco.mjtGeom.mjGEOM_CYLINDER}.get(shape)
    if gt is not None:
        g = b.add_geom(type=gt, size=list(half), rgba=BRICK, contype=0, conaffinity=0)
        if shape == "cyl_lying":
            g.quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]
    else:                           # no object in this log: an invisible placeholder (a free body needs a mass)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], rgba=[0, 0, 0, 0], contype=0,
                   conaffinity=0, mass=0.001)
    model = world.compile()
    data = mujoco.MjData(model)
    return compose.Scene(model, data, compose.load_meta(preset), [compose.Placed(rb, prefix, np.zeros(3), 0.0)], world)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-o", required=True)
    ap.add_argument("--preset", default="studio")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--slow", type=float, default=1.0, help="slow motion factor")
    ap.add_argument("--size", default="960x720")
    ap.add_argument("--az", type=float, default=90.0)
    ap.add_argument("--el", type=float, default=-8.0)
    ap.add_argument("--dist", type=float, default=0.55)
    ap.add_argument("--look", default=None, help="x,y,z camera target (default: between trunk and beak)")
    ap.add_argument("--close", action="store_true", help="beak close-up at floor level")
    ap.add_argument("--ssaa", type=float, default=1.5)
    ap.add_argument("--still", type=float, default=None)
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.split("x"))
    z = np.load(a.log, allow_pickle=True)
    t, q, names = z["t"], z["qpos"], [str(s) for s in z["joint_names"]]
    shape = str(z["object_shape"]) if "object_shape" in z.files else "none"
    half = tuple(float(v) for v in z["object_size"]) if "object_size" in z.files else (0.0, 0.0, 0.0)
    sc = build(a.preset, (int(size[0] * a.ssaa), int(size[1] * a.ssaa)), shape, half)
    m, d = sc.model, sc.data
    adrs, cols, c = [], [], 0
    for n in names:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        assert j >= 0, n
        k = 7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        adrs += list(range(m.jnt_qposadr[j], m.jnt_qposadr[j] + k)); cols += list(range(c, c + k)); c += k
    assert c == q.shape[1], (c, q.shape)
    adrs, cols = np.array(adrs), np.array(cols)
    t1 = a.t1 if a.t1 is not None else t[-1]
    tt, qq = replay.resample(t, q, a.fps * a.slow, a.t0, t1)   # --slow 2: twice as many frames, same fps
    i_ref = int(np.argmin(np.abs(t - (a.still if a.still is not None else 0.5 * (a.t0 + t1)))))
    d.qpos[adrs] = q[i_ref, cols]; mujoco.mj_forward(m, d)
    trunk = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk_base")
    lip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot/lip_up")
    if a.look:
        look = [float(v) for v in a.look.split(",")]
    elif a.close:
        look = [d.site_xpos[lip][0], d.site_xpos[lip][1], 0.012]
    else:
        look = [0.5 * (d.xpos[trunk][0] + d.site_xpos[lip][0]), d.xpos[trunk][1], 0.07]
    dist = 0.16 if a.close and a.dist == 0.55 else a.dist
    el = -1.0 if a.close and a.el == -8.0 else a.el
    rig = camera.Fixed(center=look, distance=dist, azimuth=a.az, elevation=el)
    with render.Renderer(sc, size=size, ssaa=a.ssaa, fx={"dof": None}, no_self_shadow=True, soft=4) as r:
        if a.still is not None:
            import imageio
            i = int(np.argmin(np.abs(t - a.still)))
            d.qpos[adrs] = q[i, cols]; mujoco.mj_forward(m, d)
            imageio.imwrite(a.o, r.frame(rig(0.0, sc)))
            print("wrote", a.o)
            return
        with render.Video(a.o, fps=a.fps, size=size) as v:
            for i in range(len(tt)):
                d.qpos[adrs] = qq[i, cols]; mujoco.mj_forward(m, d)
                v.write(r.frame(rig((tt[i] - tt[0]) * a.slow, sc), t=(tt[i] - tt[0]) * a.slow))
    print("wrote", a.o, len(tt), "frames")


if __name__ == "__main__":
    main()
