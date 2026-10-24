"""Export the as-built MicroDuck kinematic tree to duck_kin.json for duck_kin.py (numpy only, runs on the A7Z).

The A7Z has no MuJoCo; the approach controller needs only the chain trunk -> legs (sole vertices, for the trunk height)
and trunk -> neck -> head (the head_camera and mouth_tip sites). This reads them from the same model target_ground.py
uses (evaluate_delay.load with static_gap.cfg, mass model 'asbuilt') and writes body poses, hinge axes, qpos0, the two
sites and the convex-hull vertices of the foot-collision meshes. test_duck_kin.py checks duck_kin.py against target_ground.DuckGeometry.

    ~/.venvs/microduck-dyn/bin/python duck_kin_export.py      # writes duck_kin.json next to this file
"""
import json
import os
import sys

SIM_PYTHON = os.path.expanduser("~/.venvs/microduck-dyn/bin/python")
if os.path.realpath(sys.prefix) != os.path.realpath(os.path.dirname(os.path.dirname(SIM_PYTHON))):
    os.execv(SIM_PYTHON, [SIM_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from target_ground import DuckGeometry  # noqa: E402

FEET = ("left_foot_collision", "right_foot_collision")
SITES = ("head_camera", "mouth_tip")


def main():
    geom = DuckGeometry(model="asbuilt")
    m = geom.m
    trunk = m.body("trunk_base").id
    keep, index = [], {}
    for b in range(m.nbody):  # body ids are in tree order: a parent always precedes its children
        a = b
        while a not in (0, trunk):
            a = m.body_parentid[a]
        if a == trunk:
            index[b] = len(keep)
            keep.append(b)
    act_joint = {m.jnt(int(m.actuator(n).trnid[0])).name: n for n in geom.names}
    bodies = []
    for b in keep:
        joints = []
        if b != trunk:
            for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
                assert m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE, (m.body(b).name, m.jnt_type[j])
                jn = m.jnt(j).name
                joints.append(dict(name=act_joint.get(jn, jn), axis=m.jnt_axis[j].tolist(), pos=m.jnt_pos[j].tolist(),
                                   qpos0=float(m.qpos0[m.jnt_qposadr[j]])))
        bodies.append(dict(name=m.body(b).name, parent=-1 if b == trunk else index[int(m.body_parentid[b])],
                           pos=m.body_pos[b].tolist(), quat=m.body_quat[b].tolist(), joints=joints))
    sites = {}
    for s in SITES:
        sid = m.site(s).id
        sites[s] = dict(body=index[int(m.site_bodyid[sid])], pos=m.site_pos[sid].tolist(), quat=m.site_quat[sid].tolist())
    feet = {}
    for g in FEET:
        gid = m.geom(g).id
        mid = m.geom_dataid[gid]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        v = v[ConvexHull(v).vertices]  # the lowest point of any pose is a hull vertex: same z minimum, far fewer points
        feet[g] = dict(body=index[int(m.geom_bodyid[gid])], pos=m.geom_pos[gid].tolist(), quat=m.geom_quat[gid].tolist(),
                       verts=v.round(5).tolist())
    out = dict(source="target_ground.DuckGeometry(model='asbuilt'): evaluate_delay.load(static_gap.cfg(0.09, "
                      "model='asbuilt', kp=200, voltage=5.0))", joints=list(geom.names), bodies=bodies, sites=sites,
               feet=feet)
    path = os.path.join(HERE, "duck_kin.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{path}: {len(bodies)} bodies, feet vertices {[len(feet[g]['verts']) for g in FEET]}, "
          f"{os.path.getsize(path) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
