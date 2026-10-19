"""Sole-stack thickness in the walking sim: how thick may rubber + tape under the printed foot be?

The stack (tape + rubber) is modelled as a rigid shell offset outward from the printed foot by --sole-mm: MuJoCo's
geom margin on the two foot collision meshes (the solver treats the surface as inflated by the margin in every
direction, so the rounded edge grows with it, as a wrapped sheet does). Compliance of the stack is not modelled
(a 1 mm 50A rubber compresses about 0.03 mm under one foot, see README). Everything else is brain_sim_check.run:
as-built mass model, Brain, sensor path, real-condition flags from robot_state_check.py --print-sim-args.

--head-swap replaces the 9.3 g Pi Zero W of the as-built manifest by the A7Z at the same position (mass from
--a7z-g, sourced in README).

  ~/.venvs/microduck-dyn/bin/python sole_thickness_sim.py --sole-mm 1.0 --model asbuilt --volt 7.0 --drop 0.04 \
      --vx 0.25 --walk-at 4 --walk-s 8 --seed 0
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAC = os.path.join(HERE, "..", "..", "..", "guide", "firmware", "mac")
SIM_PYTHON = os.path.expanduser("~/.venvs/microduck-dyn/bin/python")
if os.path.realpath(sys.prefix) != os.path.realpath(os.path.dirname(os.path.dirname(SIM_PYTHON))):
    os.execv(SIM_PYTHON, [SIM_PYTHON, os.path.abspath(__file__)] + sys.argv[1:])
sys.path.insert(0, os.path.abspath(MAC))

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

import brain_sim_check as bsc  # noqa: E402

PI_POS = [0.013898458364897247, -7.9141346850318e-08, -0.050467040423555716]  # add-pi, asbuilt-mass-manifest.json
PI_G = 9.3


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sole-mm", type=float, default=0.0)
    pre.add_argument("--head-swap", action="store_true")
    pre.add_argument("--a7z-g", type=float, default=None)
    mine, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    stats = {}
    orig_load = bsc.ev.load

    def load(c0):
        if mine.head_swap:
            if mine.a7z_g is None:
                raise SystemExit("--head-swap needs --a7z-g")
            c0.setdefault("mass_parts", []).extend([
                dict(body="jaw_soft", mass_kg=-PI_G / 1000, position_m=PI_POS),
                dict(body="jaw_soft", mass_kg=mine.a7z_g / 1000, position_m=PI_POS)])
        m, d, c, names, bam = orig_load(c0)
        feet = [i for i in range(m.ngeom) if (m.geom(i).name or "") in ("left_foot_collision", "right_foot_collision")]
        assert len(feet) == 2, feet
        for g in feet:
            m.geom_margin[g] = mine.sole_mm / 1000
        stats["feet"] = feet
        return m, d, c, names, bam

    bsc.ev.load = load

    # swing-foot clearance: lowest point of the foot collision mesh above the (offset) floor, per tick while walking
    import mujoco
    step0 = mujoco.mj_step
    clear = []
    vcache = {}

    def mj_step(m, d):
        step0(m, d)
        if int(round(d.time / m.opt.timestep)) % 8:
            return
        for g in stats.get("feet", []):
            if g not in vcache:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                vcache[g] = v
            R = d.geom_xmat[g].reshape(3, 3)
            z = (vcache[g] @ R.T)[:, 2] + d.geom_xpos[g][2]
            clear.append((round(d.time, 3), g, float(z.min() - mine.sole_mm / 1000)))

    bsc.mujoco.mj_step = mj_step
    import atexit

    def report():
        if not clear:
            return
        t = np.array([c[0] for c in clear]); z = np.array([c[2] for c in clear]) * 1000
        walk_at = float(rest[rest.index("--walk-at") + 1]) if "--walk-at" in rest else 4.0
        sel = t > walk_at + 1.0
        # per 20 ms window, the higher of the two feet = swing foot height
        hi = {}
        for ti, gi, zi in clear:
            if ti > walk_at + 1.0:
                hi[ti] = max(hi.get(ti, -1), zi * 1000)
        sw = np.array(list(hi.values()))
        print(json.dumps(dict(sole_mm=mine.sole_mm, head_swap=mine.head_swap,
                              swing_peak_mm_p50=float(np.percentile(sw, 50)), swing_peak_mm_p90=float(np.percentile(sw, 90)),
                              min_foot_mm=float(z[sel].min()))), file=sys.stderr)
    atexit.register(report)
    bsc.main()


if __name__ == "__main__":
    main()
