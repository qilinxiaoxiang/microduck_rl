#!/usr/bin/env python
"""Elastic stability of a pose held by P-only servos (no policy): does the duck spring back, or buckle?

Holding constant targets, each servo acts as a torsion spring of stiffness k = 0.554 N.m/rad (BAM XL330 at firmware
kP 200, 7.4 V; robotd's standing gain 160 gives 0.443). With the soles flat on the floor the body's shape is set by
the joint angles, and the total potential energy is
    U(x) = m g z_com(x) + sum_j n_j k / 2 (x_j - x_j0)^2
in the symmetric sagittal coordinates x = (hip pitch, knee, ankle, neck pitch, head pitch) (n_j = 2 servos for the
leg joints, 1 for neck and head). A held pose is elastically stable when the Hessian of U is positive definite: the
gravity Hessian (finite differences of the CoM height through the flat-feet placement) plus the servo springs.
The lowest eigenvalue (N.m/rad) and its mode tell how far from buckling the pose is. Friction in the gearboxes can
hold a slightly unstable pose while nothing moves; any motion then lets it buckle.

  PYTHONPATH=src .venv-mjlab/bin/python scripts/static_grasp/sg_stiffness.py POSE.json [--kp 200]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sg_model as S

K200 = 0.554
NAMES = ["hip pitch", "knee", "ankle", "neck pitch", "head pitch"]
N_SERVOS = np.array([2, 2, 2, 1, 1])


def sym_q(base, x):
    q = np.array(base, dtype=float).copy()
    q[2], q[3], q[4] = x[0], x[1], x[2]
    q[11], q[12], q[13] = -x[0], -x[1], -x[2]
    q[5], q[6] = x[3], x[4]
    return q


def gravity_hessian(sm, q, theta, h=0.01):
    x0 = np.array([q[2], q[3], q[4], q[5], q[6]])

    def U(x):
        sm.place(sym_q(q, x), theta)
        return sm.mass * S.G * sm.d.subtree_com[sm.trunk][2]

    n = len(x0)
    Hm = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            e_i, e_j = np.eye(n)[i] * h, np.eye(n)[j] * h
            Hm[i, j] = Hm[j, i] = (U(x0 + e_i + e_j) - U(x0 + e_i - e_j) - U(x0 - e_i + e_j) + U(x0 - e_i - e_j)) / (4 * h * h)
    return Hm


def stability(sm, q, theta, kp=200.0):
    k = K200 * kp / 200.0
    Hg = gravity_hessian(sm, q, theta)
    Ht = Hg + np.diag(N_SERVOS * k)
    w, v = np.linalg.eigh(Ht)
    mode = v[:, 0] / np.abs(v[:, 0]).max()
    return dict(min_eig=float(w[0]), eigs=w.tolist(), mode=dict(zip(NAMES, np.round(mode, 2).tolist())),
                gravity_min_eig=float(np.linalg.eigvalsh(Hg)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pose", nargs="?")
    ap.add_argument("--kp", type=float, default=200.0)
    a = ap.parse_args()
    sm = S.StaticModel()
    poses = [("standing (home)", S.DEFAULT_POSE, -0.0873)]
    if a.pose:
        P = json.load(open(a.pose))
        q1, th = np.array(P["q"]), float(P["theta"])
        for u in (0.1, 0.25, 0.5, 0.75, 0.9):
            poses.append((f"straight path u={u}", S.DEFAULT_POSE + u * (q1 - S.DEFAULT_POSE), th * u))
        poses.append((Path(a.pose).stem, q1, th))
    for label, q, th in poses:
        r = stability(sm, q, th, a.kp)
        print(f"{label:24s} lowest stiffness {r['min_eig']:+.3f} N.m/rad ({'stable' if r['min_eig'] > 0 else 'BUCKLES'})"
              f"  gravity alone {r['gravity_min_eig']:+.3f}  mode {r['mode']}")


if __name__ == "__main__":
    main()
