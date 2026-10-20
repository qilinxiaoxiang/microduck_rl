"""Measure the closed beak's collision surfaces, not approximate visual sites.

Only reads a normally reset model. Ray intersections use the actual convex
collision pieces. This geometry diagnostic does not establish grasp success.
Coordinates are in the jaw_soft body frame; closure is mostly along local X.
"""
import argparse
import json
import numpy as np
from native_household import NativeHousehold


def ray_interval(vertices, faces, y, z):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    triangles, normals = triangles[valid], normals[valid]/lengths[valid, None]
    inward = np.einsum('ij,ij->i', normals, vertices.mean(axis=0)-triangles[:, 0]) > 0
    normals[inward] *= -1
    rhs = np.einsum('ij,ij->i', normals, triangles[:, 0])-normals[:, 1]*y-normals[:, 2]*z
    parallel = np.abs(normals[:, 0]) < 1e-9
    if np.any(rhs[parallel] < -1e-8):
        return None
    positive, negative = normals[:, 0] > 1e-9, normals[:, 0] < -1e-9
    lo = float(np.max(rhs[negative]/normals[negative, 0], initial=-np.inf))
    hi = float(np.min(rhs[positive]/normals[positive, 0], initial=np.inf))
    return [lo, hi] if lo <= hi else None


def inspect(xml, config):
    r = NativeHousehold(xml, config)
    try:
        m, d = r.model, r.data
        if abs(d.qpos[r.jaw]) > 1e-9:
            raise ValueError('Expected the ordinary closed-jaw reset')
        rotation = d.xmat[r.head].reshape(3, 3)
        pieces = {'upper': [], 'lower': []}
        bounds = {}
        for g, vertices in r.beak_meshes:
            name = m.geom(g).name
            side = 'upper' if name.startswith('beak_soft_mouth_top_') else 'lower' if name.startswith('beak_jaw_soft_') else None
            if not side:
                continue
            local = (vertices@d.geom_xmat[g].reshape(3, 3).T+d.geom_xpos[g]-d.xpos[r.head])@rotation
            mesh = m.geom_dataid[g]
            start, count = m.mesh_faceadr[mesh], m.mesh_facenum[mesh]
            pieces[side].append((local, m.mesh_face[start:start+count].copy()))
        for side, group in pieces.items():
            vertices = np.concatenate([v for v, _ in group])
            bounds[side] = [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
        rays = []
        for y in [-.05, -.038, -.025, 0.]:
            for z in [-.0755, -.074, -.072, -.065]:
                intervals = {side: [interval for v, f in group if (interval := ray_interval(v, f, y, z)) is not None]
                             for side, group in pieces.items()}
                upper = min((v[0] for v in intervals['upper']), default=None)
                lower = max((v[1] for v in intervals['lower']), default=None)
                rays.append(dict(y=y, z=z, upperX=upper, lowerX=lower,
                                 gap=None if upper is None or lower is None else upper-lower))
        return dict(scope='Closed collision geometry only, not physical acceptance', bounds=bounds, rays=rays)
    finally:
        r.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('trace'); args = parser.parse_args()
    with open(args.trace) as stream:
        evidence = json.load(stream)
    print(json.dumps(inspect(evidence['modelXml'], evidence['config']), indent=2))
