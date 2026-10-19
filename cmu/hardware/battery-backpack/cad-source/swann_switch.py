"""Swann 39 rocker switch as one closed solid in the backpack pose (rocker down in the floor pocket), for the assembly
guide, the print gate and the teammate renders. Replaces the REV-31-1387 envelope that build_bp.py places for its fit
checks (build/switch-world.stl stays as build_bp.py writes it).

Shape (mm): body 27.3 x 11.5 (11.5 measured by Shawn 2026-10-17, 27.3 from a Swann 39 listing), 18.4 deep behind the
flange and two lugs 7.2 further (photo IMG_7023 at about 25 px/mm), flange 30 x 13.8 x 2, rocker 25 x 10 ending 0.5 above the
floor face (the REV-envelope tip, 8.49 proud of the flange), two snap wings on the long faces just above the 3 mm panel.
    ~/.venvs/microduck-cad/bin/python swann_switch.py      writes build/switch_swann-world.stl"""
import json
from pathlib import Path
import numpy as np, trimesh, manifold3d as m3

HERE = Path(__file__).resolve().parent; B = HERE / 'build'
SW = json.loads((B / 'build.json').read_text())['switch']
XC, ZPF, TIP = SW['xc'], SW['z_pf'], SW['tip_z']
BODY_T, BODY_L, DEPTH, LUG = 11.5, 27.3, 18.4, 7.2
LUG_Y = (-1.8, -9.0)


def box(lo, hi, rot=None):
    lo, hi = np.array(lo, float), np.array(hi, float)
    m = trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))
    if rot is not None:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rot[0]), [1, 0, 0], rot[1]))
    return m


def parts():
    p = {'body': box([XC - BODY_T / 2, -BODY_L / 2, ZPF], [XC + BODY_T / 2, BODY_L / 2, ZPF + DEPTH]),
         'flange': box([XC - 6.9, -15.0, ZPF - 2.0], [XC + 6.9, 15.0, ZPF + 0.05]),
         'rocker': box([XC - 5.0, -12.5, TIP + 0.5], [XC + 5.0, 12.5, ZPF - 1.9])}            # 0.5 above the floor face
    for s in (1, -1):
        p[f'wing{s:+d}'] = box([XC - 4.5, s * BODY_L / 2 - 0.45 * s - 0.35, ZPF + 3.2], [XC + 4.5, s * BODY_L / 2 - 0.45 * s + 0.35, ZPF + 12.0],
                               (s * 12, [XC, s * BODY_L / 2, ZPF + 3.2]))
    for i, y in enumerate(LUG_Y):
        p[f'lug{i}'] = box([XC - 0.4, y - 2.5, ZPF + DEPTH - 0.1], [XC + 0.4, y + 2.5, ZPF + DEPTH + LUG])
    return p


def solid():
    u = None
    for m in parts().values():
        mm = m3.Manifold(m3.Mesh(vert_properties=np.asarray(m.vertices, np.float32), tri_verts=np.asarray(m.faces, np.uint32)))
        u = mm if u is None else u + mm
    o = u.to_mesh()
    t = trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
    assert t.is_watertight and t.is_volume
    return t


if __name__ == '__main__':
    t = solid(); t.export(B / 'switch_swann-world.stl')
    print(B / 'switch_swann-world.stl', np.round(t.bounds, 2).tolist(), round(t.volume, 1))
