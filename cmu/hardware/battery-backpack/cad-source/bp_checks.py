"""Battery backpack checks after build_bp.py (same checks as layout-F/post_build.py and raycast_cap_screws.py), world mm:
- print poses: frame rim (X_SPLIT) on the bed and cap rear face (X_LID) on the bed, x' = y, y' = z, z' = x (as the
  V1.16 rear_frame / rear_lid rows); bed contact, height, overhang area, watertight, one body;
- insertion with the cap off: battery out along -x, DFR1202 in along +x, switch up from below; contacts allowed only
  in the switch panel slab (snap wings) and on the DFR clip lips;
- cap path along +x onto the frame with the battery, DFR, switch and plugs in place, and the seated gap;
- outline: every vertex behind the front plate inside the y-z box outline (+0.2 mm);
- wire centrelines inside a printed part (a wire through a wall);
- screws: thread-ring material, head envelope and driver lane for the three cap screws; driver lanes for the four
  fixing screws (upper through the battery space with cap and battery out; lower under the box floor, also against
  every robot part of the guide manifest).
    python bp_checks.py   (CAD venv, bounded runner)   writes bp_checks.json and build/{frame,cap}-print.stl"""
import sys, json
from pathlib import Path
import numpy as np, trimesh
from shapely import contains_xy
from shapely.geometry import Point
HERE = Path(__file__).resolve().parent; B = HERE / 'build'
sys.path.insert(0, str(HERE))
import build_bp as BB
C = BB.C; P = BB.P; BRP = BB.BRP
sys.path.insert(0, 'microduck-sim/guide/content/interference-audit'); import ia_lib
R = json.loads((B / 'build.json').read_text()); XS, XL = R['X_SPLIT'], R['X_LID']
load = lambda n: trimesh.load(str(B / f'{n}-world.stl'), force='mesh')
fr_t, cap_t = load('frame'), load('cap'); fr, cap = P.mesh_to_m(fr_t), P.mesh_to_m(cap_t)
out = {}
import time; T0 = time.time(); stage = lambda m: print(f'[{time.time() - T0:6.1f}s] {m}', flush=True)

# ---------------- print poses
TO_PRINT = {'frame': [[0, 1, 0, 32.05], [0, 0, 1, 44.0], [1, 0, 0, -XS], [0, 0, 0, 1]],
            'cap': [[0, 1, 0, 32.05], [0, 0, 1, 34.0], [1, 0, 0, -XL], [0, 0, 0, 1]]}
pp = {}
for k, t in (('frame', fr_t), ('cap', cap_t)):
    p = t.copy(); p.apply_transform(np.array(TO_PRINT[k], float)); p.apply_translation([0, 0, -p.bounds[0][2]]); p.export(B / f'{k}-print.stl')
    at = np.max(np.abs(p.triangles[:, :, 2]), axis=1) <= 0.01; n = p.face_normals; c = p.triangles_center
    over = (n[:, 2] < -np.cos(np.radians(60))) & (c[:, 2] > 0.05)
    q = t.copy(); q.merge_vertices()
    pp[k] = dict(toPrint=TO_PRINT[k], printBounds=np.round(p.bounds, 2).tolist(), bedContactMm2=round(float(p.area_faces[at].sum()), 1),
                 heightMm=round(float(p.bounds[1][2]), 2), overhangOver60degFromVerticalMm2=round(float(p.area_faces[over].sum()), 1),
                 watertight=bool(q.is_watertight), bodies=len(q.split(only_watertight=False)), volumeMm3=round(float(q.volume), 1))
out['printPoses'] = pp

stage(' insertion')
# ---------------- insertion with the cap off
sw = P.mesh_to_m(load('switch')); dfr = P.mesh_to_m(load('dfr1202')); bat = P.mesh_to_m(load('battery'))
z_pf = R['switch']['z_pf']; xc = R['switch']['xc']
slab = P.box((xc - 9.0, -17.0, z_pf - 0.6), (BB.X_IN + 0.1, 17.0, z_pf + BB.PANEL + 0.6))       # the 3 mm panel: snap wings
pcbf = BB.DFR['back'] - BB.DFR['pcb']
clips = P.union([P.box((pcbf - 1.5, lo, 12.3), (BB.X_IN + 0.1, hi, 39.7)) for lo, hi in ((12.6, 14.2), (-14.2, -12.6))])
fr_ns = fr - slab; fr_nc = fr - clips
ins = {}
def sweep(man, S, vec, dists):
    worst = (0.0, None)
    for d in dists:
        v = (S ^ man.translate((np.asarray(vec) * d).tolist())).volume()
        if v > worst[0]:
            worst = (v, round(float(d), 2))
    return dict(maxOverlapMm3=round(worst[0], 3), atMm=worst[1])
ins['battery_out_along_minus_x'] = sweep(bat, fr, (-1, 0, 0), np.arange(0, 20.01, 0.5))
ins['dfr1202_in_along_plus_x_outside_clips'] = sweep(dfr, fr_nc, (-1, 0, 0), np.arange(0, 20.01, 0.5))
ins['dfr1202_in_along_plus_x_clips_only'] = sweep(dfr, fr ^ clips, (-1, 0, 0), np.arange(0, 20.01, 0.5))
ins['switch_up_from_below_outside_panel'] = sweep(sw, fr_ns, (0, 0, -1), np.arange(0, 45.01, 0.5))
ins['switch_seated_panel_only'] = round((sw ^ (fr ^ slab)).volume(), 2)
out['insertion'] = ins

stage(' cap path')
# ---------------- cap path (+x onto the frame) and seated gaps
inside = P.union([bat, dfr, sw, P.mesh_to_m(load('plugs'))])
cp = {'frame': (0.0, None), 'parts': (0.0, None)}
for d in np.arange(15.0, -0.01, -0.5):
    c = cap.translate([-d, 0, 0])
    for nm, S in (('frame', fr), ('parts', inside)):
        v = (S ^ c).volume()
        if v > cp[nm][0]:
            cp[nm] = (v, round(float(d), 2))
out['capPath'] = {k: dict(maxOverlapMm3=round(v[0], 3), atOffsetMm=v[1]) for k, v in cp.items()}
out['capPath']['seatedGapMm'] = dict(frame=round(float(fr.min_gap(cap, 2.0)), 3), parts=round(float(inside.min_gap(cap, 2.0)), 3))

stage(' outline')
# ---------------- outline
POLY = C.outline_poly(BB.outline(XL))
def outside(t, tol=0.2):
    v = t.vertices; v = v[v[:, 0] < BB.X_IN - 0.05]
    o = ~contains_xy(POLY.buffer(tol), v[:, 1], v[:, 2])
    if not o.any():
        return dict(maxMm=0.0, count=0)
    d = np.array([POLY.exterior.distance(Point(q[1], q[2])) for q in v[o]]); i = int(np.argmax(d))
    return dict(maxMm=round(float(d[i]), 2), at=np.round(v[o][i], 2).tolist(), count=int(o.sum()))
out['outline'] = dict(frame=outside(fr_t), cap=outside(cap_t))
out['outline']['passed'] = out['outline']['frame']['maxMm'] == 0 and out['outline']['cap']['maxMm'] == 0

stage(' wire')
# ---------------- wire centrelines inside a printed part
W = json.loads((B / 'wires.json').read_text()); pen = {}
for k, v in W.items():
    p = np.array(v['points']); q = np.vstack([a + (b - a) * t for a, b in zip(p[:-1], p[1:]) for t in np.linspace(0, 1, 12)])
    row = {}
    for nm, man in (('frame', fr), ('cap', cap)):
        e = 0.02; ins_ = np.array([(man ^ P.box(tuple(x - e), tuple(x + e))).volume() > 0.5 * (2 * e) ** 3 for x in q]); row[nm] = int(ins_.sum())
        if ins_.any():
            row[nm + 'At'] = np.round(q[ins_][len(q[ins_]) // 2], 2).tolist()
    pen[k] = row
out['wireCentrelinePenetration'] = dict(perWire=pen, passed=all(r['frame'] == 0 and r['cap'] == 0 for r in pen.values()))

stage(' screws')
# ---------------- screws
def intervals(m, y, z, x0=-90.0, d=(1.0, 0, 0)):
    loc, _, _ = m.ray.intersects_location([[x0, y, z]], [d], multiple_hits=True)
    if len(loc) == 0:
        return []
    xs = sorted(set(np.round(np.asarray(loc).reshape(-1, 3)[:, 0], 3).tolist()))
    return [xs[i:i + 2] for i in range(0, len(xs) - 1, 2)]
def ring_inside(m, x0, x1, nx, Y, Z, radii):
    """samples on lines along x (rings of radius r around the axis) that lie inside m, from ray-parity intervals per line
    (trimesh.contains needs about 4 MB per point here)."""
    xs = np.linspace(x0, x1, nx); n = 0; tot = 0
    for r in radii:
        for a in (np.linspace(0, 2 * np.pi, 12, endpoint=False) if r > 0 else [0.0]):
            iv = intervals(m, Y + r * np.cos(a), Z + r * np.sin(a), x0=-200.0)
            n += int(sum(((xs > lo) & (xs < hi)).sum() for lo, hi in iv)); tot += len(xs)
    return n, tot
sc = {}
for s in R['capScrews']:
    Y, Z, seat, tip = s['y'], s['z'], s['headSeatX'], s['tipX']; r = dict(axisYZ=[Y, Z], seatX=seat, tipX=tip)
    eng = {}
    for k, m in (('frame', fr_t), ('cap', cap_t)):
        lens = [sum(max(0.0, min(b, tip) - max(a_, seat)) for a_, b in intervals(m, Y + np.cos(a), Z + np.sin(a))) for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
        eng[k] = round(float(np.min(lens)), 2)
    r['ringMaterialSeatToTipMm'] = eng
    hd = {k: ring_inside(m, seat - 1.5, seat - 0.05, 6, Y, Z, (0.5, 1.0, 1.5, 1.9)) for k, m in (('frame', fr_t), ('cap', cap_t))}
    r['headEnvelopeSamplesInside'] = dict(frame=hd['frame'][0], cap=hd['cap'][0], samples=hd['frame'][1])
    ln = {k: ring_inside(m, XL - 0.2, seat - 1.6, 20, Y, Z, (0.0, 1.0, 1.9)) for k, m in (('frame', fr_t), ('cap', cap_t))}
    r['driverLaneDia4SamplesInside'] = dict(frame=ln['frame'][0], cap=ln['cap'][0], samples=ln['frame'][1])
    sc[f'cap y{Y:+.0f} z{Z:+.1f}'] = r
# fixing screws: head on the cover inner face x -47 (M2 pan head 1.3 tall); lane dia 4 from the head back
robot = {}
for k in ia_lib.manifest():
    if k in BB.C.__dict__.get('SKIP', set()) or k in ('rear_frame', 'rear_lid', 'xl4015_top', 'xl4015_mid', 'pd5991'):
        continue
    v, f = ia_lib.load_solid(k); tri = v[f]; tlo, thi = tri.min(1), tri.max(1)
    keep = (thi[:, 0] > -200.0) & (tlo[:, 0] < BB.X_IN) & (thi[:, 2] > -39.5) & (tlo[:, 2] < -33.5) & (np.abs(tri[:, :, 1]).max(1) > 13.0) & (np.abs(tri[:, :, 1]).min(1) < 19.0)
    if keep.any():                                                                   # faces near a lower lane only
        robot[k] = trimesh.Trimesh(v, f[keep], process=True)
    del v, f, tri
def lane_hits(Y, Z, x_end, meshes):
    hits = {}
    for a in [None] + list(np.linspace(0, 2 * np.pi, 12, endpoint=False)):
        for rr in ((0.0,) if a is None else (1.0, 2.0)):
            o = [BB.X_IN - 1.4, Y + (rr * np.cos(a) if a is not None else 0), Z + (rr * np.sin(a) if a is not None else 0)]
            for k, m in meshes.items():
                loc, _, _ = m.ray.intersects_location([o], [[-1.0, 0, 0]], multiple_hits=True)
                xs = [float(p[0]) for p in np.asarray(loc).reshape(-1, 3) if p[0] > x_end]
                if xs:
                    hits[k] = round(max(xs), 2)
    return hits
upper_parts = {'frame': fr_t, 'dfr1202': load('dfr1202'), 'switch': load('switch')}
for (Y, Z) in BRP.POST['holes']:
    if Z > 0:
        h = lane_hits(Y, Z, XS - 40.0, upper_parts)
        sc[f'fixing y{Y:+.1f} z{Z:+.1f}'] = dict(route='with the cap and battery out, straight back through the battery space', laneBlockedBy=h, clear=not h)
    else:
        h = lane_hits(Y, Z, -200.0, {'frame': fr_t, 'cap': cap_t, **robot})
        sc[f'fixing y{Y:+.1f} z{Z:+.1f}'] = dict(route='under the box floor, straight back', laneBlockedBy=h, clear=not h)
out['screws'] = sc
(HERE / 'bp_checks.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k: out[k] for k in ('printPoses', 'insertion', 'capPath', 'outline')}, indent=1))
print('wires through walls:', {k: r for k, r in pen.items() if r['frame'] or r['cap']})
print(json.dumps(sc, indent=1))
