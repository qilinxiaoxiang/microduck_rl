#!/usr/bin/env python3
"""Screen print orientations before slicing.

    /tmp/microduck-cad-venv/bin/python orientation_check.py MANIFEST.json [--key KEY ...]
        [--extra-poses EXTRA.json] [--out REPORT.json] [--max-generated 8]

MANIFEST is a release manifest in the 13-RELEASE-PIPELINE format (parts with path, sha256, sourceToPrint,
flatContact). For every part it measures the current pose and a set of alternative poses, then reports the
current pose next to the best-ranked alternative. EXTRA.json optionally adds named poses per part:
{"<key>": [{"name": "10/1 pose", "sourceToPrint": [[...4x4...]]}]}.

Metrics per pose (all in millimetres, mesh in print frame, lowest point on the bed):
  heightMm, footprintMm         size; height drives layer count and print time
  contactMm2                    triangles with all three vertices within 0.01 mm of the bed (same rule as the
                                release pipeline's flatContact check)
  firstLayerSectionMm2          area of the mesh section at z = 0.1 mm (what a 0.2 mm first layer sees)
  tipMarginMm, tipAngleDeg      signed distance from the centre of mass (projected) to the edge of the convex hull
                                of the first-layer section, and the tilt needed to tip the part over; <= 0 means
                                the part leans on its supports or brim to stay up
  overhang45/55/60Mm2           downward area steeper than 45, 55 and 60 degrees from vertical, excluding the bed
                                face. 60 degrees matches the slicer default support threshold angle of 30 degrees
                                (measured from horizontal) used by the MicroDuck Bambu and Orca profiles.
  supportProxyMm3               sum over 60-degree overhang triangles of projected area x drop height to the bed
                                or to the part below (vertical ray). A screen for support volume, not a slice.
  supportOnModelMm2             projected overhang area whose support would land on the part itself (scars a
                                second surface, harder to remove inside cavities)
  pinsVertical/pinsHorizontal   small round bosses (radius <= 2.5 mm: locating pins, posts). A vertical pin is
                                loaded across its layers when pushed sideways; reported, not penalised.
  bossFacesFacingDown           distinct planar faces of round bosses (pads, pins, posts) that face down above the
                                bed: a seat or pin tip printed on support. Penalised like a down-facing hole end.
  holes                         planar circular holes found in the CAD, with the axis angle to Z classified as
                                vertical (<= 15 deg), inclined, or horizontal (>= 75 deg), and every hole end that
                                faces down above the bed (a screw seat or mating face printed on support or bridge)

The ranking penalty is a documented screen (see PENALTY below); it orders candidates for native slicing and
functional review. It does not replace a native slice, the per-part functional review, or a printed sample.
Read-only on all inputs. Load one mesh at a time; peak memory for MicroDuck parts is well under 1 GB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Point

CONTACT_TOL = 0.01          # release pipeline flat-contact tolerance
SECTION_Z = 0.1             # mid first layer for 0.2 mm first layer
BED_EXCLUDE = 0.05          # faces entirely below this height are bed faces, not overhangs
VERTICAL_DEG, HORIZONTAL_DEG = 15.0, 75.0
SEAT_DOWN_NZ = -0.7         # hole end normal z below this = faces down (more than 45 deg)
NEAR_BED = 0.6              # a down-facing end lower than this is bridged by the first layers (inference; the
                            # mouth ring's 0.3 mm step sliced with zero support on Bambu)
BED_XY, BED_Z = 196.0, 250.0  # common central bed square of both printers (pipeline arrange margin 30 on 256)

# Penalty weights (inference, heuristic). Order of magnitude follows the priority order of the practice guide:
# stability > functional faces/holes > support > height. Checked against the V1.6 study (results.json): the screen
# ranks the chosen pose first for 12 of 13 parts. Known miss: it cannot tell that a curved face is a sliding or
# bearing seat (jaw support), so a functional review and a native slice decide every flagged case.
PENALTY = {
    'notSelfStanding': 1000.0,     # centre of mass outside the contact hull
    'noFlatContact': 300.0,        # nothing within 0.01 mm of the bed (curved or point contact)
    'lowTipAnglePerDeg': 4.0,      # per degree below 10 deg tip angle
    'smallContactPerMm2': 0.5,     # per mm2 below 50 mm2 of true contact
    'horizontalHole': 6.0,         # per functional hole printed with a horizontal axis
    'inclinedHole': 3.0,           # per hole with an inclined axis
    'seatDown': 10.0,              # per hole end facing down above the bed (rough, supported or bridged seat)
    'overhang60PerCm2': 1.0,       # per cm2 of area that the slicer will support
    'supportProxyPerCm3': 1.5,     # per cm3 of support-volume proxy
    'supportOnModelPerCm2': 1.0,   # extra per cm2 of support that lands on the part itself
    'heightPerMm': 0.15,           # time, layer count, tipping leverage
    'tooBig': 10000.0,             # does not fit the common bed area
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_mesh(path):
    m = trimesh.load(path, force='mesh', process=False)
    v, inv = np.unique(m.vertices, axis=0, return_inverse=True)
    return trimesh.Trimesh(v, inv.reshape(-1)[m.faces], process=False)


def rotation_to_down(d):
    """Shortest rotation that maps unit vector d (source frame) to (0, 0, -1)."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    t = np.array([0.0, 0.0, -1.0])
    c = float(np.dot(d, t))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(d, t)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s ** 2)


def min_area_yaw(points_xy):
    """Yaw (radians) that minimises the axis-aligned footprint rectangle of the projected hull."""
    hull = MultiPoint([tuple(p) for p in points_xy]).convex_hull
    if hull.geom_type != 'Polygon':
        return 0.0
    xy = np.asarray(hull.exterior.coords)
    best = (math.inf, 0.0)
    for a, b in zip(xy[:-1], xy[1:]):
        ang = -math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(ang), math.sin(ang)
        r = xy @ np.array([[c, s], [-s, c]])
        area = np.ptp(r[:, 0]) * np.ptp(r[:, 1])
        if area < best[0] - 1e-9:
            best = (area, ang)
    return best[1]


def pose_from_down(mesh, d):
    """Full 4x4 sourceToPrint for bed-facing source direction d: lay down, yaw to minimum footprint, drop to bed."""
    R = rotation_to_down(d)
    yaw = min_area_yaw((mesh.convex_hull.vertices @ R.T)[:, :2])
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = Rz @ R
    return normalise(mesh, T)


def normalise(mesh, T):
    """Translate so the transformed mesh is centred on XY origin with its lowest point at z = 0."""
    T = np.array(T, float)
    p = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    lo, hi = p.min(0), p.max(0)
    T[:3, 3] -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    return T


# ----------------------------------------------------------------------------------------------- features
def detect_holes(mesh, rmin=0.5, rmax=15.0):
    """Circular loops on planar facets. Inner loops (facet material outside the circle) are holes; outer loops are
    bosses/pins. Returns merged holes with axis, radius and every end (facet outward normal, centre, facet area)."""
    loops = []
    V = mesh.vertices
    for fi, (faces, normal, edges) in enumerate(zip(mesh.facets, mesh.facets_normal, mesh.facets_boundary)):
        if len(edges) < 6:
            continue
        u = np.cross(normal, [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0])
        u /= np.linalg.norm(u)
        w = np.cross(normal, u)
        cent2 = np.c_[mesh.triangles_center[faces] @ u, mesh.triangles_center[faces] @ w]
        for comp in trimesh.graph.connected_components(edges, min_len=6):
            p = V[comp]
            q = np.c_[p @ u, p @ w]
            A = np.c_[2 * q, np.ones(len(q))]
            sol, *_ = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)
            cx, cy = sol[:2]
            r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0))
            if not rmin <= r <= rmax:
                continue
            dist = np.hypot(q[:, 0] - cx, q[:, 1] - cy)
            if np.sqrt(np.mean((dist - r) ** 2)) / r > 0.06:
                continue
            ang = np.sort(np.arctan2(q[:, 1] - cy, q[:, 0] - cx))
            if np.max(np.diff(np.r_[ang, ang[0] + 2 * np.pi])) > np.pi / 2:
                continue                       # an arc, not a full circle
            inside = np.mean(np.hypot(cent2[:, 0] - cx, cent2[:, 1] - cy) < r)
            centre = cx * u + cy * w + normal * float(np.mean(p @ normal))
            kind = 'hole' if inside < 0.5 else 'boss'
            if kind == 'hole':
                # A true hole is empty just inside its rim on both sides of the facet. Material in front of the facet
                # just inside the rim means the loop outlines a protrusion (a boss, or a ring boss with its own bore);
                # material behind the centre means a solid step around a protrusion.
                ang8 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                rim = (centre + (r - 0.3) * (np.outer(np.cos(ang8), u) + np.outer(np.sin(ang8), w))) + 0.3 * normal
                if mesh.contains(rim).mean() >= 0.5 or mesh.contains([centre - 0.3 * normal])[0]:
                    kind = 'boss'
            loops.append(dict(kind=kind, r=r, axis=normal.copy(), centre=centre,
                              facet=fi, facetArea=float(mesh.facets_area[fi]), vertexIds=np.asarray(comp)))
    merged = []
    for lp in loops:
        for h in merged:
            if h['kind'] != lp['kind'] or abs(h['r'] - lp['r']) > 0.3 or abs(np.dot(h['axis'], lp['axis'])) < 0.996:
                continue
            off = lp['centre'] - h['ends'][0]['centre']
            if np.linalg.norm(off - np.dot(off, h['axis']) * h['axis']) < 0.3:
                h['ends'].append(dict(normal=lp['axis'], centre=lp['centre'], facet=lp['facet'], facetArea=lp['facetArea'],
                                      vertexIds=lp['vertexIds']))
                break
        else:
            merged.append(dict(kind=lp['kind'], r=lp['r'], axis=lp['axis'],
                               ends=[dict(normal=lp['axis'], centre=lp['centre'], facet=lp['facet'],
                                          facetArea=lp['facetArea'], vertexIds=lp['vertexIds'])]))
    return merged


# ------------------------------------------------------------------------------------------------ metrics
def section_area(mesh, z):
    try:
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if sec is None:
            return 0.0
        planar, _ = sec.to_2D()
        return float(sum(p.area for p in planar.polygons_full))
    except Exception:
        return float('nan')


def metrics(src, T, holes, intended_contact=None):
    m = src.copy()
    m.apply_transform(T)
    m.apply_translation([0, 0, -m.bounds[0, 2]])
    tri = m.triangles
    zmax = tri[:, :, 2].max(1)
    zmin = tri[:, :, 2].min(1)
    contact = zmax <= CONTACT_TOL
    contact_area = float(m.area_faces[contact].sum())
    ext = m.extents
    com = m.center_mass if m.is_volume else m.centroid
    # Tipping: convex hull of the first-layer section (z = 0.1 mm; everything the first layer bonds to the bed),
    # plus true flat contact; else everything within 0.1 mm.
    pts = tri[contact].reshape(-1, 3)[:, :2]
    seg = trimesh.intersections.mesh_plane(m, [0, 0, 1], [0, 0, SECTION_Z])
    if len(seg):
        pts = np.r_[pts.reshape(-1, 2), seg.reshape(-1, 3)[:, :2]]
    hull_basis = 'firstLayerSection+contact'
    if len(pts) < 3:
        pts = m.vertices[m.vertices[:, 2] <= 0.1][:, :2]
        hull_basis = 'vertices0.1'
    margin, hull_area = -math.inf, 0.0
    if len(pts) >= 3:
        hull = MultiPoint([tuple(p) for p in pts]).convex_hull
        if hull.geom_type == 'Polygon':
            hull_area = hull.area
            d = hull.exterior.distance(Point(com[0], com[1]))
            margin = d if hull.contains(Point(com[0], com[1])) else -d
    tip = math.degrees(math.atan2(margin, max(com[2], 1e-6))) if math.isfinite(margin) else -90.0
    n = m.face_normals
    a = m.area_faces
    above = zmin > BED_EXCLUDE
    over = {deg: float(a[above & (-n[:, 2] > math.sin(math.radians(deg)))].sum()) for deg in (45, 55, 60)}
    # Support proxy: vertical ray from each 60-degree overhang triangle centroid.
    sel = np.nonzero(above & (-n[:, 2] > math.sin(math.radians(60))))[0]
    proxy = on_model = 0.0
    if len(sel):
        c = m.triangles_center[sel]
        origins = c - np.array([0, 0, 1e-3])
        drop = c[:, 2].copy()
        landed = np.zeros(len(sel), bool)
        try:
            loc, ray_idx, _ = m.ray.intersects_location(origins, np.tile([0, 0, -1.0], (len(sel), 1)), multiple_hits=False)
            for (x, y, z), i in zip(loc, ray_idx):
                if c[i, 2] - z > 1e-3:
                    drop[i] = c[i, 2] - z
                    landed[i] = True
        except Exception:
            pass
        proj = a[sel] * -n[sel, 2]
        proxy = float((proj * drop).sum())
        on_model = float(proj[landed].sum())
    # Holes and their ends.
    R = T[:3, :3]
    hole_rows = []
    vertical = inclined = horizontal = seats_down = 0
    seat_area_down = 0.0
    for h in holes:
        if h['kind'] != 'hole':
            continue
        ax = R @ h['axis']
        tilt = math.degrees(math.acos(min(1.0, abs(ax[2]))))
        cls = 'vertical' if tilt <= VERTICAL_DEG else 'horizontal' if tilt >= HORIZONTAL_DEG else 'inclined'
        vertical += cls == 'vertical'
        inclined += cls == 'inclined'
        horizontal += cls == 'horizontal'
        ends = []
        for e in h['ends']:
            nz = float((R @ e['normal'])[2])
            ez = float(m.vertices[e['vertexIds'], 2].min())
            state = end_state(ez, nz)
            if state == 'down':
                seats_down += 1
                seat_area_down += e['facetArea']
            ends.append(dict(state=state, heightMm=round(ez, 2)))
        hole_rows.append(dict(diameterMm=round(2 * h['r'], 2), tiltDeg=round(tilt, 1), axisClass=cls, ends=ends))
    pins_v = pins_h = 0
    boss_rows, boss_down_facets = [], set()
    for h in holes:
        if h['kind'] != 'boss':
            continue
        ax = R @ h['axis']
        tilt = math.degrees(math.acos(min(1.0, abs(ax[2]))))
        if h['r'] <= 2.5:
            pins_v += tilt <= VERTICAL_DEG
            pins_h += tilt >= HORIZONTAL_DEG
        ends = []
        for e in h['ends']:
            nz = float((R @ e['normal'])[2])
            ez = float(m.vertices[e['vertexIds'], 2].min())
            state = end_state(ez, nz)
            if state == 'down':
                boss_down_facets.add(e['facet'])
            ends.append(dict(state=state, heightMm=round(ez, 2)))
        boss_rows.append(dict(diameterMm=round(2 * h['r'], 2), tiltDeg=round(tilt, 1), ends=ends))
    bed_dir = (np.linalg.inv(R) @ np.array([0, 0, -1.0]))
    fits = sorted(ext[:2]) <= sorted([BED_XY, BED_XY]) and ext[2] <= BED_Z
    out = dict(heightMm=round(float(ext[2]), 2), footprintMm=[round(float(x), 1) for x in ext[:2]],
               contactMm2=round(contact_area, 1), firstLayerSectionMm2=round(section_area(m, SECTION_Z), 1),
               contactHullMm2=round(hull_area, 1), hullBasis=hull_basis,
               comHeightMm=round(float(com[2]), 2), tipMarginMm=round(margin, 2) if math.isfinite(margin) else None,
               tipAngleDeg=round(tip, 1), overhang45Mm2=round(over[45], 1), overhang55Mm2=round(over[55], 1),
               overhang60Mm2=round(over[60], 1), supportProxyMm3=round(proxy, 0), supportOnModelMm2=round(on_model, 1),
               holesVertical=vertical, holesInclined=inclined, holesHorizontal=horizontal, holeEndsFacingDown=seats_down,
               holeEndsFacingDownFacetMm2=round(seat_area_down, 1), holes=hole_rows,
               pinsVertical=pins_v, pinsHorizontal=pins_h, bossFacesFacingDown=len(boss_down_facets), bosses=boss_rows,
               bedFacingSourceDir=[round(float(x), 4) for x in bed_dir], fitsCommonBed=bool(fits))
    if intended_contact is not None:
        out['meetsIntendedContact'] = contact_area >= intended_contact
    out['penalty'], out['penaltyTerms'] = penalty(out)
    return out


def end_state(ez, nz):
    if ez <= CONTACT_TOL and nz < -0.9:
        return 'onBed'
    if nz < SEAT_DOWN_NZ:
        return 'nearBed' if ez < NEAR_BED else 'down'
    return 'up' if nz > -SEAT_DOWN_NZ else 'side'


def penalty(x):
    t = {}
    if not x['fitsCommonBed']:
        t['tooBig'] = PENALTY['tooBig']
    if x['tipMarginMm'] is None or x['tipMarginMm'] <= 0:
        t['notSelfStanding'] = PENALTY['notSelfStanding']
    elif x['tipAngleDeg'] < 10:
        t['lowTipAngle'] = PENALTY['lowTipAnglePerDeg'] * (10 - x['tipAngleDeg'])
    if x['contactMm2'] <= 0:
        t['noFlatContact'] = PENALTY['noFlatContact']
    elif x['contactMm2'] < 50:
        t['smallContact'] = PENALTY['smallContactPerMm2'] * (50 - x['contactMm2'])
    t['horizontalHoles'] = PENALTY['horizontalHole'] * x['holesHorizontal']
    t['inclinedHoles'] = PENALTY['inclinedHole'] * x['holesInclined']
    t['seatsDown'] = PENALTY['seatDown'] * x['holeEndsFacingDown']
    t['bossFacesDown'] = PENALTY['seatDown'] * x['bossFacesFacingDown']
    t['overhang60'] = PENALTY['overhang60PerCm2'] * x['overhang60Mm2'] / 100
    t['supportProxy'] = PENALTY['supportProxyPerCm3'] * x['supportProxyMm3'] / 1000
    t['supportOnModel'] = PENALTY['supportOnModelPerCm2'] * x['supportOnModelMm2'] / 100
    t['height'] = PENALTY['heightPerMm'] * x['heightMm']
    t = {k: round(v, 2) for k, v in t.items() if v}
    return round(sum(t.values()), 2), t


# ---------------------------------------------------------------------------------------------- candidates
def generated_directions(mesh, limit):
    """Bed-facing directions from (a) trimesh stable poses on the convex hull, (b) large planar facets of the part."""
    dirs = []
    try:
        Ts, probs = trimesh.poses.compute_stable_poses(mesh, sigma=0.0, n_samples=1, threshold=0.0)
        for T, p in sorted(zip(Ts, probs), key=lambda x: -x[1])[:limit]:
            dirs.append((f'stable p={p:.2f}', np.linalg.inv(T[:3, :3]) @ np.array([0, 0, -1.0])))
    except Exception as e:  # noqa: BLE001
        print(f'  stable poses failed: {e}', file=sys.stderr)
    order = np.argsort(-mesh.facets_area)
    k = 0
    for fi in order:
        if mesh.facets_area[fi] < 20 or k >= limit:
            break
        nrm = mesh.facets_normal[fi]
        h = mesh.vertices @ nrm
        plane = float(np.mean(mesh.vertices[mesh.faces[mesh.facets[fi]].reshape(-1)] @ nrm))
        if h.max() - plane > 0.05:     # facet is not the outermost plane in its normal direction
            continue
        dirs.append((f'flat facet {mesh.facets_area[fi]:.0f} mm2', nrm))
        k += 1
    return dirs


def candidates(mesh, current_T, extras, limit):
    rows = [dict(name='current', sourceToPrint=normalise(mesh, current_T), origin='manifest')]
    for e in extras:
        rows.append(dict(name=e['name'], sourceToPrint=normalise(mesh, np.array(e['sourceToPrint'], float)), origin='extra'))
    for name, d in generated_directions(mesh, limit):
        rows.append(dict(name=name, sourceToPrint=pose_from_down(mesh, d), origin='generated'))
    kept = []
    for r in rows:
        d = np.linalg.inv(r['sourceToPrint'][:3, :3]) @ np.array([0, 0, -1.0])
        dup = next((k for k in kept if np.dot(k['_d'], d) > math.cos(math.radians(3))), None)
        if dup is None:
            r['_d'] = d
            r['sameAs'] = []
            kept.append(r)
        else:
            dup['sameAs'].append(r['name'])
    return kept


def check_part(row, extras, limit):
    path = Path(row['path'])
    if 'sha256' in row and sha256(path) != row['sha256']:
        raise SystemExit(f'{row["key"]}: source hash differs from manifest')
    mesh = load_mesh(path)
    holes = detect_holes(mesh)
    intended = float(row.get('flatContact', {}).get('minimumAreaMm2', 0)) or None
    out = []
    for c in candidates(mesh, np.array(row['sourceToPrint'], float), extras, limit):
        x = metrics(mesh, c['sourceToPrint'], holes, intended)
        out.append(dict(name=c['name'], origin=c['origin'], sameAs=c['sameAs'], sourceToPrint=c['sourceToPrint'].tolist(), **x))
    ranked = sorted(out, key=lambda r: r['penalty'])
    for i, r in enumerate(ranked, 1):
        r['rank'] = i
    cur = next(r for r in out if r['name'] == 'current')
    best_alt = next((r for r in ranked if r['name'] != 'current'), None)
    flags = []
    if cur['tipMarginMm'] is None or cur['tipMarginMm'] <= 0:
        flags.append('current pose is not self-standing on its contact')
    if cur['contactMm2'] <= 0:
        flags.append('current pose has no flat contact within 0.01 mm')
    if cur['holeEndsFacingDown']:
        flags.append(f"{cur['holeEndsFacingDown']} hole end(s) face down above the bed")
    if cur['bossFacesFacingDown']:
        flags.append(f"{cur['bossFacesFacingDown']} boss/pin face(s) face down above the bed")
    if cur['holesHorizontal']:
        flags.append(f"{cur['holesHorizontal']} hole(s) print with a horizontal axis")
    if best_alt and best_alt['penalty'] < cur['penalty'] - 5:
        flags.append(f"alternative '{best_alt['name']}' screens better by {cur['penalty'] - best_alt['penalty']:.1f}; slice both")
    summary = dict(holes=sum(h['kind'] == 'hole' for h in holes), bosses=sum(h['kind'] == 'boss' for h in holes),
                   holeDiametersMm=sorted(round(2 * h['r'], 2) for h in holes if h['kind'] == 'hole'),
                   volumeMm3=round(float(mesh.volume), 1), triangles=len(mesh.faces))
    del mesh
    return dict(key=row['key'], color=row.get('color'), source=str(path), features=summary, currentRank=cur['rank'],
                bestAlternative=best_alt['name'] if best_alt else None, flags=flags, candidates=out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--key', action='append', help='Only these part keys (repeatable)')
    ap.add_argument('--extra-poses', type=Path)
    ap.add_argument('--max-generated', type=int, default=8)
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    manifest = json.loads(args.manifest.read_text())
    extras = json.loads(args.extra_poses.read_text()) if args.extra_poses else {}
    report = dict(manifest=str(args.manifest.resolve()), manifestSha256=sha256(args.manifest), penaltyWeights=PENALTY,
                  thresholds=dict(contactTolMm=CONTACT_TOL, sectionZMm=SECTION_Z, verticalDeg=VERTICAL_DEG,
                                  horizontalDeg=HORIZONTAL_DEG, seatDownNormalZ=SEAT_DOWN_NZ), parts=[])
    for row in manifest['parts']:
        if args.key and row['key'] not in args.key:
            continue
        r = check_part(row, extras.get(row['key'], []), args.max_generated)
        report['parts'].append(r)
        cur = next(c for c in r['candidates'] if c['name'] == 'current')
        best = next((c for c in r['candidates'] if c['name'] == r['bestAlternative']), None)
        line = (f"{r['key']}: current h={cur['heightMm']} contact={cur['contactMm2']} tip={cur['tipAngleDeg']}deg "
                f"ov60={cur['overhang60Mm2']} holes V/I/H={cur['holesVertical']}/{cur['holesInclined']}/{cur['holesHorizontal']} "
                f"seatsDown={cur['holeEndsFacingDown']} bossDown={cur['bossFacesFacingDown']} penalty={cur['penalty']} rank={r['currentRank']}/{len(r['candidates'])}")
        if best:
            line += (f" | best alt '{best['name']}' h={best['heightMm']} contact={best['contactMm2']} ov60={best['overhang60Mm2']} "
                     f"holes V/I/H={best['holesVertical']}/{best['holesInclined']}/{best['holesHorizontal']} "
                     f"seatsDown={best['holeEndsFacingDown']} bossDown={best['bossFacesFacingDown']} penalty={best['penalty']}")
        print(line)
        for f in r['flags']:
            print('   flag:', f)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
