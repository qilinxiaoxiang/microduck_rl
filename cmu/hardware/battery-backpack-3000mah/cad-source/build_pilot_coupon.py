"""Pilot coupon: a small strip that measures a printer's pilot shrink against print_fit.PILOT. Rides on any plate.

Strip 6 mm tall (print Z), 7 mm deep, flat on the bed, top face up.
Left: vertical through pilots 1.5 to 2.2 mm in 0.1 steps (8 holes), labelled on the front face ("15" = 1.5).
Right: horizontal teardrop pilots 1.7 to 2.3 mm in 0.1 steps (7 holes), axis along Y through the 7 mm depth, tip to +Z,
labelled on the top face. Walls between holes are 2 mm at the largest hole (4.2 mm pitch).
Print it from one .3mf on each printer, drive an M2 x 8 machine screw and an M2 x 8 thread-forming screw into each row,
and write the smallest diameter each enters by hand driver into PRINT-FIT.md and print_fit.PRINTER_OFFSET.
Run: uv run -q --python 3.12 --with trimesh --with manifold3d --with shapely --with matplotlib python build_pilot_coupon.py
"""
import math
from pathlib import Path

import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.geometry import Polygon
from shapely.ops import unary_union

VDIAS = [round(1.5 + 0.1 * i, 1) for i in range(8)]
HDIAS = [round(1.7 + 0.1 * i, 1) for i in range(7)]
PITCH, H, D, END = 4.2, 6.0, 7.0, 2.1         # pitch, height (Z), depth (Y), end margin to first/last hole centre
HZ = 2.7                                       # horizontal hole centre height
TXT, ENGRAVE = 2.8, 0.4
L = 2 * END + PITCH * (len(VDIAS) + len(HDIAS) - 1)
FONT = FontProperties(family="DejaVu Sans", weight="bold")


def text_poly(s, cx, cy, size):
    tp = TextPath((0, 0), s, size=size, prop=FONT)
    polys = []
    for ring in tp.to_polygons():
        if len(ring) >= 3:
            polys.append(Polygon(ring).buffer(0))
    # even-odd: letters with holes ("0", "8") come as outer + inner rings
    shape = Polygon()
    for poly in sorted(polys, key=lambda q: -q.area):
        shape = shape.difference(poly) if shape.contains(poly) else shape.union(poly)
    minx, miny, maxx, maxy = shape.bounds
    return trimesh.transformations.translation_matrix((cx - (minx + maxx) / 2, cy - (miny + maxy) / 2, 0)), shape


def label(s, cx, cy, size):
    T, shape = text_poly(s, cx, cy, size)
    parts = [trimesh.creation.extrude_polygon(g, ENGRAVE + 0.1) for g in getattr(shape, "geoms", [shape])]
    m = trimesh.util.concatenate(parts)
    m.apply_transform(T)
    return m


def teardrop(d, x, z, length):
    r = d / 2
    pts = [(r * math.cos(math.radians(t)), r * math.sin(math.radians(t))) for t in range(-225, 46, 5)]
    pts.append((0.0, r * math.sqrt(2)))                                      # 45 deg tip
    prism = trimesh.creation.extrude_polygon(Polygon(pts), length)
    prism.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))  # tip -> +Z, extrusion -> -Y
    prism.apply_translation((x, length - 0.5, z))
    return prism


body = trimesh.creation.box((L, D, H))
body.apply_translation((L / 2, D / 2, H / 2))
cuts = []
for i, d in enumerate(VDIAS):
    x = END + PITCH * i
    c = trimesh.creation.cylinder(radius=d / 2, height=H + 2, sections=96)
    c.apply_translation((x, D / 2, H / 2))
    cuts.append(c)
    lab = label(f"{int(round(d * 10))}", x, 0, TXT)                    # text in XY, then stood up on the front face
    lab.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, (1, 0, 0)))   # v -> +Z, extrusion -> -Y
    lab.apply_translation((0, ENGRAVE, H / 2))
    cuts.append(lab)
for j, d in enumerate(HDIAS):
    x = END + PITCH * (len(VDIAS) + j)
    cuts.append(teardrop(d, x, HZ, D + 1.0))
    lab = label(f"{int(round(d * 10))}", x, D / 2, TXT)
    lab.apply_translation((0, 0, H - ENGRAVE))
    cuts.append(lab)
out = body.difference(trimesh.util.concatenate(cuts))
p = Path(__file__).with_name("pilot-coupon.stl")
out.export(p)
print(p, "watertight", out.is_watertight, "bounds", [[round(v, 2) for v in r] for r in out.bounds.tolist()])
print(f"volume {out.volume / 1000:.3f} cm3, PLA {out.volume / 1000 * 1.24:.2f} g")
