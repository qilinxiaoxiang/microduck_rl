"""1:1 cutting template for a wrapped foot sole (inner tube or sheet rubber) from the printed foot STL.

The foot bottom (community-44.stl, left foot; the right foot community-46.stl is its mirror) is flat over about
39 x 27 mm and curves up 1 mm about 3.5 mm past the flat and 2 mm about 5 mm past it. The sheet must cover the
curve because walking puts most floor contacts there (sole_contact_sim.py), so the template is the surface up to
WRAP_H mm above the flat, developed along rays from the foot centre: for each direction the template radius is the
arc length along the surface from the centre to the point where the surface is WRAP_H above the flat. Corners get
a narrow V dart (the wrapped corner is doubly curved; the excess at 45 deg of normal tilt is only about 1 mm).

Prints at 100 % scale on US Letter; the 50 mm bar checks the printer did not scale the page.

  /opt/homebrew/bin/python3 sole_template.py   -> sole_template.pdf (+ .png preview)
"""
import struct
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = Path(__file__).resolve().parent
STL = HERE / ".." / ".." / "assembly" / "print-parts" / "community-44.stl"
WRAP_H = 2.5   # mm above the flat that the sheet reaches (sim contacts reach 1.6 mm)
FLAT_H = 0.1   # mm: below this the bottom counts as flat
DART_W = 2.0   # mm opening of each corner dart at the outer edge
STEP = 0.25    # mm height-map grid


def height_map():
    b = STL.read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    t = np.frombuffer(b[84:], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("x", "<u2")]), count=n)["v"]
    t = t.reshape(-1, 3, 3).astype(float)
    xs = np.arange(-27.5, 27.501, STEP); ys = np.arange(-21, 21.001, STEP)
    low = np.full((len(xs), len(ys)), np.nan)
    for tri in t:
        a, b_, c = tri
        ix = np.where((xs >= tri[:, 0].min()) & (xs <= tri[:, 0].max()))[0]
        iy = np.where((ys >= tri[:, 1].min()) & (ys <= tri[:, 1].max()))[0]
        if not len(ix) or not len(iy):
            continue
        X, Y = np.meshgrid(xs[ix], ys[iy], indexing="ij")
        v0 = b_[:2] - a[:2]; v1 = c[:2] - a[:2]; d = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(d) < 1e-9:
            continue
        px = X - a[0]; py = Y - a[1]
        u = (px * v1[1] - py * v1[0]) / d; w = (v0[0] * py - v0[1] * px) / d
        ins = (u >= -1e-6) & (w >= -1e-6) & (u + w <= 1 + 1e-6)
        z = a[2] + u * (b_[2] - a[2]) + w * (c[2] - a[2])
        sub = low[np.ix_(ix, iy)]; upd = ins & (np.isnan(sub) | (z < sub)); sub[upd] = z[upd]
        low[np.ix_(ix, iy)] = sub
    return xs, ys, low


def sample(xs, ys, low, x, y):
    i = (x - xs[0]) / STEP; j = (y - ys[0]) / STEP
    i0, j0 = int(np.floor(i)), int(np.floor(j))
    if i0 < 0 or j0 < 0 or i0 + 1 >= len(xs) or j0 + 1 >= len(ys):
        return np.nan
    fi, fj = i - i0, j - j0
    q = low[i0:i0 + 2, j0:j0 + 2]
    if np.isnan(q).any():
        return np.nan
    return (q[0, 0] * (1 - fi) * (1 - fj) + q[1, 0] * fi * (1 - fj) + q[0, 1] * (1 - fi) * fj + q[1, 1] * fi * fj)


def develop(xs, ys, low, n=720):
    """Return (angles, template radius, flat radius, planar radius) per ray from the foot centre."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rho = np.zeros(n); rflat = np.zeros(n); rplan = np.zeros(n)
    ds = 0.05
    for k, a in enumerate(ang):
        s = 0.0; r = 0.0; zprev = 0.0; rf = None
        while True:
            r += ds
            z = sample(xs, ys, low, r * np.cos(a), r * np.sin(a))
            if np.isnan(z):
                rho[k] = s; rflat[k] = rf if rf is not None else r; rplan[k] = r
                break
            s += np.hypot(ds, z - zprev); zprev = z
            if rf is None and z >= FLAT_H:
                rf = s
            if z >= WRAP_H:
                rho[k] = s; rflat[k] = rf if rf is not None else s; rplan[k] = r
                break
    return ang, rho, rflat, rplan


def outline_with_darts(ang, rho, rflat):
    pts = []
    # dart directions: the four corners of the flat region (largest flat radius in each quadrant)
    corners = []
    for qlo in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
        sel = (ang >= qlo) & (ang < qlo + np.pi / 2)
        corners.append(ang[sel][np.argmax(rflat[sel])])
    for k, a in enumerate(ang):
        r = rho[k]
        for c in corners:
            da = (a - c + np.pi) % (2 * np.pi) - np.pi
            half = DART_W / 2 / r  # rad, half opening at the outer edge
            if abs(da) < half:
                # inside the dart: go down to the flat radius linearly (V shape)
                r = rflat[k] + (rho[k] - rflat[k]) * abs(da) / half
        pts.append((r * np.cos(a), r * np.sin(a)))
    return np.array(pts), corners


def main():
    xs, ys, low = height_map()
    ang, rho, rflat, rplan = develop(xs, ys, low)
    poly, corners = outline_with_darts(ang, rho, rflat)
    flat = np.array([(r * np.cos(a), r * np.sin(a)) for a, r in zip(ang, rflat)])
    ext = dict(x=(poly[:, 0].min(), poly[:, 0].max()), y=(poly[:, 1].min(), poly[:, 1].max()))
    print("template extent mm: x %.1f..%.1f  y %.1f..%.1f ; flat x %.1f..%.1f y %.1f..%.1f" % (
        *ext["x"], *ext["y"], flat[:, 0].min(), flat[:, 0].max(), flat[:, 1].min(), flat[:, 1].max()))
    print("planar radius at wrap height vs developed: mean extra %.2f mm" % np.mean(rho - rplan))

    # US Letter, 1 mm data = 1 mm paper
    W_IN, H_IN = 8.5, 11.0
    fig = plt.figure(figsize=(W_IN, H_IN))
    w_mm, h_mm = W_IN * 25.4, H_IN * 25.4
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, w_mm); ax.set_ylim(0, h_mm); ax.set_aspect("equal"); ax.axis("off")

    def draw(cx, cy, mirror, label):
        P = poly.copy(); F = flat.copy()
        if mirror:
            P[:, 1] *= -1; F[:, 1] *= -1
        ax.add_patch(Polygon(P + (cx, cy), closed=True, fill=False, lw=0.6, ec="black"))
        ax.add_patch(Polygon(F + (cx, cy), closed=True, fill=False, lw=0.4, ec="0.5", ls=(0, (2, 2))))
        ax.plot([cx - 3, cx + 3], [cy, cy], lw=0.3, c="0.5"); ax.plot([cx, cx], [cy - 3, cy + 3], lw=0.3, c="0.5")
        ax.text(cx, cy + 6, label, ha="center", va="center", fontsize=7)
        ax.text(cx, cy - 6, "front-back axis is horizontal", ha="center", va="center", fontsize=5, color="0.4")

    # two per foot (one spare each)
    for i, (cx, cy) in enumerate([(70, 220), (150, 220), (70, 160), (150, 160)]):
        mirror = i % 2 == 1
        draw(cx, cy, mirror, "LEFT foot (community-44)" if not mirror else "RIGHT foot (community-46)")

    # scale bar
    ax.plot([30, 80], [120, 120], lw=1.2, c="black"); ax.plot([30, 30], [118, 122], lw=1.2, c="black"); ax.plot([80, 80], [118, 122], lw=1.2, c="black")
    ax.text(55, 124, "50 mm  (check with a ruler; print at 100 % / actual size, no fit-to-page)", ha="center", fontsize=7)

    lines = [
        "MicroDuck foot sole template, wrapped sheet (inner tube 1 mm or sheet rubber up to 1 mm)",
        "Source: community-44.stl bottom height map; outer line = surface developed to %.1f mm above the flat;" % WRAP_H,
        "dashed line = flat region (%.0f x %.0f mm). Cut on the outer line. Corner darts are %.0f mm wide." % (
            flat[:, 0].max() - flat[:, 0].min(), flat[:, 1].max() - flat[:, 1].min(), DART_W),
        "",
        "1. Print this page at 100 %. Check the 50 mm bar.",
        "2. Cut a template, tape it to the rubber (butyl inner tube: slit open, lay flat, wash the talc off, dry).",
        "3. Cut the rubber on the outer line and the four darts. Rough the PLA bottom with the sandpaper, wipe with alcohol.",
        "4. Stick thin double-sided tape (<= 0.2 mm, PET carrier or transfer tape, not foam) on the rubber first, trim to the rubber.",
        "5. Peel, centre on the flat (dashed line matches the flat), press the middle down, then press the edges over the",
        "   curve from the centre outward so the dart edges meet. Press hard for a minute; let it sit before walking.",
        "Stack thickness tape + rubber: see README (sim sweep) for the limit.",
    ]
    for k, s in enumerate(lines):
        ax.text(15, 100 - 5 * k, s, fontsize=6.5, family="DejaVu Sans", va="top")
    out = HERE / "sole_template.pdf"
    fig.savefig(out, format="pdf")
    fig.savefig(HERE / "sole_template.png", dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
