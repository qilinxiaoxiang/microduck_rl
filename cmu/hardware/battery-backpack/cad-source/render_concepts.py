"""Renders of the two battery backpack drafts on the robot (guide/content/render.py, vtk, light background, guide part
colours), with numbered labels on the plugs and parts, combined into one sheet per layout.
    python render_concepts.py            (CAD venv, bounded runner; after concepts.py)
Writes renders/<L>_*.png and renders/<L>-sheet.png."""
import sys, json
from pathlib import Path
import numpy as np, trimesh
from PIL import Image, ImageDraw, ImageFont
HERE = Path(__file__).resolve().parent; OUT = HERE / 'renders'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, '/Users/xiangzheng/code/robotics/microduck-sim/guide/content'); import render as GR
sys.path.insert(0, '/Users/xiangzheng/code/robotics/microduck-sim/guide/content/interference-audit'); import ia_lib
G = ia_lib.load_guide(); COLOR = {p['key']: p['color'] for p in G['parts']}
hexc = lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
SKIP = {'rear_frame', 'rear_lid', 'xl4015_top', 'xl4015_mid', 'pd5991'}
COL = dict(frame=[237, 128, 26], cap=[250, 163, 64], battery=[70, 72, 82], switch=[25, 25, 28], dfr1202=[25, 70, 170],
           dfr_pads=[212, 175, 55], plugs=[240, 200, 20])
SIZE = (1000, 860)
try:
    FONT = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', 22); FONT_S = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 19)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


def tube(pts, r=0.9, seg=12):
    pts = np.asarray(pts, float); keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > 0.05:
            keep.append(i)
    pts = pts[keep]
    t = np.gradient(pts, axis=0); t /= np.linalg.norm(t, axis=1)[:, None]
    n = np.cross(t[0], [0, 0, 1.0]) if abs(t[0][2]) < 0.9 else np.cross(t[0], [1.0, 0, 0]); n /= np.linalg.norm(n)
    V = []
    for i, p in enumerate(pts):
        n = n - (n @ t[i]) * t[i]; n /= np.linalg.norm(n); b = np.cross(t[i], n)
        for k in range(seg):
            a = 2 * np.pi * k / seg; V.append(p + r * (np.cos(a) * n + np.sin(a) * b))
    F = []
    for i in range(len(pts) - 1):
        for k in range(seg):
            a0, a1 = i * seg + k, i * seg + (k + 1) % seg; b0, b1 = a0 + seg, a1 + seg
            F += [[a0, b0, a1], [a1, b0, b1]]
    return np.array(V), np.array(F)


def offset_path(pts, d):
    """parallel copy of a polyline at distance d (second wire of a pair), offset in the plane normal to z where possible."""
    pts = np.asarray(pts, float); t = np.gradient(pts, axis=0); t /= np.linalg.norm(t, axis=1)[:, None] + 1e-12
    out = []
    for p, ti in zip(pts, t):
        ref = np.array([1.0, 0, 0]) if abs(ti[0]) < 0.9 else np.array([0, 0, 1.0])
        n = np.cross(ti, ref); n /= np.linalg.norm(n) + 1e-12; out.append(p + d * n)
    return np.array(out)


def part(v, f, color, module='x'):
    return dict(v=np.asarray(v, np.float32), f=np.asarray(f, np.int64), color=color, module=module)


def robot_scene():
    sc = {}; man = ia_lib.manifest()
    for k in man:
        if k in SKIP:
            continue
        v, f = ia_lib.load_solid(k); sc[k] = part(v, f, hexc(COLOR.get(k, '#b0b0b0')), man[k].get('module', 'x'))
    return sc


def add_layout(sc, name):
    d = HERE / name
    for k in ('frame', 'cap', 'battery', 'switch', 'dfr1202', 'dfr_pads', 'plugs'):
        m = trimesh.load(str(d / f'{k}-world.stl'), force='mesh'); sc[k] = part(m.vertices, m.faces, COL[k], 'bp' if k in ('frame', 'cap') else 'bpin')
    W = json.loads((d / 'wires.json').read_text())
    for k, w in W.items():
        p = np.array(w['points']); cr = [200, 30, 30]
        if w['kind'] in ('pair', 'robot'):
            v, f = tube(offset_path(p, 0.95), 0.9); sc[f'w_{k}_a'] = part(v, f, cr, 'wires')
            v, f = tube(offset_path(p, -0.95), 0.9); sc[f'w_{k}_b'] = part(v, f, [30, 30, 30], 'wires')
    return sc


def cam(target, direction, dist, up=(0, 0, 1)):
    d = np.asarray(direction, float); d /= np.linalg.norm(d); t = np.asarray(target, float)
    return [list(t + dist * d), list(t), list(up)]


def project(view, pts, size=SIZE, angle=30.0):
    """pixel coordinates of world points for a vtk camera (perspective, vertical view angle)."""
    pos, foc, up = (np.asarray(a, float) for a in view)
    f = foc - pos; f /= np.linalg.norm(f); r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    q = np.asarray(pts, float) - pos; z = q @ f; x = q @ r; y = q @ u
    h = np.tan(np.radians(angle) / 2) * z; W, H = size
    return np.c_[W / 2 + (x / h) * (H / 2), H / 2 - (y / h) * (H / 2)]


def annotate(png, view, labels, title, side='right', colx=None):
    """labels in one column beside the model (side 'right' or 'left'), sorted by height so leader lines do not cross."""
    im = Image.open(png).convert('RGB'); dr = ImageDraw.Draw(im); W, H = im.size
    pts = [(txt, project(view, [pt])[0]) for txt, pt in labels]; pts.sort(key=lambda t: t[1][1])
    colx = colx if colx is not None else (W - 24 if side == 'right' else 24); ys = []; y = 60
    for txt, px in pts:
        y = max(y, px[1]); ys.append(y); y += 40
    over = ys[-1] - (H - 30) if ys else 0
    if over > 0:
        ys = [v - over for v in ys]
    for (txt, px), ty in zip(pts, ys):
        bb = dr.textbbox((0, 0), txt, font=FONT_S); tw, th = bb[2] - bb[0], bb[3] - bb[1]; pad = 5
        x0 = colx - tw - 2 * pad if side == 'right' else colx
        x1 = x0 + tw + 2 * pad; edge = x0 if side == 'right' else x1
        dr.line([tuple(px), (edge, ty)], fill=(20, 20, 20), width=2)
        dr.ellipse([px[0] - 5, px[1] - 5, px[0] + 5, px[1] + 5], fill=(220, 30, 30), outline=(255, 255, 255))
        dr.rectangle([x0, ty - th / 2 - pad, x1, ty + th / 2 + pad + 3], fill=(255, 255, 240), outline=(40, 40, 40))
        dr.text((x0 + pad, ty - th / 2 - 1), txt, fill=(10, 10, 10), font=FONT_S)
    dr.rectangle([0, 0, W, 36], fill=(255, 255, 255)); dr.text((10, 6), title, fill=(0, 0, 0), font=FONT)
    im.save(png); return im


def sheet(ims, out, cols=2):
    w, h = ims[0].size; rows = (len(ims) + cols - 1) // cols
    S = Image.new('RGB', (w * cols, h * rows), (255, 255, 255))
    for i, im in enumerate(ims):
        S.paste(im, ((i % cols) * w, (i // cols) * h))
    S.save(out)


def shifted(sc, keys, dx):
    out = dict(sc)
    for k in keys:
        q = dict(sc[k]); q['v'] = sc[k]['v'] + np.array([dx, 0, 0], np.float32); out[k] = q
    return out


SHELLS = {'shell_left', 'shell_right', 'rear_pack', 'torso_base', 'ring10_torso', 'ring5_torso', 'clip_torso'}
BOX = {'frame', 'cap'}
INNER = {'battery', 'switch', 'dfr1202', 'dfr_pads', 'plugs'}

if __name__ == '__main__':
    C = json.loads((HERE / 'concepts.json').read_text())
    base = robot_scene(); done = {}
    for name in [a for a in sys.argv[1:]] or list(C):
        c = C[name]; sc = add_layout(dict(base), name); pl = c['plugs']; b = c['battery']; d = c['dfr1202']
        bc = (np.array(b['lo']) + np.array(b['hi'])) / 2; dc = (np.array(d['lo']) + np.array(d['hi'])) / 2
        j = {k.split(' ')[0]: np.array(vv['centre']) for k, vv in pl.items()}
        sw = np.array(c['switch']['rocker']) if 'rocker' in c['switch'] else None
        L1 = name == 'L1'; rocker = np.array([-54.3, 0, -34.0]) if L1 else np.array([-82.5, 0, -3.0])
        swc = np.array([-54.3, 0, -14.0]) if L1 else np.array([-62.0, 0, -3.0])
        ims = []

        def shot(tag, view, title, labels, side='right', colx=None, **kw):
            f = GR.render(sc_now, [view], str(OUT / f'{name}_{tag}.png'), size=SIZE, **kw)[0]
            ims.append(annotate(f, view, labels, f'{name}: {title}', side, colx))

        INNER_HIDE = {'wires', 'battery', 'dfr1202', 'dfr_pads', 'plugs'}
        LOW_SCREWS = [-45.0, 16.1, -36.5], [-45.0, -16.1, -36.5]
        sc_now = sc
        # 1 outside, rear 3/4
        v = cam([-45, 0, 5], (-1, -0.6, 0.45), 330)
        shot('look', v, f'"{c["title"]}", outside ({c["depthBehindFrontPlate"]} mm deep; bench box 20.1)',
             [('switch rocker', rocker)], side='left', hide=INNER_HIDE)
        # 2 L1 straight from below (rear at the top) / L2 from the side
        if L1:
            v = cam([-52, 0, -30], (0, 0, -1), 210, up=(-1, 0, 0))
            shot('low', v, 'from below (rear at the top): rocker in a floor pocket',
                 [('switch rocker (press with a finger from below)', rocker), ('lower fixing screw', LOW_SCREWS[0]),
                  ('lower fixing screw', LOW_SCREWS[1])], hide=INNER_HIDE | {'left-leg', 'right-leg'})
        else:
            v = cam([-50, 0, 5], (0, 1, 0.1), 300)
            shot('low', v, 'side: the rocker sticks out 8.5 mm behind the box', [('switch rocker', rocker)],
                 hide=INNER_HIDE | {'left-leg'})
        # 3 side cut-away, everything inside visible
        v = cam([-55, 0, 3], (0, 1, 0.05), 205)
        shot('side', v, 'side, see-through (robot front to the LEFT)',
             [('battery', bc + [0, 30, 0]), ('DFR1202 5 V board', dc + [0, 13, 0]), ('switch body', swc + [0, 12, 0]),
              ('plugs 1+2 (XT30, either side of the switch)', j['J1'] + [0, 0, -6]),
              ('swap plugs 3+4 (in the pack channel)', j['J3'] + [0, 0, 6])],
             colx=1000 - 12, hide={'left-leg', 'right-leg'}, ghost=SHELLS | BOX)
        # 4 straight from behind, see-through box
        v = cam([-55, 0, 3], (-1, 0, 0.12), 215)
        shot('rear', v, 'from behind, see-through box (robot right side on the right)',
             [('1 battery to switch (XT30)', j['J1'] + [0, 0, -8]), ('2 switch to 5 V feed (XT30)', j['J2'] + [0, 0, -8]),
              ('balance plug (JST-XH)', j['balance']), ('battery', bc + [0, 10, 5]), ('switch', swc)],
             ghost=BOX | SHELLS)
        # 5 swap plugs in the empty pack channel, backpack hidden
        v = cam([-42, 0, 15], (-0.8, 0.45, 0.8), 150)
        shot('swap', v, 'the two swap plugs in the pack channel (box hidden)',
             [('3 servo swap plug (XT30)', j['J3'] + [0, 0, 13]), ('4 A7Z 5 V swap plug (MR30)', j['J4'] + [0, 0, 11.4]),
              ('robot lead up to OpenRB J4', [-38.0, -2.0, 30.0]), ('robot lead up to the neck', [-42.0, 8.0, 32.0])],
             hide={'battery', 'cap', 'frame', 'dfr1202', 'dfr_pads', 'switch'}, ghost=SHELLS)
        # 6 battery out: cap and battery pulled back
        sc_now = shifted(sc, ['cap'], -30.0)
        sc_now = shifted(sc_now, ['battery'] + [k for k in sc if k.startswith('w_bat')], -55.0)
        v = cam([-75, 0, 10], (-0.9, -0.75, 0.55), 330)
        shot('out', v, 'charging: cap off, battery out the back',
             [('cap', [-100.0, -20, 30]), ('battery', bc + [-55, 0, 5])], side='left', hide={'wires'})
        sheet(ims, OUT / f'{name}-sheet.png'); done[name] = str(OUT / f'{name}-sheet.png')
    print(json.dumps(done, indent=1))
