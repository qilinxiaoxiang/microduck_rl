"""Battery backpack (DECISIONS.md D-010) layout drafts for Shawn's pick, 2026-10-15. Renders and numbers only: nothing here
is released, gated or final.

Both drafts replace the layout F bench box on the same front plate (the V1.8 cover outline with its four M2 x 6 holes at
(+-16.1, 26.0) and (+-16.1, -36.5), x -47..-45) and keep its width (|y| <= 32.05), top (z 43) and rounded rear edges.
Contents (sources in README.md): Tattu 2S 650 (listing 57 x 31 x 12, XT30 on a 45 mm 16 AWG lead, JST-XH balance lead),
REV-31-1387 switch (REV STEP model: 30.2 x 13.6 flange, rocker 8.5 mm proud of the panel, snap wings 32.3 wide behind
a 3 mm panel, 29.6 mm from the panel face to the lead end), DFRobot DFR1202 (30 x 26 x 8, pads on a 26 mm edge).
The swap joints (robot XT30 male servo inlet, robot MR30 male A7Z lead) sit in the rear pack's empty channel behind the
OpenRB (x -45..-36.7, |y| <= 13.6, free from z -6 to 31 by probe_channel.py), reached by pulling the backpack back.

L1 "switch underneath": the bench-box silhouette 3.5 mm deeper; battery behind the rear cap; DFR standing against the
     front plate above the switch hub; switch upright in the lower tier, rocker in a recess in its floor.
L2 "switch on the back": no lower tier, 7.5 mm deeper; rocker on the rear face; battery top rear, DFR top front.
World = guide frame, mm (x forward, y left, z up). python concepts.py (CAD venv, bounded runner) writes <L>/*.stl and
concepts.json."""
import sys, json, math
from pathlib import Path
import numpy as np, trimesh
from shapely.geometry import Polygon, box as sbox, Point
from shapely.ops import unary_union
HERE = Path(__file__).resolve().parent; BP = HERE.parent; PRINT = BP.parent
sys.path.insert(0, str(BP / 'layout-D-20261014')); sys.path.insert(0, str(PRINT / 'rear-pack'))
import pack_lib as P, build_rear_pack as BRP
import shellkit as K
REV = Path('parts/rev_1.stl')  # cmu/ copy: the REV-31-1387 STEP export lived in a scratch directory; parts/rev_31_1387_switch-local.stl is its cached copy
REV_LOCAL = HERE / 'parts' / 'rev_31_1387_switch-local.stl'

T = 1.2; HW = 32.05; Z1 = 43.0; X_IN = -47.0; X_OUT = -45.0; Z_STEP_OUT = -15.2
BAT = (57.0, 31.0, 12.0)                       # listing: length (y), width (z), thickness (x)
DFR = (26.0, 30.0, 8.0)                        # y (pad edge), z, x
XT30_HALF = (16.0, 10.2, 5.2); MR30_HALF = (14.4, 10.3, 4.9)   # length, width, height (Amass listings; XT30 approximate)
REV_PANEL = 3.0; REV_FRONT_Y = 1.5            # switch local: panel front face at y 1.5 (REV bracket), rocker +y


def cover_plate():
    """V1.8 cover outline and holes (build_rear_pack.build_cover) without the DFR hold-down lip."""
    c = BRP.COVER; hw, lw, lz, ch = c['half_w'], c['low_half_w'], c['low_z'], BRP.POST['chamfer']
    base = Polygon([(-lw, c['z0']), (lw, c['z0']), (lw, lz), (hw, lz + ch), (hw, c['z1']), (-hw, c['z1']), (-hw, lz + ch), (-lw, lz)])
    top_round = unary_union([BRP.rounded(sbox(-hw, c['z1'] - 8.0, hw, c['z1']), 1.5), sbox(-hw - 1, c['z0'] - 1, hw + 1, c['z1'] - 4.0)])
    plate = P.extrude_yz(base.intersection(top_round).buffer(0).simplify(0.02), c['x0'], c['x1'])
    for (y, z) in BRP.POST['holes']:
        plate = plate - P.cyl_x(c['hole_r'], c['x0'] - 1, c['x1'] + 1, y, z)
    return plate


def outline_poly(ol):
    def rr(r):
        p = sbox(r.y0, r.z0, r.y1, r.z1); mid = 0.5 * (r.z0 + r.z1)
        top = p.intersection(sbox(r.y0 - 1, mid, r.y1 + 1, r.z1 + 1)); bot = p.difference(top)
        rnd = lambda q, rc: q.buffer(-rc, join_style=1).buffer(rc, join_style=1) if rc > 0 else q
        return unary_union([rnd(p, r.rc_top).intersection(top.buffer(0.01)), rnd(p, r.rc_bot).intersection(bot.buffer(0.01))])
    u = rr(ol.W) if ol.N is None else unary_union([rr(ol.W), rr(ol.N)])
    if ol.rf > 0:
        u = u.buffer(ol.rf, join_style=1).buffer(-ol.rf, join_style=1)
    return u.simplify(0.02)


def rev_local():
    """REV switch body in its STEP frame (mm): y = through-panel axis, rocker at +y, bracket front face y 1.5."""
    REV_LOCAL.parent.mkdir(exist_ok=True)
    if not REV_LOCAL.exists():
        trimesh.load(str(REV), force='mesh').export(REV_LOCAL)
    return trimesh.load(str(REV_LOCAL), force='mesh')


def place(m, R, t):
    out = m.copy(); out.vertices = out.vertices @ np.asarray(R, float).T + np.asarray(t, float); return out


def boxmesh(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def plug(a, axis, half, n_halves=2):
    """mated plug pair starting at point a along unit axis (two halves overlapping 6 mm), as a box mesh list."""
    L = half[0] * n_halves - (6.0 if n_halves == 2 else 0.0)
    ax = np.asarray(axis, float); ax /= np.linalg.norm(ax)
    c = np.asarray(a, float) + ax * L / 2
    ext = np.abs(ax) * L + (1 - np.abs(ax)) * 0.0
    # cross-section: width along the larger remaining world axis given by caller order (y if axis is z or x)
    return c, L


def plugbox(centre, axis, half, width_axis, n_halves=2):
    L = half[0] * n_halves - (6.0 if n_halves == 2 else 0.0)
    e = np.zeros(3); e[axis] = L; e[width_axis] = half[1]; e[3 - axis - width_axis] = half[2]
    c = np.asarray(centre, float)
    return boxmesh(c - e / 2, c + e / 2)


# ------------------------------------------------------------------ layouts
LAYOUTS = {
    'L1': dict(title='switch underneath', X_LID=-71.0, N=dict(hy=17.7, z0=-34.0), R=dict(side=6.5, top=6.5, step=4.3, nside=4.5, nbot=6.0),
               rc=dict(top=1.8, step=1.8, nbot=2.1, rf=3.0)),
    'L2': dict(title='switch on the back', X_LID=-74.6, N=None, R=dict(side=7.4, top=7.8, step=6.0), rc=dict(top=1.8, step=1.8)),
}


def outline(L):
    W = K.Rect(-HW, HW, Z_STEP_OUT, Z1, L['rc']['top'], L['rc']['step'], 'side', 'top', 'step')
    if L['N']:
        n = L['N']; N = K.Rect(-n['hy'], n['hy'], n['z0'], Z_STEP_OUT + 8.0, 0.0, L['rc']['nbot'], 'nside', 'top', 'nbot')
        return K.Outline(W=W, N=N, rf=L['rc']['rf'], R=dict(L['R']), X_LID=L['X_LID'], X_IN=X_IN, T=T)
    return K.Outline(W=W, R=dict(L['R']), X_LID=L['X_LID'], X_IN=X_IN, T=T)


def fit_battery(ol, x_front, z_top, y_half=BAT[0] / 2, clr=0.3):
    """highest battery box (thickness along x from x_front back, width along z down from z_top) that clears the inner
    cavity (rounded rear edges) by clr; returns (lo, hi)."""
    for zt in np.arange(z_top, z_top - 8.0, -0.1):
        lo = np.array([x_front - BAT[2], -y_half, zt - BAT[1]]); hi = np.array([x_front, y_half, zt])
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in np.linspace(lo[1], hi[1], 9) for z in np.linspace(lo[2], hi[2], 9)])
        if (ol.inner_np(c) <= -clr).all():
            return lo, hi
    raise RuntimeError('battery does not fit')


def build(name):
    L = LAYOUTS[name]; ol = outline(L); out = dict(title=L['title'], X_LID=L['X_LID'])
    rmax = max(L['R'].values()); X_SPLIT = round(math.ceil((L['X_LID'] + rmax) * 20) / 20, 2)
    outer, inner = ol.solids(0.3, x_shift=0.15)
    poly = outline_poly(ol)
    shell = (outer - inner)
    web = P.extrude_yz(poly.intersection(sbox(-HW - 1, -80, HW + 1, -2.0)), X_IN, X_OUT)
    frame_shell = shell ^ P.box((X_SPLIT, -60, -60), (X_IN + 0.5, 60, 60))
    cap = shell ^ P.box((L['X_LID'] - 1, -60, -60), (X_SPLIT, 60, 60))
    frame = P.union([frame_shell, cover_plate(), web])
    cuts = [P.cyl_x(BRP.COVER['hole_r'], X_IN - 1, X_OUT + 1, y, z) for (y, z) in BRP.POST['holes']]
    comp = {}; wires = {}
    sw = rev_local()
    if name == 'L1':
        # switch: axis local y -> world -z (rocker down), local x (27.5 / 32.3 wings) -> world y, local z (10.6) -> world x
        R = [[0, 0, 1], [1, 0, 0], [0, -1, 0]]        # world = R @ local: x_w = z_l, y_w = x_l, z_w = -y_l
        z_pf = ol.N.z0 + 8.49                          # panel front (lower) face: rocker tip flush with the floor
        xc = -54.3                                     # flange centre x (pocket x -47.2 .. -61.4)
        zc_l = 0.5 * (-12.21 + 1.39)
        t = [xc - zc_l, 0.15, z_pf + REV_FRONT_Y]
        comp['switch'] = place(sw, R, t)
        sb = comp['switch'].bounds
        pocket = P.box((xc - 7.1, -15.6, ol.N.z0 - 1), (xc + 7.1, 15.6, z_pf))
        cut_hole = P.box((xc - 5.3, -13.75, z_pf - 0.5), (xc + 5.3, 13.75, z_pf + REV_PANEL + 0.5))
        panel = P.box((xc - 8.3, -16.8, z_pf), (xc + 8.3, 16.8, z_pf + REV_PANEL)) - cut_hole
        pocket_walls = P.box((xc - 8.3, -16.8, ol.N.z0), (xc + 8.3, 16.8, z_pf + 0.01)) - pocket
        frame = P.union([frame, panel ^ outer, pocket_walls ^ outer])
        cuts += [pocket, cut_hole]
        out['switch'] = dict(rockerTipZ=round(float(sb[0][2]), 2), bodyTopZ=round(float(sb[1][2]), 2), panelZ=[round(z_pf, 2), round(z_pf + REV_PANEL, 2)],
                             pocket=dict(x=[xc - 7.1, xc + 7.1], absY=15.6, z=[ol.N.z0, round(z_pf, 2)]))
        # DFR standing against the front plate, pads on the lower 26 mm edge
        dlo = np.array([X_IN - 0.4 - DFR[2], -DFR[0] / 2, 11.0]); dhi = np.array([X_IN - 0.4, DFR[0] / 2, 11.0 + DFR[1]])
        bat_front = dlo[0] - 0.6
    else:
        # L2: switch axis local y -> world -x (rocker to the back), wings (local x) along y, local z along z
        R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]        # x_w = -y_l, y_w = x_l, z_w = z_l
        x_pf = L['X_LID'] + T - REV_PANEL               # panel = rear wall thickened to 3 mm locally
        zc = -3.0; zc_l = 0.5 * (-12.21 + 1.39)
        t = [x_pf + REV_FRONT_Y + REV_PANEL, 0.15, zc - zc_l]
        t[0] = L['X_LID'] + REV_FRONT_Y                 # flange on the outer rear face
        comp['switch'] = place(sw, R, t)
        sb = comp['switch'].bounds
        cut_hole = P.box((L['X_LID'] - 1, -13.75, zc - 5.3), (L['X_LID'] + REV_PANEL + 0.5, 13.75, zc + 5.3))
        boss = P.box((L['X_LID'] + 0.5, -16.6, zc - 7.5), (L['X_LID'] + REV_PANEL, 16.6, zc + 7.5))
        cap = P.union([cap, boss ^ outer]) - cut_hole
        front_win = P.box((X_IN - 1, -16.8, zc - 6.0), (X_OUT + 1, 16.8, zc + 6.0))
        cuts += [front_win]
        out['switch'] = dict(rockerRearX=round(float(sb[0][0]), 2), bodyFrontX=round(float(sb[1][0]), 2), z=[round(float(sb[0][2]), 2), round(float(sb[1][2]), 2)])
        dlo = np.array([X_IN - 0.4 - DFR[2], -DFR[0] / 2, 11.0]); dhi = np.array([X_IN - 0.4, DFR[0] / 2, 11.0 + DFR[1]])
        bat_front = dlo[0] - 0.6
    comp['dfr1202'] = boxmesh(dlo, dhi)
    pads = []
    for c_, w_ in ((2.18, 2.47), (13.13, 4.49), (23.69, 3.48)):     # VIN, GND, VOUT from the +y end (dfr-pads.json)
        y = DFR[0] / 2 - c_; pads.append(boxmesh([dlo[0] - 0.01, y - w_ / 2, dlo[2] - 0.3], [dlo[0] + 1.8, y + w_ / 2, dlo[2] + 1.8]))
    comp['dfr_pads'] = trimesh.util.concatenate(pads)
    blo, bhi = fit_battery(ol, bat_front, Z1 - T - 0.3)
    comp['battery'] = boxmesh(blo, bhi)
    out['battery'] = dict(lo=np.round(blo, 2).tolist(), hi=np.round(bhi, 2).tolist())
    out['dfr1202'] = dict(lo=np.round(dlo, 2).tolist(), hi=np.round(dhi, 2).tolist())
    # front-plate window for the backpack's two swap leads into the pack channel
    win = P.box((X_IN - 1, -12.0, 1.5), (X_OUT + 1, 12.0, 9.5)) if name == 'L1' else P.box((X_IN - 1, -12.0, 4.0), (X_OUT + 1, 12.0, 9.5))
    cuts.append(win)
    frame = frame - P.union(cuts)
    # ---------------- plugs (rendered as yellow blocks) and wires (rough centrelines)
    bz = 0.5 * (blo[2] + bhi[2]); bx = 0.5 * (blo[0] + bhi[0])
    sw_top = sb[1][2] if name == 'L1' else None
    J = {}
    J['J1 battery-switch XT30'] = plugbox([bx + 1.0, -25.0, 1.0], 2, XT30_HALF, 0) if name == 'L1' else plugbox([bx, -24.5, -3.0], 2, XT30_HALF, 0)
    J['J2 switch-feed XT30'] = plugbox([bx + 1.0, 25.0, 1.0], 2, XT30_HALF, 0) if name == 'L1' else plugbox([bx, 24.5, -3.0], 2, XT30_HALF, 0)
    J['J3 servo swap XT30 (robot inlet)'] = plugbox([-41.6, -6.5, 17.0], 2, XT30_HALF, 1)
    J['J4 A7Z swap MR30 (robot lead)'] = plugbox([-41.6, 6.5, 17.0], 2, MR30_HALF, 1)
    J['balance plug JST-XH'] = boxmesh([bx - 3, -29.5, blo[2] - 9.0], [bx + 3, -21.5, blo[2] - 4.0]) if name == 'L1' else boxmesh([bx - 3, -29.5, blo[2] - 7.0], [bx + 3, -21.5, blo[2] - 2.0])
    comp['plugs'] = trimesh.util.concatenate(list(J.values()))
    out['plugs'] = {k: dict(centre=np.round(m.bounds.mean(0), 1).tolist(), size=np.round(m.extents, 1).tolist()) for k, m in J.items()}
    j1, j2, j3, j4 = [m.bounds for m in list(J.values())[:4]]
    pin, pgnd, pout = DFR[0] / 2 - 2.18, DFR[0] / 2 - 13.13, DFR[0] / 2 - 23.69
    dpx = dlo[0] + 0.9; dpz = dlo[2] - 0.2
    if name == 'L1':
        top = sb[1][2]; sx = 0.5 * (sb[0][0] + sb[1][0])
        wires['battery lead (16 AWG)'] = ([[bx, -BAT[0] / 2, bz], [bx, -BAT[0] / 2 - 1.5, bz], [bx, -26.0, blo[2] - 3], [bx + 1.0, -25.0, j1[1][2]]], 'pair')
        wires['switch lead to J1'] = ([[sx, -5.0, top], [sx, -5.0, top + 3.0], [sx - 3, -22.0, top + 2.0], [bx + 1.0, -25.0, j1[0][2]]], 'pair')
        wires['switch lead to J2'] = ([[sx, -2.0, top], [sx, -2.0, top + 4.5], [sx - 3, 22.0, top + 3.5], [bx + 1.0, 25.0, j2[0][2]]], 'pair')
        wires['feed lead to VIN/GND'] = ([[bx + 1.0, 25.0, j2[1][2]], [bx + 1.0, 25.0, 8.0], [dpx, 14.0, 7.0], [dpx, pin, dpz]], 'pair')
        wires['servo lead to J3'] = ([[dpx, pin - 1.0, dpz], [dpx, pin - 2.0, 5.0], [-46.0, -6.5, 5.0], [-41.6, -6.5, j3[0][2]]], 'pair')
        wires['5 V lead to J4'] = ([[dpx, pout, dpz], [dpx, pout, 6.5], [-46.0, 6.5, 6.5], [-41.6, 6.5, j4[0][2]]], 'pair')
    else:
        sx1 = sb[1][0]; zc = 0.5 * (sb[0][2] + sb[1][2])
        wires['battery lead (16 AWG)'] = ([[bx, -BAT[0] / 2, bz], [bx, -BAT[0] / 2 - 1.5, bz], [bx, -25.5, blo[2] - 2], [bx, -24.5, j1[1][2]]], 'pair')
        wires['switch lead to J1'] = ([[sx1, -5.0, zc], [-42.0, -5.0, zc], [-42.0, -5.0, 2.0], [-46.0, -16.0, 1.0], [bx, -24.5, j1[0][2]]], 'pair')
        wires['switch lead to J2'] = ([[sx1, -2.0, zc + 1], [-40.5, -2.0, zc + 1], [-40.5, -2.0, 0.0], [-46.0, 16.0, 0.0], [bx, 24.5, j2[0][2]]], 'pair')
        wires['feed lead to VIN/GND'] = ([[bx, 24.5, j2[1][2]], [bx, 24.5, 8.0], [dpx, 14.0, 7.5], [dpx, pin, dpz]], 'pair')
        wires['servo lead to J3'] = ([[dpx, pin - 1.0, dpz], [dpx, pin - 2.0, 6.5], [-46.0, -6.5, 6.5], [-41.6, -6.5, j3[0][2]]], 'pair')
        wires['5 V lead to J4'] = ([[dpx, pout, dpz], [dpx, pout, 7.5], [-46.0, 6.5, 7.5], [-41.6, 6.5, j4[0][2]]], 'pair')
    # robot side (same for both backpacks): servo inlet from OpenRB J4, A7Z lead from the head notch
    wires['robot servo inlet (from J4)'] = ([[-31.5, 0.0, 25.4], [-35.0, 0.0, 28.0], [-41.0, -4.0, 29.0], [-41.6, -6.5, j3[1][2]]], 'robot')
    wires['robot A7Z lead (from the neck)'] = ([[-20.0, 2.0, 36.0], [-33.0, 4.0, 30.5], [-41.0, 5.5, 29.5], [-41.6, 6.5, j4[1][2]]], 'robot')
    out['bounds'] = dict(frame=np.round(P.from_m(frame).bounds, 2).tolist(), cap=np.round(P.from_m(cap).bounds, 2).tolist())
    out['X_SPLIT'] = X_SPLIT; out['depthBehindFrontPlate'] = round(X_IN - L['X_LID'], 2)
    out['volumes'] = dict(frame=round(frame.volume(), 1), cap=round(cap.volume(), 1))
    d = HERE / name; d.mkdir(exist_ok=True)
    for nm, man in (('frame', frame), ('cap', cap)):
        m = P.from_m(man); m.merge_vertices(); m.export(d / f'{nm}-world.stl')
    for k, m in comp.items():
        m.export(d / f'{k}-world.stl')
    paths = {}
    for k, (wp, kind) in wires.items():
        pts = K.fillet_path([tuple(map(float, p)) for p in wp], 2.0, n=6)
        paths[k] = dict(points=np.round(np.asarray(pts), 3).tolist(), kind=kind, lengthMm=round(K.path_length(pts), 1))
    json.dump(paths, open(d / 'wires.json', 'w'), indent=0)
    out['wires'] = {k: v['lengthMm'] for k, v in paths.items()}
    return out


if __name__ == '__main__':
    res = {k: build(k) for k in (sys.argv[1:] or LAYOUTS)}
    P.jdump(res, HERE / 'concepts.json')
    print(json.dumps(res, indent=1))
