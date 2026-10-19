"""MicroDuck battery backpack (DECISIONS.md D-010), print geometry, 2026-10-15. World = guide frame, mm (x forward, y left,
z up). The backpack replaces the layout F bench box on the same V1.8 cover outline and its four M2 x 6 holes at (+-16.1, 26.0)
and (+-16.1, -36.5); it keeps the bench box width (|y| <= 32.05), top (z 43), straight step (z -15.2) and narrow lower tier
(|y| <= 17.7), and puts all rear rounding into the cap.

Why the battery stands upright: the Tattu leads leave the pack's 31 x 12 end. Lying across the box (57 mm along y in a
61.7 mm cavity) leaves 2.4 mm at that end, too little to bend a 16 AWG lead. Upright (57 along z) the leads leave the top
into a 6 mm band under the top wall; the lower end drops into the narrow tier.
Contents (sources in README.md): Tattu 2S 650 (listing 57 x 31 x 12; pocket 58 x 32 x 13 until the pack is measured),
Swann 39 rocker switch (the one in hand, 2026-10-17: 30 x 13.8 flange, 27.3-27.4 x 12.2-12.3 cutout as Swann's 40
family; geometry envelope from the REV-31-1387 STEP: same flange, rocker 8.49 mm proud, body deeper than the Swann's),
DFRobot DFR1202 (30 x 26 x 8, power pads VIN/GND/VOUT on one 26 mm edge, EN/AGND/PGOOD on the other, no mounting holes).

Parts and how each is held (plain words in README.md):
- switch: rocker down in a pocket in the narrow-tier floor, flush with the floor; body up through a 27.5 x 12.3 hole in a
  3 mm panel; the switch's snap wings grip the panel.
- battery: upright behind the switch, standing on the panel and a shelf; the cap's two fences keep it centred; a foam pad
  on the cap presses it forward.
- DFR1202: power-pad edge down, back 2.2 mm off the cover on two standoff ribs, clipped by two flexible lips on its two
  pad-free 30 mm edges, one corner stop under the VIN end; a backing plate above the cover top (z 29.5) carries its upper
  part. A 2 mm foam pad on its inductor rests against the battery.
- cap: three M2 x 8 from driver holes in the cap back down columns inside the cap into frame bosses (two upper side, one
  bottom centre).
- leads: battery XT30 (J1) and switch-to-feed XT30 (J2) float in the side spaces beside the battery; the servo and 5 V
  leads leave through a window under the DFR into the rear pack's empty channel, where the two swap plugs hang.
    python build_bp.py          (CAD venv, bounded runner) writes build/*.stl, build/wires.json and build/build.json."""
import sys, json, math
from pathlib import Path
import numpy as np, trimesh, manifold3d as m3
from shapely.geometry import box as sbox, Polygon
HERE = Path(__file__).resolve().parent; OUT = HERE / 'build'; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import concepts as C
P = C.P; K = C.K; BRP = C.BRP

T = 1.2; HW = 32.05; Z1 = 43.0; X_IN = -47.0; X_OUT = -45.0; Z_STEP = -15.2; ZI1 = Z1 - T
N_HY = 17.7; N_Z0 = -34.0
R_EDGE = dict(side=7.4, top=6.5, step=4.3, nside=3.5, nbot=5.0)      # rear-edge radii (cap); nside 3.5 keeps the battery's
                                                                      # lower rear corners 0.3 inside the narrow tier
RC = dict(top=1.8, step=1.8, nbot=2.1, rf=3.0)                        # y-z outline corner radii (as layout F)
BAT_LIST = (57.0, 31.0, 12.0)                                         # Tattu listing: length (z), width (y), thickness (x)
BAT_POCKET = (58.0, 32.0, 13.0)                                       # listing + clearance until the measured pack
FOAM_GAP = 1.0                                                        # behind the listing pack: the cap's foam pad
SW_POCKET_HALF = 7.2                                                  # Swann 39 flange 13.8 / 2 + 0.3
POCKET_HY = 16.4                                                      # the relaxed snap wings (|y| 16.22) pass; 1.3 mm side walls
SW_BODY_T = 11.5                                                      # switch body thickness through the panel (world x), measured by Shawn 2026-10-17; REV STEP body 10.3
PANEL = 3.0; SW_CUT = (12.3, 27.5)                                    # x, y: Swann 40-family cutout 27.4 x 12.3 (catalogue); the Swann 39 body measured 11.5 thick (Shawn, 2026-10-17)
DFR = dict(w=26.0, h=30.0, t=8.0, pcb=1.6, back=X_IN - 2.2, z0=10.5)  # PCB back plane x -49.2, power pads at z 10.5
SCREW = dict(r_col=2.8, bore_r=2.2, clear_r=1.15, floor=3.0, pilot_r=0.8, len=8.0)   # 3.0 mm head floor + M2 x 8 (2026-10-15, the V1.17 box cap's 1.0 mm floors tore): same tip as 1.0 + M2 x 6
CAP_SCREWS = [(27.0, 24.0), (-27.0, 24.0), (0.0, -26.5)]
WINDOW = dict(y=12.0, z=(4.6, 9.4))
UPPER_SCREWS = [(y, z) for (y, z) in BRP.POST['holes'] if z > 0]


def outline(X_LID):
    W = K.Rect(-HW, HW, Z_STEP, Z1, RC['top'], RC['step'], 'side', 'top', 'step')
    N = K.Rect(-N_HY, N_HY, N_Z0, Z_STEP + 8.0, 0.0, RC['nbot'], 'nside', 'top', 'nbot')
    return K.Outline(W=W, N=N, rf=RC['rf'], R=dict(R_EDGE), X_LID=X_LID, X_IN=X_IN, T=T)


def switch_place():
    """rocker down: switch local y (through-panel, rocker +y) -> world -z, local z (10.6 / flange 13.6) -> world x,
    local x (27.5 / wings) -> world -y (a proper rotation). Flange centre x = X_IN - SW_POCKET_HALF, so the rocker pocket
    ends at the cover."""
    sw = C.rev_local(); Rm = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    xc = X_IN - SW_POCKET_HALF; z_pf = N_Z0 + 8.49; zc_l = 0.5 * (-12.21 + 1.39)
    m = C.place(sw, Rm, [xc - zc_l, -0.15, z_pf + C.REV_FRONT_Y])
    assert np.linalg.det(Rm) > 0
    v = m.vertices; above = v[v[:, 2] > z_pf + PANEL + 0.05]
    return m, dict(xc=xc, z_pf=round(z_pf, 3), body_x=[round(float(above[:, 0].min()), 3), round(float(above[:, 0].max()), 3)],
                   body_y=[round(float(above[:, 1].min()), 3), round(float(above[:, 1].max()), 3)], top_z=round(float(v[:, 2].max()), 3),
                   tip_z=round(float(v[:, 2].min()), 3))


def lip_prism(s, x_front, x_back, z0, z1, y_root, depth, chamfer):
    """flexure lip on side s (+1/-1): x-y profile with its lead-in chamfer on the rear (x_back) face, extruded z0..z1."""
    yi = y_root - depth
    pts = [(x_front, s * (y_root + 0.3)), (x_front, s * yi), (x_back + chamfer, s * yi), (x_back, s * y_root), (x_back, s * (y_root + 0.3))]   # root 0.3 into the arm
    return P.extrude_xy(Polygon(pts), z0, z1)


def build():
    sw_mesh, sw = switch_place()
    body_back = sw['body_x'][0] - (SW_BODY_T - (sw['body_x'][1] - sw['body_x'][0])) / 2   # measured body, centred on the STEP body
    bat_front = round(body_back - 0.3, 2)                             # battery pocket front: 0.3 behind the switch body
    bat_back = bat_front - BAT_POCKET[2]
    X_LID = math.floor((bat_back - T) * 20) / 20
    X_SPLIT = round(X_LID + max(R_EDGE.values()), 2)
    ol = outline(X_LID); outer, inner = ol.solids(0.3, x_shift=0.15)
    poly = C.outline_poly(ol)
    shell = outer - inner
    z_shelf = sw['z_pf'] + PANEL                                      # battery floor (panel top), z -22.51
    bat_lo = np.array([bat_back, -BAT_POCKET[1] / 2, z_shelf]); bat_hi = np.array([bat_front, BAT_POCKET[1] / 2, z_shelf + BAT_POCKET[0]])
    blo = np.array([bat_front - BAT_LIST[2], -BAT_LIST[1] / 2, z_shelf]); bhi = np.array([bat_front, BAT_LIST[1] / 2, z_shelf + BAT_LIST[0]])
    # listing pack (with the foam gap behind it) at least 0.3 inside the cavity, rounded rear edges included
    c = np.array([[x, y, z] for x in np.linspace(blo[0], bhi[0], 7) for y in np.linspace(blo[1], bhi[1], 9) for z in np.linspace(blo[2], bhi[2], 25)])
    fit = float(ol.inner_np(c).max())
    assert fit <= -0.3, f'battery {fit} from the cavity wall'

    # ---------------- frame
    web = P.extrude_yz(poly.intersection(sbox(-HW - 1, -80, HW + 1, -2.0)), X_IN, X_OUT)
    # web and cover plate share both faces (x -47 / -45): union them first; in one batch with the rest they left zero-
    # thickness sheets at x -47 (8 mesh bodies, not watertight) once X_LID moved to -74.4 (2026-10-17)
    frame_parts = [shell ^ P.box((X_SPLIT, -60, -60), (X_IN + 0.5, 60, 60)), C.cover_plate() + web]
    xc, z_pf = sw['xc'], sw['z_pf']
    pocket = P.box((xc - SW_POCKET_HALF, -POCKET_HY, N_Z0 - 1), (X_IN + 0.01, POCKET_HY, z_pf))
    cut_hole = P.box((xc - SW_CUT[0] / 2, -SW_CUT[1] / 2, z_pf - 0.5), (xc + SW_CUT[0] / 2, SW_CUT[1] / 2, z_pf + PANEL + 0.5))
    # narrow tier: solid from the rim to the cover below the battery floor, the switch pocket cut from below; the panel
    # (z_pf .. z_pf + 3) is the battery floor. Solid from the rim so nothing starts in mid-air when printed rim-down.
    floor_block = P.box((X_SPLIT, -N_HY - 1.0, N_Z0 - 1.0), (X_IN, N_HY + 1.0, z_shelf)) ^ outer      # clipped by the outer surface only
    frame_parts.append(floor_block)
    # upper backing plate for the DFR above the cover top (z 29.5), hung from the top wall; the shells reach x -45.97
    backing = P.box((X_IN - 1.2, -15.5, 29.0), (X_IN + 0.5, 15.5, ZI1 + 0.5)) ^ outer
    # outer stops at x -47, so the backing only touched the cover face to face and left a zero-thickness slit there
    # (non-manifold edge, 2026-10-17); the band z 29.0..29.5 lies inside the cover plate, so this overlap adds no outside material
    backing = backing + P.box((X_IN - 1.2, -15.5, 29.0), (X_IN + 0.5, 15.5, 29.5))
    frame_parts.append(backing)
    # DFR: standoff ribs on the pad-free 30 mm edges, flexure lips, VIN-corner stop
    pcb_front = DFR['back'] - DFR['pcb']; dz0, dz1 = DFR['z0'], DFR['z0'] + DFR['h']
    for s in (1, -1):
        lo, hi = sorted([s * 11.5, s * 13.0])
        frame_parts.append(P.box((DFR['back'], lo, dz0 + 1.5), (X_IN + 0.5, hi, 29.3)))                   # on the cover
        frame_parts.append(P.box((DFR['back'], lo, 29.0), (X_IN - 1.1, hi, dz1)))                         # on the backing plate
        flo, fhi = sorted([s * 13.2, s * 14.0])
        for (fz0, fz1, root) in ((12.5, 22.0, X_IN + 0.5), (30.0, 39.5, X_IN - 1.1)):                      # flexures (cover, backing)
            frame_parts.append(P.box((pcb_front - 1.5, flo, fz0), (root, fhi, fz1)))
            frame_parts.append(lip_prism(s, pcb_front - 0.1, pcb_front - 1.3, fz0 + 0.3, fz1 - 0.3, 13.2, 0.45, 0.45))
    frame_parts.append(P.box((pcb_front, 12.4, dz0 - 0.9), (X_IN + 0.5, 13.2, dz0 - 0.1)))              # VIN-end corner stop
    # cap screw bosses in the frame (from the rim forward); the bottom one lies inside the floor block
    pilots = []
    for (y, z) in CAP_SCREWS:
        tip = X_SPLIT - SCREW['floor'] + SCREW['len']
        if z > 0:
            s = 1 if y > 0 else -1
            frame_parts.append(P.cyl_x(SCREW['r_col'], X_SPLIT, tip + 1.0, y, z))
            web_lo, web_hi = sorted([s * abs(y), s * (HW - 0.5)])
            frame_parts.append(P.box((X_SPLIT, web_lo, z - 1.2), (tip + 1.0, web_hi, z + 1.2)) ^ outer)
        pilots.append(P.cyl_x(SCREW['pilot_r'], X_SPLIT - 0.5, tip + 0.1, y, z))
    frame = P.union(frame_parts)
    cuts = [pocket, cut_hole] + pilots
    cuts += [P.cyl_x(BRP.COVER['hole_r'], X_IN - 1, X_OUT + 1, y, z) for (y, z) in BRP.POST['holes']]
    cuts += [P.cyl_x(2.3, X_IN - 3.0, X_IN, y, z) for (y, z) in UPPER_SCREWS]          # head room for the upper cover screws
    cuts.append(P.box((X_IN - 1.5, -WINDOW['y'], WINDOW['z'][0]), (X_OUT + 1, WINDOW['y'], WINDOW['z'][1])))
    frame = frame - P.union(cuts)

    # ---------------- cap
    cap = shell ^ P.box((X_LID - 1, -60, -60), (X_SPLIT, 60, 60))
    cparts, choles = [], []
    for (y, z) in CAP_SCREWS:
        seat = X_SPLIT - SCREW['floor']
        cparts.append(P.cyl_x(SCREW['r_col'], X_LID + 0.5, X_SPLIT, y, z) ^ outer)
        choles += [P.cyl_x(SCREW['bore_r'], X_LID - 1.5, seat, y, z), P.cyl_x(SCREW['clear_r'], seat - 0.1, X_SPLIT + 0.1, y, z)]
    for s in (1, -1):                                                 # battery fences
        lo, hi = sorted([s * (BAT_POCKET[1] / 2 + 0.3), s * (BAT_POCKET[1] / 2 + 1.5)])
        cparts.append(P.box((X_LID + 0.5, lo, Z_STEP - 1.0), (X_SPLIT, hi, 30.0)) ^ outer)
    cparts.append(P.box((X_LID + 0.5, -16.8, z_shelf - 1.2), (X_SPLIT, 16.8, z_shelf)) ^ outer)   # shelf under the battery's rear half
    cap = P.union([cap] + cparts) - P.union(choles)

    # ---------------- bought parts (renders, checks)
    comp = {'switch': sw_mesh}
    comp['battery'] = C.boxmesh(blo, bhi)
    dlo = np.array([DFR['back'] - DFR['t'], -DFR['w'] / 2, dz0]); dhi = np.array([DFR['back'], DFR['w'] / 2, dz1])
    comp['dfr1202'] = C.boxmesh(dlo, dhi)
    pads = []
    for c_, w_ in ((2.18, 2.47), (13.13, 4.49), (23.69, 3.48)):      # VIN, GND, VOUT from the +y end
        y = DFR['w'] / 2 - c_; pads.append(C.boxmesh([pcb_front - 0.05, y - w_ / 2, dz0 - 0.3], [DFR['back'] + 0.05, y + w_ / 2, dz0 + 1.8]))
    comp['dfr_pads'] = trimesh.util.concatenate(pads)
    XT, MR = C.XT30_HALF, C.MR30_HALF
    J = {'J1 battery-switch XT30': C.plugbox([-66.0, -21.0, 16.0], 2, XT, 0),
         'J2 switch-feed XT30': C.plugbox([-66.0, 21.0, 16.0], 2, XT, 0),
         'J3 servo swap XT30': C.plugbox([-41.6, 6.5, 17.0], 2, XT, 1),
         'J4 A7Z swap MR30': C.plugbox([-41.6, -6.5, 17.0], 2, MR, 1),
         'balance plug JST-XH': C.boxmesh([-68.0, 13.0, 36.5], [-62.0, 21.0, 41.0])}
    comp['plugs'] = trimesh.util.concatenate([m for k, m in J.items() if not k.startswith(('J3', 'J4'))])
    comp['swap_plugs'] = trimesh.util.concatenate([m for k, m in J.items() if k.startswith(('J3', 'J4'))])
    j = {k.split(' ')[0]: m.bounds for k, m in J.items()}
    # rough lead centrelines (renders and the wire-through-wall check)
    bx = 0.5 * (blo[0] + bhi[0]); btop = bhi[2]; stop = sw['top_z']; dpx = pcb_front - 1.0; dpz = dz0 - 0.4
    vin, gnd, vout = DFR['w'] / 2 - 2.18, DFR['w'] / 2 - 13.13, DFR['w'] / 2 - 23.69
    wires = {
        'battery lead (16 AWG)': [[bx, -8.0, btop], [bx, -8.0, 38.0], [bx, -19.0, 38.0], [bx, -21.0, 32.0], [-66.0, -21.0, j['J1'][1][2]]],
        'switch lead to J1': [[xc, -4.0, stop], [xc, -4.0, 7.5], [-58.0, -20.0, 7.0], [-63.0, -21.0, -6.0], [-66.0, -21.0, j['J1'][0][2]]],
        'switch lead to J2': [[xc, 4.0, stop], [xc, 4.0, 7.5], [-58.0, 20.0, 7.0], [-63.0, 21.0, -6.0], [-66.0, 21.0, j['J2'][0][2]]],
        'feed lead to VIN/GND': [[-66.0, 21.0, j['J2'][1][2]], [-60.0, 21.0, 31.0], [-53.0, 19.5, 8.0], [dpx, 8.0, 7.5], [dpx, vin, dpz]],
        'servo lead to J3': [[dpx, vin - 1.2, dpz], [dpx, 6.5, 7.0], [-45.5, 6.5, 7.0], [-43.5, 6.5, 0.5], [-41.6, 6.5, j['J3'][0][2]]],
        '5 V lead to J4': [[dpx, vout, dpz], [dpx, -6.5, 7.6], [-45.5, -6.5, 7.6], [-43.5, -6.5, 0.5], [-41.6, -6.5, j['J4'][0][2]]],
        'robot servo inlet (from OpenRB J4)': [[-31.5, 0.0, 25.4], [-35.0, 2.0, 30.0], [-40.5, 5.5, 31.5], [-41.6, 6.5, j['J3'][1][2]]],
        'robot A7Z lead (from the neck)': [[-20.0, -2.0, 36.0], [-33.0, -4.0, 31.0], [-40.5, -5.5, 30.5], [-41.6, -6.5, j['J4'][1][2]]],
    }
    paths = {}
    for k, wp in wires.items():
        pts = K.fillet_path([tuple(map(float, p)) for p in wp], 2.0, n=6)
        paths[k] = dict(points=np.round(np.asarray(pts), 3).tolist(), kind='robot' if k.startswith('robot') else 'pair', lengthMm=round(K.path_length(pts), 1))

    # ---------------- outputs
    for nm, man in (('frame', frame), ('cap', cap)):
        m = P.from_m(man); m.merge_vertices(); m.export(OUT / f'{nm}-world.stl')
    for k, m in comp.items():
        m.export(OUT / f'{k}-world.stl')
    (OUT / 'wires.json').write_text(json.dumps(paths, indent=0))
    rec = dict(X_LID=X_LID, X_SPLIT=X_SPLIT, depthBehindCoverMm=round(X_IN - X_LID, 2), switch=sw,
               batteryPocket=dict(lo=np.round(bat_lo, 2).tolist(), hi=np.round(bat_hi, 2).tolist(), listingCavityMarginMm=round(-fit, 2),
                                  leadBandMm=round(ZI1 - bhi[2], 2)),
               battery=dict(lo=np.round(blo, 2).tolist(), hi=np.round(bhi, 2).tolist()),
               dfr1202=dict(lo=np.round(dlo, 2).tolist(), hi=np.round(dhi, 2).tolist(), gapToBatteryMm=round(float(dlo[0] - bat_front), 2)),
               capScrews=[dict(y=y, z=z, screw='M2 x %d' % SCREW['len'], headSeatX=round(X_SPLIT - SCREW['floor'], 2), tipX=round(X_SPLIT - SCREW['floor'] + SCREW['len'], 2)) for y, z in CAP_SCREWS],
               plugs={k: dict(centre=np.round(m.bounds.mean(0), 1).tolist(), size=np.round(m.extents, 1).tolist()) for k, m in J.items()},
               window=WINDOW, wiresMm={k: v['lengthMm'] for k, v in paths.items()},
               volumesMm3=dict(frame=round(frame.volume(), 1), cap=round(cap.volume(), 1)),
               bounds=dict(frame=np.round(P.from_m(frame).bounds, 2).tolist(), cap=np.round(P.from_m(cap).bounds, 2).tolist()),
               outline=dict(R_EDGE=R_EDGE, RC=RC, N=dict(hy=N_HY, z0=N_Z0), T=T))
    clash = {}
    for k, m in list(comp.items()) + [(k, v) for k, v in J.items()]:
        try:
            mm = P.mesh_to_m(m)
            clash[k] = dict(frame=round((mm ^ frame).volume(), 2), cap=round((mm ^ cap).volume(), 2))
        except Exception as e:
            clash[k] = f'not manifold: {e}'
    rec['boughtPartOverlapMm3'] = clash
    P.jdump(rec, OUT / 'build.json')
    return rec


if __name__ == '__main__':
    print(json.dumps(build(), indent=1))
