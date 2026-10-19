"""Renders for the teammate build page (bringup/battery/team): the V1.20 backpack print geometry (build_bp.py) with the
Swann 39 switch drawn to its measured shape instead of the REV envelope. The robot keeps its two leads (22 AWG solid servo
pair from J4, Dupont 5 V lead from the A7Z); only their free ends carry the male XT30 / MR30, and the bench box gets female
pigtails in its XL4015 OUT terminals (2026-10-18), so the swap plugs stay as build_bp.py places them.
    ~/.venvs/microduck-cad/bin/python render_team.py      (bounded runner; after build_bp.py)
Writes renders/team/TEAM_*.png. Print geometry is untouched.

Swann 39 shape (local to the switch, mm): body 27.3 x 11.5 (Shawn measured 11.5, 2026-10-17; 27.3 from the listing),
18.4 deep behind the flange and lugs 7.2 further (photo scale, IMG_7023 at about 25 px/mm), flange 30 x 13.8 x 2,
rocker 25 x 10 to the REV-envelope tip (8.49 proud), snap wings on the two long faces."""
import json
from pathlib import Path
import numpy as np, trimesh
import render_concepts as RC
from render_concepts import GR, HERE, SIZE, part, tube, offset_path, cam, annotate, sheet, shifted, SHELLS, BOX

B = HERE / 'build'
OUT = HERE / 'renders' / 'team'; OUT.mkdir(parents=True, exist_ok=True)
c = json.loads((B / 'build.json').read_text())
SW = c['switch']; XC = SW['xc']; ZPF = SW['z_pf']; TIP = SW['tip_z']


def bx(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return trimesh.creation.box(extents=hi - lo, transform=trimesh.transformations.translation_matrix((lo + hi) / 2))


def swann():
    """the Swann 39 from swann_switch.py: body, flange, rocker and wings as one mesh, the two lugs as another."""
    import swann_switch as SS
    p = SS.parts()
    return trimesh.util.concatenate([m for k, m in p.items() if not k.startswith('lug')]), trimesh.util.concatenate([m for k, m in p.items() if k.startswith('lug')])


def scene():
    sc = {k: v for k, v in RC.robot_scene().items() if not k.startswith(('battery_backpack', 'bp_'))}   # the guide's own backpack copy
    for k in ('frame', 'cap', 'battery', 'dfr1202', 'dfr_pads', 'plugs'):
        m = trimesh.load(str(B / f'{k}-world.stl'), force='mesh'); sc[k] = part(m.vertices, m.faces, RC.COL.get(k, RC.COL['plugs']), 'bp' if k in BOX else 'bpin')
    body, lugs = swann()
    sc['switch'] = part(body.vertices, body.faces, [28, 28, 30], 'bpin')
    sc['switch_lugs'] = part(lugs.vertices, lugs.faces, [200, 150, 70], 'bpin')
    j3 = c['plugs']['J3 servo swap XT30']['centre']; j4 = c['plugs']['J4 A7Z swap MR30']['centre']
    m = trimesh.load(str(B / 'swap_plugs-world.stl'), force='mesh'); sc['swap_plugs'] = part(m.vertices, m.faces, RC.COL['plugs'], 'bpin')
    for k, w in json.loads((B / 'wires.json').read_text()).items():
        p = np.array(w['points'])
        r = 0.6 if 'A7Z' in k else 0.9                                          # Dupont lead thinner than the 22 AWG solid pair
        v, f = tube(offset_path(p, 0.95), r); sc[f'w_{k}_a'] = part(v, f, [200, 30, 30], 'wires')
        v, f = tube(offset_path(p, -0.95), r); sc[f'w_{k}_b'] = part(v, f, [30, 30, 30], 'wires')
    return sc, j3, j4


if __name__ == '__main__':
    sc, j3, j4 = scene(); ims = []
    j3, j4 = np.array(j3), np.array(j4)
    bc = (np.array(c['battery']['lo']) + np.array(c['battery']['hi'])) / 2; dc = (np.array(c['dfr1202']['lo']) + np.array(c['dfr1202']['hi'])) / 2
    J = {k.split(' ')[0]: np.array(v['centre']) for k, v in c['plugs'].items()}
    XL = c['X_LID']; caps = [np.array([XL, s['y'], s['z']]) for s in c['capScrews']]
    rocker = np.array([XC, 0, TIP + 1.0]); swc = np.array([XC, 0, ZPF + 9.0]); LOW = [[-45.0, 16.1, -36.5], [-45.0, -16.1, -36.5]]
    sc_now = sc

    def shot(tag, view, title, labels, side='right', colx=None, **kw):
        f = GR.render(sc_now, [view], str(OUT / f'TEAM_{tag}.png'), size=SIZE, **kw)[0]
        ims.append(annotate(f, view, labels, title, side, colx))

    shot('switch', cam([XC, 0, -14], (-0.55, -0.75, 0.35), 95), 'Swann 39 in the floor panel, rocker down (box see-through)',
         [('switch body (11.5 mm thick)', swc + [0, -13.6, 0]), ('solder lugs (red wires)', [XC, -5.4, ZPF + 22.0]),
          ('snap wing over the 3 mm panel', [XC, -14.5, ZPF + 7.0]), ('rocker, flush in the floor pocket', rocker)],
         hide={'battery', 'dfr1202', 'dfr_pads', 'plugs', 'swap_plugs', 'wires', 'cap'}, ghost={'frame'} | SHELLS)
    shot('side', cam([-58, 0, 3], (0, 1, 0.05), 205), 'side, see-through (robot front to the LEFT)',
         [('battery (upright, leads out the top)', [bc[0], 15.5, -12.0]), ('DFR1202 5 V board', dc + [0, 13, 0]),
          ('Swann 39 switch', swc + [0, 12, 0]), ('plug 2 (XT30, plug 1 behind it)', J['J2'] + [0, 0, -6]),
          ('swap plugs 3 + 4 (in the pack channel)', j3 + [0, 0, 6])],
         colx=1000 - 12, hide={'left-leg', 'right-leg'}, ghost=SHELLS | BOX)
    shot('rear', cam([-58, 0, 5], (-1, 0, 0.12), 215), 'from behind, see-through box (robot right side on the right)',
         [('1 battery to switch (XT30)', J['J1'] + [0, 0, -12]), ('2 switch to 5 V feed (XT30)', J['J2'] + [0, 0, 6]),
          ('balance plug (JST-XH)', J['balance']), ('battery', bc + [0, 0, 5]), ('cap screw 3 (M2 x 8)', caps[2])],
         ghost=BOX | SHELLS)
    shot('low', cam([-55, 0, -30], (0, 0, -1), 210, up=(-1, 0, 0)), 'from below (rear at the top): press the rocker here',
         [('switch rocker', rocker), ('lower fixing screw (M2 x 6)', LOW[0]), ('lower fixing screw (M2 x 6)', LOW[1]),
          ('cap screw 3 (M2 x 8)', caps[2])], hide={'wires', 'battery', 'dfr1202', 'dfr_pads', 'plugs', 'swap_plugs', 'left-leg', 'right-leg'})
    shot('plugs', cam([-42, 0, 15], (-0.8, 0.45, 0.8), 150), 'the two swap plugs in the pack channel (box hidden)',
         [('3 servo plug (XT30): your lead D', j3 + [0, 0, 13]), ('4 A7Z 5 V plug (MR30): your lead E', j4 + [0, 0, 11.4]),
          ('robot servo pair, 22 AWG solid (from J4)', [-36.0, 2.5, 30.5]), ('robot 5 V lead, Dupont (from the neck)', [-38.0, -4.5, 31.0])],
         hide={'battery', 'cap', 'frame', 'dfr1202', 'dfr_pads', 'switch', 'switch_lugs', 'plugs'} | {k for k in sc if k.startswith(('w_battery', 'w_switch', 'w_feed'))}, ghost=SHELLS)
    sc_now = shifted(sc, ['cap'], -30.0)
    sc_now = shifted(sc_now, ['battery'] + [k for k in sc if k.startswith('w_battery')], -55.0)
    shot('out', cam([-78, 0, 10], (-0.9, -0.75, 0.55), 340), 'charging: three screws out, cap off, battery out the back',
         [('cap', [XL - 30.0, -20, 30]), ('battery', bc + [-55, 0, 5])], side='left', hide={'wires'})
    sheet(ims, OUT / 'TEAM-sheet.png'); print(OUT / 'TEAM-sheet.png')
