"""Renders of the battery backpack print geometry (build_bp.py) on the robot: same scene, colours and label style as the
draft renders (render_concepts.py), one sheet of six views.
    python render_bp.py            (CAD venv, bounded runner; after build_bp.py)
Writes renders/BP_*.png and renders/BP-sheet.png."""
import json
import numpy as np, trimesh
import render_concepts as RC
from render_concepts import GR, OUT, HERE, SIZE, part, tube, offset_path, cam, annotate, sheet, shifted, SHELLS, BOX

B = HERE / 'build'


def add_backpack(sc):
    for k in ('frame', 'cap', 'battery', 'switch', 'dfr1202', 'dfr_pads', 'plugs', 'swap_plugs'):
        m = trimesh.load(str(B / f'{k}-world.stl'), force='mesh'); sc[k] = part(m.vertices, m.faces, RC.COL.get(k, RC.COL['plugs']), 'bp' if k in BOX else 'bpin')
    for k, w in json.loads((B / 'wires.json').read_text()).items():
        p = np.array(w['points'])
        v, f = tube(offset_path(p, 0.95), 0.9); sc[f'w_{k}_a'] = part(v, f, [200, 30, 30], 'wires')
        v, f = tube(offset_path(p, -0.95), 0.9); sc[f'w_{k}_b'] = part(v, f, [30, 30, 30], 'wires')
    return sc


if __name__ == '__main__':
    c = json.loads((B / 'build.json').read_text())
    sc = add_backpack(dict(RC.robot_scene())); ims = []
    j = {k.split(' ')[0]: np.array(v['centre']) for k, v in c['plugs'].items()}
    bc = (np.array(c['battery']['lo']) + np.array(c['battery']['hi'])) / 2; dc = (np.array(c['dfr1202']['lo']) + np.array(c['dfr1202']['hi'])) / 2
    xc = c['switch']['xc']; rocker = np.array([xc, 0, -34.0]); swc = np.array([xc, 0, -12.0]); XL = c['X_LID']
    caps = [np.array([XL, s['y'], s['z']]) for s in c['capScrews']]
    LOW = [[-45.0, 16.1, -36.5], [-45.0, -16.1, -36.5]]
    INNER_HIDE = {'wires', 'battery', 'dfr1202', 'dfr_pads', 'plugs', 'swap_plugs'}
    sc_now = sc

    def shot(tag, view, title, labels, side='right', colx=None, **kw):
        f = GR.render(sc_now, [view], str(OUT / f'BP_{tag}.png'), size=SIZE, **kw)[0]
        ims.append(annotate(f, view, labels, title, side, colx))

    # 1 outside, rear 3/4
    shot('look', cam([-45, 0, 5], (-1, -0.6, 0.45), 330), f'outside: {c["depthBehindCoverMm"]} mm deep behind the front plate (bench box 20.1)',
         [('cap screw 1 (M2 x 6)', caps[0]), ('cap screw 2 (M2 x 6)', caps[1]), ('cap screw 3 (M2 x 6)', caps[2]),
          ('switch rocker (underneath)', rocker)], side='left', hide=INNER_HIDE)
    # 2 straight from below, rear at the top
    shot('low', cam([-55, 0, -30], (0, 0, -1), 210, up=(-1, 0, 0)), 'from below (rear at the top): rocker flush in a floor pocket',
         [('switch rocker (press from below)', rocker), ('lower fixing screw', LOW[0]), ('lower fixing screw', LOW[1]),
          ('cap screw 3', caps[2])], hide=INNER_HIDE | {'left-leg', 'right-leg'})
    # 3 side see-through
    shot('side', cam([-58, 0, 3], (0, 1, 0.05), 205), 'side, see-through (robot front to the LEFT): battery stands upright',
         [('battery (upright, leads out the top)', [bc[0], 15.5, -12.0]), ('DFR1202 5 V board', dc + [0, 13, 0]),
          ('switch body', swc + [0, 12, 0]), ('plug 2 (XT30, plug 1 behind it)', j['J2'] + [0, 0, -6]),
          ('swap plugs 3+4 (in the pack channel)', j['J3'] + [0, 0, 6])],
         colx=1000 - 12, hide={'left-leg', 'right-leg'}, ghost=SHELLS | BOX)
    # 4 from behind see-through
    shot('rear', cam([-58, 0, 5], (-1, 0, 0.12), 215), 'from behind, see-through box (robot right side on the right)',
         [('1 battery to switch (XT30)', j['J1'] + [0, 0, -12]), ('2 switch to 5 V feed (XT30)', j['J2'] + [0, 0, 6]),
          ('balance plug (JST-XH)', j['balance']), ('battery', bc + [0, 0, 5]), ('cap screw 3', caps[2])],
         ghost=BOX | SHELLS)
    # 5 swap plugs, box hidden
    shot('swap', cam([-42, 0, 15], (-0.8, 0.45, 0.8), 150), 'the two swap plugs in the pack channel (box hidden)',
         [('3 servo swap plug (XT30)', j['J3'] + [0, 0, 13]), ('4 A7Z 5 V swap plug (MR30)', j['J4'] + [0, 0, 11.4]),
          ('robot lead from OpenRB J4', [-36.0, 2.5, 30.5]), ('robot lead from the neck', [-38.0, -4.5, 31.0])],
         hide={'battery', 'cap', 'frame', 'dfr1202', 'dfr_pads', 'switch', 'plugs'} | {k for k in sc if k.startswith(('w_battery', 'w_switch', 'w_feed'))}, ghost=SHELLS)
    # 6 charging: cap and battery out the back
    sc_now = shifted(sc, ['cap'], -30.0)
    sc_now = shifted(sc_now, ['battery'] + [k for k in sc if k.startswith('w_battery')], -55.0)
    shot('out', cam([-78, 0, 10], (-0.9, -0.75, 0.55), 340), 'charging: three screws out, cap off, battery out the back',
         [('cap', [XL - 30.0, -20, 30]), ('battery', bc + [-55, 0, 5])], side='left', hide={'wires'})
    sheet(ims, OUT / 'BP-sheet.png'); print(OUT / 'BP-sheet.png')
