"""Whole-robot mass and COM with the battery backpack (build_bp.py), same baseline and method as layout-F/mass_estimate.py
(printed parts: CAD volume x 1.24 g/cm3 x 0.851 fill; V1.8 cover removed). Bought-part masses: battery 43 g (Tattu
listing); switch with its leads, DFR1202, plug halves and box wires are estimates, marked so. Writes bp_mass.json."""
import json
from pathlib import Path
import numpy as np, trimesh
HERE = Path(__file__).resolve().parent; B = HERE / 'build'
F = json.loads((HERE.parent / 'layout-F' / 'mass_estimate.json').read_text()); base = F['baseline']
c = json.loads((B / 'build.json').read_text()); items = []
for k in ('frame', 'cap'):
    m = trimesh.load(str(B / f'{k}-world.stl'), force='mesh')
    items.append(dict(id=k, g=round(abs(m.volume) / 1000 * 1.24 * 0.851, 2), com=[round(v, 2) for v in m.center_mass], label='CAD volume x 1.24 x 0.851'))
ctr = lambda d: [round((a + b) / 2, 2) for a, b in zip(d['lo'], d['hi'])]
items += [dict(id='battery', g=43.0, com=ctr(c['battery']), label='Tattu listing 43 g'),
          dict(id='switch_with_leads', g=14.0, com=[c['switch']['xc'], 0, -14.0], label='estimate'),
          dict(id='dfr1202', g=6.0, com=ctr(c['dfr1202']), label='estimate'),
          dict(id='plugs_and_wires', g=10.0, com=[-58.0, 0, 12.0], label='estimate: 4 XT30 halves, 1 MR30 half, box leads'),
          dict(id='rear_pack_cover_removed', g=-6.304, com=[-45.8, 0, -5.3], label='slicer (V1.8 cover)')]
added = sum(i['g'] for i in items); M = base['totalMassG'] + added
com = (np.array(base['comMm']) * base['totalMassG'] + sum(i['g'] * np.array(i['com']) for i in items)) / M
out = dict(backpack=dict(items=items, netAddedG=round(added, 1), newTotalG=round(M, 1), newComMm=[round(v, 2) for v in com],
                         comShiftMm=[round(v, 2) for v in com - np.array(base['comMm'])]),
           F_bench_box=dict(netAddedG=F['netAddedG'], newTotalG=F['newTotalG'], newComMm=F['newComMm'], comShiftMm=F['comShiftMm']),
           baseline=dict(totalMassG=base['totalMassG'], comMm=base['comMm']))
(HERE / 'bp_mass.json').write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(k, {kk: v[kk] for kk in v if kk != 'items'})
