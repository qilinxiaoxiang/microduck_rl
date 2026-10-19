import json, sys
from ramp_test import run
from static_gap import STAND, HOME
res = {}
for kind in ['ankle', 'hip', 'com']:
    for v in ([0, 4, 8, 10, 12, 14] if kind != 'com' else [0, 5, 10, 15, 20]):
        if kind == 'ankle': zero, com = {'left_ankle': -v, 'right_ankle': v}, 0
        elif kind == 'hip': zero, com = {'left_hip_pitch': -v, 'right_hip_pitch': v}, 0
        else: zero, com = {}, v
        st = run(zero, com, ramp_s=0.01, hold_s=0.01, pre_s=4.0, target=STAND)
        rp = run(zero, com, pre_s=1.0)
        gb_st = st['gbx_trace'][-1][1]
        print(json.dumps(dict(kind=kind, v=v, stand_hold_fell=st['fell_at'], stand_gbx_end=gb_st,
                              ramp_fell=rp['fell_at'], ramp_gbx=[x for x in rp['gbx_trace'] if x[0] in (1.0, 1.75, 2.5, 3.0, 5.5)])), flush=True)
