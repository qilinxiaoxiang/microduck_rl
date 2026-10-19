"""If the real robot equals the sim with its COM 10 mm forward, can a software posture trim (lean back via hip or
ankle zero offsets in the runtime mapping) bring walk61/stand61 back inside their tolerance?"""
import json
from concurrent.futures import ProcessPoolExecutor
from policy_test import run
HIP = lambda v: {'left_hip_pitch': -v, 'right_hip_pitch': v}
ANK = lambda v: {'left_ankle': -v, 'right_ankle': v}
CASES = {}
for com in (7.5, 10, 12.5):
    for v in (4, 6, 8, 10): CASES[f'com{com}-hiptrim-{v}'] = dict(com_x_mm=com, zero=HIP(-v))
    for v in (2, 4, 6): CASES[f'com{com}-anktrim-{v}'] = dict(com_x_mm=com, zero=ANK(-v))
def go(k): r = run(**CASES[k], T=13.0); return k, {x: r[x] for x in ('fell_at', 'max_tilt', 'dist_x')}
if __name__ == '__main__':
    out = {}
    with ProcessPoolExecutor(2) as ex:
        for k, r in ex.map(go, CASES): out[k] = r; print(k, r, flush=True)
    json.dump(out, open('trims.json', 'w'), indent=1)
