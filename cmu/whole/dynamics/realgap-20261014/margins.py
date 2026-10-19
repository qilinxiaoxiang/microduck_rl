"""Tolerance of walk61/stand61 to each single real-robot gap (start STAND, policy on at t=0, stand 3 s, walk 0.3 for 10 s)."""
import json
from concurrent.futures import ProcessPoolExecutor
from policy_test import run
HIP = lambda v: {'left_hip_pitch': -v, 'right_hip_pitch': v}
ANK = lambda v: {'left_ankle': -v, 'right_ankle': v}
KNEE = lambda v: {'left_knee': -v, 'right_knee': v}
CASES = {}
for v in [-10, -7.5, -5, -2.5, 2.5, 5, 7.5]: CASES[f'com{v:+}'] = dict(com_x_mm=v)
for v in [-6, -4, -2, 2, 4, 6]: CASES[f'hip{v:+}'] = dict(zero=HIP(v))
for v in [-3, -2, -1, 1, 2, 3]: CASES[f'ank{v:+}'] = dict(zero=ANK(v))
for v in [-6, -3, 3, 6]: CASES[f'knee{v:+}'] = dict(zero=KNEE(v))
for v in [-6, -4, -2, 2, 4, 6]: CASES[f'imu{v:+}'] = dict(imu_pitch=v)
def go(k): r = run(**CASES[k], T=13.0); return k, {x: r[x] for x in ('fell_at', 'max_tilt', 'dist_x')}
if __name__ == '__main__':
    out = {}
    with ProcessPoolExecutor(2) as ex:
        for k, r in ex.map(go, CASES): out[k] = r; print(k, r, flush=True)
    json.dump(out, open('margins.json', 'w'), indent=1)
