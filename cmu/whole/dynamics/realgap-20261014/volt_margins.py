"""Same single-gap tolerance as margins.py, at the community training voltage (7.4 V) vs this build's 5 V."""
import json
from concurrent.futures import ProcessPoolExecutor
from policy_test import run
HIP = lambda v: {'left_hip_pitch': -v, 'right_hip_pitch': v}
CASES = {}
for volt in (7.4,):
    for v in (2.5, 5, 7.5, 10, 15): CASES[f'v{volt}-com+{v}'] = dict(com_x_mm=v, volt=volt)
    for v in (6, 8, 10): CASES[f'v{volt}-hip+{v}'] = dict(zero=HIP(v), volt=volt)
    CASES[f'v{volt}-imu-6'] = dict(imu_pitch=-6, volt=volt)
def go(k): r = run(**CASES[k], T=13.0); return k, {x: r[x] for x in ('fell_at', 'max_tilt', 'dist_x')}
if __name__ == '__main__':
    out = {}
    with ProcessPoolExecutor(2) as ex:
        for k, r in ex.map(go, CASES): out[k] = r; print(k, r, flush=True)
    json.dump(out, open('volt_margins.json', 'w'), indent=1)
