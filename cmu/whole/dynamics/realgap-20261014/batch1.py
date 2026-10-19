import json
from concurrent.futures import ProcessPoolExecutor
from policy_test import run
HIP = lambda v: {'left_hip_pitch': -v, 'right_hip_pitch': v}
ANK = lambda v: {'left_ankle': -v, 'right_ankle': v}
CASES = {
 'nominal-STAND': dict(), 'nominal-HOME': dict(start='HOME'), 'nominal-ramp1.5': dict(ramp_s=1.5),
 'com10-STAND': dict(com_x_mm=10), 'com10-ramp1.5': dict(com_x_mm=10, ramp_s=1.5), 'com15-STAND': dict(com_x_mm=15),
 'com20-STAND': dict(com_x_mm=20),
 'hip8-STAND': dict(zero=HIP(8)), 'hip8-ramp1.5': dict(zero=HIP(8), ramp_s=1.5), 'hip12-STAND': dict(zero=HIP(12)),
 'ank4-STAND': dict(zero=ANK(4)), 'ank8-STAND': dict(zero=ANK(8)),
 'imu+10-nominal': dict(imu_pitch=10), 'imu-10-nominal': dict(imu_pitch=-10),
 'com10-imu+10': dict(com_x_mm=10, imu_pitch=10),
}
def go(k): r = run(**CASES[k]); return k, {x: r[x] for x in ('fell_at', 'max_tilt', 'dist_x')}
if __name__ == '__main__':
    with ProcessPoolExecutor(2) as ex:
        for k, r in ex.map(go, CASES): print(k, r, flush=True)
