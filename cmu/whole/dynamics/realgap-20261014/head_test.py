"""Does walk61/stand61 track head-pose commands (camera aiming) while balancing? Nominal as-built sim."""
import math, json
from concurrent.futures import ProcessPoolExecutor
from policy_test import run
CASES = {
 'stand-yaw-sweep0.6': dict(walk_at=99, head=lambda t: [0, 0, 0.6 * math.sin(2 * math.pi * t / 4), 0]),
 'stand-yaw-step0.8': dict(walk_at=99, head=lambda t: [0, 0, 0.8 if t > 2 else 0, 0]),
 'stand-pitch-sweep0.3': dict(walk_at=99, head=lambda t: [0, 0.3 * math.sin(2 * math.pi * t / 4), 0, 0]),
 'walk-yaw-sweep0.6': dict(head=lambda t: [0, 0, 0.6 * math.sin(2 * math.pi * t / 4), 0]),
}
def go(k):
    r = run(**CASES[k], T=12.0)
    tr = r['trace']; f = CASES[k]['head']
    err = [abs(row[3][2] - f(row[0])[2]) + abs(row[3][1] - f(row[0])[1]) for row in tr if row[0] > 1]
    return k, dict(fell_at=r['fell_at'], max_tilt=r['max_tilt'], mean_head_err_rad=round(sum(err) / len(err), 3),
                   sample=[(row[0], row[3]) for row in tr[4:12]])
if __name__ == '__main__':
    with ProcessPoolExecutor(2) as ex:
        for k, r in ex.map(go, CASES): print(k, json.dumps(r), flush=True)
