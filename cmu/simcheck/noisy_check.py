"""cmu/mac/brain_sim_check.py with the robot's measured sensor noise and per-phase metrics (sim only; robot code untouched).

  uv run --no-sync python cmu/simcheck/noisy_check.py [--gyro-noise-dq K] [--gyro-noise SD] [--vel-noise-dq K]
        [--vel-noise SD] <brain_sim_check.py arguments>

Noise goes on what walk_brain.Brain.tick receives, per tick, from its own stream (seeded 1000 + --seed, separate from
the sim's drop stream), the way guide/firmware/mac/jitter-20261018/jitter_sim.py does it:
  gyro (rad/s, per axis)        sd = sqrt(gyro_noise^2 + (gyro_noise_dq x rms leg joint speed)^2)
  joint velocity (rad/s, legs)  sd = sqrt(vel_noise^2 + (vel_noise_dq x rms leg joint speed)^2)
The leg speed is the sim's true one this tick. The values the suite passes and their sources are in sim_suite.py
(SENSOR_NOISE).

The printed result row is brain_sim_check's, plus metrics measured by phase (true leg speeds, the Brain's own gravity
reading and its actions, up to a fall):
  cmd_*    from the walk command (--walk-at) to its end: max_tilt_cmd (deg), action_jumps_cmd, jumps_cmd_s
  stand_*  stand61 at zero command, from the first policy tick (--stand-s) to the walk command, or to the end of the run
           when --vx and --vtheta are 0: steadiness as microduck-sim/extensions/battery-walk-20261122/gapcheck.py
           measures it on robot logs (jumps/s, leg rms speed, share of still 1 s windows, time to settle)
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAC = os.path.join(os.path.dirname(HERE), "mac")
sys.path.insert(0, MAC)

import numpy as np  # noqa: E402

HEAD = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
JUMP_RAD = 0.5            # brain_sim_check's action jump (tick-to-tick change > 0.5 rad), the real-log jitter measure
STILL_WIN, STILL_STEP = 50, 5   # gapcheck.py: 1 s windows every 0.1 s
STILL_DA, STILL_SP = 0.10, 0.25  # gapcheck.py: a window is still when no action changes > 0.10 rad, mean leg rms < 0.25


def stand_metrics(rows):
    """gapcheck.steady() on sim ticks: rows of (t, leg speeds, action or None)."""
    if len(rows) < 10:
        return None
    t = np.array([r[0] for r in rows])
    sp = np.array([float(np.sqrt(np.mean(np.square(r[1])))) for r in rows])
    da, last, jumps = np.zeros(len(rows)), None, 0
    for i, r in enumerate(rows):
        if r[2] is not None and last is not None:
            da[i] = float(np.max(np.abs(np.asarray(r[2]) - last)))
            jumps += da[i] > JUMP_RAD
        if r[2] is not None:
            last = np.asarray(r[2], float)
    dur = float(t[-1] - t[0]) + 0.02
    still, starts = [], []
    for s in range(0, max(1, len(rows) - STILL_WIN + 1), STILL_STEP):
        e = min(len(rows), s + STILL_WIN)
        if e - s < STILL_WIN // 2:
            break
        still.append(bool(da[s + 1:e].max(initial=0.0) <= STILL_DA and sp[s:e].mean() <= STILL_SP))
        starts.append(float(t[s] - t[0]))
    first = [s for s, ok in zip(starts, still) if ok]
    return dict(stand_dur=round(dur, 2), stand_jumps_s=round(jumps / dur, 2),
                stand_leg_rms=round(float(np.sqrt(np.mean(np.square(sp)))), 3),
                stand_still_frac=round(float(np.mean(still)), 3) if still else None,
                stand_settle_s=round(first[0], 2) if first else None)


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--gyro-noise-dq", type=float, default=0.0)
    pre.add_argument("--gyro-noise", type=float, default=0.0)
    pre.add_argument("--vel-noise-dq", type=float, default=0.0)
    pre.add_argument("--vel-noise", type=float, default=0.0)
    mine, rest = pre.parse_known_args()
    peek = argparse.ArgumentParser(add_help=False)
    for k, v in (("--stand-s", 1.0), ("--walk-at", 4.0), ("--walk-s", 10.0), ("--vx", 0.3), ("--vtheta", 0.0)):
        peek.add_argument(k, type=float, default=v)
    peek.add_argument("--seed", type=int, default=0)
    p, _ = peek.parse_known_args(rest)
    sys.argv = [os.path.join(MAC, "brain_sim_check.py")] + rest

    import brain_sim_check as bsc
    import walk_brain

    nrng = np.random.default_rng(1000 + p.seed)
    legs_idx = [k for k, n in enumerate(walk_brain.NAMES) if n not in HEAD]
    ticks = []  # (t, true leg speeds rad/s, Brain gravity reading, action or None)
    orig = walk_brain.Brain.tick

    def tick(self, want, t, pos_t, vel_t, quat, gyro, imu_valid, joint_age, imu_age):
        legs = np.array([vel_t[k] * walk_brain.VEL_UNIT for k in legs_idx], float)
        rms = float(np.sqrt(np.mean(np.square(legs))))
        gsd = math.hypot(mine.gyro_noise, mine.gyro_noise_dq * rms)
        vsd = math.hypot(mine.vel_noise, mine.vel_noise_dq * rms)
        if gsd > 0:
            gyro = tuple(float(g) + float(x) for g, x in zip(gyro, nrng.normal(0.0, gsd, 3)))
        if vsd > 0:
            vel_t = list(vel_t)
            for k in legs_idx:
                vel_t[k] = vel_t[k] + float(nrng.normal(0.0, vsd)) / walk_brain.VEL_UNIT
        goal, kp, info = orig(self, want, t, pos_t, vel_t, quat, gyro, imu_valid, joint_age, imu_age)
        ticks.append((float(t), legs, tuple(info.get("gb") or (0.0, 0.0, -1.0)),
                      None if info.get("action") is None else list(info["action"])))
        return goal, kp, info

    walk_brain.Brain.tick = tick
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bsc.main()
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    for line in lines[:-1]:
        print(line)
    row = json.loads(lines[-1])
    end = row.get("fell_at") if row.get("fell_at") is not None else float("inf")
    walking = abs(p.vx) > 1e-9 or abs(p.vtheta) > 1e-9
    stand_end = p.walk_at if walking else end
    stand = [(t, legs, a) for t, legs, gb, a in ticks if p.stand_s <= t < min(stand_end, end)]
    row.update(stand_metrics(stand) or {})
    if walking:
        cmd = [(t, gb, a) for t, legs, gb, a in ticks if p.walk_at <= t < min(p.walk_at + p.walk_s, end)]
        tilt, jumps, last = 0.0, 0, None
        for t, gb, a in cmd:
            g = np.asarray(gb, float)
            tilt = max(tilt, math.degrees(math.acos(max(-1.0, min(1.0, -g[2] / (np.linalg.norm(g) or 1.0))))))
            if a is not None:
                if last is not None and max(abs(x - y) for x, y in zip(a, last)) > JUMP_RAD:
                    jumps += 1
                last = a
        dur = len(cmd) * 0.02
        row.update(max_tilt_cmd=round(tilt, 1), action_jumps_cmd=jumps,
                   jumps_cmd_s=round(jumps / dur, 2) if dur > 0 else None)
    row["sensor_noise"] = vars(mine)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
