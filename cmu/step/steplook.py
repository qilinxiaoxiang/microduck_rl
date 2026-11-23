"""Step-look-step: burst, stop, look again with the camera, repeat until the ball is STOP_R ahead.

The camera closes the loop between steps; nothing tracks a position during one (the nudge chased its leg odometry into
a fall on 2026-11-21). Two arms, the same coordinator:
  walk61   the deployed policy (baseline)
  step     walk61 + the capped correction (Mjlab-WalkStep-Cap0xx-FromWalk61-MicroDuck-CMU-Current)
The arm is whichever file the running brain loaded as walk61.onnx (its run record's sha, looked up in arms.json); each
arm has its own burst table. No brain change: the step arm is a models directory with that file in walk61's place.

  look    standing, head down (never past HEAD_PITCH_MAX), until LOOK_N agreeing detections from frames captured after
          the head and body settled; the scan only before the first sighting, afterwards the head aims at the estimate
  ready   head back to neutral for HEAD_UP_S on a still body
  burst   ONE {"mode": "walk", "vx": 0.25, "vtheta": w, "for_s": T}: walk_wired ends it by itself, also if this dies
  settle  zero command until still, then look again

It stops, and never retries by itself, on: a guard or manual mode change, a body that does not come to rest, a ball
not found, too far to the side, or closer than expected, MAX_STEPS, and MAX_NO_PROGRESS steps without progress.

Installed beside seepick.py on the A7Z:
  ~/.venvs/duck-policy/bin/python steplook.py --preflight
  ~/.venvs/duck-policy/bin/python steplook.py --dry          look and print the step it would take, no motion of the legs
  ~/.venvs/duck-policy/bin/python steplook.py --single 0.3   one burst of 0.3 s between two looks (the ruler test)
  ~/.venvs/duck-policy/bin/python steplook.py                the loop
"""
import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as ap  # noqa: E402
import seepick as sp  # noqa: E402

VX = 0.25                  # m/s: walk61 steps only here
STOP_R = 0.18              # m from the foot-frame origin: with the head at 1.1 (neck 0.22 as the robot reads it) the
STOP_TOL = 0.025           # picture starts 0.14 m out, at neck 0.35 0.17 m: closer cannot be looked at between steps
R_MAX = 0.70               # m: further is for the approach, not for steps
HEAD_PITCH_MAX = 1.1       # rad absolute: 1.3 tipped the stand on 2026-11-21, up to 1.1 held on 2026-11-20
HEAD_YAW_MAX = 0.5         # rad: the range the step policy was trained to look in
SCAN = tuple(dict(head_pitch=p, head_yaw=y) for p, y in ((0.85, 0.0), (1.1, 0.0), (0.6, 0.0), (0.85, 0.5),
                                                          (0.85, -0.5)))
LOOK_N = 2                 # detections
LOOK_AGREE_M = 0.03
LOOK_MOVE_S = 0.8          # for the head to get there
LOOK_S = 3.5               # per head pose: a learned pass takes 1.1-1.25 s on the A7Z
HEAD_UP_S = 1.0            # neutral before a burst (trained: 0.6; walk61 does not step with a head command)
SETTLE_S = 1.5             # after a burst's end (walk61 keeps stepping 1-2 s)
STILL_WAIT_S = 6.0         # then this long to come to rest
BEARING_OK = math.radians(6)
BEARING_MAX = math.radians(35)  # further to the side: a person turns the duck (turning right was broken on 11/21)
VTH_MAX = 0.3              # rad/s (approach.VTH_ARC)
VTH_STRAIGHT = 0.002       # not zero on purpose: walk_brain learns its yaw trim from walks without a manual turn, and
#                            short bursts must not rewrite the robot's trim
NEAR_M = 0.10              # this close to the stop: short steps
MAX_STEPS = 10
MIN_PROGRESS_M = 0.01
MAX_NO_PROGRESS = 3
ARMS_FILE = os.path.join(HERE, "arms.json")


def load_arm(log_path, arms_file=ARMS_FILE):
    """(arm, None) for the walk61.onnx the running brain loaded, or (None, why)."""
    meta = (log_path[:-6] if log_path.endswith(".jsonl") else log_path) + ".meta.jsonl"
    try:
        with open(meta) as f:
            shas = json.loads(f.read().strip().splitlines()[-1])["model_sha"]
            sha = shas["walk61.onnx"]
        with open(arms_file) as f:
            arms = json.load(f)["arms"]
    except (OSError, ValueError, KeyError, IndexError) as e:
        return None, f"cannot tell which walk policy the brain runs ({type(e).__name__}: {e}); restart the brain"
    for arm in arms:
        if arm["sha"] == sha:
            wrong = [s for s in arm.get("slots", ["walk61"]) if shas.get(f"{s}.onnx") != sha]
            if wrong:
                return None, f"arm {arm['name']} needs its policy in {', '.join(wrong)} too; reinstall it (step.sh install)"
            return arm, None
    return None, f"the brain's walk61.onnx ({sha}) is neither arm in {arms_file}"


def plan(x, y, bursts, level=0):
    """The next burst for a ball at foot-frame (x, y): dict(T, vtheta), or dict(done=why) / dict(refuse=why).

    bursts: [[T, the furthest it moves], ...] rising. The longest one that cannot overshoot the stop window; `level`
    raises the choice by that many rungs after steps that did not move, never past what cannot reach the ball."""
    r, b = math.hypot(x, y), math.atan2(y, x)
    if abs(b) > BEARING_MAX:
        return dict(refuse=f"ball {math.degrees(b):+.0f} deg to the side: turn the duck to face it")
    if r > R_MAX:
        return dict(refuse=f"ball {r:.2f} m away, beyond {R_MAX} m: bring the duck closer")
    if r <= STOP_R + STOP_TOL:
        return dict(done=f"ball {r:.3f} m ahead, stop window {STOP_R} +- {STOP_TOL}")
    room = r - (STOP_R - STOP_TOL)
    fit = [i for i, (_, far) in enumerate(bursts) if far <= room]
    if not fit:
        return dict(done=f"ball {r:.3f} m ahead, nearer than the smallest burst can step without overshooting")
    i = fit[-1]
    if r - STOP_R < NEAR_M:
        i = min(i, level)      # close in: the smallest step, a rung up for every one that did not move
    T = float(bursts[i][0])
    w = VTH_STRAIGHT if abs(b) <= BEARING_OK else max(-VTH_MAX, min(VTH_MAX, b / T))
    return dict(T=T, vtheta=round(w, 4))


class StepLook(sp.SeePick):
    def __init__(self, kin, latest_frame, state, state_at, send, detect, arm, single=None, dry=False, now=None,
                 log=print):
        kw = {} if now is None else {"now": now}
        super().__init__(kin, latest_frame, state, state_at, send, detect, make_approach=None, tries=1, dry=dry,
                         log=log, **kw)
        self.arm, self.single = arm, single
        self.scan_i, self.found, self.est, self.look_t0 = 0, [], None, None
        self.burst, self.steps, self.no_progress = None, 0, 0

    # measure() takes a late learned detection, and carries the colour track to the last sighting, only in the phases
    # "find" / "aim" / "dock": the look is the find
    def look_head(self):
        if self.est is None:
            return dict(SCAN[self.scan_i])
        x, y = self.est
        pitch = min(HEAD_PITCH_MAX, max(0.6, 1.1 + (0.245 - math.hypot(x, y)) * 1.5))
        return dict(head_pitch=round(pitch, 3), head_yaw=round(max(-HEAD_YAW_MAX, min(HEAD_YAW_MAX, math.atan2(y, x))), 3))

    def still(self, now):
        return now - self.rock_t >= ap.STILL_S

    def start_look(self, why):
        self.found, self.look_t0 = [], None
        self.enter("find", why)

    def p_start(self, st, now):
        self.try_n = 1
        self.start_look(f"arm {self.arm['name']}")

    def p_find(self, st, now):
        if not self.still(now):
            # no look-down on a rocking body (robot 2026-11-06 fell doing that)
            self.stand()
            self.found, self.look_t0 = [], None
            if now - self.t_phase > STILL_WAIT_S + LOOK_S * len(SCAN):
                self.finish(False, "the body did not come to rest for a look")
            return
        self.stand(self.look_head())
        if self.look_t0 is None:
            self.look_t0 = now
        if now - self.look_t0 < LOOK_MOVE_S:
            return
        m = self.measure(st)
        if m is not None and m["t"] >= self.look_t0 + LOOK_MOVE_S / 2:
            self.found = [f for f in self.found if math.hypot(f["x"] - m["x"], f["y"] - m["y"]) < LOOK_AGREE_M] + [m]
            if len(self.found) >= LOOK_N:
                self.looked(*(float(np.median([f[k] for f in self.found])) for k in ("x", "y")))
                return
        if now - self.look_t0 < LOOK_S:
            return
        if self.est is not None:
            self.finish(False, f"ball not seen again after step {self.steps} (last x {self.est[0]:.3f} y {self.est[1]:.3f})")
        elif self.scan_i + 1 < len(SCAN):
            self.scan_i, self.found, self.look_t0 = self.scan_i + 1, [], None
        else:
            self.finish(False, "ball not found (put it 20-70 cm in front of the duck)")

    def looked(self, x, y):
        before, self.est = self.est, (x, y)
        if self.burst is not None and before is not None:
            moved = math.hypot(*before) - math.hypot(x, y)
            self.attempts.append(dict(self.burst, before=[round(v, 3) for v in before], after=[round(x, 3), round(y, 3)],
                                      closer_cm=round(100 * moved, 1)))
            self.log(f"step {self.steps}: {self.burst['T']} s -> {100 * moved:+.1f} cm closer")
            self.no_progress = 0 if moved >= MIN_PROGRESS_M else self.no_progress + 1
            if self.single is not None:
                self.finish(True, f"single burst {self.single} s: {100 * moved:+.1f} cm closer by the camera")
                return
            if self.no_progress >= MAX_NO_PROGRESS:
                self.finish(False, f"{MAX_NO_PROGRESS} steps in a row without {100 * MIN_PROGRESS_M:.0f} cm of progress")
                return
        p = plan(x, y, self.arm["bursts"], self.no_progress)
        if self.single is not None and "refuse" not in p:
            p = dict(T=float(self.single), vtheta=VTH_STRAIGHT)
        if "refuse" in p:
            self.finish(False, p["refuse"])
        elif "done" in p:
            self.finish(abs(math.hypot(x, y) - STOP_R) <= STOP_TOL, p["done"])
        elif self.dry:
            self.finish(True, f"dry: ball x {x:.3f} y {y:.3f}, would burst {p}")
        elif self.steps >= MAX_STEPS:
            self.finish(False, f"{MAX_STEPS} steps used, ball still {math.hypot(x, y):.2f} m away")
        else:
            self.burst = p
            self.enter("ready", f"ball x {x:.3f} y {y:.3f}: burst {p}")

    def p_ready(self, st, now):
        self.stand()           # the head back to neutral
        if now - self.t_phase > HEAD_UP_S + STILL_WAIT_S:
            self.finish(False, "the body did not come to rest before a step")
        elif now - self.t_phase >= HEAD_UP_S and self.still(now) and st.get("ok", True):
            self.steps += 1
            # written once: walk_wired restarts for_s on every rewrite
            self.send_fn({"mode": "walk", "vx": VX, "vy": 0.0, "vtheta": self.burst["vtheta"], "for_s": self.burst["T"]})
            self.enter("burst", f"step {self.steps}")

    def p_burst(self, st, now):
        if now - self.t_phase >= self.burst["T"] + 0.1:
            self.stand()
            self.enter("settle")

    def p_settle(self, st, now):
        waited = now - self.t_phase
        if waited >= SETTLE_S and self.still(now):
            self.start_look(f"after step {self.steps}")
        elif waited > SETTLE_S + STILL_WAIT_S:
            self.finish(False, f"still moving {waited:.0f} s after step {self.steps}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd", default="/tmp/duck_cmd.json")
    p.add_argument("--log", default=None)
    p.add_argument("--preflight", action="store_true", help="say which arm the brain runs and exit")
    p.add_argument("--dry", action="store_true", help="one look, print the planned burst, no step")
    p.add_argument("--single", type=float, default=None, metavar="T", help="one burst of T s (0.1-1.5) between two looks")
    p.add_argument("--max-s", type=float, default=150.0)
    p.add_argument("--trace", default=os.path.expanduser(f"~/duck-logs/steplook_{time.strftime('%m%d-%H%M%S')}.jsonl"))
    a = p.parse_args()
    if a.single is not None and not 0.1 <= a.single <= 1.5:
        p.error("--single takes 0.1 to 1.5 s")
    log_path = a.log or ap.find_log()
    if not log_path:
        sys.exit("no walk_wired log: start the brain from the app (Balance stand) first")
    tail = sp.SeeTail(log_path)
    time.sleep(0.5)
    st = tail.latest()
    arm, why = load_arm(log_path)
    if why is None and (st is None or time.time() - st["t"] > 3.0):
        why = "the brain is not ticking (crashed, or the disk is full)"
    if why is None and st["mode"] != "walk":
        why = f"the brain is in {st['mode']} mode: press Balance stand first"
    if why or a.preflight:
        print(json.dumps(dict(ok=not why, why=why or "preflight passed", arm=arm and arm["name"])))
        return
    signal.signal(signal.SIGTERM, sp._interrupt)
    import functools

    import ball_detect
    from duck_kin import DuckKin

    lo, hi = sp.COLORS["orange"]
    detect = ball_detect.BallDetect(lo, hi, ap.MIN_AREA,
                                    learned=functools.partial(ball_detect.learned_target.detect, threads=2))
    send, cam = ap.send_file(a.cmd), ap.ShmCamera()
    ctl = StepLook(DuckKin(), cam.latest, tail.latest, tail.at, send, detect, arm, single=a.single, dry=a.dry,
                   log=lambda s: print(s, flush=True))
    try:
        end = time.time() + a.max_s
        while ctl.phase not in ("done", "failed"):
            if time.time() > end:
                ctl.finish(False, f"time limit {a.max_s} s")
                break
            ctl.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        print("interrupted: walk at zero command written")
    finally:
        cam.close()
        os.makedirs(os.path.dirname(a.trace), exist_ok=True)
        with open(a.trace, "w") as f:
            f.write(json.dumps(dict(arm=arm, brain_log=log_path)) + "\n")
            for row in ctl.trace:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps(dict(result=ctl.result)) + "\n")
    print(json.dumps(ctl.result))
    print(f"trace {a.trace}")


if __name__ == "__main__":
    main()
