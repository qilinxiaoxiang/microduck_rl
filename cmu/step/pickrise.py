"""Pick-rise demo: step-look-step to the ball, then the seeing pick's own last centimetres and the kneel, holding the
ball at the end (cmu/pick_rise/EXPERIMENT-pickrise.md).

  walk   steplook.StepLook unchanged: bursts and looks until the ball is in the stop window (18 cm ahead +- 2.5)
  kneel  seepick.SeePick's T2 flow from there, with the kneel revision "dk" (seepickpix_dk61.onnx, the kneel in
         arms.json "kneel") and hold-on: aim (the head on the ball, only on a still body), align (turns and creeps,
         as T2 did on the robot), then {"mode": "seepick", "policy": "dk", "phase": 0.89, "rise_old": false}; at the
         hold phase the jaw decides, hold HOLD_S, then monitor (hold the pose, torque on). One kneel per run, no retry.

What the kneel stage changes from seepick --t2: the pick window is centred on the kneel's own target (KNEEL_X, from
its sim grid) instead of T2's (T2_WIN_*, T2_Y), and the pick command carries no yaw_off (T2's head-yaw correction; the
dk kneel never trained with it, as dockpick.py). Everything else is seepick.py as it ran on the robot.

Installed beside steplook.py on the A7Z (cmu/step/step.sh install):
  ~/.venvs/duck-policy/bin/python pickrise.py --preflight     which walk arm and kneel the brain runs, and exit
  ~/.venvs/duck-policy/bin/python pickrise.py --dry           walk and aim, print the kneel it would start, no kneel
  ~/.venvs/duck-policy/bin/python pickrise.py --kneel-only    no walk: the ball already near the feet (8-15 cm ahead)
  ~/.venvs/duck-policy/bin/python pickrise.py                 the whole demo
"""
import argparse
import json
import math
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as ap  # noqa: E402
import seepick as sp  # noqa: E402
import steplook as sl  # noqa: E402

KNEEL_SLOT = "seepickpix_dk61.onnx"   # walk_brain.PICK_REVISIONS["dk"]
POLICY = "dk"
HOLD_S = 5.0               # s held at the hold phase before monitor (seepick.HOLD_MAX_S 10)
KNEEL_X = 0.0945           # m ahead of the foot-frame origin: the kneel's best spot in sim (e2e.py --kneel-grid, fa_s3)
KNEEL_WIN_X = (-0.012, 0.012)   # m around KNEEL_X: its sim grid holds within about +-1.5 cm
KNEEL_WIN_Y = 0.015        # m: its grid spans +-1.5 cm sideways
NEAR_START = (0.06, 0.25)  # m: --kneel-only takes a ball this far ahead (the aim's lowest look sees 0.07-0.19)


def kneel_window():
    """The kneel's pick window in seepick's T2 constants (this process only)."""
    sp.T2_WIN_X = (KNEEL_X + KNEEL_WIN_X[0], KNEEL_X + KNEEL_WIN_X[1])
    sp.T2_WIN_Y = (-KNEEL_WIN_Y, KNEEL_WIN_Y)
    sp.T2_Y = 0.0
    sp.PICK_SLACK = 0.0    # nearer than the window is a step back, not a bite


def load_kneel(log_path, arms_file=sl.ARMS_FILE):
    """(kneel, None) when the brain loaded arms.json's kneel in KNEEL_SLOT, else (None, why)."""
    meta = (log_path[:-6] if log_path.endswith(".jsonl") else log_path) + ".meta.jsonl"
    try:
        with open(meta) as f:
            shas = json.loads(f.read().strip().splitlines()[-1])["model_sha"]
        with open(arms_file) as f:
            kneel = json.load(f)["kneel"]
    except (OSError, ValueError, KeyError, IndexError) as e:
        return None, f"cannot tell which kneel the brain runs ({type(e).__name__}: {e})"
    got = shas.get(KNEEL_SLOT)
    if got is None:
        return None, f"the brain has no {KNEEL_SLOT}: install the kneel (step.sh install) and restart (step.sh arm)"
    if got != kneel["sha"]:
        return None, f"the brain's {KNEEL_SLOT} is {got}, arms.json's kneel {kneel['name']} is {kneel['sha']}"
    return kneel, None


class PickRise(sl.StepLook):
    def __init__(self, kin, latest_frame, state, state_at, send, detect, arm, kneel_only=False, hold_s=HOLD_S,
                 dry=False, now=None, log=print):
        super().__init__(kin, latest_frame, state, state_at, send, detect, arm, dry=False, now=now, log=log)
        self.t2, self.policy, self.hold_s, self.tries = True, POLICY, hold_s, 1
        self.demo_dry, self.kneel_only = dry, kneel_only
        self.stage = "kneel" if kneel_only else "walk"
        self.walk_result = None

    def p_start(self, st, now):
        if st.get("jaw_deg") is None:
            sp.SeePick.finish(self, False, "the jaw is not calibrated or not read (jaw_deg missing in walk_wired's log)",
                              stop=False)
            return
        if self.kneel_only:
            self.start_kneel(None, "kneel only")
        else:
            super().p_start(st, now)

    def start_kneel(self, est, why):
        """seepick's T2 flow from the aim: a still body, the head on the ball, AIM_N detections, then align. The
        estimate only points the first aim (kneel only: the kneel's spot, which the lowest look covers): there is no
        pick_here, so a ball the aim does not see is never bitten blind on a stale guess (seepick's own blind pick
        after a creep that ran the ball under the picture stays)."""
        self.stage, self.dry = "kneel", self.demo_dry
        x, y = est if est is not None else (KNEEL_X, 0.0)
        self.try_n, self.align_n, self.app, self.meas = 1, 0, None, []
        self.obj = dict(x=float(x), y=float(y), size=sp.DEFAULT_SIZE, t=self.now(), cut=False)
        self.sight, self.lost_backs, self.blind_ok, self.blind_pick, self.aim_gate = (x, y), 0, False, False, None
        self.pick_here, self.head, self.burst_m = None, None, None
        self.enter("aim", why)

    def p_aim(self, st, now):
        super().p_aim(st, now)
        if self.phase == "align" and self.kneel_only and self.obj and not (
                NEAR_START[0] <= math.hypot(self.obj["x"], self.obj["y"]) <= NEAR_START[1]):
            self.finish(False, f"kneel only: ball at x {self.obj['x']:.3f} y {self.obj['y']:.3f}, not "
                               f"{NEAR_START[0]}-{NEAR_START[1]} m ahead")

    def p_burst(self, st, now):   # the walk's burst, or the kneel stage's creep (seepick's p_burst)
        if self.stage == "kneel":
            sp.SeePick.p_burst(self, st, now)
        else:
            super().p_burst(st, now)

    def pick_cmd(self):
        cmd = super().pick_cmd()
        cmd["object"].pop("yaw_off", None)   # T2's head-yaw correction: the dk kneel never trained with it
        return cmd

    def finish(self, ok, why, stop=True):
        if self.stage == "walk":
            self.walk_result = dict(ok=ok, why=why, steps=self.steps, aims=self.aims, est=self.est)
            if ok and self.est is not None:
                self.start_kneel(self.est, f"walk done ({why})")
                return
            why = f"walk: {why}"
        super().finish(ok, why, stop)
        self.result.update(stage=self.stage, walk=self.walk_result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd", default="/tmp/duck_cmd.json")
    p.add_argument("--log", default=None)
    p.add_argument("--preflight", action="store_true", help="say which walk arm and kneel the brain runs and exit")
    p.add_argument("--dry", action="store_true", help="walk and aim, print the kneel it would start, no kneel")
    p.add_argument("--kneel-only", action="store_true", help="no walk: the ball already 8-15 cm ahead")
    p.add_argument("--hold-s", type=float, default=HOLD_S)
    p.add_argument("--max-s", type=float, default=200.0)
    p.add_argument("--trace", default=os.path.expanduser(f"~/duck-logs/pickrise_{time.strftime('%m%d-%H%M%S')}.jsonl"))
    a = p.parse_args()
    if not 0 < a.hold_s <= sp.HOLD_MAX_S:
        p.error(f"--hold-s takes 0 to {sp.HOLD_MAX_S}")
    log_path = a.log or ap.find_log()
    if not log_path:
        sys.exit("no walk_wired log: start the brain from the app (Balance stand) first")
    tail = sp.SeeTail(log_path)
    time.sleep(0.5)
    st = tail.latest()
    arm, why = sl.load_arm(log_path)
    kneel, why_k = load_kneel(log_path)
    why = why or (None if a.dry else why_k)   # a dry run never starts the kneel
    if why is None and (st is None or time.time() - st["t"] > 3.0):
        why = "the brain is not ticking (crashed, or the disk is full)"
    if why is None and st["mode"] != "walk":
        why = f"the brain is in {st['mode']} mode: press Balance stand first"
    if why or a.preflight:
        print(json.dumps(dict(ok=not why, why=why or "preflight passed", arm=arm and arm["name"],
                              kneel=kneel and kneel["name"])))
        return
    kneel_window()
    signal.signal(signal.SIGTERM, sp._interrupt)
    import functools

    import ball_detect
    from duck_kin import DuckKin

    lo, hi = sp.COLORS["orange"]
    detect = ball_detect.BallDetect(lo, hi, ap.MIN_AREA,
                                    learned=functools.partial(ball_detect.learned_target.detect, threads=2))
    send, cam = ap.send_file(a.cmd), ap.ShmCamera()
    ctl = PickRise(DuckKin(), cam.latest, tail.latest, tail.at, send, detect, arm, kneel_only=a.kneel_only,
                   hold_s=a.hold_s, dry=a.dry, log=lambda s: print(s, flush=True))
    try:
        end = time.time() + a.max_s
        while ctl.phase not in ("done", "failed"):
            if time.time() > end and ctl.phase != "pick":   # never write over a kneel in progress
                ctl.finish(False, f"time limit {a.max_s} s")
                break
            ctl.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        send(sp.MONITOR if ctl.phase == "pick" else {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        print("interrupted: " + ("monitor (hold the pose)" if ctl.phase == "pick" else "walk at zero command") + " written")
    finally:
        cam.close()
        os.makedirs(os.path.dirname(a.trace), exist_ok=True)
        with open(a.trace, "w") as f:
            f.write(json.dumps(dict(arm=arm, kneel=kneel, brain_log=log_path)) + "\n")
            for row in ctl.trace:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps(dict(result=ctl.result)) + "\n")
    print(json.dumps(ctl.result))
    print(f"trace {a.trace}")


if __name__ == "__main__":
    main()
