"""Find an object of a named colour, walk to it, pick it up with the learned seeing pick, and retry when it fails or the
object slips out of the beak (final-goal steps 1-4, D-019; cmu/grasp/PLAN-seeing-pickup.md, the coordinator layer).

Runs on the A7Z beside walk_wired.py, like approach.py, whose controller it reuses for finding and walking. Phases:
  approach  approach.Approach with this colour's detector: scan, walk, stop with the object about 10 cm ahead
  aim       standing, head aimed at it, MEASURE_N detections -> object centre (foot frame) and width
  align     inside the pick window around the trained spot (SPOT, +-PICK_WIN): pick; to the side: turn in place toward
            it; too far: a short walk burst; too close: give up this try (walk61 does not step back)
  pick      {"mode": "seepick", "object": ...} to walk_wired; every new frame that sees the object near the last
            estimate updates it (projected with the joints and gravity logged at its capture time); walk_wired runs
            the policy, drives the jaw on the trained timeline and hands back to stand61 after the rise
  verify    after the hand-back, walk_wired's jaw_holding over VERIFY_S: holding -> hold, else retry
  hold      watch the jaw: holding lost for SLIP_S (the jaw closed past GRIP_MIN, i.e. the object fell out) is a slip
            -> the head searches and the next try starts from approach
At most --tries picks. A walk_wired guard trip (mode monitor) or stale state ends the run.

The policy was trained with the object at SPOT = (0.087, 0.0) m from the standing root (the foot-frame origin of
duck_kin) +-3-4 cm (sim: 80 % at +-3 cm, 71 % at +-5 cm, 31-39 % when the object moves before the bite).

  ~/.venvs/duck-policy/bin/python seepick.py --color orange          on the A7Z, walk_wired running, jaw calibrated
  ~/.venvs/duck-policy/bin/python seepick.py --color orange --t2      the pick is T2 (seepickpix61.onnx): the same phases,
                                                                      the pick command carries the ball's picture box
  ~/.venvs/duck-policy/bin/python seepick.py --color green --dry     approach, aim and align only, no pick
  ~/.venvs/duck-policy/bin/python seepick.py --color green --observe 10   detections only, no motion
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as ap  # noqa: E402

SPOT = (0.087, 0.0)        # m, foot frame: the trained object position (static grasp OBJECT_NOMINAL_XY)
PICK_WIN = 0.03            # m: |dx| and |dy| within this -> pick. Robot 2026-10-30: the pick at dx +1.1 dy -1.5 cm held
                           # (seepick_1030-032226); the picks at dx +4.6 (seepick_1030-044224) and dx +2.5 dy +4.9 cm
                           # (seepick_1030-044506) both came up empty, so the trained +-5 cm does not hold on the robot.
                           # Was 0.05 after seepick_1030-025920 (dx +4.1 cm: a blind 0.6 s walk burst lost the ball under the
                           # beak); now the align re-measures after every move instead (MAX_ALIGN), so a miss costs a turn
                           # or a burst, not the ball.
# T2 window, foot frame (t2-chain-20261101/README.md). Robot holds: T2 x 0.100 y -0.021, seepick61 x 0.098 y -0.015. Robot misses: T2 x 0.122
# y -0.005, seepick61 x 0.133 y 0 and x 0.112 y +0.049. As-built sim, real sphere (window_sweep.py): pad error at close grows with the offset,
# about 2 cm at dx +3 and 3 cm at dx +4, held 0 of 60 from dx +2 to +6. A wider window only bites air. Two holds: the centre is thin evidence.
# T2 closes at one spot whatever the ball (beak_at_close.py): run 9 missed at x 0.110 y -0.028 and x 0.102 y -0.007. x has no correction and stays
# about +-0.8 cm around the hold; head yaw (T2_YAW_M_RAD, walk_brain yaw_off) carries y within about +-1.8 cm of T2_Y. Sim yaw_sweep.py: dy -2 err 2.1 -> 0.8 cm.
T2_WIN_X = (0.090, 0.108)
T2_WIN_Y = (-0.045, 0.008)
T2_YAW_M_RAD = 0.060
T2_Y = -0.018              # m: where the creep and the turn steer the object; the two holds sat here, the walk veers about 2 cm per creep
NUDGE_FROM = 0.135         # m: from here in the creep is timed, and never stops on what it sees. Stop-on-sight gave 0.1 s and no move at 0.125 and
NUDGE_S = 1.2              # s  a walk until the ball left the picture at 0.124. Creeps of 1-1.5 s moved 1-5 cm (12 creeps, robot 2026-11-01)
AIM_MIN_FRAMES = 3         # T1 and T2: fewer frames than this is no sighting (x 0.037 width 8 mm from 2 frames ended run 8's good try)
AIM_GATE_M = 0.06          # m: T2's aim after a creep only takes detections this near the last sighting. Readings on the move are 1.5 cm off and the ball can
#                            shift 3 cm in y: a 4 cm gate threw out the real ball at the end of run 12's first creep, and it picked blind at dx +3.5
NEAR_LOOK_R = 0.17         # m: an object estimated nearer than this is aimed at with the lowest look (approach.NEAR_HEAD), whose
#                            picture covers x 0.07-0.19 m at the centre line (robot 2026-11-12), and the creep keeps that look
BACK_MAX_S = 1.2           # s: the step back ends here at the latest
FEET_R = 0.13              # m: last seen nearer than this and not in the aim's picture: it is under the picture at the toes
#                            (the lowest look starts at x 0.073, the toe front edge is at 0.055): step back, no new search
LOST_BACKS = 2             # step-backs for a ball lost at the feet before the normal search takes over
BACK_VX = -0.4             # m/s: robot 2026-11-12, standing start: -0.25 for 2.5 s leaned back 5 deg and never stepped (actions
#                            under 0.13); -0.4 stepped from 0.4 s, 2 s and 1.5 s runs stayed up (max tilt 11.4 deg). The as-built
#                            sim steps back at -0.20, which the robot does not.
SIGHT_LOST_S = 0.4         # s: a creep or step back with no sighting of the object for this long stops. Robot 2026-11-12
#                            (seepick_1112f, recorded frames): the creep kept the lowest look, walk61 dropped the neck 0.3 rad
#                            more once it stepped, the ball left the top of the picture after 0.3 s, the creep walked its full
#                            1.5 s unseen and the left foot came down on the ball; the duck fell.
TRACK_GAIN = 0.2           # rad of head pitch per picture height of error, per sighting: the head follows the object while walking
BACK_GATE_M = 0.12         # m: detections this near the last sighting are the object while it moves away in the picture
BLIND_R = 0.15         # m: nearer than this the object can be under the standing picture (bottom edge 0.10-0.125 m at
#                            the lowest look, ruler 2026-10-30); not seen there means pick at the last known position.
#                            Was 0.12: robot 2026-11-06 the duck turned 16 deg toward a ball at 0.12 m, lost it under
#                            the picture, its estimate read 0.121 and the run gave up without a pick. A wrong blind
#                            pick costs one empty bite; a search from there cannot see the ball either.
PICK_FAR_X = 0.105         # m: the far edge of the pick window. Every robot hold sat at x 0.098-0.100 (10-30 03:22, 11-01 run 7);
#                            the bites of 2026-11-12 at x 0.110, 0.110, 0.112 and 0.125 closed on nothing (jaw to -1.1 deg at
#                            -18 mA; a held ball stops it at 10.1 deg, -117 mA), and the sim held 0 of 60 from dx +2 cm.
PICK_SLACK = 0.01          # m: T2 only, the fixed margin on the near edge of its window (a creep from there walks over the ball).
#                            The window does not widen after MAX_ALIGN moves: the bites beyond x 0.105 all closed on nothing
TURN_RATE = ap.VTH_SPIN     # rad/s in place (0.3; 0.5 rolled the duck over on carpet, robot 2026-11-06)
TURN_MIN = math.radians(8)  # a bearing under this is not worth a turn (the in-place turn coasts)
TURN_MAX_S = 3.0
LOOK_TURNS = (-1, -1, -1, -1, -1, -1, -1)   # body turns between head scans when nothing is in view: always the same way,
#                                     7 x about 50 deg = once round. Was (-1, -1, 1, 1, 1, 1): two right, then back through the
#                                     two directions already scanned before two left, from the days of the charger lead; on
#                                     the battery it only looked again where it had looked (Shawn 2026-11-12: "再转一次身就能
#                                     看到了，结果又转回去了"). The
LOOK_SETTLE_S = 0.7                 # head back at the default pose before the turn: run 8 fell in the one spin that began with the head ramping down from 1.33 rad
LOOK_TURN_S = 3.5                   # scan covers +-60 deg; a 3.5 s spin keeps about 50 deg on the robot (2026-11-01)
BURST_MAX_S = 1.5          # s: the creep toward a too-far object ends here at the latest (a standing-start walk61 moved
#                            0-5 cm per 0.6-0.9 s command on the robot 2026-10-30: the first steps are in place)
BURST_STOP_X = 0.125       # m: the creep stops when the object is measured here or nearer; walk61 coasts about 4 cm after a
#                            stop (approach.STOP_R), which lands it at the trained spot
MAX_ALIGN = 6              # align moves per try; still outside the pick window after them: no bite, the try ends. A
#                            standing-start burst moves 0-5 cm (robot 2026-10-30, seepick_1030-044506), so 3 was not enough
BURST_INSIDE_M = 0.02      # m: a creep that starts already inside BURST_STOP_X walks until the camera shows this much; it stopped on its first tick
#                            and moved nothing, six times, then picked at dx +3.5 cm and bit air (robot 2026-11-01)
AIM_N, AIM_S = 5, 3.0      # detections for the pick measurement, and the time allowed
SETTLE_S = 1.0             # after a stop, before measuring (approach.SETTLE_S is 1.2 after a walk)
TRACK_GATE = 0.04          # m: a detection further than this from the estimate during the pick is another object
TRACK_GATE_HERE = 0.08     # m: the gate on a --pick-here try, whose estimate is a person's guess (robot 2026-10-30: the
#                            guess was 11 cm off and three true sightings during the bend were gated away)
PIX_W, PIX_H = 640, 360    # the picture T2 reads
PIX_SEEN_S = 0.15          # s: a detection this recent counts as the newest frame's (the camera runs about 15-30 fps)
PIX_BLIND_W = 16.0         # px: the width of the ball at about 10 cm when it has no detection (12 mm ball, robot frames)
CMD_PERIOD = 0.1           # s: the pick command is rewritten at least this often while the camera sees the object
PICK_MAX_S = 6.0           # the seeing pick takes 3.6 s to the hand-back (phase 0.9 x 4 s)
VERIFY_S = 0.5
SLIP_S = 0.2
HOLD_WATCH_S = 10.0        # watch for a slip this long after a pick, then done
DEFAULT_SIZE = 0.013       # m, the training mean width (BODY_SLOT_MEAN)
SIZE_RANGE = (0.008, 0.03)
# OpenCV HSV (H 0-180). Orange is approach.py's tested range; the others are starting points to check on real frames
# (orange_target.py --hist on a snapshot).
# orange: robot 2026-11-12 (seepick_1112b, office light) the 12 mm ball read H 1-5, S 67-115 (median 90) at about 0.4 m, so S >= 120
# dropped the whole ball and the scan never saw it; S 70 / H 0 found only the ball (118 px) in the same frame, floor and cabinets clear.
COLORS = {"orange": ((0, 70, 120), (22, 255, 255)), "yellow": ((22, 100, 120), (35, 255, 255)),
          "green": ((40, 80, 60), (85, 255, 255)), "blue": ((95, 100, 60), (130, 255, 255)),
          "red": ((170, 120, 90), (180, 255, 255)), "pink": ((145, 60, 120), (170, 255, 255))}


class SeePick:
    """The coordinator; camera, state, commands and the approach controller are injected (fakes in test_seepick.py).

    state() / state_at(t) return approach.parse_state dicts plus jaw_holding and sp_phase (SeeTail)."""

    def __init__(self, kin, latest_frame, state, state_at, send, detect, make_approach, tries=3, dry=False,
                 now=time.time, log=print, pick_here=None, track_gate=None, t2=False):
        self.t2 = t2
        self.kin, self.latest_frame, self.state_fn, self.state_at = kin, latest_frame, state, state_at
        self.send_fn, self.detect, self.make_approach = send, detect, make_approach
        self.tries, self.dry, self.now, self.log = tries, dry, now, log
        self.track_gate = track_gate  # m, overrides TRACK_GATE / TRACK_GATE_HERE when given
        self.pick_here = pick_here   # (x, y) foot frame: the object is already at the feet; the first try skips the
        #                              approach, aims there and picks on this estimate even if the camera cannot see
        #                              it (the beak's blind zone); frames seen during the bend still correct it
        self.helper = ap.Approach(kin, latest_frame, state, state_at, send, detect, now=now, log=lambda s: None,
                                  obj_radius=DEFAULT_SIZE / 2)   # its projection and head aiming, never stepped
        self.phase, self.t_phase, self.result = "start", now(), None
        self.app, self.try_n, self.align_n = None, 0, 0
        self.obj, self.meas, self.last_idx = None, [], None   # obj: dict x, y, size, t (capture time)
        self.turn = None          # (direction, target yaw change, start yaw)
        self.head, self.burst_m = None, None   # the aim head pose (kept during a creep), the last sighting during it
        self.cmd_t, self.picked_t, self.seen_pick = 0.0, None, False
        self.holding_since, self.lost_since = None, None
        self.trace, self.attempts = [], []
        self.blind_ok, self.blind_pick, self.last_y = False, False, 0.0   # T2: the creep ended this near; pick unseen; side it was last on
        self.aim_gate = None      # T2: the last sighting of the creep, which the next aim is gated on
        self.look_n = 0           # body turns made while looking for the object
        self.sight = None         # (x, y) of the last sighting (measure, or the approach's arrival) since the last search
        self.lost_backs = 0       # step-backs made for a ball lost at the feet since it was last seen
        self.pick_late = False    # the pick ran past PICK_MAX_S and has not handed back yet
        self.rock_t, self.step_t = -1e9, None   # last time |trunk gyro| > ap.STILL_GYRO; the previous step's time

    # --- helpers -----------------------------------------------------------------------------------------------
    def enter(self, phase, why=""):
        self.log(f"[{self.now():.1f}] {self.phase} -> {phase}{': ' + why if why else ''}")
        self.trace.append(dict(t=self.now(), phase=phase, why=why))
        self.phase, self.t_phase = phase, self.now()

    def finish(self, ok, why, stop=True):
        self.result = dict(ok=ok, why=why, tries=self.try_n, attempts=self.attempts)
        if stop:
            self.send_fn({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        self.enter("done" if ok else "failed", why)

    def stand(self, head=None):
        cmd = {"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0}
        if head:
            cmd["head"] = {k: v for k, v in head.items() if k in ("head_pitch", "head_yaw", "neck_pitch")}
        self.send_fn(cmd)

    def measure(self, st, gate=None, gate_m=None):
        """New frame -> (x, y, width, capture time) of the object centre in the capture-time foot frame, or None."""
        fr = self.latest_frame()
        if fr is None:
            return None
        frame, t_cap, idx = fr
        if idx == self.last_idx or self.now() - t_cap > ap.FRAME_MAX_AGE_S:
            return None
        self.last_idx = idx
        sc = self.state_at(t_cap) or st
        h, w = frame.shape[:2]
        # while the robot moves or bites, only a box from this very frame counts: a background box is about a second old
        det = self.helper.detect_at(frame, sc, w, h, late_ok=self.phase == "aim")
        if det is None:
            return None
        if det.get("ctx") is not None and det["ctx"] is not sc:
            sc = det["ctx"]
            t_cap = sc["t"]
        x0, y0, bw, bh = det["bbox"]
        a = self.kin.pixel_to_ground(x0, y0 + bh, sc["q"], sc["gb"], w, h, z=0.0)
        b = self.kin.pixel_to_ground(x0 + bw, y0 + bh, sc["q"], sc["gb"], w, h, z=0.0)
        size = DEFAULT_SIZE if a is None or b is None else float(np.clip(math.dist(a, b), *SIZE_RANGE))
        self.helper.obj_radius = size / 2
        p = self.helper.project(det, sc, w, h)
        if p is None or p[0] < 0.03:   # the duck's own coloured parts project under the body
            return None
        gate_given = gate_m
        gate_m = TRACK_GATE_HERE if (self.pick_here is not None and self.app is None) else TRACK_GATE
        gate_m = gate_given if gate_given is not None else gate_m
        if self.track_gate is not None:
            gate_m = self.track_gate   # --track-gate; 0 keeps the given estimate and only records what the camera saw
        if gate is not None and math.hypot(p[0] - gate[0], p[1] - gate[1]) > gate_m:
            if self.phase == "pick":
                self.seen_elsewhere = (float(p[0]), float(p[1]), float(t_cap))   # the bite may have knocked it there
            self.trace.append(dict(t=self.now(), phase=self.phase, frame=idx, gated=[p[0], p[1]], t_cap=float(t_cap),
                                   bbox=[int(v) for v in det["bbox"]], u=float(det.get("base_cx", det["cx"])),
                                   v=float(y0 + bh), w=w, h=h, q=sc["q"], gb=list(sc["gb"])))  # cam_calib obs row
            return None
        m = dict(x=float(p[0]), y=float(p[1]), size=size, t=float(t_cap), cut=bool(p[2]), frame=idx,
                 u=float(x0 + bw / 2), v=float(y0 + bh / 2), w=float(bw))
        self.trace.append(dict(t=self.now(), phase=self.phase, meas=m))
        self.sight = (m["x"], m["y"])
        return m

    def pick_cmd(self):
        o = self.obj
        obj = {"x": round(o["x"], 4), "y": round(o["y"], 4), "size": round(o["size"], 4), "t": o["t"]}
        if self.t2:
            obj.update(self.pix_box(o), yaw_off=round((o["y"] - T2_Y) / T2_YAW_M_RAD, 3))
        return {"mode": "seepick", "phase": 1.0, "object": obj}

    def pix_box(self, o):
        """T2's picture slots: the last detection's box, seen while that detection is the newest frame. A ball under the
        picture (blind zone, no detection) reads as its projection clamped to the edge, the hand-over of training
        (T2-PIXEL.md, brain_sim_check.PixCamera.start)."""
        now = self.now()
        if "u" in o:
            return {"u": round(o["u"], 2), "v": round(o["v"], 2), "w": round(o["w"], 2), "seen": now - o["t"] < PIX_SEEN_S}
        st = self.state_fn()
        p = self.kin.ground_to_pixel(o["x"], o["y"], st["q"], st["gb"], PIX_W, PIX_H, z=0.0) if st else None
        u, v = (PIX_W / 2, PIX_H - 1) if p is None else (min(max(p[0], 0.0), PIX_W - 1), min(max(p[1], 0.0), PIX_H - 1))
        return {"u": round(float(u), 2), "v": round(float(v), 2), "w": PIX_BLIND_W, "seen": False}

    # --- the machine -------------------------------------------------------------------------------------------
    def step(self):
        now = self.now()
        st = self.state_fn()
        if st is None or now - st["t"] > ap.STATE_MAX_AGE_S:
            if now - self.t_phase > ap.STALE_START_S:
                self.finish(False, "no fresh walk_wired state")
            return
        if self.phase not in ("start", "approach", "done", "failed") and st["mode"] in ("monitor", "stand"):
            # a guard (tilt, action limit) or a person stood it down: writing walk back would hand a fallen or tilted
            # robot to the policy again (2026-11-06: aim kept writing walk to a duck lying at 50 deg)
            self.finish(False, f"walk_wired is in {st['mode']} mode (a guard trip or a manual command)", stop=False)
            return
        dt, self.step_t = (now - self.step_t if self.step_t is not None else 0.0), now
        if st.get("gyro") is not None and math.sqrt(sum(g * g for g in st["gyro"])) > ap.STILL_GYRO:
            self.rock_t = now
        if self.phase in ("aim", "align") and now - self.rock_t < ap.STILL_S:
            # no look-down head pose and no pick start on a rocking body (2026-11-06: the aim lowered the head to 1.4
            # right after a turn in place, trunk gyro 1-1.5 rad/s, and the duck fell); the phase clock pauses
            self.send_fn({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
            self.t_phase += dt
            return
        getattr(self, "p_" + self.phase)(st, now)

    def p_start(self, st, now):
        if st.get("jaw_deg") is None and not self.dry:
            self.finish(False, "the jaw is not calibrated or not read (jaw_deg missing in walk_wired's log)", stop=False)
            return
        if self.pick_here is not None:
            self.try_n, self.align_n, self.app, self.meas = 1, 0, None, []
            self.enter("aim", f"try 1: pick here, estimate x {self.pick_here[0]:.3f} y {self.pick_here[1]:.3f}")
            return
        self.next_try("start")

    def retry_here(self, why):
        """Another try at the object's last known position without a search: a standing robot cannot see the pick
        spot (BLIND_R), so after a miss or a slip it aims there again and, unseen, picks blind. Counts as a try."""
        if self.try_n >= self.tries:
            self.finish(False, f"{self.tries} tries used ({why})")
            return
        self.try_n += 1
        self.blind_ok, self.blind_pick, self.aim_gate = False, False, None
        self.align_n, self.meas = 0, []
        self.enter("aim", f"try {self.try_n}: {why}")

    def next_try(self, why):
        if self.try_n >= self.tries:
            self.finish(False, f"{self.tries} tries used ({why})")
            return
        self.try_n += 1
        self.blind_ok, self.blind_pick, self.aim_gate = False, False, None
        self.align_n = 0
        self.sight, self.lost_backs = None, 0
        self.app = self.make_approach()
        self.enter("approach", f"try {self.try_n}: {why}")

    def time_limit(self, why):
        """The run's time limit (--max-s). A pick in progress is not written over: walk_wired hands back first (p_pick
        bounds that), then the next call finishes the run. True once the run is finished."""
        if self.phase == "pick":
            return False
        self.finish(False, why)
        return True

    def back_lost(self, why):
        """Step back from a ball that is under the picture at the toes, the head in the lowest look, then aim again.
        Not a try and no body turn: a search from here cannot show it, and a turn puts a foot on it."""
        self.lost_backs += 1
        self.obj, self.meas = None, []
        self.blind_ok, self.blind_pick, self.aim_gate = False, False, None
        self.head = dict(ap.NEAR_HEAD)
        self.enter("back", f"{why}: step back {self.lost_backs} of {LOST_BACKS}")

    def p_approach(self, st, now):
        self.app.step()
        if self.app.phase == "done":
            self.meas = []
            if self.app.result.get("x") is not None:
                self.sight = (self.app.result["x"], self.app.result["y"])
            self.enter("aim", f"arrived at x {self.app.result.get('x')} y {self.app.result.get('y')}")
        elif self.app.phase == "failed":
            why = self.app.result.get("why")
            if "stand mode" in why:   # a guard stood it down, usually a fall: no try can run (run 10 burned three in 0.8 s)
                self.finish(False, why, stop=False)
                return
            if why == ap.LOST_AT_FEET and self.lost_backs < LOST_BACKS:
                self.back_lost("approach: lost at the feet")
                return
            if why == ap.NOT_IN_VIEW and self.look_n < len(LOOK_TURNS):   # looking around does not use up a pick try
                self.enter("look", f"nothing in view: body turn {self.look_n + 1} of {len(LOOK_TURNS)}")
            else:
                self.next_try(f"approach: {why}")

    def p_look(self, st, now):
        if now - self.t_phase < LOOK_SETTLE_S:
            self.stand()
            return
        if now - self.t_phase >= LOOK_SETTLE_S + LOOK_TURN_S:
            self.stand()
            self.look_n += 1
            self.app = self.make_approach()
            self.enter("approach", "scan again")
            return
        self.send_fn({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": LOOK_TURNS[self.look_n] * (-1 if self.last_y > 0 else 1) * TURN_RATE,
                      "for_s": ap.CMD_FOR_S})

    def p_aim(self, st, now):
        est = (self.obj["x"], self.obj["y"]) if self.obj else ((self.app.est if self.app else None) or self.pick_here or SPOT)
        head = self.helper.aim_head(st, *est)
        if math.hypot(*est) < NEAR_LOOK_R or self.lost_backs:   # after a step back from a ball lost at the feet: lowest look
            head = dict(ap.NEAR_HEAD)
        self.head = head
        self.stand(head)
        if now - self.t_phase < SETTLE_S:
            return
        m = self.measure(st, gate=self.aim_gate, gate_m=AIM_GATE_M) if self.aim_gate else self.measure(st)
        if m is not None:
            self.meas.append(m)
        if len(self.meas) >= AIM_N or (now - self.t_phase > SETTLE_S + AIM_S and len(self.meas) >= AIM_MIN_FRAMES):
            med = {k: float(np.median([q[k] for q in self.meas])) for k in ("x", "y", "size")}
            self.obj = dict(med, t=max(q["t"] for q in self.meas), cut=any(q["cut"] for q in self.meas))
            self.last_y, self.blind_ok, self.aim_gate = med["y"], False, None
            self.lost_backs = 0
            self.log(f"  object x {med['x']:.3f} y {med['y']:.3f} width {med['size'] * 1000:.0f} mm "
                     f"from {len(self.meas)} frames (spot {SPOT[0]:.3f} {SPOT[1]:.3f})")
            self.enter("align")
        elif now - self.t_phase > SETTLE_S + AIM_S:
            self.aim_gate = None
            if self.t2 and self.blind_ok and self.obj["x"] < T2_WIN_X[0] - PICK_SLACK:
                self.obj = None    # last seen already past the beak: find it again, do not bite the floor
                self.next_try("last seen past the beak")
                return
            if self.t2 and self.blind_ok:
                # the creep walked on from x <= NUDGE_FROM and the ball left the picture (it starts at 0.105): this is the pick
                # place, and another step runs it past the beak. No box: T2 gets the training hand-over, clamped to the edge
                self.log(f"  under the picture after the creep: picking, last seen x {self.obj['x']:.3f} y {self.obj['y']:.3f}")
                self.obj = {k: self.obj[k] for k in ("x", "y", "size")} | {"t": now}
                self.blind_pick = True
                self.enter("align")
                return
            if self.pick_here is not None and self.app is None:
                self.obj = dict(x=float(self.pick_here[0]), y=float(self.pick_here[1]), size=DEFAULT_SIZE, t=now)
                self.log("  not seen (blind zone): picking on the given estimate")
                self.enter("align")
                return
            # No pick on an object the camera does not show. The lowest look (approach.NEAR_HEAD) shows the floor from
            # 0.07 to 0.19 m ahead (robot 2026-11-12), which holds the whole pick window; "not seen" there means the
            # ball is not there. The old branch bit the last known position (seepick_1112a try 2, 1112c both tries:
            # Shawn "为什么还在瞎试 这个时候视野里根本不可能有球").
            # Last seen nearer than FEET_R (or already stepping back for a ball lost at the feet): it is under the
            # picture at the toes, so the robot steps back and aims again; a new search from here would not show it.
            last = (self.obj["x"], self.obj["y"]) if self.obj else self.sight
            if self.lost_backs < LOST_BACKS and (self.lost_backs or (last is not None and math.hypot(*last) < FEET_R)):
                self.back_lost("not seen at the pick spot, last seen at the feet")
                return
            self.obj = None
            self.next_try("not seen at the pick spot")

    def in_window(self, x, y):
        """The same window on every align move: it does not widen after MAX_ALIGN."""
        if self.t2:
            # nearer than the window but not "too close" still picks: a creep from there walks over the ball (run 12, x 0.086 to under the beak)
            return T2_WIN_X[0] - PICK_SLACK <= x <= T2_WIN_X[1] and T2_WIN_Y[0] <= y <= T2_WIN_Y[1]
        return SPOT[0] - PICK_WIN <= x <= PICK_FAR_X and abs(y - SPOT[1]) <= PICK_WIN

    def p_align(self, st, now):
        dx, dy = self.obj["x"] - SPOT[0], self.obj["y"] - SPOT[1]
        win = PICK_WIN
        if self.blind_pick or self.in_window(self.obj["x"], self.obj["y"]):
            self.blind_pick = False
            if self.dry:
                self.finish(True, f"dry run: would pick at dx {dx * 100:+.1f} dy {dy * 100:+.1f} cm")
                return
            self.attempts.append(dict(try_n=self.try_n, x=round(self.obj["x"], 3), y=round(self.obj["y"], 3),
                                      size=round(self.obj["size"], 3)))
            self.seen_pick, self.picked_t = False, None
            self.send_fn(self.pick_cmd())
            self.cmd_t = now
            self.seen_elsewhere = None
            self.enter("pick", f"dx {dx * 100:+.1f} dy {dy * 100:+.1f} cm")
            return
        if self.align_n >= MAX_ALIGN:
            self.next_try(f"still off the spot after {MAX_ALIGN} moves (dx {dx * 100:+.1f} dy {dy * 100:+.1f} cm)")
            return
        if self.obj["x"] < (T2_WIN_X[0] - PICK_SLACK if self.t2 else SPOT[0] - win):
            # walk61 walks backward (Pollen trains lin_vel_x -0.4..0.4); the robot needs BACK_VX to start stepping
            self.align_n += 1
            self.head = dict(ap.NEAR_HEAD)
            self.enter("back", f"too close (dx {dx * 100:+.1f} cm): step back")
            return
        self.align_n += 1
        bearing = math.atan2(self.obj["y"] - T2_Y if self.t2 else self.obj["y"], self.obj["x"])
        off_side = not T2_WIN_Y[0] <= self.obj["y"] <= T2_WIN_Y[1] if self.t2 else abs(dy) > win
        if off_side and abs(bearing) >= TURN_MIN:
            self.turn = (1 if bearing > 0 else -1, abs(bearing), st["yaw"])
            self.enter("turning", f"bearing {math.degrees(bearing):+.0f} deg")
        else:
            self.burst_m = self.obj   # the aim measurement: if it sat cut on the bottom edge, the creep's first step hides it
            if self.head is None:   # align entered without an aim (tests, a given estimate): look at the object
                self.head = self.helper.aim_head(st, self.obj["x"], self.obj["y"])
            self.enter("burst", f"dx {dx * 100:+.1f} cm")

    def track_head(self, m):
        """Head pitch command that keeps a sighting near the middle of the picture while the body walks (the neck and
        the trunk move under the camera): a sighting high in the picture lifts the head, one low in it lowers it."""
        if m is None or not self.head:
            return
        err = m["v"] / PIX_H - 0.5
        hp = self.head.get("head_pitch", ap.NEAR_HEAD["head_pitch"]) + TRACK_GAIN * err
        self.head = dict(self.head, head_pitch=round(max(0.35, min(ap.NEAR_HEAD["head_pitch"], hp)), 3))

    def p_back(self, st, now):
        """Step back from an object that is nearer than the pick window, the head in the lowest look so the camera
        keeps it; stop when it is measured at the pick spot or at BACK_MAX_S, then aim and measure again. A ball lost at
        the feet has no position to gate on: the first sighting is taken as it comes."""
        m = self.measure(st, gate=(self.obj["x"], self.obj["y"]), gate_m=BACK_GATE_M) if self.obj else self.measure(st)
        if m is not None:
            self.obj = m
            self.track_head(m)
        if (m is not None and m["x"] >= SPOT[0]) or now - self.t_phase >= BACK_MAX_S:
            why = f"object at x {m['x']:.3f}" if m is not None and m["x"] >= SPOT[0] else f"{BACK_MAX_S} s"
            self.stand(self.head)
            self.obj, self.meas = None, []
            self.enter("aim", f"step back done: {why}")
            return
        self.send_fn({"mode": "walk", "vx": BACK_VX, "vy": 0.0, "vtheta": 0.0, "for_s": ap.CMD_FOR_S,
                      "head": {k: v for k, v in self.head.items() if k in ("head_pitch", "head_yaw", "neck_pitch")}})

    def p_turning(self, st, now):
        if now - self.t_phase < LOOK_SETTLE_S:   # head to the default pose first: it must not move during the spin (approach.walk)
            self.stand()
            return
        d, target, yaw0 = self.turn
        done = abs(ap.wrap(st["yaw"] - yaw0)) >= target - math.radians(3)
        if done or now - self.t_phase > LOOK_SETTLE_S + TURN_MAX_S:
            self.stand()
            self.obj, self.meas = None, []
            self.enter("aim", "turned" if done else "turn time limit")
            return
        self.send_fn({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": d * TURN_RATE, "for_s": ap.CMD_FOR_S})

    def p_burst(self, st, now):
        """Creep toward a too-far object with the head on it: walk while the camera still sees it beyond BURST_STOP_X;
        stop at that mark (walk61 coasts the last centimetres), when it drops under the picture, or at BURST_MAX_S; then
        aim and measure again. Robot 2026-10-30: a blind 0.6 s burst moved anything from 0 to 5 cm."""
        ref = self.burst_m or self.obj
        m = self.measure(st, gate=(ref["x"], ref["y"]))
        if m is not None:
            self.burst_m = m
            self.track_head(m)
        nudge = self.t2 and self.obj["x"] <= NUDGE_FROM
        stop_x = BURST_STOP_X if self.obj["x"] > BURST_STOP_X else self.obj["x"] - BURST_INSIDE_M
        if not nudge and m is not None and m["x"] <= stop_x:
            why = f"object at x {m['x']:.3f}"
        elif m is None and self.burst_m is not None and now - self.burst_m["t"] > 0.4 and (
                self.burst_m.get("cut") or self.obj["x"] <= BURST_STOP_X or nudge):
            why = "object under the picture"   # this near a blind walk runs the object past the beak
        elif not self.t2 and now - max(self.t_phase, (self.burst_m or {}).get("t", 0.0)) > SIGHT_LOST_S:
            why = f"object not seen for {SIGHT_LOST_S} s"   # never walk at a near object the camera does not show
        elif now - self.t_phase >= (NUDGE_S if nudge else BURST_MAX_S):
            why = f"{NUDGE_S if nudge else BURST_MAX_S} s"
        else:
            vth = 0.0
            if self.t2 and self.burst_m:   # the walk veers: 0.158 y -0.004 became 0.110 y -0.040 in one creep (robot run 8)
                b = math.atan2(self.burst_m["y"] - T2_Y, self.burst_m["x"])
                vth = round(max(-ap.VTH_ARC, min(ap.VTH_ARC, ap.K_TURN * b)), 3)
            cmd = {"mode": "walk", "vx": ap.VX, "vy": 0.0, "vtheta": vth, "for_s": ap.CMD_FOR_S}
            if self.head:
                cmd["head"] = {k: v for k, v in self.head.items() if k in ("head_pitch", "head_yaw", "neck_pitch")}
            self.send_fn(cmd)
            return
        self.stand(self.head)
        # a ball that left the bottom of the picture keeps its last sighting: the aim's blind zone picks there. Without it
        # the aim fell back on the walk's stale estimate and started a new try (robot 2026-11-01, try 1 of both runs)
        if self.t2:    # the aim gates on this sighting; a detection anywhere else is not the ball
            # picking unseen only makes sense when the ball went under the picture or the timed creep ran: a creep stopped at its mark still sees it
            self.obj, self.blind_ok = dict(self.burst_m or self.obj), nudge or why == "object under the picture"
            self.last_y, self.aim_gate = self.obj["y"], (self.obj["x"], self.obj["y"])
        else:
            self.obj = dict(self.burst_m) if why == "object under the picture" and self.burst_m else None
        self.meas, self.burst_m = [], None
        self.enter("aim", f"creep done: {why}")

    def p_pick(self, st, now):
        if st["mode"] == "seepick":
            self.seen_pick = True
        elif self.pick_late:   # the late pick has handed back: now the run ends
            self.pick_late = False
            self.finish(False, f"seeing pick did not hand back within {PICK_MAX_S} s (handed back after {now - self.t_phase:.1f} s)")
            return
        elif self.seen_pick:   # walk_wired handed back to stand61 after the rise
            self.enter("verify")
            return
        if now - self.t_phase > PICK_MAX_S:
            if st["mode"] == "seepick":
                # the pick is still running: no mode change is written into the bend or the rise; wait for the hand-back
                self.pick_late = True
                return
            self.finish(False, f"seeing pick did not hand back within {PICK_MAX_S} s (mode {st['mode']})")
            return
        if (st.get("sp_phase") or 0.0) >= 0.85:
            return   # never rewrite the pick command around the hand-back
        m = self.measure(st, gate=(self.obj["x"], self.obj["y"]))
        if m is not None:
            self.obj = m
        if m is not None or now - self.cmd_t >= CMD_PERIOD:
            self.send_fn(self.pick_cmd())
            self.cmd_t = now

    def p_verify(self, st, now):
        if now - self.t_phase < VERIFY_S:
            return
        rows = [self.state_at(t) for t in np.linspace(now - VERIFY_S, now, 5)]
        held = sum(bool(r and r.get("jaw_holding")) for r in rows) >= 3
        self.attempts[-1]["holding"] = held
        if held:
            self.holding_since, self.lost_since = now, None
            self.enter("hold", "the jaw holds something")
        elif getattr(self, "seen_elsewhere", None) is not None:
            # robot 2026-11-12 (seepick_1112a): the ball was in the beak, the bite knocked it 5 cm aside, the camera saw
            # it there (gated out as "another object") and the second try still bit the old spot blind. Seen away from
            # the estimate during the pick = it moved: look for it again.
            x, y, _ = self.seen_elsewhere
            self.seen_elsewhere, self.obj = None, None
            self.next_try(f"nothing between the bills after the rise; seen at x {x:.3f} y {y:.3f} during the pick")
        elif self.obj is not None and math.hypot(self.obj["x"], self.obj["y"]) < BLIND_R:
            self.retry_here("nothing between the bills after the rise")   # a miss leaves it about where it was
        else:
            self.next_try("nothing between the bills after the rise")

    def p_hold(self, st, now):
        if st.get("jaw_holding"):
            self.lost_since = None
        else:
            self.lost_since = self.lost_since or now
            if now - self.lost_since >= SLIP_S:
                self.attempts[-1]["slipped_after_s"] = round(now - self.holding_since, 2)
                # it drops in front of the beak, about at the trained spot, under the standing picture
                self.obj = dict(x=SPOT[0], y=SPOT[1], size=self.obj["size"] if self.obj else DEFAULT_SIZE, t=now)
                self.retry_here("slipped out of the beak")
                return
        if now - self.holding_since >= HOLD_WATCH_S:
            self.finish(True, "holding", stop=False)

    p_done = p_failed = lambda self, st, now: None


class SeeTail(ap.LogTail):
    """approach.LogTail plus the jaw and seeing-pick fields of walk_wired's log rows."""

    def poll(self):
        self.buf += self.f.read()
        *lines, self.buf = self.buf.split(b"\n")
        for ln in lines:
            try:
                row = json.loads(ln)
                s = ap.parse_state(row)
            except (ValueError, KeyError, TypeError):
                continue
            s.update(jaw_holding=bool(row.get("jaw_holding")), jaw_deg=row.get("jaw_deg"), sp_phase=row.get("sp_phase"))
            self.rows.append(s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--color", default="orange", choices=sorted(COLORS))
    p.add_argument("--hsv-lo", default=None, help="override the colour table, e.g. 40,80,60")
    p.add_argument("--hsv-hi", default=None)
    p.add_argument("--colour-only", action="store_true", help="orange: the colour filter alone, no learned detector")
    p.add_argument("--cmd", default="/tmp/duck_cmd.json")
    p.add_argument("--log", default=None, help="walk_wired log (default: the --log of the running walk_wired)")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--t2", action="store_true", help="pick with T2, the picture policy (seepickpix61.onnx)")
    p.add_argument("--dry", action="store_true", help="approach, aim and align only, no pick")
    p.add_argument("--observe", type=float, default=None, help="s; print detections only, no motion")
    p.add_argument("--max-s", type=float, default=180.0)
    p.add_argument("--pick-here", nargs=2, type=float, metavar=("X", "Y"),
                   help="m, foot frame: the object is already at the feet; skip the approach and pick there")
    p.add_argument("--track-gate", type=float, default=None, metavar="M",
                   help="override the tracking gate; 0 = never move the estimate on a sighting (sightings are traced)")
    p.add_argument("--trace", default=os.path.expanduser(f"~/duck-logs/seepick_{time.strftime('%m%d-%H%M%S')}.jsonl"))
    a = p.parse_args()
    import orange_target
    from duck_kin import DuckKin
    lo, hi = COLORS[a.color]
    lo = tuple(int(v) for v in a.hsv_lo.split(",")) if a.hsv_lo else lo
    hi = tuple(int(v) for v in a.hsv_hi.split(",")) if a.hsv_hi else hi
    detect = lambda f, accept=None: orange_target.detect(f, lo, hi, ap.MIN_AREA, accept=accept)  # noqa: E731
    if a.color == "orange" and not a.colour_only:   # the ball: the colour filter proposes, the learned detector decides
        import ball_detect
        import functools
        detect = ball_detect.BallDetect(lo, hi, ap.MIN_AREA,
                                        learned=functools.partial(ball_detect.learned_target.detect, threads=2))
        print(f"detector: colour filter confirmed by the learned detector ({ball_detect.learned_target.MODEL})")
    log_path = a.log or ap.find_log()
    if not log_path:
        sys.exit("no walk_wired log found: start the brain from the app first")
    kin = DuckKin()
    tail = SeeTail(log_path)
    cam = ap.ShmCamera()
    send = ap.send_file(a.cmd)
    ctl = SeePick(kin, cam.latest, tail.latest, tail.at, send, detect, tries=a.tries, dry=a.dry, t2=a.t2,
                  pick_here=tuple(a.pick_here) if a.pick_here else None, track_gate=a.track_gate,
                  make_approach=lambda: ap.Approach(kin, cam.latest, tail.latest, tail.at, send, detect,
                                                    log=lambda s: print("  approach " + s, flush=True)),
                  log=lambda s: print(s, flush=True))
    print(f"log {log_path}, colour {a.color} HSV {lo}..{hi}, tries {a.tries}{', dry run' if a.dry else ''}", flush=True)
    try:
        if a.observe is not None:
            end = time.time() + a.observe
            while time.time() < end:
                st = tail.latest()
                m = ctl.measure(st) if st else None
                print(f"{time.strftime('%H:%M:%S')} " + (f"x {m['x']:.3f} y {m['y']:.3f} width {m['size'] * 1000:.0f} mm"
                                                         if m else "not seen"), flush=True)
                time.sleep(0.3)
            return
        end = time.time() + a.max_s
        while ctl.phase not in ("done", "failed"):
            if time.time() > end and ctl.time_limit(f"time limit {a.max_s} s"):
                break
            ctl.step()
            time.sleep(0.05)
    except KeyboardInterrupt:
        send({"mode": "walk", "vx": 0.0, "vy": 0.0, "vtheta": 0.0})
        print("interrupted: walk at zero command written (stand61)")
    finally:
        cam.close()
    os.makedirs(os.path.dirname(a.trace), exist_ok=True)
    with open(a.trace, "w") as f:
        for row in ctl.trace:
            f.write(json.dumps(row) + "\n")
        f.write(json.dumps(dict(result=ctl.result)) + "\n")
    print(json.dumps(ctl.result))
    print(f"trace {a.trace}")


if __name__ == "__main__":
    main()
