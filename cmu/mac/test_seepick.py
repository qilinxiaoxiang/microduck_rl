"""seepick.SeePick on test_approach's fake robot: the approach, aim and align run as there; a seeing pick lasts
PICK_S in mode "seepick", then the fake hands back to walk with the jaw holding or not as each case scripts.

    python3 -m pytest -q test_seepick.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import approach as apc  # noqa: E402
import seepick as sp  # noqa: E402
from test_approach import KIN, FakeDuck  # noqa: E402

PICK_S = 3.6


class PickDuck(FakeDuck):
    """outcomes: one entry per pick, "hold", "miss" or ("slip", s); the object leaves the floor while held and lands
    0.09 m ahead of the robot when it slips."""

    def __init__(self, obj, outcomes, mode_during_pick="seepick"):
        super().__init__(obj)
        self.outcomes, self.mode_during_pick = list(outcomes), mode_during_pick
        self.pick_t, self.picks, self.holding, self.slip_at, self.in_beak = None, [], False, None, False
        self.pick_s = PICK_S   # a test makes the pick late
        self.over = []         # (t, cmd): mode changes written while walk_wired runs the pick or a guard holds it

    def send(self, cmd):
        if cmd.get("mode") == "seepick" and self.mode != "seepick" and self.pick_t is None:
            self.pick_t, self.mode = self.t, self.mode_during_pick
            self.picks.append(dict(cmd["object"], sent_t=self.t))
            self.outcome = self.outcomes.pop(0)
            return
        if self.mode in ("seepick", "monitor"):
            if cmd.get("mode") != "seepick":
                self.over.append((round(self.t, 2), dict(cmd)))
            return   # walk_wired owns the mode until the hand-back (or the guard holds it)
        super().send(cmd)

    def advance(self, dt=0.05):
        if self.pick_t is not None and self.mode == "seepick" and self.t - self.pick_t >= self.pick_s:
            self.mode, self.pick_t, self.cmd = "walk", None, {"mode": "walk", "vx": 0.0, "vtheta": 0.0}
            if self.outcome != "miss":
                self.holding, self.in_beak = True, True
                self.slip_at = self.t + self.outcome[1] if isinstance(self.outcome, tuple) else None
        if self.slip_at is not None and self.t >= self.slip_at:
            self.holding, self.in_beak, self.slip_at = False, False, None
            self.obj = (self.X + 0.09 * math.cos(self.th), self.Y + 0.09 * math.sin(self.th))
        super().advance(dt)

    def state(self):
        s = super().state()
        s.update(jaw_holding=self.holding, jaw_deg=8.0 if self.holding else 2.0,
                 sp_phase=None if self.pick_t is None else (self.t - self.pick_t) / 4.0)
        return s

    def detect(self, frame, accept=None):
        return None if self.in_beak else super().detect(frame, accept)


def run(duck, tries=3, dry=False, max_s=120.0):
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(),
                                                        duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None),
                     tries=tries, dry=dry, now=lambda: duck.t, log=lambda s: None)
    while ctl.phase not in ("done", "failed") and duck.t < max_s:
        ctl.step()
        duck.advance()
    return ctl


def test_walks_aligns_and_picks_once():
    duck = PickDuck((0.6, 0.05), ["hold"])
    ctl = run(duck)
    assert ctl.result["ok"], ctl.result
    assert ctl.try_n == 1 and len(duck.picks) == 1
    o = duck.picks[0]
    assert abs(o["x"] - sp.SPOT[0]) <= sp.PICK_WIN + 0.02 and abs(o["y"] - sp.SPOT[1]) <= sp.PICK_WIN + 0.02, o
    assert 0 <= o["sent_t"] - o["t"] < 1.0   # capture time handed to the brain, which turns it into the age


def test_a_miss_is_retried():
    duck = PickDuck((0.5, -0.04), ["miss", "hold"])
    ctl = run(duck)
    assert ctl.result["ok"], ctl.result
    assert ctl.try_n == 2 and [a.get("holding") for a in ctl.result["attempts"]] == [False, True]


def test_a_slip_searches_and_picks_again():
    duck = PickDuck((0.5, 0.0), [("slip", 2.0), "hold"])
    ctl = run(duck)
    assert ctl.result["ok"], ctl.result
    assert ctl.result["attempts"][0]["slipped_after_s"] >= sp.SLIP_S
    assert len(duck.picks) == 2


def test_gives_up_after_the_tries():
    duck = PickDuck((0.5, 0.0), ["miss", "miss"])
    ctl = run(duck, tries=2)
    assert not ctl.result["ok"] and "2 tries used" in ctl.result["why"]


def test_dry_run_never_picks():
    duck = PickDuck((0.5, 0.03), [])
    ctl = run(duck, dry=True)
    assert ctl.result["ok"] and "dry run" in ctl.result["why"]
    assert not duck.picks


def test_guard_trip_during_the_pick_ends_the_run():
    duck = PickDuck((0.5, 0.0), ["hold"], mode_during_pick="monitor")
    ctl = run(duck)
    assert not ctl.result["ok"] and "monitor" in ctl.result["why"]
    t_pick = duck.picks[0]["sent_t"]
    assert not any(t > t_pick for t, c in duck.sent)   # the guard's hold is left alone, nothing is walked


def test_an_object_to_the_side_turns_before_the_pick():
    duck = PickDuck((0.5, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=None, now=lambda: duck.t, log=lambda s: None)
    ctl.obj = dict(x=0.09, y=0.07, size=0.013, t=0.0)
    ctl.enter("align")
    ctl.step()
    assert ctl.phase == "turning"
    for _ in range(80):
        ctl.step()
        duck.advance()
        if ctl.phase != "turning":
            break
    assert any(c.get("vtheta", 0) > 0 for _, c in duck.sent)   # left, toward +y


def test_an_object_out_of_view_is_found_by_turning_the_body():
    duck = PickDuck((-0.1, -0.5), ["hold"])   # behind the right shoulder: outside every head pose of the scan
    ctl = run(duck, max_s=240.0)
    assert ctl.look_n >= 1 and duck.picks and ctl.try_n == 1, ctl.result
    assert ctl.result["ok"], ctl.result


def test_nothing_anywhere_gives_up_after_the_body_turns():
    duck = PickDuck((0.6, 0.0), [])
    duck.detect = lambda frame, accept=None: None
    ctl = run(duck, max_s=400.0)
    assert ctl.phase == "failed" and ctl.look_n == len(sp.LOOK_TURNS) and not duck.picks


def test_too_close_is_not_picked():
    duck = PickDuck((0.5, 0.0), [])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(),
                                                        duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None),
                     now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    ctl.obj = dict(x=0.02, y=0.0, size=0.013, t=0.0)
    ctl.enter("align")
    ctl.step()
    assert ctl.phase == "back" and ctl.try_n == 1 and not duck.picks   # too close: step back, never bite (see below)


def test_a_too_far_object_is_crept_up_to_with_the_camera_then_picked():
    """The object 7 cm beyond the trained spot: no blind burst; the creep walks while the camera sees it, stops at
    BURST_STOP_X and the pick happens inside PICK_WIN (robot 2026-10-30: picks at +4.6 cm came up empty)."""
    duck = PickDuck((0.16, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=None, now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    ctl.obj = dict(x=0.16, y=0.0, size=0.013, t=0.0)
    ctl.enter("align")
    phases = []
    for _ in range(400):
        ctl.step()
        duck.advance()
        if ctl.phase not in phases:
            phases.append(ctl.phase)
        if duck.picks or ctl.phase in ("done", "failed"):
            break
    assert phases[:2] == ["burst", "aim"], phases
    assert duck.picks, (phases, ctl.result)
    o = duck.picks[0]
    assert sp.SPOT[0] - sp.PICK_WIN <= o["x"] <= sp.PICK_FAR_X and abs(o["y"] - sp.SPOT[1]) <= sp.PICK_WIN, o
    assert all(c.get("vx", 0) <= apc.VX for _, c in duck.sent)


def test_an_object_beyond_the_hold_range_is_not_bitten_from_there():
    """Robot 2026-11-12: the bites at x 0.110, 0.110, 0.112 and 0.125 all closed on nothing; every robot hold sat at
    x 0.098-0.100. Measured at 11.9 cm the robot moves closer first; it does not bite from there."""
    duck = PickDuck((0.119, 0.012), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=None, now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    ctl.obj = dict(x=0.119, y=0.012, size=0.013, t=0.0)
    ctl.enter("align")
    ctl.step()
    assert ctl.phase == "burst" and not duck.picks, ctl.phase
    assert not ctl.in_window(0.119, 0.012) and ctl.in_window(0.100, -0.015)


def t2_at(duck):
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=None, now=lambda: duck.t, log=lambda s: None, t2=True)
    ctl.try_n = 1
    ctl.pix_box = lambda o: {}
    return ctl


def test_t2_picks_inside_the_held_placements_without_a_move():
    duck = PickDuck((0.100, -0.021), ["hold"])
    ctl = t2_at(duck)
    ctl.obj = dict(x=0.100, y=-0.021, size=0.013, t=0.0)
    ctl.enter("align")
    ctl.step()
    assert duck.picks and ctl.phase == "pick"


def test_t2_does_not_pick_at_the_miss_of_run_2():
    """x 0.122 y -0.005 bit air on the robot and sim holds 0 of 6 at dx +3: it moves on, and the timed creep really walks."""
    duck = PickDuck((0.122, -0.005), ["hold"])
    ctl = t2_at(duck)
    ctl.obj = dict(x=0.122, y=-0.005, size=0.013, t=0.0)
    ctl.enter("align")
    ctl.step()
    assert not duck.picks and ctl.phase == "burst"
    while ctl.phase == "burst":
        ctl.step()
        duck.advance()
    assert duck.t >= 0.4 and any(c.get("vx", 0) > 0 for _, c in duck.sent)


def test_t2_false_detection_after_the_creep_does_not_end_the_try():
    """Run 8: ball last seen at x 0.124 y -0.021, creep, then x 0.037 y 0.050 width 8 mm from 2 frames: pick at the last sighting."""
    duck = PickDuck((0.09, -0.02), ["hold"])
    ctl = t2_at(duck)
    ctl.obj = dict(x=0.124, y=-0.021, size=0.014, t=duck.t, u=1.0, v=1.0, w=1.0)
    ctl.blind_ok, ctl.aim_gate = True, (0.124, -0.021)
    fake = iter([dict(x=0.037, y=0.050, size=0.008, t=0.0, cut=False)] * 2)
    real = sp.SeePick.measure
    ctl.measure = lambda st, gate=None, gate_m=None: (lambda m: None if m is None or (gate and math.hypot(m["x"] - gate[0], m["y"] - gate[1]) > gate_m) else m)(next(fake, None))
    ctl.enter("aim")
    for _ in range(400):
        ctl.step()
        duck.advance()
        if duck.picks:
            break
    assert duck.picks and duck.picks[0]["x"] == 0.124 and "u" not in ctl.obj, ctl.phase


def test_t2_never_walks_at_a_ball_inside_or_nearer_than_the_window():
    """Run 12: x 0.086 y -0.019 started a creep and the duck walked over the ball."""
    for x in (0.086, 0.081, 0.100):
        duck = PickDuck((x, -0.019), ["hold"])
        ctl = t2_at(duck)
        ctl.obj = dict(x=x, y=-0.019, size=0.013, t=0.0)
        ctl.enter("align")
        ctl.step()
        assert duck.picks and not any(c.get("vx", 0) > 0 for _, c in duck.sent), x


def test_t2_a_ball_last_seen_past_the_beak_is_looked_for_not_bitten():
    duck = PickDuck((0.06, -0.004), ["hold"])
    ctl = t2_at(duck)
    ctl.make_approach = lambda: type("A", (), {"phase": "search", "step": lambda s: None})()
    ctl.obj = dict(x=0.063, y=-0.004, size=0.013, t=duck.t)
    ctl.blind_ok, ctl.aim_gate = True, (0.063, -0.004)
    ctl.measure = lambda st, gate=None, gate_m=None: None
    ctl.enter("aim")
    for _ in range(200):
        ctl.step()
        duck.advance()
        if ctl.phase == "approach":
            break
    assert ctl.phase == "approach" and not duck.picks


def test_t2_creep_steers_toward_the_held_side():
    duck = PickDuck((0.158, -0.05), ["hold"])
    ctl = t2_at(duck)
    ctl.obj = dict(x=0.158, y=-0.05, size=0.013, t=0.0)
    ctl.head = {}
    ctl.burst_m = dict(ctl.obj)
    ctl.t_phase, ctl.phase = duck.t, "burst"
    ctl.measure = lambda st, gate=None, gate_m=None: None
    ctl.step()
    assert duck.sent[-1][1]["vtheta"] < 0


def test_t2_pick_command_carries_the_box_and_the_blind_zone_clamps_to_the_edge():
    import walk_brain

    class Kin:
        def ground_to_pixel(self, x, y, q, gb, w, h, z=0.0):
            return (330.0, 400.0)

    c = sp.SeePick.__new__(sp.SeePick)
    c.t2, c.kin, c.now, c.state_fn = True, Kin(), lambda: 10.0, lambda: {"q": {}, "gb": [0, 0, -1]}
    c.obj = dict(x=0.09, y=0.0, size=0.012, t=9.95, u=330.0, v=200.0, w=20.0)
    o = c.pick_cmd()["object"]
    assert walk_brain.is_pix_object(o) and o["u"] == 330.0 and o["seen"] is True
    c.obj = dict(x=0.09, y=0.0, size=0.012, t=9.0)
    o = c.pick_cmd()["object"]
    assert o["v"] == 359 and o["seen"] is False and walk_brain.is_pix_object(o)
    c.t2 = False
    assert not walk_brain.is_pix_object(c.pick_cmd()["object"])


def test_a_guard_stand_during_the_aim_ends_the_run_without_walk():
    """Robot 2026-11-06: the aim kept writing walk to a duck the tilt guard had stood down (lying at 50 deg)."""
    duck = PickDuck((0.6, 0.05), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(),
                                                        duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None),
                     now=lambda: duck.t, log=lambda s: None)
    while ctl.phase != "aim" and duck.t < 60:
        ctl.step()
        duck.advance()
    assert ctl.phase == "aim"
    duck.mode = "stand"
    n = len(duck.sent)
    for _ in range(20):
        ctl.step()
        duck.advance()
    assert ctl.phase == "failed" and "stand" in ctl.result["why"]
    assert not [c for _, c in duck.sent[n:] if c.get("mode") == "walk"], duck.sent[n:]


def test_no_look_down_or_pick_while_the_body_rocks():
    """Robot 2026-11-06: the aim lowered the head to 1.4 just after a turn in place (gyro 1-1.5 rad/s); the duck fell."""
    duck = PickDuck((0.6, 0.05), ["hold"])
    rock = {"on": False}
    base = duck.state
    duck.state = lambda: dict(base(), gyro=(0.0, 1.2 if rock["on"] else 0.02, 0.0))
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(),
                                                        duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None),
                     now=lambda: duck.t, log=lambda s: None)
    while ctl.phase != "aim" and duck.t < 60:
        ctl.step()
        duck.advance()
    rock["on"] = True
    n = len(duck.sent)
    for _ in range(60):
        ctl.step()
        duck.advance()
    bad = [c for _, c in duck.sent[n:] if c.get("mode") == "seepick" or
           (c.get("head") or {}).get("head_pitch", 0) > apc.DEFAULT_HEAD["head_pitch"] + 1e-6]
    assert not bad, bad[:3]
    rock["on"] = False
    while ctl.phase not in ("done", "failed") and duck.t < 120:
        ctl.step()
        duck.advance()
    assert ctl.result["ok"], ctl.result


def test_a_ball_not_seen_at_the_pick_spot_is_never_bitten():
    """Robot 2026-11-12: the lowest look shows the floor from 0.07 m, so the pick window is in the picture; a ball not
    seen there is not there. The old rule bit the last known position (seepick_1112a try 2, 1112c both tries).
    Last seen at the feet (0.121 m) it may be under the picture at the toes: two step-backs with a new aim each, no
    body turn, and when it is still unseen the try ends without a bite."""
    duck = PickDuck((0.6, 0.0), ["hold"])
    duck.detect = lambda frame, accept=None: None      # the ball is under the picture
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: None, tries=1, now=lambda: duck.t, log=lambda s: None)
    ctl.try_n, ctl.align_n, ctl.meas = 1, 0, []
    ctl.obj = dict(x=0.117, y=-0.031, size=0.012, t=0.0)   # |est| 0.121 m
    ctl.enter("aim")
    while ctl.phase not in ("done", "failed") and duck.t < 40:
        ctl.step()
        duck.advance()
    assert ctl.phase == "failed" and ctl.result["attempts"] == [] and not duck.picks, (ctl.phase, ctl.result)
    assert [r["phase"] for r in ctl.trace].count("back") == sp.LOST_BACKS
    assert not any(c.get("vtheta") for _, c in duck.sent)


class KnockDuck(PickDuck):
    """The bite touches the ball and knocks it away: 2 s into the pick it lies 0.20 m ahead and 0.08 m to the left."""

    def advance(self, dt=0.05):
        if self.pick_t is not None and self.outcome == "miss" and not getattr(self, "knocked", False) \
                and self.t - self.pick_t >= 2.0:
            self.knocked = True
            c, s = math.cos(self.th), math.sin(self.th)
            self.obj = (self.X + 0.20 * c - 0.08 * s, self.Y + 0.20 * s + 0.08 * c)
        super().advance(dt)


def test_a_ball_knocked_away_by_the_bite_is_looked_for_not_bitten_blind():
    """Robot 2026-11-12 (seepick_1112a): try 1 had the ball in the beak and knocked it 5 cm aside; the camera saw it
    there during the pick, and try 2 still bit the old spot blind (Shawn: "第二次完全是瞎试 因为球完全不在范围内")."""
    duck = KnockDuck((0.5, 0.0), ["miss", "hold"])
    ctl = run(duck, tries=5)
    whys = [r.get("why", "") for r in ctl.trace]
    assert not any("blind zone" in w for w in whys)
    # the look at the old spot finds nothing and bites nothing; the search finds the ball and the second bite holds
    assert ctl.result["ok"] and len(duck.picks) == 2, ctl.result
    assert any("not seen at the pick spot" in w for w in whys), whys


def test_an_object_too_close_is_stepped_back_from():
    """Shawn 2026-11-12: "同时现在没有后退 为什么 完全应该有后退". An object nearer than the pick window: the robot
    walks backward at the speed that steps on the robot (BACK_VX), then aims again; the try is not given up."""
    duck = PickDuck((0.04, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(),
                                                        duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None),
                     now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    ctl.obj = dict(x=0.04, y=0.0, size=0.013, t=0.0)
    ctl.enter("align")
    x0 = duck.obj[0]
    while ctl.phase in ("align", "back") and duck.t < 10.0:
        ctl.step()
        duck.advance()
    assert ctl.phase == "aim" and ctl.try_n == 1, (ctl.phase, ctl.try_n)
    backs = [c for _, c in duck.sent if c.get("mode") == "walk" and c.get("vx", 0) < 0]
    assert backs and all(c["vx"] == sp.BACK_VX for c in backs)
    assert not duck.picks


def test_a_creep_stops_when_the_object_leaves_the_picture():
    """Robot 2026-11-12 (seepick_1112f): the creep held the lowest look, the neck dropped once walk61 stepped, the ball
    left the top of the picture and the creep walked on unseen for 1 s; the left foot came down on the ball and the
    duck fell. A creep with no sighting for SIGHT_LOST_S stops."""
    duck = PickDuck((0.16, 0.0), [])
    seen = {"on": True}
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send,
                     lambda f, accept=None: duck.detect(f, accept) if seen["on"] else None,
                     make_approach=lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(),
                                                        duck.send, duck.detect, now=lambda: duck.t, log=lambda s: None),
                     now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    ctl.obj = dict(x=0.16, y=0.0, size=0.013, t=0.0)
    ctl.enter("align")
    ctl.step()
    assert ctl.phase == "burst"
    seen["on"] = False
    t0 = duck.t
    while ctl.phase == "burst" and duck.t < t0 + 3.0:
        ctl.step()
        duck.advance()
    assert ctl.phase == "aim" and duck.t - t0 <= sp.SIGHT_LOST_S + 0.15, (ctl.phase, duck.t - t0)
    assert any("not seen" in r.get("why", "") for r in ctl.trace)


def real_approach(duck):
    return lambda: apc.Approach(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                                now=lambda: duck.t, log=lambda s: None)


def back_cmds(duck):
    return [c for _, c in duck.sent if c.get("mode") == "walk" and c.get("vx", 0) < 0]


def test_a_ball_lost_at_the_feet_after_the_walk_is_stepped_back_from():
    """The walk runs on and the ball ends under the picture at the toes (approach.LOST_AT_FEET). No try is used and
    the body does not turn: the robot steps back with the lowest look, sees the ball again and picks it."""
    duck = PickDuck((0.6, 0.0), ["hold"])
    duck.coast_m = 0.09
    ctl = run(duck)
    assert any("lost at the feet" in r.get("why", "") for r in ctl.trace), [r.get("why") for r in ctl.trace]
    assert ctl.result["ok"] and ctl.try_n == 1 and ctl.look_n == 0 and len(duck.picks) == 1, ctl.result
    backs = back_cmds(duck)
    assert backs and all(c["vx"] == sp.BACK_VX and c["head"]["head_pitch"] == apc.NEAR_HEAD["head_pitch"] for c in backs)
    assert not any(c.get("vtheta") for _, c in duck.sent if not c.get("vx"))   # no turn in place


def test_not_seen_in_the_aim_after_a_sighting_at_the_toes_steps_back():
    """Last seen at 0.10 m and the aim shows nothing: the ball is under the picture at the toes. The robot steps back
    and aims again in the same try; it does not start a new search."""
    duck = PickDuck((0.055, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=real_approach(duck), now=lambda: duck.t, log=lambda s: None)
    ctl.try_n, ctl.align_n, ctl.meas = 1, 0, []
    ctl.obj = dict(x=0.10, y=0.0, size=0.013, t=0.0)
    ctl.enter("aim")
    while ctl.phase not in ("done", "failed") and duck.t < 60:
        ctl.step()
        duck.advance()
    assert back_cmds(duck) and ctl.try_n == 1 and ctl.look_n == 0, (ctl.try_n, ctl.look_n, ctl.result)
    assert ctl.result["ok"] and len(duck.picks) == 1, ctl.result


def test_the_pick_window_does_not_widen_after_the_align_moves():
    """After MAX_ALIGN moves the old window grew by PICK_SLACK on every edge and bit from x 0.110, where every robot
    bite closed on nothing. Outside the window after the moves: no bite, the try ends."""
    stub = lambda: type("A", (), {"phase": "search", "step": lambda s: None})()   # noqa: E731
    for t2, x, y in ((False, sp.PICK_FAR_X + 0.005, 0.0), (False, sp.SPOT[0] - sp.PICK_WIN - 0.005, 0.0),
                     (False, 0.09, sp.PICK_WIN + 0.005), (True, sp.T2_WIN_X[1] + 0.005, sp.T2_Y),
                     (True, 0.100, sp.T2_WIN_Y[1] + 0.005)):
        duck = PickDuck((x, y), ["hold"])
        ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                         make_approach=stub, now=lambda: duck.t, log=lambda s: None, t2=t2)
        ctl.pix_box = lambda o: {}
        ctl.try_n, ctl.align_n = 1, sp.MAX_ALIGN
        ctl.obj = dict(x=x, y=y, size=0.013, t=0.0)
        ctl.enter("align")
        ctl.step()
        assert not duck.picks and ctl.phase == "approach" and ctl.try_n == 2, (t2, x, y, ctl.phase)


def test_t1_aim_needs_three_frames():
    """Two frames are no sighting for T1 either (run 8: x 0.037 width 8 mm from 2 frames): no pick position from them."""
    duck = PickDuck((0.09, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: type("A", (), {"phase": "search", "step": lambda s: None})(),
                     now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    two = iter([dict(x=0.09, y=0.0, size=0.013, t=0.0, cut=False)] * (sp.AIM_MIN_FRAMES - 1))
    ctl.measure = lambda st, gate=None, gate_m=None: next(two, None)
    ctl.enter("aim")
    phases = []
    while ctl.phase == "aim" and duck.t < 20:
        ctl.step()
        duck.advance()
        phases.append(ctl.phase)
    assert "align" not in phases and ctl.phase == "approach" and not duck.picks, ctl.phase


def test_a_late_pick_is_not_written_over():
    """A pick still running at PICK_MAX_S: no walk command is written into the bend or the rise. The run ends once
    walk_wired has handed back."""
    duck = PickDuck((0.5, 0.0), ["hold"])
    duck.pick_s = sp.PICK_MAX_S + 2.0
    ctl = run(duck)
    assert not duck.over, duck.over[:2]
    assert not ctl.result["ok"] and "did not hand back" in ctl.result["why"], ctl.result
    assert duck.mode == "walk" and duck.t >= duck.picks[0]["sent_t"] + duck.pick_s


def test_the_time_limit_waits_for_a_pick_in_progress():
    """--max-s running out during a pick: the pick hands back first, then the run finishes."""
    duck = PickDuck((0.5, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=real_approach(duck), now=lambda: duck.t, log=lambda s: None)
    while not (ctl.phase == "pick" and duck.mode == "seepick") and duck.t < 60:
        ctl.step()
        duck.advance()
    assert ctl.phase == "pick"
    assert ctl.time_limit("time limit") is False and ctl.phase == "pick" and ctl.result is None
    while ctl.phase == "pick" and duck.t < 80:
        assert ctl.time_limit("time limit") is False
        ctl.step()
        duck.advance()
    assert not duck.over, duck.over[:2]
    assert ctl.time_limit("time limit") is True and ctl.phase == "failed" and ctl.result["why"] == "time limit"
