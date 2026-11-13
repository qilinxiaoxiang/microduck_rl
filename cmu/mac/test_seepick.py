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

    def send(self, cmd):
        if cmd.get("mode") == "seepick" and self.mode != "seepick" and self.pick_t is None:
            self.pick_t, self.mode = self.t, self.mode_during_pick
            self.picks.append(dict(cmd["object"], sent_t=self.t))
            self.outcome = self.outcomes.pop(0)
            return
        if self.mode in ("seepick", "monitor"):
            return   # walk_wired owns the mode until the hand-back (or the guard holds it)
        super().send(cmd)

    def advance(self, dt=0.05):
        if self.pick_t is not None and self.mode == "seepick" and self.t - self.pick_t >= PICK_S:
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
    assert ctl.phase == "approach" and ctl.try_n == 2 and not duck.picks


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
    # the last sighting before the ball left the picture: its true place is nearer, so the window carries the slack
    assert abs(o["x"] - sp.SPOT[0]) <= sp.PICK_WIN + sp.PICK_SLACK and abs(o["y"] - sp.SPOT[1]) <= sp.PICK_WIN, o
    assert all(c.get("vx", 0) <= apc.VX for _, c in duck.sent)


def test_a_creep_starting_inside_the_stop_mark_still_walks():
    """Object at 12.2 cm, dx +3.5, inside BURST_STOP_X: the creep must move, not stop on its first tick (robot 2026-11-01)."""
    duck = PickDuck((0.122, 0.0), ["hold"])
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=None, now=lambda: duck.t, log=lambda s: None)
    ctl.try_n = 1
    ctl.obj = dict(x=0.122, y=0.0, size=0.013, t=0.0)
    ctl.enter("align")
    assert ctl.phase == "align"
    ctl.step()
    duck.advance()
    assert ctl.phase == "burst"
    while ctl.phase == "burst":
        ctl.step()
        duck.advance()
    assert any(c.get("vx", 0) > 0 for _, c in duck.sent)
    assert duck.X > 0.01


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


def test_a_ball_lost_just_past_the_old_blind_edge_is_picked_not_given_up():
    """Robot 2026-11-06: after a 16 deg turn toward the ball the estimate read 0.121 m, just over the old 0.12 m blind
    edge; unseen, the run gave up without a bite. Unseen inside BLIND_R (0.15 m) picks at the last known position."""
    duck = PickDuck((0.6, 0.0), ["hold"])
    duck.detect = lambda frame, accept=None: None      # the ball is under the picture
    ctl = sp.SeePick(KIN, lambda: duck.frame, duck.state, lambda t: duck.state(), duck.send, duck.detect,
                     make_approach=lambda: None, tries=1, now=lambda: duck.t, log=lambda s: None)
    ctl.try_n, ctl.align_n, ctl.meas = 1, 0, []
    ctl.obj = dict(x=0.117, y=-0.031, size=0.012, t=0.0)   # |est| 0.121 m
    ctl.enter("aim")
    while ctl.phase == "aim" and duck.t < 20:
        ctl.step()
        duck.advance()
    assert ctl.phase in ("align", "pick", "turning"), (ctl.phase, ctl.result)
