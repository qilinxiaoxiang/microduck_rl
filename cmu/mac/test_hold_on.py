"""Hold-on (cmu/pick_rise/ROBOT-TEST.md): T2 keeps acting through the rise and holds at seepick.HOLD_PHASE, short of
walk_wired's hand-back to stand61, then seepick.py writes monitor. The brain half runs walk_brain.Brain with stand61
standing in for both pick policies (plumbing only, as test_teleop_modes); the coordinator half runs on
test_seepick's fake robot.

    python3 -m pytest -q test_hold_on.py
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seepick as sp  # noqa: E402
import walk_brain as wb  # noqa: E402
import walk_wired as ww  # noqa: E402
from test_seepick import PickDuck, t2_at  # noqa: E402
from test_teleop_modes import UPRIGHT_SENSOR_QUAT, WALK0  # noqa: E402

PIX_OBJ = {"u": 330.0, "v": 250.0, "w": 28.0, "t": 99.9, "seen": True}


def run_brain(pick, ticks):
    """Brain ticks: 10 standing on stand61, then the seeing pick with the jaw holding; (policy, sp_phase) per tick."""
    brain = wb.Brain(os.path.join(wb.WHOLE, "runtime-overlay", "models"))
    brain.sessions = dict(brain.sessions, seepick=brain.sessions["stand"], seepickpix=brain.sessions["stand"],
                          seepickpix_r1=brain.sessions["stand"])
    pos = [wb.to_tick(n, wb.STAND[n]) for n in wb.NAMES]
    out = []
    for k in range(10 + ticks):
        want = WALK0 if k < 10 else dict(pick, _holding=True)
        _, _, info = brain.tick(want, 100 + k * 0.02, pos, [0] * 14, UPRIGHT_SENSOR_QUAT, (0.0, 0.0, 0.0), True, 0.004, 0.006)
        if k >= 10:
            out.append((info.get("policy"), info.get("sp_phase")))
    return out


def test_brain_default_hands_a_holding_t2_to_seepick61_at_the_rise():
    out = run_brain({"mode": "seepick", "object": PIX_OBJ}, 120)   # phase 0.6
    swap = next(ph for pol, ph in out if pol == "seepick")
    assert out[0][0] == "seepickpix"
    assert wb.SP_RISE_OLD_PHASE <= swap < wb.SP_RISE_OLD_PHASE + 0.01


def test_brain_rise_old_false_keeps_t2_and_the_phase_cap_holds():
    ticks = int(sp.HOLD_PHASE * wb.SP_PERIOD / 0.02) + 50   # past the cap by 1 s
    out = run_brain({"mode": "seepick", "phase": sp.HOLD_PHASE, "rise_old": False, "object": PIX_OBJ}, ticks)
    assert {pol for pol, _ in out} == {"seepickpix"}
    assert max(ph for _, ph in out) == pytest.approx(sp.HOLD_PHASE, abs=1e-4)
    assert max(ph for _, ph in out) < 0.9   # walk_wired.SP_DONE_PHASE: the hand-back to stand61 never fires


class HoldDuck(PickDuck):
    """walk_wired with hold-on: the phase runs at 1/4 per s up to the command's cap and the mode stays seepick (no
    hand-back below 0.90); the jaw holds from phase 0.40; monitor holds the pose; trip_at (s) is a guard trip."""

    def __init__(self, obj, trip_at=None):
        super().__init__(obj, ["hold"])
        self.cap, self.trip_at, self.sent = 1.0, trip_at, []

    def send(self, cmd):
        self.sent.append((self.t, cmd))
        if cmd.get("mode") == "seepick":
            self.cap = cmd.get("phase", 1.0)
            if self.pick_t is None:
                self.pick_t, self.mode = self.t, "seepick"
                self.picks.append(dict(cmd["object"], sent_t=self.t))
        elif cmd.get("mode") == "monitor":
            self.mode = "monitor"

    def phase_now(self):
        return None if self.pick_t is None else min(self.cap, (self.t - self.pick_t) / 4.0)

    def advance(self, dt=0.05):
        self.t += dt
        if self.mode == "seepick" and self.phase_now() >= 0.40:
            self.holding, self.in_beak = True, True
        if self.trip_at is not None and self.pick_t is not None and self.t - self.pick_t >= self.trip_at:
            self.mode, self.trip_at = "monitor", None

    def state(self):
        s = super().state()
        s.update(mode=self.mode, sp_phase=self.phase_now())
        return s


def run_hold(duck, hold_s, max_s=20.0):
    ctl = t2_at(duck)
    ctl.hold_s = hold_s
    ctl.obj = dict(x=0.100, y=-0.021, size=0.013, t=0.0)
    ctl.enter("align")
    while ctl.phase not in ("done", "failed") and duck.t < max_s:
        ctl.step()
        duck.advance()
    return ctl


def test_hold_on_holds_for_hold_s_then_writes_monitor_never_stand61():
    duck = HoldDuck((0.100, -0.021))
    ctl = run_hold(duck, 5.0)
    picks = [c for _, c in duck.sent if c.get("mode") == "seepick"]
    assert picks and all(c["phase"] == sp.HOLD_PHASE and c["rise_old"] is False for c in picks)
    assert not any(c.get("mode") == "walk" for _, c in duck.sent)
    assert duck.sent[-1][1] == sp.MONITOR
    assert ctl.result["ok"], ctl.result
    att = ctl.result["attempts"][-1]
    assert att["holding"] and att["holding_at_hold"] and att["held_s"] == pytest.approx(5.0, abs=0.1)
    # the phase reaches the cap at HOLD_PHASE x 4 s; monitor follows hold_s later
    assert duck.sent[-1][0] - duck.pick_t == pytest.approx(sp.HOLD_PHASE * 4.0 + 5.0, abs=0.15)


def test_a_guard_trip_during_hold_on_ends_the_run_without_a_command():
    duck = HoldDuck((0.100, -0.021), trip_at=5.0)   # 1.4 s into the hold
    ctl = run_hold(duck, 5.0)
    n_sent = len(duck.sent)
    assert ctl.phase == "failed" and "monitor mode" in ctl.result["why"]   # SeePick.step's guard-trip stop
    assert not any(c.get("mode") in ("walk", "monitor") for _, c in duck.sent)
    assert len(duck.sent) == n_sent


def test_a_hand_back_to_walk_during_hold_on_ends_the_run_without_a_command():
    duck = HoldDuck((0.100, -0.021))
    ctl = t2_at(duck)
    ctl.hold_s = 5.0
    ctl.obj = dict(x=0.100, y=-0.021, size=0.013, t=0.0)
    ctl.enter("align")
    while duck.t < 5.0:
        ctl.step()
        duck.advance()
    duck.mode = "walk"   # someone pressed Y (stand61) on the remote
    n_sent = len(duck.sent)
    ctl.step()
    assert ctl.phase == "failed" and "ended by walk_wired (mode walk)" in ctl.result["why"]
    assert len(duck.sent) == n_sent and ctl.result["attempts"][-1]["holding"]


def test_without_hold_on_the_pick_command_is_unchanged():
    ctl = t2_at(PickDuck((0.100, -0.021), ["hold"]))
    ctl.obj = dict(x=0.100, y=-0.021, size=0.013, t=0.0)
    cmd = ctl.pick_cmd()
    assert cmd["phase"] == 1.0 and "rise_old" not in cmd


@pytest.mark.parametrize("t2,hold_s", [(False, 5.0), (True, 0.0), (True, sp.HOLD_MAX_S + 1)])
def test_hold_on_needs_t2_and_a_bounded_hold(t2, hold_s):
    duck = PickDuck((0.1, 0.0), [])
    with pytest.raises(ValueError):
        sp.SeePick(None, lambda: None, duck.state, lambda t: None, duck.send, duck.detect, make_approach=None,
                   t2=t2, hold_s=hold_s)


# --- the remote's two pick buttons: the baseline and revision r1 side by side (walk_brain.PICK_REVISIONS) ---

def test_brain_policy_r1_runs_the_revision_and_keeps_its_own_rise():
    out = run_brain({"mode": "seepick", "policy": "r1", "object": PIX_OBJ}, 120)   # no rise_old key, phase 0.6
    assert {pol for pol, _ in out} == {"seepickpix_r1"}


def test_brain_r1_hold_on_is_the_remote_button_command():
    ticks = int(sp.HOLD_PHASE * wb.SP_PERIOD / 0.02) + 50
    out = run_brain({"mode": "seepick", "phase": sp.HOLD_PHASE, "rise_old": False, "policy": "r1", "object": PIX_OBJ}, ticks)
    assert {pol for pol, _ in out} == {"seepickpix_r1"}
    assert max(ph for _, ph in out) < 0.9


def test_brain_loads_the_revision_file_beside_the_baseline(tmp_path):
    models = os.path.join(wb.WHOLE, "runtime-overlay", "models")
    for f in os.listdir(models):
        os.symlink(os.path.join(models, f), tmp_path / f)
    for name in ("seepickpix61.onnx", "seepickpix_r161.onnx"):
        os.symlink(os.path.join(models, "stand61.onnx"), tmp_path / name)
    assert {"seepickpix", "seepickpix_r1"} <= set(wb.Brain(str(tmp_path)).sessions)


@pytest.mark.parametrize("want,sessions,why", [
    ({"policy": "r1", "object": PIX_OBJ}, ("seepickpix", "seepickpix_r1"), None),
    ({"object": PIX_OBJ}, ("seepickpix",), None),
    ({"policy": "r1", "object": PIX_OBJ}, ("seepickpix",), "no seepickpix_r161.onnx"),
    ({"policy": "r9", "object": PIX_OBJ}, ("seepickpix", "seepickpix_r1"), "unknown pick policy"),
    ({"policy": "r1", "object": {"x": 0.1, "y": 0.0, "size": 0.013, "t": 1.0}}, ("seepick", "seepickpix_r1"),
     "needs a picture object"),
])
def test_walk_wired_refuses_a_revision_it_cannot_run(want, sessions, why):
    block = ww.seepick_block(dict(want, mode="seepick"), True, sessions)
    assert block is None if why is None else why in block


def test_seepick_policy_rides_on_the_pick_command():
    ctl = t2_at(PickDuck((0.100, -0.021), ["hold"]))
    ctl.policy, ctl.hold_s = "r1", 5.0
    ctl.obj = dict(x=0.100, y=-0.021, size=0.013, t=0.0)
    cmd = ctl.pick_cmd()
    assert (cmd["policy"], cmd["phase"], cmd["rise_old"]) == ("r1", sp.HOLD_PHASE, False)


def test_a_revision_needs_t2():
    duck = PickDuck((0.1, 0.0), [])
    with pytest.raises(ValueError):
        sp.SeePick(None, lambda: None, duck.state, lambda t: None, duck.send, duck.detect, make_approach=None, policy="r1")
