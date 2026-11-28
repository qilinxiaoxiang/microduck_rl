"""steplook.py on a fake robot: the real camera projection, bursts that move it a given distance per for_s.

    uv run --no-sync --with pytest --with pyserial pytest cmu/step/test_steplook.py      (repo root)
"""
import hashlib
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "mac")):
    if os.path.exists(os.path.join(d, "test_approach.py")):
        sys.path.insert(0, d)
        break
sys.path.insert(0, HERE)
import steplook as sl  # noqa: E402
from test_approach import KIN, STAND, H, W  # noqa: E402

ARM = dict(name="test", bursts=[[0.2, 0.03], [0.4, 0.05], [0.7, 0.08], [1.0, 0.12]])


class Frame:
    shape = (H, W, 3)

    def __init__(self, box):
        self.box = box


class Duck:
    """walk_wired as the coordinator sees it: a burst moves it `gain` x the table's distance once, then it stands."""

    def __init__(self, ball, gain=0.8, mode="walk", turn_gain=0.45):
        self.t, self.ball, self.gain, self.mode, self.turn_gain = 100.0, list(ball), gain, mode, turn_gain
        self.q, self.sent, self.bursts, self.last, self.i = dict(STAND), [], [], None, 0
        self.gyro = [0.0, 0.0, 0.0]

    def send(self, c):
        self.sent.append((self.t, dict(c)))
        if c.get("vx") and c != self.last:
            far = min(ARM["bursts"], key=lambda b: abs(b[0] - c["for_s"]))[1] * self.gain
            x, y = self.ball[0] - far, self.ball[1]
            a = -c["vtheta"] * c["for_s"]
            self.ball = [x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)]
            self.bursts.append((self.t, dict(c)))
        elif c.get("vx") == 0.0 and c.get("vtheta") and c.get("for_s") and c != self.last:
            a = -c["vtheta"] * c["for_s"] * self.turn_gain     # a turn in place: the ball swings round the duck
            x, y = self.ball
            self.ball = [x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)]
            self.bursts.append((self.t, dict(c)))
        self.last = dict(c)
        head = c.get("head") or {}
        for n in ("head_pitch", "head_yaw"):
            self.q[n] = head.get(n, STAND[n])
        self.q["neck_pitch"] = 0.22   # what the robot reads standing

    def state(self, t=None):
        return dict(t=self.t, mode=self.mode, gb=[0.0, 0.0, -1.0], q=dict(self.q), gyro=list(self.gyro), yaw=0.0, ok=True)

    def frame(self):
        self.i += 1
        return Frame(None), self.t, self.i


def make(duck, **kw):
    ctl = sl.StepLook(KIN, duck.frame, duck.state, duck.state, duck.send, lambda f, accept=None: None, ARM,
                      now=lambda: duck.t, log=lambda s: None, **kw)

    def measure(st, gate=None, gate_m=None):
        uv = KIN.ground_to_pixel(duck.ball[0], duck.ball[1], duck.q, (0, 0, -1), W, H, z=0.0065)
        if uv is None or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            return None
        return dict(x=duck.ball[0], y=duck.ball[1], t=duck.t)

    ctl.measure = measure
    return ctl


def drive(duck, ctl, limit=240.0):
    end = duck.t + limit
    while ctl.phase not in ("done", "failed") and duck.t < end:
        ctl.step()
        duck.t += 0.05
    assert ctl.result is not None
    return ctl


@pytest.mark.parametrize("ball", [(0.30, 0.0), (0.40, 0.08), (0.45, -0.10), (0.26, 0.02)])
def test_steps_to_the_window_without_overshooting(ball):
    d = Duck(ball)
    c = drive(d, make(d))
    assert c.result["ok"], c.result
    assert sl.STOP_R - sl.STOP_TOL - 0.005 <= math.hypot(*d.ball) <= sl.STOP_R + sl.STOP_TOL + 0.01


def test_the_table_distance_never_overshoots():
    for cm in range(18, 70):
        p = sl.plan(cm / 100, 0.0, ARM["bursts"])
        if "T" in p:
            far = dict(ARM["bursts"])[p["T"]]
            assert cm / 100 - far >= sl.STOP_R - sl.STOP_TOL - 1e-9


def test_each_burst_is_written_once_with_the_head_up_and_stops_itself():
    d = Duck((0.40, 0.0))
    drive(d, make(d))
    walks = [c for _, c in d.sent if c.get("vx")]
    assert walks and len(walks) == len(d.bursts)
    for t, c in d.bursts:
        assert c["for_s"] > 0 and "head" not in c
        before = [x for u, x in d.sent if t - sl.HEAD_UP_S + 0.1 < u < t]
        assert before and all("head" not in x for x in before)


def test_straight_bursts_do_not_teach_the_yaw_trim():
    d = Duck((0.40, 0.0))
    drive(d, make(d))
    assert all(abs(c["vtheta"]) > 1e-3 for _, c in d.bursts)   # walk_brain: |vtheta| > 1e-3 is a manual turn


def test_the_head_never_goes_past_what_held_on_the_robot():
    d = Duck((0.24, 0.03))
    drive(d, make(d))
    heads = [c["head"] for _, c in d.sent if c.get("head")]
    assert heads and all(h["head_pitch"] <= 1.1 and abs(h["head_yaw"]) <= sl.HEAD_YAW_MAX for h in heads)
    assert not any("neck_pitch" in h for h in heads)


def test_a_robot_that_does_not_move_is_stopped():
    d = Duck((0.40, 0.0), gain=0.0)
    c = drive(d, make(d))
    assert not c.result["ok"] and len(d.bursts) == sl.MAX_NO_PROGRESS


def test_a_ball_to_the_side_gets_turns_in_place_then_is_refused_without_a_step():
    # 2026-11-26: it used to be refused at once (a person turned the duck); now it turns in place first. A duck whose
    # turns do nothing ends refused after MAX_AIMS turns, and never takes a forward step toward a ball that far off.
    d = Duck((0.25, 0.22), turn_gain=0.0)
    c = drive(d, make(d))
    assert not c.result["ok"]
    assert len(d.bursts) == sl.MAX_AIMS and all(b["vx"] == 0.0 for _, b in d.bursts)


def test_a_ball_to_the_side_is_turned_to_and_reached():
    d = Duck((0.25, 0.22))
    c = drive(d, make(d))
    assert c.result["ok"], c.result


def test_a_guard_trip_ends_the_run_without_writing_walk():
    d = Duck((0.40, 0.0))
    c = make(d)
    while not d.bursts:
        c.step()
        d.t += 0.05
    n, d.mode = len(d.sent), "stand"
    c.step()
    assert c.phase == "failed" and len(d.sent) == n


def test_no_look_down_and_no_step_on_a_rocking_body():
    d = Duck((0.40, 0.0))
    d.gyro = [1.0, 0.0, 0.0]
    c = drive(d, make(d))
    assert not c.result["ok"] and not d.bursts and not any(x.get("head") for _, x in d.sent)


def test_dry_and_single():
    d = Duck((0.40, 0.0))
    assert drive(d, make(d, dry=True)).result["ok"] and not d.bursts
    d = Duck((0.40, 0.0))
    assert drive(d, make(d, single=0.4)).result["ok"] and [c["for_s"] for _, c in d.bursts] == [0.4]


def test_the_arm_comes_from_what_the_brain_loaded(tmp_path):
    log, arms = tmp_path / "b.jsonl", tmp_path / "arms.json"
    sha = hashlib.sha256(b"policy").hexdigest()[:16]
    (tmp_path / "b.meta.jsonl").write_text(json.dumps(dict(model_sha={"walk61.onnx": sha})) + "\n")
    arms.write_text(json.dumps(dict(arms=[dict(ARM, sha=sha)])))
    assert sl.load_arm(str(log), str(arms))[0]["name"] == "test"
    arms.write_text(json.dumps(dict(arms=[dict(ARM, sha="other")])))
    arm, why = sl.load_arm(str(log), str(arms))
    assert arm is None and "neither arm" in why


def test_a_run_that_ends_outside_the_window_is_not_a_success():
    d = Duck((0.23, 0.0))
    c = make(d)
    c.arm = dict(name="coarse", bursts=[[0.15, 0.084]])
    drive(d, c)
    assert not c.result["ok"] and not d.bursts


def test_smoothed_straight_bursts_leave_the_robots_yaw_trim_alone(tmp_path):
    """The staged brain, fed a straight burst through walk_wired's low-pass, must not rewrite the trim file."""
    import walk_brain as wb
    from test_teleop_modes import UPRIGHT_SENSOR_QUAT

    f = tmp_path / "trim.json"
    f.write_text('{"yaw_trim": -0.196}')
    models = os.path.join(HERE, "..", "whole", "runtime-overlay", "models")
    for vth, learns in ((sl.VTH_STRAIGHT, False), (0.0, True)):
        f.write_text('{"yaw_trim": -0.196}')
        brain = wb.Brain(models, yaw_trim_file=str(f))
        pos, cmd = [wb.to_tick(n, wb.STAND[n]) for n in wb.NAMES], [0.0, 0.0, 0.0]
        for k in range(220):
            on = 20 <= k < 85   # 1.3 s
            for i, v in enumerate((0.25 if on else 0.0, 0.0, vth if on else 0.0)):
                cmd[i] += 0.2 * (v - cmd[i])
                cmd[i] = 0.0 if abs(cmd[i]) < 1e-4 else cmd[i]
            brain.tick(dict(mode="walk", vx=cmd[0], vy=cmd[1], vtheta=cmd[2]), 100 + k * 0.02, pos, [0] * 14,
                       UPRIGHT_SENSOR_QUAT, (0.0, 0.0, 0.0), True, 0.004, 0.006)
        assert ("updated" in json.loads(f.read_text())) == learns


def test_a_two_slot_arm_needs_both_slots(tmp_path):
    log, arms = tmp_path / "b.jsonl", tmp_path / "arms.json"
    arms.write_text(json.dumps(dict(arms=[dict(ARM, sha="s", slots=["walk61", "stand61"])])))
    (tmp_path / "b.meta.jsonl").write_text(json.dumps(dict(model_sha={"walk61.onnx": "s", "stand61.onnx": "x"})) + "\n")
    arm, why = sl.load_arm(str(log), str(arms))
    assert arm is None and "stand61" in why
    (tmp_path / "b.meta.jsonl").write_text(json.dumps(dict(model_sha={"walk61.onnx": "s", "stand61.onnx": "s"})) + "\n")
    assert sl.load_arm(str(log), str(arms))[0]["name"] == "test"


def test_plan_aims_in_the_window_before_it_finishes():
    assert "done" in sl.plan(0.19, 0.0, ARM["bursts"])
    p = sl.plan(0.19, 0.19 * math.tan(math.radians(15)), ARM["bursts"])
    assert p.get("aim") and p["vx"] == 0.0 and p["vtheta"] > 0 and sl.AIM_T[0] <= p["T"] <= sl.AIM_T[1]
    assert sl.plan(0.19, -0.04, ARM["bursts"])["vtheta"] < 0
    assert "done" in sl.plan(0.19, 0.04, ARM["bursts"], aims=sl.MAX_AIMS)


@pytest.mark.parametrize("y", [0.04, -0.05])
def test_an_off_centre_ball_in_the_window_is_faced_before_the_run_ends(y):
    d = Duck((0.19, y))      # already in range: only a turn in place can centre it
    c = drive(d, make(d))
    assert c.result["ok"], c.result
    assert abs(math.atan2(d.ball[1], d.ball[0])) <= sl.AIM_TOL + 1e-6
    assert d.bursts and all(b["vx"] == 0.0 for _, b in d.bursts)


def test_turns_that_do_not_turn_end_the_run_after_max_aims_not_as_no_progress():
    d = Duck((0.19, 0.05), turn_gain=0.0)
    c = drive(d, make(d))
    assert c.result["ok"], c.result          # still in the range window: ends as before, just not centred
    assert sum(b["vx"] == 0.0 for _, b in d.bursts) == sl.MAX_AIMS


def test_a_ball_drifting_to_the_side_on_the_way_gets_a_turn_not_a_refusal():
    p = sl.plan(0.40, -0.40 * math.tan(math.radians(30)), ARM["bursts"])
    assert p.get("aim") and p["vtheta"] < 0
    far_side = sl.plan(0.40, -0.40 * math.tan(math.radians(50)), ARM["bursts"], aims=sl.MAX_AIMS)
    assert "refuse" in far_side          # turns used up: refused as before
