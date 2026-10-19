"""pickup.Pickup on a fake robot: the phase machine from look to a standing grip, and its failure exits.

The fake keeps the object's position in the foot frame and the robot's head pose, moves the object by the commanded
walk bursts (vx * SPEED_RATIO, turn at TURN_RATE) and answers the ground-pick and jaw commands the way walk_wired.py
would (phase ramps at 1/4 per s, hand-back to walk mode after the rise, jaw holding when an object is within reach).

  ~/.venvs/microduck-walk/bin/python -m unittest test_pickup -v
"""
import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import pickup as pk  # noqa: E402
from walk_brain import NAMES, STAND  # noqa: E402


class FakeDuck:
    """Enough of walk_wired + the world for the phase machine: commands change the state over simulated time."""

    def __init__(self, obj_xy, visible_fov_deg=70.0, grip_ok=True):
        self.t = 0.0
        self.obj = list(obj_xy)          # object in the foot frame (x forward, y left)
        self.q = dict(STAND)
        self.mode, self.torque, self.policy = "walk", True, "stand"
        self.gp_phase, self.gp_target = None, None
        self.jaw_goal, self.jaw_deg, self.jaw_holding = None, 0.0, False
        self.burst = None                # (end_t, vx, vtheta)
        self.fov = math.radians(visible_fov_deg)
        self.grip_ok = grip_ok
        self.sent = []
        self.head_target = {}

    # walk_wired side
    def send(self, cmd):
        self.sent.append((round(self.t, 2), dict(cmd)))
        self.mode = cmd["mode"]
        if cmd["mode"] == "walk":
            self.gp_phase, self.gp_target = None, None
            self.head_target = dict(cmd.get("head") or {})
            if cmd.get("for_s"):
                self.burst = (self.t + cmd["for_s"], cmd.get("vx", 0.0), cmd.get("vtheta", 0.0))
                self.burst_len = cmd["for_s"]
        elif cmd["mode"] == "groundpick":
            self.gp_target = cmd["phase"]
            if self.gp_phase is None:
                self.gp_phase = 0.0
        if "jaw" in cmd:
            self.jaw_goal = cmd["jaw"]

    def advance(self, dt=0.1):
        self.t += dt
        # head follows its target quickly
        for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll"):
            tgt = self.head_target.get(k, STAND[k])
            self.q[k] += max(-0.5 * dt / 0.1, min(0.5 * dt / 0.1, tgt - self.q[k]))
        if self.burst and self.mode == "walk":
            end, vx, vth = self.burst
            if self.t <= end:
                since = self.t - (end - self.burst_len)
                dx = math.copysign(pk.walk_dist(since) - pk.walk_dist(since - dt), vx) if vx else 0.0
                self.obj[0] -= dx
                a = vth * pk.TURN_REAL * dt  # the body turns slower than commanded (sim: 0.54 rad/s per rad/s)
                x, y = self.obj
                self.obj = [x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)]
            else:
                self.burst = None
        if self.mode == "groundpick" and self.gp_target is not None:
            step = dt / 4.0
            self.gp_phase = min(self.gp_target, self.gp_phase + step) if self.gp_target > self.gp_phase else self.gp_phase
            if self.gp_phase >= 0.9 and self.gp_target >= 0.9:
                self.mode, self.gp_phase, self.gp_target = "walk", None, None
        if self.jaw_goal == "open":
            self.jaw_deg, self.jaw_holding = 20.0, False
        elif self.jaw_goal == "grip":
            near = pk.in_window(self.obj[0] - pk.MOUTH_X, self.obj[1])
            crouched = self.gp_phase is not None and self.gp_phase >= 0.3
            self.jaw_holding = bool(self.grip_ok and near and (crouched or self.jaw_holding))
            self.jaw_deg = 8.0 if self.jaw_holding else -3.0

    def state(self):
        return dict(t=self.t, mode=self.mode, policy=self.policy, torque=self.torque, gb=[0.0, 0.0, -1.0],
                    q=dict(self.q), imu_age=0.01, gp_phase=self.gp_phase, jaw_deg=self.jaw_deg,
                    jaw_holding=self.jaw_holding, jaw_goal=self.jaw_goal, fresh=True)

    def observe(self, q, gb, object_z):
        """The object is seen when it lies within the head's yaw cone and the head looks down enough."""
        if q["head_pitch"] < 0.6 or self.mode != "walk":
            return None
        x, y = self.obj
        bearing = math.atan2(y, x)
        if abs(bearing - q["head_yaw"]) > self.fov / 2:
            return None
        r = math.hypot(x, y)
        # floor bands of the three poses (as-built sim, true camera): NEAR 0.06..0.34, MID 0.07..0.40, LOOK 0.15..2.4
        if q["neck_pitch"] < -0.2:
            lo, hi = 0.055, 0.34
        elif q["head_pitch"] > 0.95:
            lo, hi = 0.07, 0.40
        else:
            lo, hi = 0.149, 2.4
        if not lo < r < hi:
            return None
        return dict(x=x, y=y, area=500, cx=320, cy=240, frame=0)


def run(duck, max_s=120.0, **kw):
    logs = []
    p = pk.Pickup(duck.observe, duck.state, duck.send, now=lambda: duck.t, log=logs.append, **kw)
    while p.phase not in ("done", "failed") and duck.t < max_s:
        p.step()
        duck.advance()
    return p, logs


class PickupMachine(unittest.TestCase):
    def test_right_front_object_is_found_approached_and_lifted(self):
        duck = FakeDuck(obj_xy=(0.55, -0.25))  # right-front, 0.6 m away
        p, logs = run(duck)
        self.assertEqual(p.phase, "done", "\n".join(logs))
        self.assertTrue(p.result["ok"], p.result)
        modes = [c["mode"] for _, c in duck.sent]
        self.assertIn("groundpick", modes)
        gp = [c for _, c in duck.sent if c["mode"] == "groundpick"]
        self.assertEqual([c["jaw"] for c in gp], ["open", "grip", "grip"])
        self.assertEqual([c["phase"] for c in gp], [pk.GP_PHASE, pk.GP_PHASE, 1.0])
        # the final object position was inside the pickup window when the crouch was commanded
        t_crouch = next(t for t, c in duck.sent if c["mode"] == "groundpick")
        self.assertLess(abs(duck.obj[0] - pk.MOUTH_X), 0.03)
        self.assertLess(abs(duck.obj[1]), 0.03)
        # the search looked right first
        first_yaw = next(c["head"]["head_yaw"] for _, c in duck.sent if c.get("head", {}).get("head_yaw"))
        self.assertLess(first_yaw, 0)
        self.assertGreater(t_crouch, 2.0)

    def test_object_behind_needs_body_turns(self):
        duck = FakeDuck(obj_xy=(-0.4, -0.3))  # behind on the right
        p, logs = run(duck)
        self.assertEqual(p.phase, "done", "\n".join(logs))
        turns = [c for _, c in duck.sent if c["mode"] == "walk" and c.get("vtheta", 0) < 0 and c.get("vx", 0) > 0]
        self.assertTrue(turns)  # arcs: walk61 never turns in place
        self.assertTrue(p.result["ok"])

    def test_no_object_fails_after_the_search_turns(self):
        duck = FakeDuck(obj_xy=(5.0, 5.0))  # out of range
        p, logs = run(duck, max_s=300)
        self.assertEqual(p.phase, "failed")
        self.assertEqual(p.result["why"], "no orange object found")
        turns = [c for _, c in duck.sent if c["mode"] == "walk" and c.get("vtheta", 0) < 0]
        self.assertEqual(len(turns), pk.MAX_TURNS)

    def test_guard_trip_while_walking_fails(self):
        duck = FakeDuck(obj_xy=(0.6, 0.0))
        orig = duck.advance

        def advance(dt=0.1):
            orig(dt)
            if duck.t > 6.0:
                duck.mode = "stand"  # walk_wired's tilt guard switched to stand
        duck.advance = advance
        p, logs = run(duck)
        self.assertEqual(p.phase, "failed")
        self.assertIn("stand mode", p.result["why"])

    def test_one_guard_trip_is_recovered(self):
        """walk61 diverges once mid-burst (walk_wired ramps to STAND); the pickup re-stands on stand61 and finishes."""
        duck = FakeDuck(obj_xy=(0.6, 0.0))
        orig = duck.advance
        tripped = []

        def advance(dt=0.1):
            orig(dt)
            if duck.mode == "walk" and duck.burst and not tripped and duck.t > 4.0:
                duck.mode, duck.burst, tripped[:] = "stand", None, [duck.t]  # the guard: STAND ramp, burst dropped
        duck.advance = advance
        p, logs = run(duck, max_s=200)
        self.assertEqual(p.phase, "done", "\n".join(logs))
        self.assertTrue(tripped)
        self.assertEqual(p.trips, 1)
        # the first command after the trip came no sooner than the 3.5 s STAND settle and re-entered walk mode
        after = [(t, c) for t, c in duck.sent if t > tripped[0]]
        self.assertGreaterEqual(after[0][0] - tripped[0], 3.4)
        self.assertEqual(after[0][1]["mode"], "walk")
        self.assertEqual(after[0][1].get("vx", 0.0), 0.0)

    def test_bursts_walk_with_the_head_up_and_relook_before_measuring(self):
        duck = FakeDuck(obj_xy=(0.6, 0.0))
        p, logs = run(duck)
        self.assertEqual(p.phase, "done", "\n".join(logs))
        bursts = [c for _, c in duck.sent if c["mode"] == "walk" and c.get("for_s")]
        self.assertTrue(bursts)
        self.assertTrue(all("head" not in c for c in bursts))
        self.assertLessEqual(max(c["for_s"] for c in bursts), pk.BURST_MAX_S)
        # every burst is followed by a zero command carrying the look pose
        for i, (t, c) in enumerate(duck.sent):
            if c["mode"] == "walk" and c.get("for_s"):
                nxt = duck.sent[i + 1][1]
                self.assertEqual(nxt.get("vx", 0.0), 0.0)
                self.assertIn("head", nxt)

    def test_failed_grip_retries_then_gives_up(self):
        duck = FakeDuck(obj_xy=(0.3, 0.0), grip_ok=False)
        p, logs = run(duck, max_s=200)
        self.assertEqual(p.phase, "failed", "\n".join(logs))
        self.assertEqual(p.result["why"], "grip failed")
        grips = [c for _, c in duck.sent if c.get("jaw") == "grip"]
        self.assertEqual(len(grips), pk.GRIP_RETRIES + 1)
        rises = [c for _, c in duck.sent if c["mode"] == "groundpick" and c["phase"] == 1.0]
        self.assertEqual(len(rises), pk.GRIP_RETRIES + 1)

    def test_dry_run_stops_before_the_crouch(self):
        duck = FakeDuck(obj_xy=(0.4, 0.1))
        p, logs = run(duck, dry=True)
        self.assertEqual(p.phase, "done")
        self.assertIn("dry run", p.result["why"])
        self.assertFalse([c for _, c in duck.sent if c["mode"] == "groundpick"])

    def test_stale_state_fails(self):
        duck = FakeDuck(obj_xy=(0.4, 0.1))
        p = pk.Pickup(duck.observe, lambda: dict(duck.state(), t=-10.0), duck.send, now=lambda: duck.t, log=lambda s: None)
        while p.phase not in ("done", "failed") and duck.t < 30:
            p.step()
            duck.advance()
        self.assertEqual(p.result["why"], "no fresh walk_wired state")


if __name__ == "__main__":
    unittest.main()
