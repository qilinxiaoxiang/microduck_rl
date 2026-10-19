"""Unit tests for jaw_limits.py and the pure planning part of set_jaw_limit.py. No hardware, no dependencies.

Run from this folder:  python3 -m unittest test_jaw_limits -v
"""
import hashlib
import itertools
import math
import unittest
from pathlib import Path

import jaw_limits as jl
import set_jaw_limit as sjl

HERE = Path(__file__).resolve().parent
SIM_MIRROR = HERE.parents[2] / "microduck-simulator/app/src/game/jaw-limits.json"
SIM_TABLE = HERE.parents[2] / "microduck-simulator/app/src/game/jaw-pose-caps.json"
STRAIGHT = {"neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0}
RULES = ("jaw", "head_pitch")


def zero(closed_deg=-10.0, open_deg=10.0):
    return jl.JawZero(math.radians(closed_deg), math.radians(open_deg))


class Numbers(unittest.TestCase):
    def test_agreed_values(self):
        self.assertEqual(jl.JAW_MAX_OPEN_DEG, 25.0)
        self.assertEqual(jl.HEAD_PITCH_MAX_WHILE_JAW_OPEN_DEG, 65.0)
        self.assertEqual(jl.JAW_OPEN_THRESHOLD_DEG, 2.0)
        self.assertEqual(jl.JAW_HARDWARE_LIMIT_MARGIN_DEG, 2.0)
        self.assertEqual(jl.JAW_HARDWARE_LIMIT_HEAD_YAW_DEG, 60.0)
        self.assertEqual((jl.JAW_ID, jl.HEAD_PITCH_ID), (15, 11))

    def test_cap_clears_every_measured_yaw_and_the_trained_opening(self):
        # margin-yaw.json minimum first contact 26.25 deg; trained policy maximum 18.33 deg.
        self.assertLess(jl.JAW_MAX_OPEN_DEG, 26.25)
        self.assertGreater(jl.JAW_MAX_OPEN_DEG, 18.33)

    def test_simulator_mirror_is_byte_identical(self):
        if not SIM_MIRROR.exists():
            self.skipTest("simulator checkout not present")
        canonical = (HERE / "jaw_limits.json").read_bytes()
        self.assertEqual(hashlib.sha256(SIM_MIRROR.read_bytes()).hexdigest(), hashlib.sha256(canonical).hexdigest(),
                         "Copy guide/firmware/pi/jaw_limits.json over " + str(SIM_MIRROR))


class Ticks(unittest.TestCase):
    def test_rustypot_mapping(self):
        self.assertEqual(jl.rad_to_tick(0.0), 2048)
        self.assertAlmostEqual(jl.tick_to_rad(2048), 0.0)
        self.assertEqual(jl.rad_to_tick(math.pi / 2), 3072)
        self.assertEqual(jl.rad_to_tick(jl.tick_to_rad(2787)), 2787)

    def test_head_pitch_cap_tick(self):
        # 2048 + 65 deg * 4096 / 360 = 2787.56: the last whole tick at or below 65 deg.
        self.assertEqual(jl.HEAD_PITCH_MAX_TICK, 2787)
        self.assertLessEqual(math.degrees(jl.tick_to_rad(jl.HEAD_PITCH_MAX_TICK)), 65.0)
        self.assertGreater(math.degrees(jl.tick_to_rad(jl.HEAD_PITCH_MAX_TICK + 1)), 65.0)


class JawCap(unittest.TestCase):
    def test_below_and_at_cap_unchanged(self):
        z = zero()
        for opening in (0.0, 18.33, 24.99, 25.0):
            out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(opening)}, z, measured_head=STRAIGHT)
            self.assertAlmostEqual(z.opening_deg(out["jaw"]), opening, places=9)
            self.assertEqual(clamps, [])

    def test_above_cap_clamped_to_cap(self):
        z = zero()
        for opening in (25.01, 30.0, 40.0, 170.0):
            out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(opening)}, z, measured_head=STRAIGHT)
            self.assertAlmostEqual(z.opening_deg(out["jaw"]), 25.0, places=9)
            self.assertEqual(clamps, ["jaw"])

    def test_negative_opening_direction(self):
        z = zero(closed_deg=30.0, open_deg=12.0)
        self.assertEqual(z.sign, -1.0)
        out, clamps = jl.limit_targets({"jaw": math.radians(30.0 - 31.0)}, z, measured_head=STRAIGHT)
        self.assertAlmostEqual(math.degrees(out["jaw"]), 5.0, places=9)
        self.assertEqual(clamps, ["jaw"])
        out, clamps = jl.limit_targets({"jaw": math.radians(10.0)}, z, measured_head=STRAIGHT)
        self.assertEqual(clamps, [])

    def test_squeeze_past_closed_is_not_limited(self):
        z = zero()
        for head in (STRAIGHT, None):
            out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(-2.0)}, z, measured_head=head)
            self.assertAlmostEqual(z.opening_deg(out["jaw"]), -2.0, places=9)
            self.assertEqual(clamps, [])

    def test_other_joints_pass_through(self):
        z = zero()
        out, clamps = jl.limit_targets({"neck_pitch": 1.2, "head_yaw": 2.9, "left_knee": -1.0}, z, measured_jaw_rad=z.jaw_rad_at(20))
        self.assertEqual(out, {"neck_pitch": 1.2, "head_yaw": 2.9, "left_knee": -1.0})
        self.assertEqual(clamps, [])

    def test_too_close_references_rejected(self):
        with self.assertRaises(ValueError):
            jl.JawZero(0.1, 0.1 + math.radians(0.2))

    def test_non_finite_rejected(self):
        z = zero()
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                jl.limit_targets({"jaw": bad}, z)
            with self.assertRaises(ValueError):
                jl.limit_targets({"head_pitch": bad, "jaw": z.closed_rad}, z)


class HeadInterlock(unittest.TestCase):
    def run_case(self, jaw_opening, head_deg, measured=None):
        z = zero()
        targets = {"head_pitch": math.radians(head_deg)}
        if jaw_opening is not None:
            targets["jaw"] = z.jaw_rad_at(jaw_opening)
        out, clamps = jl.limit_targets(targets, z, None if measured is None else z.jaw_rad_at(measured), STRAIGHT)
        return math.degrees(out["head_pitch"]), [c for c in clamps if c in RULES]

    def test_closed_jaw_allows_any_head_pitch(self):
        for jaw in (0.0, 1.0, 2.0):
            head, clamps = self.run_case(jaw, 85.0)
            self.assertAlmostEqual(head, 85.0)
            self.assertEqual(clamps, [])

    def test_open_jaw_caps_head_pitch(self):
        for jaw in (2.0001, 5.0, 18.33, 25.0):
            head, clamps = self.run_case(jaw, 85.0)
            self.assertAlmostEqual(head, 65.0)
            self.assertEqual(clamps, ["head_pitch"])

    def test_head_pitch_boundary(self):
        self.assertEqual(self.run_case(18.0, 65.0), (65.0, []))
        self.assertAlmostEqual(self.run_case(18.0, 64.9)[0], 64.9)
        head, clamps = self.run_case(18.0, 65.01)
        self.assertAlmostEqual(head, 65.0)
        self.assertEqual(clamps, ["head_pitch"])

    def test_negative_head_pitch_untouched(self):
        head, clamps = self.run_case(18.33, -44.0)
        self.assertAlmostEqual(head, -44.0)
        self.assertEqual(clamps, [])

    def test_measured_open_jaw_counts(self):
        # Commanded closed while the jaw still holds an object or has not closed yet.
        head, clamps = self.run_case(0.0, 80.0, measured=16.0)
        self.assertAlmostEqual(head, 65.0)
        self.assertEqual(clamps, ["head_pitch"])
        # Commanded open, measured still closed: the command decides.
        head, clamps = self.run_case(18.0, 80.0, measured=0.0)
        self.assertAlmostEqual(head, 65.0)

    def test_unknown_jaw_counts_as_open(self):
        head, clamps = self.run_case(None, 80.0)
        self.assertAlmostEqual(head, 65.0)
        head, clamps = self.run_case(None, 80.0, measured=1.0)
        self.assertAlmostEqual(head, 80.0)

    def test_jaw_cap_and_head_cap_together(self):
        z = zero()
        out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(35.0), "head_pitch": math.radians(84.7)}, z, measured_head=STRAIGHT)
        cap, closed, _ = jl.pose_jaw_cap_deg({"head_pitch": math.radians(65.0)}, STRAIGHT)
        self.assertAlmostEqual(z.opening_deg(out["jaw"]), min(25.0, cap))
        self.assertAlmostEqual(math.degrees(out["head_pitch"]), 65.0)
        self.assertEqual(clamps[:2], ["jaw", "head_pitch"])
        self.assertEqual(clamps[2:], [] if cap >= 25.0 else ["jaw_closed_zone" if closed else "jaw_pose"])


class GoalTicks(unittest.TestCase):
    def test_jaw_ticks_positive_direction(self):
        z = jl.JawZero.from_ticks(1000, 1200)
        cap = z.max_open_tick()
        self.assertEqual(cap, 1284)  # 1000 + floor(25 * 4096 / 360 = 284.44)
        st = {12: 2048, 11: 2048, 13: 2048, 14: 2048}
        self.assertEqual(jl.limit_goal_ticks({15: 1284}, z, measured_head_ticks=st), ({15: 1284}, []))
        self.assertEqual(jl.limit_goal_ticks({15: 1285}, z, measured_head_ticks=st), ({15: 1284}, ["jaw"]))
        self.assertEqual(jl.limit_goal_ticks({15: 990}, z, measured_head_ticks=st), ({15: 990}, []))

    def test_jaw_ticks_negative_direction(self):
        z = jl.JawZero.from_ticks(3000, 2800)
        self.assertEqual(z.max_open_tick(), 2716)  # 3000 - 284.44, rounded towards closed
        st = {12: 2048, 11: 2048, 13: 2048, 14: 2048}
        self.assertEqual(jl.limit_goal_ticks({15: 2700}, z, measured_head_ticks=st), ({15: 2716}, ["jaw"]))
        self.assertEqual(jl.limit_goal_ticks({15: 2716}, z, measured_head_ticks=st), ({15: 2716}, []))

    def test_head_ticks(self):
        z = jl.JawZero.from_ticks(1000, 1200)
        st = {12: 2048, 11: 2048, 13: 2048, 14: 2048}

        def run(goals, **kw):
            out, clamps = jl.limit_goal_ticks(goals, z, measured_head_ticks=st, **kw)
            return out, [c for c in clamps if c in RULES]
        # 1100 is 8.8 deg open, inside every pose cap between head_pitch 0 and 65 at yaw 0, roll 0.
        self.assertGreater(jl.pose_jaw_cap_deg({"head_pitch": math.radians(65)}, STRAIGHT)[0], 8.8)
        self.assertEqual(run({12: 2048, 11: 3000, 13: 2048, 14: 2048, 15: 1100}), ({12: 2048, 11: 2787, 13: 2048, 14: 2048, 15: 1100}, ["head_pitch"]))
        self.assertEqual(run({12: 2048, 11: 2787, 13: 2048, 14: 2048, 15: 1100}), ({12: 2048, 11: 2787, 13: 2048, 14: 2048, 15: 1100}, []))
        self.assertEqual(run({11: 3000, 15: 1010}), ({11: 3000, 15: 1010}, []))
        self.assertEqual(run({11: 3000, 15: 1000}, measured_jaw_tick=1150), ({11: 2787, 15: 1000}, ["head_pitch"]))
        self.assertEqual(run({11: 3000, 1: 77}), ({11: 2787, 1: 77}, ["head_pitch"]))


def deg_pose(pitch=None, yaw=None, roll=None, neck=0.0):
    return {k: math.radians(v) for k, v in (("neck_pitch", neck), ("head_pitch", pitch), ("head_yaw", yaw),
                                           ("head_roll", roll)) if v is not None}


class PoseCap(unittest.TestCase):
    table = jl.POSE_TABLE
    shape = tuple(jl.POSE_AXES[n][2] for n in jl.POSE_JOINTS)

    def node(self, i):
        v = self.table["first_contact_deg"][i[0]][i[1]][i[2]][i[3]]
        return self.table["clear_to_deg"] if v is None else v

    def at(self, i, frac=(0, 0, 0, 0)):
        return {n: math.radians(jl.POSE_AXES[n][0] + (i[a] + frac[a]) * jl.POSE_AXES[n][1])
                for a, n in enumerate(jl.POSE_JOINTS)}

    def box_min(self, lo, hi):
        return min(self.node(i) for i in itertools.product(*(range(a, b + 1) for a, b in zip(lo, hi))))

    def test_grid_covers_the_runtime_joint_ranges(self):
        # calibration.template.json = MJCF: neck_pitch -90..60, head_pitch -90..90, head_yaw -170..170, head_roll -25..25.
        self.assertEqual(jl.POSE_JOINTS, ("neck_pitch", "head_pitch", "head_yaw", "head_roll"))
        self.assertEqual(jl.POSE_AXES["neck_pitch"], (-90.0, 15.0, 11))
        self.assertEqual(jl.POSE_AXES["head_pitch"], (-90.0, 5.0, 37))
        self.assertEqual(jl.POSE_AXES["head_yaw"], (-170.0, 10.0, 35))
        self.assertEqual(jl.POSE_AXES["head_roll"], (-25.0, 5.0, 11))
        self.assertEqual(self.table["clear_to_deg"], 27.0)
        self.assertEqual(jl.POSE_TABLE_PATH.name, "jaw_pose_caps.json")
        self.assertEqual((jl.POSE_IDS, jl.POSE_SIGN), ({"neck_pitch": 12, "head_pitch": 11, "head_yaw": 13, "head_roll": 14},
                                                       dict.fromkeys(jl.POSE_JOINTS, 1.0)))

    def test_simulator_table_is_byte_identical(self):
        if not SIM_TABLE.exists():
            self.skipTest("simulator checkout not present")
        self.assertEqual(hashlib.sha256(SIM_TABLE.read_bytes()).hexdigest(),
                         hashlib.sha256(jl.POSE_TABLE_PATH.read_bytes()).hexdigest(),
                         "Run jaw-neck-check/scripts/build_pose_caps.py to copy jaw_pose_caps.json over " + str(SIM_TABLE))

    def test_every_node_reads_its_own_value(self):
        for i in itertools.product(*(range(n) for n in self.shape)):
            self.assertEqual(jl.pose_contact_deg(self.at(i)), self.node(i), i)

    def test_inside_a_cell_the_smallest_corner_counts(self):
        for i in ((0, 0, 0, 0), (6, 20, 10, 5), (9, 35, 33, 9), (3, 9, 28, 4), (5, 29, 17, 9)):
            self.assertEqual(jl.pose_contact_deg(self.at(i, (0.5, 0.5, 0.5, 0.5))), self.box_min(i, [x + 1 for x in i]), i)

    def test_node_boundaries(self):
        # Exactly on a node uses that node only; just past it brings in the next node.
        for i in ((6, 9, 28, 5), (6, 29, 17, 10), (2, 18, 17, 5)):
            q = self.at(i)
            self.assertEqual(jl.pose_contact_deg(q), self.node(i))
            for axis, step in ((0, 1), (1, 1), (2, -1)):
                j = list(i)
                j[axis] += step
                name = jl.POSE_JOINTS[axis]
                moved = dict(q, **{name: q[name] + step * math.radians(1e-4)})
                self.assertEqual(jl.pose_contact_deg(moved), min(self.node(i), self.node(j)), (i, axis))

    def test_commanded_and_measured_span_a_box(self):
        cmd, meas = deg_pose(10, -20, 5, neck=-30), deg_pose(42, 15, -5, neck=-10)
        want = self.box_min((4, 20, 15, 3), (6, 27, 19, 6))
        self.assertEqual(jl.pose_contact_deg(cmd, meas), want)
        self.assertEqual(jl.pose_contact_deg(meas, cmd), want)

    def test_unknown_pose_is_most_conservative(self):
        every = min(jl._POSE_FLAT)
        self.assertEqual(jl.POSE_TABLE_MIN_CONTACT_DEG, every)
        self.assertEqual(jl.pose_contact_deg(), every)
        self.assertEqual(jl.pose_contact_deg({}, None), every)
        # One joint missing takes its whole axis; a value off the grid counts as unknown.
        last = [n - 1 for n in self.shape]
        self.assertEqual(jl.pose_contact_deg(deg_pose(pitch=0, yaw=60)), self.box_min((6, 18, 23, 0), (6, 18, 23, last[3])))
        self.assertEqual(jl.pose_contact_deg(deg_pose(95, 0, 0)), self.box_min((6, 0, 17, 5), (6, last[1], 17, 5)))
        self.assertEqual(jl.pose_contact_deg(deg_pose(0, 0, 0, neck=None)), self.box_min((0, 18, 17, 5), (last[0], 18, 17, 5)))
        z = zero()
        cap, closed = jl.cap_for_contact(every)
        out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(20.0)}, z)
        self.assertAlmostEqual(z.opening_deg(out["jaw"]), min(20.0, cap), places=9)
        self.assertEqual(clamps, [] if cap >= 20.0 else ["jaw_closed_zone" if closed else "jaw_pose"])

    def test_edge_tolerance(self):
        # Up to 0.5 deg past an edge counts as the last cell (simulated limits overshoot by up to 0.28 deg); further out
        # the joint is unknown.
        self.assertEqual(jl.JAW_POSE_EDGE_TOLERANCE_DEG, 0.5)
        q = deg_pose(0, 0, 0, neck=-90.3)
        self.assertEqual(jl.pose_contact_deg(q), self.box_min((0, 18, 17, 5), (1, 18, 17, 5)))
        q = deg_pose(0, 0, 25.4)
        self.assertEqual(jl.pose_contact_deg(q), self.box_min((6, 18, 17, 9), (6, 18, 17, 10)))
        q = deg_pose(0, 0, 0, neck=-90.6)
        self.assertEqual(jl.pose_contact_deg(q), self.box_min((0, 18, 17, 5), (self.shape[0] - 1, 18, 17, 5)))

    def test_reductions_match_brute_force(self):
        import random
        rng = random.Random(20261013)
        for _ in range(200):
            q = {n: math.radians(jl.POSE_AXES[n][0] + rng.random() * jl.POSE_AXES[n][1] * (jl.POSE_AXES[n][2] - 1))
                 for n in jl.POSE_JOINTS if rng.random() < 0.7}
            lo, hi = [], []
            for a, n in enumerate(jl.POSE_JOINTS):
                if n in q:
                    r = jl._axis_nodes(n, math.degrees(q[n]), math.degrees(q[n]))
                    lo.append(r[0]), hi.append(r[1])
                else:
                    lo.append(0), hi.append(self.shape[a] - 1)
            self.assertEqual(jl.pose_contact_deg(q), self.box_min(lo, hi), q)

    def test_cap_formula(self):
        self.assertEqual(jl.cap_for_contact(27.0), (25.0, False))
        self.assertEqual(jl.cap_for_contact(40.0), (25.0, False))
        self.assertEqual(jl.cap_for_contact(20.0), (18.0, False))
        self.assertEqual(jl.cap_for_contact(7.0), (5.0, False))
        self.assertEqual(jl.cap_for_contact(6.99), (0.0, True))
        self.assertEqual(jl.cap_for_contact(0.0), (0.0, True))

    def test_straight_head_keeps_the_full_cap(self):
        self.assertEqual(jl.pose_jaw_cap_deg(STRAIGHT, STRAIGHT)[:2], (25.0, False))

    def test_pose_cap_clamps_the_jaw(self):
        z = zero()
        head = deg_pose(60, 0, 25)
        cap, closed, contact = jl.pose_jaw_cap_deg(head, head)
        self.assertLess(cap, 18.33)
        self.assertFalse(closed)
        out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(24.0), **head}, z, measured_head=head)
        self.assertAlmostEqual(z.opening_deg(out["jaw"]), cap, places=9)
        self.assertEqual(clamps, ["jaw_pose"])
        self.assertEqual(out["head_roll"], head["head_roll"])
        below = jl.limit_targets({"jaw": z.jaw_rad_at(cap - 0.5), **head}, z, measured_head=head)
        self.assertEqual(below[1], [])

    def test_closed_zone_holds_the_jaw_closed(self):
        z = zero()
        zone = deg_pose(-45, 140, 0)
        cap, closed, contact = jl.pose_jaw_cap_deg(zone, zone)
        self.assertEqual((cap, closed), (0.0, True))
        self.assertLess(contact, jl.JAW_MIN_USABLE_OPEN_DEG + jl.JAW_POSE_CAP_MARGIN_DEG)
        out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(10.0), **zone}, z, measured_head=zone)
        self.assertAlmostEqual(z.opening_deg(out["jaw"]), 0.0, places=9)
        self.assertEqual(clamps, ["jaw_closed_zone"])
        out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(-1.0), **zone}, z, measured_head=zone)
        self.assertAlmostEqual(z.opening_deg(out["jaw"]), -1.0, places=9)  # squeeze past closed stays allowed
        self.assertEqual(clamps, [])
        # Commanded clear of the zone while the head is still in it: the measured pose keeps the jaw closed.
        out, clamps = jl.limit_targets({"jaw": z.jaw_rad_at(10.0), **STRAIGHT}, z, measured_head=zone)
        self.assertEqual(clamps, ["jaw_closed_zone"])

    def test_goal_ticks_use_the_pose_cap(self):
        z = jl.JawZero.from_ticks(1000, 1200)
        zone_ticks = {12: 2048, 11: jl.rad_to_tick(math.radians(-45)), 13: jl.rad_to_tick(math.radians(140)), 14: 2048}
        out, clamps = jl.limit_goal_ticks({15: 1150, **zone_ticks}, z, measured_head_ticks=zone_ticks)
        self.assertEqual((out[15], clamps), (1000, ["jaw_closed_zone"]))
        head = deg_pose(60, 0, 25)
        ticks = {12: 2048, 11: jl.rad_to_tick(head["head_pitch"]), 13: 2048, 14: jl.rad_to_tick(head["head_roll"])}
        out, clamps = jl.limit_goal_ticks({15: 1280, **ticks}, z, measured_head_ticks=ticks)
        self.assertEqual(clamps, ["jaw_pose"])
        cap = jl.pose_jaw_cap_deg({n: jl.tick_to_rad(ticks[jl.POSE_IDS[n]]) for n in jl.POSE_JOINTS})[0]
        self.assertEqual(out[15], z.max_open_tick(cap))
        out, clamps = jl.limit_goal_ticks({15: 1150}, z)  # no head data at all: most conservative cap
        self.assertLessEqual(z.opening_deg_from_tick(out[15]), jl.cap_for_contact(jl.POSE_TABLE_MIN_CONTACT_DEG)[0] + 1e-9)


class SetJawLimitPlan(unittest.TestCase):
    def test_increasing_ticks(self):
        p = sjl.plan_limit(closed_tick=1000, contact_tick=1300, current_min=0, current_max=4095, operating_mode=3)
        self.assertEqual(p["register"], "max_position_limit")
        self.assertEqual(p["new_limit_tick"], 1300 - 23)
        self.assertEqual(p["new_min"], 0)
        self.assertEqual(p["new_max"], 1277)
        self.assertAlmostEqual(p["contact_opening_deg"], 300 * 360 / 4096)
        self.assertTrue(p["writable"])
        self.assertTrue(p["enforced_in_current_mode"])

    def test_decreasing_ticks(self):
        p = sjl.plan_limit(closed_tick=3000, contact_tick=2700, current_min=0, current_max=4095, operating_mode=3)
        self.assertEqual(p["register"], "min_position_limit")
        self.assertEqual(p["new_limit_tick"], 2723)
        self.assertEqual((p["new_min"], p["new_max"]), (2723, 4095))
        self.assertTrue(p["writable"])

    def test_limit_is_at_least_two_degrees_short(self):
        p = sjl.plan_limit(closed_tick=1000, contact_tick=1300, current_min=0, current_max=4095, operating_mode=3)
        self.assertGreaterEqual((1300 - p["new_limit_tick"]) * 360 / 4096, 2.0)

    def test_mode5_is_flagged_not_enforced(self):
        p = sjl.plan_limit(closed_tick=1000, contact_tick=1300, current_min=0, current_max=4095, operating_mode=5)
        self.assertFalse(p["enforced_in_current_mode"])
        self.assertTrue(p["writable"])
        self.assertTrue(any("Operating Mode 5" in w for w in p["warnings"]))

    def test_wrap_or_out_of_range_refused(self):
        for closed, contact in ((4050, 4300), (100, -150), (-20, 200)):
            p = sjl.plan_limit(closed_tick=closed, contact_tick=contact, current_min=0, current_max=4095, operating_mode=3)
            self.assertFalse(p["writable"], (closed, contact))

    def test_closed_must_stay_inside_window(self):
        p = sjl.plan_limit(closed_tick=1000, contact_tick=1300, current_min=1100, current_max=4095, operating_mode=3)
        self.assertFalse(p["writable"])

    def test_implausible_contact_refused(self):
        for contact in (1000, 1010, 1000 + 40):  # no motion, below the 2 deg margin
            p = sjl.plan_limit(closed_tick=1000, contact_tick=contact, current_min=0, current_max=4095, operating_mode=3)
            self.assertFalse(p["writable"], contact)

    def test_early_contact_warns(self):
        p = sjl.plan_limit(closed_tick=1000, contact_tick=1000 + round(20 * 4096 / 360), current_min=0, current_max=4095, operating_mode=3)
        self.assertTrue(p["writable"])
        self.assertTrue(any("earlier than" in w for w in p["warnings"]))

    def test_contact_inside_software_cap_warns(self):
        inside = sjl.plan_limit(closed_tick=1000, contact_tick=1000 + round(24.5 * 4096 / 360), current_min=0, current_max=4095, operating_mode=3)
        self.assertTrue(inside["writable"])
        self.assertTrue(any("inside the 25.0 deg software cap" in w for w in inside["warnings"]))
        cad = sjl.plan_limit(closed_tick=1000, contact_tick=1000 + round(26.25 * 4096 / 360), current_min=0, current_max=4095, operating_mode=3)
        self.assertEqual(cad["warnings"], [])
        self.assertAlmostEqual(cad["limit_opening_deg"], (299 - 23) * 360 / 4096)  # about 24.26 deg


if __name__ == "__main__":
    unittest.main()
