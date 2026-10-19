"""MicroDuck jaw (ID15) and head-pitch (ID11) safety limits, shared by every real-robot command path.

The numbers live in jaw_limits.json and the CAD contact table jaw_pose_caps.json next to this file; keep the three files
together when copying them to the Pi or into a runtime bundle. Pure Python 3.9+, no dependencies, no hardware access.

Rules (HANDOFF-20261013 open item 3, jaw-neck-check):
  1. While the jaw is open by more than jaw_open_threshold_deg (2), head_pitch never exceeds
     head_pitch_max_while_jaw_open_deg (65). The jaw opening used here is the larger of the commanded and the measured
     opening (the command capped at jaw_max_open_deg); with neither known the jaw counts as open.
  2. Jaw opening, measured from the calibrated closed jaw, never exceeds the pose cap:
     min(jaw_max_open_deg (25), first contact - jaw_pose_cap_margin_deg (2)). First contact comes from the CAD table
     jaw_pose_caps.json over neck_pitch, head_pitch, head_yaw and head_roll: the smallest value over the grid nodes
     surrounding every pose between the commanded head pose (after rule 1) and the measured one. A joint with neither
     value takes its whole range, so an unknown pose gets the most conservative cap. A cap below jaw_min_usable_open_deg (5)
     becomes 0: the jaw is held closed and the clamp is reported as "jaw_closed_zone". Closing past the closed
     reference is not limited, because the current-limited grip squeezes by commanding slightly past contact.

Apply limit_targets() (radians) or limit_goal_ticks() (XL330 ticks) at the last point before goal positions are written
to the bus.
"""
import json
import math
from pathlib import Path

_SPEC_PATH = Path(__file__).resolve().with_name("jaw_limits.json")
SPEC = json.loads(_SPEC_PATH.read_text())

JAW_ID = int(SPEC["jaw_id"])
HEAD_PITCH_ID = int(SPEC["head_pitch_id"])
JAW_MAX_OPEN_DEG = float(SPEC["jaw_max_open_deg"])
JAW_OPEN_THRESHOLD_DEG = float(SPEC["jaw_open_threshold_deg"])
HEAD_PITCH_MAX_WHILE_JAW_OPEN_DEG = float(SPEC["head_pitch_max_while_jaw_open_deg"])
JAW_HARDWARE_LIMIT_MARGIN_DEG = float(SPEC["jaw_hardware_limit_margin_deg"])
JAW_HARDWARE_LIMIT_HEAD_YAW_DEG = float(SPEC["jaw_hardware_limit_head_yaw_deg"])
TICKS_PER_REV = int(SPEC["xl330_ticks_per_rev"])
CENTER_TICK = int(SPEC["xl330_center_tick"])
HEAD_PITCH_SIGN = float(SPEC["head_pitch_sign"])
HEAD_PITCH_OFFSET_RAD = float(SPEC["head_pitch_offset_rad"])
POSE_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
POSE_IDS = {name: int(SPEC[name + "_id"]) for name in POSE_JOINTS}
POSE_SIGN = {name: float(SPEC[name + "_sign"]) for name in POSE_JOINTS}
POSE_OFFSET_RAD = {name: float(SPEC[name + "_offset_rad"]) for name in POSE_JOINTS}
JAW_POSE_CAP_MARGIN_DEG = float(SPEC["jaw_pose_cap_margin_deg"])
JAW_MIN_USABLE_OPEN_DEG = float(SPEC["jaw_min_usable_open_deg"])
JAW_POSE_EDGE_TOLERANCE_DEG = float(SPEC["jaw_pose_edge_tolerance_deg"])
POSE_TABLE_PATH = _SPEC_PATH.with_name(SPEC["jaw_pose_cap_table"])
POSE_TABLE = json.loads(POSE_TABLE_PATH.read_text())

TICKS_PER_DEG = TICKS_PER_REV / 360.0
EPS_DEG = 1e-9  # comparisons ignore floating-point rounding, so a target exactly at a limit passes unchanged
HEAD_PITCH_MAX_WHILE_JAW_OPEN_RAD = math.radians(HEAD_PITCH_MAX_WHILE_JAW_OPEN_DEG)


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Non-finite " + name)
    return value


def rad_to_tick(rad):
    """rustypot 1.4.2 Xl330 AnglePosition.to_raw: trunc(4096 (pi + rad) / (2 pi))."""
    return int(TICKS_PER_REV * (math.pi + _finite(rad, "angle")) / (2 * math.pi))


def tick_to_rad(tick):
    """rustypot 1.4.2 Xl330 AnglePosition.from_raw: 2 pi raw / 4096 - pi (2048 is 0 rad)."""
    return 2 * math.pi * _finite(tick, "tick") / TICKS_PER_REV - math.pi


def head_pitch_model_to_hardware_rad(model_rad):
    return _finite(model_rad, "head_pitch") * HEAD_PITCH_SIGN + HEAD_PITCH_OFFSET_RAD


def head_pitch_hardware_to_model_rad(hardware_rad):
    return (_finite(hardware_rad, "head_pitch") - HEAD_PITCH_OFFSET_RAD) * HEAD_PITCH_SIGN


def _head_pitch_max_tick():
    exact = CENTER_TICK + math.degrees(head_pitch_model_to_hardware_rad(HEAD_PITCH_MAX_WHILE_JAW_OPEN_RAD)) * TICKS_PER_DEG
    # Largest tick at or below the limit when ticks grow with head_pitch, smallest at or above it otherwise.
    return math.floor(exact + 1e-9) if HEAD_PITCH_SIGN > 0 else math.ceil(exact - 1e-9)


HEAD_PITCH_MAX_TICK = _head_pitch_max_tick()


def _load_pose_table(table):
    if tuple(table["joints"]) != POSE_JOINTS:
        raise ValueError("jaw_pose_caps.json joints must be " + ", ".join(POSE_JOINTS))
    axes = {}
    for name in POSE_JOINTS:
        start, step, count = table["axes_deg"][name]
        axes[name] = (float(start), float(step), int(count))
    clear = float(table["clear_to_deg"])
    shape = tuple(axes[j][2] for j in POSE_JOINTS)
    flat = []

    def walk(node, depth):
        if depth == len(shape):
            v = clear if node is None else float(node)
            if not math.isfinite(v) or v < 0:
                raise ValueError("jaw_pose_caps.json value")
            flat.append(min(v, clear))
            return
        if not isinstance(node, list) or len(node) != shape[depth]:
            raise ValueError("jaw_pose_caps.json grid shape at " + POSE_JOINTS[depth])
        for child in node:
            walk(child, depth + 1)

    walk(table["first_contact_deg"], 0)
    return axes, shape, flat


POSE_AXES, _POSE_SHAPE, _POSE_FLAT = _load_pose_table(POSE_TABLE)
POSE_TABLE_MIN_CONTACT_DEG = min(_POSE_FLAT)
_REDUCED = {(False,) * len(POSE_JOINTS): (_POSE_FLAT, _POSE_SHAPE)}


def _reduced(unknown):
    """(flat, shape) of the table minimised over the unknown axes, built from the next larger table."""
    if unknown not in _REDUCED:
        last = max(a for a, u in enumerate(unknown) if u)
        parent = tuple(u and a != last for a, u in enumerate(unknown))
        flat, shape = _reduced(parent)
        axis = sum(1 for a in range(last) if not parent[a])       # position of `last` among the parent's axes
        inner = 1
        for n in shape[axis + 1:]:
            inner *= n
        n = shape[axis]
        out = []
        for base in range(0, len(flat), n * inner):
            out.extend(map(min, *(flat[base + k * inner:base + (k + 1) * inner] for k in range(n))))
        _REDUCED[unknown] = (out, shape[:axis] + shape[axis + 1:])
    return _REDUCED[unknown]


# Built once at import, so a lookup with unknown joints never stalls a control tick.
for _unknown in sorted({tuple(bool(m >> a & 1) for a in range(len(POSE_JOINTS))) for m in range(2 ** len(POSE_JOINTS))},
                       key=sum):
    _reduced(_unknown)


def _axis_nodes(name, lo_deg, hi_deg):
    """Index range of the grid nodes surrounding [lo, hi]; None if the range is unknown or leaves the table.

    A value at most JAW_POSE_EDGE_TOLERANCE_DEG past an edge counts as lying in the last cell at that edge."""
    start, step, count = POSE_AXES[name]
    end = start + step * (count - 1)
    tol = JAW_POSE_EDGE_TOLERANCE_DEG + 1e-9
    if lo_deg is None or lo_deg < start - tol or hi_deg > end + tol:
        return None
    i0 = int(math.floor((lo_deg - start) / step + 1e-9))
    i1 = int(math.ceil((hi_deg - start) / step - 1e-9))
    if lo_deg < start - 1e-9:
        i0, i1 = 0, max(i1, 1)
    if hi_deg > end + 1e-9:
        i0, i1 = min(i0, count - 2), count - 1
    return max(0, i0), min(count - 1, i1)


def pose_contact_deg(commanded=None, measured=None):
    """Smallest first-contact jaw angle (deg) over every head pose between commanded and measured.

    commanded, measured: {joint: model radians} for any of POSE_JOINTS (missing joints allowed).
    """
    ranges = []
    for name in POSE_JOINTS:
        vals = [math.degrees(_finite(d[name], name)) for d in (commanded, measured) if d and d.get(name) is not None]
        ranges.append(_axis_nodes(name, min(vals), max(vals)) if vals else None)
    flat, shape = _reduced(tuple(r is None for r in ranges))
    known = [r for r in ranges if r is not None]
    strides, acc = [], 1
    for n in reversed(shape):
        strides.append(acc)
        acc *= n
    strides.reverse()
    offsets = [0]
    for (lo, hi), stride in zip(known, strides):
        offsets = [o + k * stride for o in offsets for k in range(lo, hi + 1)]
    return min(flat[o] for o in offsets)


def cap_for_contact(contact_deg):
    """(cap_deg, closed_zone) for a first-contact angle: min(25, contact - 2), or 0 when below the usable opening."""
    cap = min(JAW_MAX_OPEN_DEG, contact_deg - JAW_POSE_CAP_MARGIN_DEG)
    if cap < JAW_MIN_USABLE_OPEN_DEG - EPS_DEG:
        return 0.0, True
    return cap, False


def pose_jaw_cap_deg(commanded=None, measured=None):
    """(cap_deg, closed_zone, contact_deg) for the head poses between commanded and measured (model radians)."""
    contact = pose_contact_deg(commanded, measured)
    cap, closed = cap_for_contact(contact)
    return cap, closed, contact


def pose_model_to_hardware_rad(name, model_rad):
    return _finite(model_rad, name) * POSE_SIGN[name] + POSE_OFFSET_RAD[name]


def pose_hardware_to_model_rad(name, hardware_rad):
    return (_finite(hardware_rad, name) - POSE_OFFSET_RAD[name]) * POSE_SIGN[name]


class JawZero:
    """Closed-jaw reference and opening direction, from the robot's own calibration; never a default.

    closed_rad/open_rad are ID15 rustypot hardware radians (phase1 calibration.json jaw.closed_rad and jaw.open_rad,
    encoder 2048 = 0 rad, sign +1, offset 0). Use from_ticks() with raw Present Position readings instead.
    """

    def __init__(self, closed_rad, open_rad):
        self.closed_rad = _finite(closed_rad, "jaw closed reference")
        open_rad = _finite(open_rad, "jaw open reference")
        if abs(open_rad - self.closed_rad) < math.radians(0.5):
            raise ValueError("Jaw open and closed references are too close to give an opening direction")
        self.sign = 1.0 if open_rad > self.closed_rad else -1.0

    @classmethod
    def from_ticks(cls, closed_tick, open_tick):
        return cls(tick_to_rad(closed_tick), tick_to_rad(open_tick))

    @classmethod
    def from_calibration(cls, calibration):
        jaw = calibration["jaw"]
        return cls(jaw["closed_rad"], jaw["open_rad"])

    def opening_deg(self, jaw_rad):
        return math.degrees((_finite(jaw_rad, "jaw") - self.closed_rad) * self.sign)

    def jaw_rad_at(self, opening_deg):
        return self.closed_rad + self.sign * math.radians(_finite(opening_deg, "jaw opening"))

    @property
    def closed_tick_exact(self):
        return TICKS_PER_REV * (math.pi + self.closed_rad) / (2 * math.pi)

    def opening_deg_from_tick(self, tick):
        return (_finite(tick, "jaw tick") - self.closed_tick_exact) * self.sign / TICKS_PER_DEG

    def max_open_tick(self, cap_deg=None):
        """Goal tick for an opening of cap_deg (default JAW_MAX_OPEN_DEG), rounded toward the closed jaw."""
        cap_deg = JAW_MAX_OPEN_DEG if cap_deg is None else _finite(cap_deg, "jaw cap")
        exact = self.closed_tick_exact + self.sign * cap_deg * TICKS_PER_DEG
        return math.floor(exact + 1e-9) if self.sign > 0 else math.ceil(exact - 1e-9)


def clamp_jaw_rad(jaw_rad, zero):
    """Return (jaw_rad, clamped) with the opening capped at JAW_MAX_OPEN_DEG."""
    if zero.opening_deg(jaw_rad) > JAW_MAX_OPEN_DEG + EPS_DEG:
        return zero.jaw_rad_at(JAW_MAX_OPEN_DEG), True
    return float(jaw_rad), False


def jaw_counts_as_open(commanded_opening_deg=None, measured_opening_deg=None):
    known = [_finite(x, "jaw opening") for x in (commanded_opening_deg, measured_opening_deg) if x is not None]
    if not known:
        return True
    return max(known) > JAW_OPEN_THRESHOLD_DEG + EPS_DEG


def limit_head_pitch_rad(head_pitch_rad, jaw_open):
    """Return (head_pitch_rad, clamped) for the model-frame head_pitch angle."""
    head_pitch_rad = _finite(head_pitch_rad, "head_pitch")
    if jaw_open and math.degrees(head_pitch_rad) > HEAD_PITCH_MAX_WHILE_JAW_OPEN_DEG + EPS_DEG:
        return HEAD_PITCH_MAX_WHILE_JAW_OPEN_RAD, True
    return head_pitch_rad, False


def _jaw_pose_clamp(opening_deg, commanded_head, measured_head):
    """(cap_deg, clamp name or None) for a commanded jaw opening at the given head poses."""
    cap, closed, _ = pose_jaw_cap_deg(commanded_head, measured_head)
    if opening_deg > cap + EPS_DEG:
        return cap, "jaw_closed_zone" if closed else "jaw_pose"
    return cap, None


def limit_targets(targets, zero, measured_jaw_rad=None, measured_head=None):
    """Limit a name -> radians goal mapping.

    'jaw' is in ID15 hardware radians; 'neck_pitch', 'head_pitch', 'head_yaw' and 'head_roll' are model radians.
    measured_head maps any of those names to measured model radians. Other names pass through. Returns (limited_dict, clamps), clamps
    naming each rule that changed a goal: 'jaw' (25 deg), 'head_pitch' (65 deg interlock), 'jaw_pose' (pose cap) or
    'jaw_closed_zone' (pose cap below the usable opening, jaw held closed).
    """
    out = {name: _finite(value, name) for name, value in targets.items()}
    clamps = []
    commanded = None
    if "jaw" in out:
        out["jaw"], changed = clamp_jaw_rad(out["jaw"], zero)
        if changed:
            clamps.append("jaw")
        commanded = zero.opening_deg(out["jaw"])
    measured = None if measured_jaw_rad is None else zero.opening_deg(measured_jaw_rad)
    if "head_pitch" in out:
        out["head_pitch"], changed = limit_head_pitch_rad(out["head_pitch"], jaw_counts_as_open(commanded, measured))
        if changed:
            clamps.append("head_pitch")
    if "jaw" in out:
        head = {name: out[name] for name in POSE_JOINTS if name in out}
        cap, clamp = _jaw_pose_clamp(commanded, head, measured_head)
        if clamp:
            out["jaw"] = zero.jaw_rad_at(cap)
            clamps.append(clamp)
    return out, clamps


def limit_goal_ticks(goals, zero, measured_jaw_tick=None, measured_head_ticks=None):
    """Limit an ID -> raw Goal Position tick mapping for direct DYNAMIXEL SDK writers.

    ID15 and ID11 are limited; IDs 12, 11, 13 and 14 and measured_head_ticks (ID -> Present Position tick) give the
    head pose for the jaw pose cap. Other IDs pass through. Returns (limited_dict, clamps) as limit_targets.
    """
    out = {int(i): int(t) for i, t in goals.items()}
    clamps = []
    commanded = None
    if JAW_ID in out:
        if zero.opening_deg_from_tick(out[JAW_ID]) > JAW_MAX_OPEN_DEG + EPS_DEG:
            out[JAW_ID] = zero.max_open_tick()
            clamps.append("jaw")
        commanded = zero.opening_deg_from_tick(out[JAW_ID])
    measured = None if measured_jaw_tick is None else zero.opening_deg_from_tick(measured_jaw_tick)
    if HEAD_PITCH_ID in out and jaw_counts_as_open(commanded, measured):
        beyond = out[HEAD_PITCH_ID] > HEAD_PITCH_MAX_TICK if HEAD_PITCH_SIGN > 0 else out[HEAD_PITCH_ID] < HEAD_PITCH_MAX_TICK
        if beyond:
            out[HEAD_PITCH_ID] = HEAD_PITCH_MAX_TICK
            clamps.append("head_pitch")
    if JAW_ID in out:
        def model(ticks):
            return {name: pose_hardware_to_model_rad(name, tick_to_rad(ticks[POSE_IDS[name]]))
                    for name in POSE_JOINTS if ticks and ticks.get(POSE_IDS[name]) is not None}
        cap, clamp = _jaw_pose_clamp(commanded, model(out), model(measured_head_ticks))
        if clamp:
            out[JAW_ID] = zero.max_open_tick(cap)
            clamps.append(clamp)
    return out, clamps
