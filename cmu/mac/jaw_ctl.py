"""Jaw servo (ID15) for walk_wired.py: calibration by hand, current-limited grip, the shared jaw/head limits.

The jaw is not one of the 14 policy joints: walk_wired.py reads its position with the body and writes its goal
separately. Its closed position depends on how the horn was fitted (guide step "jaw"), so it is measured once on the
robot and kept in pi/jaw_calibration.json: with the jaw torque off, close the beak by hand and send
{"jaw_mark": "closed"}, open it by hand and send {"jaw_mark": "open"}; the file holds both ticks and the opening
direction follows from them (jaw_limits.JawZero). Until both marks exist no jaw goal is written.

Grip: ID15 runs in Operating Mode 5 (current-based position control, as the phase 1 runtime commissioned it) with
Goal Current GOAL_CURRENT_MA, so a "grip" command aims SQUEEZE_DEG past the closed reference and the current limit,
not the position loop, holds the object. A jaw that stops GRIP_MIN_DEG or more from closed while gripping holds
something. Every goal passes jaw_limits (pi/jaw_limits.py): opening at most 25 deg, less where the commanded or
measured head pose brings the jaw near other parts (jaw_pose_caps.json). jaw_limits.json's own head sign/offset table
is not used: head poses are handed over in model radians from walk_brain's to_model, which carries this build's signs.

Control-file keys (beside "mode"): "jaw": opening in degrees from closed (rest at 2, not 0: a goal on the closed
stop keeps the current-limited servo pushing and humming), or "grip"; "jaw_torque": false releases the
jaw (calibration, or to free an object by hand); "jaw_mark": "closed" | "open" records the present tick.
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PI_DIR = os.path.join(HERE, "..", "pi")
sys.path.insert(0, PI_DIR)
import jaw_limits as jl  # noqa: E402

JAW_ID = jl.JAW_ID  # 15
CAL_PATH = os.path.join(PI_DIR, "jaw_calibration.json")
ADDR_OPERATING_MODE, ADDR_CURRENT_LIMIT, ADDR_TORQUE, ADDR_P_GAIN = 11, 38, 64, 84
ADDR_GOAL_CURRENT, ADDR_GOAL, ADDR_PRESENT_CURRENT, ADDR_PRESENT_POS = 102, 116, 126, 132
MODE_CURRENT_POSITION = 5
GOAL_CURRENT_MA = 100     # grip force limit; the phase 1 runtime accepted 20..150 mA (XL330 1 mA per LSB)
CURRENT_LIMIT_MA = 150    # EEPROM Current Limit, the ceiling Goal Current may use
KP_JAW = 200              # position P gain in mode 5; 400 buzzed audibly on the unloaded jaw (2026-10-16)
SQUEEZE_DEG = -3.0        # "grip" aims this far past closed; the servo stops on the object or at the closed stop
GRIP_MIN_DEG = 3.0        # measured opening at or above this while gripping = something is between the bills
OPEN_DEG = 20.0           # "open" shortcut; the pose cap may lower it
TICK = 2 * math.pi / 4096
POSE_JOINTS = jl.POSE_JOINTS  # neck_pitch, head_pitch, head_yaw, head_roll


def tick_to_hw_rad(tick):
    return (float(tick) - 2048) * TICK


def hw_rad_to_tick(rad):
    return int(round(2048 + rad / TICK))


class JawCtl:
    def __init__(self, cal_path=CAL_PATH, kp=KP_JAW):
        self.cal_path = cal_path
        self.kp = int(kp)
        self.cal = {}
        self.zero = None
        self.on = False            # jaw torque enabled by this process
        self.goal_deg = None       # wanted opening (deg) or "grip"; None = nothing written
        self.last_tick = None      # latest Present Position
        self.last_goal_tick = None
        self.clamps = []
        self.mode_checked = None
        if os.path.exists(cal_path):
            self.cal = json.load(open(cal_path))
            self._build_zero()

    def _build_zero(self):
        if "closed_tick" in self.cal and "open_tick" in self.cal:
            self.zero = jl.JawZero.from_ticks(self.cal["closed_tick"], self.cal["open_tick"])
        else:
            self.zero = None

    @property
    def calibrated(self):
        return self.zero is not None

    def mark(self, which, tick):
        """Record the present tick as the closed or open reference (jaw torque off, beak moved by hand)."""
        if which not in ("closed", "open"):
            raise ValueError(which)
        if tick is None:
            raise RuntimeError("no ID15 reading to mark")
        self.cal[f"{which}_tick"] = int(tick)
        self.cal[f"{which}_marked"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.cal_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cal, f, indent=1)
        os.replace(tmp, self.cal_path)
        try:
            self._build_zero()
        except ValueError as e:  # both marks at the same tick: the beak was not moved between them
            self.zero = None
            raise ValueError(f"{e}; both marks are saved, move the beak and mark again") from None
        return self.cal

    def opening_deg(self, tick=None):
        tick = self.last_tick if tick is None else tick
        if tick is None or self.zero is None:
            return None
        return self.zero.opening_deg_from_tick(tick)

    def goal_tick(self, want, commanded_head, measured_head, measured_tick=None):
        """Goal Position tick for want (deg or "grip") under the shared limits; (tick, clamps)."""
        if self.zero is None:
            raise RuntimeError("jaw not calibrated: mark closed and open first")
        deg = SQUEEZE_DEG if want == "grip" else OPEN_DEG if want == "open" else float(want)
        targets = {"jaw": self.zero.jaw_rad_at(deg)}
        targets.update({n: float(commanded_head[n]) for n in POSE_JOINTS if commanded_head and n in commanded_head})
        measured_jaw = None if measured_tick is None else tick_to_hw_rad(measured_tick)
        out, clamps = jl.limit_targets(targets, self.zero, measured_jaw_rad=measured_jaw,
                                       measured_head={n: float(measured_head[n]) for n in POSE_JOINTS if n in measured_head})
        return hw_rad_to_tick(out["jaw"]), clamps

    def holding(self):
        """True while a grip command stopped short of the closed reference (an object between the bills)."""
        deg = self.opening_deg()
        return self.goal_deg == "grip" and deg is not None and deg >= GRIP_MIN_DEG

    # --- bus access through walk_wired.Link (sync_read / sync_write on the shared port) ---------------------------
    @staticmethod
    def _read1(link, addr, size=1, tries=10):
        for _ in range(tries):
            got = link.sync_read(addr, size, [JAW_ID], timeout=0.03, allow_alert=True)
            if JAW_ID in got:
                return int.from_bytes(got[JAW_ID], "little", signed=size == 4)
        return None

    def setup(self, link):
        """Put ID15 in current-based position mode with the grip current and enable its torque where it is."""
        mode = self._read1(link, ADDR_OPERATING_MODE)
        if mode is None:
            raise RuntimeError("ID15 does not answer")
        if mode != MODE_CURRENT_POSITION:
            link.sync_write(ADDR_TORQUE, 1, {JAW_ID: 0})
            link.sync_write(ADDR_OPERATING_MODE, 1, {JAW_ID: MODE_CURRENT_POSITION})  # EEPROM, torque must be off
            link.sync_write(ADDR_CURRENT_LIMIT, 2, {JAW_ID: CURRENT_LIMIT_MA})
            time.sleep(0.05)
            mode = self._read1(link, ADDR_OPERATING_MODE)
            if mode != MODE_CURRENT_POSITION:
                raise RuntimeError(f"ID15 operating mode is {mode}, not {MODE_CURRENT_POSITION}")
        self.mode_checked = mode
        pos = self._read1(link, ADDR_PRESENT_POS, 4)
        if pos is None:
            raise RuntimeError("ID15 position read failed")
        link.sync_write(ADDR_GOAL_CURRENT, 2, {JAW_ID: GOAL_CURRENT_MA})
        link.sync_write(ADDR_P_GAIN, 2, {JAW_ID: self.kp})
        link.sync_write(ADDR_GOAL, 4, {JAW_ID: pos})  # hold where it is, then enable
        link.sync_write(ADDR_TORQUE, 1, {JAW_ID: 1})
        self.on, self.last_tick, self.last_goal_tick = True, pos, pos
        return pos

    def release(self, link):
        link.sync_write(ADDR_TORQUE, 1, {JAW_ID: 0})
        self.on, self.goal_deg, self.last_goal_tick = False, None, None

    def write(self, link, commanded_head, measured_head):
        """Write the goal for self.goal_deg (if any); returns the tick written or None."""
        if not self.on or self.goal_deg is None or self.zero is None:
            return None
        tick, self.clamps = self.goal_tick(self.goal_deg, commanded_head, measured_head, self.last_tick)
        if tick != self.last_goal_tick:
            link.sync_write(ADDR_GOAL, 4, {JAW_ID: tick})
            self.last_goal_tick = tick
        return tick
