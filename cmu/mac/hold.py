"""Graded hold: how much a person has taken over the MicroDuck's support, and what the 50 Hz loop does with it.

Two inputs, two outputs (Shawn 2026-10-21: a gradual transition, no on/off freeze; the hand leaving gives the balance
back at once):

  grip  head pads covered (L1 R1 L2 R2 M, D-013), weighted by pad area, and how long they have stayed covered:
        engagement e rises toward the coverage with GRIP_TAU (a longer grip counts more) and falls with LETGO_TAU.
  load  the share of the robot's weight still on its feet (1 standing, 0 in the air): the summed Present Current of
        the ten leg servos between the calibrated in-the-air and standing values (walk_wired), or the foot contact
        force over m*g (brain_sim_check). Unknown until calibrated: then only the grip acts.

  s     policy authority, the factor on stand61/walk61's actions (goal = default + s * action): 1 - held, where
        held = max(A_TOUCH * e, h_load) and h_load = (1 - load) * grip gate. The hand alone lowers it to
        1 - A_TOUCH; feet off the floor with the hand on the head lower it to 0, which is the default stance itself,
        the pose wanted at set-down. Once a lift has landed, the hand term is dropped until the hand leaves. It falls at most DOWN_RATE per s and rises at most UP_RATE per s.
  kp    leg P gain from KP_RL down to KP_LEG_MIN with h_load (the legs hang at 0.01 N m, scratch neck_hang.py), back up
        within LOAD_TAU of the load returning (no slow ramp); neck_pitch and head_pitch up to KP_NECK_HOLD with h_load: hanging from the head puts
        0.20 / 0.11 N m on them against 0.026 / 0.024 standing (as-built model, 0.84 kg, 2026-10-21), so the neck
        stiffens at once and softens slowly. head_yaw and head_roll carry almost nothing and stay at KP_RL.

The unload term counts only while the hand is on the head (grip gate): feet unloaded without a hand are a fall, a kick
or a stumble, where the policy needs all its authority. A touch block missing or stale for over STALE_KEEP_S counts as
no hand; a shorter dropout keeps the last pad levels.

Shadow by default: walk_wired always computes and logs the estimate; it changes the goals and gains only while the
hold file (HOLD_FILE, {"on": true, "cal": {...}}) says on and the leg load is calibrated (without the load share the
hand term alone acted in the sim and one of two pivot-hand set-downs fell, 2026-10-22). hold_session.py switches it and writes the calibration.
The file lives in the A7Z home, so the switch and the calibration survive a reboot and a brain restart (Shawn 2026-10-23).
"""
import json
import math
import os

PADS = ("L1", "R1", "L2", "R2", "M")
PAD_AREA_MM2 = {"L1": 324.0, "R1": 324.0, "L2": 144.0, "R2": 144.0, "M": 144.0}  # 18 x 18 and 12 x 12 mm (D-013)
LEG_JOINTS = ("left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
              "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle")
NECK_JOINTS = ("neck_pitch", "head_pitch")
HOLD_FILE = os.path.expanduser("~/.duck_hold.json")

# Defaults until hold_session.py calibrates: a pad counts from PAD_LO counts over its rest baseline and is fully covered
# at PAD_HI (the live page's bar is full at 2000; the per-pad rise through the 1.7 mm wall is not measured yet).
PAD_LO, PAD_HI = 150.0, 1200.0
GRIP_FULL_MM2 = 500.0       # covered area that counts as a full grip (a palm on L1 + R1 is 648)
BASELINE_TAU = 5.0          # s, rest baseline of each pad, tracked only while the pad reads untouched
GRIP_TAU = 0.6              # s, engagement rising toward the coverage: a longer grip counts more
LETGO_TAU = 0.06            # s, engagement falling when the hand leaves
TOUCH_STALE_S = 0.2         # an older block is stale
STALE_KEEP_S = 0.5          # stale this long: no hand; shorter: the last pad levels stand
LOAD_TAU = 0.06             # s, leg current smoothing
GATE_LO, GATE_HI = 0.1, 0.4  # engagement over which unloaded feet count as held (linear in between)
A_TOUCH = 0.6               # the hand alone takes at most this share of the authority, until a lift has landed
AIRBORNE, LANDED = 0.5, 0.8  # h_load that counts as lifted; load share that counts as set down again
LAND_S = 0.1                # s the load share must stay over LANDED, read with the legs paused, to count as set down
S_QUIET = 0.1               # authority at or below which the leg current is the floor's push, not the policy's motion
REGRIP_S = 0.5              # s the pads may read no hand mid-lift (a regrip) before the lift ends
STILL_W, STILL_S = 0.2, 0.2  # |gyro| under STILL_W rad/s for STILL_S: the body rests on something (in the hand 0.25-1.0)
DOWN_RATE = 4.0             # authority falls at most this per s (0.25 s from full to none)
UP_RATE = 10.0              # and comes back at most this per s (0.1 s)
KP_LEG_MIN = 80             # legs in the air
KP_NECK_HOLD = 500          # neck_pitch / head_pitch with the body hanging from the head
KP_SOFTEN_S = 0.3           # s for a full soften; stiffening is immediate
SNAP = 0.02                 # the smoothed grip and load land on their input when this close
KP_STEP = 8                 # walk_wired writes a gain only when it moved this much


def clip01(x):
    return 0.0 if x <= 0 else 1.0 if x >= 1 else x


def valid_cal(cal):
    """The calibration if every field has the right shape, else None (a bad file must never stop the control loop)."""
    if not isinstance(cal, dict):
        return None
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)  # noqa: E731
    for k in ("pad_lo", "pad_hi"):
        if k in cal and not (isinstance(cal[k], list) and len(cal[k]) == len(PADS) and all(num(v) for v in cal[k])):
            return None
    for k in ("grip_full_mm2", "legs_stand_ma", "legs_air_ma", "a_touch"):
        if k in cal and cal[k] is not None and not num(cal[k]):
            return None
    if not num(cal.get("grip_full_mm2", 1.0)) or cal.get("grip_full_mm2", 1.0) <= 0:
        return None
    return cal


def read_hold_file(path=HOLD_FILE):
    """{"on": bool, "cal": dict, "error": str or None}; a missing or broken file, or a malformed calibration, is off with
    no calibration. Only a JSON true switches it on."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {"on": False, "cal": {}, "error": None}
    cal = valid_cal(d.get("cal") or {}) if isinstance(d, dict) else None
    if cal is None:
        return {"on": False, "cal": {}, "error": f"malformed hold file {path}, ignored"}
    return {"on": d.get("on") is True, "cal": dict(cal), "error": None}


def write_hold_file(on, cal, path=HOLD_FILE):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"on": bool(on), "cal": cal}, f)
    os.replace(tmp, path)


class HoldEstimator:
    """update() once per control tick; the state is the newest estimate (attributes below)."""

    def __init__(self, cal=None, kp_rl=200):
        self.kp_rl = kp_rl
        self.base = None          # per-pad rest baseline (counts)
        self.e = 0.0              # grip engagement 0..1
        self.load_f = None        # smoothed load share, None while uncalibrated
        self.s = 1.0              # policy authority
        self.h_load = 0.0         # unloaded share under a grip
        self.h_soft = 0.0         # h_load as the gains follow it (slow down, immediate up)
        self.stale_s = 0.0        # s the touch block has been missing or old
        self.cov = 0.0
        self.levels = [0.0] * len(PADS)
        self.airborne = False     # the feet have been off the ground under this grip
        self.landed = False       # and are back on it: the hand term stays off until the hand leaves
        self.land_s = 0.0         # s of quiet foot load seen so far in this lift
        self.gap_s = 0.0          # s the pads have read no hand during this lift
        self.still_s = 0.0        # s the body gyro has stayed under STILL_W
        self.set_cal(cal or {})

    def set_cal(self, cal):
        """cal: {"pad_lo": [5], "pad_hi": [5], "grip_full_mm2", "legs_stand_ma", "legs_air_ma", "a_touch"}."""
        self.cal = cal
        self.pad_lo = list(cal.get("pad_lo") or [PAD_LO] * len(PADS))
        self.pad_hi = list(cal.get("pad_hi") or [PAD_HI] * len(PADS))
        self.grip_full = float(cal.get("grip_full_mm2", GRIP_FULL_MM2))
        self.a_touch = float(cal.get("a_touch", A_TOUCH))
        st, air = cal.get("legs_stand_ma"), cal.get("legs_air_ma")
        self.legs_range = (float(air), float(st)) if st is not None and air is not None and st > air else None

    def pad_levels(self, counts, age, dt):
        """0..1 per pad from raw counts; updates the rest baseline of untouched pads."""
        if counts is None or age is None or age > TOUCH_STALE_S:
            return [0.0] * len(PADS)
        if self.base is None:
            self.base = [float(c) for c in counts]
        out = []
        for k, c in enumerate(counts):
            d = c - self.base[k]
            if d < -self.pad_lo[k]:
                # a hand only raises the count; a reading this far under the baseline is a bad block. Following it
                # latched R2 on 2026-10-30 02:09:32 (one read 1400 under, the baseline sank, the rest value then read
                # as a full hand for good, so a lift never ended and the set-down duck sagged at kp 80 and fell)
                out.append(0.0)
                continue
            if d < self.pad_lo[k]:  # untouched: follow slow drift (temperature, humidity)
                self.base[k] += (c - self.base[k]) * min(1.0, dt / BASELINE_TAU)
                d = c - self.base[k]
            out.append(clip01((d - self.pad_lo[k]) / max(1.0, self.pad_hi[k] - self.pad_lo[k])))
        return out

    def load_share(self, legs_ma, load=None):
        """Weight share on the feet from the summed leg current (mA), or a ready share (the simulator); None if unknown."""
        if load is None:
            if self.legs_range is None or legs_ma is None:
                return None
            air, st = self.legs_range
            load = (legs_ma - air) / (st - air)
        return clip01(load)

    def update(self, dt, counts=None, age=None, legs_ma=None, load=None, gyro=None):
        """counts: raw pad counts (PADS order) or None; age: s since that block; legs_ma: summed |current| of the ten
        leg servos, or load: a ready 0..1 share; gyro: body angular rate magnitude (rad/s), None if unknown.
        Returns self."""
        self.still_s = self.still_s + dt if gyro is not None and gyro < STILL_W else 0.0
        fresh = counts is not None and age is not None and age <= TOUCH_STALE_S
        self.stale_s = 0.0 if fresh else self.stale_s + dt
        if fresh or self.stale_s > STALE_KEEP_S:
            self.levels = self.pad_levels(counts, age, dt)
        # else: a short read dropout keeps the last levels (an I2C hiccup must not read as the hand leaving: that
        # would end the touchdown latch with the hand still on)
        area = sum(PAD_AREA_MM2[p] * v for p, v in zip(PADS, self.levels))
        self.cov = clip01(area / self.grip_full)
        tau = GRIP_TAU if self.cov > self.e else LETGO_TAU
        self.e += (self.cov - self.e) * min(1.0, dt / tau)
        if abs(self.cov - self.e) < SNAP:  # land exactly, so an untouched head gives exactly full authority
            self.e = self.cov
        raw = self.load_share(legs_ma, load)
        if raw is None:
            self.load_f = None
        else:
            self.load_f = raw if self.load_f is None else self.load_f + (raw - self.load_f) * min(1.0, dt / LOAD_TAU)
            if abs(raw - self.load_f) < SNAP:
                self.load_f = raw
        gate = clip01((self.e - GATE_LO) / (GATE_HI - GATE_LO))
        self.h_load = 0.0 if self.load_f is None else (1.0 - self.load_f) * gate
        # set down with the hand still on the head: the robot must balance on its own feet now, so only the load term
        # counts until the hand leaves (sim 2026-10-22, pivot hand: the hand term kept s at 0.4 after touchdown, the
        # policy's raw action grew to 1.57 rad and the release jumped 1.0-1.3 rad and tilted it 14-33 deg)
        # A lift ends only when the hand lets go (e < GATE_LO) or the feet are set down. A hand shifting on the head
        # in the air is still a hand (2026-10-30 01:33: coverage 0.46 -> 0.17 for 0.15 s shrank the gate, the
        # authority came back, the policy's own leg motion drew 329 mA, read as load 0.80 = "landed", and walk61 at zero
        # command got full authority in the air). Set-down counts only on load read while the legs hold the pause pose
        # (s <= S_QUIET) for LAND_S, never on current the policy's motion makes.
        # A regrip empties the pads too (2026-10-30 02:15:45: R1 -> L2, coverage under GATE_LO for 0.17 s, e under it
        # for 0.5 s as it climbs back with GRIP_TAU; the lift ended, walk61 flailed in the hand for 3 s and the
        # policy's own current read as load 1.0, so the lift never came back). Mid-lift the hand counts as gone only
        # after REGRIP_S, or at once when the body is still: set down and let go, |gyro| 0.02-0.05 rad/s, against
        # 0.25-1.0 in the hand. Leg current at kp 80 cannot tell the floor (110 mA set down, 101-125 in the air).
        hand = self.e >= GATE_LO or self.cov >= GATE_LO
        mid_lift = self.airborne and not self.landed
        self.gap_s = self.gap_s + dt if mid_lift and not hand else 0.0
        if not hand and not (mid_lift and self.gap_s < REGRIP_S and self.still_s < STILL_S):
            self.airborne = self.landed = False
            self.land_s = self.gap_s = 0.0
        elif mid_lift:
            quiet_load = self.load_f is not None and self.load_f > LANDED and self.s <= S_QUIET
            self.land_s = self.land_s + dt if quiet_load else 0.0
            self.landed = self.land_s >= LAND_S
        elif self.h_load > AIRBORNE:
            self.airborne = True
        lifted = self.airborne and not self.landed
        if lifted:
            self.h_load = 1.0  # the legs stay soft and the neck stiff for the whole lift
        held = 1.0 if lifted else self.h_load if self.landed else max(self.a_touch * self.e, self.h_load)
        target = 1.0 - held
        if target < self.s:
            self.s = max(target, self.s - DOWN_RATE * dt)
        else:
            self.s = min(target, self.s + UP_RATE * dt)
        if self.h_load >= self.h_soft:  # softening the legs goes slowly
            self.h_soft = min(self.h_load, self.h_soft + dt / KP_SOFTEN_S)
        else:                            # load back on the feet: full gain at once
            self.h_soft = self.h_load
        return self

    def gains(self):
        """{joint: P gain}: legs soften with the slow h_soft; the neck stiffens with the immediate h_load and
        relaxes with h_soft, so it never lags a lift."""
        leg = int(round(self.kp_rl - (self.kp_rl - KP_LEG_MIN) * self.h_soft))
        neck = int(round(self.kp_rl + (KP_NECK_HOLD - self.kp_rl) * max(self.h_load, self.h_soft)))
        out = {n: leg for n in LEG_JOINTS}
        out.update({n: neck for n in NECK_JOINTS})
        out.update({"head_yaw": self.kp_rl, "head_roll": self.kp_rl})
        return out

    def record(self):
        """The per-tick log entry."""
        r = lambda x: None if x is None else round(x, 3)  # noqa: E731
        return {"lv": [r(v) for v in self.levels], "cov": r(self.cov), "e": r(self.e), "load": r(self.load_f),
                "h_load": r(self.h_load), "s": r(self.s), "kp_leg": self.gains()["left_knee"],
                "kp_neck": self.gains()["neck_pitch"], "landed": self.landed, "gap": r(self.gap_s),
                "still": r(self.still_s)}


def legs_current_ma(cur, names):
    """Sum of |Present Current| over the ten leg servos; None when one of them has not answered."""
    vals = [cur[k] for k, n in enumerate(names) if n in LEG_JOINTS]
    return None if any(v is None for v in vals) else float(sum(abs(v) for v in vals))


def tilt_deg(gb):
    return math.degrees(math.acos(max(-1.0, min(1.0, -gb[2]))))
