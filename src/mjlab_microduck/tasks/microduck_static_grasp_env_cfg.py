"""Microduck STATIC grasp: the ground-grasp move, pulled into a statically stable crouch whose open beak hovers 1-2 mm
above the floor (Rémi, 2026-10-26: "starting from the normal grasp position, go a bit lower ... up to one millimetre
from the floor ... we don't want to touch it").

The pose comes from the static study (scripts/static_grasp/sg_search.py, notes/static-grasp): both lips within 2 mm of
the floor, the centre of mass 1.37 cm inside the feet (standing: 1.67 cm), 0.19 N.m worst holding torque, every joint
in its range; it holds for 20 s with constant targets and no policy. What a fixed-target script cannot do is GET there
and back: at the robot's servo gain the half-crouch on the way buckles (sg_stiffness.py). So the policy brings the
duck down and up, and holds the pose in between.

Built on the ground-grasp env (same 61-D obs / 14 actions, scripted mouth, object, DR, timeline: robotd's A-button
pick, 4 s period, hand-over at phase 0.70), with:
  - the mouth opening of the static pose; the object where that pose puts the pads (8.7 cm in front of the feet);
  - in the hold window (phase 0.22-0.36, the mouth closes at 0.30-0.36): a Gaussian pull toward the static pose,
    the two lips toward 1.5 mm, stillness, the centre of mass over the feet;
  - any beak or head floor contact costs, from the first newton (the earlier "new grasp" leaned 2-3 N on its beak);
  - the 90° bisector preference off (the static pose's open beak is at 78°).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, TerminationTermCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_ground_grasp_env_cfg import make_microduck_ground_grasp_env_cfg

# notes/static-grasp/poses/grasp_sym_1.json (servo order: left leg 5, neck, head pitch, yaw, roll, right leg 5)
STATIC_POSE = [0.1062, -0.0185, -1.5508, 0.8395, 1.5327, -1.5461, -1.2343, 0.0, 0.0,
               -0.1062, 0.0185, 1.5508, -0.8395, -1.5327]
STATIC_JAW_RAD = 0.386           # 22°: the jaw opening of that pose
OBJECT_NOMINAL_XY = (0.087, 0.0)  # the pads' midpoint in that pose, from the standing root (= the foot sites' midpoint)
HOLD_START, HOLD_END = 0.22, 0.36
LIP_TARGET = 0.0015

STATIC_POSE_WEIGHT = 4.0
LIP_HEIGHT_WEIGHT = 4.0
COM_SUPPORT_WEIGHT = 2.0
HOLD_STILL_WEIGHT = 0.05
BEAK_TOUCH_WEIGHT = -2.0          # per step with the head or jaw on the floor (> 0.05 N)
BEAK_FORCE_WEIGHT = -1.5          # per N, from 0 N (the ground-grasp env starts at 1 N for the head, 3 N for the jaw)


# Strict variant (run 2): run 1 (sg-r1-frompick-1026) grasped 6/6 by iteration 500 by LEANING on the beak (touch
# 0.16-0.32 s, up to 15 N, CoM up to 1.9 cm in front of the toes): the touch costs were small next to grasp_held (15).
STRICT = dict(beak_touch_weight=-10.0, beak_touch_threshold=0.2, beak_force_weight=-3.0, com_support_weight=6.0,
              com_outside_floor=-3.0, static_pose_weight=6.0, static_pose_std=0.5, lip_wide_weight=2.0,
              lip_wide_std=0.02, hold_start=0.20)


# Retimed (run 3): run 2 (sg-r2-strict-1026, cancelled at it ~550) kept the CoM inside the feet but never hovered:
# the inherited approach target hovers 6 cm above the object until phase 0.20 and reaches it at 0.29, one tick
# before the bite, so the beak arrived with a 6 N impact and rested on the floor through the bite. Here the beak
# arrives by 0.24, holds still and low until 0.34, bites over 0.34-0.40, and the duck is up by 0.66 (robotd hands
# back to the stand policy at 0.70). The beak timeline is played by the client (duck-static-grasp-test), not robotd.
TIMELINE = dict(hover_at=0.16, down_at=0.24, hold_start=0.24, close_start=0.34, close_end=0.40, held_from=0.44,
                return_start=0.44, return_end=0.66, still_weight=0.2)


def _retime(cfg, t):
    m = cfg.actions["mouth"]
    m.close_start, m.close_end = t["close_start"], t["close_end"]
    r = cfg.rewards
    r["grasp_held"].params["phase_start"] = t["held_from"]
    r["grasp_lift"].params["phase_start"] = t["close_start"] + 0.03
    r["approach_track"].params.update(hover_at=t["hover_at"], down_at=t["down_at"], phase_end=t["close_end"])
    r["object_pushed"].params["phase_end"] = t["close_end"]
    r["approach_lateral"].params["phase_end"] = t["close_start"]
    for name in ("ground_pick_return_pose_legs", "ground_pick_return_pose_neck", "return_upright"):
        r[name].params["hold_end"] = t["return_start"]
        r[name].params["rise_end"] = t["return_end"]
    r["neck_vel_descent"].params["hold_end"] = t["close_start"]


# Hover (run 4, resumed from run 3): run 3 · 500 grasped 6/6 with the CoM inside the feet, but yawed the head 13° and
# rested a corner of the beak on the floor at ~1 N through the hold. Keep the head level, and END the episode when the
# beak presses the floor above 0.5 N during the hold (light contact during the bite stays priced, not terminated).
HOVER = dict(head_level_weight=40.0, contact_end_threshold=0.5)


def make_microduck_static_grasp_env_cfg(play: bool = False, strict: bool = False, retimed: bool = False,
                                        hover: bool = False) -> ManagerBasedRlEnvCfg:
    retimed = retimed or hover
    cfg = make_microduck_ground_grasp_env_cfg(play=play)
    k = STRICT if (strict or retimed) else {}
    t = TIMELINE if retimed else None
    hold_start = t["hold_start"] if t else k.get("hold_start", HOLD_START)
    hold_end = t["close_end"] if t else HOLD_END            # pose and balance through the bite
    lip_end = t["close_start"] if t else 0.31               # lips and stillness until the bite starts
    still_end = t["close_start"] if t else 0.30

    cfg.actions["mouth"].open_rad = STATIC_JAW_RAD
    cfg.events["reset_object"].params["nominal_xy"] = OBJECT_NOMINAL_XY
    cfg.rewards.pop("beak_bisector", None)

    for name in ("head_floor_impact", "jaw_floor_impact"):
        cfg.rewards[name].weight = k.get("beak_force_weight", BEAK_FORCE_WEIGHT)
        cfg.rewards[name].params["threshold"] = 0.0
    cfg.rewards["beak_floor_touch"] = RewardTermCfg(
        func=microduck_mdp.beak_floor_touch, weight=k.get("beak_touch_weight", BEAK_TOUCH_WEIGHT),
        params={"sensor_names": ("head_floor_contact", "jaw_floor_contact"),
                "threshold": k.get("beak_touch_threshold", 0.05)},
    )
    cfg.rewards["static_pose"] = RewardTermCfg(
        func=microduck_mdp.static_pose_track, weight=k.get("static_pose_weight", STATIC_POSE_WEIGHT),
        params={"target": STATIC_POSE, "command_name": "twist", "std": k.get("static_pose_std", 0.25),
                "phase_start": hold_start, "phase_end": hold_end},
    )
    cfg.rewards["lip_height"] = RewardTermCfg(
        func=microduck_mdp.lip_height_track, weight=LIP_HEIGHT_WEIGHT,
        params={"command_name": "twist", "target": LIP_TARGET, "std": 0.004,
                "phase_start": hold_start + 0.01 if t else HOLD_START + 0.02, "phase_end": lip_end},
    )
    cfg.rewards["hold_still"] = RewardTermCfg(
        func=microduck_mdp.hold_stillness, weight=t["still_weight"] if t else HOLD_STILL_WEIGHT,
        params={"command_name": "twist", "phase_start": hold_start + 0.01 if t else HOLD_START + 0.02,
                "phase_end": still_end},
    )
    cfg.rewards["com_support"] = RewardTermCfg(
        func=microduck_mdp.com_support_margin, weight=k.get("com_support_weight", COM_SUPPORT_WEIGHT),
        params={"command_name": "twist", "phase_start": hold_start, "phase_end": hold_end,
                "outside_floor": k.get("com_outside_floor", -1.0)},
    )
    if strict or retimed:
        cfg.rewards["lip_height_wide"] = RewardTermCfg(
            func=microduck_mdp.lip_height_track, weight=k["lip_wide_weight"],
            params={"command_name": "twist", "target": LIP_TARGET, "std": k["lip_wide_std"],
                    "phase_start": hold_start, "phase_end": lip_end},
        )
    if t:
        _retime(cfg, t)
    if hover:
        cfg.rewards["head_level"] = RewardTermCfg(
            func=microduck_mdp.head_level_penalty, weight=HOVER["head_level_weight"],
            params={"command_name": "twist", "phase_start": hold_start, "phase_end": t["close_end"]},
        )
        cfg.terminations["beak_on_floor"] = TerminationTermCfg(
            func=microduck_mdp.beak_floor_contact_in_window, time_out=False,
            params={"command_name": "twist", "phase_start": hold_start - 0.02, "phase_end": t["close_start"],
                    "threshold": HOVER["contact_end_threshold"]},
        )
    return cfg
