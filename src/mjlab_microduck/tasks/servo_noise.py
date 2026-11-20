"""Servo measurement noise levels for the fetch chain (nudge, dock, kneel), applied identically to every task.

Level 0 is the velocity recipe's (joint_pos +-0.001 rad white, joint_vel +-0.25 rad/s, encoder bias +-0.015 rad
constant per env). Level 0's position noise is below the XL330's own resolution (4096 counts / rev = 0.0015 rad),
so it trains no tolerance to encoder jitter at all.

  level 1 (train): joint_pos +-0.004 rad (~ +-2.6 counts of jitter under load and vibration), joint_vel +-0.5 rad/s
                   (the firmware's velocity is differenced counts), encoder bias +-0.026 rad (1.5 deg horn /
                   calibration error)
  level 2 (stress): twice level 1's white noise, bias +-0.04 rad; for checking margins, not for training

All values are uniform and zero-centred, like the recipe's. The dock commander's own head-joint input (neck pitch,
head pitch, head yaw) had no noise and read the true joint position; from level 1 on it reads the biased position
with the same white noise as every other joint reading.
"""
from __future__ import annotations

from mjlab.utils.noise import UniformNoiseCfg as Unoise

SERVO_NOISE = {
    1: dict(pos=0.004, vel=0.5, bias=0.026),
    2: dict(pos=0.008, vel=1.0, bias=0.04),
}


def apply_servo_noise(cfg, level: int):
    """Set every corrupted obs group's joint readings to `level` (in place; returns cfg). Critic groups (no
    corruption) keep the true state."""
    if not level:
        return cfg
    lv = SERVO_NOISE[level]
    for grp in cfg.observations.values():
        if not grp.enable_corruption:
            continue
        for name, amp in (("joint_pos", lv["pos"]), ("joint_vel", lv["vel"]), ("head_joints", lv["pos"])):
            term = grp.terms.get(name)
            if term is None:
                continue
            term.noise = Unoise(n_min=-amp, n_max=amp)
            if name == "head_joints":
                term.params = {**term.params, "biased": True}
    ev = cfg.events.get("encoder_bias")
    assert ev is not None, "servo noise needs the encoder_bias startup event"
    ev.params["bias_range"] = (-lv["bias"], lv["bias"])
    return cfg


def noisy_task_id(task_id: str, level: int) -> str:
    """Mjlab-X-MicroDuck-CMU-Plugged -> Mjlab-X-SN1-MicroDuck-CMU-Plugged."""
    head, sep, tail = task_id.partition("-MicroDuck")
    assert sep, task_id
    return f"{head}-SN{level}{sep}{tail}"
