"""Servo measurement noise levels for the fetch chain (nudge, dock, kneel), applied identically to every task.

Level 0 is the velocity recipe's (joint_pos +-0.001 rad white, joint_vel +-0.25 rad/s, encoder bias +-0.015 rad
constant per env). Level 0's position noise is below the XL330's own resolution (4096 counts / rev = 0.0015 rad),
so it trains no tolerance to encoder jitter at all.

  level 1 (train): joint_pos +-0.004 rad (~ +-2.6 counts of jitter under load and vibration), joint_vel +-0.5 rad/s
                   (the firmware's velocity is differenced counts), encoder bias +-0.026 rad (1.5 deg horn /
                   calibration error)
  level 2 (stress): twice level 1's white noise, bias +-0.04 rad; for checking margins, not for training
  level 3 (worst):  four times level 1's white noise, bias +-0.06 rad (3.4 deg); trained on by the cautious step
                    policies of 2026-11-22 (brain_sim_check.py --servo-noise has the same levels)

apply_servo_noise_range(cfg, lo, hi) draws a level per env at every reset, uniform in [lo, hi] on the level scale (the
white-noise amplitude doubles per level: level t = level 1 x 2^(t-1)), so one run trains from quiet to worst reads.
The bias stays the startup event's, per env, over the top level's range.

All values are uniform and zero-centred, like the recipe's. The dock commander's own head-joint input (neck pitch,
head pitch, head yaw) had no noise and read the true joint position; from level 1 on it reads the biased position
with the same white noise as every other joint reading.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# per-env white-noise scale (fraction of the top level's amplitude) by key; written at reset by draw_noise_level,
# read by EpisodeScaledUniformNoiseCfg. A module registry because the managers deepcopy their term cfgs.
NOISE_SCALE: dict[str, torch.Tensor] = {}

SERVO_NOISE = {
    1: dict(pos=0.004, vel=0.5, bias=0.026),
    2: dict(pos=0.008, vel=1.0, bias=0.04),
    3: dict(pos=0.016, vel=2.0, bias=0.06),
}


def apply_servo_noise(cfg, level: int, vel: bool = True):
    """Set every corrupted obs group's joint readings to `level` (in place; returns cfg). Critic groups (no
    corruption) keep the true state. vel=False leaves joint_vel's noise as it is (robot_conditions.py sets it)."""
    if not level:
        return cfg
    lv = SERVO_NOISE[level]
    for grp in cfg.observations.values():
        if not grp.enable_corruption:
            continue
        for name, amp in (("joint_pos", lv["pos"]), ("joint_vel", lv["vel"]), ("head_joints", lv["pos"])):
            if name == "joint_vel" and not vel:
                continue
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


@dataclass
class EpisodeScaledUniformNoiseCfg(Unoise):
    """Uniform noise in [n_min, n_max] times a per-env scale from NOISE_SCALE[key] (1 until the first reset)."""

    key: str = "servo"

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        n_min = self._get_cached_tensor("n_min", self.n_min, data.device)
        n_max = self._get_cached_tensor("n_max", self.n_max, data.device)
        noise = torch.rand_like(data) * (n_max - n_min) + n_min
        scale = NOISE_SCALE.get(self.key)
        if scale is not None and data.dim() >= 1 and scale.shape[0] == data.shape[0]:
            noise = noise * scale.to(data.device).view(-1, *([1] * (data.dim() - 1)))
        return data + noise


def draw_noise_level(env, env_ids, key: str, lo: float, hi: float):
    """Reset event: a servo-noise level per env, uniform in [lo, hi], stored as its amplitude over level hi's."""
    n = env.num_envs
    if key not in NOISE_SCALE or NOISE_SCALE[key].shape[0] != n:
        NOISE_SCALE[key] = torch.ones(n, device=env.device)
    if env_ids is None:
        env_ids = torch.arange(n, device=env.device)
    t = torch.rand(len(env_ids), device=env.device) * (hi - lo) + lo
    NOISE_SCALE[key][env_ids] = torch.pow(2.0, t - hi)


def apply_servo_noise_range(cfg, lo: float, hi: int, key: str = "servo", vel: bool = True):
    """Like apply_servo_noise(cfg, hi, vel), but each env draws its level in [lo, hi] at every reset."""
    from mjlab.managers import EventTermCfg

    apply_servo_noise(cfg, hi, vel)
    lv = SERVO_NOISE[hi]
    for grp in cfg.observations.values():
        if not grp.enable_corruption:
            continue
        for name, amp in (("joint_pos", lv["pos"]), ("joint_vel", lv["vel"]), ("head_joints", lv["pos"])):
            if name == "joint_vel" and not vel:
                continue
            if name in grp.terms:
                grp.terms[name].noise = EpisodeScaledUniformNoiseCfg(n_min=-amp, n_max=amp, key=key)
    cfg.events["servo_noise_level"] = EventTermCfg(func=draw_noise_level, mode="reset",
                                                   params={"key": key, "lo": float(lo), "hi": float(hi)})
    return cfg
