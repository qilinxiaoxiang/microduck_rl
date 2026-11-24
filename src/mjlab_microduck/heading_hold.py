"""Heading hold and the press-speed bucket in training: straight walks commanded as the robot's brain commands them.

Why (robotics checkout, microduck-sim/extensions/battery-walk-20261122/SIM-AUDIT-20261123.md top section item 1): on
the robot the brain (guide/firmware/mac/walk_brain.py, Brain.step) feeds the policy a turn input computed from the IMU
heading on every walk, so the policy's own yaw motion comes back to it as a command one tick later. Training never
showed that loop: mjlab's velocity commands are piecewise constant (rel_heading_envs = 0 in the velocity recipe). And
the robot's straight press is vx 0.25 m/s, while mjlab's forward zero-turn bucket sets vx = |vx|.clamp(min=0.3).

Everything here is opt-in (scratch_worker.py; every option at its default = this module is never used, the recipe's
VelocityCommandCommandOnlyCfg stays in place).

The forward zero-turn bucket: an env with is_forward_env, not standing, vx > 0, vy = 0, wz = 0 after mjlab's sampling
and the turn-in-place bucket (mjlab rel_forward_envs 0.2 of the envs at each resample).

Press speed (--press-speed-share Q, --press-speed V, default 0 / 0.25): at each resample a bucket env gets vx = V with
probability Q.

Heading hold (--heading-hold SHARE): at each resample a bucket env becomes a heading-hold env with probability SHARE,
drawn independently of the press-speed draw. Its gain K is drawn in [LO, HI] (--heading-hold-gain, default 3.0 3.0 =
walk_brain.py HEADING_KP, fixed). Two laws (--heading-hold-law):
  p      (default) wz = clip(K * wrap(target - heading), -1, 1), target = the env's heading at the resample.
  brain  walk_brain.py Brain.step for a straight walk (vtheta = 0, moving), each control step with dt = the env step:
           err = wrap(target - heading_used)
           i_new = clip(i + err dt, -I_MAX, I_MAX); u_new = trim + KP err + KI i_new
           if |u_new| < TURN_MAX or |i_new| < |i|: i = i_new          (the brain's anti-windup)
           wz = clip(trim + KI i + KP err, -TURN_MAX, TURN_MAX)
         KP = K, KI = K * 2/3 (brain 3.0 and 2.0), I_MAX 1.5 rad s, TURN_MAX 1.0. At the resample: i = 0, target =
         heading_used, trim uniform in [-T, T] (--heading-hold-trim T, default 0), and heading_used = the heading 0, 1
         or 2 control ticks old (age drawn per env; the brain reads an IMU yaw 0 to 2 ticks old), from a 3-slot history.
         The history runs for every env on every step and is kept across a resample, so the target is the aged
         heading at the resample, not the true one.
         The history and the integral advance only on real env steps (dt > 0), not on the reset-time compute(0).
         After an episode reset the target and history are taken from the heading after mjlab's sim.forward()
         (reset() below), never from the ended episode's.

What still differs from the brain under law brain:
  - The trim is a random constant per command segment; the brain learns its trim across walks (learn_yaw_trim,
    TRIM_RATE 0.5) from the robot's own drift.
  - The sim has no plant yaw bias (the robot's left-foot drift that the integral and trim exist to cancel), so a trim
    here is a command offset the policy must walk against, not a correction of a real bias.
  - The heading source is the sim's true heading (aged 0 to 2 ticks); the robot's is the IMU yaw with its own noise and
    the servo-read drops.
  - Only straight forward walks are held. The brain also holds heading on walking turns (target advancing at vtheta,
    with the clamp anti-windup on the target) and on turns in place.
"""
from __future__ import annotations

from dataclasses import dataclass as _dataclass

import torch
from mjlab.utils.lab_api.math import wrap_to_pi

from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg

BRAIN_HEADING_KP = 3.0  # walk_brain.py HEADING_KP
BRAIN_HEADING_KI = 2.0  # walk_brain.py HEADING_KI
BRAIN_HEADING_I_MAX = 1.5  # walk_brain.py HEADING_I_MAX (rad s)
BRAIN_TURN_MAX = 1.0  # walk_brain.py TURN_MAX = CMD_WZ_MAX (walking turn clamp since 2026-11-22)
BRAIN_IMU_AGE_MAX = 2  # control ticks
LAWS = ("p", "brain")
HIST = BRAIN_IMU_AGE_MAX + 1


class HeadingHoldVelocityCommand(VelocityCommandCommandOnly):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.is_heading_hold_env = torch.zeros(n, dtype=torch.bool, device=dev)
        self.is_press_speed_env = torch.zeros(n, dtype=torch.bool, device=dev)
        self.heading_hold_target = torch.zeros(n, device=dev)
        self.heading_hold_error = torch.zeros(n, device=dev)
        self.heading_hold_gain = torch.full((n,), float(cfg.heading_hold_gain[0]), device=dev)
        # law brain state
        self.heading_hold_i = torch.zeros(n, device=dev)
        self.heading_hold_trim = torch.zeros(n, device=dev)
        self.heading_hold_age = torch.zeros(n, dtype=torch.long, device=dev)
        self.heading_hist = torch.zeros(n, HIST, device=dev)  # [:, 0] newest
        self._anchor_pending = torch.zeros(n, dtype=torch.bool, device=dev)
        self._dt = 0.0

    def reset(self, env_ids):
        # mjlab calls this inside _reset_idx, before sim.forward(): heading_w (xquat) still holds the heading of the
        # episode that just ended, while the reset event has written a new yaw. The target and history taken in
        # _resample_command are therefore stale for these envs; _update_command (which mjlab calls after forward())
        # takes them again from the new heading. A resample in the middle of an episode reads a fresh heading.
        out = super().reset(env_ids)
        self._anchor_pending[env_ids] = True
        return out

    def compute(self, dt: float) -> None:
        self._dt = float(dt)
        super().compute(dt)

    def heading_used(self, ids=None) -> torch.Tensor:
        """The heading the law reads: true heading (law p), or the history slot of the env's age (law brain)."""
        if self.cfg.heading_hold_law == "p":
            h = self.robot.data.heading_w
            return h if ids is None else h[ids]
        hist = self.heading_hist if ids is None else self.heading_hist[ids]
        age = self.heading_hold_age if ids is None else self.heading_hold_age[ids]
        return hist.gather(1, age[:, None]).squeeze(1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        cmd = self.vel_command_b[env_ids]
        bucket = (self.is_forward_env[env_ids] & ~self.is_standing_env[env_ids] & (cmd[:, 0] > 0.0)
                  & (cmd[:, 1] == 0.0) & (cmd[:, 2] == 0.0))
        n = len(env_ids)
        q = float(self.cfg.rel_press_speed_envs)
        if q > 0.0:
            press = bucket & (torch.rand(n, device=self.device) < q)
            self.is_press_speed_env[env_ids] = press
            pid = env_ids[press]
            self.vel_command_b[pid, 0] = float(self.cfg.press_speed)
        share = float(self.cfg.rel_heading_hold_envs)
        if share <= 0.0:
            return
        self.is_heading_hold_env[env_ids] = bucket & (torch.rand(n, device=self.device) < share)
        lo, hi = (float(x) for x in self.cfg.heading_hold_gain)
        if hi > lo:
            self.heading_hold_gain[env_ids] = torch.empty(n, device=self.device).uniform_(lo, hi)
        else:
            self.heading_hold_gain[env_ids] = lo
        self.heading_hold_error[env_ids] = 0.0
        if self.cfg.heading_hold_law == "brain":
            # the history is kept across a resample in the middle of an episode: the target is the heading the brain
            # would read at that moment (the env's age slot), as walk_brain.py latches its held heading from the IMU
            # yaw it has. Only an episode reset refills the history (reset() and _update_command).
            self.heading_hold_age[env_ids] = torch.randint(0, BRAIN_IMU_AGE_MAX + 1, (n,), device=self.device)
            self.heading_hold_i[env_ids] = 0.0
            t = float(self.cfg.heading_hold_trim)
            self.heading_hold_trim[env_ids] = (torch.empty(n, device=self.device).uniform_(-t, t) if t > 0.0
                                               else torch.zeros(n, device=self.device))
        self.heading_hold_target[env_ids] = self.heading_used(env_ids)

    def _update_command(self) -> None:
        super()._update_command()
        if float(self.cfg.rel_heading_hold_envs) <= 0.0:
            return
        brain = self.cfg.heading_hold_law == "brain"
        if brain and self._dt > 0.0:  # a real env step: the IMU history moves one tick
            self.heading_hist = torch.cat([self.robot.data.heading_w[:, None], self.heading_hist[:, :-1]], dim=1)
        if self._anchor_pending.any():
            pend = self._anchor_pending
            h = self.robot.data.heading_w[pend]
            if brain:
                self.heading_hist[pend] = h[:, None].expand(-1, HIST)
            self.heading_hold_target[pend] = h
            self._anchor_pending[pend] = False
        ids = self.is_heading_hold_env.nonzero(as_tuple=False).flatten()
        if len(ids) == 0:
            return
        err = wrap_to_pi(self.heading_hold_target[ids] - self.heading_used(ids))
        self.heading_hold_error[ids] = err
        m = float(self.cfg.heading_hold_max)
        kp = self.heading_hold_gain[ids]
        if not brain:
            self.vel_command_b[ids, 2] = torch.clip(kp * err, -m, m)
            return
        ki = kp * (BRAIN_HEADING_KI / BRAIN_HEADING_KP)
        trim, i = self.heading_hold_trim[ids], self.heading_hold_i[ids]
        if self._dt > 0.0:
            i_new = torch.clip(i + err * self._dt, -BRAIN_HEADING_I_MAX, BRAIN_HEADING_I_MAX)
            u_new = trim + kp * err + ki * i_new
            i = torch.where((u_new.abs() < m) | (i_new.abs() < i.abs()), i_new, i)
            self.heading_hold_i[ids] = i
        self.vel_command_b[ids, 2] = torch.clip(trim + ki * i + kp * err, -m, m)

    def tick_stats(self) -> dict:
        """Per-tick numbers for the worker's metrics line (shares are of all envs)."""
        h = self.is_heading_hold_env
        out = dict(press_share=float(self.is_press_speed_env.float().mean()), hold_share=float(h.float().mean()))
        if h.any():
            wz = self.vel_command_b[h, 2]
            out.update(hold_abs_wz=float(wz.abs().mean()), hold_sd_wz=float(wz.std(unbiased=False)),
                       hold_clamp_share=float((wz.abs() >= float(self.cfg.heading_hold_max) - 1e-6).float().mean()))
        return out


@_dataclass(kw_only=True)
class HeadingHoldVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    rel_heading_hold_envs: float = 0.0  # share of forward zero-turn envs that get heading feedback
    heading_hold_gain: tuple[float, float] = (BRAIN_HEADING_KP, BRAIN_HEADING_KP)  # K drawn per env per resample
    heading_hold_max: float = BRAIN_TURN_MAX
    heading_hold_law: str = "p"
    heading_hold_trim: float = 0.0  # law brain: trim uniform in [-T, T] per resample
    rel_press_speed_envs: float = 0.0  # share of forward zero-turn envs commanded at press_speed
    press_speed: float = 0.25

    def build(self, env) -> HeadingHoldVelocityCommand:
        return HeadingHoldVelocityCommand(self, env)


def apply_heading_hold(cfg, share: float, gain=(BRAIN_HEADING_KP, BRAIN_HEADING_KP), law: str = "p",
                       trim: float = 0.0, press_share: float = 0.0, press_speed: float = 0.25) -> dict | None:
    """Replace cfg.commands['twist'] (a VelocityCommandCommandOnlyCfg) by the command above with the same fields.
    share 0 and press_share 0 change nothing and return None; otherwise returns the record of what was applied."""
    if share <= 0.0 and press_share <= 0.0:
        return None
    assert 0.0 <= share <= 1.0 and 0.0 <= press_share <= 1.0, (share, press_share)
    assert law in LAWS, law
    lo, hi = float(gain[0]), float(gain[1])
    assert 0.0 < lo <= hi, gain
    assert trim >= 0.0 and (trim == 0.0 or law == "brain"), "a trim needs law brain"
    old = cfg.commands["twist"]
    assert type(old) is VelocityCommandCommandOnlyCfg, type(old)
    # the recipe keeps mjlab's heading_command=True with rel_heading_envs 0: the absolute-target law acts on no env
    assert old.rel_heading_envs == 0.0, "mjlab's own heading mode already acts on some envs"
    assert old.rel_forward_envs > 0.0, "no forward zero-turn bucket to act on"
    assert tuple(old.ranges.ang_vel_z) == (-BRAIN_TURN_MAX, BRAIN_TURN_MAX), old.ranges.ang_vel_z
    cfg.commands["twist"] = HeadingHoldVelocityCommandCfg(
        **vars(old), rel_heading_hold_envs=float(share), heading_hold_gain=(lo, hi), heading_hold_max=BRAIN_TURN_MAX,
        heading_hold_law=law, heading_hold_trim=float(trim), rel_press_speed_envs=float(press_share),
        press_speed=float(press_speed))
    rf = float(old.rel_forward_envs)
    rec = dict(forward_bucket=f"is_forward_env, not standing, vx > 0, vy = wz = 0 (rel_forward_envs {rf})",
               press_speed=dict(share_of_bucket=float(press_share), vx=float(press_speed),
                                expected_share_of_all_envs=round(float(press_share) * rf, 4)))
    if share > 0.0:
        rec["heading_hold"] = dict(
            share_of_bucket=float(share), expected_share_of_all_envs=round(float(share) * rf, 4), law=law,
            gain=[lo, hi], clamp=BRAIN_TURN_MAX, target="heading_used at each command resample",
            ki="K * 2/3" if law == "brain" else None, i_max=BRAIN_HEADING_I_MAX if law == "brain" else None,
            trim=[-float(trim), float(trim)] if law == "brain" else None,
            imu_age_ticks=[0, BRAIN_IMU_AGE_MAX] if law == "brain" else [0, 0],
            draws="independent of the press-speed draw")
    return rec
