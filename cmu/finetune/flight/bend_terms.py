"""Bend reward candidates for the with-pack groundpick fine-tune (extensions/flight-20261110/train/BEND-PLAN.md section 5).

The bend audit (bend_audit.py, 2026-11-11) found that Pollen's ground-pick reward pays a robot that never bends more
than groundpick61's bend (stand / intended 1.06 without the pack, 1.21 with it) and one that stays down about as much
(0.90 / 1.07). Two windowed costs separate them, both costs (>= 0, used with a negative weight), in metres:
  mouth_high_at_hold  phase in [lo, hi) (the low hold, PICKUP.md's phase 0.33-0.425): mouth tip height above z_ok
  mouth_low_at_rest   phase in [lo, hi) (the standing rest after the rise, 0.80-1.0): mouth tip height below z_up
A standing robot pays the first, a robot that stays down pays the second, the intended bend pays neither.
"""
from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _phase(env, command_name: str = "twist"):
    return env.command_manager.get_term(command_name)._gp_phase


def _mouth_z(env, asset_cfg):
    asset = env.scene[asset_cfg.name]
    return asset.data.site_pos_w[:, asset_cfg.site_ids[0], 2]


def mouth_high_at_hold(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
                       lo: float = 0.33, hi: float = 0.45, z_ok: float = 0.03, command_name: str = "twist") -> torch.Tensor:
    ph = _phase(env, command_name)
    gate = ((ph >= lo) & (ph < hi)).float()
    return gate * torch.clamp(_mouth_z(env, asset_cfg) - z_ok, min=0.0)


def mouth_low_at_rest(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", site_names=["mouth_tip"]),
                      lo: float = 0.85, hi: float = 1.0, z_up: float = 0.18, command_name: str = "twist") -> torch.Tensor:
    ph = _phase(env, command_name)
    gate = ((ph >= lo) & (ph < hi)).float()
    return gate * torch.clamp(z_up - _mouth_z(env, asset_cfg), min=0.0)


def held_phase_command_class():
    """GroundPickPhaseCommand that, in a fraction of envs, stops the phase at PICKUP.md's crouch phase for hold_s once
    per cycle, then lets it run on: the runtime command (walk_brain groundpick: descend to phase 0.33 at 1/4 per s,
    hold, then rise to 1.0). Pollen's env never holds (0.2 s at the bottom), so without this the fine-tune never sees
    the state the acceptance tests. Parameters in HOLD (set by flight_worker.apply_delta from delta["phase_hold"]);
    the held envs are drawn again at every reset and every cycle wrap."""
    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommand

    class HeldPhaseCommand(GroundPickPhaseCommand):
        HOLD = dict(phase=0.33, hold_s=4.0, frac=0.5)

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            n = self.num_envs
            self._hold_left = torch.zeros(n, device=self.device)
            self._held_cycle = torch.zeros(n, dtype=torch.bool, device=self.device)
            self._hold_env = torch.rand(n, device=self.device) < self.HOLD["frac"]

        def compute(self, dt: float) -> None:
            p = float(self.HOLD["phase"])
            holding = self._hold_left > 0
            prev = self._gp_phase
            new = (prev + dt / self._period) % 1.0
            wrapped = new < prev
            cross = self._hold_env & ~self._held_cycle & ~holding & (prev < p) & ((new >= p) | wrapped)
            new = torch.where(cross | holding, torch.full_like(new, p), new)
            self._hold_left = torch.where(cross, torch.full_like(new, float(self.HOLD["hold_s"])),
                                          torch.clamp(self._hold_left - dt * holding.float(), min=0.0))
            self._held_cycle |= cross
            w = wrapped & ~cross & ~holding
            if bool(w.any()):
                self._held_cycle[w] = False
                self._hold_env[w] = torch.rand(int(w.sum()), device=self.device) < self.HOLD["frac"]
            self._gp_phase = new
            self.vel_command_b[:, 0] = torch.cos(2 * torch.pi * new)
            self.vel_command_b[:, 1] = torch.sin(2 * torch.pi * new)
            self.vel_command_b[:, 2] = 0.0

        def reset(self, env_ids):
            out = super().reset(env_ids)
            if env_ids is not None and len(env_ids) > 0:
                self._hold_left[env_ids] = 0.0
                self._held_cycle[env_ids] = self._gp_phase[env_ids] >= float(self.HOLD["phase"])
                self._hold_env[env_ids] = torch.rand(len(env_ids), device=self.device) < self.HOLD["frac"]
            return out

    return HeldPhaseCommand


def held_phase_cfg(old_cfg, hold: dict):
    """A copy of the GroundPickPhaseCommandCfg whose build() makes the held command (mjlab builds command terms with
    cfg.build(env), and GroundPickPhaseCommandCfg.build names its class directly, so class_type alone does nothing)."""
    from dataclasses import dataclass
    from mjlab_microduck.tasks.mdp import GroundPickPhaseCommandCfg

    cls = held_phase_command_class()
    cls.HOLD = dict(cls.HOLD, **hold)

    @dataclass(kw_only=True)
    class HeldPhaseCommandCfg(GroundPickPhaseCommandCfg):
        def build(self, env):
            return cls(self, env)

    return HeldPhaseCommandCfg(**{**vars(old_cfg), "class_type": cls}), cls.HOLD
