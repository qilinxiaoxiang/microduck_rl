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
