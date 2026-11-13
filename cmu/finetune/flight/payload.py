"""Back-load randomisation for the universal stand/walk policy (extensions/flight-20261110/train/UNIVERSAL-PLAN.md).

The robot is compiled WITHOUT the pack (--manifest flightpack-none.json). Each env then carries a payload on trunk_base
built from the real pack's parts (manifest diff nominal minus none: 74 parts, all on trunk_base), split into the two
flight batteries and the rest, so every sampled load has the mass, COM and inertia the real pack gives in that state:

  full pack                 rest x 1, batteries x 1      (weight 0.40: the pack stays fitted, FP-R4)
  batteries out (charging)  rest x 1, batteries x 0      (weight 0.25)
  no pack                   0                            (weight 0.10)
  any load 0 to ~1.1 kg     whole pack x s, s ~ U(0, smax) (weight 0.25)

The state is drawn once per env (fixed for the run, like legged_gym's startup base-mass randomisation). The function
replaces the `randomize_com` reset event on trunk_base and folds in the two trunk_base randomisations it would
otherwise fight with: the startup ±5 % mass/inertia scale (randomize_mass_inertia, removed by the delta) and the
per-reset COM offset (randomize_com, whose `ranges` the com_range curriculum still sets). The COM offset is applied to
the loaded trunk (base + payload), as the round-1 pack runs did on the pack-compiled trunk.
Everything is written through pseudo-inertia matrices (linear in the mass distribution), so it is exact and does not
accumulate across resets: each reset rebuilds from the compile-time defaults.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

BATTERY_IDS = ("battery-L", "battery-R")  # 2 x 151 g, the mass report's "battery" group (302 g)
STATES = ("full", "nobatt", "none", "scaled")


def _quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def part_inertia(p) -> np.ndarray:
    """inertia about the part COM, the same rule as finetune_worker.apply_asbuilt_masses (box from dimensions_m)"""
    if "inertia_about_com_kg_m2" in p:
        return np.asarray(p["inertia_about_com_kg_m2"], float)
    m = float(p["mass_kg"])
    size = np.asarray(p.get("dimensions_m", [.02, .02, .02]), float)
    Q = _quat_mat(p.get("rotation_wxyz", [1, 0, 0, 0]))
    return Q @ np.diag(m / 12 * (np.sum(size ** 2) - size ** 2)) @ Q.T


def part_J(parts) -> np.ndarray:
    """4x4 pseudo-inertia (body frame, about the body origin) of manifest parts: [[Sigma, h], [h^T, m]]"""
    J = np.zeros((4, 4))
    for p in parts:
        m = float(p["mass_kg"])
        c = np.asarray(p["position_m"], float)
        Ic = part_inertia(p)
        Io = Ic + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        J[:3, :3] += 0.5 * np.trace(Io) * np.eye(3) - Io
        J[:3, 3] += m * c
        J[3, :3] += m * c
        J[3, 3] += m
    return J


def pack_parts(nominal_manifest: str, none_manifest: str):
    n = json.loads(Path(nominal_manifest).read_text())["parts"]
    zi = {p["id"] for p in json.loads(Path(none_manifest).read_text())["parts"]}
    pack = [p for p in n if p["id"] not in zi]
    assert pack and all(p["body"] == "trunk_base" for p in pack), "payload model expects every pack part on trunk_base"
    batt = [p for p in pack if p["id"] in BATTERY_IDS]
    assert len(batt) == len(BATTERY_IDS), f"battery parts {BATTERY_IDS} not found in the pack manifest"
    rest = [p for p in pack if p["id"] not in BATTERY_IDS]
    return rest, batt


def _sample_states(n, weights, smax, gen, device):
    w = torch.tensor([weights[s] for s in STATES], dtype=torch.float32)
    kind = torch.multinomial(w / w.sum(), n, replacement=True, generator=gen)
    s_rest = torch.zeros(n)
    s_batt = torch.zeros(n)
    s_rest[kind == 0], s_batt[kind == 0] = 1.0, 1.0
    s_rest[kind == 1] = 1.0
    sc = torch.rand(n, generator=gen) * smax
    s_rest[kind == 3], s_batt[kind == 3] = sc[kind == 3], sc[kind == 3]
    return kind.to(device), s_rest.to(device), s_batt.to(device)


def trunk_payload(env, env_ids, ranges, asset_cfg, payload: dict):
    """reset event on trunk_base: default x alpha-scale (startup draw) + payload (startup draw), COM offset per reset"""
    from mjlab.envs.mdp.dr.body import _decompose_pseudo_inertia_J, _get_entity_indices, _reconstruct_pseudo_inertia_J
    dev = env.device
    asset = env.scene[asset_cfg.name]
    idx = _get_entity_indices(asset.indexing, asset_cfg, "body", False)
    assert len(idx) == 1, "trunk_payload acts on trunk_base only"
    st = getattr(env, "_payload_state", None)
    if st is None:  # first call (the startup reset): draw the per-env scale and payload once
        gen = torch.Generator().manual_seed(int(payload.get("seed", 0)))
        n = env.num_envs
        kind, s_rest, s_batt = _sample_states(n, payload["weights"], payload["smax"], gen, dev)
        lo, hi = payload["mass_scale"]
        alpha = (torch.log(torch.tensor(lo)) / 2 + torch.rand(n, generator=gen) * (torch.log(torch.tensor(hi)) - torch.log(torch.tensor(lo))) / 2).to(dev)
        J_rest = torch.tensor(payload["J_rest"], dtype=torch.float32, device=dev)
        J_batt = torch.tensor(payload["J_batt"], dtype=torch.float32, device=dev)
        st = env._payload_state = dict(kind=kind, s_rest=s_rest, s_batt=s_batt, alpha=alpha, J_rest=J_rest, J_batt=J_batt)
        kg = s_rest * J_rest[3, 3] + s_batt * J_batt[3, 3]
        counts = {s: int((kind == i).sum()) for i, s in enumerate(STATES)}
        print(f"[payload] {n} envs {counts}; payload kg min {kg.min():.3f} mean {kg.mean():.3f} max {kg.max():.3f}; "
              f"first 8: {[round(float(x), 3) for x in kg[:8]]}", flush=True)
        env._payload_obs = torch.zeros(n, 4, device=dev)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev)
    env_ids = env_ids.to(dev).long()
    k = len(env_ids)
    dm = env.sim.get_default_field("body_mass")[idx]
    dp = env.sim.get_default_field("body_ipos")[idx]
    di = env.sim.get_default_field("body_inertia")[idx]
    dq = env.sim.get_default_field("body_iquat")[idx]
    J0 = _reconstruct_pseudo_inertia_J(dm, dp, di, dq)[0]  # (4, 4) bare trunk_base
    a = st["alpha"][env_ids]
    J = J0.unsqueeze(0) * torch.exp(2 * a).view(k, 1, 1)  # mass and inertia x e^(2 alpha), COM unchanged
    J = J + st["s_rest"][env_ids].view(k, 1, 1) * st["J_rest"] + st["s_batt"][env_ids].view(k, 1, 1) * st["J_batt"]
    m, c, inert, q = _decompose_pseudo_inertia_J(J)
    lo, hi = ranges
    delta = torch.rand(k, 3, device=dev) * (hi - lo) + lo  # COM offset of the loaded trunk (the curriculum's range)
    cx = payload.get("com_x_whole_m")
    if cx:  # NEXT-PLAN.md universal3: x offset set in whole-robot terms, U(-cx, cx), at every load
        m_rest = env.sim.get_default_field("body_mass").sum() - dm.sum()  # every body but trunk_base
        whole = (m.view(k) + m_rest) / m.view(k)
        delta[:, 0] = (torch.rand(k, device=dev) * 2 - 1) * float(cx) * whole
    c = c + delta
    J2 = _reconstruct_pseudo_inertia_J(m, c, inert, q)  # same inertia about the moved COM
    m, c, inert, q = _decompose_pseudo_inertia_J(J2)
    e = env_ids.view(k, 1)
    b = torch.as_tensor(idx, device=dev).view(1, 1)
    env.sim.model.body_mass[e, b] = m.view(k, 1)
    env.sim.model.body_ipos[e, b] = c.view(k, 1, 3)
    env.sim.model.body_inertia[e, b] = inert.view(k, 1, 3)
    env.sim.model.body_iquat[e, b] = q.view(k, 1, 4)
    pay = st["s_rest"][env_ids] * st["J_rest"][3, 3] + st["s_batt"][env_ids] * st["J_batt"][3, 3]
    env._payload_obs[env_ids] = torch.cat([pay.view(k, 1), c], dim=1)


try:  # same per-world field expansion and set_const recompute as mjlab's dr.pseudo_inertia
    from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
    trunk_payload = requires_model_fields("body_mass", "body_ipos", "body_inertia", "body_iquat",
                                          recompute=RecomputeLevel.set_const)(trunk_payload)
except ImportError:  # manifest tools without mjlab
    pass


def payload_obs(env):
    """privileged critic observation: payload kg and the loaded trunk_base COM (body frame, m)"""
    po = getattr(env, "_payload_obs", None)
    return po if po is not None else torch.zeros(env.num_envs, 4, device=env.device)


def payload_cfg(nominal_manifest, none_manifest, weights=None, smax=1.15, mass_scale=(0.95, 1.05), seed=0, com_x_whole_m=None):
    rest, batt = pack_parts(nominal_manifest, none_manifest)
    return dict(J_rest=part_J(rest).tolist(), J_batt=part_J(batt).tolist(),
                weights=weights or dict(full=0.40, nobatt=0.25, none=0.10, scaled=0.25), smax=smax,
                mass_scale=mass_scale, seed=seed, com_x_whole_m=com_x_whole_m,
                rest_kg=sum(p["mass_kg"] for p in rest), batt_kg=sum(p["mass_kg"] for p in batt))
