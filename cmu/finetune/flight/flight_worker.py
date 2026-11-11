"""Flight-pack fine-tune: cmu/finetune/modal/finetune_worker.py with three registered pack deltas.

The flight pack (extensions/flight-20261110, 936 g, four EDFs + two batteries + wing plate) doubles the duck's mass.
This file adds delta keys to the unchanged worker and then runs its main(), so every flag, guard, cap, checkpoint
and export path is the worker's own. Plan and reasons: microduck-sim/extensions/flight-20261110/train/TRAINING-PLAN.md.

New delta keys (applied after the worker's own apply_delta):
  trunk_mass_scale (lo, hi)  replaces the trunk_base +-5 % pseudo-inertia DR (startup, per env): mass and inertia of
                             trunk_base scale together; with the pack manifest's trunk_base mass this spans the
                             whole robot 1.70-2.10 kg (TRAINING-PLAN.md, "Randomisation")
  trunk_inertia_d (lo, hi)   pseudo-inertia axis stretch d1=d2=d3 range on trunk_base (second moments x e^(2d))
  standing_frac f            every standing_envs curriculum stage set to f (1.0 = stand-only task for stand61)
  drop_z (lo, hi)            reset_base root height range (m); the walk env's own is (0.12, 0.13): a higher top
                             drops the robot a few cm onto its feet at episode start (landing hand-over)
Run it exactly like the worker:  python cmu/finetune/flight/flight_worker.py --delta packwalk ...
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal"))
import finetune_worker as fw  # noqa: E402

# trunk_base mass in the nominal pack model: 1.291 kg of the robot's 1.899 kg (measured 2026-11-10 by
# extensions/flight-20261110/train/check_pack_model.py on the training model); 1.70-2.10 kg total -> trunk x 0.846-1.156
PACK_TRUNK_KG = 1.291
PACK_TOTAL_KG = 1.899
_lo = (1.70 - (PACK_TOTAL_KG - PACK_TRUNK_KG)) / PACK_TRUNK_KG
_hi = (2.10 - (PACK_TOTAL_KG - PACK_TRUNK_KG)) / PACK_TRUNK_KG
PACK_DR = dict(trunk_mass_scale=(round(_lo, 4), round(_hi, 4)), trunk_inertia_d=(-0.05, 0.05), drop_z=(0.12, 0.15))
# walk: ft3's EMA velocity tracking (without it a walk61 fine-tune drifts to standing, ft1/ft2) + the pack DR;
# everything else stays Pollen's recipe as walk61 was trained (no DELTA_STAGED actuator/noise widening)
fw.DELTAS["packwalk"] = dict(track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)), **PACK_DR)
# stand: stand61's job in the runtime is zero command until the first step: all envs standing, same pack DR
fw.DELTAS["packstand"] = dict(track_lin_vel=dict(tau_s=0.5, std=math.sqrt(0.05)), standing_frac=1.0, **PACK_DR)
# scratch: the original recipe from random weights on the pack model (no init policy, curricula from step 0)
fw.DELTAS["packscratch"] = dict(PACK_DR)

# round 2 (RUNLOG 21:37-22:00): the round-1 walkers hold the pack but walk with leg torque at the 0.538 N m clamp
# (p99 at the limit, 13-40 % of steps >= 95 % of it). torque_margin adds a hinge cost on every actuator's |force|
# above `limit`; the weight is set from the degenerate audit (flight_audit.py --jobs packwalk_tq), not guessed.
TQ_LIMIT = 0.45  # N m: 84 % of the 0.538 N m clamp
fw.DELTAS["packwalk_tq"] = dict(fw.DELTAS["packwalk"], torque_margin=dict(limit=TQ_LIMIT, weight=-1.0))


def actuator_force_over(env, limit: float):
    """sum over actuators of max(0, |actuator force| - limit), N m (a cost: >= 0, used with a negative weight)"""
    import torch
    f = env.scene["robot"].data.actuator_force
    return torch.sum(torch.clamp(f.abs() - limit, min=0.0), dim=1)


_orig_apply = fw.apply_delta


def apply_delta(env_cfg, delta: dict):
    env_cfg = _orig_apply(env_cfg, delta)
    ev = env_cfg.events
    if "trunk_mass_scale" in delta:
        lo, hi = delta["trunk_mass_scale"]
        p = ev["randomize_mass_inertia"].params
        p["alpha_range"] = (math.log(lo) / 2.0, math.log(hi) / 2.0)
        if "trunk_inertia_d" in delta:
            p["d_range"] = tuple(delta["trunk_inertia_d"])
    if "standing_frac" in delta:
        for s in env_cfg.curriculum["standing_envs"].params["standing_stages"]:
            s["rel_standing_envs"] = float(delta["standing_frac"])
        # also the command term's own start value: commands drawn before the curriculum's first update keep it
        # until their next resample (the Kaggle smoke fk-smoke-b showed moving envs in the stand job without this)
        env_cfg.commands["twist"].rel_standing_envs = float(delta["standing_frac"])
    if "drop_z" in delta:
        ev["reset_base"].params["pose_range"]["z"] = tuple(delta["drop_z"])
    if "torque_margin" in delta:
        from mjlab.managers import RewardTermCfg
        tm = delta["torque_margin"]
        env_cfg.rewards["torque_margin"] = RewardTermCfg(func=actuator_force_over, weight=float(tm["weight"]),
                                                         params={"limit": float(tm["limit"])})
    print(f"[pack] delta applied: { {k: delta[k] for k in delta if k != 'track_lin_vel'} }", flush=True)
    return env_cfg


def _tq_weight_from_argv():
    """--tq-weight W (round 2): the torque_margin weight chosen from the audit; removed before the worker parses"""
    if "--tq-weight" in sys.argv:
        i = sys.argv.index("--tq-weight")
        w = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
        fw.DELTAS["packwalk_tq"]["torque_margin"] = dict(limit=TQ_LIMIT, weight=w)


fw.apply_delta = apply_delta

if __name__ == "__main__":
    # finetune_worker commits the Modal Volume "microduck-rl" from inside save() when `import modal` works; Modal puts
    # its client on the subprocess path, but these runs mount their own Volume (flight_modal.py commits it), so the
    # worker must not find modal: its own try/except then skips the in-worker commit
    sys.modules["modal"] = None
    _tq_weight_from_argv()
    # --scratch: train from random weights (skip the ONNX init and the curricula collapse); handled here so the
    # worker's own flag list stays unchanged
    if "--scratch" in sys.argv:
        sys.argv.remove("--scratch")
        fw.load_walk61_into_actor = lambda actor, onnx_path, init_std: (None, None)
        fw.compare_onnx = lambda a, b, n=256, seed=0: 0.0
        fw.CURRICULA_DONE_STEP = 0
    fw.main()
