"""CPU tests of the flight-pack options of scratch_worker.py (pack body, thrust event, mass ramp, told input).

Run (no GPU, a few minutes on CPU): python -m pytest cmu/finetune/kaggle/test_pack_thrust.py -q
PACK_TEST_BEFORE_SNAPSHOT=<json> enables test_current_cfg_unchanged_against_snapshot: the canonical dump
(cfg_dump()) of the --model current cfg built by the worker before the pack change; without it that test skips.
"""
from __future__ import annotations

import dataclasses
import enum
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PACK = "flightpack-v1-w918-20261124"
B2 = REPO / "cmu/whole/runtime-overlay/models-bw-s2c/checkpoints/bw-s2c-b2_model_4999.pt"


def _worker():
    spec = importlib.util.spec_from_file_location("scratch_worker_under_test", HERE / "scratch_worker.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


sw = _worker()


def canon(x, depth=0):
    """Canonical JSON-able view of a cfg: dataclass fields, dicts in order, callables by qualified name."""
    if depth > 40:
        return "<deep>"
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        d = {"__type__": type(x).__module__ + "." + type(x).__qualname__}
        for f in dataclasses.fields(x):
            d[f.name] = canon(getattr(x, f.name), depth + 1)
        for k, v in vars(x).items():
            if k not in d and not k.startswith("__"):
                d["+" + k] = canon(v, depth + 1)
        return d
    if isinstance(x, dict):
        return {str(k): canon(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [canon(v, depth + 1) for v in x]
    if isinstance(x, enum.Enum):
        return str(x)
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, type) or callable(x):
        return f"<fn {getattr(x, '__module__', '?')}.{getattr(x, '__qualname__', type(x).__qualname__)}>"
    return f"<{type(x).__qualname__} {x!r}>"[:300]


def cfg_dump():
    return {c: canon(sw.make_env_cfg("current", conditions=c)) for c in ("official", "robot")}


def build_env(cfg, n=4):
    from mjlab.envs import ManagerBasedRlEnv
    cfg.scene.num_envs = n
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()
    return env


def pack_cfg(**kw):
    from mjlab_microduck.tasks.cmu_asbuilt import apply_pack_options
    cfg = sw.make_env_cfg(PACK, conditions=kw.pop("conditions", "robot"))
    rec = apply_pack_options(cfg, PACK, **kw)
    return cfg, rec


def trunk(env):
    a = env.scene["robot"]
    ids, _ = a.find_bodies("trunk_base")
    return a, ids[0], int(a.indexing.body_ids[ids[0]])


def weight(env):
    from mjlab_microduck.tasks.mdp import _robot_weight_n
    return _robot_weight_n(env, env.scene["robot"])


def thrust_tick(env, level):
    t = env.event_manager.get_term_cfg("pack_thrust")
    t.func(env, None, **dict(t.params, hold_level=level))


# ---------------------------------------------------------------- (a) current model unchanged
def test_current_cfg_unchanged_against_snapshot():
    snap = os.environ.get("PACK_TEST_BEFORE_SNAPSHOT")
    if not snap:
        pytest.skip("PACK_TEST_BEFORE_SNAPSHOT not set")
    before = json.loads(Path(snap).read_text())
    after = json.loads(json.dumps(cfg_dump(), sort_keys=True))
    assert after == before


def test_current_cfg_has_no_pack_terms_and_refuses_pack_flags():
    for c in ("official", "robot"):
        cfg = sw.make_env_cfg("current", conditions=c)
        assert not {"pack_thrust", "pack_mass_ramp"} & set(cfg.events)
        assert "pack_thrust_lift" not in cfg.observations["actor"].terms
    with pytest.raises(SystemExit):
        sw.main(["--run-name", "x", "--model", "current", "--max-hours", "1", "--thrust-lift", "0", "0.5"])


# ---------------------------------------------------------------- (b) pack body mass
def test_pack_body_mass_matches_manifest():
    from mjlab_microduck.tasks.cmu_asbuilt import PACK_VARIANTS, robot_body_inertials
    man = json.loads((REPO / "cmu/model/manifests" / f"{PACK}.json").read_text())
    base_g = sum(b["mass"] for b in robot_body_inertials(PACK_VARIANTS[PACK]).values()) * 1000
    want = base_g + float(man["derived"]["net_delta_g"])
    cfg, _ = pack_cfg(conditions="official")
    env = build_env(cfg, n=2)
    got = float(env.sim.mj_model.body_mass.sum()) * 1000
    env.close()
    assert abs(got - want) < 2.0, (got, want, base_g)


# ---------------------------------------------------------------- (c, d, e) thrust event
@pytest.fixture(scope="module")
def thrust_env():
    cfg, rec = pack_cfg(thrust_lift=(0.5, 0.5), zero_share=0.0, drop_share=0.0, imbalance=0.0)
    env = build_env(cfg)
    yield env, rec
    env.close()


def test_lag_63_percent_in_one_tau(thrust_env):
    env, rec = thrust_env
    t = env.event_manager.get_term_cfg("pack_thrust")
    t.func.reset(None)
    tau, dt = rec["thrust"]["tau"], env.step_dt
    frac = []
    for _ in range(12):
        thrust_tick(env, 0.5)
        frac.append(float(env._pack_thrust_lift.mean()) / 0.5)
    k63 = next(i + 1 for i, f in enumerate(frac) if f >= 1 - math.exp(-1) - 1e-6)
    assert abs(k63 * dt - tau) <= dt + 1e-9, (k63, frac)


def test_constant_lift_vertical_force_and_tilt(thrust_env):
    env, _ = thrust_env
    for _ in range(80):
        thrust_tick(env, 0.5)
    a, local, gid = trunk(env)
    pose = a.data.root_link_pose_w.clone()
    pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    a.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    thrust_tick(env, 0.5)
    f = env.sim.data.xfrc_applied[:, gid, 0:3]
    w = weight(env)
    assert torch.allclose(f[:, 2] / w, torch.full_like(w, 0.5), rtol=0.02), (f[:, 2] / w)
    # tilt 25 deg about x then 10 deg about y: the force follows the trunk's z axis
    from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz
    q = quat_from_euler_xyz(torch.full((env.num_envs,), math.radians(25)), torch.full((env.num_envs,), math.radians(10)),
                            torch.zeros(env.num_envs))
    pose[:, 3:7] = q
    a.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    thrust_tick(env, 0.5)
    f = env.sim.data.xfrc_applied[:, gid, 0:3]
    z_t = quat_apply(a.data.body_link_quat_w[:, local], torch.tensor([[0.0, 0.0, 1.0]]).expand(env.num_envs, 3))
    cos = (f * z_t).sum(-1) / f.norm(dim=-1)
    assert (cos > 0.9999).all(), cos
    assert torch.allclose(f.norm(dim=-1) / w, torch.full_like(w, 0.5), rtol=0.02)


def test_zero_share_one_applies_no_force():
    cfg, _ = pack_cfg(thrust_lift=(0.0, 0.75), zero_share=1.0)
    env = build_env(cfg)
    _, _, gid = trunk(env)
    for _ in range(30):
        env.step(torch.zeros(env.num_envs, 14))
    assert float(env.sim.data.xfrc_applied[:, gid].abs().max()) == 0.0
    assert float(env._pack_thrust_lift.abs().max()) == 0.0
    env.close()


def test_drop_share_one_drops_target():
    cfg, _ = pack_cfg(thrust_lift=(0.6, 0.6), zero_share=0.0, drop_share=1.0, imbalance=0.0)
    env = build_env(cfg)
    t = env.event_manager.get_term_cfg("pack_thrust")
    seen = torch.zeros(env.num_envs, dtype=torch.bool)
    for _ in range(int(6.0 / env.step_dt)):  # every hold (2-6 s) drops to zero somewhere inside it
        t.func(env, None, **t.params)
        seen |= t.func._level == 0
    assert seen.all(), seen
    env.close()


# ---------------------------------------------------------------- mass ramp
def _full_inertia(m, i):
    import mujoco
    import numpy as np
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(m["iquat"], dtype=float))
    R = R.reshape(3, 3)
    return R @ np.diag(m["inertia"]) @ R.T


def test_mass_ramp_endpoints():
    import numpy as np
    from mjlab_microduck.tasks.cmu_asbuilt import PACK_VARIANTS, robot_body_inertials
    cfg, rec = pack_cfg(conditions="official", ramp_steps=24 * 100)
    for k in ("randomize_mass_inertia", "randomize_com", "randomize_head_com"):  # compare against the exact bodies
        cfg.events.pop(k, None)
    for k in ("com_range", "head_com_range"):  # curricula that drive the COM randomisation removed above
        cfg.curriculum.pop(k, None)
    env = build_env(cfg, n=2)
    a = env.scene["robot"]
    term = env.event_manager.get_term_cfg("pack_mass_ramp")
    for f, variant in ((0.0, PACK_VARIANTS[PACK]), (1.0, PACK)):
        term.func(env, None, **dict(term.params, fraction=f))
        want = robot_body_inertials(variant)
        m = env.sim.model
        for name in rec["mass_ramp"]["bodies"] + ["trunk_base"]:
            gid = int(a.indexing.body_ids[a.find_bodies(name)[0][0]])
            for e in range(env.num_envs):
                got = dict(mass=float(m.body_mass[e, gid]), ipos=m.body_ipos[e, gid].tolist(),
                           inertia=m.body_inertia[e, gid].tolist(), iquat=m.body_iquat[e, gid].tolist())
                assert abs(got["mass"] - want[name]["mass"]) < 1e-6, (f, name, got["mass"], want[name]["mass"])
                assert np.allclose(got["ipos"], want[name]["ipos"], atol=1e-6), (f, name)
                assert np.allclose(_full_inertia(got, 0), _full_inertia(want[name], 0), atol=1e-8, rtol=1e-4), (f, name)
        tot = float(m.body_mass[0, a.indexing.body_ids].sum()) * 1000
        assert abs(tot - sum(b["mass"] for b in want.values()) * 1000) < 0.01
    # the scheduled fraction counts from start_counter and resets do not accumulate
    term.func.start_counter = env.common_step_counter
    env.reset()
    assert term.func.last_fraction == 0.0
    m0 = env.sim.model.body_mass.clone()
    env.reset()
    assert torch.equal(m0, env.sim.model.body_mass)
    env.close()


# ---------------------------------------------------------------- (f) padded 61 -> 62 load
def test_padded_checkpoint_same_action(tmp_path):
    from tensordict import TensorDict
    if not B2.exists():
        pytest.skip(f"{B2} missing")
    d = torch.load(B2, map_location="cpu", weights_only=False)
    padded = sw.pad_checkpoint_for_told(str(B2), tmp_path / "p.pt")
    assert padded["widths"]["actor_state_dict"] == (61, 62)
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab_microduck.tasks import MicroduckOnPolicyRunner
    cfg, _ = pack_cfg(thrust_lift=(0.0, 0.6), told=True)
    env = build_env(cfg)
    assert env.observation_manager.group_obs_dim["actor"][0] == 62
    agent = sw.make_agent_cfg("t", 1, 1, 1)
    wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = MicroduckOnPolicyRunner(wenv, dataclasses.asdict(agent), None, "cpu")
    runner.load(padded["path"], map_location="cpu")
    pol = runner.get_inference_policy("cpu")
    x61 = torch.randn(16, 61) * 0.5
    sd = d["actor_state_dict"]
    h = (x61 - sd["obs_normalizer._mean"]) / (sd["obs_normalizer._std"] + 1e-2)
    for i in (0, 2, 4):
        h = torch.nn.functional.elu(h @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"])
    ref = h @ sd["mlp.6.weight"].T + sd["mlp.6.bias"]
    with torch.no_grad():
        out = pol(TensorDict({"actor": torch.cat([x61, torch.zeros(16, 1)], -1)}, batch_size=[16]))
    assert torch.allclose(out, ref, atol=1e-5), (out - ref).abs().max()
    # lift estimate is 0 while the delivered lift is 0 and lags otherwise
    for _ in range(10):
        env.step(torch.zeros(env.num_envs, 14))
        est, lift = env._pack_thrust_lift_est[:, 0], env._pack_thrust_lift
        assert (est[lift == 0] == 0).all()
    env.close()
