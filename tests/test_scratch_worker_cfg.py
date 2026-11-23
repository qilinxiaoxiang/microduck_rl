"""The from-scratch route (cmu/finetune/kaggle/scratch_worker.py) trains Pollen's recipe unchanged.

CPU only: builds cfgs and compiles the robot models, never steps an env. For --model stock the env cfg must equal
make_microduck_velocity_env_cfg() field by field; for --model current only the robot's spec_fn may differ, and the
compiled models may differ only in body masses/inertias and the ankle joint range (the heel stop). The runner cfg must
equal MicroduckRlCfg apart from the bookkeeping fields. Run with -s to see the diff printed.
"""
import dataclasses
import enum
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cmu/finetune/kaggle"))
import scratch_worker as sw  # noqa: E402

ROBOT_SPEC_FN = "scene.entities['robot'].spec_fn"


def _fn_id(f):
    """Stable identity of a callable: module.qualname, plus its closure values for factory-made functions."""
    name = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', type(f).__name__)}"
    cells = getattr(f, "__closure__", None) or ()
    return (name, tuple(repr(c.cell_contents) for c in cells if not callable(c.cell_contents)))


def cfg_diff(a, b, path="", out=None, seen=None):
    """Every path where two cfg trees differ (dataclasses, dicts, sequences, callables, arrays, scalars)."""
    out = [] if out is None else out
    seen = set() if seen is None else seen
    if id(a) in seen and id(b) in seen:
        return out
    seen.update((id(a), id(b)))
    if type(a) is not type(b):
        out.append((path, f"type {type(a).__name__} != {type(b).__name__}"))
    elif dataclasses.is_dataclass(a) and not isinstance(a, type):
        for f in dataclasses.fields(a):
            cfg_diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}" if path else f.name, out, seen)
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append((f"{path}[{k!r}]", "only in " + ("route" if k in a else "upstream")))
            else:
                cfg_diff(a[k], b[k], f"{path}[{k!r}]", out, seen)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} != {len(b)}"))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                cfg_diff(x, y, f"{path}[{i}]", out, seen)
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b):
            out.append((path, "array differs"))
    elif isinstance(a, type) or isinstance(a, enum.Enum):
        if a is not b:
            out.append((path, f"{a} != {b}"))
    elif callable(a):
        if _fn_id(a) != _fn_id(b):
            out.append((path, f"{_fn_id(a)} != {_fn_id(b)}"))
    elif hasattr(a, "__dict__") and not isinstance(a, (str, bytes, int, float, bool)):
        cfg_diff(vars(a), vars(b), path, out, seen)
    elif a != b:
        out.append((path, f"{a!r} != {b!r}"))
    return out


def _report(label, diffs):
    print(f"\n[{label}] {len(diffs)} difference(s) from upstream")
    for p, d in diffs:
        print(f"  {p}: {d}")


@pytest.fixture(scope="module")
def upstream():
    return make_microduck_velocity_env_cfg(play=False)


def test_stock_env_cfg_equals_upstream(upstream):
    diffs = cfg_diff(sw.make_env_cfg("stock"), upstream)
    _report("stock env cfg", diffs)
    assert diffs == []


def test_current_env_cfg_differs_only_in_robot_model(upstream):
    diffs = cfg_diff(sw.make_env_cfg("current"), upstream)
    _report("current env cfg", diffs)
    assert [p for p, _ in diffs] == [ROBOT_SPEC_FN]


def test_the_diff_sees_recipe_changes(upstream):
    """Negative control: the comparison catches the changes earlier routes made (weights, DR ranges, noise, curricula,
    command ranges, the fitted robot conditions)."""
    from mjlab_microduck.robot_conditions import apply_robot_conditions
    c = sw.make_env_cfg("stock")
    c.rewards["track_linear_velocity"].weight *= 1.01
    c.events["foot_friction"].params["ranges"] = (0.4, 1.3)
    c.observations["actor"].terms["base_ang_vel"].noise.n_max *= 2
    c.curriculum["action_rate_weight"].params["weight_stages"][-1]["weight"] = -0.123
    c.commands["twist"].ranges.lin_vel_x = (-0.3, 0.3)
    found = cfg_diff(c, upstream)
    assert len(found) >= 5, found
    rc = sw.make_env_cfg("stock")
    apply_robot_conditions(rc)
    assert len(cfg_diff(rc, upstream)) >= 3


def test_two_builds_of_the_route_are_independent(upstream):
    """No shared mutable state: building `current` does not change a later `stock` build or the upstream cfg."""
    sw.make_env_cfg("current")
    assert cfg_diff(sw.make_env_cfg("stock"), make_microduck_velocity_env_cfg(play=False)) == []


def _model_facts(cfg):
    m = cfg.scene.entities["robot"].spec_fn().compile()
    bodies = {m.body(i).name: (float(m.body_mass[i]), m.body_ipos[i].copy(), m.body_inertia[i].copy()) for i in range(m.nbody)}
    joints = {m.joint(i).name: m.jnt_range[i].copy() for i in range(m.njnt)}
    coll = sorted((m.geom(i).name, int(m.geom_contype[i]), int(m.geom_conaffinity[i])) for i in range(m.ngeom)
                  if m.geom_contype[i] or m.geom_conaffinity[i])
    return m, bodies, joints, coll


def test_current_model_differs_only_in_mass_and_ankle_stop(upstream):
    m0, b0, j0, c0 = _model_facts(upstream)
    m1, b1, j1, c1 = _model_facts(sw.make_env_cfg("current"))
    print(f"\n[current model] total mass {m0.body_mass.sum() * 1000:.1f} g (stock) -> {m1.body_mass.sum() * 1000:.1f} g")
    assert set(b0) == set(b1) and set(j0) == set(j1)
    changed_bodies = sorted(n for n in b0 if not (np.isclose(b0[n][0], b1[n][0]) and np.allclose(b0[n][1], b1[n][1])
                                                  and np.allclose(b0[n][2], b1[n][2])))
    for n in changed_bodies:
        print(f"  body {n}: {b0[n][0] * 1000:.1f} g -> {b1[n][0] * 1000:.1f} g")
    changed_joints = sorted(n for n in j0 if not np.allclose(j0[n], j1[n]))
    for n in changed_joints:
        print(f"  joint {n} range {np.round(j0[n], 3).tolist()} -> {np.round(j1[n], 3).tolist()}")
    assert changed_joints == ["left_ankle", "right_ankle"]
    assert np.allclose(j1["left_ankle"], [-1.36, 1.36])
    # collision set, actuators, joints, DOFs are the stock model's: the mass parts and the visual layer add no contact
    assert c0 == c1, "collision geoms differ"
    assert (m0.nu, m0.nq, m0.nv, m0.nbody) == (m1.nu, m1.nq, m1.nv, m1.nbody)
    assert 1000 < m1.body_mass.sum() * 1000 < 1070  # 1034.8 g current robot (cmu/model/current.json)


def test_agent_cfg_is_the_official_runner_cfg():
    cfg = sw.make_agent_cfg("bw-s1-cur-s1", seed=2, max_iters=4000, save_interval=250)
    diffs = cfg_diff(cfg, MicroduckRlCfg)
    _report("runner cfg", diffs)
    assert {p for p, _ in diffs} <= set(sw.AGENT_BOOKKEEPING)
    a = cfg.actor
    assert a.distribution_cfg["init_std"] == 1.0 and cfg.algorithm.learning_rate == 1e-3
    assert cfg.algorithm.entropy_coef == 0.01 and a.hidden_dims == (512, 256, 128)
    assert cfg.save_interval == MicroduckRlCfg.save_interval == 250 and cfg.num_steps_per_env == 24


def test_kaggle_preset_flags_exist_in_the_worker():
    import kaggle_job as kj
    args = kj.worker_args("scratch", "bw-s1-cur-s1", ["--model", "current", "--seed", "1"])
    assert kj.flag_value(args, "--max-iters") == "4000" and kj.flag_value(args, "--max-hours") == "5.8"
    with pytest.raises(SystemExit):
        kj.worker_args("scratch", "x", ["--seed", "1"])  # no --model
    with pytest.raises(SystemExit):
        kj.worker_args("scratch", "x", ["--model", "current", "--delta", "batreal"])  # a fine-tune flag
