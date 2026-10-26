"""Cfg invariants for the CMU as-built mass variants (cmu/model, tasks/cmu_asbuilt.py).

CPU only: builds cfgs and compiles the MuJoCo models, never steps an env.
"""
import copy
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.robot.microduck_constants import get_walk_spec
from mjlab_microduck.tasks.cmu_asbuilt import CMU_VELOCITY_TASKS, load_apply_masses, make_cmu_velocity_env_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "cmu" / "model"))
import make_variants  # noqa: E402

apply_masses = load_apply_masses()

EXPECTED_TASKS = (
    "Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged",
    "Mjlab-Velocity-Rough-MicroDuck-CMU-Plugged",
    "Mjlab-Velocity-Flat-MicroDuck-CMU-Battery",
    "Mjlab-Velocity-Rough-MicroDuck-CMU-Battery",
)


def _total_g(model: mujoco.MjModel) -> float:
    return float(model.body_mass.sum()) * 1000


@pytest.fixture(scope="module")
def measured():
    return {v: make_variants.measure(v) for v in ("official", "asbuilt", "plugged", "battery")}


# ── (a) registry ─────────────────────────────────────────────────────────────

def test_four_cmu_tasks_registered():
    tasks = set(list_tasks())
    assert set(EXPECTED_TASKS) <= tasks
    assert {t for t, _, _ in CMU_VELOCITY_TASKS} == set(EXPECTED_TASKS)


@pytest.mark.parametrize("task_id", EXPECTED_TASKS)
def test_registered_cfg_is_a_velocity_cfg_with_a_distinct_experiment(task_id):
    from mjlab.tasks.registry import load_rl_cfg
    cfg = load_env_cfg(task_id)
    assert "robot" in cfg.scene.entities
    rl = load_rl_cfg(task_id)
    variant = task_id.rsplit("-", 1)[-1].lower()
    assert rl.experiment_name == f"microduck_velocity_cmu_{variant}"
    assert rl.experiment_name != "velocity"


# ── (b) spec compiles and the mass matches make_variants ─────────────────────

@pytest.mark.parametrize("task_id", EXPECTED_TASKS)
def test_task_spec_compiles_with_make_variants_mass(task_id, measured):
    variant = task_id.rsplit("-", 1)[-1].lower()
    spec = load_env_cfg(task_id).scene.entities["robot"].spec_fn()
    model = spec.compile()
    assert model.nbody == 16
    assert abs(_total_g(model) - measured[variant]["total_g"]) < 0.5
    # the walk model has no keyframe; HOME comes from the joint table and must give the same COM as the scene
    _, com_mm = apply_masses.mass_and_com(model)
    assert np.allclose(com_mm, measured[variant]["com_mm"], atol=0.05)


def test_play_cfg_uses_the_same_masses(measured):
    for variant in ("plugged", "battery"):
        train = make_cmu_velocity_env_cfg(variant).scene.entities["robot"].spec_fn().compile()
        play = make_cmu_velocity_env_cfg(variant, play=True).scene.entities["robot"].spec_fn().compile()
        assert np.allclose(train.body_mass, play.body_mass)
        assert abs(_total_g(train) - measured[variant]["total_g"]) < 0.5


# ── (c) variant totals differ from asbuilt by the signed delta sums ──────────

def test_variant_totals_are_asbuilt_plus_signed_deltas(measured):
    plugged_delta = make_variants.net_delta_g("plugged")
    battery_delta = make_variants.net_delta_g("battery")  # relative to plugged
    assert measured["plugged"]["total_g"] - measured["asbuilt"]["total_g"] == pytest.approx(plugged_delta, abs=0.01)
    assert measured["battery"]["total_g"] - measured["plugged"]["total_g"] == pytest.approx(battery_delta, abs=0.01)
    assert measured["battery"]["total_g"] - measured["asbuilt"]["total_g"] == pytest.approx(plugged_delta + battery_delta, abs=0.01)
    # the deltas are the ones the design states: Pi (9.3 g) -> A7Z (18 g) + antenna (1.5 g) + the rear box lid
    # (7.80 g slicer, on since 2026-10-26) with its four M2 x 10 (0.316 g each); box with lid -> backpack
    lid_g = 7.80 + 4 * 0.31567
    # the A7Z left the head for the roof saddle (2026-10-26): + saddle body 17.743 g (V1.33 slicer), - the three Pi-era
    # board screws E-pi (0.671 g), - the CSI ribbon (1.6 g); the IMU, UART, 5 V and amplifier leads are re-run to the
    # saddle header (their totals before and after are summed from the manifests)
    def g(manifest, prefixes):
        parts = apply_masses.load_manifest(manifest)["parts"]
        return sum(p["mass_kg"] for p in parts if p["id"].startswith(prefixes)) * 1000
    leads = tuple(f"add-cable-{c}-" for c in ("imu", "uart", "pi-lead", "amp-lrc", "amp-bclk", "amp-din", "amp-gnd", "amp-vin"))
    leads_delta = g(make_variants.MANIFEST_DIR / "plugged.json", leads) - g(apply_masses.BASE_MANIFEST, leads)
    saddle_g = 17.743 - (0.237 + 0.197 + 0.237) - 1.6 + leads_delta
    assert -4.0 < leads_delta < -3.0  # shorter torso leads (about -6.0 g) against longer amplifier leads (+1.4 g)
    assert plugged_delta == pytest.approx(-9.3 + 18.0 + 1.5 + lid_g + saddle_g, abs=0.01)
    assert battery_delta == pytest.approx(97.1 - 59.513 - lid_g, abs=0.01)
    # the battery removal is in every variant and asbuilt = official + manifest - 90 g
    base = apply_masses.load_manifest(apply_masses.BASE_MANIFEST)
    manifest_g = sum(p["mass_kg"] for p in base["parts"]) * 1000
    assert measured["asbuilt"]["total_g"] - measured["official"]["total_g"] == pytest.approx(manifest_g - 90.0, abs=0.01)


def test_manifests_match_their_variant_files():
    for name in ("plugged", "battery"):
        on_disk = apply_masses.load_manifest(apply_masses.MANIFEST_DIR / f"{name}.json")["parts"]
        _, regenerated, _, _, _ = make_variants.resolve_variant(make_variants.VARIANT_DIR / f"{name}.json")
        assert [p["id"] for p in on_disk] == [p["id"] for p in regenerated], f"{name}: run make_variants.py"
        assert sum(p["mass_kg"] for p in on_disk) == pytest.approx(sum(p["mass_kg"] for p in regenerated), abs=1e-9)
    battery_ids = {p["id"] for p in apply_masses.variant_parts("battery")}
    assert "add-pi" not in battery_ids and "add-a7z" in battery_ids and "add-bp-battery-tattu-2s-650" in battery_ids
    assert not any(i.startswith(("add-box-", "add-screw-C-xl4015-", "add-screw-C-pd5991-", "add-cable-pw-")) for i in battery_ids)
    assert "remove-np_f970" in battery_ids


# ── (d) actor observation terms unchanged (61-D contract) ────────────────────

@pytest.mark.parametrize("variant", ("plugged", "battery"))
@pytest.mark.parametrize("rough", (False, True))
def test_actor_obs_terms_unchanged(variant, rough):
    base = make_microduck_velocity_env_cfg(rough=rough)
    cmu = make_cmu_velocity_env_cfg(variant, rough=rough)
    assert list(cmu.observations.keys()) == list(base.observations.keys())
    for group in base.observations:
        assert list(cmu.observations[group].terms.keys()) == list(base.observations[group].terms.keys())
    assert list(cmu.observations["actor"].terms.keys()) == [
        "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel", "actions", "command", "head_command", "body_command",
    ]
    assert cmu.actions.keys() == base.actions.keys()
    assert cmu.scene.entities["robot"].init_state == base.scene.entities["robot"].init_state
    assert cmu.scene.entities["robot"].articulation == base.scene.entities["robot"].articulation


# ── (e) apply_parts_to_spec touches only the named bodies, inertias stay PD ──

def test_apply_parts_leaves_other_bodies_identical_and_inertias_positive_definite():
    reference = get_walk_spec().compile()
    spec = get_walk_spec()
    parts = apply_masses.variant_parts("battery")
    touched = {p["body"] for p in parts}
    report = apply_masses.apply_parts_to_spec(spec, parts)
    model = spec.compile()
    assert set(report["bodies"]) == touched
    for i in range(model.nbody):
        name = model.body(i).name
        if name in touched:
            assert model.body_mass[i] == pytest.approx(report["bodies"][name]["after_g"] / 1000, abs=1e-5)
        else:
            assert model.body_mass[i] == reference.body_mass[i]
            assert np.array_equal(model.body_ipos[i], reference.body_ipos[i])
            assert np.array_equal(model.body_inertia[i], reference.body_inertia[i])
            assert np.array_equal(model.body_iquat[i], reference.body_iquat[i])
        if model.body_mass[i] > 0:
            I = model.body_inertia[i]
            assert np.all(I > 0), name
            assert I.max() <= I.sum() - I.max() + 1e-12, name  # triangle inequality: a physical body
    assert model.njnt == reference.njnt and model.ngeom == reference.ngeom and model.nsite == reference.nsite
    assert np.array_equal(model.geom_size, reference.geom_size)


def test_apply_parts_rejects_a_body_going_massless():
    spec = get_walk_spec()
    with pytest.raises(AssertionError):
        apply_masses.apply_parts_to_spec(spec, [dict(id="x", body="hip_l", mass_kg=-1.0, position_m=[0, 0, 0], dimensions_m=[.01, .01, .01])])


def test_parts_do_not_alias_between_calls():
    a = apply_masses.variant_parts("plugged")
    b = apply_masses.variant_parts("plugged")
    a[0]["mass_kg"] += 1.0
    assert b[0]["mass_kg"] != a[0]["mass_kg"]
    assert copy.deepcopy(b) == apply_masses.variant_parts("plugged")
