"""The robot as it is now is the default everywhere (cmu/model/current.json). CPU only, never steps an env.

  uv run --with pytest pytest tests/test_current_model.py
"""
import hashlib
import importlib
import inspect
import json
import pkgutil
import sys
from pathlib import Path

import pytest

import mjlab_microduck.tasks  # noqa: F401  (registers tasks)
from mjlab.tasks.registry import list_tasks

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRENT = REPO_ROOT / "cmu" / "model" / "current.json"
sys.path.insert(0, str(REPO_ROOT / "cmu" / "model"))
import visual  # noqa: E402

CURRENT_IDS = (
    "Mjlab-Velocity-Flat-MicroDuck-CMU-Current",
    "Mjlab-Velocity-Rough-MicroDuck-CMU-Current",
    "Mjlab-AimPick-Flat-MicroDuck-CMU-Current",
)
# defaults a task factory may carry for a parameter named `variant`
ALLOWED_DEFAULTS = ("current", inspect.Parameter.empty)


@pytest.fixture(scope="module")
def current():
    return json.loads(CURRENT.read_text())


def test_current_manifest_exists_and_hash_matches(current):
    path = REPO_ROOT / current["mass_manifest"]
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == current["mass_manifest_sha256"]


def test_current_visual_exists(current):
    path = REPO_ROOT / current["visual"]
    assert path.is_file(), path
    assert path.parent.is_dir()


def test_apply_masses_resolves_current(current):
    from mjlab_microduck.tasks.cmu_asbuilt import load_apply_masses
    am = load_apply_masses()
    assert am.resolve_variant("current") == current["variant"]
    assert am.variant_parts("current") == am.variant_parts(current["variant"])
    cli = inspect.getsource(am.main)
    assert "default=CURRENT" in cli


def _task_modules():
    """Every module of the tasks package that imports (testbench_env_cfg.py does not import against this mjlab and
    registers nothing; a module that cannot be imported cannot hand out a model)."""
    pkg = mjlab_microduck.tasks
    for info in pkgutil.iter_modules(pkg.__path__):
        try:
            yield importlib.import_module(f"{pkg.__name__}.{info.name}")
        except ImportError:
            continue


def test_no_task_factory_defaults_to_a_retired_variant():
    bad = []
    for module in _task_modules():
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if fn.__module__ != module.__name__:
                continue
            param = inspect.signature(fn).parameters.get("variant")
            if param is not None and param.default not in ALLOWED_DEFAULTS:
                bad.append(f"{module.__name__}.{name}(variant={param.default!r})")
    assert not bad, f"task factories defaulting to a retired model: {bad}"


def test_current_task_ids_registered():
    assert set(CURRENT_IDS) <= set(list_tasks())


def test_current_tasks_build_the_current_robot(current):
    """-CMU-Current compiles to the same masses as the dated variant current.json names, and draws its visual."""
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.tasks.cmu_asbuilt import make_cmu_velocity_env_cfg
    dated = make_cmu_velocity_env_cfg(current["variant"]).scene.entities["robot"].spec_fn().compile()
    for task_id in CURRENT_IDS:
        spec = load_env_cfg(task_id).scene.entities["robot"].spec_fn()
        names = [g.name for g in spec.geoms]
        assert any(n.startswith("cmu_") for n in names), f"{task_id}: no CMU visual geoms"
        model = spec.compile()
        if "Velocity" in task_id:
            assert abs(model.body_mass.sum() - dated.body_mass.sum()) < 1e-9, task_id


def test_visual_variant_for_manifest(tmp_path):
    manifests = REPO_ROOT / "cmu" / "model" / "manifests"
    assert visual.visual_variant_for_manifest(manifests / "plugged.json") == ""
    assert visual.visual_variant_for_manifest(manifests / "battery-20261104.json") == "battery-20261104"
    # a flight-pack manifest names its base robot at the start of "build"; the real ones live in the robotics
    # checkout beside this repository (extensions/flight-*/), so a clone checks the rule on a stand-in
    stand_in = tmp_path / "flightpack-nominal.json"
    stand_in.write_text(json.dumps({"build": "battery-20261104 + flight pack (stand-in)", "parts": []}))
    flight = [stand_in] + sorted(REPO_ROOT.parent.glob("extensions/flight-*/train/manifests/flightpack-*.json"))
    v2 = "battery-v2m5-20261121"  # the new battery backpack, current 2026-11-22 to 11-23, drawn by visual-20261122
    w918 = "battery-v2m5-w918-20261123"  # the same robot with masses matched to the weighed 918 g, current since 11-23
    for path in flight:
        build = json.loads(path.read_text()).get("build", "")
        if build.startswith("flightpack-"):  # hand-spliced on another payload manifest (flightpack-d2-bv2-nominal.json):
            with pytest.raises(ValueError):    # no robot named, so no sim may load it
                visual.visual_variant_for_manifest(path)
            continue
        want = v2 if build.startswith(v2 + " ") else "battery-20261104"
        assert visual.visual_variant_for_manifest(path) == want, path
    assert visual.visual_variant_for_manifest(manifests / f"{v2}.json") == v2 != visual.current_model()["variant"]
    assert visual.visual_variant_for_manifest(manifests / f"{w918}.json") == w918 == visual.current_model()["variant"]
    on_v2 = tmp_path / "flightpack-on-v2.json"
    on_v2.write_text(json.dumps({"build": f"{v2} + flight pack (stand-in)", "parts": []}))
    assert visual.visual_variant_for_manifest(on_v2) == v2


def test_backpack_v2_variant():
    """battery-v2m5-20261121: battery-20261104 with its 18 add-bp-* parts replaced by the 11 add-bv2-* parts; the
    committed manifest is what make_variants writes from the variant file; its visual layer (visual-20261122, also the
    current robot's) draws the new backpack and not the old one, without touching the physics."""
    import mujoco
    import apply_masses
    import make_variants
    name = "battery-v2m5-20261121"
    on_disk = apply_masses.load_manifest(apply_masses.MANIFEST_DIR / f"{name}.json")
    _, regenerated, removed, added, chain = make_variants.resolve_variant(make_variants.VARIANT_DIR / f"{name}.json")
    assert on_disk["parts"] == regenerated and on_disk["build"].startswith(name + " (the robot from 2026-11-22")
    assert len(removed) == 18 and all(p["id"].startswith("add-bp-") for p in removed)
    assert len(added) == 11 and all(p["id"].startswith("add-bv2-") for p in added)
    assert not any(p["id"].startswith("add-bp-") for p in on_disk["parts"])
    assert sum(p["mass_kg"] for p in added) * 1000 == pytest.approx(177.45, abs=0.01)
    assert chain[-1].endswith("variants/battery-20261104.json")
    drawn = {p["key"] for p in json.loads((REPO_ROOT / visual.current_model()["visual"]).read_text())["parts"]}
    assert {"bv2_frame", "bv2_cap", "bv2_battery", "bv2_switch", "bv2_dfr1202"} <= drawn
    assert not drawn & {"battery_backpack_frame", "battery_backpack_cap", "battery_hatch", "bp_battery"}
    assert not any(k.startswith("fp_") for k in drawn) and "roof_plate" not in drawn
    spec = mujoco.MjSpec.from_file(str(apply_masses.SCENE_XML))
    m0 = spec.compile()
    assert visual.apply_visual_to_spec(spec, name) > 0
    m1 = spec.compile()
    assert abs(m1.body_mass.sum() - m0.body_mass.sum()) < 1e-12


def test_weighed_918_variant_is_current():
    """battery-v2m5-w918-20261123: battery-v2m5-20261121 with its total mass matched to the robot weighed at 918 g on
    2026-11-23 (extensions/battery-walk-20261122/mass-audit-20261123 in the robotics checkout); the committed manifest is
    what make_variants writes; current.json names it; it is drawn by the same visual layer as battery-v2m5-20261121."""
    import apply_masses
    import make_variants
    name = "battery-v2m5-w918-20261123"
    assert visual.current_model()["variant"] == name
    assert visual.current_model()["visual"] == "cmu/model/visual-20261122/visual.json"
    on_disk = apply_masses.load_manifest(apply_masses.MANIFEST_DIR / f"{name}.json")
    _, regenerated, _, _, chain = make_variants.resolve_variant(make_variants.VARIANT_DIR / f"{name}.json")
    assert on_disk["parts"] == regenerated and on_disk["build"].startswith(name + " (")
    assert chain[-1].endswith("variants/battery-v2m5-20261121.json")
    assert make_variants.measure(name)["total_g"] == pytest.approx(918.0, abs=0.01)
    import mujoco
    a, _ = apply_masses.build_scene(name)
    b, _ = apply_masses.build_scene("battery-v2m5-20261121")
    na, nb = visual.apply_visual_to_spec(a, name), visual.apply_visual_to_spec(b, "battery-v2m5-20261121")
    assert na == nb > 0
    assert {g.name for g in a.geoms if g.name.startswith("cmu_")} == {g.name for g in b.geoms if g.name.startswith("cmu_")}
    assert isinstance(a.compile(), mujoco.MjModel)

