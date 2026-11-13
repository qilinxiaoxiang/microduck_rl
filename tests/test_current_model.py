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
    for path in flight:
        assert visual.visual_variant_for_manifest(path) == "battery-20261104", path
