"""CMU as-built mass variants of the velocity task (Pollen's recipe on the measured robot masses).

Which model to use: the `-CMU-Current` tasks (`Mjlab-Velocity-Flat-MicroDuck-CMU-Current`, `...-Rough-...`). They
resolve `cmu/model/current.json`, the one pointer to the robot as it is now (battery-20261104 on 2026-11-11), when
this module is imported. The dated ids (`-CMU-Plugged`, `-CMU-Battery`, `-CMU-Battery-20261104`) are historical:
kept so old runs reproduce, never the default for new training.

`make_cmu_velocity_env_cfg(variant)` is `make_microduck_velocity_env_cfg` with the robot's `spec_fn` wrapped so
that the signed mass parts of `cmu/model` (`variant_parts(variant)`) are applied to the MJCF bodies as explicit
inertials before mjlab compiles the scene, and (for battery-20261104, which the current variant resolves to) the
robot's own visual layer is drawn. Everything else (observations, 61-D contract, rewards, DR, BAM actuators, HOME
frame) is Pollen's. `cmu/model/apply_masses.py` is loaded by file path from the repository root; `cmu/` is not an
installed package.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APPLY_MASSES = _REPO_ROOT / "cmu" / "model" / "apply_masses.py"
_VISUAL = _REPO_ROOT / "cmu" / "model" / "visual.py"

CURRENT = "current"
# Dated variants that have their own visual layer. Older variants keep the legacy look they were trained with.
_VISUAL_VARIANTS = ("battery-20261104",)
# "current" first: the robot as it is now. The dated ones are historical.
CMU_VARIANTS = (CURRENT, "plugged", "battery", "battery-20261104")
HISTORICAL_VARIANTS = ("plugged", "battery", "battery-20261104")
# (task id, variant, rough)
CMU_VELOCITY_TASKS = tuple(
    (f"Mjlab-Velocity-{'Rough' if rough else 'Flat'}-MicroDuck-CMU-{variant.capitalize()}", variant, rough)
    for variant in CMU_VARIANTS
    for rough in (False, True)
)


def _load(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: the CMU tasks need the repository checkout (cmu/model)")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_apply_masses():
    return _load(_APPLY_MASSES, "cmu_model_apply_masses")


def load_visual():
    return _load(_VISUAL, "cmu_model_visual")


def resolve_variant(variant: str) -> str:
    """'current' -> the dated variant cmu/model/current.json names (manifest hash checked); a dated one -> itself."""
    return load_apply_masses().resolve_variant(variant)


def cmu_robot_spec_fn(base_spec_fn, variant: str, tag: str = "cmu"):
    """Wrap a robot spec_fn: apply the variant's signed mass parts and, for a variant with its own visual layer
    (the current robot), draw it. Shared by every CMU task family so the current robot looks the same everywhere."""
    apply_masses = load_apply_masses()
    resolved = apply_masses.resolve_variant(variant)
    parts = apply_masses.variant_parts(resolved)
    visual_variant = ""
    if resolved in _VISUAL_VARIANTS:
        visual = load_visual()
        visual_variant = visual.visual_variant_for_manifest(apply_masses.variant_manifest_path(resolved))
    label = f"{variant}={resolved}" if variant != resolved else variant

    def spec_fn():
        spec = base_spec_fn()
        report = apply_masses.apply_parts_to_spec(spec, parts)
        if visual_variant:
            load_visual().apply_visual_to_spec(spec, visual_variant)
        print(f"[{tag}:{label}] total mass {report['total_before_g']:.1f} g -> {report['total_after_g']:.1f} g; "
              f"bodies patched: {sorted(report['bodies'])}")
        return spec

    return spec_fn


def make_cmu_velocity_env_cfg(variant: str = CURRENT, play: bool = False, rough: bool = False):
    """Velocity cfg on a CMU mass variant. Default `current` (cmu/model/current.json); dated variants are historical."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    robot = cfg.scene.entities["robot"]
    spec_fn = cmu_robot_spec_fn(robot.spec_fn, variant)
    cfg.scene.entities = {**cfg.scene.entities, "robot": dataclasses.replace(robot, spec_fn=spec_fn)}
    return cfg


def run_label(variant: str) -> str:
    """'current-battery-20261104' for the alias (wandb runs record which robot 'current' meant), else the variant."""
    return f"{CURRENT}-{resolve_variant(CURRENT)}" if variant == CURRENT else variant


def cmu_rl_cfg(variant: str):
    return dataclasses.replace(MicroduckRlCfg, experiment_name=f"microduck_velocity_cmu_{variant}",
                               run_name=f"velocity_cmu_{run_label(variant)}")


def register_cmu_tasks(runner_cls) -> None:
    for task_id, variant, rough in CMU_VELOCITY_TASKS:
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_cmu_velocity_env_cfg(variant, rough=rough),
            play_env_cfg=make_cmu_velocity_env_cfg(variant, play=True, rough=rough),
            rl_cfg=cmu_rl_cfg(variant),
            runner_cls=runner_cls,
        )
