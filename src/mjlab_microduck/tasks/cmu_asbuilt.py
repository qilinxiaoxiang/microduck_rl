"""CMU as-built mass variants of the velocity task (Pollen's recipe on the measured robot masses).

`make_cmu_velocity_env_cfg(variant)` is `make_microduck_velocity_env_cfg` with the robot's `spec_fn` wrapped so
that the signed mass parts of `cmu/model` (`variant_parts(variant)`) are applied to the MJCF bodies as explicit
inertials before mjlab compiles the scene. Everything else (observations, 61-D contract, rewards, DR, BAM
actuators, HOME frame) is Pollen's. `cmu/model/apply_masses.py` is loaded by file path from the repository
root; `cmu/` is not an installed package.
"""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from .microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
_APPLY_MASSES = _REPO_ROOT / "cmu" / "model" / "apply_masses.py"

CMU_VARIANTS = ("plugged", "battery")
# (task id, variant, rough)
CMU_VELOCITY_TASKS = tuple(
    (f"Mjlab-Velocity-{'Rough' if rough else 'Flat'}-MicroDuck-CMU-{variant.capitalize()}", variant, rough)
    for variant in CMU_VARIANTS
    for rough in (False, True)
)


def load_apply_masses():
    if not _APPLY_MASSES.exists():
        raise FileNotFoundError(f"{_APPLY_MASSES} not found: the CMU tasks need the repository checkout (cmu/model)")
    spec = importlib.util.spec_from_file_location("cmu_model_apply_masses", _APPLY_MASSES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cmu_velocity_env_cfg(variant: str, play: bool = False, rough: bool = False):
    apply_masses = load_apply_masses()
    parts = apply_masses.variant_parts(variant)
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn

    def spec_fn():
        spec = base_spec_fn()
        report = apply_masses.apply_parts_to_spec(spec, parts)
        print(f"[cmu:{variant}] total mass {report['total_before_g']:.1f} g -> {report['total_after_g']:.1f} g; "
              f"bodies patched: {sorted(report['bodies'])}")
        return spec

    cfg.scene.entities = {**cfg.scene.entities, "robot": dataclasses.replace(robot, spec_fn=spec_fn)}
    return cfg


def cmu_rl_cfg(variant: str):
    return dataclasses.replace(MicroduckRlCfg, experiment_name=f"microduck_velocity_cmu_{variant}",
                               run_name=f"velocity_cmu_{variant}")


def register_cmu_tasks(runner_cls) -> None:
    for task_id, variant, rough in CMU_VELOCITY_TASKS:
        register_mjlab_task(
            task_id=task_id,
            env_cfg=make_cmu_velocity_env_cfg(variant, rough=rough),
            play_env_cfg=make_cmu_velocity_env_cfg(variant, play=True, rough=rough),
            rl_cfg=cmu_rl_cfg(variant),
            runner_cls=runner_cls,
        )
