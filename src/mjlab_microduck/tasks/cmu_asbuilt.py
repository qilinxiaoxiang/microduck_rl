"""CMU as-built mass variants of the velocity task (Pollen's recipe on the measured robot masses).

Which model to use: the `-CMU-Current` tasks (`Mjlab-Velocity-Flat-MicroDuck-CMU-Current`, `...-Rough-...`). They
resolve `cmu/model/current.json`, the one pointer to the robot as it is now (battery-v2m5-w918-20261123 since 2026-11-23:
the battery backpack with the 2S 3000 mAh battery, masses matched to the weighed 918 g), when this module is imported. The dated ids (`-CMU-Plugged`,
`-CMU-Battery`, `-CMU-Battery-20261104`, `-CMU-Battery-v2m5-20261121`, `-CMU-Battery-v2m5-w918-20261123`) name a fixed robot; all but the current one are historical:
kept so old runs reproduce, never the default for new training.

`make_cmu_velocity_env_cfg(variant)` is `make_microduck_velocity_env_cfg` with the robot's `spec_fn` wrapped so
that the signed mass parts of `cmu/model` (`variant_parts(variant)`) are applied to the MJCF bodies as explicit
inertials before mjlab compiles the scene, and (for battery-20261104, battery-v2m5-20261121 and battery-v2m5-w918-20261123, which the current
variant resolves to) the robot's own visual layer is drawn. Everything else (observations, 61-D contract, rewards, DR, BAM actuators, HOME
frame) is Pollen's, except that the `current` tasks also train under the real robot's measured sensor and actuator
conditions (`mjlab_microduck/robot_conditions.py`, docs/SIM-TO-REAL-GAPS.md). `cmu/model/apply_masses.py` is loaded by file path from the repository root; `cmu/` is not an
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
_VISUAL_VARIANTS = ("battery-20261104", "battery-v2m5-20261121", "battery-v2m5-w918-20261123",
                    # opt-in foot-contact siblings of w918 (never current, no task id): drawn as w918
                    "battery-v2m5-w918-sole1-20261123", "battery-v2m5-w918-box1-20261123")
# "current" first: the robot as it is now. The dated ones are historical.
CMU_VARIANTS = (CURRENT, "plugged", "battery", "battery-20261104", "battery-v2m5-20261121",
                "battery-v2m5-w918-20261123")
HISTORICAL_VARIANTS = ("plugged", "battery", "battery-20261104", "battery-v2m5-20261121")
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
    # the manifest's dated foot-contact layer (cmu/model/foot_contact.py), the same one the judging sim
    # (evaluate_delay.load, brain_sim_check.py) applies; None for every manifest without the key (current included)
    foot_layer = apply_masses.variant_foot_contact(resolved)

    def spec_fn():
        spec = base_spec_fn()
        report = apply_masses.apply_parts_to_spec(spec, parts)
        if foot_layer:
            apply_masses.foot_contact_module().apply_foot_contact(spec, foot_layer)
        if visual_variant:
            load_visual().apply_visual_to_spec(spec, visual_variant)
        print(f"[{tag}:{label}] total mass {report['total_before_g']:.1f} g -> {report['total_after_g']:.1f} g; "
              f"bodies patched: {sorted(report['bodies'])}"
              + (f"; foot contact {foot_layer['name']}" if foot_layer else ""), flush=True)
        return spec

    spec_fn.foot_contact = foot_layer["name"] if foot_layer else None  # read by apply_foot_contact_to_cfg
    return spec_fn


# A foot-contact layer turns the printed-foot meshes' floor contact off and adds footc_left / footc_right on the ankle
# bodies. mjlab applies the robot's CollisionCfg AFTER spec_fn, and FULL_COLLISION (microduck_constants.py) would turn
# the printed meshes back on (they match ".*_collision") and switch the footc geoms off (disable_other_geoms), so the
# robot cfg gets this collision set instead: same values as FULL_COLLISION for every other geom, the footc geoms take
# the printed feet's condim 3, priority 1 and friction 1.0, and the printed feet stay off. solref/solimp are left to
# the layer (None here = untouched).
FOOT_CONTACT_GEOMS = ("footc_left", "footc_right")
_FOOTC_RE = r"^footc_(left|right)$"
_PRINTED_FOOT_RE = r"^(left|right)_foot_collision$"


def foot_contact_collision_cfg(base):
    return dataclasses.replace(
        base,
        geom_names_expr=tuple(base.geom_names_expr) + (_FOOTC_RE,),
        contype={_PRINTED_FOOT_RE: 0, ".*": 1},
        conaffinity={_PRINTED_FOOT_RE: 0, ".*": 1},
        condim={_FOOTC_RE: 3, **dict(base.condim)},
        priority={_FOOTC_RE: 1, **dict(base.priority)},
        friction={_FOOTC_RE: (1.0,), **dict(base.friction)},
    )


def apply_foot_contact_to_cfg(cfg, variant: str) -> str | None:
    """When the variant's manifest names a foot-contact layer, make the rest of the env use it (in place): the robot's
    collision set (above), the feet contact sensor (air time, contact rewards) and the foot friction randomisation
    point at footc_left / footc_right (left first, as before), and the spawn height range rises by the layer's drop
    below the printed soles (apply_masses.foot_contact_drop_m, about 1 mm), so the layer starts as high above the floor
    as the printed soles did. Returns the layer name, or None and changes nothing."""
    robot = cfg.scene.entities["robot"]
    layer_name = getattr(robot.spec_fn, "foot_contact", None)
    if not layer_name:
        return None
    collisions = tuple(robot.collisions)
    assert len(collisions) == 1, f"foot contact expects one robot CollisionCfg, found {len(collisions)}"
    cfg.scene.entities = {**cfg.scene.entities,
                          "robot": dataclasses.replace(robot, collisions=(foot_contact_collision_cfg(collisions[0]),))}
    sensors = []
    for s in cfg.scene.sensors:
        if getattr(s, "name", None) == "feet_ground_contact":
            assert s.primary.pattern == r"^(left_foot_collision|right_foot_collision)$", s.primary.pattern
            s = dataclasses.replace(s, primary=dataclasses.replace(s.primary, pattern=r"^(footc_left|footc_right)$"))
        sensors.append(s)
    cfg.scene.sensors = tuple(sensors)
    ev = cfg.events["foot_friction"]
    ev.params = dict(ev.params, asset_cfg=dataclasses.replace(ev.params["asset_cfg"], geom_names=FOOT_CONTACT_GEOMS))
    drop = load_apply_masses().foot_contact_drop_m(robot.spec_fn())
    assert 0.0 <= drop < 0.005, f"{layer_name}: layer {drop * 1000:.2f} mm below the printed soles; expected about 1 mm"
    rb = cfg.events["reset_base"]
    lo, hi = rb.params["pose_range"]["z"]
    rb.params = dict(rb.params, pose_range=dict(rb.params["pose_range"], z=(lo + drop, hi + drop)))
    cfg.__dict__["_foot_contact_drop_m"] = drop  # read by tests and the worker's record
    return layer_name


def make_cmu_velocity_env_cfg(variant: str = CURRENT, play: bool = False, rough: bool = False,
                              robot_conditions: bool | None = None):
    """Velocity cfg on a CMU mass variant. Default `current` (cmu/model/current.json); dated variants are historical.

    robot_conditions: apply the real robot's measured sensor and actuator conditions
    (`mjlab_microduck.robot_conditions.ROBOT_CONDITIONS`: leg-speed gyro noise, joint velocity and gravity noise, one
    tick old joint position, 4 % servo read drops, BAM friction x1.0-1.5). Default (None): on for `current`, off for
    the dated variants, so historical runs reproduce. docs/SIM-TO-REAL-GAPS.md has the evidence for each value."""
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
    robot = cfg.scene.entities["robot"]
    spec_fn = cmu_robot_spec_fn(robot.spec_fn, variant)
    cfg.scene.entities = {**cfg.scene.entities, "robot": dataclasses.replace(robot, spec_fn=spec_fn)}
    apply_foot_contact_to_cfg(cfg, variant)  # inert unless the variant's manifest names a foot-contact layer
    if robot_conditions is None:
        robot_conditions = variant == CURRENT
    if robot_conditions:
        from mjlab_microduck.robot_conditions import apply_robot_conditions
        apply_robot_conditions(cfg)
    return cfg


# ---------------------------------------------------------------- flight pack (opt-in; never current, no task id)
# pack variant -> the robot it was built on (the mass ramp's fraction-0 body)
PACK_VARIANTS = {"flightpack-v1-w918-20261124": "battery-v2m5-w918-20261123"}


def thrust_sites(variant: str) -> dict:
    """cmu/model/<variant>/thrust_sites.json: fan sites and axes in the trunk_base frame (m), lag."""
    path = _REPO_ROOT / "cmu" / "model" / variant / "thrust_sites.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: the thrust event needs the pack variant's fan sites")
    import json
    d = json.loads(path.read_text())
    assert d.get("body", "trunk_base") == "trunk_base", d.get("body")
    fans = d["fans"]
    assert fans and all(len(f["position_m"]) == 3 and len(f["thrust_axis"]) == 3 for f in fans)
    for f in fans:
        assert max(abs(v) for v in f["position_m"]) < 0.5, f"{f['name']}: position not in metres?"
    lag = d.get("lag", {})
    return dict(path=str(path), names=[f["name"] for f in fans], positions=[list(map(float, f["position_m"])) for f in fans],
                axes=[list(map(float, f["thrust_axis"])) for f in fans],
                tau_s=float(lag.get("tau_up_s", 0.10)), tau_down_s=float(lag.get("tau_down_s", lag.get("tau_up_s", 0.10))))


def robot_body_inertials(variant: str) -> dict:
    """Compile the robot alone on a variant and return {body: mass, ipos, inertia, iquat} (MuJoCo body frame)."""
    import mujoco
    base = make_microduck_velocity_env_cfg().scene.entities["robot"].spec_fn
    m = cmu_robot_spec_fn(base, variant, tag="inertials")().compile()
    return {m.body(i).name: dict(mass=float(m.body_mass[i]), ipos=[float(v) for v in m.body_ipos[i]],
                                 inertia=[float(v) for v in m.body_inertia[i]], iquat=[float(v) for v in m.body_iquat[i]])
            for i in range(1, m.nbody)}


def pack_changed_bodies(variant: str) -> dict:
    """Base-robot inertials of the bodies whose inertials differ between the pack variant and its base."""
    import numpy as np
    a, b = robot_body_inertials(PACK_VARIANTS[variant]), robot_body_inertials(variant)
    assert set(a) == set(b), set(a) ^ set(b)
    out = {}
    for k in a:
        if (abs(a[k]["mass"] - b[k]["mass"]) > 1e-7 or not np.allclose(a[k]["ipos"], b[k]["ipos"], atol=1e-7)
                or not np.allclose(a[k]["inertia"], b[k]["inertia"], atol=1e-10)
                or not np.allclose(np.abs(a[k]["iquat"]), np.abs(b[k]["iquat"]), atol=1e-6)):
            out[k] = a[k]
    return out


def apply_pack_options(cfg, variant: str, thrust_lift=None, zero_share: float = 0.25, drop_share: float = 0.15,
                       lag: float | None = None, imbalance: float = 0.10, told: bool = False,
                       ramp_steps: int = 0, hold_s=(2.0, 6.0), est_delay_ticks=(2, 6), est_noise: float = 0.05,
                       levels: int = 0, level_init_max: int = 5, gate_speed: float = 0.04,
                       min_fwd_s: float = 2.0) -> dict:
    """Add the pack's opt-in terms to a cfg built on a PACK_VARIANTS body (in place); returns the record.
    thrust_lift (lo, hi): the pack_thrust step event (mdp.pack_thrust). told: the delayed noisy lift estimate as the
    actor's last input (62nd) and the true lift as the critic's last input. ramp_steps > 0: the pack_mass_ramp reset
    event over that many env steps. All off: the cfg is untouched."""
    from mjlab.managers import EventTermCfg, ObservationTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from . import mdp as microduck_mdp
    assert variant in PACK_VARIANTS, variant
    rec = {}
    if thrust_lift is not None:
        sites = thrust_sites(variant)
        tau = sites["tau_s"] if lag is None else float(lag)
        params = dict(positions=sites["positions"], axes=sites["axes"], lift_range=tuple(map(float, thrust_lift)),
                      tau=tau, zero_share=float(zero_share), drop_share=float(drop_share), imbalance=float(imbalance),
                      hold_s=tuple(hold_s), est_delay_ticks=tuple(est_delay_ticks), est_noise=float(est_noise),
                      asset_cfg=SceneEntityCfg("robot"))
        cfg.events["pack_thrust"] = EventTermCfg(func=microduck_mdp.pack_thrust, mode="step", params=params)
        rec["thrust"] = dict({k: v for k, v in params.items() if k != "asset_cfg"}, sites=sites["path"],
                             fans=sites["names"])
    if told:
        assert thrust_lift is not None, "told needs the thrust event"
        cfg.observations["actor"].terms["pack_thrust_lift"] = ObservationTermCfg(
            func=microduck_mdp.pack_thrust_lift, params={"estimate": True})
        cfg.observations["critic"].terms["pack_thrust_lift"] = ObservationTermCfg(
            func=microduck_mdp.pack_thrust_lift, params={"estimate": False})
        assert list(cfg.observations["actor"].terms)[-1] == "pack_thrust_lift"
        rec["told"] = dict(actor="delayed noisy estimate (last input)", critic="true lift (last input)",
                           delay_ticks=list(est_delay_ticks), noise_sd_of_weight=est_noise)
    if ramp_steps > 0:
        base = pack_changed_bodies(variant)
        cfg.events["pack_mass_ramp"] = EventTermCfg(
            func=microduck_mdp.pack_mass_ramp, mode="reset",
            params=dict(base_inertials=base, ramp_steps=int(ramp_steps), asset_cfg=SceneEntityCfg("robot")))
        assert list(cfg.events)[-1] == "pack_mass_ramp"  # after the reset COM DR it composes with
        rec["mass_ramp"] = dict(ramp_steps=int(ramp_steps), base_variant=PACK_VARIANTS[variant], bodies=sorted(base))
    if levels > 0:
        assert ramp_steps <= 0, "pack levels replace the iteration ramp"
        assert levels >= 2, levels
        base = pack_changed_bodies(variant)
        init = (0, min(int(level_init_max), levels - 1))
        cfg.events["pack_mass_ramp"] = EventTermCfg(
            func=microduck_mdp.pack_mass_ramp, mode="reset",
            params=dict(base_inertials=base, ramp_steps=0, asset_cfg=SceneEntityCfg("robot"), levels=int(levels),
                        level_init=init, gate_speed=float(gate_speed), min_fwd_s=float(min_fwd_s)))
        assert list(cfg.events)[-1] == "pack_mass_ramp"
        cfg.events["pack_level_track"] = EventTermCfg(func=microduck_mdp.pack_level_track, mode="step",
                                                      params=dict(command_name="twist"))
        rec["mass_levels"] = dict(levels=int(levels), fractions=[round(k / (levels - 1), 4) for k in range(levels)],
                                  level_init=list(init), gate_speed=float(gate_speed), min_fwd_s=float(min_fwd_s),
                                  forward_tick="command vx >= 0.1", base_variant=PACK_VARIANTS[variant],
                                  bodies=sorted(base),
                                  rule="episode end, envs with >= min_fwd_s of forward command: up one if no fall and "
                                       "mean speed along command >= gate; down one if fell or speed < gate/2; past "
                                       "the top redrawn uniformly over all levels; others keep")
    return rec


def apply_cmd_progress(cfg, weight: float, min_speed: float = 0.1) -> dict:
    """Add the command-progress reward term (mdp.cmd_progress) at `weight` per s; weight 0 leaves the cfg untouched."""
    if not weight:
        return {}
    from mjlab.managers import RewardTermCfg
    from . import mdp as microduck_mdp
    cfg.rewards["cmd_progress"] = RewardTermCfg(func=microduck_mdp.cmd_progress, weight=float(weight),
                                                params=dict(command_name="twist", min_speed=float(min_speed)))
    return dict(weight=float(weight), min_speed=float(min_speed),
                value="clip(v along command / commanded speed, 0, 1) on ticks with |planar command| > min_speed")


def apply_upright_progress_gate(cfg, window_s: float = 1.0, v_ref: float = 0.10, terms=("upright",),
                                min_speed: float = 0.1) -> dict:
    """Wrap each named reward term in mdp.upright_progress_gate: under a walk command (planar command > min_speed) its
    positive part is multiplied by clip(mean speed along the command over the last window_s / v_ref, 0, 1); at zero
    command it is unchanged. Weights and the wrapped term's params are kept."""
    from mjlab.managers import RewardTermCfg
    from . import mdp as microduck_mdp
    assert window_s > 0 and v_ref > 0, (window_s, v_ref)
    done = []
    for name in terms:
        old = cfg.rewards[name]
        assert old.func is not microduck_mdp.upright_progress_gate, f"{name}: already gated"
        params = dict(old.params)
        assert not any(k.startswith("gate_") for k in params), name
        params.update(gate_inner=old.func, gate_command="twist", gate_window_s=float(window_s),
                      gate_v_ref=float(v_ref), gate_min_speed=float(min_speed))
        cfg.rewards[name] = RewardTermCfg(func=microduck_mdp.upright_progress_gate, weight=old.weight, params=params)
        done.append(dict(term=name, weight=old.weight, inner=getattr(old.func, "__name__", str(old.func))))
    return dict(terms=done, window_s=float(window_s), v_ref=float(v_ref), min_speed=float(min_speed),
                rule="under a walk command the positive part is multiplied by clip(mean speed along the command over "
                     "the last window_s / v_ref, 0, 1); unchanged at zero command")


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
