"""CMU variants of Pollen's ground grasp and static grasp (upstream branch static-grasp, 2026-10-26).

`make_cmu_grasp_env_cfg(base, variant, ball=...)` takes one of Pollen's grasp env factories and changes two things:

- The robot carries the CMU as-built masses (`cmu/model`, `variant_parts(variant)`), as in `cmu_asbuilt.py`.
- With `ball=True` the object is the one Shawn will put in front of the robot (2026-10-26): a 3D-printed orange ball
  that is not quite round, 1.0-1.5 cm across. An ellipsoid whose three radii are drawn independently per episode
  (5.3-7.2 mm), 0.5-2.5 g (printed PLA), friction 0.35-1.1: PLA ball on the printed PLA lip pads at the low end,
  a rubber-lined beak (bike inner tube) at the high end. Pollen's objects are 1.4-2.6 cm boxes at friction 1.1-1.6
  for soft TPU pads, which our beak does not have.

Rewards, timeline, actor contract (61-D, blind) and the 14-action output are Pollen's.
"""
from __future__ import annotations

import dataclasses

import mujoco
from mjlab.entity import EntityCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp import dr

from .cmu_asbuilt import load_apply_masses

BALL_RADIUS = 0.00625                 # nominal 1.25 cm ball
BALL_RADIUS_SCALE = (0.85, 1.15)      # per axis, independent: 1.06-1.44 cm across, up to 1.35 : 1 lumpy
BALL_MASS = 0.0013                    # solid PLA 1.25 cm sphere ~1.3 g; printed parts are lighter
BALL_MASS_ALPHA = (-0.48, 0.33)       # pseudo_inertia: mass x e^(2 alpha) -> 0.5-2.5 g
BALL_FRICTION = (0.35, 1.1)


def get_ball_object_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="object", pos=[0.15, 0.0, BALL_RADIUS])
    body.add_freejoint(name="object_free")
    body.add_geom(name="object_geom", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[BALL_RADIUS] * 3,
                  mass=BALL_MASS, rgba=[1.0, 0.45, 0.05, 1.0], condim=4, friction=[0.7, 0.02, 0.0001])
    return spec


BALL_OBJECT_CFG = EntityCfg(
    spec_fn=get_ball_object_spec,
    init_state=EntityCfg.InitialStateCfg(pos=(0.15, 0.0, BALL_RADIUS)),
)


def make_cmu_grasp_env_cfg(base_factory, variant: str = "plugged", ball: bool = True, **kwargs):
    cfg = base_factory(**kwargs)
    apply_masses = load_apply_masses()
    parts = apply_masses.variant_parts(variant)
    robot = cfg.scene.entities["robot"]
    base_spec_fn = robot.spec_fn

    def spec_fn():
        spec = base_spec_fn()
        report = apply_masses.apply_parts_to_spec(spec, parts)
        print(f"[cmu:{variant}] total mass {report['total_before_g']:.1f} g -> {report['total_after_g']:.1f} g")
        return spec

    entities = {**cfg.scene.entities, "robot": dataclasses.replace(robot, spec_fn=spec_fn)}
    if ball:
        entities["object"] = BALL_OBJECT_CFG
        obj = SceneEntityCfg("object", geom_names=("object_geom",))
        cfg.events["object_size"] = EventTermCfg(
            func=dr.geom_size, mode="reset",
            params={"asset_cfg": obj, "operation": "scale", "ranges": BALL_RADIUS_SCALE},
        )
        cfg.events["object_mass"].params["alpha_range"] = BALL_MASS_ALPHA
        cfg.events["object_friction"].params["ranges"] = BALL_FRICTION
    cfg.scene.entities = entities
    return cfg
