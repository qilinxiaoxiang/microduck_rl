import dataclasses as _dc
from mjlab_microduck.train_hook import maybe_submit_to_hf_jobs

# `train <task> ... --hf-jobs` submits to HF Jobs and exits here, before any
# of the cfg imports below: this module is what mjlab's plugin loader pulls
# in, and it is the only train path no install order can take from us (see
# train_hook.py). A no-op without the flag.
maybe_submit_to_hf_jobs()

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class MicroduckOnPolicyRunner(VelocityOnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        # resolve_symmetry_config injects _env into train_cfg["algorithm"]["symmetry_cfg"]
        # in-place, sharing the same dict object with self.alg.symmetry.  Replace the
        # train_cfg reference with a copy that omits _env so dump_yaml can serialize the
        # config (MjSpec is not picklable), without touching the PPO's internal reference.
        alg = train_cfg.get("algorithm", {})
        sym = alg.get("symmetry_cfg") if isinstance(alg, dict) else None
        if isinstance(sym, dict) and "_env" in sym:
            alg["symmetry_cfg"] = {k: v for k, v in sym.items() if k != "_env"}


from .microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    MicroduckRlCfg,
)
from .microduck_standup_env_cfg import (
    make_microduck_standup_env_cfg,
    MicroduckStandUpRlCfg,
)
from .microduck_velstand_env_cfg import (
    make_microduck_velstand_env_cfg,
    MicroduckVelStandRlCfg,
)
from .microduck_ground_pick_env_cfg import (
    make_microduck_ground_pick_env_cfg,
    MicroduckGroundPickRlCfg,
)
from .microduck_ball_kick_env_cfg import (
    make_microduck_ball_kick_env_cfg,
    MicroduckBallKickRlCfg,
)
from .microduck_sitstand_env_cfg import (
    make_microduck_sitstand_env_cfg,
    MicroduckSitStandRlCfg,
)
from .microduck_velocity_rollers_env_cfg import (
    make_microduck_velocity_rollers_env_cfg,
    MicroduckRollersRlCfg,
)
from .microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
    MicroduckSwizzleRlCfg,
)
from .microduck_roller_crouch_env_cfg import (
    make_microduck_roller_crouch_env_cfg,
    MicroduckRollerCrouchRlCfg,
)
from .microduck_roller_slope_env_cfg import (
    make_microduck_roller_slope_env_cfg,
    MicroduckRollerSlopeRlCfg,
)
from .microduck_roller_standup_env_cfg import (
    make_microduck_roller_standup_env_cfg,
    MicroduckRollerStandUpRlCfg,
)
from .microduck_spin_env_cfg import (
    make_microduck_spin_env_cfg,
    MicroduckSpinRlCfg,
)
from .microduck_roulade_env_cfg import (
    make_microduck_roulade_env_cfg,
    MicroduckRouladeRlCfg,
)
from .backlash import make_backlash_variant

# Standard velocity task
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(rough=True),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# VelStand — walking + fall recovery + body pose control in one policy.
register_mjlab_task(
    task_id="Mjlab-VelStand-Flat-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-VelStand-Rough-MicroDuck",
    env_cfg=make_microduck_velstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_velstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckVelStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Stand-up task — robot starts inverted (lying on back) and must stand up
register_mjlab_task(
    task_id="Mjlab-StandUp-Flat-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(),
    play_env_cfg=make_microduck_standup_env_cfg(play=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-StandUp-Rough-MicroDuck",
    env_cfg=make_microduck_standup_env_cfg(rough=True),
    play_env_cfg=make_microduck_standup_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# SitStand task — commanded sit ↔ stand in one policy, gently, head commandable
register_mjlab_task(
    task_id="Mjlab-SitStand-Flat-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-SitStand-Rough-MicroDuck",
    env_cfg=make_microduck_sitstand_env_cfg(rough=True),
    play_env_cfg=make_microduck_sitstand_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckSitStandRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Ground-pick task — crouch, touch the ground with the mouth tip, return to stand
register_mjlab_task(
    task_id="Mjlab-GroundPick-Flat-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# BallKick task — kick a 70mm/15g ball forward hard with the right foot from a
# standing start (flat terrain only — a ball on rough terrain is another task).
register_mjlab_task(
    task_id="Mjlab-BallKick-Flat-MicroDuck",
    env_cfg=make_microduck_ball_kick_env_cfg(),
    play_env_cfg=make_microduck_ball_kick_env_cfg(play=True),
    rl_cfg=MicroduckBallKickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundPick-Rough-MicroDuck",
    env_cfg=make_microduck_ground_pick_env_cfg(rough=True),
    play_env_cfg=make_microduck_ground_pick_env_cfg(play=True, rough=True),
    rl_cfg=MicroduckGroundPickRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller skate velocity task (passive-wheel model; historical task id kept)
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck-Rollers",
    env_cfg=make_microduck_velocity_rollers_env_cfg(),
    play_env_cfg=make_microduck_velocity_rollers_env_cfg(play=True),
    rl_cfg=MicroduckRollersRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller SWIZZLE task — clean classic swizzle (symmetric, feet grounded).
register_mjlab_task(
    task_id="Mjlab-Velocity-Swizzle-MicroDuck",
    env_cfg=make_microduck_velocity_swizzle_env_cfg(),
    play_env_cfg=make_microduck_velocity_swizzle_env_cfg(play=True),
    rl_cfg=MicroduckSwizzleRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerCrouch-Flat-MicroDuck",
    env_cfg=make_microduck_roller_crouch_env_cfg(),
    play_env_cfg=make_microduck_roller_crouch_env_cfg(play=True),
    rl_cfg=MicroduckRollerCrouchRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-RollerSlope-Flat-MicroDuck",
    env_cfg=make_microduck_roller_slope_env_cfg(),
    play_env_cfg=make_microduck_roller_slope_env_cfg(play=True),
    rl_cfg=MicroduckRollerSlopeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roller STANDUP — se relever sur rollers (policy dédiée, départ au sol).
register_mjlab_task(
    task_id="Mjlab-RollerStandUp-Flat-MicroDuck",
    env_cfg=make_microduck_roller_standup_env_cfg(),
    play_env_cfg=make_microduck_roller_standup_env_cfg(play=True),
    rl_cfg=MicroduckRollerStandUpRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Spin task — rotation rapide sur place, sur rollers (slot ground-pick).
register_mjlab_task(
    task_id="Mjlab-Spin-Flat-MicroDuck",
    env_cfg=make_microduck_spin_env_cfg(),
    play_env_cfg=make_microduck_spin_env_cfg(play=True),
    rl_cfg=MicroduckSpinRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Roulade — forward roll over the flat head top, land back on the feet.
register_mjlab_task(
    task_id="Mjlab-Roulade-Flat-MicroDuck",
    env_cfg=make_microduck_roulade_env_cfg(),
    play_env_cfg=make_microduck_roulade_env_cfg(play=True),
    rl_cfg=MicroduckRouladeRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Backlash variants — ±1° serial gear play per servo + encoder-through-backlash
# actuator feedback and joint obs (see tasks/backlash.py). Each family keeps its
# base task's collision model: Velocity → robot_walk_backlash.xml,
# VelStand → robot_allcollisions_backlash.xml, StandUp → robot_groundcontact_backlash.xml. Obs/action dims are
# unchanged vs the base tasks.
from mjlab_microduck.robot.microduck_constants import (
    MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG,
    MICRODUCK_BACKLASH_ROBOT_CFG,
    MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG,
    MICRODUCK_WALK_BACKLASH_ROBOT_CFG,
)

# (task_id, make_fn, make_kwargs, rl_cfg, backlash robot cfg). Task ids mirror
# the base ids with "-Backlash" inserted. Walk-model tasks get the walk
# backlash robot, roller tasks the wheels+backlash robot, the rest the
# groundcontact backlash robot — same model as their base task in each case.
_BL_GROUNDCONTACT = MICRODUCK_BACKLASH_ROBOT_CFG
_BL_WALK = MICRODUCK_WALK_BACKLASH_ROBOT_CFG
_BL_ROLLERS = MICRODUCK_ROLLERS_BACKLASH_ROBOT_CFG
_BL_ALLCOLLISIONS = MICRODUCK_ALLCOLLISIONS_BACKLASH_ROBOT_CFG
_BACKLASH_TASKS = (
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-Velocity-Rough-Backlash-MicroDuck", make_microduck_velocity_env_cfg, {"rough": True}, MicroduckRlCfg, _BL_WALK),
    ("Mjlab-VelStand-Flat-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-VelStand-Rough-Backlash-MicroDuck", make_microduck_velstand_env_cfg, {"rough": True}, MicroduckVelStandRlCfg, _BL_ALLCOLLISIONS),
    ("Mjlab-StandUp-Flat-Backlash-MicroDuck", make_microduck_standup_env_cfg, {}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-StandUp-Rough-Backlash-MicroDuck", make_microduck_standup_env_cfg, {"rough": True}, MicroduckStandUpRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Flat-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-SitStand-Rough-Backlash-MicroDuck", make_microduck_sitstand_env_cfg, {"rough": True}, MicroduckSitStandRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Flat-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-GroundPick-Rough-Backlash-MicroDuck", make_microduck_ground_pick_env_cfg, {"rough": True}, MicroduckGroundPickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-BallKick-Flat-Backlash-MicroDuck", make_microduck_ball_kick_env_cfg, {}, MicroduckBallKickRlCfg, _BL_GROUNDCONTACT),
    ("Mjlab-Velocity-Flat-Backlash-MicroDuck-Rollers", make_microduck_velocity_rollers_env_cfg, {}, MicroduckRollersRlCfg, _BL_ROLLERS),
    ("Mjlab-Velocity-Swizzle-Backlash-MicroDuck", make_microduck_velocity_swizzle_env_cfg, {}, MicroduckSwizzleRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerCrouch-Flat-Backlash-MicroDuck", make_microduck_roller_crouch_env_cfg, {}, MicroduckRollerCrouchRlCfg, _BL_ROLLERS),
    ("Mjlab-RollerSlope-Flat-Backlash-MicroDuck", make_microduck_roller_slope_env_cfg, {}, MicroduckRollerSlopeRlCfg, _BL_ROLLERS),
)
for _task_id, _make_cfg, _kw, _rl_cfg, _robot_cfg in _BACKLASH_TASKS:
    register_mjlab_task(
        task_id=_task_id,
        env_cfg=make_backlash_variant(_make_cfg(**_kw), _robot_cfg),
        play_env_cfg=make_backlash_variant(_make_cfg(play=True, **_kw), _robot_cfg),
        rl_cfg=_rl_cfg,
        runner_cls=MicroduckOnPolicyRunner,
    )

# CMU as-built mass variants (cmu/model): the velocity recipe on the measured robot, -CMU-Plugged / -CMU-Battery.
from .cmu_asbuilt import register_cmu_tasks
register_cmu_tasks(MicroduckOnPolicyRunner)

# CMU aiming crouch (cmu_aim_pick_env_cfg.py): the ground pick with an object the beak has to reach.
from .cmu_aim_pick_env_cfg import register_aim_pick_tasks
register_aim_pick_tasks(MicroduckOnPolicyRunner)

# ── Ground grasp (2026-10-10): the ground-pick slot with an opening beak + a real object ──
from pathlib import Path as _Path

from .microduck_ground_grasp_env_cfg import (
    make_microduck_ground_grasp_env_cfg,
    MicroduckGroundGraspRlCfg,
)


class GroundGraspFromPickRunner(MicroduckOnPolicyRunner):
    """Warm start from TODAY's ground-pick move (Rémi's suggestion): the actor MLP and its observation normalizer are
    copied from the deployed ONNX (same 61 -> 512 -> 256 -> 128 -> 14 ELU network; the exporter bakes
    (obs - mean) / std into it). The critic stays fresh (its observations differ: object, mouth, contacts), the
    exploration std starts at INIT_STD so the inherited move is not drowned in noise, and the normalizer count is huge
    so the inherited statistics stay put. A real resume (--agent.resume) loads over this as usual."""

    INIT_ONNX = _Path(__file__).parent / "init_policies" / "alpha_ground_pick.onnx"
    INIT_STD = 0.25
    # A FRESH Adam moves every weight by ~lr on its first steps (m/sqrt(v) ~ sign(g)): at the cfg's 1e-3 that shifted
    # the inherited actions by ~1 rad in ONE update (measured on the smoke run). Start low; the adaptive KL schedule
    # raises it (x1.5 per update while KL < desired/2) once the moments are warm.
    INIT_LR = 1e-4

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        self._init_actor_from_onnx()

    def _init_actor_from_onnx(self):
        import onnx
        import torch
        from onnx import numpy_helper

        init = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(self.INIT_ONNX)).graph.initializer}
        actor = self.alg.actor
        sd = actor.state_dict()
        std_key = next(k for k in init if k.startswith("onnx::Div"))
        std = init[std_key] - actor.obs_normalizer.eps      # the exporter bakes (obs - mean) / (std + eps)
        new = {
            "obs_normalizer._mean": init["obs_normalizer._mean"],
            "obs_normalizer._std": std,
            "obs_normalizer._var": std ** 2,
        }
        for k in ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias",
                  "mlp.6.weight", "mlp.6.bias"):
            new[k] = init[k]
        for k, v in new.items():
            t = torch.as_tensor(v, dtype=sd[k].dtype, device=sd[k].device).reshape(sd[k].shape)
            sd[k].copy_(t)
        sd["obs_normalizer.count"].fill_(1e9)
        sd["distribution.std_param"].fill_(self.INIT_STD)
        actor.load_state_dict(sd)
        self.alg.learning_rate = self.INIT_LR
        for g in self.alg.optimizer.param_groups:
            g["lr"] = self.INIT_LR
        print(f"[ground_grasp] actor initialised from {self.INIT_ONNX.name} (std {self.INIT_STD}, lr {self.INIT_LR})")


register_mjlab_task(
    task_id="Mjlab-GroundGrasp-Flat-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-GroundGrasp-FromPick-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Variant: same, plus a Gaussian pull of the beak pitch toward 60° in the grasp window (the static pose study's sweet
# spot: both lips < 1 cm from the floor) — for when the task alone keeps today's straight-down beak.
register_mjlab_task(
    task_id="Mjlab-GroundGrasp-FromPickPitch60-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(beak_pitch_pref_weight=3.0, beak_pitch_target=60.0),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True, beak_pitch_pref_weight=3.0, beak_pitch_target=60.0),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Tall objects standing up (small toys, 3-5 cm): the beak takes the top slice. Fine-tuned from a floor-grasp checkpoint.
register_mjlab_task(
    task_id="Mjlab-GroundGrasp-Tall-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(tall=True),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True, tall=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)

# Flat objects (2x2-Lego-like, 0.8-1.2 cm tall): no bisector preference, RL picks the beak angle.
register_mjlab_task(
    task_id="Mjlab-GroundGrasp-Flat-Objects-MicroDuck",
    env_cfg=make_microduck_ground_grasp_env_cfg(flat=True),
    play_env_cfg=make_microduck_ground_grasp_env_cfg(play=True, flat=True),
    rl_cfg=MicroduckGroundGraspRlCfg,
    runner_cls=MicroduckOnPolicyRunner,
)


# ── Static grasp (2026-10-26): the ground-grasp move held in a statically stable crouch, beak 1-2 mm off the floor ──
import copy as _copy

from .microduck_static_grasp_env_cfg import make_microduck_static_grasp_env_cfg

MicroduckStaticGraspRlCfg = _copy.deepcopy(MicroduckGroundGraspRlCfg)
MicroduckStaticGraspRlCfg.experiment_name = "static_grasp"
MicroduckStaticGraspRlCfg.run_name = "static_grasp"

register_mjlab_task(
    task_id="Mjlab-StaticGrasp-FromPick-MicroDuck",
    env_cfg=make_microduck_static_grasp_env_cfg(),
    play_env_cfg=make_microduck_static_grasp_env_cfg(play=True),
    rl_cfg=MicroduckStaticGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Strict: beak contact and a CoM past the toes cost far more (run 1 learned to lean on its beak), wider pose / lip pulls.
register_mjlab_task(
    task_id="Mjlab-StaticGrasp-Strict-FromPick-MicroDuck",
    env_cfg=make_microduck_static_grasp_env_cfg(strict=True),
    play_env_cfg=make_microduck_static_grasp_env_cfg(play=True, strict=True),
    rl_cfg=MicroduckStaticGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Retimed: the beak arrives by phase 0.24, holds still and low until 0.34, bites 0.34-0.40 (run 2 never hovered).
register_mjlab_task(
    task_id="Mjlab-StaticGrasp-Retimed-FromPick-MicroDuck",
    env_cfg=make_microduck_static_grasp_env_cfg(retimed=True),
    play_env_cfg=make_microduck_static_grasp_env_cfg(play=True, retimed=True),
    rl_cfg=MicroduckStaticGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# Hover: retimed + level head + episode ends if the beak presses the floor (> 0.5 N) during the hold (run 3 rested on it).
register_mjlab_task(
    task_id="Mjlab-StaticGrasp-Hover-FromPick-MicroDuck",
    env_cfg=make_microduck_static_grasp_env_cfg(hover=True),
    play_env_cfg=make_microduck_static_grasp_env_cfg(play=True, hover=True),
    rl_cfg=MicroduckStaticGraspRlCfg,
    runner_cls=GroundGraspFromPickRunner,
)

# CMU grasp variants (cmu_grasp_env_cfg.py): Pollen's ground / static grasp on the CMU as-built masses, with the
# 1.0-1.5 cm lumpy printed ball (-Ball-) Shawn tests on the robot.
from .cmu_grasp_env_cfg import make_cmu_grasp_env_cfg

for _base_name, _factory, _kw, _rl in (
    ("GroundGrasp-FromPick", make_microduck_ground_grasp_env_cfg, {}, MicroduckGroundGraspRlCfg),
    ("StaticGrasp-Hover-FromPick", make_microduck_static_grasp_env_cfg, {"hover": True}, MicroduckStaticGraspRlCfg),
):
    for _ball in (True, False):
        register_mjlab_task(
            task_id=f"Mjlab-{_base_name}-MicroDuck-CMU-Plugged{'-Ball' if _ball else ''}",
            env_cfg=make_cmu_grasp_env_cfg(_factory, "plugged", ball=_ball, **_kw),
            play_env_cfg=make_cmu_grasp_env_cfg(_factory, "plugged", ball=_ball, play=True, **_kw),
            rl_cfg=_rl,
            runner_cls=GroundGraspFromPickRunner,
        )

# CMU seeing pick (cmu_seeing_pick_env_cfg.py): the static hover grasp with the object seen by the head camera in
# body slots 55-60, several object shapes, a placement window and mid-descent kicks (cmu/grasp/PLAN-seeing-pickup.md).
from .cmu_seeing_pick_env_cfg import (
    BODY_SLOT_MEAN as _SP_MEAN,
    BODY_SLOT_START as _SP_START,
    BODY_SLOT_STD as _SP_STD,
    make_seeing_pick_env_cfg,
    seeing_pick_rl_cfg,
)


class SeeingPickFromPickRunner(GroundGraspFromPickRunner):
    """The ground-pick warm start, plus the body slots: the inherited network saw zeros there (normaliser std 0.01),
    so their mean / std are set to the camera values' scale and their first-layer columns start at zero. At
    iteration 0 the policy therefore moves exactly like alpha_ground_pick and learns to use the slots."""

    def _init_actor_from_onnx(self):
        super()._init_actor_from_onnx()
        import torch

        actor = self.alg.actor
        sl = slice(_SP_START, _SP_START + len(_SP_MEAN))
        with torch.no_grad():
            norm = actor.obs_normalizer
            norm._mean[..., sl] = torch.tensor(_SP_MEAN, device=norm._mean.device)
            norm._std[..., sl] = torch.tensor(_SP_STD, device=norm._std.device)
            norm._var[..., sl] = norm._std[..., sl] ** 2
            actor.mlp[0].weight[:, sl] = 0.0
        print(f"[seeing_pick] body slots {sl.start}-{sl.stop - 1}: normaliser reset, first-layer columns zeroed")


register_mjlab_task(
    task_id="Mjlab-SeeingPick-FromPick-MicroDuck-CMU-Plugged",
    env_cfg=make_seeing_pick_env_cfg("plugged"),
    play_env_cfg=make_seeing_pick_env_cfg("plugged", play=True),
    rl_cfg=seeing_pick_rl_cfg(MicroduckStaticGraspRlCfg),
    runner_cls=SeeingPickFromPickRunner,
)

# CMU seeing pick that watches the picture (cmu_seeing_pix_env_cfg.py, T2): body slots 55-60 are the detection's
# (u, v, width, age, holding, visible) instead of a floor position. Warm start from the trained T2 export spx-b_1800
# (cmu/planning/fetch-20261103 task D): its normaliser already carries the pixel slots' statistics, so the ONNX init
# is copied unchanged. Plugged reproduces the source run's body; Battery-20261104 is the dated, physically
# unqualified battery model, whose head/beak contact geometry is still the walking baseline's.
from .cmu_seeing_pix_env_cfg import seeing_pix_rl_cfg
from .cmu_seeing_pix_rise_env_cfg import make_seeing_pix_handoff_env_cfg, seeing_pix_rise_rl_cfg


class SeeingPixFromSpxRunner(GroundGraspFromPickRunner):
    INIT_ONNX = _Path(__file__).parent / "init_policies" / "spx-b_1800.onnx"


# SeeingPickPix: the source T2 task (+ hand-over logging). SeeingPickPixRise: the same task whose main term pays the
# hold after the rise only in stand61's admission state (cmu_seeing_pix_rise_env_cfg.py), the task D comparison.
for _suffix, _variant in (("Plugged", "plugged"), ("Battery-20261104", "battery-20261104")):
    for _name, _rise, _rl in (("SeeingPickPix", False, seeing_pix_rl_cfg),
                              ("SeeingPickPixRise", True, seeing_pix_rise_rl_cfg)):
        register_mjlab_task(
            task_id=f"Mjlab-{_name}-FromSpx-MicroDuck-CMU-{_suffix}",
            env_cfg=make_seeing_pix_handoff_env_cfg(_variant, rise=_rise),
            play_env_cfg=make_seeing_pix_handoff_env_cfg(_variant, play=True, rise=_rise),
            rl_cfg=_rl(MicroduckStaticGraspRlCfg),
            runner_cls=SeeingPixFromSpxRunner,
        )

# Kneel-rise (cmu_kneel_rise_env_cfg.py, 2026-11-18): the open-loop half of the split pick. Blind (holding bit only),
# docked starts, the phase held at 0.90 and the episode run 2.5 s past the hand-over, neck-rate cost while holding.
# Warm start from rise-s1 @ 999 with the blinded slots' first-layer columns zeroed.
from .cmu_kneel_rise_env_cfg import BLIND_ZERO as _KR_ZERO
from .cmu_kneel_rise_env_cfg import INIT_ONNX as _KR_INIT
from .cmu_kneel_rise_env_cfg import DOCKFIT_C as _KR_DOCKFIT_C
from .cmu_kneel_rise_env_cfg import kneel_rise_rl_cfg, make_kneel_rise_env_cfg


class KneelRiseFromRiseRunner(GroundGraspFromPickRunner):
    """Plus a critic warm-up: for the first CRITIC_WARMUP_ITERS updates the actor is frozen and only the fresh critic
    trains (at CRITIC_WARMUP_LR). Run kr-a (no warm-up) lost the inherited bite within ~60 iterations (held at the
    hand-over 0.29 -> 0) to updates driven by an untrained critic, and spent ~400 iterations relearning it."""

    INIT_ONNX = _KR_INIT
    CRITIC_WARMUP_ITERS = 50
    CRITIC_WARMUP_LR = 1e-3

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        update, alg, n = self.alg.update, self.alg, [0]
        actor_params = list(alg.actor.parameters())

        def warm_update():
            warm = n[0] < self.CRITIC_WARMUP_ITERS
            if warm:
                for prm in actor_params:
                    prm.requires_grad_(False)
                self._set_lr(self.CRITIC_WARMUP_LR)
            try:
                out = update()
            finally:
                for prm in actor_params:
                    prm.requires_grad_(True)
            n[0] += 1
            if warm:
                self._set_lr(self.INIT_LR)   # the adaptive schedule raised it on KL = 0; restart from the init lr
                if n[0] == self.CRITIC_WARMUP_ITERS:
                    print(f"[kneel_rise] critic warm-up done after {n[0]} updates: actor unfrozen")
            return out

        self.alg.update = warm_update

    def _set_lr(self, lr):
        self.alg.learning_rate = lr
        for g in self.alg.optimizer.param_groups:
            g["lr"] = lr

    def _init_actor_from_onnx(self):
        super()._init_actor_from_onnx()
        import torch

        cols = [_SP_START + i for i in _KR_ZERO]
        with torch.no_grad():
            self.alg.actor.mlp[0].weight[:, cols] = 0.0
        print(f"[kneel_rise] blinded body slots {cols}: first-layer columns zeroed")


for _suffix, _variant in (("Plugged", "plugged"), ("Battery-20261104", "battery-20261104")):
    register_mjlab_task(
        task_id=f"Mjlab-KneelRise-FromRise-MicroDuck-CMU-{_suffix}",
        env_cfg=make_kneel_rise_env_cfg(_variant),
        play_env_cfg=make_kneel_rise_env_cfg(_variant, play=True),
        rl_cfg=kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg),
        runner_cls=KneelRiseFromRiseRunner,
    )



class KneelRiseSightedFromRiseRunner(KneelRiseFromRiseRunner):
    """The sighted variant: rise-s1's camera columns are kept. rise-s1 untrained is already the best kneel (held at end
    0.67 / 0.49 / 0.34 at +-1 / 3 / 5 cm, 7-9 % falls), and ks-a (adaptive lr, entropy 0.01, warm-up 50) degraded it
    to 0.50 / 0.29 / 0.16 by iteration 250 while the action std grew 0.25 -> 0.31. Hence a longer warm-up and a low
    lr; launch with --agent.algorithm.schedule fixed --agent.algorithm.entropy-coef 0.0."""

    INIT_LR = 3e-5
    CRITIC_WARMUP_ITERS = 100

    def _init_actor_from_onnx(self):
        GroundGraspFromPickRunner._init_actor_from_onnx(self)


for _suffix, _variant in (("Plugged", "plugged"), ("Battery-20261104", "battery-20261104")):
    register_mjlab_task(
        task_id=f"Mjlab-KneelRise-Sighted-FromRise-MicroDuck-CMU-{_suffix}",
        env_cfg=make_kneel_rise_env_cfg(_variant, sighted=True),
        play_env_cfg=make_kneel_rise_env_cfg(_variant, play=True, sighted=True),
        rl_cfg=kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, sighted=True),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )
    # the sighted kneel on the dock's end states (near side included); warm start from ks-f's checkpoint
    register_mjlab_task(
        task_id=f"Mjlab-KneelRise-DockFit-MicroDuck-CMU-{_suffix}",
        env_cfg=make_kneel_rise_env_cfg(_variant, dockfit=True),
        play_env_cfg=make_kneel_rise_env_cfg(_variant, play=True, dockfit=True),
        rl_cfg=kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )
    # ... and on the ball alone (the fetch target); warm start from ks-g's checkpoint
    register_mjlab_task(
        task_id=f"Mjlab-KneelRise-DockFit-Ball-MicroDuck-CMU-{_suffix}",
        env_cfg=make_kneel_rise_env_cfg(_variant, dockfit=True, ball_only=True),
        play_env_cfg=make_kneel_rise_env_cfg(_variant, play=True, dockfit=True, ball_only=True),
        rl_cfg=kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )
    # ... on bd-c's ends (the forward dock); warm start from ks-h's checkpoint
    register_mjlab_task(
        task_id=f"Mjlab-KneelRise-DockFitC-Ball-MicroDuck-CMU-{_suffix}",
        env_cfg=make_kneel_rise_env_cfg(_variant, dockfit=True, ball_only=True, dock_fit=_KR_DOCKFIT_C),
        play_env_cfg=make_kneel_rise_env_cfg(_variant, play=True, dockfit=True, ball_only=True,
                                             dock_fit=_KR_DOCKFIT_C),
        rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
                           experiment_name="kneel_rise_ball_c", run_name="kneel_rise_ball_c"),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )

# Control for the sighted kneel's degradation (2026-11-18, RESULTS.md round 4): the same fine-tune pipeline (ONNX
# init, fresh critic, fixed lr, entropy 0, curricula pinned at rise-s1's step) in rise-s1's OWN task, start window at
# its operating value. Diagnostic only.
from .cmu_kneel_rise_env_cfg import SOURCE_STEP as _KR_SRC_STEP
from .cmu_kneel_rise_env_cfg import pin_curricula_at as _pin_curricula_at


def _make_rise_control_cfg(variant, play=False):
    from . import cmu_seeing_pix_env_cfg as _pix

    cfg = make_seeing_pix_handoff_env_cfg(variant, play=play, rise=True)
    _pin_curricula_at(cfg, _KR_SRC_STEP)
    cfg.events["reset_object"].params["window"] = _pix.PIX_WINDOW_MAX
    cfg.curriculum.pop("pix_window", None)
    return cfg


register_mjlab_task(
    task_id="Mjlab-SeeingPickPixRise-Ctl-MicroDuck-CMU-Plugged",
    env_cfg=_make_rise_control_cfg("plugged"),
    play_env_cfg=_make_rise_control_cfg("plugged", play=True),
    rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, sighted=True), experiment_name="kneel_rise_ctl",
                       run_name="kneel_rise_ctl"),
    runner_cls=KneelRiseSightedFromRiseRunner,
)

# Ball dock (cmu_ball_dock_env_cfg.py, 2026-11-18): the closed-loop half of the split pick. A commander (vx, vy, wz,
# head pitch / yaw) over the frozen deployed walk61, seeing the ball through the U20CAM model, rewarded for standing
# still with the ball at the kneel-rise landing spot.
from .cmu_ball_dock_env_cfg import NUDGE_LOW_ONNX as _BD_NUDGE
from .cmu_ball_dock_env_cfg import DOCK_TARGET_DX as _BD_TARGET_DX
from .cmu_ball_dock_env_cfg import BallDockRlCfg, make_ball_dock_env_cfg

for _suffix, _variant in (("Plugged", "plugged"), ("Battery-20261104", "battery-20261104")):
    register_mjlab_task(
        task_id=f"Mjlab-BallDock-Walk61-MicroDuck-CMU-{_suffix}",
        env_cfg=make_ball_dock_env_cfg(_variant),
        play_env_cfg=make_ball_dock_env_cfg(_variant, play=True),
        rl_cfg=BallDockRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )
    register_mjlab_task(
        task_id=f"Mjlab-BallDock-Nudge-MicroDuck-CMU-{_suffix}",
        env_cfg=make_ball_dock_env_cfg(_variant, walk_onnx=_BD_NUDGE),
        play_env_cfg=make_ball_dock_env_cfg(_variant, play=True, walk_onnx=_BD_NUDGE),
        rl_cfg=_dc.replace(BallDockRlCfg, experiment_name="ball_dock_nudge", run_name="ball_dock_nudge",
                           max_iterations=2500),
        runner_cls=MicroduckOnPolicyRunner,
    )
    # bd-c: the dock over the nudge, target moved DOCK_TARGET_DX ahead; resume from bd-b's checkpoint
    register_mjlab_task(
        task_id=f"Mjlab-BallDock-Nudge-Fwd-MicroDuck-CMU-{_suffix}",
        env_cfg=make_ball_dock_env_cfg(_variant, walk_onnx=_BD_NUDGE, target_dx=_BD_TARGET_DX),
        play_env_cfg=make_ball_dock_env_cfg(_variant, play=True, walk_onnx=_BD_NUDGE, target_dx=_BD_TARGET_DX),
        rl_cfg=_dc.replace(BallDockRlCfg, experiment_name="ball_dock_fwd", run_name="ball_dock_fwd",
                           max_iterations=600),
        runner_cls=MicroduckOnPolicyRunner,
    )

# Walk nudge (cmu_walk_nudge_env_cfg.py, 2026-11-18): walk61 fine-tuned to step to a target 1-6 cm away and stop, the
# twist slots carrying gain x the remaining error. Warm start from walk61 with the critic warm-up.
from .cmu_walk_nudge_env_cfg import INIT_ONNX as _WN_INIT
from .cmu_walk_nudge_env_cfg import make_walk_nudge_env_cfg, walk_nudge_rl_cfg


class WalkNudgeFromWalk61Runner(KneelRiseSightedFromRiseRunner):
    INIT_ONNX = _WN_INIT


for _suffix, _variant in (("Plugged", "plugged"), ("Battery-20261104", "battery-20261104")):
    register_mjlab_task(
        task_id=f"Mjlab-WalkNudge-FromWalk61-MicroDuck-CMU-{_suffix}",
        env_cfg=make_walk_nudge_env_cfg(_variant),
        play_env_cfg=make_walk_nudge_env_cfg(_variant, play=True),
        rl_cfg=walk_nudge_rl_cfg(MicroduckRlCfg),
        runner_cls=WalkNudgeFromWalk61Runner,
    )
    register_mjlab_task(
        task_id=f"Mjlab-WalkNudge-Cam-FromWalk61-MicroDuck-CMU-{_suffix}",
        env_cfg=make_walk_nudge_env_cfg(_variant, camera=True),
        play_env_cfg=make_walk_nudge_env_cfg(_variant, play=True, camera=True),
        rl_cfg=walk_nudge_rl_cfg(MicroduckRlCfg, camera=True),
        runner_cls=WalkNudgeFromWalk61Runner,
    )


# Round 7 A/B: the bd-c-fitted ball kneel with the static-pose pull at LOW_POSE_WEIGHT (control: the task above)
register_mjlab_task(
    task_id="Mjlab-KneelRise-DockFitC-Ball-LowPose-MicroDuck-CMU-Plugged",
    env_cfg=make_kneel_rise_env_cfg("plugged", dockfit=True, ball_only=True, dock_fit=_KR_DOCKFIT_C, low_pose=True),
    play_env_cfg=make_kneel_rise_env_cfg("plugged", play=True, dockfit=True, ball_only=True, dock_fit=_KR_DOCKFIT_C,
                                         low_pose=True),
    rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
                       experiment_name="kneel_rise_ball_lowpose", run_name="kneel_rise_ball_lowpose"),
    runner_cls=KneelRiseSightedFromRiseRunner,
)

# Servo measurement noise (servo_noise.py, 2026-11-20): the chain's tasks with every joint reading at noise level 1
# (train) and 2 (stress test). The dock's SN1 training twin runs over wn-sn1 (the nudge retrained at level 1).
from .cmu_ball_dock_env_cfg import NUDGE_SN_ONNX as _BD_NUDGE_SN
from .servo_noise import apply_servo_noise as _sn, noisy_task_id as _sn_id

for _lvl in (1, 2):
    _base = "Mjlab-WalkNudge-FromWalk61-MicroDuck-CMU-Plugged"
    register_mjlab_task(
        task_id=_sn_id(_base, _lvl),
        env_cfg=_sn(make_walk_nudge_env_cfg("plugged"), _lvl),
        play_env_cfg=_sn(make_walk_nudge_env_cfg("plugged", play=True), _lvl),
        rl_cfg=_dc.replace(walk_nudge_rl_cfg(MicroduckRlCfg), experiment_name=f"walk_nudge_sn{_lvl}",
                           run_name=f"walk_nudge_sn{_lvl}"),
        runner_cls=WalkNudgeFromWalk61Runner,
    )
    _base = "Mjlab-BallDock-Nudge-MicroDuck-CMU-Plugged"       # dock evals (dock_check --walk-onnx picks the low level)
    register_mjlab_task(
        task_id=_sn_id(_base, _lvl),
        env_cfg=_sn(make_ball_dock_env_cfg("plugged", walk_onnx=_BD_NUDGE), _lvl),
        play_env_cfg=_sn(make_ball_dock_env_cfg("plugged", play=True, walk_onnx=_BD_NUDGE), _lvl),
        rl_cfg=_dc.replace(BallDockRlCfg, experiment_name=f"ball_dock_nudge_sn{_lvl}",
                           run_name=f"ball_dock_nudge_sn{_lvl}"),
        runner_cls=MicroduckOnPolicyRunner,
    )
    _base = "Mjlab-BallDock-Nudge-Fwd-MicroDuck-CMU-Plugged"   # the forward dock over wn-sn1
    register_mjlab_task(
        task_id=_sn_id(_base, _lvl),
        env_cfg=_sn(make_ball_dock_env_cfg("plugged", walk_onnx=_BD_NUDGE_SN, target_dx=_BD_TARGET_DX), _lvl),
        play_env_cfg=_sn(make_ball_dock_env_cfg("plugged", play=True, walk_onnx=_BD_NUDGE_SN,
                                                target_dx=_BD_TARGET_DX), _lvl),
        rl_cfg=_dc.replace(BallDockRlCfg, experiment_name=f"ball_dock_fwd_sn{_lvl}", run_name=f"ball_dock_fwd_sn{_lvl}",
                           max_iterations=600),
        runner_cls=MicroduckOnPolicyRunner,
    )
    _base = "Mjlab-KneelRise-Sighted-FromRise-MicroDuck-CMU-Plugged"   # kneel evals (kr_eval)
    register_mjlab_task(
        task_id=_sn_id(_base, _lvl),
        env_cfg=_sn(make_kneel_rise_env_cfg("plugged", sighted=True), _lvl),
        play_env_cfg=_sn(make_kneel_rise_env_cfg("plugged", play=True, sighted=True), _lvl),
        rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, sighted=True),
                           experiment_name=f"kneel_rise_sighted_sn{_lvl}", run_name=f"kneel_rise_sighted_sn{_lvl}"),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )
    _base = "Mjlab-KneelRise-DockFitC-Ball-MicroDuck-CMU-Plugged"      # the kneel on bd-c's ends, ball only
    register_mjlab_task(
        task_id=_sn_id(_base, _lvl),
        env_cfg=_sn(make_kneel_rise_env_cfg("plugged", dockfit=True, ball_only=True, dock_fit=_KR_DOCKFIT_C), _lvl),
        play_env_cfg=_sn(make_kneel_rise_env_cfg("plugged", play=True, dockfit=True, ball_only=True,
                                                 dock_fit=_KR_DOCKFIT_C), _lvl),
        rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
                           experiment_name=f"kneel_rise_ball_c_sn{_lvl}", run_name=f"kneel_rise_ball_c_sn{_lvl}"),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )

# bd-wide: the retry dock (wide spawns, half the resets) at servo noise level 1 over wn-sn1; resume from bd-sn1
register_mjlab_task(
    task_id="Mjlab-BallDock-Nudge-Fwd-Wide-SN1-MicroDuck-CMU-Plugged",
    env_cfg=_sn(make_ball_dock_env_cfg("plugged", walk_onnx=_BD_NUDGE_SN, target_dx=_BD_TARGET_DX, wide=True), 1),
    play_env_cfg=_sn(make_ball_dock_env_cfg("plugged", play=True, walk_onnx=_BD_NUDGE_SN, target_dx=_BD_TARGET_DX,
                                            wide=True), 1),
    rl_cfg=_dc.replace(BallDockRlCfg, experiment_name="ball_dock_wide_sn1", run_name="ball_dock_wide_sn1",
                       max_iterations=800),
    runner_cls=MicroduckOnPolicyRunner,
)

# Walk step (cmu_walk_step_env_cfg.py, 2026-11-22): a capped L1 correction on frozen walk61 that turns a short burst
# into a short repeatable step, for step-look-step. Trained on the current body only.
from .cmu_walk_step_env_cfg import CAPS as _WS_CAPS
from .cmu_walk_step_env_cfg import make_walk_step_env_cfg, walk_step_rl_cfg


class WalkStepResidualRunner(MicroduckOnPolicyRunner):
    """Critic warm-up: the zero correction (= walk61) is frozen while the fresh critic learns its returns."""

    CRITIC_WARMUP_ITERS = 50
    CRITIC_WARMUP_LR = 1e-3

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        alg, update = self.alg, self.alg.update
        actor_params, lr = list(alg.actor.parameters()), alg.learning_rate

        def warm_update():
            warm = self.current_learning_iteration < self.CRITIC_WARMUP_ITERS
            for p in actor_params:
                p.requires_grad_(not warm)
            alg.learning_rate = self.CRITIC_WARMUP_LR if warm else lr
            for g in alg.optimizer.param_groups:
                g["lr"] = alg.learning_rate
            return update()

        alg.update = warm_update


for _cap in _WS_CAPS:
    register_mjlab_task(
        task_id=f"Mjlab-WalkStep-Cap{round(_cap * 100):03d}-FromWalk61-MicroDuck-CMU-Current",
        env_cfg=make_walk_step_env_cfg(),
        play_env_cfg=make_walk_step_env_cfg(play=True),
        rl_cfg=walk_step_rl_cfg(MicroduckRlCfg, _cap),
        runner_cls=WalkStepResidualRunner,
    )

from .cmu_walk_step_env_cfg import NOISE_VARIANTS as _WS_NOISE

for _tag, _noise in _WS_NOISE.items():
    register_mjlab_task(
        task_id=f"Mjlab-WalkStep-{_tag}-FromWalk61-MicroDuck-CMU-Current",
        env_cfg=make_walk_step_env_cfg(noise=_noise),
        play_env_cfg=make_walk_step_env_cfg(play=True, noise=_noise),
        rl_cfg=walk_step_rl_cfg(MicroduckRlCfg, 0.15),
        runner_cls=WalkStepResidualRunner,
    )

from .cmu_walk_step_env_cfg import RC_VARIANTS as _WS_RC

for _tag, _noise in _WS_RC.items():
    register_mjlab_task(
        task_id=f"Mjlab-WalkStep-{_tag}-FromWalk61-MicroDuck-CMU-Current",
        env_cfg=make_walk_step_env_cfg(noise=_noise, robot_conditions=True),
        play_env_cfg=make_walk_step_env_cfg(play=True, noise=_noise, robot_conditions=True),
        rl_cfg=walk_step_rl_cfg(MicroduckRlCfg, 0.15),
        runner_cls=WalkStepResidualRunner,
    )

# CMU current-body ground pick and sit-stand (cmu_skill_env_cfg.py): Pollen's skill cfgs with only the body swapped.
from .cmu_skill_env_cfg import register_cmu_skill_tasks
register_cmu_skill_tasks(MicroduckOnPolicyRunner)

# Pick on the current body (2026-11-25, cmu/pick_rise/REFINE.md): the noise-trained ball kneel (ks-sn1's task) on the
# robot as it is now (918 g battery body). ks-sn1 @ 749 falls in 47 % of episodes here (0.7 % on Plugged).
#   KneelRise-DockFitC-Ball-<COND>-...-Current : the task under one test condition (kr_eval --task); SN1 is also the
#       full fine-tune's training task (kw-a, warm start from ks-sn1's checkpoint).
#   KneelRes-DockFitC-Ball-<COND>-...-Current  : the same task with the actor a frozen ks-sn1 @ 749 plus a capped
#       learned correction (residual_model.py, as rcn030 on walk61); cap and L1 from the CLI.
# COND: SN0..SN3 servo read noise level; RC = main's robot conditions (robot_conditions.py) at SN1; RCSN2 / RCSN3 the
# same at level 2 / 3; RCSNR13 = robot conditions with the level drawn per episode in [1, 3] (rcn030's recipe).
from .cmu_skill_env_cfg import apply_skill_conditions as _skill_cond
from .residual_model import ResidualModelCfg as _ResCfg
from .servo_noise import apply_servo_noise_range as _snr

KNEEL_BASE_ONNX = _Path(__file__).resolve().parents[3] / "cmu" / "kneel" / "policies" / "ks-sn1_749.onnx"


def _kneel_cond(cfg, cond: str):
    if cond.startswith("RC"):
        _skill_cond(cfg, "robot")
        cond = cond[2:] or "SN1"
    if cond == "SNR13":
        return _snr(cfg, 1, 3)
    return _sn(cfg, int(cond[2:]))


def _kneel_cfg(cond: str, play: bool = False):
    return _kneel_cond(make_kneel_rise_env_cfg("current", play=play, dockfit=True, ball_only=True,
                                              dock_fit=_KR_DOCKFIT_C), cond)


def kneel_res_rl_cfg(cap: float = 0.15, l1: float = 0.3):
    rl = _dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
                     experiment_name="kneel_res", run_name="kneel_res", max_iterations=1000)
    rl.actor = _ResCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=False,
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "log"},
                       base_onnx=str(KNEEL_BASE_ONNX), cap=cap, l1=l1)
    a = rl.algorithm
    a.schedule, a.learning_rate, a.entropy_coef = "fixed", 3e-4, 0.0
    a.symmetry_cfg = None
    return rl


for _cond in ("SN0", "SN1", "SN2", "SN3", "RC", "RCSN2", "RCSN3", "RCSNR13"):
    register_mjlab_task(
        task_id=f"Mjlab-KneelRise-DockFitC-Ball-{_cond}-MicroDuck-CMU-Current",
        env_cfg=_kneel_cfg(_cond),
        play_env_cfg=_kneel_cfg(_cond, play=True),
        rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
                           experiment_name="kneel_rise_ball_c_w918", run_name="kneel_rise_ball_c_w918",
                           max_iterations=750),
        runner_cls=KneelRiseSightedFromRiseRunner,
    )
KNEEL_BASE_CKPT = _Path(__file__).resolve().parents[3] / "logs" / "rsl_rl" / "kneel_rise_ball_c_sn1" / "ks-sn1-src" / "model_749.pt"


class KneelResCriticRunner(WalkStepResidualRunner):
    CRITIC_CKPT = None   # None: KNEEL_BASE_CKPT
    """The residual kneel with ks-sn1's trained critic (2026-11-25). With a fresh critic the correction never left zero
    (res_c30_rcn @ 999: mean |correction| 0.001 rad of a 0.30 cap; value loss 7 -> 17): the critic did not learn
    this episodic task's returns well enough for the advantages to outweigh the L1 pull. kw-a, which trained well,
    starts from that critic too."""

    CRITIC_WARMUP_ITERS = 20

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)
        import torch as _torch
        ckpt = self.CRITIC_CKPT or KNEEL_BASE_CKPT
        sd = _torch.load(ckpt, map_location=device, weights_only=False)["critic_state_dict"]
        self.alg.critic.load_state_dict(sd)
        print(f"[{type(self).__name__}] critic from {ckpt}")


for _cond in ("SN1", "RCSNR13"):
    register_mjlab_task(
        task_id=f"Mjlab-KneelResC-DockFitC-Ball-{_cond}-MicroDuck-CMU-Current",
        env_cfg=_kneel_cfg(_cond),
        play_env_cfg=_kneel_cfg(_cond, play=True),
        rl_cfg=kneel_res_rl_cfg(),
        runner_cls=KneelResCriticRunner,
    )

for _cond in ("SN1", "RCSNR13"):
    register_mjlab_task(
        task_id=f"Mjlab-KneelRes-DockFitC-Ball-{_cond}-MicroDuck-CMU-Current",
        env_cfg=_kneel_cfg(_cond),
        play_env_cfg=_kneel_cfg(_cond, play=True),
        rl_cfg=kneel_res_rl_cfg(),
        runner_cls=WalkStepResidualRunner,
    )

# Wide placement (2026-11-25, REFINE.md "docking spread"): the residual pick for a dock that misses by more. The
# resc runs hold 0.4-0.7 within +-1 cm of the landing spot and ~0 beyond +-3 cm. Here only 30 % of the starts follow
# the dock-shaped Gaussian (DOCKFIT_C); 70 % are uniform over the window, which starts at +-2 cm and widens by 0.5 cm
# (to +-5 cm, near side clipped at r_min) each time the held EMA passes 0.3. The frozen base cannot unlearn the bite.
from mjlab.managers import CurriculumTermCfg as _CurrCfg
from .cmu_seeing_pix_env_cfg import pix_window_curriculum as _pix_window_curr

WIDE = dict(frac=0.3, window_start=0.02, window_step=0.005, window_max=0.05, advance=0.15)  # 0.3 was never
# reached under robot conditions (held EMA 0.17-0.20 at +-2 cm after 350 iterations), so the window never widened


def _kneel_wide_cfg(cond: str, play: bool = False):
    cfg = _kneel_cfg(cond, play=play)
    ev = cfg.events["reset_object"]
    ev.params.update(dock_frac=WIDE["frac"], window=WIDE["window_max"] if play else WIDE["window_start"])
    if not play:
        cfg.curriculum["dock_window"] = _CurrCfg(
            func=_pix_window_curr, params={"event_name": "reset_object", "window_step": WIDE["window_step"],
                                           "window_max": WIDE["window_max"], "advance_success": WIDE["advance"]})
    return cfg


for _cond in ("SN1", "RCSNR13"):
    register_mjlab_task(
        task_id=f"Mjlab-KneelResC-Wide-DockFitC-Ball-{_cond}-MicroDuck-CMU-Current",
        env_cfg=_kneel_wide_cfg(_cond),
        play_env_cfg=_kneel_wide_cfg(_cond, play=True),
        rl_cfg=kneel_res_rl_cfg(),
        runner_cls=KneelResCriticRunner,
    )


# Fine step (2026-11-25, REFINE.md "docking spread"): the last ~10 cm before the kneel. rcn030's bursts under 0.5 s do
# not step and its 1.3 s burst spreads 3.8-9.4 cm, so a blind final burst lands within +-1.5 cm only ~6 % of the time.
# The nudge (wn-sn1: walk61 trained to step to a target 1-6 cm away, within 1 cm 0.966 on Plugged) frozen with the 918 g
# body's pitch trim, plus a capped correction trained with wn-sn1's critic (the kneel's recipe); targets widened to
# -4..+12 cm forward and +-5 cm sideways; main's robot conditions plus servo read noise drawn in [1, 3].
from .cmu_walk_nudge_env_cfg import make_walk_nudge_env_cfg as _mk_nudge

NUDGE_BASE_ONNX = _Path(__file__).resolve().parents[3] / "cmu" / "kneel" / "policies" / "wn-sn1_1249.onnx"
NUDGE_CRITIC_CKPT = (_Path(__file__).resolve().parents[3] / "logs" / "rsl_rl" / "walk_nudge_sn1" /
                     "2026-11-20_00-33-47_wn-sn1" / "model_1249.pt")
FINE_TARGET_X, FINE_TARGET_Y = (-0.04, 0.12), (-0.05, 0.05)


def _fine_step_cfg(play: bool = False):
    cfg = _mk_nudge("current", play=play)
    cfg.commands["twist"].target_x, cfg.commands["twist"].target_y = FINE_TARGET_X, FINE_TARGET_Y
    # make_cmu_velocity_env_cfg already applies main's robot conditions on the current body
    assert "robot_conditions" in str(cfg.observations["actor"].terms["base_ang_vel"].func) or \
        "k" in cfg.observations["actor"].terms["base_ang_vel"].params, "expected robot conditions on the current body"
    return _snr(cfg, 1, 3)


def fine_step_rl_cfg(cap: float = 0.30, l1: float = 0.05, trim: float = 6.0):
    rl = _dc.replace(walk_nudge_rl_cfg(MicroduckRlCfg), experiment_name="fine_step", run_name="fine_step",
                     max_iterations=1000)
    rl.actor = _ResCfg(hidden_dims=(256, 128, 64), activation="elu", obs_normalization=False,
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.05, "std_type": "log"},
                       base_onnx=str(NUDGE_BASE_ONNX), cap=cap, l1=l1, base_trim_deg=trim)
    a = rl.algorithm
    a.schedule, a.learning_rate, a.entropy_coef = "fixed", 3e-4, 0.0
    a.symmetry_cfg = None
    return rl


class FineStepRunner(KneelResCriticRunner):
    CRITIC_CKPT = NUDGE_CRITIC_CKPT


register_mjlab_task(
    task_id="Mjlab-FineStep-FromNudge-RCSNR13-MicroDuck-CMU-Current",
    env_cfg=_fine_step_cfg(),
    play_env_cfg=_fine_step_cfg(play=True),
    rl_cfg=fine_step_rl_cfg(),
    runner_cls=FineStepRunner,
)


# Hand-off starts (2026-11-26, REFINE.md "e2e harness"): the residual kneel continued from resc_c30t45_rcn @ 749 with
# HANDOFF_FRAC of the resets starting from the states the e2e harness recorded when the fine step handed over (joint
# pose and velocity, tilt, height, ball offset; e2e.py --save-handoff, RC and SN1 twins). The rest keep the dock-shaped
# placement. Under robot conditions resc_c30t45_rcn holds 0.43 of the hand-offs, missing 0.25 of all episodes.
from .kneel_handoff import reset_pix_object_handoff as _reset_handoff

HANDOFF_DIR = _Path(__file__).resolve().parents[3] / "cmu" / "pick_rise" / "refine" / "handoff"
HANDOFF_FILES = tuple(str(HANDOFF_DIR / f"ho_{c}.pt") for c in ("RC", "SN1"))
HANDOFF_FRAC = 0.6


def _kneel_handoff_cfg(cond: str, play: bool = False, frac: float = HANDOFF_FRAC):
    cfg = _kneel_cfg(cond, play=play)
    ev = cfg.events["reset_object"]
    ev.func = _reset_handoff
    ev.params.update(handoff_paths=HANDOFF_FILES, frac=frac)
    return cfg


for _cond in ("SN1", "RCSNR13"):
    register_mjlab_task(
        task_id=f"Mjlab-KneelResC-Handoff-DockFitC-Ball-{_cond}-MicroDuck-CMU-Current",
        env_cfg=_kneel_handoff_cfg(_cond),
        play_env_cfg=_kneel_handoff_cfg(_cond, play=True),
        rl_cfg=kneel_res_rl_cfg(),
        runner_cls=KneelResCriticRunner,
    )

register_mjlab_task(   # the dose arm: 90 % hand-off starts
    task_id="Mjlab-KneelResC-Handoff90-DockFitC-Ball-RCSNR13-MicroDuck-CMU-Current",
    env_cfg=_kneel_handoff_cfg("RCSNR13", frac=0.9),
    play_env_cfg=_kneel_handoff_cfg("RCSNR13", play=True, frac=0.9),
    rl_cfg=kneel_res_rl_cfg(),
    runner_cls=KneelResCriticRunner,
)


# Full kneel fine-tune on the 918 g body (2026-11-27, REFINE.md "kneel_full"): the residual (frozen ks-sn1 + 0.30 cap)
# is at its ceiling under robot conditions, so every weight trains. Start: ks-sn1 @ 749 with the residual's 4.5 deg
# IMU pitch trim folded into its first layer (cmu/pick_rise/refine/fold_trim.py; trim alone: SN1 held 0.32 / fell 0.31
# against 0.20 / 0.47). Conditions come in by stages, each a resume of the last (KneelFull-SN1 -> -RC -> -RCSNR13): the
# earlier full fine-tunes went straight to their final conditions and kw-b (robot conditions) degraded everywhere. The
# world is the e2e harness's: the walking tasks' body randomisation (trunk CoM +-8 mm, not +-2.5 cm) and IMU mount
# misalignment +-2 deg (not +-6); the kneel's own IMU noise and age stay. Half of the resets start from the recorded
# hand-offs (gaze on), the rest from the dock-shaped placement with the head aimed.
FULL_DR_TERMS = ("base_com", "randomize_com", "randomize_head_com", "foot_friction")
FULL_IMU_MISALIGN_DEG = 2.0
FULL_HANDOFF_FRAC = 0.5


def _kneel_full_cfg(cond: str, play: bool = False):
    cfg = _kneel_handoff_cfg(cond, play=play, frac=FULL_HANDOFF_FRAC)
    walk = make_walk_step_env_cfg()
    for k in FULL_DR_TERMS:
        if k in walk.events:
            cfg.events[k] = walk.events[k]
        else:
            cfg.events.pop(k, None)
    for k in ("com_range", "head_com_range"):   # the kneel's curricula rewrite the CoM ranges while it runs
        cfg.curriculum.pop(k, None)
    actor = cfg.observations["actor"].terms
    for t in ("base_ang_vel", "projected_gravity"):
        term = actor[t] = _copy.deepcopy(actor[t])
        assert "max_angle_deg" in term.params, (t, term.params)
        term.params = dict(term.params, max_angle_deg=FULL_IMU_MISALIGN_DEG)
    return cfg


class KneelFullRunner(KneelRiseSightedFromRiseRunner):
    """The sighted kneel's warm start (critic warm-up, actor frozen) with the init lr taken from the rl cfg
    (--agent.algorithm.learning-rate), so the two arms differ only in schedule and lr."""

    CRITIC_WARMUP_ITERS = 50

    def __init__(self, env, train_cfg: dict, log_dir=None, device="cpu", **kwargs):
        self.INIT_LR = float(train_cfg["algorithm"]["learning_rate"])
        super().__init__(env, train_cfg, log_dir, device, **kwargs)


for _cond in ("SN1", "RC", "RCSNR13"):
    register_mjlab_task(
        task_id=f"Mjlab-KneelFull-{_cond}-MicroDuck-CMU-Current",
        env_cfg=_kneel_full_cfg(_cond),
        play_env_cfg=_kneel_full_cfg(_cond, play=True),
        rl_cfg=_dc.replace(kneel_rise_rl_cfg(MicroduckStaticGraspRlCfg, dockfit=True, ball_only=True),
                           experiment_name="kneel_full", run_name="kneel_full", max_iterations=500),
        runner_cls=KneelFullRunner,
    )
