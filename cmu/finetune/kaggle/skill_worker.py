"""Pollen's ground-pick or sit-stand recipe trained on the current robot body (sibling of scratch_worker.py).

What runs: tasks/cmu_skill_env_cfg.py, i.e. Pollen's make_microduck_ground_pick_env_cfg / make_microduck_sitstand_env_cfg
with only the robot's spec_fn swapped to the CMU body variant (`--model current` = cmu/model/current.json), and the
task's own runner cfg (MicroduckGroundPickRlCfg / MicroduckSitStandRlCfg) with only the bookkeeping fields set (seed,
max_iterations, save_interval, logger, run_name). `learn(init_at_random_ep_len=True)` as mjlab's train script calls it.
  --task groundpick   ground pick; --bend-recipe pollen (Pollen's reward, default) or bend4 (BEND-PLAN.md / NEXT-PLAN.md
                      bend4 ported to this body: mouth costs, torque hinge, pitch band, knee margin, rest stance, held
                      crouch, in-episode hand-back to --handback-walker, bend guard; tasks/cmu_skill_env_cfg.py)
  --task sitstand     sit-stand (Pollen: 1250 iterations from scratch, then resumed to 2250, docs/sitstand_policy.md)
  --conditions official|robot   robot = the measured sensing/actuation conditions scratch_worker.py applies with
                      `--conditions robot` (cmu_skill_env_cfg.apply_skill_conditions; ground pick IMU age 0-3 -> 0-2)

Start (one of, default from scratch with random weights and curricula from step 0):
  --init-checkpoint PATH   continue a finished run of this worker (full rsl_rl .pt: actor, critic, optimizer, iteration,
                           common_step_counter), loaded by scratch_worker.load_init_checkpoint; numbering continues.
  --init-onnx-actor PATH   warm start the actor from an exported ONNX (Pollen's groundpick61 / sitstand61: normaliser
                           and MLP weights; action std --init-std); the critic is new and trains alone for
                           --warmup-iters iterations (default 100) at the configured lr with the schedule fixed and the
                           actor frozen: MLP weights, action std and the observation normaliser (rsl_rl's
                           actor.update_normalization is a no-op while warm; mean / var / std / count must be
                           bit-identical at the unfreeze or the run stops). After the warm-up --max-lr (default 0 = off)
                           caps the adaptive lr, from the unfreeze on. The re-exports model_0.onnx and
                           model_warmup_end.onnx (just before the unfreeze) must reproduce the source to 1e-4. --warm-counter zero (default): curricula start at step 0, as from scratch; final sets
                           common_step_counter to the step where every staged curriculum is at its last stage. The
                           counter at start is in run_record.json "curriculum"; the log prints the iterations at which
                           the sit curriculum adds its seated disturbances and pushes.

Run controls (they change nothing the recipe computes): a full checkpoint and an ONNX (runner export, normaliser
baked in) every --save-interval iterations as model_<it>.pt / model_<it>.onnx, plus the final one; status.json
heartbeat after every save and every metric row (stage, iteration, updated, DONE / FAILED at the end) and STATUS;
metrics.jsonl every --metric-interval iterations with the mean episode length, mean reward and every reward term's
weighted value per second; --max-hours wall cap (clean stop with a final save).
Body check before any step: run_record.json "body" holds the manifest path, its sha256, the built model's total mass;
--expect-manifest-sha HEX exits 3 before the env is built when the manifest differs.

Kaggle: `kaggle_job.py push skill --run-name A --pair-run-name B -- --task sitstand --seed 1 -- --task sitstand --seed 2`.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scratch_worker import (  # noqa: E402  (sibling module; stdlib-only at import)
    StopTraining,
    body_manifest,
    built_body,
    check_expected_manifest,
    check_init_checkpoint,
    load_init_checkpoint,
    sha256_file,
)

TASKS = {
    "groundpick": "Mjlab-GroundPick-Flat-MicroDuck-CMU-Current",
    "sitstand": "Mjlab-SitStand-Flat-MicroDuck-CMU-Current",
}
MODELS = ("current",)


def add_sit_target_args(ap):
    """Sit-stand target override flags (default off; cmu_skill_env_cfg.apply_sit_targets). Poses are
    'servo_name=rad,...' merged over Pollen's SIT keyframe (--sit-pose, --sit-spawn-pose) or over HOME (--stand-pose);
    the action offset, default_joint_pos and the observation stay HOME. skill_reads.py and skill_audit.py take the same
    flags so reads and audits score against the same targets as the run."""
    g = ap.add_argument_group("sit-stand targets (sitstand only, default Pollen's)")
    g.add_argument("--sit-pose", default=None, help="SIT keyframe joints, e.g. left_ankle=-0.357,right_ankle=0.357")
    g.add_argument("--stand-pose", default=None, help="STAND reward target joints over HOME (reward only)")
    g.add_argument("--stand-z", type=float, default=None, help="STAND_Z (Pollen 0.115)")
    g.add_argument("--sit-z", type=float, default=None, help="SIT_Z (Pollen 0.060)")
    g.add_argument("--sit-spawn-pose", default=None, help="seated spawn joints (default: the SIT keyframe in use)")
    g.add_argument("--sit-spawn-z", default=None, help="seated spawn height LO:HI (Pollen 0.06:0.075)")
    g.add_argument("--sit-upright-pitch-deg", type=float, default=None,
                   help="seated upright reference, deg nose-down (forward positive), following the posture blend "
                        "(0 at stand); applies to upright_linear, posture_composite's upright factor, posture_stillness")
    return ap


def sit_targets_from_args(a) -> dict | None:
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    t = dict(sit_pose=sk.parse_pose(getattr(a, "sit_pose", None)), stand_pose=sk.parse_pose(getattr(a, "stand_pose", None)),
             stand_z=getattr(a, "stand_z", None), sit_z=getattr(a, "sit_z", None),
             spawn_pose=sk.parse_pose(getattr(a, "sit_spawn_pose", None)),
             spawn_z=tuple(float(x) for x in a.sit_spawn_z.split(":")) if getattr(a, "sit_spawn_z", None) else None,
             upright_pitch_deg=getattr(a, "sit_upright_pitch_deg", None))
    t = {k: v for k, v in t.items() if v is not None}
    return t or None


def make_env_cfg(task: str, model: str = "current", recipe: str = "pollen", conditions: str = "official",
                 play: bool = False, walker: str | None = None, sit_targets: dict | None = None):
    from mjlab_microduck.tasks import cmu_skill_env_cfg as sk
    if task == "groundpick":
        if sit_targets:
            raise SystemExit("the sit-stand target flags need --task sitstand")
        kw = {"walker": walker} if walker else {}
        return sk.make_cmu_ground_pick_env_cfg(model, recipe=recipe, conditions=conditions, play=play, **kw)
    if task == "sitstand":
        if recipe != "pollen":
            raise SystemExit("--bend-recipe is a ground-pick option (the sit-stand command has no phase)")
        kw = {"sit_targets": sit_targets} if sit_targets else {}
        return sk.make_cmu_sitstand_env_cfg(model, conditions=conditions, play=play, **kw)
    raise SystemExit(f"--task {task!r}: choose from {sorted(TASKS)}")


def staged_iterations(env_cfg, names=None) -> dict:
    """curriculum term -> list of (iteration, stage value) for every *_stages list (step / 24 = iteration)"""
    out = {}
    for n, term in (env_cfg.curriculum or {}).items():
        if names and n not in names:
            continue
        for k, v in (getattr(term, "params", None) or {}).items():
            if k.endswith("_stages") and isinstance(v, (list, tuple)):
                out[n] = [(int(st["step"]) // 24, {kk: vv for kk, vv in st.items() if kk != "step"}) for st in v
                          if isinstance(st, dict) and "step" in st]
    return out


SIT_DISTURBANCES = ("push_magnitude", "seated_head_jerk_prob", "seated_tip_range")


def _install_stop_shim():
    """bend_terms.bend_guard raises `finetune_worker.StopTraining`; map it to this worker's StopTraining (the modal
    finetune worker is not imported here)."""
    import types
    if "finetune_worker" not in sys.modules:
        sys.modules["finetune_worker"] = types.SimpleNamespace(StopTraining=StopTraining)


def base_agent_cfg(task: str):
    if task == "groundpick":
        from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import MicroduckGroundPickRlCfg
        return MicroduckGroundPickRlCfg
    from mjlab_microduck.tasks.microduck_sitstand_env_cfg import MicroduckSitStandRlCfg
    return MicroduckSitStandRlCfg


AGENT_BOOKKEEPING = ("seed", "max_iterations", "save_interval", "logger", "run_name")


def make_agent_cfg(task: str, run_name: str, seed: int, max_iters: int, save_interval: int):
    cfg = copy.deepcopy(base_agent_cfg(task))
    cfg.seed = seed
    cfg.max_iterations = max_iters
    cfg.save_interval = save_interval
    cfg.logger = "tensorboard"
    cfg.run_name = run_name
    return cfg


def curricula_final_step(env_cfg) -> int:
    """The env step at which every step-staged curriculum of the cfg is at its last stage (max 'step' over every
    *_stages list in the curriculum params), plus one iteration's 24 steps."""
    last = 0
    for term in (env_cfg.curriculum or {}).values():
        for k, v in (getattr(term, "params", None) or {}).items():
            if k.endswith("_stages") and isinstance(v, (list, tuple)):
                for st in v:
                    if isinstance(st, dict) and "step" in st:
                        last = max(last, int(st["step"]))
    return last + 24


def _actor(alg):
    a = getattr(alg, "actor", None)
    return a if a is not None else alg.get_policy()


def load_onnx_into_actor(actor, onnx_path: str, init_std: float):
    """Copy of flight_worker.load_onnx_into_actor (finetune_worker.load_walk61_into_actor allowing a normaliser std of
    exactly 0, as groundpick61's never-varying command slots export): normaliser mean / std and the MLP weights from
    the exported graph (Sub, Div, Gemm/Elu ...), action std = init_std."""
    import onnx
    import torch
    from onnx import numpy_helper
    m = onnx.load(onnx_path)
    W = {i.name: torch.tensor(numpy_helper.to_array(i)).clone() for i in m.graph.initializer}
    eps = float(getattr(actor.obs_normalizer, "eps", 0.0))
    std = next(v for k, v in W.items() if k.startswith("onnx::Div")) - eps
    assert float(std.min()) > -1e-6, float(std.min())
    std = std.clamp(min=0.0)
    mean = W["obs_normalizer._mean"]
    sd = actor.state_dict()
    new = {}
    for k in sd:
        if k == "obs_normalizer._mean":
            new[k] = mean
        elif k == "obs_normalizer._var":
            new[k] = std ** 2
        elif k == "obs_normalizer._std":
            new[k] = std
        elif k == "obs_normalizer.count":
            new[k] = torch.tensor(10 ** 9, dtype=sd[k].dtype)
        elif k.startswith("mlp."):
            new[k] = W[k]
        elif k == "distribution.std_param":
            new[k] = torch.full_like(sd[k], init_std)
        elif k == "distribution.log_std_param":
            new[k] = torch.full_like(sd[k], float(torch.log(torch.tensor(init_std))))
        else:
            raise KeyError(f"unexpected actor key {k}")
    assert new["mlp.0.weight"].shape == sd["mlp.0.weight"].shape, (new["mlp.0.weight"].shape, sd["mlp.0.weight"].shape)
    actor.load_state_dict({k: v.to(sd[k].device, sd[k].dtype) for k, v in new.items()}, strict=True)
    return dict(zero_std_slots=int((std == 0).sum()))


GUARD_OBSERVE = {"knee_squat": 1e9, "pitch_min_deg": -1e9, "fall_factor": 1e12}


def apply_guard_mode(env_cfg, mode: str):
    """bend4's bend_guard: observe = its window lines still print (hold knee, pitch, mouth, falls) but its squat and
    falling tests can never be true, so it never raises StopTraining; stop = its own thresholds. None without a guard."""
    term = (env_cfg.rewards or {}).get("bend_guard")
    if term is None:
        return None
    if mode == "observe":
        term.params.update(GUARD_OBSERVE)
    return dict(mode=mode, params=dict(term.params))


class WarmupFreeze:
    """The actor during the critic-only warm-up: every actor parameter (MLP weights and the action std / log-std) has
    requires_grad False, and actor.update_normalization (called by rsl_rl PPO.process_env_step every env step) is a
    no-op, so the observation normaliser's mean / var / std / count stay bit-identical. release() restores both and
    returns the check (identical parameters and normaliser buffers, skipped normaliser updates)."""

    def __init__(self, actor):
        self.actor = actor
        self.active = False
        self.skipped = 0

    def _norm_state(self):
        nz = getattr(self.actor, "obs_normalizer", None)
        return {k: v.detach().clone() for k, v in (nz.state_dict().items() if nz is not None else [])}

    def freeze(self):
        a = self.actor
        self.params0 = {k: p.detach().clone() for k, p in a.named_parameters()}
        self.norm0 = self._norm_state()
        for p in a.parameters():
            p.requires_grad_(False)

        def no_update(obs):
            self.skipped += 1
        a.update_normalization = no_update  # instance attribute shadows the class method
        self.active = True
        return self

    def release(self) -> dict:
        import torch
        a = self.actor
        if "update_normalization" in a.__dict__:
            del a.__dict__["update_normalization"]
        for p in a.parameters():
            p.requires_grad_(True)
        self.active = False
        norm1 = self._norm_state()
        params_same = all(torch.equal(self.params0[k], p.detach()) for k, p in a.named_parameters())
        norm_same = set(norm1) == set(self.norm0) and all(torch.equal(self.norm0[k], norm1[k]) for k in norm1)
        return dict(params_identical=params_same, normaliser_identical=norm_same,
                    normaliser_buffers=sorted(norm1), normaliser_updates_skipped=self.skipped)


def capped_lr(lr: float, max_lr: float, warm: bool):
    """--max-lr: the lr to set after an update, or None to leave it; no cap while the critic warms up"""
    if warm or not max_lr or lr <= max_lr:
        return None
    return max_lr


def compare_onnx(a: str, b: str, n: int = 256, seed: int = 0) -> float:
    import numpy as np
    import onnxruntime as ort
    sa, sb = ort.InferenceSession(a), ort.InferenceSession(b)
    obs = np.random.default_rng(seed).normal(0, 0.5, size=(n, 61)).astype(np.float32)
    d = 0.0
    for o in obs:
        ya = sa.run(None, {sa.get_inputs()[0].name: o[None]})[0]
        yb = sb.run(None, {sb.get_inputs()[0].name: o[None]})[0]
        d = max(d, float(np.abs(ya - yb).max()))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--model", choices=MODELS, default="current")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--save-interval", type=int, default=250)
    ap.add_argument("--metric-interval", type=int, default=50)
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--bend-recipe", choices=("pollen", "bend4"), default="pollen",
                    help="groundpick only: pollen = Pollen's reward; bend4 = the audited bend4 recipe on this body")
    ap.add_argument("--handback-walker", default=None,
                    help="bend4: frozen walker ONNX for the in-episode hand-back (default "
                         "cmu/whole/runtime-overlay/models-bw-s2c/bw-s2c-b2.onnx)")
    ap.add_argument("--bend-guard", choices=("observe", "stop"), default="observe",
                    help="bend4: observe (default) = the guard logs its 250-iteration windows and never stops the run "
                         "(K1 kill decisions are the coordinator's); stop = bend_terms.bend_guard's own StopTraining")
    ap.add_argument("--conditions", choices=("official", "robot"), default="official",
                    help="robot = the robot's measured sensing and actuation conditions (scratch_worker --conditions robot)")
    ap.add_argument("--max-lr", type=float, default=0.0, help="cap on the adaptive PPO lr after every update (0 = off)")
    ap.add_argument("--init-checkpoint", default=None, help="full rsl_rl checkpoint of this worker to continue")
    ap.add_argument("--init-sha256", default=None, help="expected sha256 of --init-checkpoint (kaggle_job.py sets it)")
    ap.add_argument("--init-onnx-actor", default=None, help="warm start the actor from an exported 61-input ONNX")
    ap.add_argument("--init-onnx-sha256", default=None, help="expected sha256 of --init-onnx-actor (kaggle_job.py sets it)")
    ap.add_argument("--init-std", type=float, default=0.25, help="action std after --init-onnx-actor")
    ap.add_argument("--warmup-iters", type=int, default=100, help="critic-only iterations after --init-onnx-actor")
    ap.add_argument("--warm-counter", choices=("final", "zero"), default="zero",
                    help="common_step_counter after --init-onnx-actor: 0 (default) or the final curriculum stage")
    ap.add_argument("--expect-manifest-sha", default=None)
    ap.add_argument("--gpu", default="?")
    ap.add_argument("--rate-per-hour", type=float, default=0.0, help="accepted for kernel_entry.py; unused")
    ap.add_argument("--device", default="cuda:0")
    add_sit_target_args(ap)
    args = ap.parse_args(argv)
    if args.bend_recipe != "pollen" and args.task != "groundpick":
        ap.error("--bend-recipe needs --task groundpick")
    if args.handback_walker and args.bend_recipe != "bend4":
        ap.error("--handback-walker needs --bend-recipe bend4")
    if args.handback_walker and not Path(args.handback_walker).is_file():
        ap.error(f"--handback-walker {args.handback_walker}: no such file")
    if args.init_checkpoint and args.init_onnx_actor:
        ap.error("--init-checkpoint and --init-onnx-actor are exclusive")
    if args.init_sha256 and not args.init_checkpoint:
        ap.error("--init-sha256 needs --init-checkpoint")
    if args.init_onnx_sha256 and not args.init_onnx_actor:
        ap.error("--init-onnx-sha256 needs --init-onnx-actor")
    if args.max_iters < 1:
        ap.error("--max-iters must be >= 1")
    task_id = TASKS[args.task] + ("-Bend4" if args.bend_recipe == "bend4" else "") + (
        "+robot-conditions" if args.conditions == "robot" else "")

    body = body_manifest(args.model)
    print(f"[skill] body: variant {body['variant']}, manifest {body['mass_manifest']} sha256 {body['mass_manifest_sha256']}",
          flush=True)
    check_expected_manifest(body, args.expect_manifest_sha)
    init = check_init_checkpoint(args.init_checkpoint, args.init_sha256) if args.init_checkpoint else None
    onnx_init = None
    if args.init_onnx_actor:
        p = Path(args.init_onnx_actor)
        if not p.is_file():
            raise SystemExit(f"--init-onnx-actor {p}: no such file")
        sha = sha256_file(p)
        if args.init_onnx_sha256 and sha != args.init_onnx_sha256:
            raise SystemExit(f"--init-onnx-actor {p}: sha256 {sha} != --init-onnx-sha256 {args.init_onnx_sha256}")
        onnx_init = dict(path=str(p.resolve()), sha256=sha, init_std=args.init_std, warmup_iters=args.warmup_iters,
                         warm_counter=args.warm_counter)

    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    import mjlab_microduck.tasks  # noqa: F401  (registers the CMU skill ids; load_runner_cls needs them)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = args.device
    t_start = time.time()
    t0_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_dir = Path(args.root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.glob("model_*.pt")):
        raise SystemExit(f"{run_dir} already holds checkpoints; give a new --run-name (--init-checkpoint continues one)")
    if init and Path(init["path"]).parent == run_dir.resolve():
        raise SystemExit(f"--init-checkpoint {init['path']} lies in the new run directory; use a new --run-name")

    def status(stage, **kw):
        (run_dir / "status.json").write_text(json.dumps(dict(
            stage=stage, run_name=args.run_name, task=task_id, started=t0_iso,
            updated=time.strftime("%Y-%m-%dT%H:%M:%S"), **kw), indent=1, default=str))

    status("build")
    _install_stop_shim()
    sit_targets = sit_targets_from_args(args)
    if sit_targets and args.task != "sitstand":
        raise SystemExit("the sit-stand target flags need --task sitstand")
    if sit_targets:
        task_id += "+sit-targets"
    env_cfg = make_env_cfg(args.task, args.model, recipe=args.bend_recipe, conditions=args.conditions,
                           walker=args.handback_walker, sit_targets=sit_targets)
    if sit_targets:
        print(f"[skill] sit-stand targets overridden: {json.dumps(env_cfg.__dict__.get('_sit_targets'))}", flush=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    guard_mode = apply_guard_mode(env_cfg, args.bend_guard)
    agent_cfg = make_agent_cfg(args.task, args.run_name, args.seed, args.max_iters, args.save_interval)
    stages = staged_iterations(env_cfg)
    if args.task == "sitstand":
        for n in SIT_DISTURBANCES:
            if n in stages:
                print(f"[skill] sit curriculum {n}: " + "; ".join(f"iteration {i}: {v}" for i, v in stages[n]), flush=True)
    record = dict(run_name=args.run_name, task=task_id, skill=args.task, model=args.model, bend_recipe=args.bend_recipe,
                  conditions=args.conditions, robot_conditions=env_cfg.__dict__.get("_robot_conditions"),
                  bend4=env_cfg.__dict__.get("_bend4"), bend_guard=guard_mode, max_lr=args.max_lr,
                  sit_targets=env_cfg.__dict__.get("_sit_targets"),
                  curriculum=dict(stages_by_iteration=stages, final_step=curricula_final_step(env_cfg)),
                  route=("continue" if init else "warm start from ONNX" if onnx_init else "from scratch")
                  + " (Pollen's skill recipe, current body)",
                  seed=args.seed, num_envs=args.num_envs, max_iters=args.max_iters, save_interval=args.save_interval,
                  metric_interval=args.metric_interval, max_hours=args.max_hours, gpu=args.gpu, device=device,
                  started=t0_iso, argv=sys.argv[1:] if argv is None else list(argv),
                  reward_weights={k: v.weight for k, v in env_cfg.rewards.items()},
                  agent_cfg=dataclasses.asdict(agent_cfg),
                  body=dict(body, expect_manifest_sha=args.expect_manifest_sha))
    if init:
        record["init_checkpoint"] = dict(init)
    if onnx_init:
        record["init_onnx_actor"] = dict(onnx_init)
    cur = Path(__file__).resolve().parents[3] / "cmu/model/current.json"
    record["current"] = {k: json.loads(cur.read_text())[k] for k in ("variant", "mass_manifest", "mass_manifest_sha256")}

    def write_record():
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))

    write_record()
    env = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        record["body"].update(built_body(env, body))
        print(f"[skill] body built: total mass {record['body']['total_mass_g']:.2f} g", flush=True)
        write_record()
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_base = load_runner_cls(TASKS[args.task]) or MjlabOnPolicyRunner
        out = {"stop": None, "onnx_missing": []}

        class SkillRunner(runner_base):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                rm = self.env.unwrapped.reward_manager
                self.terms = list(rm.active_terms)
                self.wsum = torch.zeros(len(self.terms))
                self.wsteps = 0
                self.last_row = None
                self.saves = 0
                self.warm = False
                orig_step = self.logger.process_env_step

                def hooked_step(rewards, dones, extras, intrinsic_rewards=None):
                    self.wsum += rm._step_reward.mean(0).detach().float().cpu()
                    self.wsteps += 1
                    return orig_step(rewards, dones, extras, intrinsic_rewards)
                self.logger.process_env_step = hooked_step
                orig_log = self.logger.log

                def hooked_log(*a, **k):
                    r = orig_log(*a, **k)
                    self.after_iteration(int(k["it"]))
                    return r
                self.logger.log = hooked_log
                orig_update = self.alg.update

                def clamped_update(*a, **k):  # --max-lr: cap the adaptive lr after every update, after the warm-up
                    r = orig_update(*a, **k)
                    self.apply_lr_cap()
                    return r
                self.alg.update = clamped_update
                self.freeze = None

            def apply_lr_cap(self):
                lr = capped_lr(self.alg.learning_rate, args.max_lr, self.warm)
                if lr is not None:
                    self.alg.learning_rate = lr
                    for g in self.alg.optimizer.param_groups:
                        g["lr"] = lr

            def end_warmup(self, it):
                runner_dir = run_dir
                self.export_policy_to_onnx(str(runner_dir), "model_warmup_end.onnx")
                d1 = compare_onnx(onnx_init["path"], str(runner_dir / "model_warmup_end.onnx"))
                chk = self.freeze.release()
                self.warm = False
                self.alg.schedule = "adaptive"
                self.apply_lr_cap()  # the first actor update already runs under the cap
                chk.update(iteration=it, model_warmup_end_max_abs_diff=d1, lr_after=self.alg.learning_rate)
                record["init_onnx_actor"]["warmup_end"] = chk
                write_record()
                print(f"[skill] critic warm-up done at iteration {it}; actor unfrozen; {json.dumps(chk)}", flush=True)
                if d1 >= 1e-4 or not chk["params_identical"] or not chk["normaliser_identical"]:
                    raise SystemExit(f"warm-up changed the actor: {json.dumps(chk)}")

            def after_iteration(self, it):
                elapsed = time.time() - t_start
                if self.warm and it >= args.warmup_iters - 1:
                    self.end_warmup(it)
                if it % args.metric_interval == 0:
                    lb, rb = self.logger.lenbuffer, self.logger.rewbuffer
                    row = dict(iter=it, elapsed_s=round(elapsed, 1),
                               mean_ep_len_steps=round(float(np.mean(lb)), 1) if len(lb) else None,
                               mean_reward=round(float(np.mean(rb)), 4) if len(rb) else None,
                               terms_per_s={n: round(float(v) / max(self.wsteps, 1), 4) for n, v in zip(self.terms, self.wsum)},
                               action_std=round(float(self.alg.get_policy().output_std.mean()), 4),
                               lr=self.alg.learning_rate, warm=self.warm, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    self.wsum.zero_()
                    self.wsteps = 0
                    with open(run_dir / "metrics.jsonl", "a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(f"[skill] {json.dumps(row)}", flush=True)
                    self.last_row = row
                    status("train", iter=it, last=row)
                if elapsed >= args.max_hours * 3600:
                    out["stop"] = f"wall cap {args.max_hours} h reached at iteration {it}"
                    print("[skill] STOP:", out["stop"], flush=True)
                    raise StopTraining(out["stop"])

            def save(self, path, infos=None):
                super().save(path, infos)
                it = self.current_learning_iteration
                onnx = run_dir / f"{run_dir.name}.onnx"
                if onnx.exists():
                    shutil.copy(onnx, run_dir / f"model_{it}.onnx")
                else:
                    out["onnx_missing"].append(it)
                    if self.saves == 0:
                        raise SystemExit("ONNX export failed at the first save; stopping before the GPU hours are spent")
                self.saves += 1
                status("train", iter=it, saved=f"model_{it}.pt", onnx=f"model_{it}.onnx" if onnx.exists() else None,
                       last=self.last_row)

        runner = SkillRunner(env, dataclasses.asdict(agent_cfg), str(run_dir), device)
        if init:
            loaded = load_init_checkpoint(runner, init, agent_cfg.num_steps_per_env)
            record["init_checkpoint"] = loaded
            print(f"[skill] continuing {loaded['path']} at iteration {loaded['first_iteration']}, common_step_counter "
                  f"{loaded['common_step_counter_after_load']}", flush=True)
        elif onnx_init:
            actor = _actor(runner.alg)
            onnx_init.update(load_onnx_into_actor(actor, onnx_init["path"], args.init_std))
            counter = curricula_final_step(env_cfg) if args.warm_counter == "final" else 0
            env.unwrapped.common_step_counter = counter
            runner.current_learning_iteration = 0
            runner.export_policy_to_onnx(str(run_dir), "model_0.onnx")
            d0 = compare_onnx(onnx_init["path"], str(run_dir / "model_0.onnx"))
            onnx_init.update(common_step_counter=counter, model0_max_abs_diff=d0)
            print(f"[skill] actor from {onnx_init['path']} (sha256 {onnx_init['sha256'][:16]}), common_step_counter "
                  f"{counter}, model_0.onnx max abs diff {d0:.2e}", flush=True)
            if d0 >= 1e-4:
                raise SystemExit(f"model_0.onnx differs from {onnx_init['path']} by {d0:.2e} (>= 1e-4)")
            if args.warmup_iters > 0:
                runner.warm = True
                runner.freeze = WarmupFreeze(actor).freeze()
                runner.alg.schedule = "fixed"
            record["init_onnx_actor"] = onnx_init
        record["curriculum"]["common_step_counter_at_start"] = int(env.unwrapped.common_step_counter)
        if record.get("bend4"):
            record["bend4"]["walker_sha256"] = sha256_file(record["bend4"]["walker"])
        write_record()
        status("loaded" if (init or onnx_init) else "start", iter=runner.current_learning_iteration)
        print(f"[skill] {task_id}: {record['route']}, seed {args.seed}, {args.num_envs} envs, {args.max_iters} iterations, "
              f"common_step_counter {env.unwrapped.common_step_counter}", flush=True)
        stop_reason = "max_iters reached"
        try:
            runner.learn(num_learning_iterations=args.max_iters, init_at_random_ep_len=True)
        except StopTraining as e:
            stop_reason = str(e)
            it = runner.current_learning_iteration
            if not (run_dir / f"model_{it}.pt").exists():
                runner.save(str(run_dir / f"model_{it}.pt"))
        elapsed = time.time() - t_start
        record.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), final_iter=runner.current_learning_iteration,
                      stop_reason=stop_reason, elapsed_s=round(elapsed, 1), onnx_missing=out["onnx_missing"],
                      last_metrics=runner.last_row)
        write_record()
        (run_dir / "STATUS").write_text(f"DONE {stop_reason}\n")
        status("done", iter=runner.current_learning_iteration, reason=stop_reason, last=runner.last_row)
        print(f"[skill] DONE: {stop_reason}; final iteration {runner.current_learning_iteration}; {elapsed / 3600:.2f} h",
              flush=True)
        return 0
    except BaseException as e:  # noqa: BLE001  (every failure ends with FAILED in the heartbeat)
        reason = f"{type(e).__name__}: {e}"
        record.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), stop_reason=reason,
                      traceback=traceback.format_exc()[-4000:])
        write_record()
        (run_dir / "STATUS").write_text(f"FAILED {reason}\n")
        status("failed", reason=reason)
        print(f"[skill] FAILED: {reason}", flush=True)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
