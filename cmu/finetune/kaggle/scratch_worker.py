"""Pollen's walking recipe trained from scratch, unchanged except for the robot model (battery-walk stage S1).

What runs: task Mjlab-Velocity-Flat-MicroDuck exactly as `uv run train` builds it (make_microduck_velocity_env_cfg and
MicroduckRlCfg: lr 1e-3 adaptive, init_std 1.0, entropy 0.01, 512-256-128 ELU, obs normalization), random weights,
curricula from step 0, `learn(init_at_random_ep_len=True)` as mjlab's train script calls it. No ONNX load, no warm-up,
no delta, no robot conditions. The only choice is the robot model:
  --model stock    Pollen's walk model untouched (the reproduction control arm)
  --model current  the robot as it is now: cmu/model/current.json through make_cmu_velocity_env_cfg("current",
                   robot_conditions=False), i.e. the variant's signed mass parts and the ankle heel stop, nothing else
tests/test_scratch_worker_cfg.py checks that both cfgs equal the upstream task cfg apart from the robot model.

--conditions (current model only; S1 ran `official`):
  official  the cfg above, no robot conditions (default)
  gyro      official plus only the leg-speed gyro noise (src/mjlab_microduck/robot_conditions.py: actor base_ang_vel
            + N(0, (K x rms leg joint speed)^2) per axis per tick, on top of the stock +-0.03), with K drawn once per
            env at env creation, uniform in GYRO_LEGSPEED_RANGE (0.15, 0.30) (robot random part 0.16-0.22,
            gyro-signal-20261123; until 2026-11-23 the route used the constant K = 0.3)
  robot     all six ROBOT_CONDITIONS as make_cmu_velocity_env_cfg("current", robot_conditions=True) applies them,
            except that the gyro factor is the same per-env K as in `gyro` (the builder keeps the constant 0.3),
            plus two conditions only this set has (robot_conditions.py):
              zero_shift_range  the brain's hidden zero shift as an unknown: per env, once at creation, the real
                                joint sits d from where the policy believes it, d uniform in [0, shift] (left hip roll
                                [0, 0.096], right hip roll [-0.061, 0], head roll [-0.061, 0], head yaw [-0.087, 0]
                                rad), through mjlab's encoder bias (bias -= d; actor joint_pos and action target both
                                in the believed frame, critic true), on top of the stock +-0.015 rad bias
              imu_max_lag       actor base_ang_vel and projected_gravity 0 to 2 policy ticks old (official 0 to 1)
  run_record.json "robot_conditions" records the dict applied (route_conditions()).
`stock` takes only `official`: the conditions were measured on the current robot and no stock builder applies them.

Run controls added around the recipe (they change nothing it computes):
  - checkpoints and an ONNX (runner export path, normalizer baked in, metadata attached) every --save-interval
    iterations as model_<it>.pt / model_<it>.onnx, plus the final iteration;
  - every --metric-interval iterations one row in metrics.jsonl: mean speed along the command (body vx) over envs
    commanded forward faster than 0.15 m/s, mean episode length, track_linear_velocity and track_angular_velocity
    weighted reward per second; status.json is the heartbeat (stage, iteration, last update);
  - --max-hours wall cap (stops cleanly with a final checkpoint and ONNX);
  - --early-stop-iter N / --early-stop-speed S: at iteration N, if the forward speed of the last metric window is
    below S, stop, write FAILED and exit 2 (2026-11-12: a from-scratch pair learned to stand and burned 10 h).
Every run ends with STATUS (DONE / FAILED and the reason) and run_record.json.

--init-checkpoint PATH (continue a finished run; default none = the from-scratch route above, unchanged):
  the env and runner are built exactly as above (same --model / --conditions), then the checkpoint is loaded through
  the runner's own load (mjlab/rl/runner.py MjlabOnPolicyRunner.load -> rsl_rl PPO.load): actor (weights, action std,
  observation normaliser), critic (weights, normaliser), Adam state, iteration, and env.common_step_counter, so every
  step-based curriculum is at the stage the source run ended at (S1: 96000 steps, all final stages are <= 48000).
  The worker refuses MICRODUCK_WARM_START (it would zero the counter) and stops if the counter after the load is below
  (source iteration + 1) x num_steps_per_env. The PPO adaptive learning rate starts at the restored optimizer's lr
  (rsl_rl would otherwise restart it at the cfg's 1e-3). The action std is the loaded one (never reset; logged).
  Numbering continues the source: S1 ends at iteration 3999, so the continuation trains iterations 4000.. and saves
  model_4000.pt, model_4250.pt, ..., the last one model_<3999 + max_iters>.pt, in its own run directory.
  --max-iters is the number of MORE iterations; --early-stop-iter N is relative to this run's first iteration (checked
  at absolute iteration first + N, S1 continuation: 4000 + N; from scratch first = 0, so N stays absolute as before);
  --metric-interval and --save-interval stay on the absolute iteration number.
  --init-sha256 HEX (optional) must match the file's sha256 (kaggle_job.py passes it).

--model also takes the opt-in foot-contact variants FOOT_VARIANTS (battery-v2m5-w918-sole1-20261123 crowned sole,
battery-v2m5-w918-box1-20261123 flat box): the current robot's parts and 918 g plus the layer their manifest's
"foot_contact" key names (cmu/model/foot_contact.py, as the judging sim loads it; tasks/cmu_asbuilt.py
apply_foot_contact_to_cfg points collisions, the feet contact sensor and the foot friction randomisation at the footc
geoms and raises the spawn by the layer's drop). --model current builds exactly what it built before (no layer).

Straight-walk commands as the robot's brain gives them (src/mjlab_microduck/heading_hold.py; all options at their
defaults leave the recipe's command cfg untouched):
  --press-speed-share Q, --press-speed V (0, 0.25): a forward zero-turn env gets vx = V with probability Q per resample.
  --heading-hold SHARE (0): that share of the forward zero-turn envs gets its yaw command from heading feedback.
  --heading-hold-law p|brain (p): p = clip(K * wrap(own heading at resample - heading), -1, 1); brain = walk_brain.py's
      straight-walk PI law with its anti-windup, a trim and a 0-2 tick old heading (that module's docstring).
  --heading-hold-gain LO HI (3.0 3.0 = HEADING_KP, fixed): with LO < HI, K is drawn per env at each resample.
  --heading-hold-trim T (0.0, law brain only): trim uniform in [-T, T] per resample.
  Every --metric-interval the metrics row carries hold_share, hold_abs_wz, hold_sd_wz, hold_clamp_share and
  press_share (window means of per-tick values; shares of all envs); run_record.json "commands" holds the settings.

Body check (written before any training step): run_record.json "body" holds the mass manifest path, its sha256, the
foot-contact layer (or null) and the total mass of the built env model; the log prints them. --expect-manifest-sha HEX
exits with status 3 before the env is built when the manifest's sha256 differs; status 3 also follows, before learning,
when the built model's footc geoms do not match the layer the manifest names. The mass is recorded, not checked (the
w918 manifests build 918.0 g; tests/test_foot_contact_training.py checks that).

Flight pack (--model flightpack-v1-w918-20261124, PACK_VARIANTS; every option off by default, other models refuse them;
terms in tasks/mdp.py pack_thrust / pack_mass_ramp, wired by tasks/cmu_asbuilt.py apply_pack_options):
  --thrust-lift LO HI       fan thrust event: lift/weight target per env in [LO, HI] (<= 0.75), redrawn every 2-6 s,
                            --thrust-zero-share (0.25) envs at zero, --thrust-drop-share (0.15) of redraws drop to zero
                            mid-hold, --thrust-lag (thrust_sites.json, 0.10 s) first-order, --thrust-imbalance (0.10)
                            per-fan share spread; force at each fan site along its trunk-fixed axis.
  --thrust-told             62nd actor input: lift estimate 2-6 ticks old with sd 0.05 noise (0 while fans are off);
                            critic gets the true lift; a 61-input --init-checkpoint is padded (pad_checkpoint_for_told).
  --pack-ramp-iters N       pack mass, COM, inertia ramp from the 918 g base to full over this run's first N iterations.
  --pack-levels N           per-env pack levels 0..N-1 (fraction k/(N-1)) instead of the ramp: start 0..--pack-level-init-max,
                            at each episode end up one (no fall, speed along forward command >= --pack-gate-speed),
                            down one (fall, or speed < half the gate), only for envs with >= 2 s of forward command;
                            past the top redrawn over all levels. Metrics rows: level mean, top share, per-band
                            (levels 0-3, 4-7, 8-10) forward speed and falls per episode.
  --cmd-progress-weight W   reward W x clip(v along command / commanded speed, 0, 1) per s on ticks with a planar
                            command above 0.1 m/s (mdp.cmd_progress; any model).
  --max-lr X                cap on the adaptive learning rate (any model).
Walking command options of tasks/cmu_press_commands.py are added when that module exists (press_module()).

Kaggle: `kaggle_job.py push scratch ...` (cmu/finetune/kaggle/README.md). kernel_entry.py passes --root, --gpu and
--rate-per-hour; the last is accepted and ignored (no dollar cap on Kaggle; --max-hours is the cap).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

TASK = "Mjlab-Velocity-Flat-MicroDuck"
FOOT_VARIANTS = ("battery-v2m5-w918-sole1-20261123", "battery-v2m5-w918-box1-20261123")  # opt-in, never the default
PACK_VARIANTS = ("flightpack-v1-w918-20261124",)  # opt-in flight-pack body (tasks/cmu_asbuilt.py PACK_VARIANTS)
MODELS = ("stock", "current") + FOOT_VARIANTS + PACK_VARIANTS
CONDITIONS = ("official", "gyro", "robot")
FWD_CMD_MIN = 0.15  # m/s: an env counts as commanded forward above this (the worker's "moving" band)


class StopTraining(Exception):
    pass


def make_env_cfg(model: str, play: bool = False, conditions: str = "official", heading_hold: float = 0.0,
                 heading_hold_gain=(3.0, 3.0), heading_hold_law: str = "p", heading_hold_trim: float = 0.0,
                 press_speed_share: float = 0.0, press_speed: float = 0.25):
    """The env cfg of the route: the upstream velocity cfg, with the current robot's masses for `current`, and for
    `current` the robot conditions chosen by `conditions` (module docstring); heading_hold > 0 or press_speed_share > 0
    swaps in the straight-walk command (heading_hold.py), both 0 leave the cfg exactly as before."""
    cfg = _make_env_cfg(model, play, conditions)
    if heading_hold > 0.0 or press_speed_share > 0.0:
        from mjlab_microduck.heading_hold import apply_heading_hold
        cfg.__dict__["_heading_hold_record"] = apply_heading_hold(
            cfg, heading_hold, heading_hold_gain, law=heading_hold_law, trim=heading_hold_trim,
            press_share=press_speed_share, press_speed=press_speed)
    return cfg


def _make_env_cfg(model: str, play: bool = False, conditions: str = "official"):
    if conditions not in CONDITIONS:
        raise SystemExit(f"--conditions {conditions!r}: choose from {CONDITIONS}")
    if model == "stock":
        if conditions != "official":
            raise SystemExit(f"--conditions {conditions} needs --model current: the robot conditions were measured on "
                             f"the current robot and no stock builder applies them")
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
        return make_microduck_velocity_env_cfg(play=play)
    if model == "current" or model in FOOT_VARIANTS or model in PACK_VARIANTS:
        from mjlab_microduck.tasks.cmu_asbuilt import make_cmu_velocity_env_cfg
        cfg = make_cmu_velocity_env_cfg(model, play=play, robot_conditions=False)
        if conditions != "official":
            from mjlab_microduck.robot_conditions import apply_robot_conditions
            apply_robot_conditions(cfg, route_conditions(conditions))
        return cfg
    raise SystemExit(f"--model {model!r}: choose from {MODELS}")


def route_conditions(conditions: str) -> dict:
    """The robot-condition dict the route applies (and run_record.json records) for `conditions`: ROBOT_CONDITIONS
    (robot) or none of it (gyro) except the leg-speed gyro noise, whose factor is drawn per env in
    GYRO_LEGSPEED_RANGE instead of the constant gyro_noise_legspeed."""
    from mjlab_microduck.robot_conditions import GYRO_LEGSPEED_RANGE, IMU_MAX_LAG, ROBOT_CONDITIONS, ZERO_SHIFT_RANGE
    if conditions == "official":
        return {}
    if conditions == "gyro":
        return {"gyro_noise_legspeed_range": GYRO_LEGSPEED_RANGE}
    if conditions == "robot":
        c = {k: v for k, v in ROBOT_CONDITIONS.items() if k != "gyro_noise_legspeed"}
        return dict(c, gyro_noise_legspeed_range=GYRO_LEGSPEED_RANGE, zero_shift_range=dict(ZERO_SHIFT_RANGE),
                    imu_max_lag=IMU_MAX_LAG)
    raise SystemExit(f"--conditions {conditions!r}: choose from {CONDITIONS}")


REPO_ROOT = Path(__file__).resolve().parents[3]


def body_manifest(model: str) -> dict:
    """The mass manifest a --model builds from: repository path, sha256 and the foot-contact layer it names. Read from
    the files the env builder reads (cmu/model/apply_masses.py), before anything is built."""
    if model == "stock":
        return dict(variant="stock", mass_manifest=None, mass_manifest_sha256=None, foot_contact=None)
    from mjlab_microduck.tasks.cmu_asbuilt import load_apply_masses
    am = load_apply_masses()
    variant = am.resolve_variant(model)
    path = am.variant_manifest_path(variant)
    layer = am.variant_foot_contact(variant)
    return dict(variant=variant, mass_manifest=str(Path(path).resolve().relative_to(REPO_ROOT)),
                mass_manifest_sha256=sha256_file(path), foot_contact=layer["name"] if layer else None)


def check_expected_manifest(body: dict, expect: str | None) -> None:
    """--expect-manifest-sha: exit 3 before the env is built when the manifest is not the one the launch names."""
    if expect is None:
        return
    if body["mass_manifest_sha256"] != expect.strip().lower():
        print(f"[scratch] BODY CHECK FAILED: manifest {body['mass_manifest']} sha256 {body['mass_manifest_sha256']} "
              f"!= --expect-manifest-sha {expect}", flush=True)
        raise SystemExit(3)


def built_body(env, body: dict) -> dict:
    """What the built env model reports: total mass (g, nominal, before startup randomisation) and the footc geoms."""
    import mujoco
    m = env.sim.mj_model
    names = [m.geom(i).name for i in range(m.ngeom)]
    footc = sorted(n for n in names if n.split("/")[-1].startswith("footc_"))
    return dict(total_mass_g=round(float(m.body_mass.sum()) * 1000, 2), footc_geoms=footc,
                mujoco=mujoco.__version__)


def make_agent_cfg(run_name: str, seed: int, max_iters: int, save_interval: int):
    """MicroduckRlCfg (the task's registered runner cfg) with only the bookkeeping fields set."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg
    cfg = copy.deepcopy(MicroduckRlCfg)
    cfg.seed = seed
    cfg.max_iterations = max_iters
    cfg.save_interval = save_interval
    cfg.logger = "tensorboard"
    cfg.run_name = run_name
    return cfg


# fields make_agent_cfg sets; every other runner field must equal MicroduckRlCfg (the test checks it)
AGENT_BOOKKEEPING = ("seed", "max_iterations", "save_interval", "logger", "run_name")
WARM_START_ENV = "MICRODUCK_WARM_START"  # mdp.py Patch 5: zeroes the step counter after a load; refused here


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_init_checkpoint(path, expect_sha256=None) -> dict:
    """Before anything is built: the file exists, its hash matches, it is a full rsl_rl checkpoint with a step counter,
    and the warm-start patch is off. Returns path, sha256, source iteration and source step counter."""
    import torch
    if os.environ.get(WARM_START_ENV, "") not in ("", "0"):
        raise SystemExit(f"{WARM_START_ENV}={os.environ[WARM_START_ENV]} would reset common_step_counter to 0 after the "
                         f"load (curricula restart); --init-checkpoint continues a run, unset it")
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--init-checkpoint {path}: no such file")
    sha = sha256_file(p)
    if expect_sha256 and sha != expect_sha256:
        raise SystemExit(f"--init-checkpoint {path}: sha256 {sha} != --init-sha256 {expect_sha256}")
    d = torch.load(p, map_location="cpu", weights_only=False)
    missing = [k for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter", "infos") if k not in d]
    if missing:
        raise SystemExit(f"--init-checkpoint {path}: not a full rsl_rl checkpoint (missing {missing})")
    counter = ((d["infos"] or {}).get("env_state") or {}).get("common_step_counter")
    if counter is None:
        raise SystemExit(f"--init-checkpoint {path}: no infos.env_state.common_step_counter (not saved by mjlab's runner)")
    return dict(path=str(p.resolve()), sha256=sha, source_iter=int(d["iter"]), source_common_step_counter=int(counter))


def load_init_checkpoint(runner, info: dict, num_steps_per_env: int) -> dict:
    """Load through the runner's own path (MjlabOnPolicyRunner.load: PPO.load of actor, critic, optimizer and the
    iteration, then env.unwrapped.common_step_counter), assert the counter, continue the numbering and sync the
    adaptive lr to the restored optimizer. Returns what was restored (for run_record/status)."""
    import torch
    runner.load(info["path"], map_location=runner.device)
    u = runner.env.unwrapped
    floor = (info["source_iter"] + 1) * num_steps_per_env
    if u.common_step_counter < floor or u.common_step_counter != info["source_common_step_counter"]:
        raise SystemExit(f"after loading {info['path']}: common_step_counter {u.common_step_counter}, expected "
                         f"{info['source_common_step_counter']} (>= {floor}); curricula would not be at the source's stage")
    if runner.current_learning_iteration != info["source_iter"]:
        raise SystemExit(f"after loading: iteration {runner.current_learning_iteration} != checkpoint {info['source_iter']}")
    runner.current_learning_iteration = info["source_iter"] + 1  # learn() runs source+1 .. source+max_iters
    alg = runner.alg
    lr_saved = alg.optimizer.param_groups[0]["lr"]
    lr_cfg = alg.learning_rate
    alg.learning_rate = lr_saved  # rsl_rl keeps its own copy (cfg 1e-3) and writes it into the optimizer each update
    # output_std needs a sampled distribution; read the loaded parameter itself (scalar noise: std_param, log: exp)
    sd = alg.get_policy().state_dict()
    if "distribution.std_param" in sd:
        std = sd["distribution.std_param"].detach().float().flatten().cpu()
    elif "distribution.log_std_param" in sd:
        std = torch.exp(sd["distribution.log_std_param"].detach().float()).flatten().cpu()
    else:
        raise SystemExit(f"after loading: no distribution std parameter in the actor ({sorted(sd)[:8]} ...)")
    return dict(**info, common_step_counter_after_load=int(u.common_step_counter),
                first_iteration=runner.current_learning_iteration, lr_restored=lr_saved, lr_cfg=lr_cfg,
                action_std_mean=round(float(std.mean()), 4), action_std_min=round(float(std.min()), 4),
                action_std_max=round(float(std.max()), 4))


def press_module():
    """tasks/cmu_press_commands.py (owned by the command work) or None when it is absent."""
    try:
        from mjlab_microduck.tasks import cmu_press_commands
    except ImportError:
        return None
    return cmu_press_commands


def pad_checkpoint_for_told(src: str, dst) -> dict:
    """Write a copy of a 61-input checkpoint whose actor and critic take one more input at the end (the lift estimate):
    first-layer weight gets a zero column, the obs normaliser mean 0 / var 1 / std 1 for it, and Adam's moments for
    those two weights a zero column, so the actor's output is unchanged while the new input is 0. Returns the widths."""
    import torch
    d = torch.load(src, map_location="cpu", weights_only=False)
    widths = {}
    for key in ("actor_state_dict", "critic_state_dict"):
        sd = d[key]
        w = sd["mlp.0.weight"]
        widths[key] = (int(w.shape[1]), int(w.shape[1]) + 1)
        sd["mlp.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 1, dtype=w.dtype)], dim=1)
        for name, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
            if name in sd:
                v = sd[name]
                sd[name] = torch.cat([v, torch.full((*v.shape[:-1], 1), fill, dtype=v.dtype)], dim=-1)
    st = d["optimizer_state_dict"]["state"]
    old = {widths["actor_state_dict"][0], widths["critic_state_dict"][0]}
    hit = 0
    for i, s in st.items():
        for k in ("exp_avg", "exp_avg_sq"):
            v = s.get(k)
            if v is not None and v.dim() == 2 and v.shape[1] in old and v.shape[0] == 512:
                s[k] = torch.cat([v, torch.zeros(v.shape[0], 1, dtype=v.dtype)], dim=1)
                hit += 1
    if hit != 4:
        raise SystemExit(f"pad_checkpoint_for_told: expected 4 first-layer Adam moments, padded {hit}")
    torch.save(d, dst)
    return dict(path=str(dst), widths=widths)


def cap_learning_rate(alg, max_lr: float) -> None:
    """Upper bound on PPO's adaptive learning rate: every assignment to alg.learning_rate (rsl_rl's adaptive schedule,
    the worker's resume sync) is clamped to max_lr, and the optimizer follows."""
    cls = type(alg)

    class CappedLr(cls):
        @property
        def learning_rate(self):
            return self.__dict__["_capped_lr"]

        @learning_rate.setter
        def learning_rate(self, v):
            self.__dict__["_capped_lr"] = min(float(v), float(max_lr))

    lr = alg.__dict__.pop("learning_rate")
    alg.__class__ = CappedLr
    alg.learning_rate = lr
    for g in alg.optimizer.param_groups:
        g["lr"] = alg.learning_rate


def fall_cost_term(env, term_name: str = "fell_over"):
    """1 / step_dt on the tick an env terminates by its fell_over termination (not on a tick that is also a time-out),
    else 0. The termination manager runs before the reward manager in the same step (mjlab ManagerBasedRlEnv.step),
    so this reads the current tick's termination. With weight -C and the reward manager's dt scaling
    (scale_rewards_by_dt), the reward PPO receives on that tick is exactly -C."""
    tm = env.termination_manager
    fell = tm.get_term(term_name) & ~tm.time_outs
    return fell.float() / env.step_dt


def apply_fall_cost(cfg, cost: float) -> dict:
    """--fall-cost C: add reward term fall_cost (weight -C on fall_cost_term). 0 = no term (cfg unchanged)."""
    if not cost:
        return {}
    from mjlab.managers import RewardTermCfg
    assert cost > 0 and "fell_over" in cfg.terminations and not cfg.terminations["fell_over"].time_out, cost
    assert "fall_cost" not in cfg.rewards
    assert getattr(cfg, "scale_rewards_by_dt", True), "fall_cost_term assumes dt-scaled rewards"
    cfg.rewards["fall_cost"] = RewardTermCfg(func=fall_cost_term, weight=-float(cost),
                                             params=dict(term_name="fell_over"))
    return dict(cost=float(cost), term="fell_over", rule="reward -C once on the tick an env terminates by fell_over "
                "(bad_orientation), not on time-outs; value 1/step_dt x weight -C, scaled by dt -> -C per fall")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--root", default="runs")
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--conditions", choices=CONDITIONS, default="official",
                    help="robot conditions in training (current model only): official = none (S1), gyro = leg-speed "
                         "gyro noise only, robot = all ROBOT_CONDITIONS")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iters", type=int, default=4000)
    ap.add_argument("--save-interval", type=int, default=250)
    ap.add_argument("--metric-interval", type=int, default=50)
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--early-stop-iter", type=int, default=2000, help="0 = off")
    ap.add_argument("--early-stop-speed", type=float, default=0.03, help="m/s along the forward command")
    ap.add_argument("--gpu", default="?")
    ap.add_argument("--rate-per-hour", type=float, default=0.0, help="accepted for kernel_entry.py; unused")
    ap.add_argument("--device", default="cuda:0", help="torch/warp device (default cuda:0; cpu for a local smoke)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="continue a finished run of this worker: full rsl_rl checkpoint (model_<it>.pt); --max-iters "
                         "MORE iterations, numbering continues, curricula stay at the source's stage (docstring)")
    ap.add_argument("--init-sha256", default=None, help="expected sha256 of --init-checkpoint (optional)")
    ap.add_argument("--heading-hold", type=float, default=0.0,
                    help="share of forward zero-turn envs whose yaw command is heading feedback (0 = off, the recipe)")
    ap.add_argument("--heading-hold-gain", type=float, nargs=2, default=[3.0, 3.0], metavar=("LO", "HI"),
                    help="heading feedback gain K; LO < HI draws K per env at each resample (default 3.0 3.0, fixed)")
    ap.add_argument("--heading-hold-law", choices=("p", "brain"), default="p",
                    help="p = proportional (default); brain = walk_brain.py straight-walk PI law, trim, IMU age")
    ap.add_argument("--heading-hold-trim", type=float, default=0.0,
                    help="law brain: trim drawn uniform in [-T, T] at each resample (default 0)")
    ap.add_argument("--press-speed-share", type=float, default=0.0,
                    help="share of forward zero-turn envs commanded at --press-speed (default 0 = off)")
    ap.add_argument("--press-speed", type=float, default=0.25, help="vx of the press-speed envs (m/s, default 0.25)")
    ap.add_argument("--expect-manifest-sha", default=None,
                    help="sha256 the body's mass manifest must have; exit 3 before training otherwise")
    # flight-pack options (PACK_VARIANTS only; all off by default)
    ap.add_argument("--thrust-lift", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="pack model: fan lift as a fraction of weight, target drawn per env in [LO, HI] (<= 0.75) and "
                         "redrawn every 2-6 s (default off = no thrust event)")
    ap.add_argument("--thrust-zero-share", type=float, default=0.25, help="share of envs (per episode) held at zero lift")
    ap.add_argument("--thrust-drop-share", type=float, default=0.15,
                    help="share of redraws whose target drops to zero at a random time inside the hold")
    ap.add_argument("--thrust-lag", type=float, default=None, help="first-order thrust lag, s (default thrust_sites.json)")
    ap.add_argument("--thrust-imbalance", type=float, default=0.10, help="per-episode fan share spread (fraction)")
    ap.add_argument("--thrust-told", action="store_true",
                    help="append a delayed (2-6 ticks) noisy (sd 5 %% of weight) lift estimate as the 62nd actor input "
                         "(critic: true lift); a 61-input --init-checkpoint is padded with a zero column")
    ap.add_argument("--pack-ramp-iters", type=int, default=0,
                    help="pack model: the pack's mass, COM and inertia ramp from the base robot to full over the first "
                         "N iterations of this run (0 = full pack from the first step)")
    ap.add_argument("--pack-levels", type=int, default=0,
                    help="pack model: per-env pack levels 0..N-1 (fraction k/(N-1) of the pack's mass, COM, inertia), "
                         "set at reset, promoted/demoted at episode end (mdp.pack_mass_ramp.update_levels); 0 = off")
    ap.add_argument("--pack-level-init-max", type=int, default=5, help="start levels drawn uniformly in 0..this")
    ap.add_argument("--pack-gate-speed", type=float, default=0.04,
                    help="promotion speed along the forward command (m/s); below half of it counts as standing")
    ap.add_argument("--cmd-progress-weight", type=float, default=0.0,
                    help="weight (per s) of the command-progress reward term mdp.cmd_progress (0 = off)")
    ap.add_argument("--upright-progress-gate", action="store_true",
                    help="under a walk command multiply the positive upright reward by clip(mean speed along the "
                         "command over the last --upright-gate-window s / --upright-gate-vref, 0, 1); unchanged at "
                         "zero command (mdp.upright_progress_gate; default off)")
    ap.add_argument("--progress-gate-terms", default=None,
                    help="comma list of positive reward terms to gate the same way as --upright-progress-gate (same "
                         "multiplier, window and v_ref; unchanged at zero command), e.g. "
                         "upright,track_angular_velocity,pose,head_pose_tracking; replaces --upright-progress-gate")
    ap.add_argument("--fall-cost", type=float, default=0.0,
                    help="one-off reward cost C (reward units, as PPO receives it) on the tick an env terminates by "
                         "falling over (fell_over), never on time-outs (default 0 = off, cfg unchanged)")
    ap.add_argument("--upright-gate-window", type=float, default=1.0, help="gate window, s (default 1.0)")
    ap.add_argument("--upright-gate-vref", type=float, default=0.10, help="gate reference speed, m/s (default 0.10)")
    ap.add_argument("--max-lr", type=float, default=None, help="upper bound on the adaptive learning rate (default none)")
    press = press_module()
    if press is not None:
        press.add_press_arguments(ap)
    args = ap.parse_args(argv)
    pack = args.model in PACK_VARIANTS
    if not pack and (args.thrust_lift is not None or args.thrust_told or args.pack_ramp_iters):
        ap.error(f"--thrust-lift / --thrust-told / --pack-ramp-iters need --model {PACK_VARIANTS[0]}")
    if args.thrust_lift is not None and not 0.0 <= args.thrust_lift[0] <= args.thrust_lift[1] <= 0.75:
        ap.error("--thrust-lift needs 0 <= LO <= HI <= 0.75")
    if args.thrust_told and args.thrust_lift is None:
        ap.error("--thrust-told needs --thrust-lift")
    for name in ("thrust_zero_share", "thrust_drop_share"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            ap.error(f"--{name.replace('_', '-')} is a share in [0, 1]")
    if not 0.0 <= args.thrust_imbalance < 1.0:
        ap.error("--thrust-imbalance must lie in [0, 1)")
    if args.thrust_lag is not None and not 0.0 < args.thrust_lag <= 1.0:
        ap.error("--thrust-lag must lie in (0, 1] s")
    if args.pack_ramp_iters < 0:
        ap.error("--pack-ramp-iters must be >= 0")
    if args.pack_levels and not pack:
        ap.error(f"--pack-levels needs --model {PACK_VARIANTS[0]}")
    if args.pack_levels and args.pack_ramp_iters:
        ap.error("--pack-levels replaces --pack-ramp-iters; give one")
    if args.pack_levels and not (2 <= args.pack_levels <= 101 and 0 <= args.pack_level_init_max < args.pack_levels):
        ap.error("--pack-levels needs 2..101 levels and 0 <= --pack-level-init-max < --pack-levels")
    if not 0.0 < args.pack_gate_speed <= 1.0:
        ap.error("--pack-gate-speed must lie in (0, 1] m/s")
    if args.cmd_progress_weight < 0.0:
        ap.error("--cmd-progress-weight must be >= 0")
    if not (0.0 < args.upright_gate_window <= 10.0 and 0.0 < args.upright_gate_vref <= 1.0):
        ap.error("--upright-gate-window must lie in (0, 10] s and --upright-gate-vref in (0, 1] m/s")
    if not 0.0 <= args.fall_cost <= 1000.0:
        ap.error("--fall-cost must lie in [0, 1000]")
    gate_terms = None
    if args.progress_gate_terms is not None:
        if args.upright_progress_gate:
            ap.error("--progress-gate-terms replaces --upright-progress-gate; give one")
        gate_terms = tuple(t.strip() for t in args.progress_gate_terms.split(",") if t.strip())
        if not gate_terms or len(set(gate_terms)) != len(gate_terms):
            ap.error("--progress-gate-terms needs one or more distinct reward term names")
    if not (args.upright_progress_gate or gate_terms) and (args.upright_gate_window != 1.0
                                                            or args.upright_gate_vref != 0.10):
        ap.error("--upright-gate-window / --upright-gate-vref need --upright-progress-gate or --progress-gate-terms")
    if args.max_lr is not None and not 0.0 < args.max_lr <= 1e-2:
        ap.error("--max-lr must lie in (0, 0.01]")
    if not 0.0 <= args.heading_hold <= 1.0:
        ap.error("--heading-hold is a share in [0, 1]")
    if not 0.0 < args.heading_hold_gain[0] <= args.heading_hold_gain[1]:
        ap.error("--heading-hold-gain needs 0 < LO <= HI")
    if args.heading_hold_gain != [3.0, 3.0] and args.heading_hold == 0.0:
        ap.error("--heading-hold-gain needs --heading-hold > 0")
    if args.heading_hold_law != "p" and args.heading_hold == 0.0:
        ap.error("--heading-hold-law needs --heading-hold > 0")
    if args.heading_hold_trim != 0.0 and args.heading_hold_law != "brain":
        ap.error("--heading-hold-trim needs --heading-hold-law brain")
    if not 0.0 <= args.heading_hold_trim <= 1.0:
        ap.error("--heading-hold-trim is a turn input in [0, 1]")
    if not 0.0 <= args.press_speed_share <= 1.0:
        ap.error("--press-speed-share is a share in [0, 1]")
    if args.press_speed != 0.25 and args.press_speed_share == 0.0:
        ap.error("--press-speed needs --press-speed-share > 0")
    if not 0.0 < args.press_speed <= 0.4:
        ap.error("--press-speed must lie in the trained forward range (0, 0.4] m/s")
    if args.model == "stock" and args.expect_manifest_sha:
        ap.error("--expect-manifest-sha needs a CMU model (stock has no mass manifest)")
    if args.model == "stock" and args.conditions != "official":
        ap.error(f"--conditions {args.conditions} needs --model current (no stock builder applies robot conditions)")
    if args.init_sha256 and not args.init_checkpoint:
        ap.error("--init-sha256 needs --init-checkpoint")
    if args.max_iters < 1:
        ap.error("--max-iters must be >= 1")
    body = body_manifest(args.model)
    print(f"[scratch] body: variant {body['variant']}, manifest {body['mass_manifest']} sha256 "
          f"{body['mass_manifest_sha256']}, foot contact {body['foot_contact']}", flush=True)
    check_expected_manifest(body, args.expect_manifest_sha)
    init = check_init_checkpoint(args.init_checkpoint, args.init_sha256) if args.init_checkpoint else None

    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401  (registers Mjlab-*-MicroDuck through the plugin entry point)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = args.device
    t_start = time.time()
    run_dir = Path(args.root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.glob("model_*.pt")):
        raise SystemExit(f"{run_dir} already holds checkpoints; this route never resumes (curricula would restart)"
                         + ("; --init-checkpoint starts a new run: give it a new --run-name" if init else ""))
    if init and Path(init["path"]).parent == run_dir.resolve():
        raise SystemExit(f"--init-checkpoint {init['path']} lies in the new run directory {run_dir}; use a new --run-name")

    def status(stage, **kw):
        extra = {"init_checkpoint": record.get("init_checkpoint", init)} if init else {}
        (run_dir / "status.json").write_text(json.dumps(dict(stage=stage, run_name=args.run_name,
                                                             conditions=args.conditions, started=t0_iso,
                                                             updated=time.strftime("%Y-%m-%dT%H:%M:%S"), **extra, **kw),
                                                        indent=1, default=str))

    record = {}

    t0_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    status("build")
    env_cfg = make_env_cfg(args.model, conditions=args.conditions, heading_hold=args.heading_hold,
                           heading_hold_gain=tuple(args.heading_hold_gain), heading_hold_law=args.heading_hold_law,
                           heading_hold_trim=args.heading_hold_trim, press_speed_share=args.press_speed_share,
                           press_speed=args.press_speed)
    if press is not None:
        press.apply_press_options(env_cfg, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = make_agent_cfg(args.run_name, args.seed, args.max_iters, args.save_interval)
    pack_record = None
    if pack:
        from mjlab_microduck.tasks.cmu_asbuilt import apply_pack_options
        pack_record = apply_pack_options(
            env_cfg, args.model, thrust_lift=args.thrust_lift, zero_share=args.thrust_zero_share,
            drop_share=args.thrust_drop_share, lag=args.thrust_lag, imbalance=args.thrust_imbalance,
            told=args.thrust_told, ramp_steps=args.pack_ramp_iters * agent_cfg.num_steps_per_env,
            levels=args.pack_levels, level_init_max=args.pack_level_init_max, gate_speed=args.pack_gate_speed)
    progress_record = None
    if args.cmd_progress_weight:
        from mjlab_microduck.tasks.cmu_asbuilt import apply_cmd_progress
        progress_record = apply_cmd_progress(env_cfg, args.cmd_progress_weight)
    gate_record = None
    if args.upright_progress_gate:
        from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
        gate_record = apply_upright_progress_gate(env_cfg, window_s=args.upright_gate_window,
                                                  v_ref=args.upright_gate_vref)
    elif gate_terms:
        from mjlab_microduck.tasks.cmu_asbuilt import apply_upright_progress_gate
        missing = [t for t in gate_terms if t not in env_cfg.rewards]
        if missing:
            raise SystemExit(f"--progress-gate-terms: no reward term {missing} (have {sorted(env_cfg.rewards)})")
        gate_record = apply_upright_progress_gate(env_cfg, window_s=args.upright_gate_window,
                                                  v_ref=args.upright_gate_vref, terms=gate_terms)
    fall_record = apply_fall_cost(env_cfg, args.fall_cost)

    record = dict(run_name=args.run_name, task=TASK, route="scratch (Pollen recipe unchanged)", model=args.model,
                  conditions=args.conditions, seed=args.seed, num_envs=args.num_envs, max_iters=args.max_iters, save_interval=args.save_interval,
                  max_hours=args.max_hours, early_stop=dict(iter=args.early_stop_iter, speed=args.early_stop_speed),
                  gpu=args.gpu, started=t0_iso)
    if init:
        record["route"] = "continue (Pollen recipe unchanged, from a finished run's checkpoint)"
        record["init_checkpoint"] = dict(init)
    if args.conditions != "official":
        record["robot_conditions"] = route_conditions(args.conditions)
        record["gyro_noise_legspeed_range_note"] = ("leg-speed gyro factor K drawn once per env at env creation, "
                                                    "uniform in gyro_noise_legspeed_range")
    if args.model == "current":
        from mjlab_microduck.tasks.cmu_asbuilt import resolve_variant
        cur = Path(__file__).resolve().parents[3] / "cmu/model/current.json"
        record["current"] = dict(variant=resolve_variant("current"), **{k: json.loads(cur.read_text())[k]
                                                                        for k in ("mass_manifest", "mass_manifest_sha256")})
    record["body"] = dict(body, expect_manifest_sha=args.expect_manifest_sha)
    record["command_options"] = dict(heading_hold=args.heading_hold, heading_hold_law=args.heading_hold_law,
                                     heading_hold_gain=list(args.heading_hold_gain),
                                     heading_hold_trim=[-args.heading_hold_trim, args.heading_hold_trim],
                                     heading_hold_imu_age_ticks=[0, 2] if args.heading_hold_law == "brain" else [0, 0],
                                     press_speed_share=args.press_speed_share, press_speed=args.press_speed)
    if args.heading_hold > 0.0 or args.press_speed_share > 0.0:
        record["commands"] = env_cfg.__dict__["_heading_hold_record"]
    if press is not None:
        record["press"] = press.press_record(args)
    if pack:
        record["pack"] = dict(pack_record, thrust_lift=args.thrust_lift, thrust_told=args.thrust_told,
                              pack_ramp_iters=args.pack_ramp_iters, pack_levels=args.pack_levels)
    if progress_record:
        record["cmd_progress"] = progress_record
    if gate_record:
        record["upright_progress_gate"] = gate_record
    if fall_record:
        record["fall_cost"] = fall_record
    if args.max_lr is not None:
        record["max_lr"] = args.max_lr
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    record["body"].update(built_body(env, body))
    print(f"[scratch] body built: total mass {record['body']['total_mass_g']:.2f} g, foot contact "
          f"{body['foot_contact']}, footc geoms {record['body']['footc_geoms']}", flush=True)
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
    if bool(record["body"]["footc_geoms"]) != bool(body["foot_contact"]):
        print(f"[scratch] BODY CHECK FAILED: manifest foot contact {body['foot_contact']} but built footc geoms "
              f"{record['body']['footc_geoms']}", flush=True)
        raise SystemExit(3)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_base = load_runner_cls(TASK) or MjlabOnPolicyRunner
    out = {"stop": None, "failed": False, "onnx_missing": []}

    class ScratchRunner(runner_base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            u = self.env.unwrapped
            robot, cm, rm = u.scene["robot"], u.command_manager, u.reward_manager
            names = list(rm.active_terms)
            idx = {n: names.index(n) for n in ("track_linear_velocity", "track_angular_velocity") if n in names}
            self.win = dict(speed=0.0, speed_n=0, steps=0, **{n: 0.0 for n in idx})
            twist = cm.get_term("twist")
            self.cmd_stats = getattr(twist, "tick_stats", None)  # heading_hold.py command only
            self.cwin = {}  # name -> [sum, count] of per-tick command stats
            self.last_row = None
            self.first_it = 0  # this run's first iteration (source iteration + 1 after --init-checkpoint)
            self.saves = 0
            orig_step = self.logger.process_env_step

            def hooked_step(rewards, dones, extras, intrinsic_rewards=None):
                cmd, v = cm.get_command("twist"), robot.data.root_link_lin_vel_b
                fwd = cmd[:, 0] > FWD_CMD_MIN
                if fwd.any():
                    self.win["speed"] += float(v[fwd, 0].mean())
                    self.win["speed_n"] += 1
                for n, i in idx.items():  # _step_reward = weighted term value per second
                    self.win[n] += float(rm._step_reward[:, i].mean())
                self.win["steps"] += 1
                if self.cmd_stats is not None:
                    for k, val in self.cmd_stats().items():
                        s = self.cwin.setdefault(k, [0.0, 0])
                        s[0] += val
                        s[1] += 1
                return orig_step(rewards, dones, extras, intrinsic_rewards)
            self.logger.process_env_step = hooked_step
            orig_log = self.logger.log

            def hooked_log(*a, **k):
                r = orig_log(*a, **k)
                self.after_iteration(int(k["it"]))
                return r
            self.logger.log = hooked_log

        def after_iteration(self, it):
            elapsed = time.time() - t_start
            if it % args.metric_interval == 0:
                w = self.win
                lb = self.logger.lenbuffer
                row = dict(iter=it, elapsed_s=round(elapsed, 1),
                           speed_along_fwd_cmd=round(w["speed"] / w["speed_n"], 4) if w["speed_n"] else None,
                           mean_ep_len_steps=round(float(np.mean(lb)), 1) if len(lb) else None,
                           mean_reward=round(float(np.mean(self.logger.rewbuffer)), 4) if len(self.logger.rewbuffer) else None,
                           **{f"{n}_per_s": round(w[n] / w["steps"], 4) for n in w if n.startswith("track_") and w["steps"]},
                           action_std=round(float(self.alg.get_policy().output_std.mean()), 4),
                           lr=self.alg.learning_rate, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
                           **{k: round(s / c, 4) for k, (s, c) in self.cwin.items() if c})
                self.cwin = {}
                if pack:  # flight-pack state at this iteration (instantaneous, all envs)
                    u = self.env.unwrapped
                    lift = getattr(u, "_pack_thrust_lift", None)
                    if lift is not None:
                        row["pack_lift_mean"] = round(float(lift.mean()), 4)
                    if args.pack_ramp_iters:
                        row["pack_ramp_fraction"] = round(
                            u.event_manager.get_term_cfg("pack_mass_ramp").func.last_fraction, 4)
                    if args.pack_levels:
                        row.update(u.event_manager.get_term_cfg("pack_mass_ramp").func.metrics())
                with open(run_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[scratch] {json.dumps(row)}", flush=True)
                self.last_row = row
                self.win = {k: 0 if k.endswith("_n") or k == "steps" else 0.0 for k in self.win}
                status("train", iter=it, last=row)
            if args.early_stop_iter and it - self.first_it == args.early_stop_iter:
                s = self.last_row and self.last_row.get("speed_along_fwd_cmd")
                if s is None or s < args.early_stop_speed:
                    out["failed"] = True
                    out["stop"] = (f"early stop at iteration {it}: speed along the forward command {s} m/s < "
                                   f"{args.early_stop_speed} (the recipe learned to stand on this body)")
            if not out["stop"] and elapsed >= args.max_hours * 3600:
                out["stop"] = f"wall cap {args.max_hours} h reached at iteration {it}"
            if out["stop"]:
                print("[scratch] STOP:", out["stop"], flush=True)
                raise StopTraining(out["stop"])

        def save(self, path, infos=None):
            super().save(path, infos)  # model_<it>.pt + <run_name>.onnx (VelocityOnPolicyRunner: normalizer + metadata)
            it = self.current_learning_iteration
            onnx = run_dir / f"{run_dir.name}.onnx"
            if onnx.exists():
                shutil.copy(onnx, run_dir / f"model_{it}.onnx")
            else:
                out["onnx_missing"].append(it)
                if self.saves == 0:
                    raise SystemExit("ONNX export failed at the first save; stopping before the GPU hours are spent")
            self.saves += 1

    runner = ScratchRunner(env, dataclasses.asdict(agent_cfg), str(run_dir), device)
    if args.max_lr is not None:
        cap_learning_rate(runner.alg, args.max_lr)
    if init:
        load_info = init
        if args.thrust_told:
            import torch
            w = torch.load(init["path"], map_location="cpu", weights_only=False)["actor_state_dict"]["mlp.0.weight"]
            want = env.unwrapped.observation_manager.group_obs_dim["actor"][0]
            if w.shape[1] == want - 1:
                padded = pad_checkpoint_for_told(init["path"], run_dir / "init_padded_told.pt")
                record["init_checkpoint_padded"] = padded
                load_info = dict(init, path=padded["path"])
            elif w.shape[1] != want:
                raise SystemExit(f"--init-checkpoint actor takes {w.shape[1]} inputs; the env gives {want}")
        loaded = load_init_checkpoint(runner, load_info, agent_cfg.num_steps_per_env)
        loaded["path"], loaded["sha256"] = init["path"], init["sha256"]
        runner.first_it = loaded["first_iteration"]
        record["init_checkpoint"] = loaded
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
        status("loaded")
        print(f"[scratch] continuing {loaded['path']} (sha256 {loaded['sha256'][:16]}, iteration {loaded['source_iter']}): "
              f"common_step_counter {loaded['common_step_counter_after_load']} (curricula at the source's stage), "
              f"action std mean {loaded['action_std_mean']} (min {loaded['action_std_min']}, max {loaded['action_std_max']}, "
              f"not reset), lr {loaded['lr_restored']:.3g} (cfg {loaded['lr_cfg']:.3g}); iterations "
              f"{runner.first_it}..{runner.first_it + args.max_iters - 1}", flush=True)
    if args.max_lr is not None:
        for g in runner.alg.optimizer.param_groups:  # the restored optimizer's lr, clamped as well
            g["lr"] = runner.alg.learning_rate
        print(f"[scratch] adaptive lr capped at {args.max_lr:.3g}; starts at {runner.alg.learning_rate:.3g}", flush=True)
    if pack and args.pack_ramp_iters:
        ramp = env.unwrapped.event_manager.get_term_cfg("pack_mass_ramp").func
        ramp.start_counter = int(env.unwrapped.common_step_counter)  # counted from this run's start, not the source's
        record["pack"]["mass_ramp_start_counter"] = ramp.start_counter
        (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
        print(f"[scratch] pack mass ramp over {args.pack_ramp_iters} iterations from step counter {ramp.start_counter}",
              flush=True)
    print(f"[scratch] {TASK} {'continued' if init else 'from scratch'}, model {args.model}, conditions {args.conditions}, "
          f"seed {args.seed}, {args.num_envs} envs, {args.max_iters} iterations, common_step_counter "
          f"{env.unwrapped.common_step_counter}", flush=True)
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
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
    verdict = "FAILED" if out["failed"] else "DONE"
    (run_dir / "STATUS").write_text(f"{verdict} {stop_reason}\n")
    status(verdict.lower(), iter=runner.current_learning_iteration, reason=stop_reason, last=runner.last_row)
    print(f"[scratch] {verdict}: {stop_reason}; final iteration {runner.current_learning_iteration}; {elapsed / 3600:.2f} h", flush=True)
    env.close()
    return 2 if out["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
