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
  gyro      official plus only the leg-speed gyro noise of ROBOT_CONDITIONS (src/mjlab_microduck/robot_conditions.py:
            actor base_ang_vel + N(0, (0.3 x rms leg joint speed)^2) per axis per tick, on top of the stock +-0.03)
  robot     make_cmu_velocity_env_cfg("current", robot_conditions=True): all six ROBOT_CONDITIONS
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

Kaggle: `kaggle_job.py push scratch ...` (cmu/finetune/kaggle/README.md). kernel_entry.py passes --root, --gpu and
--rate-per-hour; the last is accepted and ignored (no dollar cap on Kaggle; --max-hours is the cap).
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
from pathlib import Path

TASK = "Mjlab-Velocity-Flat-MicroDuck"
MODELS = ("stock", "current")
CONDITIONS = ("official", "gyro", "robot")
FWD_CMD_MIN = 0.15  # m/s: an env counts as commanded forward above this (the worker's "moving" band)


class StopTraining(Exception):
    pass


def make_env_cfg(model: str, play: bool = False, conditions: str = "official"):
    """The env cfg of the route: the upstream velocity cfg, with the current robot's masses for `current`, and for
    `current` the robot conditions chosen by `conditions` (module docstring)."""
    if conditions not in CONDITIONS:
        raise SystemExit(f"--conditions {conditions!r}: choose from {CONDITIONS}")
    if model == "stock":
        if conditions != "official":
            raise SystemExit(f"--conditions {conditions} needs --model current: the robot conditions were measured on "
                             f"the current robot and no stock builder applies them")
        from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg
        return make_microduck_velocity_env_cfg(play=play)
    if model == "current":
        from mjlab_microduck.tasks.cmu_asbuilt import make_cmu_velocity_env_cfg
        if conditions == "robot":
            return make_cmu_velocity_env_cfg("current", play=play, robot_conditions=True)
        cfg = make_cmu_velocity_env_cfg("current", play=play, robot_conditions=False)
        if conditions == "gyro":
            from mjlab_microduck.robot_conditions import ROBOT_CONDITIONS, apply_robot_conditions
            apply_robot_conditions(cfg, {"gyro_noise_legspeed": ROBOT_CONDITIONS["gyro_noise_legspeed"]})
        return cfg
    raise SystemExit(f"--model {model!r}: choose from {MODELS}")


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
    args = ap.parse_args(argv)
    if args.model == "stock" and args.conditions != "official":
        ap.error(f"--conditions {args.conditions} needs --model current (no stock builder applies robot conditions)")

    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401  (registers Mjlab-*-MicroDuck through the plugin entry point)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = "cuda:0"
    t_start = time.time()
    run_dir = Path(args.root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.glob("model_*.pt")):
        raise SystemExit(f"{run_dir} already holds checkpoints; this route never resumes (curricula would restart)")

    def status(stage, **kw):
        (run_dir / "status.json").write_text(json.dumps(dict(stage=stage, run_name=args.run_name,
                                                             conditions=args.conditions, started=t0_iso,
                                                             updated=time.strftime("%Y-%m-%dT%H:%M:%S"), **kw), indent=1))

    t0_iso = time.strftime("%Y-%m-%dT%H:%M:%S")
    status("build")
    env_cfg = make_env_cfg(args.model, conditions=args.conditions)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    agent_cfg = make_agent_cfg(args.run_name, args.seed, args.max_iters, args.save_interval)

    record = dict(run_name=args.run_name, task=TASK, route="scratch (Pollen recipe unchanged)", model=args.model,
                  conditions=args.conditions, seed=args.seed, num_envs=args.num_envs, max_iters=args.max_iters, save_interval=args.save_interval,
                  max_hours=args.max_hours, early_stop=dict(iter=args.early_stop_iter, speed=args.early_stop_speed),
                  gpu=args.gpu, started=t0_iso)
    if args.conditions != "official":
        from mjlab_microduck.robot_conditions import ROBOT_CONDITIONS
        record["robot_conditions"] = (dict(ROBOT_CONDITIONS) if args.conditions == "robot" else
                                      {"gyro_noise_legspeed": ROBOT_CONDITIONS["gyro_noise_legspeed"]})
    if args.model == "current":
        from mjlab_microduck.tasks.cmu_asbuilt import resolve_variant
        cur = Path(__file__).resolve().parents[3] / "cmu/model/current.json"
        record["current"] = dict(variant=resolve_variant("current"), **{k: json.loads(cur.read_text())[k]
                                                                        for k in ("mass_manifest", "mass_manifest_sha256")})
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
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
            self.last_row = None
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
                           lr=self.alg.learning_rate, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
                with open(run_dir / "metrics.jsonl", "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[scratch] {json.dumps(row)}", flush=True)
                self.last_row = row
                self.win = {k: 0 if k.endswith("_n") or k == "steps" else 0.0 for k in self.win}
                status("train", iter=it, last=row)
            if args.early_stop_iter and it == args.early_stop_iter:
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
                if it == 0:
                    raise SystemExit("ONNX export failed at the first save; stopping before the GPU hours are spent")

    runner = ScratchRunner(env, dataclasses.asdict(agent_cfg), str(run_dir), device)
    print(f"[scratch] {TASK} from scratch, model {args.model}, conditions {args.conditions}, seed {args.seed}, {args.num_envs} envs, "
          f"{args.max_iters} iterations, common_step_counter {env.unwrapped.common_step_counter}", flush=True)
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
