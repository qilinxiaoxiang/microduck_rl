"""DAgger distillation of the privileged teacher into a recurrent student
(extensions/flight-20261110/train/TEACHER-STUDENT-PLAN.md; 16-831 lecture 22: PPO teacher, history student by DAgger).

Run through flight_worker.py --distill (the Kaggle entry always starts flight_worker.py), or directly:
  python cmu/finetune/flight/distill_worker.py --run-name s1 --root <dir> --teacher-onnx <teacher.onnx> --core lstm \
      --manifest <flightpack-none.json> --max-iters 400 --max-hours 2.5 --max-dollars 1 --rate-per-hour 0

Env: the universal3 env (delta universal3, 61 actor inputs, payload in every env, head-pose commands at full range).
Each iteration is one segment of --bptt-steps control ticks (default 100 = 2 s) in every env:
  - the teacher labels every visited state from [the student's own 61 inputs | teacher.priv_obs(env)];
  - who drives: the teacher in all envs for the first --warmup-iters iterations (their falls and speed are the
    teacher's baseline in this env), then a share beta of the envs, beta falling linearly to 0 over --beta-iters
    (Ross 2011's mixture); the student drives the rest, half of them with Gaussian action noise --act-noise;
  - the student's state runs through every env, is zeroed where an episode ends, and is carried (detached) from one
    segment to the next, so the memory is not limited to one segment;
  - update: mean squared action error, backpropagated through the whole segment from the segment's start state.
Stops on task metrics only (never on the loss): time cap, or after --stop-after iterations two saves in a row with
the student-driven envs not stepping (speed along a 0.15-0.3 m/s command below max(0.03, half the teacher's)) or
falling more than 4 x the teacher's rate + 1 per env-minute.
Outputs in <root>/<run-name>/: model_<it>.pt and model_<it>.onnx at every save, <run-name>.onnx (final),
<run-name>_best.onnx (fewest student falls among saves with beta 0 and speed at least 0.8 x the teacher's),
progress.json / progress.csv, run_record.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_env(fw, delta_name, manifest, audit, remove_battery_kg, num_envs, seed, device, pin=None, mutate=None):
    """the worker's env for a delta (optionally pinned to one payload state; mutate(env_cfg) runs after the delta),
    wrapped for rsl_rl's step interface"""
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    env_cfg = load_env_cfg(fw.TASK, play=False)
    delta = dict(fw.DELTAS[delta_name])
    if pin:
        delta["payload"] = dict(delta["payload"], weights=pin)
    fw.apply_delta(env_cfg, delta)
    if mutate:
        mutate(env_cfg)
    env_cfg.scene.entities["robot"] = fw.apply_asbuilt_masses(env_cfg.scene.entities["robot"], manifest, audit, remove_battery_kg)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    venv = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(fw.TASK).clip_actions)
    env.common_step_counter = fw.CURRICULA_DONE_STEP
    return env, venv, delta


class Tally:
    """falls, env-steps and speed along the command for one group of envs (teacher-driven or student-driven)"""

    def __init__(self):
        self.steps = self.falls = self.mov_n = self.low_n = 0
        self.mov_sum = self.low_sum = 0.0

    def add(self, mask, fell, along, moving, low):
        self.steps += int(mask.sum())
        self.falls += int((fell & mask).sum())
        m, lo = moving & mask, low & mask
        self.mov_n += int(m.sum())
        self.mov_sum += float(along[m].sum())
        self.low_n += int(lo.sum())
        self.low_sum += float(along[lo].sum())

    def out(self):
        return dict(env_min=round(self.steps * 0.02 / 60, 2),
                    falls_per_env_min=round(self.falls / max(self.steps * 0.02 / 60, 1e-9), 3) if self.steps else None,
                    speed=round(self.mov_sum / self.mov_n, 4) if self.mov_n else None,
                    low_speed=round(self.low_sum / self.low_n, 4) if self.low_n else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--root", default="/vol/runs")
    ap.add_argument("--teacher-onnx", required=True)
    ap.add_argument("--core", choices=("lstm", "mamba"), required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--delta", default="universal3")
    ap.add_argument("--manifest", default="/root/assets/manifests/flightpack-none.json")
    ap.add_argument("--audit", default="/root/assets/existing-model-mass-audit.json")
    ap.add_argument("--remove-battery-kg", type=float, default=0.09)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--max-iters", type=int, default=400)
    ap.add_argument("--bptt-steps", type=int, default=100, help="control ticks per segment and per backpropagation")
    ap.add_argument("--warmup-iters", type=int, default=2, help="iterations with the teacher driving every env (baseline)")
    ap.add_argument("--beta-iters", type=int, default=40, help="iterations over which the teacher-driven share falls to 0")
    ap.add_argument("--act-noise", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--mb-envs", type=int, default=512, help="envs per minibatch (each a whole segment)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-final", type=float, default=2e-4)
    ap.add_argument("--save-interval", type=int, default=20)
    ap.add_argument("--stop-after", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--max-dollars", type=float, default=1.0)   # accepted for the shared launcher presets; unused
    ap.add_argument("--rate-per-hour", type=float, default=0.0)
    ap.add_argument("--gpu", default="?")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    import flight_worker as fl
    import recurrent as rc
    import teacher as tc
    fw = fl.fw
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")
    t_start = time.time()
    run_dir = Path(a.root) / a.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = a.device

    env, venv, delta = build_env(fw, a.delta, a.manifest, a.audit, a.remove_battery_kg, a.num_envs, a.seed, dev)
    # episodes start at random ages (as rsl_rl's init_at_random_ep_len), so time-outs do not arrive in one tick
    env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))
    obs = venv.get_observations()
    assert obs["actor"].shape[1] == rc.OBS, obs["actor"].shape
    teacher = tc.TorchPolicy(a.teacher_onnx).to(dev)
    assert teacher.width == rc.OBS + tc.PRIV_N, teacher.width
    student = rc.Student(a.core, teacher.mean[:, :rc.OBS].cpu(), teacher.div[:, :rc.OBS].cpu(), d=a.hidden, layers=a.layers).to(dev)
    spec = student.spec()
    print(f"[distill] student {spec}; segment {a.bptt_steps} ticks ({a.bptt_steps * 0.02:.2f} s), {a.num_envs} envs, "
          f"{a.epochs} epochs x minibatches of {a.mb_envs} envs", flush=True)
    opt = torch.optim.Adam(student.parameters(), lr=a.lr)
    robot, cm = env.scene["robot"], env.command_manager
    N, T = a.num_envs, a.bptt_steps

    def export(name, run_path):
        p = str(run_dir / name)
        rc.export_onnx(student, p)
        meta = dict(get_base_metadata(env, run_path=run_path))
        meta.update(state_inputs="state_in", state_outputs="state_out", recurrent_core=spec["core"],
                    state_dim=str(spec["state_dim"]), target_slots=str(spec["target_slots"]), bptt_steps=str(T))
        attach_metadata_to_onnx(p, meta)
        d = rc.check_onnx(student, p)
        assert d < 1e-3, f"{name}: ONNX differs from the torch student by {d}"
        return d

    d0 = export("model_0.onnx", "model_0")
    print(f"[distill] model_0.onnx: obs [1, 61] + state_in [1, {spec['state_dim']}]; ONNX vs torch over a sequence {d0:.1e}", flush=True)
    record = dict(run_name=a.run_name, task=fw.TASK, gpu=a.gpu, args=vars(a), student=spec, delta={k: v for k, v in delta.items()},
                  teacher_onnx=a.teacher_onnx, teacher_rebuild_max_abs=teacher.rebuild_max_abs,
                  started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))

    state = student.init_state(N, dev)
    base = Tally()                       # teacher-driven envs, whole run
    win_s, win_loss, win_err, win_n = Tally(), 0.0, 0.0, 0
    best, bad, stop, last_t, last_it, rows = None, 0, "max_iters reached", time.time(), 0, []
    half_noise = (torch.arange(N, device=dev) % 2 == 0)
    for it in range(1, a.max_iters + 1):
        beta = 1.0 if it <= a.warmup_iters else max(0.0, 1.0 - (it - a.warmup_iters) / max(a.beta_iters, 1))
        tmask = torch.rand(N, device=dev) < beta
        O = torch.empty(T, N, rc.OBS, device=dev)
        L = torch.empty(T, N, rc.ACT, device=dev)
        D = torch.empty(T, N, dtype=torch.bool, device=dev)
        state0 = state.clone()
        student.eval()
        with torch.no_grad():
            for t in range(T):
                o = obs["actor"].to(dev)
                lab = teacher(torch.cat([o, tc.priv_obs(env).to(dev)], dim=1))
                act_s, state = student.step(o, state)
                win_err += float(((act_s - lab) ** 2).mean(dim=1)[~tmask].sum())
                win_n += int((~tmask).sum())
                act_s = act_s + a.act_noise * torch.randn_like(act_s) * half_noise.view(N, 1)
                O[t], L[t] = o, lab
                obs, _, dones, extras = venv.step(torch.where(tmask.view(N, 1), lab, act_s).to(env.device))
                d = dones.to(dev).bool()
                to = extras.get("time_outs", torch.zeros_like(d)).to(dev).bool()
                D[t] = d
                state = state * (~d).view(N, 1)
                cmd, v = cm.get_command("twist"), robot.data.root_link_lin_vel_b
                along = (v[:, 0] * torch.sign(cmd[:, 0])).to(dev)
                moving = (cmd[:, 0].abs() > 0.15).to(dev)
                low = moving & (cmd[:, 0].abs() < 0.3).to(dev)
                base.add(tmask, d & ~to, along, moving, low)
                win_s.add(~tmask, d & ~to, along, moving, low)
        student.train()
        lr = a.lr + (a.lr_final - a.lr) * min(1.0, it / max(a.max_iters, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        keep = (~D).float()
        for _ in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for i in range(0, N, a.mb_envs):
                idx = perm[i:i + a.mb_envs]
                s, loss = state0[idx], 0.0
                for t in range(T):
                    act, s = student.step(O[t, idx], s)
                    loss = loss + ((act - L[t, idx]) ** 2).mean()
                    s = s * keep[t, idx].view(-1, 1)
                loss = loss / T
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step()
                win_loss = 0.9 * win_loss + 0.1 * float(loss) if win_loss else float(loss)

        capped = time.time() - t_start >= a.max_hours * 3600   # checked every iteration; the state is saved below
        if it % a.save_interval == 0 or it == a.max_iters or capped:
            now = time.time()
            elapsed = now - t_start
            its = (it - last_it) / (now - last_t)
            last_t, last_it = now, it
            s_out, b_out = win_s.out(), base.out()
            row = dict(iter=it, elapsed_s=round(elapsed, 1), it_per_s=round(its, 4), beta=round(beta, 3), lr=lr,
                       loss=round(win_loss, 5), student_action_mse=round(win_err / win_n, 5) if win_n else None,
                       student=s_out, teacher_base=b_out, time=time.strftime("%Y-%m-%dT%H:%M:%S"))
            rows.append(row)
            torch.save(dict(student=student.state_dict(), spec=spec, iter=it, args=vars(a)), run_dir / f"model_{it}.pt")
            dx = export(f"model_{it}.onnx", str(run_dir / f"model_{it}.pt"))
            (run_dir / "progress.json").write_text(json.dumps(dict(row, rows=rows), indent=1))
            with open(run_dir / "progress.csv", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[cap] it {it} elapsed {elapsed / 60:.1f} min it/s {its:.4f} beta {beta:.2f} loss {win_loss:.5f} "
                  f"student mse {row['student_action_mse']} student {s_out} teacher {b_out} onnx {dx:.1e}", flush=True)
            tf, ts, tl = b_out["falls_per_env_min"], b_out["speed"], b_out["low_speed"]
            sf, ss, sl = s_out["falls_per_env_min"], s_out["speed"], s_out["low_speed"]
            if beta == 0.0 and sf is not None and ss is not None and tf is not None and ts is not None:
                if ss >= 0.8 * ts and (best is None or sf < best["falls"]):
                    best = dict(iter=it, falls=sf, speed=ss)
                    torch.save(dict(student=student.state_dict(), spec=spec, iter=it, args=vars(a)), run_dir / "model_best.pt")
                not_stepping = sl is not None and sl < max(0.03, 0.5 * (tl or 0.0))
                falling = sf > 4.0 * tf + 1.0
                bad = bad + 1 if (it >= a.stop_after and (not_stepping or falling)) else 0
                if bad >= 2:
                    stop = (f"task-metric stop at iter {it}: student falls {sf} per env-min (teacher {tf}), low-command speed "
                            f"{sl} (teacher {tl}) on two saves")
            win_s, win_err, win_n = Tally(), 0.0, 0
            if elapsed >= a.max_hours * 3600:
                stop = f"time cap {a.max_hours} h reached at iter {it}"
            if stop != "max_iters reached":
                print("[cap] STOP:", stop, flush=True)
                break

    final_it = it
    if not (run_dir / f"model_{final_it}.pt").exists():
        torch.save(dict(student=student.state_dict(), spec=spec, iter=final_it, args=vars(a)), run_dir / f"model_{final_it}.pt")
    export(f"{a.run_name}.onnx", str(run_dir / f"model_{final_it}.pt"))
    if (run_dir / "model_best.pt").exists():
        student.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=dev, weights_only=False)["student"])
        export(f"{a.run_name}_best.onnx", str(run_dir / "model_best.pt"))
    elapsed = time.time() - t_start
    record.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), final_iter=final_it, stop_reason=stop, best=best,
                  elapsed_s=round(elapsed, 1), teacher_base=base.out(), onnx=str(run_dir / f"{a.run_name}.onnx"))
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, default=str))
    print(f"[done] {stop}; final iter {final_it}; best {best}; {elapsed / 60:.1f} min; onnx {record['onnx']}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
