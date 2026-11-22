"""Measured adaptation time of a recurrent student (TEACHER-STUDENT-PLAN.md, section "Adaptation time"). Mac mini CPU.

  FLIGHT_MANIFESTS=<dir> python cmu/finetune/flight/adapt_eval.py --student-pt model_best.pt --teacher-onnx t.onnx \
      --load full --envs 64 --out adapt_full.json

The universal3 env pinned to one load (full | nobatt | none), random pushes off, the student driving without noise.
Per tick and env: the student's action error against the teacher's label, sqrt(mean over the 14 joints of the squared
difference), rad. Three events, 250 ticks (5 s) each:
  reset   every env reset, student state zero (tick 0)
  push    0.26 m/s forward velocity step on every env (the 0.5 N s acceptance shove on a 1.9 kg robot)
  reentry the student's state zeroed while it moves (the policy re-entered, the load unknown again)
For each event: the median error over the envs still upright at ticks 0, 5, 10, 25, 50, 100, the steady value (median
of the last 50 ticks) and the settle time: the first tick from which the median error stays at or below
1.5 x steady + 0.005 rad for 10 ticks in a row. Envs that fall are dropped from the later ticks and counted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flight_worker as fl  # noqa: E402
import recurrent as rc  # noqa: E402
import teacher as tc  # noqa: E402
from distill_worker import build_env  # noqa: E402

PIN = {"full": dict(full=1.0, nobatt=0.0, none=0.0, scaled=0.0), "nobatt": dict(full=0.0, nobatt=1.0, none=0.0, scaled=0.0),
       "none": dict(full=0.0, nobatt=0.0, none=1.0, scaled=0.0)}
PHASE = 250
MARKS = (0, 5, 10, 25, 50, 100)


def settle(e):
    steady = float(np.nanmedian(e[-50:]))
    thr = 1.5 * steady + 0.005
    ok = e <= thr
    for t in range(len(e) - 10):
        if ok[t:t + 10].all():
            return t, steady, thr
    return None, steady, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-pt", required=True)
    ap.add_argument("--teacher-onnx", required=True)
    ap.add_argument("--load", choices=sorted(PIN), required=True)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--delta", default="universal3", help="the env the student was distilled in (universal3 | universal4)")
    a = ap.parse_args()
    fw = fl.fw
    md = os.environ["FLIGHT_MANIFESTS"]
    push = {}

    def mutate(cfg):
        push["func"] = cfg.events["push_robot"].func
        push["asset_cfg"] = cfg.events["push_robot"].params["asset_cfg"]
        cfg.events["push_robot"].interval_range_s = (1e6, 1e6 + 1.0)

    env, venv, _ = build_env(fw, a.delta, str(Path(md) / "flightpack-none.json"),
                             str(Path(__file__).resolve().parents[2] / "whole/electrical/existing-model-mass-audit.json"),
                             0.09, a.envs, a.seed, "cpu", pin=PIN[a.load], mutate=mutate)
    student = rc.load_checkpoint(a.student_pt)
    teacher = tc.TorchPolicy(a.teacher_onnx)
    N = a.envs
    err = np.full((3 * PHASE, N), np.nan)
    alive = torch.ones(N, dtype=torch.bool)
    falls = [0, 0, 0]
    with torch.no_grad():
        env.reset()
        obs = venv.get_observations()
        state = student.init_state(N)
        for t in range(3 * PHASE):
            ph = t // PHASE
            if t == PHASE:
                push["func"](env, torch.arange(N), velocity_range={"x": (0.26, 0.26)}, asset_cfg=push["asset_cfg"])
            if t == 2 * PHASE:
                state = student.init_state(N)
            o = obs["actor"]
            lab = teacher(torch.cat([o, tc.priv_obs(env)], dim=1))
            act, state = student.step(o, state)
            e = ((act - lab) ** 2).mean(dim=1).sqrt()
            err[t, alive.numpy()] = e[alive].numpy()
            obs, _, dones, _ = venv.step(act)
            d = dones.bool() & alive
            falls[ph] += int(d.sum())
            alive &= ~dones.bool()
    res = dict(student=a.student_pt, teacher=a.teacher_onnx, load=a.load, envs=N, spec=student.spec(), phase_ticks=PHASE,
               upright_at_end=int(alive.sum()), events={})
    for i, name in enumerate(("reset", "push", "reentry")):
        seg = err[i * PHASE:(i + 1) * PHASE]
        med = np.array([np.nanmedian(r) if np.isfinite(r).any() else np.nan for r in seg])
        t_s, steady, thr = settle(med)
        res["events"][name] = dict(settle_ticks=t_s, settle_s=None if t_s is None else round(t_s * 0.02, 2),
                                   steady_rad=round(steady, 4), threshold_rad=round(thr, 4), falls=falls[i],
                                   median_error_rad={str(m): round(float(med[m]), 4) for m in MARKS},
                                   peak_rad=round(float(np.nanmax(med[:100])), 4))
        print(f"[adapt] {a.load} {name}: settle {t_s} ticks, steady {steady:.4f} rad, error at {MARKS}: "
              f"{[round(float(med[m]), 4) for m in MARKS]}, falls {falls[i]}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
