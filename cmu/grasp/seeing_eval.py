"""Evaluation battery for trained seeing-pick policies (cmu/grasp/policies/*.onnx), one 4 s episode per env.

Cases, each run on every policy:
  w1        object within +-1 cm of the landing spot (where the curriculum starts)
  w3        +-3 cm (the window the runs reached)
  w5        +-5 cm (the operating window the plan asks for)
  w3_kick   +-3 cm, every episode kicks the object 1-3 cm before the bite
  w3_dist   +-3 cm, a 7 mm distractor 4-10 cm away in every episode
  w3_blind  +-3 cm with the body slots (camera input) zeroed: if success holds, the policy is not using the camera
The warm start (alpha_ground_pick, body slots zeroed) runs w1 and w3 as the baseline.

Kaggle (GPU): python3 cmu/finetune/kaggle/kaggle_job.py push bench --worker seeing_eval --run-name <name>
Local:        python cmu/grasp/seeing_eval.py --root /tmp/se --run-name t --num-envs 16 --device cpu
Output: <root>/<run-name>/seeing-eval.json (copied into cmu/grasp/ after the fetch).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeing_audit as A  # noqa: E402
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402
from mjlab_microduck.tasks import cmu_seeing_pick_env_cfg as sp  # noqa: E402

POLICIES = Path(__file__).resolve().parent / "policies"
if not any(POLICIES.glob("*.onnx")):    # the Kaggle source pack drops .onnx; kernel_entry.py puts them in the assets
    POLICIES = Path("/root/assets/policies")
CASES = {"w1": (sp.WINDOW_START, 0.0, 0.0), "w3": (0.03, 0.0, 0.0), "w5": (sp.WINDOW_OPERATING, 0.0, 0.0),
         "w3_kick": (0.03, 1.0, 0.0), "w3_dist": (0.03, 0.0, 1.0), "w3_blind": (0.03, 0.0, 0.0)}


def make_env(num_envs, seed, window, kick, distractor, device):
    cfg = load_env_cfg(A.TASK, play=False)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.curriculum.pop("seeing_window", None)
    cfg.events["reset_object"].params.update(window_xy=window, kick_prob=kick, distractor_prob=distractor)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    env.common_step_counter = A.CURRICULA_DONE_STEP
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-hours", type=float, default=0.5)
    p.add_argument("--gpu", default="")
    p.add_argument("--rate-per-hour", default="0")
    p.add_argument("--pattern", default="*.onnx")
    p.add_argument("--cases", default=",".join(CASES))
    a = p.parse_args()
    device = a.device if (a.device != "cuda" or torch.cuda.is_available()) else sys.exit("no CUDA device")
    out_dir = Path(a.root) / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = load_rl_cfg(A.TASK).clip_actions
    cases = a.cases.split(",")
    runs = [(q.stem, A.OnnxMlp(q, device), cases) for q in sorted(POLICIES.glob(a.pattern))]
    runs.append(("warm_start", A.OnnxMlp(A.BOW, device), [c for c in ("w1", "w3") if c in cases]))
    res, t0 = {}, time.time()
    for name, pol, cases in runs:
        res[name] = {}
        for case in cases:
            window, kick, dist = CASES[case]
            torch.manual_seed(a.seed)
            env = make_env(a.num_envs, a.seed, window, kick, dist, device)
            zero = (A.BODY,) if (case.endswith("blind") or name == "warm_start") else ()
            r = A.rollout(env, pol, zero, a.steps, clip)
            env.close()
            r.pop("terms_per_s")
            res[name][case] = r
            print(f"[seeing-eval] {name} {case}: success {r['success']} fell {r['fell']} "
                  f"err {r.get('xy_err_at_close_m')} ({time.time() - t0:.0f} s)", flush=True)
            if time.time() - t0 > a.max_hours * 3600 * 0.9:
                break
    result = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), task=A.TASK, device=str(device), seed=a.seed,
                  envs=a.num_envs, steps=a.steps, wall_s=round(time.time() - t0),
                  policies={n: A.md5(POLICIES / f"{n}.onnx") for n, _, _ in runs if n != "warm_start"},
                  cases={k: dict(window=v[0], kick=v[1], distractor=v[2]) for k, v in CASES.items()}, results=res)
    (out_dir / "seeing-eval.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
