"""CPU PPO. No imitation labels, analytical policy fitting, or paid services."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from functools import partial
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from jaw_env import MicroDuckJawEnv, ROOT


def export_actor(model, destination):
    modules = [*model.policy.mlp_extractor.policy_net, model.policy.action_net]
    layers = []
    for i, module in enumerate(modules):
        if isinstance(module, torch.nn.Linear):
            layers.append(dict(weight=module.weight.detach().cpu().tolist(),
                               bias=module.bias.detach().cpu().tolist(),
                               activation="tanh" if i+1 < len(modules) and isinstance(modules[i+1], torch.nn.Tanh) else "linear"))
    actor = dict(version=1, observationSize=20, actionMeaning=["close", "open"],
                 decisionHz=10, layers=layers)
    destination.write_text(json.dumps(actor, separators=(",", ":")) + "\n")


class EvidenceCallback(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.started = time.monotonic()
        self.episodes = 0

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.episodes += 1
                row = dict(step=self.num_timesteps, episodeNumber=self.episodes,
                           wallSeconds=time.monotonic()-self.started, **info)
                row.pop("terminal_observation", None)
                with (self.output / "episodes.jsonl").open("a") as f:
                    f.write(json.dumps(row, default=lambda a: a.tolist()) + "\n")
                print(json.dumps(dict(step=row["step"], episode=row["episodeNumber"],
                                      success=info["acceptance"]["accepted"],
                                      hold=info["acceptance"]["heldSeconds"],
                                      reward=info.get("episode", {}).get("r"),
                                      seconds=round(row["wallSeconds"]))), flush=True)
        return True

    def _on_rollout_end(self):
        self.model.save(self.output / f"checkpoint-{self.num_timesteps}")
        export_actor(self.model, self.output / f"actor-{self.num_timesteps}.json")


def make_env(trace_dir):
    return Monitor(MicroDuckJawEnv(trace_dir=trace_dir,timestep=.000125))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2048)
    p.add_argument("--seed", type=int, default=92610)
    p.add_argument("--envs", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    files = ["app/src/game/jaw-rl.js", "app/tests/jaw-rl-server.mjs", "rl/jaw_env.py", "rl/train_jaw.py",
             "app/src/game/head-reach-controller.js", "app/src/game/whole-body-core.js",
             "app/src/game/whole-body-acceptance.js", "app/src/game/physics-model.js",
             "app/src/game/table-pick-scene.js", "app/public/policies/BEST_alpha_walking.onnx"]
    manifest = dict(createdAt=datetime.now(timezone.utc).isoformat(), seed=args.seed,
                    steps=args.steps, envs=args.envs, algorithm="Stable-Baselines3 PPO",
                    device="cpu", trainingTimestep=.000125, evaluationTimestep=.0000625,
                    resume=str(args.resume) if args.resume else None,
                    resumeSha256=hashlib.sha256(args.resume.read_bytes()).hexdigest() if args.resume else None,
                    scope="jaw only; frozen balance and scripted reach/lift",
                    sources={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    env = SubprocVecEnv([partial(make_env, args.output/"traces") for _ in range(args.envs)], start_method="spawn")
    env.seed(args.seed)
    try:
        if args.resume:
            model = PPO.load(args.resume, env=env, device="cpu")
        else:
            model = PPO("MlpPolicy", env, learning_rate=.001, n_steps=128,
                        batch_size=64, n_epochs=10, gamma=.99, gae_lambda=.95,
                        ent_coef=.015, policy_kwargs=dict(net_arch=dict(pi=[32,32], vf=[32,32])),
                        seed=args.seed, device="cpu", verbose=0)
        model.save(args.output/"initial")
        export_actor(model, args.output/"initial-actor.json")
        started = time.monotonic()
        model.learn(total_timesteps=args.steps, callback=EvidenceCallback(args.output))
        model.save(args.output/"final")
        export_actor(model, args.output/"final-actor.json")
        (args.output/"completed.json").write_text(json.dumps(dict(steps=model.num_timesteps, wallSeconds=time.monotonic()-started))+"\n")
    finally:
        env.close()
