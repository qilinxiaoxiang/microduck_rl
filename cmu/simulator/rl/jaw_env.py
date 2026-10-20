"""P0: learn jaw actuation while retaining audited balance and reach skills."""
from pathlib import Path
import json
import subprocess
import gzip
import os
import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class MicroDuckJawEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, randomize=True, fixed_config=None, trace_dir=None, timestep=.0000625):
        super().__init__()
        self.observation_space = gym.spaces.Box(-5, 5, (20,), np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.randomize = randomize
        self.fixed_config = fixed_config
        self.timestep = timestep
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.episode_number = 0
        self.process = subprocess.Popen(
            ["node", "tests/jaw-rl-server.mjs"], cwd=ROOT / "app",
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )

    def rpc(self, **request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Physics worker exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_number += 1
        if options and "config" in options:
            config = options["config"]
        elif self.fixed_config is not None:
            config = self.fixed_config
        elif self.randomize:
            u = self.np_random.uniform
            config = dict(x=float(u(-.483, -.477)), y=float(u(-.001, .001)),
                          mass=float(u(.0015, .004)), friction=float(u(.65, .9)))
        else:
            config = {}
        result = self.rpc(op="reset", config={"dt":self.timestep, **config})
        return np.asarray(result["observation"], np.float32), result["info"]

    def step(self, action):
        result = self.rpc(op="step", action=int(action))
        if result["terminated"] and self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"worker-{os.getpid()}-episode-{self.episode_number}.json.gz"
            with gzip.open(path, "wt") as f:
                json.dump(self.rpc(op="evidence"), f, separators=(",", ":"))
            result["info"]["trace"] = str(path)
        return (np.asarray(result["observation"], np.float32), result["reward"],
                result["terminated"], result["truncated"], result["info"])

    def close(self):
        if self.process.poll() is None:
            try:
                self.rpc(op="close")
                self.process.wait(timeout=10)
            except (BrokenPipeError, RuntimeError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=10)
        self.process.stdin.close()
        self.process.stdout.close()
