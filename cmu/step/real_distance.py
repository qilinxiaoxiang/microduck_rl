"""How far each policy's actions are from walk61's on 645 logged states of walk61 walking on the robot.

    uv run --no-sync python cmu/step/real_distance.py --policy step=<onnx> [...]

For scale: t2b 0.22 (walked like walk61 on the robot), ft3 0.43 (lost balance 3 of 3), T1 0.71, nudge 1.05 (fell).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

STARTER = Path(__file__).resolve().parents[1] / "planning" / "fetch-20261103" / "starter" / "motion"
sys.path.insert(0, str(STARTER))
import replay  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", action="append", required=True, metavar="NAME=ONNX")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = replay.Policy("walk61")
    x, _, _ = replay.reconstructed(replay.load("walk61-real.jsonl"), base, {"walk"})
    ref, res = base.run(x, "velocity"), dict(states=len(x), policies={})
    for spec in a.policy:
        name, path = spec.split("=", 1)
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = np.stack([s.run(None, {s.get_inputs()[0].name: r[None]})[0][0] for r in x])
        d = got - ref
        res["policies"][name] = dict(mean_l2=round(float(np.linalg.norm(d, axis=1).mean()), 4),
                                     mean_l1=round(float(np.abs(d).sum(1).mean()), 4),
                                     max_per_joint=round(float(np.abs(d).max()), 4))
        print(name, res["policies"][name])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
