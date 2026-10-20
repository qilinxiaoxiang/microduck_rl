"""Summarize actual completed training and preserve its failure accounting."""
from pathlib import Path
import json
import hashlib
import importlib.metadata
import subprocess
import sys
from jaw_env import ROOT

if __name__=="__main__":
    output=ROOT/"rl/results"
    output.mkdir(exist_ok=True)
    runs=[ROOT/"rl/runs/run-92610",ROOT/"rl/runs/run-92610-cont2"]
    episodes=[];completed=[]
    for run in runs:
        completed.append(json.loads((run/"completed.json").read_text()))
        episodes.extend(json.loads(line) for line in (run/"episodes.jsonl").read_text().splitlines())
        manifest=json.loads((run/"manifest.json").read_text())
        for name,digest in manifest["sources"].items():
            path=ROOT/name if name.endswith(".onnx") else run/"sources"/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,(run,name)
    identity=lambda c:tuple(c.get(k) for k in ("x","y","mass","friction"))
    training_conditions={identity(e["config"]) for e in episodes}
    heldout=json.loads((ROOT/"rl/runs/heldout-initial/cases.json").read_text())
    overlap=training_conditions&{identity(c) for c in heldout}
    assert not overlap
    report=dict(trainingDecisions=sum(r["steps"] for r in completed),
                trainingWallSeconds=sum(r["wallSeconds"] for r in completed),
                completedEpisodes=len(episodes),
                stochasticTrainingSuccesses=sum(e["acceptance"]["accepted"] for e in episodes),
                rejectedPenetrationEpisodes=sum(e["summary"]["maxPenetration"]>=.0005 for e in episodes),
                maximumTrainingPenetrationMm=max(e["summary"]["maxPenetration"] for e in episodes)*1000,
                uniqueTrainingConditions=len(training_conditions),heldoutConditionOverlap=len(overlap),
                trainingTimestep=.000125,evaluationTimestep=.0000625,
                seeds=[92610],resumeRestartsEnvironmentStream=True,
                python=sys.version,node=subprocess.check_output(["node","--version"],text=True).strip(),
                packages={p:importlib.metadata.version(p) for p in ["stable-baselines3","gymnasium","torch","numpy"]},
                originalTrainingSourceSnapshotsVerified=True)
    (output/"training-summary.json").write_text(json.dumps(report,indent=2)+"\n")
    files=[*list((ROOT/"rl").glob("*.py")),ROOT/"rl/requirements.txt",ROOT/"app/package-lock.json",ROOT/"app/package.json",
           ROOT/"app/tests/jaw-rl-server.mjs",ROOT/"app/tests/jaw-actor-eval.mjs",ROOT/"Play-Jaw-RL.command",
           ROOT/"app/src/table-pick.js",ROOT/"app/table-pick.html"]
    for name in ["jaw-rl","head-reach-controller","table-pick-controller","table-pick-scene",
                 "whole-body-core","whole-body-acceptance","contact-observer","physics-model","beak-colliders","constants"]:
        files.append(ROOT/f"app/src/game/{name}.js")
    (output/"execution-source-hashes.json").write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+"\n")
    print(json.dumps(report,indent=2))
