"""Verify the browser actor against actual PPO inference, not its own copy."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from jaw_env import ROOT

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--actor",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    model=PPO.load(args.model,device="cpu")
    obs=np.random.default_rng(92699).uniform(-5,5,(1000,20)).astype(np.float32)
    with torch.no_grad():
        tensor=torch.as_tensor(obs)
        features=model.policy.extract_features(tensor)
        latent=model.policy.mlp_extractor.forward_actor(features)
        logits=model.policy.action_net(latent).numpy()
    predicted,_=model.predict(obs,deterministic=True)
    js=subprocess.run(["node","tests/jaw-actor-eval.mjs"],cwd=ROOT/"app",text=True,
                      input=json.dumps(dict(actor=json.loads(args.actor.read_text()),observations=obs.tolist())),
                      capture_output=True,check=True)
    result=json.loads(js.stdout)
    delta=float(np.max(np.abs(logits-np.asarray([r["logits"] for r in result]))))
    mismatches=int(np.sum(predicted!=np.asarray([r["action"] for r in result])))
    evidence=dict(observations=len(obs),maxLogitDifference=delta,actionMismatches=mismatches,
                  checkpointSha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  actorSha256=hashlib.sha256(args.actor.read_bytes()).hexdigest(),
                  accepted=mismatches==0 and delta<2e-5)
    args.output.write_text(json.dumps(evidence,indent=2)+"\n")
    print(evidence)
    assert evidence["accepted"],evidence
