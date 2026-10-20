"""Cross-language actor readback against PPO.predict, not the export itself."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from stable_baselines3 import PPO
from benchmark_native import ROOT

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--actor',type=Path,required=True);p.add_argument('--initial',type=Path);args=p.parse_args()
torch.set_num_threads(1);model=PPO.load(args.model,device='cpu');actor=json.loads(args.actor.read_text())
observations=np.random.default_rng(92652).uniform(-5,5,(256,93)).astype(np.float32)
reference=model.predict(observations,deterministic=True)[0]
result=subprocess.run(['node','app/tests/check-household-residual-export.mjs'],cwd=ROOT,input=json.dumps(dict(actor=actor,observations=observations.tolist())),text=True,capture_output=True,check=True)
error=float(np.max(np.abs(reference-np.array(json.loads(result.stdout)))))
assert error<1e-5,error
report=dict(samples=len(observations),maxAbsoluteActionError=error,optimizerUpdates=model._n_updates)
if args.initial:
    initial=PPO.load(args.initial,device='cpu');before=initial.policy.state_dict();after=model.policy.state_dict()
    actor_keys=[k for k in after if k.startswith(('mlp_extractor.policy_net.','action_net.'))]
    report['actorParameterDeltaL2']=sum(float(((after[k]-before[k])**2).sum()) for k in actor_keys)**.5
    assert report['actorParameterDeltaL2']>0,'Actor did not change'
print(json.dumps(report))
