"""Read saved optimizer/actor state to verify a real learned parameter update."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--initial",type=Path,required=True)
    p.add_argument("--final",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    initial=PPO.load(args.initial,device="cpu")
    final=PPO.load(args.final,device="cpu")
    names=[n for n,_ in initial.policy.named_parameters() if "policy_net" in n or "action_net" in n]
    a=dict(initial.policy.named_parameters());b=dict(final.policy.named_parameters())
    delta=torch.cat([(b[n]-a[n]).detach().flatten() for n in names])
    result=dict(actorParameters=int(delta.numel()),changedParameters=int((delta!=0).sum()),
                actorDeltaL2=float(torch.linalg.norm(delta)),ppoOptimizerEpochs=final._n_updates,
                initialSha256=hashlib.sha256(args.initial.read_bytes()).hexdigest(),
                finalSha256=hashlib.sha256(args.final.read_bytes()).hexdigest())
    result["accepted"]=result["actorDeltaL2"]>1e-5 and final._n_updates>0
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(result)
    assert result["accepted"]
