"""Independent full-reset evaluation of a residual PPO policy.

No cached prefix is restored here. Walking, floor support, pickup, holding
and release are all physically executed and sent to the unchanged verifier.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import PPO
from native_household import NativeHousehold
from run_native_household import ControllerBridge,source_hashes,ROOT

def evaluate(config,model_path,out):
    if out.exists():raise FileExistsError(out)
    model=PPO.load(model_path,device='cpu');bridge=ControllerBridge();r=None;hashes=source_hashes();started=time.monotonic()
    try:
        contract=bridge.request(op='residual-contract')
        if getattr(model,'household_contract',None)!=contract:raise ValueError('Model observation/action contract mismatch')
        init=bridge.request(op='init',config=config);r=NativeHousehold(init['xml'],init['config']);rows=[];phase='manual';action=np.zeros(14);decisions=0;inner_time=0
        for tick in range(config.get('ticks',3000)):
            if inner_time>=contract['prefixTime']:
                if decisions%5==0:
                    obs=np.array(bridge.request(op='observe',snapshot=r.snapshot(),context=r.controller_context(False)),np.float32)
                    action=model.predict(obs,deterministic=True)[0]
                decisions+=1
            result=bridge.request(op='step',snapshot=r.snapshot(),meta=r.metadata(),context=r.controller_context(True),summary=r.summary,residualAction=action.tolist())
            row=r.step(result['ctrl']);row['phase']=phase=result['phase'];row['reachServo']=result.get('reachServo');row['alignment']=result.get('alignment');row['residualAction']=action.tolist();rows.append(row)
            inner_time=result.get('innerTime') or 0
            if phase in ['done','failed'] or not r.summary['finite'] or r.summary['solverReset']:break
        events=bridge.request(op='events')
        evidence=dict(config=init['config'],phase=phase,rows=rows,summary=r.summary,modelAudit=r.model_audit,modelXml=r.xml,trainingPrefixReset=False,
                      wallSeconds=time.monotonic()-started,provenance=dict(engine='MuJoCo 3.11.0 native',residualInference='PPO.predict deterministic CPU',
                      policySha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),sources=hashes,sourcesUnchanged=hashes==source_hashes()),**events)
        evidence['acceptance']=bridge.request(op='evaluate',evidence={key:evidence[key] for key in ['phase','rows','summary','modelAudit']})
        if phase not in ['done','failed']:
            evidence['incomplete']=True;evidence['failure']='Full-reset evaluation exhausted its control-step budget'
            evidence['acceptance']['accepted']=False;evidence['acceptance']['failures'].append(evidence['failure'])
        if not evidence['provenance']['sourcesUnchanged']:
            evidence['acceptance']['accepted']=False;evidence['acceptance']['failures'].append('Sources changed during evaluation')
        out.write_text(json.dumps(evidence))
        return dict(file=out.name,config=config,phase=phase,acceptance=evidence['acceptance'],summary=r.summary,wallSeconds=evidence['wallSeconds'])
    finally:
        if r:r.close()
        bridge.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    torch.set_num_threads(1);cases=json.loads(args.cases.read_text())
    if not 1<=len(cases)<=64 or not 1<=args.workers<=4:raise ValueError('Use a bounded evaluation')
    args.output.mkdir(parents=True,exist_ok=False)
    for name in [*source_hashes(),'rl/evaluate_household_residual.py']:
        target=args.output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    shutil.copyfile(args.model,args.output/'policy.zip');shutil.copyfile(args.cases,args.output/'cases.json')
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluate,config,args.model,args.output/f'trial-{i:03d}.json') for i,config in enumerate(cases)]
        for future in as_completed(futures):
            result=future.result();results.append(result);print(json.dumps(result),flush=True)
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))
