"""Fresh-process, deterministic evaluation. Never updates the policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import gzip
import numpy as np
import torch
from stable_baselines3 import PPO
from jaw_env import MicroDuckJawEnv


def cases(seed, count, all_random=False):
    rng = np.random.default_rng(seed)
    rows = [] if all_random else [dict(label="nominal")]
    for i in range(count-len(rows)):
        # Training interior; independent continuous values and RNG seed.
        rows.append(dict(label=f"unseen-{i+1}", x=float(rng.uniform(-.483,-.477)),
                         y=float(rng.uniform(-.001,.001)), mass=float(rng.uniform(.0015,.004)),
                         friction=float(rng.uniform(.65,.9))))
    return rows


def trial(job):
    model_path, method, config, output = job
    torch.set_num_threads(1)
    env = MicroDuckJawEnv(randomize=False)
    try:
        policy = PPO.load(model_path, device="cpu") if method=="learned" else None
        obs, _ = env.reset(options=dict(config=config))
        rng = np.random.default_rng(1926)
        actions = []
        while True:
            if policy:
                action, _ = policy.predict(obs, deterministic=True)
                action = int(action)
            elif method=="open": action=1
            elif method=="closed": action=0
            elif method=="random": action=int(rng.integers(2))
            else: action=0
            response=env.rpc(op="step", action=action, scripted=method=="scripted")
            if method!="scripted": actions.append(action)
            obs=np.asarray(response["observation"],np.float32)
            if response["terminated"]: break
        evidence = env.rpc(op="evidence")
        with gzip.open(output,"wt") as f: json.dump(evidence,f,separators=(",",":"))
        return dict(config=config, method=method, acceptance=evidence["acceptance"],
                    summary=evidence["summary"], modelAudit=evidence["modelAudit"], reward=evidence["reward"],
                    wallSeconds=evidence["wallSeconds"], trace=str(output),
                    actionCounts={str(i):actions.count(i) for i in range(2)})
    finally:
        env.close()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path)
    p.add_argument("--method",choices=["learned","scripted","open","closed","random"],default="learned")
    p.add_argument("--seed",type=int,default=92620)
    p.add_argument("--count",type=int,default=4)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--cases",type=Path)
    p.add_argument("--all-random",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    configs=json.loads(args.cases.read_text()) if args.cases else cases(args.seed,args.count,args.all_random)
    (args.output/"cases.json").write_text(json.dumps(configs,indent=2)+"\n")
    jobs=[(args.model,args.method,c,args.output/f"trial-{i}.json.gz") for i,c in enumerate(configs)]
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(trial,jobs):
            results.append(row)
            print(json.dumps(row),flush=True)
    summary=dict(generatedAt=datetime.now(timezone.utc).isoformat(), method=args.method,
                 model=str(args.model) if args.model else None,
                 modelSha256=hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model else None,
                 seed=args.seed,successes=sum(r["acceptance"]["accepted"] for r in results),
                 trials=len(results),results=results)
    (args.output/"evaluation.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k!="results"}),flush=True)
