"""At most two local prediction jobs; each owns four forecast threads.

Each child has a hard process-group deadline and a distinct output directory.
This collects model-based teacher data, not reinforcement-learning results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from run_native_household import ROOT

def run(index,cases,out,seconds):
    directory=out/f'case-{index:03d}'
    command=[sys.executable,'rl/evaluate_predictive_reach.py','--cases',str(cases),'--index',str(index),'--axis-grasp','--output',str(directory),'--seconds',str(seconds)]
    with (out/f'case-{index:03d}.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:status=p.wait(timeout=seconds+90)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            status='deadline'
    trial=directory/'trial-000.json'
    if status!=0 or not trial.exists():return dict(index=index,incomplete=True,status=status)
    record=json.loads(trial.read_text())
    return dict(index=index,acceptance=record['acceptance'],summary=record['summary'],updates=record['plannerUpdates'],seconds=record['wallSeconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--indices',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=480);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    cases=a.cases.resolve();indices=[int(i) for i in a.indices.split(',')];count=len(json.loads(cases.read_text()))
    if not 1<=len(indices)<=16 or len(set(indices))!=len(indices) or not all(0<=i<count for i in indices) or not 1<=a.seconds<=600 or not 1<=a.workers<=2:raise ValueError('Use 1-16 unique explicit cases and 1-2 workers, each bounded to 600 seconds')
    a.output.mkdir(parents=True,exist_ok=False);records=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,i,cases,a.output,a.seconds) for i in indices]
        for future in as_completed(futures):
            record=future.result();records.append(record);print(json.dumps(record),flush=True)
            (a.output/'summary.json').write_text(json.dumps(sorted(records,key=lambda r:r['index']),indent=2))
