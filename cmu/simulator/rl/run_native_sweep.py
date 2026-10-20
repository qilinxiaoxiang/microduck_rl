"""Small bounded local sweeps; each process owns its physics and audit state."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
from run_native_household import run, source_hashes, ROOT


def sweep(cases, output, workers=4):
    configs=json.loads(Path(cases).read_text())
    if not isinstance(configs,list) or not 1<=len(configs)<=128:raise ValueError('Use 1–128 explicit cases')
    if not 1<=workers<=4:raise ValueError('Use 1–4 workers')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    source=out/'source';source.mkdir()
    for name in source_hashes():
        destination=source/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    shutil.copyfile(cases,source/'cases.json')
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending={executor.submit(run,c,out/f'trial-{i:03d}.json',True):i for i,c in enumerate(configs)}
        for future in as_completed(pending):
            index=pending[future]
            try:
                record=future.result();result=dict(index=index,config=record['config'],phase=record['phase'],acceptance=record['acceptance'],summary=record['summary'],wallSeconds=record['wallSeconds'])
            except Exception as error:
                result=dict(index=index,config=configs[index],incomplete=True,error=repr(error))
                # Never overwrite a completed trace with a cleanup failure.
                (out/f'error-{index:03d}.json').write_text(json.dumps(result))
            results.append(result);print(json.dumps(result),flush=True)
    (out/'summary.json').write_text(json.dumps(sorted(results,key=lambda r:r['index']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cases',required=True);parser.add_argument('--output',required=True);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();sweep(args.cases,args.output,args.workers)
