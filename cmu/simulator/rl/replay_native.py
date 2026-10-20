"""Replay an archived motor sequence, retaining full native physical evidence."""
import argparse
import json
from pathlib import Path
import time
import mujoco
from native_household import NativeHousehold

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    original=json.loads(args.trace.read_text())
    runtime=NativeHousehold(original['modelXml'],original['config'])
    started=time.perf_counter();rows=[]
    try:
        for source in original['rows']:
            row=runtime.step(source['ctrl']);row['phase']=source['phase'];rows.append(row)
        result=dict(config=original['config'],phase=original['phase'],rows=rows,summary=runtime.summary,
                    modelAudit=runtime.model_audit,modelXml=runtime.xml,
                    provenance=dict(engine=mujoco.mj_versionString(),mode='recorded motor replay, not a new policy',sourceTrace=str(args.trace)),
                    wallSeconds=time.perf_counter()-started)
        args.output.write_text(json.dumps(result))
        print(json.dumps({key:value for key,value in result.items() if key not in ['rows','modelXml']}))
    finally:runtime.close()
