"""Cache physically executed walk/crouch prefixes for training resets only."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from native_household import NativeHousehold
from run_native_household import ControllerBridge,source_hashes,ROOT

def prepare(config,destination):
    bridge=ControllerBridge();r=None
    try:
        contract=bridge.request(op='residual-contract')
        init=bridge.request(op='init',config=config);r=NativeHousehold(init['xml'],init['config']);rows=[]
        for tick in range(3000):
            result=bridge.request(op='step',snapshot=r.snapshot(),meta=r.metadata(),context=r.controller_context(True),summary=r.summary)
            row=r.step(result['ctrl']);row['phase']=result['phase'];rows.append(row)
            if result['phase']=='failed':raise RuntimeError('Failed before feedback prefix')
            if result['phase']=='reach' and (result.get('innerTime') or 0)>=contract['prefixTime']:break
        else:raise RuntimeError('No pre-feedback state reached')
        s=r.summary
        if s['nonFootSupport'] or not s['finite'] or s['solverReset'] or s['maxExternal'] or max(s['maxPenetration'],s['maxRobotPenetration'])>=.0005:raise RuntimeError('Unsafe prefix rejected')
        state=np.zeros(mujoco.mj_stateSize(r.model,mujoco.mjtState.mjSTATE_INTEGRATION))
        mujoco.mj_getState(r.model,r.data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
        record=dict(contract=contract,config=init['config'],modelXml=init['xml'],physicsState=state.tolist(),lastAction=r.last_action.tolist(),summary=s,contact=r.contact,rows=rows,
                    controllerState=bridge.request(op='capture-prefix'),provenance=dict(mode='physical prefix for training reset, not a full trial',sources=source_hashes()))
        Path(destination).write_text(json.dumps(record))
        return dict(path=str(destination),time=r.data.time,config=config)
    finally:
        if r:r.close()
        bridge.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cases=json.loads(args.cases.read_text())
    if not isinstance(cases,list) or not 1<=len(cases)<=32:raise ValueError('Use 1–32 explicit training starts')
    args.output.mkdir(parents=True,exist_ok=False)
    hashes=source_hashes()
    for name in hashes:
        destination=args.output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,destination)
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending={pool.submit(prepare,c,args.output/f'prefix-{i:03d}.json'):i for i,c in enumerate(cases)}
        for future in as_completed(pending):print(json.dumps(future.result()),flush=True)
    if hashes!=source_hashes():raise RuntimeError('Runtime sources changed during prefix generation')
    (args.output/'manifest.json').write_text(json.dumps(dict(cases=cases,sources=hashes,scope='training only; full reset and unseen starts required for acceptance'),indent=2))
