"""Counterfactual motor replay: stop rising once the whole item clears ground.

This is a physical diagnostic, not closed-loop policy acceptance. The live
simulation always begins at ordinary reset and receives only motor targets.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from native_household import NativeHousehold

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--clearance',type=float,default=.04);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
if not .03<=a.clearance<=.08:raise ValueError('Expected a bounded lift clearance')
e=json.loads(a.trace.read_text());r=NativeHousehold(e['modelXml'],e['config']);rows=[];position_error=0.;orientation_error=0.;stop=None
try:
    for row in e['rows']:
        s=r.step(row['ctrl']);s['phase']=row['phase'];rows.append(s)
        position_error=max(position_error,float(np.linalg.norm(np.array(s['root'][:3])-row['root'][:3])))
        q=np.array(s['root'][3:]);v=np.array(row['root'][3:]);orientation_error=max(orientation_error,float(2*np.arccos(np.clip(abs(q@v)/(np.linalg.norm(q)*np.linalg.norm(v)),0,1))))
        if s['phase']=='lift' and s['shape']['minZ']>a.clearance and min(s['contact']['lowerForce'],s['contact']['upperForce'])>.002:
            stop=s['time'];break
    if stop is not None:
        ctrl=r.data.ctrl.copy()
        for tick in range(250):
            s=r.step(ctrl);s['phase']='hold';rows.append(s)
        release_height=s['payload'][2]
        for tick in range(150):
            ctrl[14]=min(.32,ctrl[14]+.012);s=r.step(ctrl);s['phase']='release';rows.append(s)
    out=dict(scope='Counterfactual recorded-motor replay and constant-motor hold; not a policy success',sourceSha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),stop=stop,
             maxRootPositionDifferenceMeters=position_error,maxRootOrientationDifferenceRadians=orientation_error,summary=r.summary,rows=rows,
             holdTail=[dict(time=s['time'],minZ=s['shape']['minZ'],pair=[s['contact']['lowerForce'],s['contact']['upperForce']],nonFoot=s['contact']['nonFootSupport']) for s in rows if s['phase']=='hold'][-5:],
             releaseFall=release_height-rows[-1]['payload'][2] if stop is not None else None)
    a.output.write_text(json.dumps(out));print(json.dumps({k:v for k,v in out.items() if k!='rows'}))
finally:r.close()
