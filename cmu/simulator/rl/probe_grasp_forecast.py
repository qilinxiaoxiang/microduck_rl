"""Compare closure forecasts against recorded outcomes at a replayed boundary.

A motor replay reconstructs the boundary; report its discrepancy explicitly.
The replay and forecast are diagnosis, not new closed-loop acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from native_household import NativeHousehold
from grasp_forecast import forecast_grasp

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
d=json.loads(a.trace.read_text());r=NativeHousehold(d['modelXml'],d['config']);error=0.;position_error=0.;orientation_error=0.;result=None
try:
    for row in d['rows']:
        if row['phase']=='clamp':
            result=forecast_grasp(r,row['ctrl'][:14]);break
        current=r.step(row['ctrl']);error=max(error,float(np.max(np.abs(np.array(current['root'])-row['root']))))
        position_error=max(position_error,float(np.linalg.norm(np.array(current['root'][:3])-row['root'][:3])))
        qa=np.array(current['root'][3:]);qb=np.array(row['root'][3:])
        orientation_error=max(orientation_error,float(2*np.arccos(np.clip(abs(qa@qb)/(np.linalg.norm(qa)*np.linalg.norm(qb)),0,1))))
    result=dict(scope='recorded-motor boundary replay and isolated closure forecast; not a new policy',sourceSha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),
                maxRootReplayDifference=error,maxRootPositionDifferenceMeters=position_error,maxRootOrientationDifferenceRadians=orientation_error,
                legacyDifferenceNote='maxRootReplayDifference mixes position and quaternion components; do not interpret it as metres',time=r.data.time,forecast=result)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:r.close()
