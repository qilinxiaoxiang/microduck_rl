"""Name forbidden contacts during an exact recorded-motor replay.

Read-only with respect to the source trace. This is diagnosis, not a new
closed-loop acceptance run. All body placement happens at ordinary reset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from native_household import NativeHousehold

p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
d=json.loads(a.trace.read_text());r=NativeHousehold(d['modelXml'],d['config']);found=[];max_difference=0
try:
    for row in d['rows']:
        result=r.step(row['ctrl']);max_difference=max(max_difference,float(np.max(np.abs(np.array(result['root'])-row['root']))))
        if not result['contact']['nonFootSupport']:continue
        for i in range(r.data.ncon):
            c=r.data.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            if min(g1,g2)<0:continue
            b1,b2=r.model.geom_bodyid[[g1,g2]]
            static1=r.model.body_weldid[b1]==0;static2=r.model.body_weldid[b2]==0
            robot=g2 if static1 else g1 if static2 else -1
            if robot<0 or robot in r.feet:continue
            body=int(r.model.geom_bodyid[robot])
            while body>0 and body!=r.trunk:body=int(r.model.body_parentid[body])
            if body!=r.trunk:continue
            force=np.zeros(6);mujoco.mj_contactForce(r.model,r.data,i,force)
            if np.linalg.norm(force[:3])<=1e-5:continue
            found.append(dict(time=r.data.time,phase=row['phase'],geoms=[mujoco.mj_id2name(r.model,mujoco.mjtObj.mjOBJ_GEOM,g) for g in [g1,g2]],force=float(np.linalg.norm(force[:3]))))
    result=dict(scope='recorded motor replay diagnosis, not a new policy',sourceSha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),maxRootReplayDifference=max_difference,contacts=found)
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:r.close()
