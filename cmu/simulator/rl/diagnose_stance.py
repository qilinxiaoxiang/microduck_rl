"""Read-only geometry diagnostics from recorded poses, never acceptance evidence.

Reconstructing a pose does not reproduce dynamics, contact forces, or stability.
Only this offline diagnostic writes qpos; the live controller remains motor-only.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from native_household import NativeHousehold

parser=argparse.ArgumentParser()
parser.add_argument('trace',type=Path)
parser.add_argument('--after',type=float,default=0)
parser.add_argument('--stride',type=int,default=50)
args=parser.parse_args()
record=json.loads(args.trace.read_text())
r=NativeHousehold(record['modelXml'],record['config'])
try:
    for index,row in enumerate(record['rows']):
        if row['time']<args.after or index%args.stride:continue
        r.data.qpos[:7]=row['root'];r.data.qpos[r.qa]=row['joint']
        r.data.qpos[r.jaw]=row['jaw']
        if r.p is not None:r.data.qpos[r.p:r.p+7]=row['payload']
        mujoco.mj_forward(r.model,r.data)
        k=r.controller_context(True)['kinematics']
        feet=np.array([f['point'] for f in k['feet']]);com=np.array(k['com']['point'])
        print(json.dumps(dict(time=row['time'],phase=row['phase'],com=com.tolist(),feet=feet.tolist(),comFromFeet=(com-feet.mean(axis=0)).tolist(),root=row['root'],error=row.get('reachServo',{}).get('error') if row.get('reachServo') else None)))
finally:r.close()
