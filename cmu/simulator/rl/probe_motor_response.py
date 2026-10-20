"""Offline dynamics-response diagnostic, never full-trial acceptance evidence.

Clone a physically executed training prefix, then measure independent motor
perturbations in separate prediction data. The source state's integration
bytes must remain unchanged by every prediction. No rollout is substituted
for a live trajectory or successful trial.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import mujoco
import numpy as np
from native_household import NativeHousehold
from native_contact_step import ContactStepper
from benchmark_native import JOINTS
from run_native_household import ControllerBridge

def state(m,d):
    a=np.zeros(mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(m,d,a,mujoco.mjtState.mjSTATE_INTEGRATION)
    return a

def point(r,d):
    R=d.xmat[r.head].reshape(3,3)
    return d.xpos[r.head]+R@np.array([-.017,-.05,-.072])

def probe(prefix_path,output,until=6.,horizon=.2,epsilon=.03):
    prefix=json.loads(Path(prefix_path).read_text());r=NativeHousehold(prefix['modelXml'],prefix['config']);bridge=ControllerBridge();started=time.monotonic()
    try:
        bridge.request(op='restore-prefix',state=prefix['controllerState'])
        mujoco.mj_setState(r.model,r.data,np.array(prefix['physicsState']),mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(r.model,r.data)
        r.last_action=np.array(prefix['lastAction'],np.float32);r.summary=prefix['summary'];r.contact=prefix['contact']
        for _ in range(1000):
            result=bridge.request(op='step',snapshot=r.snapshot(),meta=r.metadata(),context=r.controller_context(True),summary=r.summary)
            r.step(result['ctrl'])
            if result['phase']=='failed':raise RuntimeError('Source failed before requested probe time')
            if (result.get('innerTime') or 0)>=until:break
        else:raise RuntimeError('Probe time not reached')
        before=state(r.model,r.data);q=mujoco.MjData(r.model);stepper=ContactStepper(r.model,q,JOINTS)
        def predict(ctrl):
            mujoco.mj_copyData(q,r.model,r.data);q.ctrl[:]=ctrl
            audit=stepper.step(round(horizon/r.model.opt.timestep))
            return dict(point=point(r,q).tolist(),joint=q.qpos[r.qa].tolist(),minimumBeakZ=min(float((v@q.geom_xmat[g].reshape(3,3).T+q.geom_xpos[g])[:,2].min()) for g,v in r.beak_meshes),
                nonFoot=bool(audit[10]),penetration=float(max(audit[7],audit[8])),jointExcess=float(audit[11]),finite=bool(audit[12]),solverReset=bool(audit[13]))
        ctrl=r.data.ctrl.copy();baseline=predict(ctrl);columns=[];rollouts=[]
        for j in range(14):
            pair=[]
            for sign in [-1,1]:
                candidate=ctrl.copy();candidate[j]+=sign*epsilon;pair.append(predict(candidate))
            columns.append(((np.array(pair[1]['point'])-pair[0]['point'])/(2*epsilon)).tolist());rollouts.append(pair)
        np.testing.assert_array_equal(before,state(r.model,r.data))
        target=np.array(r.snapshot()['payload'][:3]);error=target-np.array(baseline['point']);J=np.array(columns).T
        correction=J.T@np.linalg.solve(J@J.T+np.eye(3)*.005**2,error)
        correction*=min(1,.06/max(np.max(np.abs(correction)),1e-12))
        candidate=ctrl.copy();candidate[:14]+=correction
        result=dict(scope='offline dynamics sensitivity only, not acceptance',prefixSha256=hashlib.sha256(Path(prefix_path).read_bytes()).hexdigest(),time=r.data.time,innerTime=until,horizon=horizon,epsilon=epsilon,
            sourceStateUnchanged=True,beforePoint=point(r,r.data).tolist(),target=target.tolist(),baseline=baseline,columns=columns,rollouts=rollouts,
            predictedCorrection=correction.tolist(),correctedPrediction=predict(candidate),wallSeconds=time.monotonic()-started)
        np.testing.assert_array_equal(before,state(r.model,r.data))
        Path(output).write_text(json.dumps(result,indent=2))
        print(json.dumps({k:result[k] for k in ['scope','time','sourceStateUnchanged','target','baseline','predictedCorrection','correctedPrediction','wallSeconds']}))
        stepper.close()
    finally:r.close();bridge.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--output',required=True);p.add_argument('--until',type=float,default=6.)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    probe(a.prefix,a.output,a.until)
