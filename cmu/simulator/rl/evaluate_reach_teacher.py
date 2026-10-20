"""Full ordinary-reset evaluation of the fast supervised reach pilot."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from native_household import NativeHousehold
from run_native_household import ControllerBridge,source_hashes,ROOT

def actor(model,observation):
    if model.get('type')!='household-teacher-motor-delta' or model.get('version')!=1:raise ValueError('Wrong teacher policy contract')
    x=np.asarray(observation)
    if x.shape!=(93,) or not np.all(np.isfinite(x)):raise ValueError('Invalid observation')
    x=np.clip((x-model['mean'])/model['scale'],-10,10)
    for layer in model['layers']:
        x=np.array(layer['weight'])@x+layer['bias']
        if layer['activation']=='tanh':x=np.tanh(x)
        elif layer['activation']!='linear':raise ValueError('Invalid activation')
    if x.shape!=(14,) or not np.all(np.isfinite(x)):raise ValueError('Invalid motor action')
    return np.clip(x,-1,1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--actor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    c=json.loads(a.cases.read_text())[a.index];c={**c,'dynamicsReach':True,'reachTimeout':30,'stanceLift':False,'axisGrasp':True,'ticks':4000}
    model=json.loads(a.actor.read_text());a.output.mkdir(parents=True,exist_ok=False);hashes=source_hashes()
    for name in ['rl/train_reach_teacher.py','rl/evaluate_reach_teacher.py']:hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    for name in hashes:
        dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    shutil.copyfile(a.actor,a.output/'source/actor.json')
    bridge=ControllerBridge();init=bridge.request(op='init',config=c);r=NativeHousehold(init['xml'],init['config']);rows=[];phase='manual';inner=0;last_time=-1;plan=None;started=time.monotonic();updates=0
    try:
        for tick in range(c['ticks']):
            context=r.controller_context(True)
            if phase=='reach' and inner>=4.38:
                if plan is None or r.data.time-last_time>=.199:
                    obs=bridge.request(op='observe',snapshot=r.snapshot(),context=context)
                    change=actor(model,obs)*model['deltaScale'];ctrl=np.clip(r.data.ctrl[:14]+change,r.model.jnt_range[r.joint_ids,0]+.04,r.model.jnt_range[r.joint_ids,1]-.04)
                    updates+=1;last_time=r.data.time;plan=dict(ctrl=ctrl.tolist(),blocked=False,audit=dict(type='supervised actor',physicalForecast=False,update=updates,action=change.tolist()))
                context['motorPlan']=plan
            result=bridge.request(op='step',snapshot=r.snapshot(),meta=r.metadata(),context=context,summary=r.summary)
            row=r.step(result['ctrl']);phase=row['phase']=result['phase'];inner=result.get('innerTime') or 0;row['reachServo']=result.get('reachServo');row['motorPlan']=context.get('motorPlan');rows.append(row)
            audit=r.summary
            if audit['nonFootSupport'] or max(audit['maxPenetration'],audit['maxRobotPenetration'])>=.0005 or audit['maxJointLimitExcess']>=.002 or audit['solverReset'] or audit['maxExternal'] or any(audit['warningCounts'].values()):
                phase='failed';break
            if phase in ['done','failed'] or not r.summary['finite'] or time.monotonic()-started>120:break
        events=bridge.request(op='events')
        e=dict(config=init['config'],phase=phase,rows=rows,summary=r.summary,modelAudit=r.model_audit,modelXml=r.xml,trainingPrefixReset=False,
               policyUpdates=updates,wallSeconds=time.monotonic()-started,provenance=dict(engine='MuJoCo 3.11.0 native',controller='supervised motor-delta actor, not PPO or MPC',sources=hashes,actorSha256=hashlib.sha256(a.actor.read_bytes()).hexdigest()),**events)
        e['acceptance']=bridge.request(op='evaluate',evidence={key:e[key] for key in ['phase','rows','summary','modelAudit']})
        if phase not in ['done','failed']:e['incomplete']=True;e['failure']='Evaluation budget exhausted';e['acceptance']['accepted']=False
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value for name,value in hashes.items()):e['failure']='Sources changed';e['acceptance']['accepted']=False
        (a.output/'trial-000.json').write_text(json.dumps(e));print(json.dumps({key:value for key,value in e.items() if key not in ['rows','modelXml','provenance']}))
    finally:r.close();bridge.close()
