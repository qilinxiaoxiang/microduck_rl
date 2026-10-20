"""Full normal-reset predictive-control development evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from native_household import NativeHousehold
from run_native_household import ControllerBridge,source_hashes,ROOT
from predictive_reach import PredictiveReach

p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=int,default=480);p.add_argument('--axis-grasp',action='store_true');p.add_argument('--grasp-forecast',action='store_true');a=p.parse_args()
if not 1<=a.seconds<=600:raise ValueError('Use a bounded development evaluation')
c=json.loads(a.cases.read_text())[a.index];c={**c,'dynamicsReach':True,'reachTimeout':30,'stanceLift':False,'ticks':4000}
if a.axis_grasp:c['axisGrasp']=True
if a.grasp_forecast:c['graspForecast']=True
a.output.mkdir(parents=True,exist_ok=False);hashes=source_hashes()
for name in ['rl/predictive_reach.py','rl/evaluate_predictive_reach.py','rl/prediction_stepper.py','rl/grasp_forecast.py']:hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
for name in hashes:
    dest=a.output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
bridge=ControllerBridge();init=bridge.request(op='init',config=c);r=NativeHousehold(init['xml'],init['config']);planner=PredictiveReach(r);rows=[];phase='manual';started=time.monotonic();inner=0;failure=None;teacher=[]
planner.deadline=started+a.seconds
interrupted_forecast=None
def progress(value):
    pending=a.output/'progress.tmp';pending.write_text(json.dumps(value));pending.replace(a.output/'progress.json')
    print(json.dumps(value),flush=True)
try:
    for tick in range(c['ticks']):
        context=r.controller_context(True)
        if phase=='reach' and inner>=4.38 or c.get('dynamicsLift') and phase in ['lift','hold','carried']:
            previous_updates=planner.updates
            forecast_started=time.monotonic()
            try:context['motorPlan']=planner.update('transport' if phase in ['lift','hold','carried'] else 'reach')
            except TimeoutError as error:
                interrupted_forecast=forecast_started;failure=str(error);break
            if planner.updates!=previous_updates:
                progress(dict(time=r.data.time,phase=phase,updates=planner.updates,plannerSeconds=planner.total_seconds,elapsedSeconds=time.monotonic()-started,plan=context['motorPlan']))
                teacher.append(dict(time=r.data.time,phase=phase,observation=bridge.request(op='observe',snapshot=r.snapshot(),context=context),
                    targets=context['motorPlan']['ctrl'],beforeCtrl=r.data.ctrl[:14].tolist(),audit=context['motorPlan']['audit'],blocked=context['motorPlan']['blocked']))
        result=bridge.request(op='step',snapshot=r.snapshot(),meta=r.metadata(),context=context,summary=r.summary)
        if result.get('graspNode') is not None:planner.set_cloth_node(result['graspNode'])
        row=r.step(result['ctrl']);phase=row['phase']=result['phase'];inner=result.get('innerTime') or 0;row['reachServo']=result.get('reachServo');row['alignment']=result.get('alignment');row['motorPlan']=context.get('motorPlan');rows.append(row)
        if tick%50==0:progress(dict(time=r.data.time,phase=phase,updates=planner.updates,error=result.get('reachServo',{}).get('error'),plannerSeconds=planner.total_seconds,elapsedSeconds=time.monotonic()-started))
        if phase in ['done','failed']:break
        if time.monotonic()-started>a.seconds:failure='Predictive-control wall-time budget exhausted';break
    planner.close() # Drain bounded in-flight forecasts before final time/cost readback.
    events=bridge.request(op='events')
    e=dict(config=init['config'],phase=phase,rows=rows,summary=r.summary,modelAudit=r.model_audit,modelXml=r.xml,trainingPrefixReset=False,plannerUpdates=planner.updates,plannerWallSeconds=planner.total_seconds,
        wallSeconds=time.monotonic()-started,provenance=dict(engine='MuJoCo 3.11.0 native',controller='isolated short-horizon model-based motor planning, not PPO',sources=hashes),**events)
    e['teacherSamples']=teacher
    if interrupted_forecast is not None:e['interruptedForecastWallSeconds']=time.monotonic()-interrupted_forecast
    e['acceptance']=bridge.request(op='evaluate',evidence={key:e[key] for key in ['phase','rows','summary','modelAudit']})
    if failure or phase not in ['done','failed']:
        e['incomplete']=True;e['failure']=failure or 'Control-step budget exhausted';e['acceptance']['accepted']=False;e['acceptance']['failures'].append(e['failure'])
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value for name,value in hashes.items()):
        e['acceptance']['accepted']=False;e['acceptance']['failures'].append('Sources changed during evaluation')
    (a.output/'trial-000.json').write_text(json.dumps(e));print(json.dumps({key:value for key,value in e.items() if key not in ['rows','modelXml','provenance','teacherSamples']}))
finally:planner.close();r.close();bridge.close()
