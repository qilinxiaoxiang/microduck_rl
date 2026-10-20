// The original JS controller and ONNX inference, with physics supplied by the
// native MuJoCo process. One JSON request/response per 50 Hz motor decision.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {fileURLToPath} from 'node:url';
import {JSDOM} from 'jsdom';
import * as ort from 'onnxruntime-node';
import {preparePhysicsXml} from '../src/game/physics-model.js';
import {householdConfig,makeHouseholdPickXml} from '../src/game/household-pick-scene.js';
import {HouseholdPickController} from '../src/game/household-pick-controller.js';
import {GroundPickController} from '../src/game/ground-pick-controller.js';
import {JawActionFilter} from '../src/game/jaw-rl.js';
import {POLICIES,DEFAULT_POSE} from '../src/game/constants.js';
import {evaluateHouseholdTrial} from '../src/game/household-pick-acceptance.js';
import {ReachResidual,residualObservation,validateResidualAction,RESIDUAL_CONTRACT} from '../src/game/household-residual.js';
const root=fileURLToPath(new URL('../',import.meta.url)),dom=new JSDOM('').window,sessions={};
let c,ctl,filter,r,k,context,residual;
function makeRuntime(meta){
  const r={head:0,m:{jnt_qposadr:meta.jntQposadr,jnt_range:meta.jntRange},qa:meta.qa,va:meta.va,d:{ctrl:new Float64Array(15),xmat:new Float64Array(9),qfrc_bias:[]}};
  r.policy=async(name,command)=>{
    sessions[name]??=await ort.InferenceSession.create(path.join(root,'public',POLICIES[name]),{executionProviders:['cpu'],intraOpNumThreads:1,interOpNumThreads:1});
    const obs=new Float32Array(61);obs.set(context.obsPrefix);obs.set(command,48);
    const output=await sessions[name].run({obs:new ort.Tensor('float32',obs,[1,61])});
    for(let j=0;j<14;j++)r.d.ctrl[j]=DEFAULT_POSE[j]+output.actions.data[j];
  };
  r.pointJacobian=(point,indices=[5,6,7,8])=>indices.map(j=>{
    const a=context.axes[j],o=context.anchors[j],v=point.map((p,i)=>p-o[i]);
    return [a[1]*v[2]-a[2]*v[1],a[2]*v[0]-a[0]*v[2],a[0]*v[1]-a[1]*v[0]];
  });
  r.jointAxes=(indices=[5,6,7,8])=>indices.map(j=>context.axes[j]);
  r.beakClearance=()=>context.clearance;
  r.predictedReachPlan=()=>context.motorPlan;
  r.upperSurfaceGap=()=>context.surfaceGap;
  r.stanceKinematics=point=>{
    const data=context.kinematics,v=point.map((p,i)=>p-data.headPoint[i]),rotation=data.headRotation;
    const position=data.headPosition.map(row=>row.slice());
    for(let j=0;j<20;j++){
      position[0][j]+=rotation[1][j]*v[2]-rotation[2][j]*v[1];
      position[1][j]+=rotation[2][j]*v[0]-rotation[0][j]*v[2];
      position[2][j]+=rotation[0][j]*v[1]-rotation[1][j]*v[0];
    }
    return {grip:{position,rotation},feet:data.feet,com:data.com};
  };
  return r;
}
const lines=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
for await(const line of lines){
  try{
    const request=JSON.parse(line);let response;
    if(request.op==='init'){
      c=householdConfig(request.config);ctl=c.approach?new HouseholdPickController(c):new GroundPickController(c);filter=new JawActionFilter();residual=new ReachResidual();k=0;r=null;
      const prepared=preparePhysicsXml(fs.readFileSync(path.join(root,'public/robot/mjlab/robot_allcollisions.xml'),'utf8'),dom);
      response={config:c,xml:makeHouseholdPickXml(prepared.xml,dom,c)};
    }else if(request.op==='step'){
      if(request.residual!==undefined)throw Error('Unbounded legacy residual input is not supported');
      if(request.residualAction)validateResidualAction(request.residualAction);
      r??=makeRuntime(request.meta);context=request.context;r.summary=request.summary;
      const s=request.snapshot;r.d.ctrl.set(s.ctrl);r.d.qfrc_bias=context.bias;r.d.xmat.set(context.headOrientation);
      if(c.approach&&k===(c.autoAt??50))ctl.begin(s);
      if(c.approach&&ctl.phase==='carried'){ctl.carriedSince??=s.time;if(s.time-ctl.carriedSince>(c.carrySeconds??2))ctl.release(s);}
      const manual=new Float32Array(13);if(c.manualCommand)manual.set(c.manualCommand);
      const target=await ctl.actuate(r,s,manual);r.d.ctrl[14]=filter.update(c.open?1:target===0?0:1);
      if(request.residualAction)residual.apply(r,ctl,request.residualAction);
      const servo=ctl.inner?.reachServo??ctl.reachServo;
      k++;response={ctrl:Array.from(r.d.ctrl),phase:ctl.phase,innerTime:ctl.inner?.time,graspNode:ctl.graspNode,alignment:ctl.alignment,reachServo:servo?{offset:servo.offset.slice(),error:servo.error?.slice(),point:servo.point?.slice(),clearance:servo.clearance}:undefined};
    }else if(request.op==='residual-contract')response=RESIDUAL_CONTRACT;
    else if(request.op==='observe'){
      response=residualObservation(request.snapshot,request.context.obsPrefix,request.context.headOrientation,ctl,residual.value);
    }else if(request.op==='capture-prefix'){
      // Only before the first feedback step: no class instances are silently
      // converted into plain objects. Used for training reset acceleration.
      if(!ctl.inner||ctl.inner.reachServo||ctl.inner.time>4.3)throw Error('Capture requires a pre-feedback physical prefix');
      response={c,ctl,filter,residual,k};
    }else if(request.op==='restore-prefix'){
      const saved=request.state;
      if(!saved.ctl.inner||saved.ctl.inner.reachServo||saved.ctl.inner.time>4.3)throw Error('Invalid training prefix');
      c=saved.c;ctl=Object.assign(new HouseholdPickController(c),saved.ctl);
      ctl.inner=Object.assign(new GroundPickController(c),saved.ctl.inner);
      filter=Object.assign(new JawActionFilter(),saved.filter);residual=Object.assign(new ReachResidual(),saved.residual);k=saved.k;r=null;response={phase:ctl.phase};
    }else if(request.op==='events')response={events:ctl.events,graspEvents:ctl.inner?.events,attempts:ctl.attempts};
    else if(request.op==='evaluate')response=evaluateHouseholdTrial(request.evidence,{requireRelease:!c.holdUntilRelease});
    else if(request.op==='close'){for(const session of Object.values(sessions))await session.release();dom.close();lines.close();console.log('{}');break;}
    else throw Error('Unknown operation');
    console.log(JSON.stringify(response));
  }catch(error){console.log(JSON.stringify({error:error.stack??error.message}));}
}
