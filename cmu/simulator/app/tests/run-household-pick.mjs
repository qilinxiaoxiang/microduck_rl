import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {wholeBodyRuntime} from './whole-body-runtime.mjs';
import {dom} from './physics-fixture.mjs';
import {householdConfig,makeHouseholdPickXml} from '../src/game/household-pick-scene.js';
import {GroundPickController} from '../src/game/ground-pick-controller.js';
import {JawActionFilter} from '../src/game/jaw-rl.js';
import {HouseholdPickController} from '../src/game/household-pick-controller.js';
import {evaluateHouseholdTrial} from '../src/game/household-pick-acceptance.js';
import {ReachResidual,residualObservation,RESIDUAL_CONTRACT} from '../src/game/household-residual.js';
import {householdResidualActor} from '../src/game/household-residual-actor.js';
const c=householdConfig(JSON.parse(process.argv[2]??'{}'));
const sourceFiles=fs.readdirSync(new URL('../src/game/',import.meta.url)).filter(f=>f.endsWith('.js')).sort();
const hash=x=>createHash('sha256').update(x).digest('hex');
const sources=Object.fromEntries(sourceFiles.map(f=>[f,hash(fs.readFileSync(new URL('../src/game/'+f,import.meta.url)))]));
const r=await wholeBodyRuntime({mutate:x=>makeHouseholdPickXml(x,dom,c)});
const ctl=c.approach?new HouseholdPickController(c):new GroundPickController(c),filter=new JawActionFilter(),rows=[];
const residual=c.residualActor?new ReachResidual():null,actor=c.residualActor?JSON.parse(fs.readFileSync(c.residualActor)):null;
const residualActorSha256=actor?hash(fs.readFileSync(c.residualActor)):null;
let residualTicks=0,residualAction=Array(14).fill(0);
const objectYaw=c.objectYaw??0;
r.reset([c.x,c.y,c.z+.0005,Math.cos(objectYaw/2),0,0,Math.sin(objectYaw/2)],{rootOffset:c.rootOffset});let contact={};
for(let k=0;k<(c.ticks??850);k++){
  const before=r.snapshot(contact);
  if(c.approach&&k===(c.autoAt??50))ctl.begin(before);
  if(c.approach&&ctl.phase==='carried'){ctl.carriedSince??=before.time;if(before.time-ctl.carriedSince>(c.carrySeconds??2))ctl.release(before);}
  const command=new Float32Array(13);if(c.manualCommand)command.set(c.manualCommand);
  if(residual&&(ctl.inner?.time??0)>=RESIDUAL_CONTRACT.prefixTime){
    if(residualTicks%5===0)residualAction=householdResidualActor(actor,residualObservation(before,Array.from(r.observation().slice(0,48)),Array.from(r.d.xmat.subarray(r.head*9,r.head*9+9)),ctl,residual.value));
    residualTicks++;
  }
  const target=await ctl.actuate(r,before,command),jaw=filter.update(c.open?1:target===0?0:1);
  if(residual)residual.apply(r,ctl,residualAction);
  contact=r.step(jaw);const servo=ctl.inner?.reachServo??ctl.reachServo;
  rows.push({...r.snapshot(contact),phase:ctl.phase,alignment:ctl.alignment,reachServo:servo?{offset:[...servo.offset],error:[...servo.error],point:[...servo.point],stable:servo.stable,orientationError:servo.orientationError?.slice(),clearance:servo.clearance}:undefined});
  if(c.stopAtAligned&&ctl.phase==='settle')break;
  if(c.output&&k%50===0)fs.writeFileSync(c.output,JSON.stringify({config:c,phase:ctl.phase,incomplete:true,modelAudit:r.modelAudit,summary:r.summary,rows}));
  if(k%100===0&&c.progress)console.error(JSON.stringify({k,phase:ctl.phase,root:rows.at(-1).root.slice(0,3),grasp:rows.at(-1).graspPoint,p:rows.at(-1).payload.slice(0,3),c:contact}));
  if(r.summary.solverReset||!r.summary.finite||['done','failed'].includes(ctl.phase))break;
}
const policies=Object.fromEntries(['BEST_alpha_walking.onnx','alpha_ground_pick.onnx'].map(f=>[f,hash(fs.readFileSync(new URL('../public/policies/'+f,import.meta.url)))]));
const provenance={sources,policies,residualActorSha256,residualInference:actor?'shared JavaScript deterministic PPO actor':null,runnerSha256:hash(fs.readFileSync(new URL(import.meta.url))),modelSha256:hash(r.xml),engine:'MuJoCo 3.11.0',inference:'onnxruntime-node CPU',sourcesUnchanged:sourceFiles.every(f=>sources[f]===hash(fs.readFileSync(new URL('../src/game/'+f,import.meta.url))))};
const result={config:c,phase:ctl.phase,events:ctl.events,graspEvents:ctl.inner?.events,attempts:ctl.attempts,alignment:ctl.alignment,modelAudit:r.modelAudit,summary:r.summary,provenance,modelXml:r.xml,rows};
if(!['done','failed'].includes(ctl.phase)&&!c.stopAtAligned&&!(c.holdUntilRelease&&ctl.phase==='carried')){
  result.incomplete=true;result.failure='Control-step budget exhausted before the requested sequence completed';
}
result.acceptance=evaluateHouseholdTrial(result,{requireRelease:!c.holdUntilRelease});
if(result.incomplete){result.acceptance.accepted=false;result.acceptance.failures.push(result.failure);}
if(!provenance.sourcesUnchanged){result.acceptance.accepted=false;result.acceptance.failures.push('试验期间源文件发生变化');}
if(c.output)fs.writeFileSync(c.output,JSON.stringify(result)+'\n');
console.log(JSON.stringify({...result,rows:undefined,modelXml:undefined,opposingFrames:rows.filter(s=>s.contact.upperForce>.002&&s.contact.lowerForce>.002).length,last:rows.at(-1)}));await r.dispose();
