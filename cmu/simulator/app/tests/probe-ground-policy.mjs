// Dynamic development probe: upstream ground-pick motor policy with optional
// motor residuals from config. No supports; never the final acceptance runner.
import fs from "node:fs";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom} from "./physics-fixture.mjs";
import {makeGroundPickXml,GROUND_PICK_DEFAULTS} from "../src/game/ground-pick-scene.js";
import {rootAngles} from "../src/game/table-pick-controller.js";
const c={...GROUND_PICK_DEFAULTS,...JSON.parse(process.argv[2]??"{}")};
const r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom,c)}),rows=[];
r.reset([c.x,c.y,c.z+.0005,1,0,0,0]);
let contact={},frozen,frozenHead,lastPitch;
for(let k=0;k<(c.ticks??250);k++){
  const command=new Float32Array(13),phase=Math.min(c.holdPhase??Infinity,Math.max(0,(k-50)*.02/(c.period??4)));
  const policy=k<50||phase>.7?"walk":"groundpick";
  if(policy==="groundpick"){command[0]=Math.cos(2*Math.PI*phase);command[1]=Math.sin(2*Math.PI*phase);}
  if(c.freeze&&k*.02>=2.2){frozen??=Array.from(r.d.ctrl);frozenHead??=r.qa.slice(5,7).map(q=>r.d.qpos[q]);}
  if(frozen)r.d.ctrl.set(frozen);else await r.policy(policy,command);
  const blend=Math.max(0,Math.min(1,(k*.02-2.2)/(c.ramp??3)));
  if(c.crouch){const [a,b,h]=c.crouchCoeffs??[.7,1,.3];for(const [j,v] of [[2,a],[3,b],[4,h],[11,-a],[12,-b],[13,-h]])r.d.ctrl[j]+=v*c.crouch*blend;}
  if(c.lean){r.d.ctrl[4]-=c.lean*blend;r.d.ctrl[13]+=c.lean*blend;}
  if(c.neck)r.d.ctrl[5]+=c.neck*blend;
  if(c.head)r.d.ctrl[6]+=c.head*blend;
  if(c.targetHead&&frozenHead)for(let j=0;j<2;j++)r.d.ctrl[5+j]=frozenHead[j]+(c.targetHead[j]-frozenHead[j])*blend+r.d.qfrc_bias[r.va[5+j]]/.55;
  const pitch=rootAngles(Array.from(r.d.qpos.subarray(3,7))).pitch;
  if(c.worldAngle!==undefined&&frozenHead){
    const neck=frozenHead[0]+((c.neckTarget??-1.5)-frozenHead[0])*blend;
    const h=neck-pitch+Math.PI/2+c.worldAngle*blend;
    r.d.ctrl[5]=neck+r.d.qfrc_bias[r.va[5]]/.55;
    r.d.ctrl[6]=Math.max(-1.53,Math.min(1.53,h))+r.d.qfrc_bias[r.va[6]]/.55;
  }
  if(c.balance&&frozen){const correction=.8*(pitch-(c.pitchTarget??.55))+.08*(pitch-(lastPitch??pitch))/.02;r.d.ctrl[4]+=correction;r.d.ctrl[13]-=correction;}
  lastPitch=pitch;
  if(c.bound)for(let j=0;j<14;j++){
    const ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]);
    r.d.ctrl[j]=Math.max(r.m.jnt_range[ji*2]+(c.margin??.015),Math.min(r.m.jnt_range[ji*2+1]-(c.margin??.015),r.d.ctrl[j]));
  }
  const jaw=c.closeAt!==undefined&&k*.02>=c.closeAt?Math.max(0,.32-.6*(k*.02-c.closeAt)):(c.jaw??.32);
  contact=r.step(jaw);
  const s=r.snapshot(contact);rows.push({...s,phase});
  if(r.summary.solverReset)break;
  if(contact.nonFootPair)rows.at(-1).nonFootNames=contact.nonFootPair.map(g=>r.mj.mj_id2name(r.m,r.mj.mjtObj.mjOBJ_GEOM.value,g));
  if(k%50===0&&!c.quiet)console.log(JSON.stringify({t:s.time,phase,point:s.graspPoint,root:s.root,head:s.joint.slice(5,9),feet:contact.feetForce,other:contact.nonFootSupport,robotDepth:contact.robotMaxPenetration}));
}
const result={config:c,rows,summary:r.summary,modelAudit:r.modelAudit};
if(c.output)fs.writeFileSync(c.output,JSON.stringify(result)+"\n");
console.log(JSON.stringify({summary:r.summary,minimumPoint:rows.reduce((a,b)=>a.graspPoint[2]<b.graspPoint[2]?a:b)}));
await r.dispose();
