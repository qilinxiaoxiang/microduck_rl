import assert from "node:assert/strict";
import {evaluateGroundTrial} from "../src/game/ground-pick-acceptance.js";
import {readPayloadContacts} from "../src/game/contact-observer.js";
import {GroundPickController} from "../src/game/ground-pick-controller.js";
import {stableGroundLanding} from "../../rl/ground_review_helpers.mjs";
// Both signs of floating-clock error must select identical motor phases.
async function clockTrace(error){
 const ctrl=new GroundPickController({z:.0142,closeAt:4.6}),runtime={d:{ctrl:new Float64Array(15),qfrc_bias:new Float64Array(14)},qa:Array.from({length:14},(_,i)=>i),va:Array.from({length:14},(_,i)=>i),m:{jnt_qposadr:Array.from({length:14},(_,i)=>i),jnt_range:Array.from({length:28},(_,i)=>i%2?2:-2)},async policy(){}};
 for(let tick=0;tick<=230;tick++)await ctrl.actuate(runtime,{time:tick*.02+error,root:[0,0,0,1,0,0,0],joint:Array(14).fill(0),contact:{}});
 return ctrl.events;
}
assert.deepEqual(await clockTrace(-1e-10),await clockTrace(1e-10));
assert.deepEqual((await clockTrace(0)).map(e=>e.time),[1,2.2,4.6000000000000005]);
// A thin pen may clear the floor without its center rising 40 mm. The
// experimental whole-body lift switches on measured whole-object clearance,
// with opposing contact; the independent hold/release verifier is unchanged.
async function liftTransition(minZ,upperForce=.1){
 const ctl=new GroundPickController({z:.0062,stanceLift:true,holdUntilRelease:true});
 Object.assign(ctl,{phase:'lift',riseStart:0,age:4});
 const r={d:{ctrl:new Float64Array(15),qfrc_bias:new Float64Array(14)},qa:Array.from({length:14},(_,i)=>i),va:Array.from({length:14},(_,i)=>i),m:{jnt_qposadr:Array.from({length:14},(_,i)=>i),jnt_range:Array.from({length:28},(_,i)=>i%2?2:-2)},async policy(){}};
 await ctl.actuate(r,{time:4,root:[0,0,.1,1,0,0,0],joint:Array(14).fill(0),payload:[0,0,.042,1,0,0,0],shape:{minZ},contact:{upperForce,lowerForce:.1,nonMouthSupport:false}});
 return ctl.phase;
}
assert.equal(await liftTransition(.036),'hold');
assert.equal(await liftTransition(.03),'lift');
assert.equal(await liftTransition(.036,0),'lift');
// A timeout must not switch to a frozen low pose or call a learned policy.
for(const phase of ['failed','clamp','lift','reach']){
 const ctl=new GroundPickController({z:.0062,holdUntilRelease:true,reachFeedback:true});
 Object.assign(ctl,{phase,age:20,riseStart:0});
 const previous=Float64Array.from({length:15},(_,j)=>j*.01);
 const r={d:{ctrl:previous.slice()},async policy(){assert.equal(phase,'reach');}};
 if(phase==='reach'){
  ctl.frozen=previous.slice();ctl.frozenHead=[0,0];ctl.frozenGaze=[0,0];
  ctl.reachServo={error:[0,0,0],apply(){r.d.ctrl.fill(1);return false;}};
 }
 const target=await ctl.actuate(r,{time:10,root:[0,0,.1,1,0,0,0],joint:Array(14).fill(0),payload:[0,0,.0062,1,0,0,0],shape:{minZ:0},contact:{lowerForce:0,upperForce:0}});
 assert.equal(ctl.phase,'failed');assert.equal(target,.32);assert.deepEqual(r.d.ctrl,previous);
}
const sample=(time,phase,z)=>({time,phase,root:[0,0,.12,1,0,0,0],payload:[0,0,z,1,0,0,0],graspPoint:[0,0,z],headQuat:[1,0,0,0],jaw:.32,shape:{minZ:z-.0142},contact:{lowerForce:phase==="hold"?.1:0,upperForce:phase==="hold"?.1:0,feetForce:7,nonFootSupport:false,nonMouthSupport:phase!=="hold",floorForce:phase==="hold"?0:.005}});
const landed=Array.from({length:16},(_,i)=>sample(i*.02,"release",.0142));
assert.equal(stableGroundLanding(landed),.32);
for(const mutate of [s=>delete s.contact.floorForce,s=>s.contact.floorForce=0,s=>s.shape.minZ=.001,s=>s.jaw=.1,s=>s.contact.upperForce=.1,s=>s.time-=1]){
 const invalid=structuredClone(landed);mutate(invalid.at(-1));assert.ok(stableGroundLanding(invalid)<.3);
}
const trial={phase:"done",config:{z:.0142},summary:{finite:true,solverReset:false,maxExternal:0,maxPenetration:.0001,maxRobotPenetration:.0001,nonFootSupport:false,maxJointLimitExcess:.0001,minRadialScale:.6},modelAudit:{freePayload:true,freeBase:true,flexAllNodesFree:true,equalities:0,mocapBodies:0,pedestalPresent:false,terrainPresent:false,floorHeight:0,deformable:true,flexVertices:99},rows:[...Array.from({length:50},(_,i)=>sample((i+1)*.02,"settle",.0142)),...Array.from({length:65},(_,i)=>sample(1.02+i*.02,"hold",.1)),sample(2.32,"release",.1),sample(3.5,"done",.0142)]};
assert.equal(evaluateGroundTrial(trial).accepted,true);
for(const mutate of [t=>t.modelAudit.pedestalPresent=true,t=>t.summary.solverReset=true,t=>t.summary.minRadialScale=-.01,t=>t.summary.nonFootSupport=true,t=>t.summary.maxExternal=.001,t=>t.summary.maxJointLimitExcess=.0021,t=>t.summary.maxPenetration=.0005,t=>t.rows.forEach(s=>s.contact.upperForce=0),t=>t.rows.forEach(s=>s.contact.nonMouthSupport=true),t=>t.rows.at(-1).shape.minZ=.01,t=>t.rows[75].time+=1]){
  const bad=structuredClone(trial);mutate(bad);assert.equal(evaluateGroundTrial(bad).accepted,false);
}
// Flexible contacts use geom=-1 plus a flex id; do not confuse them with
// missing geoms or classify arbitrary head/neck contact as a mouth rim.
const force=new Float64Array([.1,0,0,0,0,0]),scratch={GetView:()=>force};
const model={geom_bodyid:[0,1,1,1,1,1]},audit={robotGeoms:new Set([3,4,5]),feetGeoms:new Set(),head:1,mouthRimGeoms:new Set([5])};
const check=(other,z)=>{
 const c={geom1:-1,geom2:other,flex:[0,-1],dist:-.00001,efc_address:0,pos:[0,0,z],delete(){}};
 const data={ncon:1,contact:{get:()=>c,delete(){}},xpos:new Float64Array(6),xmat:new Float64Array([...Array(9).fill(0),1,0,0,0,1,0,0,0,1])};
 return readPayloadContacts({mj_contactForce(){}},model,data,[2],[3],[4],scratch,audit,0);
};
assert.equal(check(3,-.06).lowerForce,.1);assert.equal(check(4,-.06).upperForce,.1);
assert.equal(check(0,0).floorForce,.1);
assert.equal(check(5,-.077).mouthRimForce,.1);assert.equal(check(5,-.077).nonMouthSupport,false);
assert.equal(check(5,-.02).nonMouthSupport,true);assert.equal(check(5,-.02).mouthRimForce,undefined);
console.log("PASS: independent ground verifier, release, physical failure guards and flex/mouth-rim classification");
