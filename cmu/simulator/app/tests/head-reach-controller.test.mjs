import assert from "node:assert/strict";
import {HeadReachController} from "../src/game/head-reach-controller.js";
import {evaluateWholeBodyTrial} from "../src/game/whole-body-acceptance.js";
const state=()=>({root:[-.6,0,.117,1,0,0,0],graspPoint:[-.494,0,.195],payload:[-.48,0,.195,1,0,0,0],
  headJac:[[.02,-.01],[.03,.02]],joint:Array(14).fill(0),jaw:.12,
  contact:{lowerForce:.3,upperForce:.28,otherSupport:true,nonFootSupport:false,feetForce:7.2,maxPenetration:0}});
{
  const s=state(),original=structuredClone(s),ctl=new HeadReachController();
  Object.assign(ctl,{phase:"clamp",gripPoint:[-.494,.195]});
  for(let i=0;i<20;i++)ctl.update(s);
  assert.equal(ctl.phase,"lift");assert.equal(ctl.hold,0);
  assert.deepEqual(s,original,"controller must not mutate physics readback");
}
for(const invalid of ["pedestal","noUpper","bodySupport","noFeet","notHigh","penetration"]){
  const ctl=new HeadReachController(),s=state();
  Object.assign(ctl,{phase:"hold",liftStarted:0,gripPoint:[-.494,.195]});
  s.graspPoint[2]=.235;s.payload[2]=.24;s.contact.otherSupport=false;
  if(invalid==="pedestal")s.contact.otherSupport=true;
  if(invalid==="noUpper")s.contact.upperForce=0;
  if(invalid==="bodySupport")s.contact.nonFootSupport=true;
  if(invalid==="noFeet")s.contact.feetForce=0;
  if(invalid==="notHigh")s.payload[2]=.20;
  if(invalid==="penetration")s.contact.maxPenetration=.0006;
  for(let i=0;i<70;i++)ctl.update(s);
  assert.equal(ctl.hold,0,invalid);
}
{
  const ctl=new HeadReachController(),s=state();
  ctl.phase="reach";s.joint[5]=-1.3;s.graspPoint[0]=-.4946;
  assert.ok(ctl.update(s).ankleLean>0,"ankle correction must not stop outside the mouth alignment tolerance");
}
const rows=Array.from({length:60},(_,i)=>({t:(i+1)*.02,phase:"hold",point:[0,0,.24],object:[.01,0,.24],
  headQuat:[1,0,0,0],jaw:.12,contact:{lowerForce:.3,upperForce:.28,otherSupport:false,nonFootSupport:false,feetForce:7.2,maxPenetration:0}}));
rows.push({...structuredClone(rows[0]),t:1.22,phase:"done",jaw:.32,object:[.01,0,.003],contact:{lowerForce:0,upperForce:0}});
const trial={phase:"done",rows,config:{z:.195},summary:{maxExternal:0,finite:true,maxPenetration:0,maxRobotPenetration:0,nonFootSupport:false,maxJointLimitExcess:0},modelAudit:{freePayload:true,freeBase:true,equalities:0,mocapBodies:0}};
assert.equal(evaluateWholeBodyTrial(trial).accepted,true);
for(const mutate of [
  r=>r.summary.maxExternal=1,r=>r.summary.maxJointLimitExcess=.1,r=>r.modelAudit.freeBase=false,
  r=>r.modelAudit.equalities=1,r=>r.rows.splice(0,20),r=>r.rows.at(-1).contact.lowerForce=.02,
  r=>r.rows[35].object[0]+=.005,r=>r.rows[30].contact.otherSupport=true,
]){const broken=structuredClone(trial);mutate(broken);assert.equal(evaluateWholeBodyTrial(broken).accepted,false);}
console.log("PASS: end-point controller and independent negative acceptance gates");
