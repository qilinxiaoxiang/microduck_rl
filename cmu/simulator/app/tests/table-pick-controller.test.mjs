import assert from "node:assert/strict";
import {TablePickController} from "../src/game/table-pick-controller.js";

const state=()=>({root:[-.55,0,.117,1,0,0,0],graspPoint:[-.428,0,.195],
  payload:[-.42,0,.195,1,0,0,0],joint:Array(14).fill(0),
  contact:{lowerForce:.09,upperForce:.07,maxPenetration:.00001,otherSupport:true,feetForce:8,nonFootSupport:false}});
{
  const ctl=new TablePickController({height:.195});
  Object.assign(ctl,{phase:"clamp",neck:-1.16});
  const s=state();
  for(let i=0;i<11;i++)ctl.update(s);
  assert.equal(ctl.phase,"lift","clamp must not skip lifting on its transition tick");
  const out=ctl.update(s);
  assert.ok(out.head[1]<out.head[0],"positive head-pitch rotation lowers the bill; lifting must initially tilt upward");
  assert.equal(out.jaw,0);
  assert.equal(ctl.hold,0,"contact on the pedestal is not an elevated hold");
}
for(const invalid of ["pedestal","noUpper","bodySupport","noFeet","notHigh"]){
  const ctl=new TablePickController({height:.195}),s=state();
  Object.assign(ctl,{phase:"hold",liftStarted:0});
  s.graspPoint[2]=.229;s.payload[2]=.23;s.contact.otherSupport=false;
  if(invalid==="pedestal")s.contact.otherSupport=true;
  if(invalid==="noUpper")s.contact.upperForce=0;
  if(invalid==="bodySupport")s.contact.nonFootSupport=true;
  if(invalid==="noFeet")s.contact.feetForce=0;
  if(invalid==="notHigh")s.payload[2]=.20;
  for(let i=0;i<70;i++)ctl.update(s);
  assert.equal(ctl.hold,0,`${invalid} must not pass hold acceptance`);
}
{
  const ctl=new TablePickController(),s=state();s.root[2]=.04;
  ctl.update(s);assert.equal(ctl.phase,"failed","fallen robot cannot claim task success");
}
console.log("PASS task controller: phase boundary, lift direction, negative hold gates and fall detection");
