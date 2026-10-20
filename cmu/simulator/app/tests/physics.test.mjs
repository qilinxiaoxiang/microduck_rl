import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { scene, id, mj, meshDir, appDir } from "./physics-fixture.mjs";
import parts from "../src/game/beak-colliders.js";
import { readPayloadContacts, createContactWindow, addContactSample, averageContactWindow } from "../src/game/contact-observer.js";
import { TIMESTEP, DECIMATION } from "../src/game/constants.js";

const results=[],failures=[];
const check=(condition,message)=>assert.ok(condition,message);
assert.equal(TIMESTEP*DECIMATION,.02,"policy interface must remain 50 Hz");
for(const asset of Object.values(parts)) {
  assert.equal(crypto.createHash("sha256").update(fs.readFileSync(path.join(meshDir,asset.source))).digest("hex"),asset.sha256);
  for(const f of asset.parts)check(fs.statSync(path.join(meshDir,f)).size>84,`missing collider ${f}`);
}
// Regression for a copied Embind vector and handles on early-continue paths.
{
  let vectors=0,contacts=0,reads=0;
  const data={ncon:2,get contact(){reads++;return {get(i){return {geom1:99,geom2:i,delete(){contacts++;}};},delete(){vectors++;}};}};
  readPayloadContacts(null,null,data,[42],[],[],{});
  assert.deepEqual([reads,vectors,contacts],[1,1,2]);
}
{
  const s=scene();
  assert.equal(s.model.opt.timestep,TIMESTEP);
  assert.equal(s.model.neq,0,"no payload weld or equality attachment");
  assert.equal(s.model.nu,15,"14 policy actuators plus physical beak hinge");
  check(!s.xml.includes("beak_upper_contact_geom"),"obsolete box proxy survived");
  s.dispose();
}

function bench({label,x=.075,y=0,mass=.0025,open=false,dt=TIMESTEP}) {
  const z=.124;
  const s=scene("legs",true,{x,y,z},xml=>xml.replace('mass="0.0025"',`mass="${mass}"`).replace(`timestep="${TIMESTEP}"`,`timestep="${dt}"`));
  const m=s.model,d=s.data;
  const p=m.jnt_qposadr[id(m,"JOINT","payload_freejoint")], jaw=m.jnt_qposadr[id(m,"JOINT","beak_joint")];
  const payload=[id(m,"GEOM","payload_geom")],body=m.geom_bodyid[payload[0]];
  assert.equal(m.body_mass[body],mass);
  const lower=["jaw","jaw_soft"].flatMap(source=>parts[source].parts.map((_,i)=>id(m,"GEOM",`beak_${source}_contact_${i}`)));
  const upper=parts.soft_mouth_top.parts.map((_,i)=>id(m,"GEOM",`beak_soft_mouth_top_contact_${i}`));
  const scratch=new mj.DoubleBuffer(6);
  mj.mj_resetData(m,d);
  // Initialization only. There are no payload pose/velocity/force writes in
  // the rollout below. A physical one-axis position servo lifts the head.
  d.qpos.set([x,y,z,1,0,0,0],p);d.qpos[jaw]=.32;mj.mj_forward(m,d);
  const result={label,dt,x,y,mass,initialPenetration:0,maxPenetration:0,holdMaxPenetration:0,
    holdSupportedFraction:0,liftAt3s:0,dropAfterRelease:0,maxPayloadExternal:0,heapStart:d.qpos.buffer.byteLength};
  let held=0,holdSteps=0,zAt3=0,instantHeld=0,instantSteps=0;
  let window=createContactWindow();
  const start=performance.now();
  try {
    for(let k=0;k<Math.round(4/dt);k++) {
      const t=k*dt;
      d.ctrl[1]=.05*Math.min(1,Math.max(0,t-1));
      d.ctrl[0]=open||t>=3?.32:0;
      mj.mj_step(m,d);
      const c=readPayloadContacts(mj,m,d,payload,lower,upper,scratch);
      addContactSample(window,c);
      if(k===0)result.initialPenetration=c.maxPenetration;
      result.maxPenetration=Math.max(result.maxPenetration,c.maxPenetration);
      for(const value of d.xfrc_applied.subarray(body*6,body*6+6))result.maxPayloadExternal=Math.max(result.maxPayloadExternal,Math.abs(value));
      check(Number.isFinite(d.qpos[p+2]),`${label}: non-finite state`);
      if(t>=2 && t<3) {
        instantSteps++;
        const supported=c.lowerForce>.002 && c.upperForce>.002 && !c.otherSupport && d.qpos[p+2]>z+.03;
        if(supported)instantHeld++;
        result.holdMaxPenetration=Math.max(result.holdMaxPenetration,c.maxPenetration);
        zAt3=d.qpos[p+2];
      }
      if((k+1)%Math.round(.02/dt)===0) {
        const averaged=averageContactWindow(window);
        if(t>=2 && t<3) {
          holdSteps++;
          if(averaged.lowerForce>.002 && averaged.upperForce>.002 && !averaged.otherSupport && d.qpos[p+2]>z+.03)held++;
        }
        window=createContactWindow();
      }
    }
    result.holdSupportedFraction=held/holdSteps;
    result.holdInstantOpposingFraction=instantHeld/instantSteps;
    result.liftAt3s=zAt3-z;
    result.dropAfterRelease=zAt3-d.qpos[p+2];
    result.finalContacts=readPayloadContacts(mj,m,d,payload,lower,upper,scratch);
    result.finalJawAngle=d.qpos[jaw];
    result.heapEnd=d.qpos.buffer.byteLength;
    result.wallSeconds=(performance.now()-start)/1000;
    results.push(result);console.log(JSON.stringify(result));
    check(result.initialPenetration<.00001,`${label}: intersecting initial pose`);
    check(result.maxPenetration<.001,`${label}: impact penetration > 1 mm`);
    assert.equal(result.maxPayloadExternal,0,`${label}: external payload force`);
    if(open)check(result.holdSupportedFraction===0,`${label}: open mouth must not count as a grip`);
    else {
      check(result.holdSupportedFraction>.95,`${label}: unsupported hold (${result.holdSupportedFraction})`);
      check(result.holdMaxPenetration<.00005,`${label}: sustained penetration > 0.05 mm`);
      check(result.liftAt3s>.03,`${label}: failed lift`);
      check(result.finalJawAngle>.31,`${label}: jaw did not open`);
      check(result.finalContacts.upperForce<.002,`${label}: opposing grip remained after opening`);
      // A deeper object can rest on the lower bill like a tray after the
      // upper pad releases. Do not falsely call one-sided support a grasp.
      // The centered nominal case must additionally fall freely.
      if(x===.075)check(result.dropAfterRelease>.03,`${label}: nominal release did not fall`);
    }
    check(result.heapEnd-result.heapStart<16*1024*1024,`${label}: unbounded contact read memory`);
  } finally {scratch.delete();s.dispose();}
}
for(const item of [
  {label:"center"},{label:"5mm deeper",x:.07},{label:"5mm shallower",x:.08},
  {label:"2mm left",y:.002},{label:"2mm right",y:-.002},
  {label:"half mass",mass:.00125},{label:"double mass",mass:.005},
  {label:"open-mouth negative control",open:true},
  {label:"half-timestep convergence",dt:TIMESTEP/2},
]) {try{bench(item);}catch(error){failures.push({label:item.label,error:error.message});}}
const nominal=results.find(r=>r.label==="center"),fine=results.find(r=>r.label==="half-timestep convergence");
if(nominal && fine && Math.abs(nominal.liftAt3s-fine.liftAt3s)>.0002)failures.push({label:"convergence",error:"lift differs by > 0.2 mm at half timestep"});

// Free fall checks gravity, floor contact and no invisible attachment.
{
  const s=scene(),m=s.model,d=s.data;
  const p=m.jnt_qposadr[id(m,"JOINT","payload_freejoint")];
  const payload=[id(m,"GEOM","payload_geom")],scratch=new mj.DoubleBuffer(6);
  mj.mj_resetData(m,d);d.qpos.set([.5,.5,.2,1,0,0,0],p);mj.mj_forward(m,d);
  let maxPenetration=0;
  for(let k=0;k<Math.round(1/TIMESTEP);k++) {
    mj.mj_step(m,d);
    maxPenetration=Math.max(maxPenetration,readPayloadContacts(mj,m,d,payload,[],[],scratch).maxPenetration);
    if(k===Math.round(.1/TIMESTEP)-1)check(Math.abs(d.qpos[p+2]-(.2-.5*9.81*.1**2))<.001,"wrong free-fall acceleration");
  }
  check(Math.abs(d.qpos[p+2]-.003)<.0001,"floor did not stop free payload");
  check(maxPenetration<.001,"free-fall floor impact excessive");
  results.push({label:"free fall",maxPenetration,restZ:d.qpos[p+2]});scratch.delete();s.dispose();
}
const output=path.join(appDir,"test-results/physics-validation.json");
fs.mkdirSync(path.dirname(output),{recursive:true});
fs.writeFileSync(output,JSON.stringify({generatedAt:new Date().toISOString(),engine:"MuJoCo WASM 3.11.0",scope:"isolated physical head-lift fixture, not whole-body pickup or hardware validation",results,failures},null,2)+"\n");
assert.deepEqual(failures,[],"physics acceptance failures (see JSON artifact)");
console.log(`PASS: ${results.length} dynamic cases; geometry, model, handle lifetime. ${output}`);
