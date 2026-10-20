import assert from 'node:assert/strict';
import {ReachResidual,residualObservation,RESIDUAL_OBS_SIZE,RESIDUAL_CONTRACT} from '../src/game/household-residual.js';
import {householdResidualActor} from '../src/game/household-residual-actor.js';
const residual=new ReachResidual(),r={d:{ctrl:new Float64Array(15)},qa:Array.from({length:14},(_,j)=>j),m:{jnt_qposadr:Array.from({length:14},(_,j)=>j),jnt_range:Array.from({length:28},(_,j)=>j%2?2:-2)}};
for(const action of [Array(13).fill(0),[NaN,...Array(13).fill(0)]]){
 assert.throws(()=>residual.apply(r,{phase:'reach',time:5},action));assert.deepEqual(Array.from(r.d.ctrl),Array(15).fill(0));
}
residual.apply(r,{phase:'reach',time:5},Array(14).fill(1));
assert.ok(Math.abs(r.d.ctrl[5]-.036)<1e-12);assert.ok(Math.abs(r.d.ctrl[0]-.030)<1e-12);assert.equal(r.d.ctrl[14],0);
for(const phase of ['manual','lift','failed']){
 const before=Array.from(r.d.ctrl);residual.apply(r,{phase,time:5},Array(14).fill(1));assert.deepEqual(Array.from(r.d.ctrl),before);assert.deepEqual(residual.value,Array(14).fill(0));
}
const s={root:[0,0,.1,1,0,0,0],payload:[.1,0,.01,1,0,0,0],graspPoint:[.09,0,.02],shape:{minZ:.004},contact:{}};
const obs=residualObservation(s,Array(48).fill(0),[1,0,0,0,1,0,0,0,1],{phase:'reach',age:1});
assert.equal(obs.length,RESIDUAL_OBS_SIZE);assert.ok(obs.every(Number.isFinite));
const c={phase:'reach',age:1,c:{kind:'pen'}},R=[1,0,0,0,1,0,0,0,1];
const rolled={...s,payload:[.1,0,.01,Math.cos(1),0,Math.sin(1),0]};
assert.deepEqual(residualObservation(s,Array(48).fill(0),R,c),residualObservation(rolled,Array(48).fill(0),R,c),'axial roll must not change the residual heading observation');
assert.throws(()=>householdResidualActor({version:1,observationSize:93,actionSize:14},obs),/Incompatible/);
assert.throws(()=>householdResidualActor({...RESIDUAL_CONTRACT,scales:Array(14).fill(.1)},obs),/Incompatible/);
const before=Array.from(r.d.ctrl);residual.apply(r,{phase:'reach',time:3.0},Array(14).fill(1));assert.deepEqual(Array.from(r.d.ctrl),before);
console.log('PASS: shared residual observations, bounded motor-only corrections, atomic invalid actions and phase mask');
