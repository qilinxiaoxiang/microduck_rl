import assert from 'node:assert/strict';
import {GroundPickController} from '../src/game/ground-pick-controller.js';
for(const stopLiftClearance of [NaN,.02,.09])assert.throws(()=>new GroundPickController({stopLiftClearance}));
const config={stopLiftClearance:.04,holdUntilRelease:true,z:.0062};
const make=()=>Object.assign(new GroundPickController(config),{phase:'lift',riseStart:4});
const r={d:{ctrl:Float64Array.from({length:15},(_,i)=>i*.01)},policy:async()=>{throw Error('standing policy was called');}};
const controls=Array.from(r.d.ctrl),s={time:5,root:[0,0,.1,1,0,0,0],joint:Array(14).fill(0),payload:[0,0,.05],jaw:.22,shape:{minZ:.041},contact:{upperForce:.3,lowerForce:.3,feetForce:7,nonMouthSupport:false,nonFootSupport:false}};
const ctl=make();assert.equal(await ctl.actuate(r,s),0);assert.equal(ctl.phase,'hold');assert.deepEqual(Array.from(r.d.ctrl),controls);
for(let k=1;k<66;k++)await ctl.actuate(r,{...s,time:5+.02*k});
assert.equal(ctl.phase,'carried');assert.deepEqual(Array.from(r.d.ctrl),controls);
ctl.release({...s,time:6.4});assert.equal(await ctl.actuate(r,{...s,time:6.42}),.32);assert.deepEqual(Array.from(r.d.ctrl),controls);
for(const change of [{shape:{minZ:.039}},{contact:{...s.contact,upperForce:0}},{contact:{...s.contact,nonFootSupport:true}}]){
  const c=make();await assert.rejects(()=>c.actuate(r,{...s,...change}),/standing policy/);assert.equal(c.phase,'lift');
}
console.log('PASS: optional lift stopping requires real whole-object clearance and opposing contact, keeps motor targets, and allows natural release');
