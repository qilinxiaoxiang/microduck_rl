// Phase/motor contract checks; these synthetic fixtures are not pickup evidence.
import assert from 'node:assert/strict';
import {GroundPickController} from '../src/game/ground-pick-controller.js';
import {HouseholdPredictiveServo} from '../src/game/household-predictive-servo.js';
const c={kind:'pen',length:.14,dynamicsReach:true,dynamicsLift:true,stopLiftClearance:.04,holdUntilRelease:true,z:.0062};
let plan={stage:'reach',ctrl:Array(14).fill(.2),blocked:false};
const r={head:0,d:{ctrl:new Float64Array(15),xmat:Float64Array.from([1,0,0,0,1,0,0,0,1])},
  predictedReachPlan:()=>plan,beakClearance:()=>({height:.01}),policy:async()=>{throw Error('Unexpected learned body-policy switch');}};
const ctl=Object.assign(new GroundPickController(c),{phase:'lift',reachServo:new HouseholdPredictiveServo(c)});
const s={time:7,root:[0,0,.1,1,0,0,0],joint:Array(14).fill(0),graspPoint:[0,0,.05],payload:[0,0,.05,1,0,0,0],shape:{minZ:.039},jaw:.22,
  contact:{lowerForce:.2,upperForce:.2,feetForce:7,nonMouthSupport:false,nonFootSupport:false}};
assert.equal(await ctl.actuate(r,s),0);assert.deepEqual(Array.from(r.d.ctrl),Array(15).fill(0),'phase boundary keeps actual motor targets until transport plan');
plan={...plan,stage:'transport'};
assert.equal(await ctl.actuate(r,{...s,time:7.02}),0);assert.deepEqual(Array.from(r.d.ctrl.slice(0,14)),plan.ctrl);
await ctl.actuate(r,{...s,time:7.04,shape:{minZ:.041}});assert.equal(ctl.phase,'hold');
plan={...plan,ctrl:Array(14).fill(.23)};
for(let k=0;k<65;k++)await ctl.actuate(r,{...s,time:7.06+k*.02,shape:{minZ:.041}});
assert.equal(ctl.phase,'carried');assert.deepEqual(Array.from(r.d.ctrl.slice(0,14)),plan.ctrl,'carried phase continues feedback');
ctl.release({...s,ctrl:Array.from(r.d.ctrl)});plan={...plan,ctrl:Array(14).fill(.9)};
assert.equal(await ctl.actuate(r,{...s,time:8.4}),.32);assert.deepEqual(Array.from(r.d.ctrl.slice(0,14)),Array(14).fill(.23),'release retains last actual targets');
const failed=Object.assign(new GroundPickController(c),{phase:'lift',reachServo:new HouseholdPredictiveServo(c)});
plan={...plan,blocked:true};const previous=Array.from(r.d.ctrl);
assert.equal(await failed.actuate(r,s),.32);assert.equal(failed.phase,'failed');assert.deepEqual(Array.from(r.d.ctrl),previous,'blocked plan does not partly apply');
console.log('PASS: predictive lift/hold/carried use only motor targets; phase boundaries, blocked plans and natural-release commands remain explicit');
