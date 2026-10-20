import assert from 'node:assert/strict';
import {householdHeading} from '../src/game/household-grasp-target.js';
const multiply=([a,b,c,d],[w,x,y,z])=>[a*w-b*x-c*y-d*z,a*x+b*w+c*z-d*y,a*y-b*z+c*w+d*x,a*z+b*y-c*x+d*w];
const wrap=x=>Math.atan2(Math.sin(x),Math.cos(x));
for(const yaw of [-2.8,-1.2,0,.4,1.5,2.9])for(let i=-80;i<=80;i++){
  const roll=i*Math.PI/40;
  const q=multiply([Math.cos(yaw/2),0,0,Math.sin(yaw/2)],[Math.cos(roll/2),0,Math.sin(roll/2),0]);
  for(const sign of [-1,1]){
    const h=householdHeading([0,0,0,...q.map(v=>v*sign)],'pen',yaw+.1);
    assert.ok(Math.abs(wrap(h-yaw))<1e-12,'shaft heading must ignore full axial rotations and quaternion sign');
  }
}
assert.ok(Math.abs(householdHeading([0,0,0,0,0,0,1],'pen',0))<1e-12,'opposite shaft directions are equivalent');
assert.equal(householdHeading([0,0,0,Math.SQRT1_2,Math.SQRT1_2,0,0],'pen',.7),.7,'vertical shaft has no horizontal heading');
assert.ok(Math.abs(householdHeading([0,0,0,Math.SQRT1_2,0,0,Math.SQRT1_2],'sock')-Math.PI/2)<1e-12);
assert.throws(()=>householdHeading([0,0,0,0,0,0,0],'pen'));
console.log('PASS: pen shaft heading is invariant to axial roll, sign and reversed endpoints');
