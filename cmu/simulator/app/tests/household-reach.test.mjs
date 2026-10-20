import assert from 'node:assert/strict';
import {wholeBodyRuntime} from './whole-body-runtime.mjs';
import {dom} from './physics-fixture.mjs';
import {makeHouseholdPickXml} from '../src/game/household-pick-scene.js';
import {HouseholdReachServo} from '../src/game/household-reach-servo.js';
import {HouseholdStanceServo,solveWithLowerBound} from '../src/game/household-stance-servo.js';
const identity=[[1,0,0],[0,1,0],[0,0,1]];
assert.deepEqual(solveWithLowerBound(identity,[.1,.2,.3],[0,0,1],0),[.1,.2,.3],'upward motion must not be attracted back to the floor');
assert.deepEqual(solveWithLowerBound(identity,[.1,.2,-.3],[0,0,1],0),[.1,.2,0],'only unsafe downward motion is constrained');
assert.deepEqual(solveWithLowerBound(identity,[.1,.2,-.3],[0,0,1],-.1),[.1,.2,-.1],'controlled descent remains possible above the safety boundary');
const r=await wholeBodyRuntime({mutate:x=>makeHouseholdPickXml(x,dom,{kind:'pen'})});
const local=[-.017,-.05,-.072],point=()=>{const R=r.d.xmat.subarray(r.head*9,r.head*9+9),p=r.d.xpos.subarray(r.head*3,r.head*3+3);return Array.from(p,(v,i)=>v+local.reduce((sum,x,j)=>sum+R[i*3+j]*x,0));};
// Finite-difference FK is an isolated kinematics test, never a physical trial.
for(const pose of [[-.7,-.3,.2,.1],[-1.48,-.7,-.1,-.05]]){
  for(let k=0;k<4;k++)r.d.qpos[r.qa[5+k]]=pose[k];r.mj.mj_forward(r.m,r.d);
  const p=point(),jac=r.pointJacobian(p),eps=1e-6;
  const stance=r.stanceKinematics(p);
  for(let j=0;j<14;j++){
    r.d.qpos[r.qa[j]]+=eps;r.mj.mj_forward(r.m,r.d);
    const shifted=r.stanceKinematics(point());
    for(let axis=0;axis<3;axis++){
      assert.ok(Math.abs((point()[axis]-p[axis])/eps-stance.grip.position[axis][6+j])<1e-6);
      assert.ok(Math.abs((shifted.com.point[axis]-stance.com.point[axis])/eps-stance.com.position[axis][6+j])<1e-6);
      for(let foot=0;foot<2;foot++)assert.ok(Math.abs((shifted.feet[foot].point[axis]-stance.feet[foot].point[axis])/eps-stance.feet[foot].position[axis][6+j])<1e-6);
    }
    r.d.qpos[r.qa[j]]-=eps;r.mj.mj_forward(r.m,r.d);
  }
  const clearance=r.beakClearance(),floor=r.mj.mj_name2id(r.m,r.mj.mjtObj.mjOBJ_GEOM.value,'floor'),segment=new r.mj.DoubleBuffer(6);
  const distance=r.mj.mj_geomDistance(r.m,r.d,floor,clearance.geom,1,segment);segment.delete();
  assert.ok(Math.abs(distance-clearance.height)<1e-6);
  for(let k=0;k<4;k++){
    r.d.qpos[r.qa[5+k]]+=eps;r.mj.mj_forward(r.m,r.d);const shifted=point();
    for(let axis=0;axis<3;axis++)assert.ok(Math.abs((shifted[axis]-p[axis])/eps-jac[k][axis])<1e-6);
    r.d.qpos[r.qa[5+k]]-=eps;r.mj.mj_forward(r.m,r.d);
  }
  const before=Array.from(r.d.qpos),forces=Array.from(r.d.qfrc_applied),external=Array.from(r.d.xfrc_applied);
  const servo=new HouseholdReachServo({poseFeedback:true});servo.apply(r,r.snapshot({}));
  assert.deepEqual(Array.from(r.d.qpos),before);assert.deepEqual(Array.from(r.d.qfrc_applied),forces);assert.deepEqual(Array.from(r.d.xfrc_applied),external);
  assert.ok(Array.from(r.d.ctrl).every(Number.isFinite));assert.ok(servo.offset.every(x=>Math.abs(x)<=.003000000001));
  const full=new HouseholdStanceServo({});full.apply(r,r.snapshot({}));
  assert.deepEqual(Array.from(r.d.qpos),before);assert.deepEqual(Array.from(r.d.qfrc_applied),forces);assert.deepEqual(Array.from(r.d.xfrc_applied),external);
  assert.ok(Array.from(r.d.ctrl).every(Number.isFinite));assert.ok(full.offset.every(x=>Math.abs(x)<=.006000000001));
  const supported=new HouseholdStanceServo({supportGravityGain:1,headGravityFeedback:true});supported.apply(r,r.snapshot({}));supported.hold(r);
  assert.ok(supported.support.baseResidual.every(x=>Math.abs(x)<1e-5));
  assert.deepEqual(Array.from(r.d.qpos),before);assert.deepEqual(Array.from(r.d.qfrc_applied),forces);assert.deepEqual(Array.from(r.d.xfrc_applied),external);
  assert.ok(Array.from(r.d.ctrl).every(Number.isFinite));
}
await r.dispose();console.log('PASS: analytical point Jacobian vs finite differences; feedback writes bounded motor increments only');
