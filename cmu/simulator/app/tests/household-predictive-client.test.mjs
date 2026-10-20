import assert from 'node:assert/strict';
import loadMujoco from '@mujoco/mujoco';
import {createPredictiveClient} from '../src/household-predictive-client.js';
const mj=await loadMujoco(),vfs=new mj.MjVFS();
const xml='<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>';
const m=mj.MjModel.from_xml_string(xml,vfs),d=new mj.MjData(m),r={mj,m,d};
const originalFetch=globalThis.fetch;let requests=[],mutate=false;
globalThis.fetch=async(url,options)=>{
  const input=JSON.parse(options.body);requests.push(input);
  if(new URL(url).pathname==='/init')return {ok:true,json:async()=>({session:'test-only'})};
  // A zero-filled detached Float64Array would lose both the z position and
  // the quaternion. Check the real WASM state's known nonzero coordinates.
  assert.equal(input.state[3],1);assert.equal(input.state[4],1);
  if(mutate)d.qpos[0]+=.1; // Isolated test fixture, never a runtime action.
  return {ok:true,json:async()=>({sequence:input.sequence,time:d.time,plan:{stage:input.stage,ctrl:Array(14).fill(.01),audit:{sourceStateUnchanged:true}}})};
};
try{
  await assert.rejects(()=>createPredictiveClient(r,xml,{},{endpoint:'https://example.com'}));
  const client=await createPredictiveClient(r,xml,{dynamicsLift:true}),before=Array.from(d.qpos);
  const plan=await client.update();assert.equal(plan.ctrl.length,14);
  assert.deepEqual(Array.from(d.qpos),before);assert.equal(client.busy,false);
  assert.equal(requests.length,2);await client.update();assert.equal(requests.length,2,'reuse plan only inside the 0.2 s interval');
  assert.equal(r.predictedReachPlan(),plan);
  await client.update('transport');assert.equal(requests.length,3,'a new stage bypasses the time cache');
  assert.equal(client.plan.stage,'transport');
  await assert.rejects(()=>client.update('teleport'),/阶段/);
  await client.reset();mutate=true;
  await assert.rejects(()=>client.update(),/状态变化/);assert.equal(client.busy,false);
  console.log('PASS: actual WASM state is read without mutation, cached plans expire, stale results are rejected');
}finally{globalThis.fetch=originalFetch;d.delete();m.delete();vfs.delete();}
