import assert from 'node:assert/strict';
import {wholeBodyRuntime} from './whole-body-runtime.mjs';
import {dom} from './physics-fixture.mjs';
import {householdConfig,makeHouseholdPickXml} from '../src/game/household-pick-scene.js';
for(const kind of ['pen','sock']){
 const c=householdConfig({kind}),r=await wholeBodyRuntime({mutate:x=>makeHouseholdPickXml(x,dom,c)});
 assert.equal(r.modelAudit.freePayload,true);assert.equal(r.modelAudit.freeBase,true);assert.equal(r.modelAudit.equalities,0);assert.equal(r.modelAudit.mocapBodies,0);
 assert.equal(r.modelAudit.floorHeight,0);assert.equal(r.modelAudit.pedestalPresent,false);assert.ok(Math.abs(r.modelAudit.payloadMass-c.mass)<1e-12);
 if(kind==='sock'){assert.equal(r.modelAudit.flexVertices,42);assert.equal(r.modelAudit.flexAllNodesFree,true);assert.equal(r.m.flex_dim[r.payloadFlexId],2);}
 r.reset([c.x,c.y,c.z+.0005,1,0,0,0]);const s=r.snapshot({});assert.ok(Number.isFinite(s.shape.minZ));assert.ok(Math.abs(s.payload[0]-c.x)<1e-8);
 r.step(.32);assert.equal(r.summary.finite,true);assert.equal(r.summary.maxExternal,0);
 const syncState=Array.from(r.d.qpos),syncSummary={...r.summary};
 r.reset([c.x,c.y,c.z+.0005,1,0,0,0]);await r.stepAsync(.32);
 assert.deepEqual(Array.from(r.d.qpos),syncState);assert.deepEqual(r.summary,syncSummary);
 r.reset([c.x+.05,c.y+.03,c.z+.0005,Math.SQRT1_2,0,0,Math.SQRT1_2]);const rotated=r.snapshot({}).payload;assert.ok(Math.abs(rotated[0]-c.x-.05)<1e-8);
 assert.ok(Math.abs(rotated[3]-Math.SQRT1_2)<1e-8);assert.ok(Math.abs(rotated[6]-Math.SQRT1_2)<1e-8);
 r.reset();const root=r.snapshot({}).root;r.reset(undefined,{rootOffset:[-.03,.015,.1]});const shifted=r.snapshot({}).root;
 assert.ok(Math.abs(shifted[0]-root[0]+.03)<1e-10);assert.ok(Math.abs(shifted[1]-root[1]-.015)<1e-10);assert.ok(Math.abs(shifted[6]-Math.sin(.05))<1e-10);
 assert.throws(()=>r.reset(undefined,{rootOffset:[NaN,0]}));
 await r.dispose();
}
console.log('PASS: free pen and 42-node unpinned cloth, exact mass, collision floor, rotated reset, finite zero-force physics');
