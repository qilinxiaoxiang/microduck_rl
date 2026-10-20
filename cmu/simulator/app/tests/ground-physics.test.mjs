import assert from "node:assert/strict";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom} from "./physics-fixture.mjs";
import {makeGroundPickXml,GROUND_PICK_DEFAULTS as c} from "../src/game/ground-pick-scene.js";
const r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom)});
const reset=()=>r.reset([c.x,c.y,c.z+.0005,1,0,0,0]);
assert.equal(r.modelAudit.flexVertices,99);assert.equal(r.modelAudit.flexAllNodesFree,true);
assert.equal(r.modelAudit.pedestalPresent,false);assert.equal(r.modelAudit.terrainPresent,false);
assert.equal(r.modelAudit.equalities,0);assert.equal(r.modelAudit.mocapBodies,0);
assert.ok(Math.abs(r.modelAudit.payloadMass-.0005)<1e-12);
reset();r.step(.32);assert.equal(r.summary.finite,true);assert.equal(r.summary.maxExternal,0);
// Adversarial unit check, not a pickup trial: force an invalid input and
// confirm that MuJoCo's subsequent finite reset cannot pass our audit.
r.d.qvel[0]=NaN;r.step(.32);
assert.equal(r.summary.finite,false);assert.equal(r.summary.solverReset,true);
reset();assert.equal(r.summary.solverReset,false);assert.equal(r.summary.resetAt,null);
assert.deepEqual(r.summary.warningCounts,{});r.d.qfrc_applied[0]=.001;r.step(.32);
assert.equal(r.summary.maxExternal,.001);reset();r.step(.32);assert.equal(r.summary.maxExternal,0);
console.log("PASS: genuine free 99-node ball, exact mass, no platform/attachment, solver-reset and applied-force rejection");
await r.dispose();
