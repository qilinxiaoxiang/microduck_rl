import assert from "node:assert/strict";
import { paceControlLoop } from "../src/game/control-loop-clock.js";

for (const [current, expectedDelay, expectedNext] of [[105,15,120],[120,0,120],[160,0,160]]) {
  const sleeps=[];
  const next=await paceControlLoop(100,20,{now:()=>current,sleep:async ms=>sleeps.push(ms)});
  assert.deepEqual(sleeps,[expectedDelay],"even an overdue loop must yield a browser task");
  assert.equal(next,expectedNext,"do not accumulate catch-up debt");
}
let inputHandled=false;
setTimeout(()=>{inputHandled=true;},0);
await paceControlLoop(0,0);
assert.equal(inputHandled,true,"an overdue physics tick must allow pending UI work");
console.log("PASS control-loop pacing: on time, late, and browser task liveness");
