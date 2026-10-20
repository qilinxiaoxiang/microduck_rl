import assert from 'node:assert/strict';
import {timedForward} from '../src/household-drive.js';
assert.deepEqual(timedForward({phase:'manual',paused:false,time:3},2),{time:3,type:'forward-two-seconds',until:5});
assert.equal(timedForward({phase:'manual',paused:false,time:3},.5).until,3.5);
assert.equal(timedForward({phase:'reach',paused:false,time:3},2),null);
assert.equal(timedForward({phase:'manual',paused:true,time:3},2),null);
for(const duration of [0,-1,10,NaN])assert.throws(()=>timedForward({phase:'manual',paused:false,time:3},duration));
assert.throws(()=>timedForward({phase:'manual',paused:false,time:NaN},2));
console.log('PASS: bounded timed walking inputs, paused/nonmanual guards and simulation-time deadlines');
