import assert from 'node:assert/strict';
import {closestPenBite} from '../src/game/household-grasp-target.js';
const close=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-12));
close(closestPenBite([1,2,3,1,0,0,0],[1.02,2.03,3.01],.14),[1,2.03,3]);
close(closestPenBite([1,2,3,1,0,0,0],[1,3,3],.14),[1,2.05,3]);
close(closestPenBite([1,2,3,Math.SQRT1_2,0,0,Math.SQRT1_2],[1.03,2.01,3],.14),[1.03,2,3]);
close(closestPenBite([1,2,3,1,0,0,0],[1,3,3],.02),[1,2,3]);
assert.throws(()=>closestPenBite([1,2,3,1,0,0,0],[1,2,NaN],.14));
console.log('Pen shaft target: projection, endpoint margin, rotation, short object and invalid input pass');
