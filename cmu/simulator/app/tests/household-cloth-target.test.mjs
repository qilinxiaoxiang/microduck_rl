import assert from 'node:assert/strict';
import {rotateVector,selectRaisedClothNode} from '../src/game/household-cloth-target.js';
import {HouseholdPickController} from '../src/game/household-pick-controller.js';
const p=[.01,-.02,.003];assert.deepEqual(rotateVector([1,0,0,0],p),p);assert.deepEqual(rotateVector([-1,0,0,0],p),p);
const s={time:1,root:[-.08,0,.1,1,0,0,0],payload:[0,0,.003,1,0,0,0],headQuat:[-1,0,0,0],graspPoint:[0,.04,.1],clothPoints:[[0,0,.003],[.017,.017,.006],[.017,-.017,.006],[.1,-.01,.007]]};
const original=JSON.stringify(s);assert.equal(selectRaisedClothNode(s),2);assert.equal(JSON.stringify(s),original);
assert.equal(selectRaisedClothNode({...s,clothPoints:[[0,0,.001]]}),null);
assert.throws(()=>selectRaisedClothNode({...s,clothPoints:[[0,0,NaN]]}));
assert.throws(()=>new HouseholdPickController({kind:'pen',raisedCloth:true,stanceFeedback:true}));
assert.equal(new HouseholdPickController({kind:'sock',raisedCloth:true,stanceFeedback:true,dynamicsReach:true}).begin(s),true);
const ctl=new HouseholdPickController({kind:'sock',raisedCloth:true,stanceFeedback:true});
assert.equal(ctl.begin(s),true);assert.equal(ctl.graspNode,2);assert.deepEqual(ctl.target,s.clothPoints[2]);
const moved={...s,clothPoints:s.clothPoints.map(v=>v.slice())};moved.clothPoints[2]=[.02,-.016,.02];moved.clothPoints[1]=[0,-.01,.03];
ctl.phase='reach';ctl.start=0;ctl.inner={phase:'reach',hold:0,actuate:async(r,seen)=>{
  assert.deepEqual(seen.taskGraspPoint,moved.clothPoints[2],'track the selected material node, not a newly highest vertex');
  assert.deepEqual(seen.payload,s.payload,'physical acceptance keeps its original material-point observation');return 0;
}};
assert.equal(await ctl.actuate({},moved),0);
console.log('PASS: raised-fold selection uses measured geometry, treats q and -q identically, and tracks one free material node without changing payload evidence');
