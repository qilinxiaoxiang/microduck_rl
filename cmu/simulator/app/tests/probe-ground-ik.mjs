// OFFLINE feasibility only: writes a separate kinematic planning state, never
// evidence of a physical pickup. Outputs motor-pose candidates for dynamic tests.
import fs from "node:fs";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom,id} from "./physics-fixture.mjs";
import {makeGroundPickXml} from "../src/game/ground-pick-scene.js";
const c=JSON.parse(process.argv[2]??"{}");
const r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom,{softBall:false})});
const rows=JSON.parse(fs.readFileSync("rl/runs/ground-development/upstream-open.json")).rows;
const s=rows.find(x=>x.time>=2.2);
r.d.qpos.set(s.root,0);for(let j=0;j<14;j++)r.d.qpos[r.qa[j]]=s.joint[j];r.mj.mj_forward(r.m,r.d);
const feet=["left_foot_collision","right_foot_collision"].map(n=>id(r.m,"GEOM",n));
const foot=g=>{const p=r.d.geom_xpos.subarray(g*3,g*3+3),R=r.d.geom_xmat.subarray(g*9,g*9+9);return [p[0],p[2],Math.atan2(R[2],R[8])*.05];};
const targets=feet.map(foot),com0=r.d.subtree_com[r.trunk*3];
const start=Array.from(r.d.qpos),result=[];
for(const pitch of c.pitches??[.3,.4,.5,.6,.7])for(const height of c.heights??[.07,.075,.08,.085]){
 r.d.qpos.set(start);r.d.qpos[2]=height;r.d.qpos.set([Math.cos(pitch/2),0,Math.sin(pitch/2),0],3);
 r.d.qpos[r.qa[5]]=-1.50;r.d.qpos[r.qa[6]]=-1.50-pitch+Math.PI/2+(c.worldAngle??0);
 for(let iter=0;iter<60;iter++){
   r.mj.mj_forward(r.m,r.d);
   if(iter%5===0)r.d.qpos[0]+=Math.max(-.002,Math.min(.002,com0-r.d.subtree_com[r.trunk*3]));
   for(let side=0;side<2;side++){
    r.mj.mj_forward(r.m,r.d);const base=foot(feet[side]),err=targets[side].map((x,k)=>x-base[k]),J=[];
    for(const j of side===0?[2,3,4]:[11,12,13]){const old=r.d.qpos[r.qa[j]];r.d.qpos[r.qa[j]]=old+.0001;r.mj.mj_forward(r.m,r.d);J.push(foot(feet[side]).map((x,k)=>(x-base[k])/.0001));r.d.qpos[r.qa[j]]=old;}
    // Damped Jacobian transpose, small step robust near joint limits.
    for(let k=0;k<3;k++){const j=(side===0?[2,3,4]:[11,12,13])[k],dq=J[k].reduce((a,x,i)=>a+x*err[i],0)/(J[k].reduce((a,x)=>a+x*x,0)+.001);r.d.qpos[r.qa[j]]=Math.max(-1.53,Math.min(1.53,r.d.qpos[r.qa[j]]+.5*dq));}
   }
 }
 r.mj.mj_forward(r.m,r.d);const snap=r.snapshot({}),errors=feet.map((g,i)=>foot(g).map((v,k)=>v-targets[i][k]));
 let nonfoot=false;const contacts=r.d.contact;for(let i=0;i<r.d.ncon;i++){const co=contacts.get(i);if(co.dist<-.0005&&[co.geom1,co.geom2].includes(0)&&![co.geom1,co.geom2].some(g=>feet.includes(g)))nonfoot=true;co.delete();}contacts.delete();
 result.push({pitch,height,root:snap.root,point:snap.graspPoint,sites:snap.sites,joint:snap.joint,errors,com:r.d.subtree_com[r.trunk*3],nonfoot});
}
if(c.output)fs.writeFileSync(c.output,JSON.stringify(result)+"\n");
console.log(JSON.stringify(result));await r.dispose();
