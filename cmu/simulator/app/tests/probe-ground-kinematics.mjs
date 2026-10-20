// Offline pose/Jacobian inspection; never a dynamic acceptance trial.
import fs from "node:fs";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom,id} from "./physics-fixture.mjs";
import {makeGroundPickXml} from "../src/game/ground-pick-scene.js";
const r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom,{softBall:false})});
const rows=JSON.parse(fs.readFileSync(new URL("../../rl/runs/ground-development/upstream-open.json",import.meta.url))).rows;
const s=rows.reduce((a,b)=>a.graspPoint[2]<b.graspPoint[2]?a:b);
r.d.qpos.set(s.root,0);for(let j=0;j<14;j++)r.d.qpos[r.qa[j]]=s.joint[j];r.mj.mj_forward(r.m,r.d);
const feet=["left_foot_collision","right_foot_collision"].map(n=>id(r.m,"GEOM",n));
const pos=()=>feet.map(g=>Array.from(r.d.geom_xpos.subarray(g*3,g*3+3)));
const base=pos(),result=[];
for(const j of [2,3,4,11,12,13]){
  const q=r.d.qpos[r.qa[j]];r.d.qpos[r.qa[j]]=q+.0001;r.mj.mj_forward(r.m,r.d);
  const joint=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]);
  result.push({j,axis:Array.from(r.d.xaxis.subarray(joint*3,joint*3+3)),derivative:pos().map((p,i)=>p.map((v,k)=>(v-base[i][k])/.0001))});
  r.d.qpos[r.qa[j]]=q;
}
console.log(JSON.stringify({root:s.root,joint:s.joint,feet:base,derivatives:result},null,2));
await r.dispose();
