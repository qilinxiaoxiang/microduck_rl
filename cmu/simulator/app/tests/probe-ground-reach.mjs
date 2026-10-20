// Kinematic feasibility only: these pose writes never count as pickup evidence.
import fs from "node:fs";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom} from "./physics-fixture.mjs";
import {makeGroundPickXml} from "../src/game/ground-pick-scene.js";
const c=JSON.parse(process.argv[2]??"{}"),r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom,c)});
const rows=[];
for(let n=-1.5;n<=1.05;n+=.15)for(let h=-1.5;h<=1.5;h+=.15){
  r.d.qpos[r.qa[5]]=n;r.d.qpos[r.qa[6]]=h;r.d.qpos[r.jaw]=.32;r.mj.mj_forward(r.m,r.d);
  const s=r.snapshot({}),hr=r.d.xmat.subarray(r.head*9,r.head*9+9);
  rows.push({neck:n,head:h,point:s.graspPoint,tip:s.sites.mouth_tip,
    axis:[-hr[2],-hr[5],-hr[8]],headHeight:r.d.xpos[r.head*3+2]});
}
rows.sort((a,b)=>a.point[2]-b.point[2]);
console.log(JSON.stringify(rows.slice(0,15),null,2));
if(c.output)fs.writeFileSync(c.output,JSON.stringify(rows)+"\n");
await r.dispose();
