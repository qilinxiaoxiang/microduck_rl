// Material fixture only, not a robot pickup: gravity drop, motor-driven plate
// compression and withdrawal. Shape evidence comes directly from flex vertices.
import fs from "node:fs";
import {scene,dom,mj} from "./physics-fixture.mjs";
import {makeGroundPickXml,GROUND_PICK_DEFAULTS} from "../src/game/ground-pick-scene.js";
const c={...GROUND_PICK_DEFAULTS,...JSON.parse(process.argv[2]??"{}")};
const full=scene("legs",false,null,x=>makeGroundPickXml(x,dom,c));
const doc=new dom.DOMParser().parseFromString(full.xml,"text/xml");
const body=doc.querySelector('body[name="payload"]');body.setAttribute("pos",`0 0 ${c.radius+.005}`);
const xml=`<mujoco><option timestep="${c.dt}" integrator="implicitfast" iterations="80" tolerance="1e-10"/><default>${doc.querySelector('default[class="soft_ball_nodes"]').outerHTML}</default><worldbody><geom name="floor" type="plane" size="1 1 .1" contype="1" conaffinity="32" friction=".8 .0002 .00001" solref=".00025 1" solimp=".99 .999 .001"/>${body.outerHTML}<body name="plate" pos="0 0 .045"><joint name="press" type="slide" axis="0 0 1" damping=".03"/><geom name="plate_geom" type="box" size=".03 .03 .001" mass=".002" contype="1" conaffinity="32" solref=".00025 1" solimp=".99 .999 .001"/></body></worldbody><actuator><position joint="press" kp="10" forcerange="-.1 .1"/></actuator></mujoco>`;
full.dispose();
const m=mj.MjModel.from_xml_string(xml),d=new mj.MjData(m),rows=[];
const jid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT.value,"press"),q=m.jnt_qposadr[jid];
for(let k=0;k<Math.round(4.5/c.dt);k++){
  const t=d.time;
  // Hold, descend slowly by 28 mm, dwell, then withdraw.
  d.ctrl[0]=t<1?0:t<2?-.028*(t-1):t<2.5?-.028:t<3?-.028*(1-(t-2.5)/.5):0;
  mj.mj_step(m,d);
  if(k%Math.round(.01/c.dt)===0){
    const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    for(let i=0;i<m.flex_vertnum[0];i++)for(let j=0;j<3;j++){const v=d.flexvert_xpos[i*3+j];lo[j]=Math.min(lo[j],v);hi[j]=Math.max(hi[j],v);}
    let depth=0,force=0;const contacts=d.contact,buf=new mj.DoubleBuffer(6);
    for(let i=0;i<d.ncon;i++){const contact=contacts.get(i);depth=Math.max(depth,-contact.dist);mj.mj_contactForce(m,d,i,buf);force=Math.max(force,buf.GetView()[0]);contact.delete();}contacts.delete();buf.delete();
    const maxRadialDisplacement=Math.max(...Array.from(d.qpos.subarray(7,q),Math.abs));
    rows.push({time:d.time,dimensions:hi.map((v,j)=>v-lo[j]),maxRadialDisplacement,minZ:lo[2]-.0002,center:Array.from(d.qpos.slice(0,3)),plateZ:.044+d.qpos[q],penetration:depth,maxContactForce:force});
  }
}
const summary={mass:m.body_subtreemass[1],equalities:m.neq,vertices:m.flex_vertnum[0],rest:rows.find(r=>r.time>.95),compressed:rows.filter(r=>r.time>2.3&&r.time<2.5).at(-1),recovered:rows.at(-1),maxPenetration:Math.max(...rows.map(r=>r.penetration)),finite:rows.every(r=>r.dimensions.every(Number.isFinite))};
summary.recoveryFraction=1-summary.recovered.maxRadialDisplacement/summary.compressed.maxRadialDisplacement;
if(c.output)fs.writeFileSync(c.output,JSON.stringify({config:c,summary,rows,xml})+"\n");
console.log(JSON.stringify(summary));d.delete();m.delete();
