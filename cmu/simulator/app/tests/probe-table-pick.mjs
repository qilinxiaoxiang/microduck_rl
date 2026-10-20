import fs from "node:fs";
import crypto from "node:crypto";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom} from "./physics-fixture.mjs";
import {TablePickController} from "../src/game/table-pick-controller.js";
import {HeadReachController} from "../src/game/head-reach-controller.js";
import {evaluateWholeBodyTrial} from "../src/game/whole-body-acceptance.js";
import {makeTablePickXml,TABLE_PICK_DEFAULTS} from "../src/game/table-pick-scene.js";
const config=JSON.parse(process.argv[2]??"{}");
const sourcePaths=["src/game/table-pick-controller.js","src/game/head-reach-controller.js","src/game/whole-body-core.js","src/game/table-pick-scene.js","src/game/whole-body-acceptance.js","tests/probe-table-pick.mjs","tests/whole-body-runtime.mjs","src/game/physics-model.js","src/game/contact-observer.js","src/game/constants.js"];
const provenance=Object.fromEntries(sourcePaths.map(p=>{const source=fs.readFileSync(p,"utf8");return [p,{sha256:crypto.createHash("sha256").update(source).digest("hex"),source}];}));
const options={...TABLE_PICK_DEFAULTS,...config},target=[options.x,options.y,options.z];
const r=await wholeBodyRuntime({mutate:xml=>makeTablePickXml(xml,dom,options)});
provenance.modelSha256=crypto.createHash("sha256").update(r.xml).digest("hex");
provenance.policySha256=crypto.createHash("sha256").update(fs.readFileSync("public/policies/BEST_alpha_walking.onnx")).digest("hex");
r.reset([...target.slice(0,2),target[2]+.0005,1,0,0,0]);
const Controller=options.cartesian?HeadReachController:TablePickController;
const ctl=new Controller({height:target[2],lift:options.lift,biteOffset:options.biteOffset}),rows=[];
let contact={},lastPhase,frozenLegs;
for(let k=0;k<1800;k++) {
  const before=r.snapshot(contact),out=ctl.update(before);
  if(config.headCommands)for(let h=0;h<4;h++)out.command[3+h]=Math.max(-1.1,Math.min(1.1,out.head[h]-(h<2?.3490658503988659:0)));
  if(frozenLegs)r.d.ctrl.set(frozenLegs);
  else {
    await r.policy("walk",out.command);
    if(config.freezeLegs&&out.phase==="align"&&ctl.age>1)frozenLegs=Array.from(r.d.ctrl);
  }
  r.applyTask(out);
  contact=r.step(config.open?.32:out.jaw);
  if(k%25===0||lastPhase!==out.phase||["clamp","lift","hold","release"].includes(out.phase)){const s=r.snapshot(contact);rows.push({t:s.time,phase:out.phase,root:s.root,point:s.graspPoint,object:s.payload,head:s.joint.slice(5,9),headQuat:s.headQuat,jaw:s.jaw,contact});}
  lastPhase=out.phase;
  if(["done","failed"].includes(out.phase))break;
}
const result={config:options,provenance,phase:ctl.phase,events:ctl.events,rows,summary:r.summary,modelAudit:r.modelAudit};
result.acceptance=evaluateWholeBodyTrial(result);
if(config.output)fs.writeFileSync(config.output,JSON.stringify(result,null,2)+"\n");
console.log(JSON.stringify({...result,provenance:undefined,rows:rows.filter((s,i)=>i===0||i===rows.length-1||s.phase!==rows[i-1].phase).map(s=>({...s,root:s.root.slice(0,3),object:s.object.slice(0,3)}))}));
await r.dispose();
