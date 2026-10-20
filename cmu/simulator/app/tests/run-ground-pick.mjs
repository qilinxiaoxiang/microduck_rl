import fs from "node:fs";
import {createHash} from "node:crypto";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom} from "./physics-fixture.mjs";
import {GROUND_PICK_DEFAULTS,makeGroundPickXml} from "../src/game/ground-pick-scene.js";
import {GroundPickController} from "../src/game/ground-pick-controller.js";
import {evaluateGroundTrial} from "../src/game/ground-pick-acceptance.js";
import {JawActionFilter,jawObservation,jawActor,JAW_RL_PHASES} from "../src/game/jaw-rl.js";
const c={...GROUND_PICK_DEFAULTS,...JSON.parse(process.argv[2]??"{}")},r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom,c)});
const hash=x=>createHash("sha256").update(x).digest("hex"),sourcePaths=["app/tests/run-ground-pick.mjs","app/src/game/ground-pick-controller.js","app/src/game/ground-pick-scene.js","app/src/game/ground-pick-acceptance.js","app/src/game/whole-body-core.js","app/src/game/contact-observer.js","app/src/game/jaw-rl.js","app/src/game/physics-model.js","app/src/game/beak-colliders.js"];
const sources=Object.fromEntries(sourcePaths.map(p=>[p,hash(fs.readFileSync(p))]));
const provenance={engine:"MuJoCo 3.11.0",inference:"onnxruntime-node 1.27.0 cpu",sources,modelSha256:hash(r.xml),jawPolicySha256:c.actor?hash(fs.readFileSync(c.actor)):null,policySha256:hash(fs.readFileSync("app/public/policies/BEST_alpha_walking.onnx")),groundPolicySha256:hash(fs.readFileSync("app/public/policies/alpha_ground_pick.onnx"))};
const ctl=new GroundPickController(c),filter=new JawActionFilter(),rows=[],actor=c.actor?JSON.parse(fs.readFileSync(c.actor)):null;
r.reset([c.x,c.y,c.z+.0005,1,0,0,0]);let contact={},action=1,learnedTicks=0;
for(let k=0;k<(c.ticks??850);k++){
  const before=r.snapshot(contact),learning=actor&&JAW_RL_PHASES.includes(ctl.phase);
  if(learning&&learnedTicks%5===0)action=jawActor(actor,Array.from(new Float32Array(jawObservation(before,ctl,filter.target)))).action;
  const target=await ctl.actuate(r,before),jaw=filter.update(learning?action:target===0?0:1);if(learning)learnedTicks++;
  contact=r.step(jaw);const s=r.snapshot(contact);rows.push({...s,phase:ctl.phase,jawTarget:jaw,learnedAction:learning?action:null});
  if(contact.otherGeoms)rows.at(-1).otherGeomNames=Object.fromEntries(Object.entries(contact.otherGeoms).map(([g,f])=>[r.mj.mj_id2name(r.m,r.mj.mjtObj.mjOBJ_GEOM.value,Number(g)),f]));
  if(r.summary.solverReset||!r.summary.finite||["done","failed"].includes(ctl.phase))break;
}
const result={config:c,phase:ctl.phase,events:ctl.events,modelAudit:r.modelAudit,summary:r.summary,provenance,rows};
provenance.sourcesUnchanged=sourcePaths.every(p=>sources[p]===hash(fs.readFileSync(p)));
result.acceptance=evaluateGroundTrial(result);
if(!provenance.sourcesUnchanged){result.acceptance.accepted=false;result.acceptance.failures.push("source changed during trial");}
if(c.output)fs.writeFileSync(c.output,JSON.stringify(result)+"\n");
const both=rows.filter(s=>s.contact.lowerForce>.002&&s.contact.upperForce>.002);
console.log(JSON.stringify({phase:ctl.phase,events:ctl.events,summary:r.summary,acceptance:result.acceptance,opposingFrames:both.length,...(c.verbose?{firstOpposing:both[0],last:rows.at(-1)}:{})}));await r.dispose();
