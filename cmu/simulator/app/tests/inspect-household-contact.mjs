// Offline diagnosis only: reconstructed coordinates never count as a rollout.
import fs from 'node:fs';
import {wholeBodyRuntime} from './whole-body-runtime.mjs';
import {dom} from './physics-fixture.mjs';
import {makeHouseholdPickXml} from '../src/game/household-pick-scene.js';
const trial=JSON.parse(fs.readFileSync(process.argv[2])),s=trial.rows.find(s=>s.time>=Number(process.argv[3]??5.3));
const r=await wholeBodyRuntime({mutate:x=>makeHouseholdPickXml(x,dom,trial.config)});
r.d.qpos.set(s.root);r.d.qpos.set(s.payload,r.p);for(let i=0;i<14;i++)r.d.qpos[r.qa[i]]=s.joint[i];r.d.qpos[r.jaw]=s.jaw;r.d.ctrl.set(s.ctrl);r.mj.mj_forward(r.m,r.d);
const contacts=r.d.contact,buf=new r.mj.DoubleBuffer(6),out=[];
for(let i=0;i<r.d.ncon;i++){const c=contacts.get(i);const names=[c.geom1,c.geom2].map(g=>r.mj.mj_id2name(r.m,r.mj.mjtObj.mjOBJ_GEOM.value,g));if(names.includes('payload_geom')){r.mj.mj_contactForce(r.m,r.d,i,buf);out.push({names,pos:Array.from(c.pos),dist:c.dist,force:Array.from(buf.GetView())});}c.delete();}contacts.delete();buf.delete();console.log(JSON.stringify({payload:s.payload,contacts:out}));await r.dispose();
