// P0: same held-out environments, fresh process per controller, immutable
// source hashes in every trial. No training or parameter search happens here.
import fs from "node:fs";
import path from "node:path";
import {spawn} from "node:child_process";
import {createHash} from "node:crypto";
import {fileURLToPath} from "node:url";
const root=fileURLToPath(new URL("../",import.meta.url)),dir=path.resolve(root,process.argv[3]??"rl/runs/ground-validation-clockfixed-20260904");
if(fs.existsSync(dir)&&fs.readdirSync(dir).length)throw Error(`Use a new evidence directory; refusing to overwrite ${dir}`);
fs.mkdirSync(dir,{recursive:true});
const actors={baseline:null,learned:"app/public/policies/jaw-ppo.json",untrained:"app/public/policies/jaw-ppo-initial.json"};
const variants=[{name:"nominal",config:{}},{name:"x_back_1p5mm",config:{x:-.5335}},{name:"y_right_1p5mm",config:{y:.0075}},{name:"mass_plus_20pct",config:{mass:.0006}}];
const jobs=variants.flatMap(v=>Object.entries(actors).map(([mode,actor])=>({name:`${v.name}-${mode}`,mode,variant:v.name,config:{...v.config,...(actor?{actor}:{})}})));
jobs.push({name:"stiffer-learned",mode:"learned",variant:"young_plus_20pct",config:{young:6000,actor:actors.learned}},{name:"coarse-learned",mode:"learned",variant:"27_nodes",config:{meshCount:3,actor:actors.learned}});
jobs.push({name:"half-step-learned",mode:"learned",variant:"half_timestep",config:{dt:.00003125,actor:actors.learned}});
const selected=process.argv[2]?jobs.filter(j=>new RegExp(process.argv[2]).test(j.name)):jobs;
const results=[];let index=0;
const children=new Set();
for(const signal of ["SIGINT","SIGTERM"])process.once(signal,()=>{
 for(const child of children)child.kill("SIGTERM");
 fs.writeFileSync(path.join(dir,"cancelled.json"),JSON.stringify({signal,completed:results.length,inFlight:children.size,time:new Date().toISOString()})+"\n");
 process.exit(130);
});
const digest=file=>createHash("sha256").update(fs.readFileSync(file)).digest("hex");
async function worker(){while(index<selected.length){const job=selected[index++],output=path.join(dir,job.name+".json"),began=Date.now();
 console.log(`START ${job.name}`);
 const code=await new Promise(resolve=>{const p=spawn(process.execPath,["app/tests/run-ground-pick.mjs",JSON.stringify({...job.config,output})],{cwd:root,stdio:["ignore","pipe","pipe"]});children.add(p);let log="";p.stdout.on("data",b=>log+=b);p.stderr.on("data",b=>log+=b);p.on("error",e=>log+=String(e));p.on("close",code=>{children.delete(p);fs.writeFileSync(path.join(dir,job.name+".log"),log);resolve(code);});});
 const trial=code===0&&fs.existsSync(output)?JSON.parse(fs.readFileSync(output)):null;
 results.push({...job,exitCode:code,seconds:(Date.now()-began)/1000,evidence:path.relative(root,output),evidenceSha256:trial?digest(output):null,acceptance:trial?.acceptance,provenance:trial?.provenance});
 console.log(`END ${job.name}: ${trial?.acceptance.accepted?"PASS":code?"PROCESS FAILURE":"REJECTED"} ${JSON.stringify(trial?.acceptance??{})}`);
 fs.writeFileSync(path.join(dir,"summary.json"),JSON.stringify({variants,results},null,2)+"\n");
}}
await Promise.all(Array.from({length:3},worker));
console.log(JSON.stringify({completed:results.length,passed:results.filter(r=>r.acceptance?.accepted).length,summary:path.join(dir,"summary.json")}));
