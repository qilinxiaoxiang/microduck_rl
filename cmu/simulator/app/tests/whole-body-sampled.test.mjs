import assert from "node:assert/strict";
import fs from "node:fs";
import {execFile} from "node:child_process";
import {promisify} from "node:util";
const execute=promisify(execFile),seed=Number(process.argv[2]??92604),directory=`test-results/whole-body/sampled-${seed}`;
fs.mkdirSync(directory,{recursive:true});
let bits=seed;
const uniform=()=>{bits^=bits<<13;bits^=bits>>>17;bits^=bits<<5;return(bits>>>0)/4294967296;};
const cases=Array.from({length:6},(_,i)=>({label:`sample-${i+1}`,x:-.48+(uniform()*2-1)*.0048,
  y:(uniform()*2-1)*.0018,mass:.0015+uniform()*.0033,friction:.65+uniform()*.2}));
const results=[],failures=[];let next=0;
async function worker(){while(next<cases.length){const c=cases[next++],output=`${directory}/${c.label}.json`;
  await execute(process.execPath,["tests/probe-table-pick.mjs",JSON.stringify({...c,output})],{maxBuffer:4*1024*1024});
  const r=JSON.parse(fs.readFileSync(output));const result={config:c,acceptance:r.acceptance,summary:r.summary,artifact:output};
  results.push(result);console.log(JSON.stringify(result));if(!r.acceptance.accepted)failures.push(result);
}}
await Promise.all([worker(),worker()]);results.sort((a,b)=>a.config.label.localeCompare(b.config.label));
fs.writeFileSync(`${directory}/validation.json`,JSON.stringify({seed,generatedAt:new Date().toISOString(),selection:seed===92604?"regression of six combinations that exposed the approach deadband; original failed artifacts remain in held-out/":"six seeded interior parameter combinations; training/test use documented in validation report",results,failures},null,2)+"\n");
assert.deepEqual(failures,[]);console.log(`PASS: six parameter combinations, seed ${seed}`);
