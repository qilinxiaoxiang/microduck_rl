import assert from "node:assert/strict";
import fs from "node:fs";
import {execFile} from "node:child_process";
import {promisify} from "node:util";
const execute=promisify(execFile),directory="test-results/whole-body";
fs.mkdirSync(directory,{recursive:true});
const cases=[
  {label:"nominal"},{label:"x-minus-5mm",x:-.485},{label:"x-plus-5mm",x:-.475},
  {label:"y-minus-2mm",y:-.002},{label:"y-plus-2mm",y:.002},
  {label:"half-mass",mass:.00125},{label:"double-mass",mass:.005},
  {label:"combined",x:-.475,y:.002,mass:.005},
  {label:"lower-friction-heavy",friction:.6,mass:.005},
  {label:"half-step",dt:.0000625},
  {label:"open-negative",open:true,expected:false},
  {label:"weak-jaw-negative",jawTorque:.006,mass:.005,expected:false},
];
const results=[],failures=[];let next=0;
async function worker(){
  while(next<cases.length){
    const config=cases[next++],output=`${directory}/${config.label}.json`,started=Date.now();
    try{
      await execute(process.execPath,["tests/probe-table-pick.mjs",JSON.stringify({...config,output})],{maxBuffer:4*1024*1024});
      const trial=JSON.parse(fs.readFileSync(output,"utf8"));
      const row={label:config.label,expected:config.expected!==false,phase:trial.phase,config:trial.config,
        acceptance:trial.acceptance,summary:trial.summary,modelAudit:trial.modelAudit,
        sources:Object.fromEntries(Object.entries(trial.provenance).map(([p,v])=>[p,v.sha256??v])),
        wallSeconds:(Date.now()-started)/1000,artifact:output};
      results.push(row);console.log(JSON.stringify(row));
      if(row.acceptance.accepted!==row.expected)failures.push({label:row.label,failures:row.acceptance.failures});
    }catch(error){failures.push({label:config.label,error:String(error)});console.error(config.label,String(error));}
  }
}
await Promise.all([worker(),worker()]);
results.sort((a,b)=>cases.findIndex(c=>c.label===a.label)-cases.findIndex(c=>c.label===b.label));
const nominal=results.find(r=>r.label==="nominal"),fine=results.find(r=>r.label==="half-step");
const convergence=nominal&&fine?{heldLiftDifference:Math.abs(nominal.acceptance.minHeldLift-fine.acceptance.minHeldLift),peakLiftDifference:Math.abs(nominal.summary.maxZ-fine.summary.maxZ)}:null;
if(!convergence||convergence.heldLiftDifference>.001||convergence.peakLiftDifference>.001)failures.push({label:"step-convergence",convergence});
fs.writeFileSync(`${directory}/validation.json`,JSON.stringify({generatedAt:new Date().toISOString(),scope:"deterministic free-standing reachable table-edge pickup; not walking, ground pickup, vision or hardware validation",results,convergence,failures},null,2)+"\n");
assert.deepEqual(failures,[],"whole-body acceptance failures; see retained trial records");
console.log(`PASS: ${results.length} whole-body cases including negative controls and half-step convergence`);
