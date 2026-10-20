// P0: read-only promotion review of frozen source/actor evidence. This does not
// control the robot, train a policy, tune parameters or erase failed trials.
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {evaluateGroundTrial} from "../app/src/game/ground-pick-acceptance.js";
import {stableGroundLanding} from "./ground_review_helpers.mjs";
const root=path.resolve(import.meta.dirname,".."),read=p=>JSON.parse(fs.readFileSync(path.resolve(root,p))),hash=p=>createHash("sha256").update(fs.readFileSync(path.resolve(root,p))).digest("hex");
const nominalDir=process.argv[2]??"rl/runs/ground-validation-clockfixed-20260904",perturbDir=process.argv[3]??"rl/runs/ground-perturbations-clockfixed-20260904",browserPath=process.argv[4]??"rl/runs/ground-browser-complete.json",output=process.argv[5]??"rl/results/ground-validation.json";
const refinementDir=process.argv[6]??"rl/runs";
const failures=[],check=(ok,label)=>{if(!ok)failures.push(label);};
// Require 0.3 s of continuous stable ground contact AFTER the real release,
// independently of the controller's own phase/stability counters.
function review(file,mode,name){const t=read(file),a=evaluateGroundTrial(t),landingSeconds=stableGroundLanding(t.rows);
 for(const [p,h] of Object.entries(t.provenance.sources??{}))check(hash(p)===h,`${name}: changed source ${p}`);
 check(t.provenance.sourcesUnchanged!==false,`${name}: source changed during run`);
 if(mode!=="baseline")check(t.provenance.jawPolicySha256===hash(mode==="learned"?"app/public/policies/jaw-ppo.json":"app/public/policies/jaw-ppo-initial.json"),`${name}: actor mismatch`);
 const acceptance={...a,landingSeconds,accepted:a.accepted&&landingSeconds>=.3};
 if(a.accepted&&landingSeconds<.3)acceptance.failures=[...a.failures,"release landing not continuously stable for 0.3 s"];
 return {name,mode,file,sha256:hash(file),config:t.config,phase:t.phase,acceptance,summary:t.summary,modelAudit:t.modelAudit,provenance:t.provenance,events:t.events};
}
const trialNames=["nominal","x_back_1p5mm","y_right_1p5mm","mass_plus_20pct"],rows=[];
for(const name of trialNames)for(const mode of ["baseline","learned","untrained"]){const dir=name==="nominal"&&mode!=="untrained"?nominalDir:perturbDir;rows.push(review(`${dir}/${name}-${mode}.json`,mode,name));}
for(const name of trialNames){
 const paired=rows.filter(r=>r.name===name),physical=c=>{const {actor,output,...rest}=c;return rest;};
 check(paired.every(r=>JSON.stringify(physical(r.config))===JSON.stringify(physical(paired[0].config))),`${name}: physical conditions are not paired`);
 check(paired.every(r=>r.modelAudit.timestep===.0000625),`${name}: execution timestep differs`);
}
const nominal=rows.find(r=>r.name==="nominal"&&r.mode==="learned"),fine=review(`${nominalDir}/half-step-learned.json`,"learned","half_timestep");
const materialVariants=[review(`${perturbDir}/stiffer-learned.json`,"learned","young_plus_20pct"),review(`${perturbDir}/coarse-learned.json`,"learned","27_nodes")];
const refinements=[review(`${refinementDir}/ground-finer-219-learned.json`,"learned","219_nodes"),review(`${refinementDir}/ground-finer-219-half-step.json`,"learned","219_nodes_half_timestep"),review(`${refinementDir}/ground-finer-219-back-half-step.json`,"learned","219_nodes_back_1p5mm_half_timestep")];
check(nominal.acceptance.accepted,"nominal learned pickup failed");
check(rows.find(r=>r.name==="nominal"&&r.mode==="baseline").acceptance.accepted,"nominal scripted baseline failed");
check(!rows.find(r=>r.name==="nominal"&&r.mode==="untrained").acceptance.accepted,"untrained nominal actor also completed task; no demonstrated improvement");
check(fine.acceptance.accepted,"half-step nominal failed");
const halfStepClearanceDifferenceMm=Math.abs(nominal.acceptance.minHeldClearance-fine.acceptance.minHeldClearance)*1000;
check(halfStepClearanceDifferenceMm<1,"nominal clearance changed by at least 1 mm at half timestep");
check(JSON.stringify(nominal.events.slice(0,3))===JSON.stringify(fine.events.slice(0,3)),"physics timestep changed prescribed approach clock");
const counts=Object.fromEntries(["baseline","learned","untrained"].map(mode=>[mode,rows.filter(r=>r.mode===mode&&r.acceptance.accepted).length]));
check(counts.learned>counts.untrained,"no complete-task improvement on paired cases");
const browser=review(browserPath,"learned","browser"),bp=browser.provenance,np=nominal.provenance;
check(browser.acceptance.accepted,"actual browser did not pass");
check(bp.controlMode==="learned","browser mode mismatch");
for(const key of ["policySha256","groundPolicySha256","jawPolicySha256"])check(bp[key]===np[key],`browser ${key} mismatch`);
// Chrome preserves the leading XML declaration while JSDOM omits it. Check
// the complete exported XML, allowing ONLY that declaration to differ. No
// physical attributes, geometry, whitespace or other contents are normalized.
const browserXml=read(browserPath).modelXml??"",textHash=s=>createHash("sha256").update(s).digest("hex");
const modelParity={rawEqual:bp.modelSha256===np.modelSha256,exportMatchesBrowserHash:textHash(browserXml)===bp.modelSha256,withoutXmlDeclarationMatchesNode:textHash(browserXml.replace(/^<\?xml[^?]*\?>/,""))===np.modelSha256};
check(modelParity.exportMatchesBrowserHash&&modelParity.withoutXmlDeclarationMatchesNode,"browser model differs beyond leading XML declaration");
for(const [key,p] of Object.entries({controllerSha256:"app/src/game/ground-pick-controller.js",coreSha256:"app/src/game/whole-body-core.js",sceneSha256:"app/src/game/ground-pick-scene.js",jawAdapterSha256:"app/src/game/jaw-rl.js"}))check(bp[key]===hash(p),`browser ${key} source mismatch`);
const result={generatedAt:new Date().toISOString(),accepted:failures.length===0,failures,problem:"P0",scope:"Fixed 99-vertex simulation example: frozen table-trained jaw PPO transferred to free-standing ground pickup of a radial elastic ball. No ground-specific training, whole-body RL or hardware/material validation.",counts,pairedConditions:4,halfStepClearanceDifferenceMm,modelParity,meshRefinement:{allPassed:refinements.every(r=>r.acceptance.accepted),trials:refinements,notPartOfPairedSuccessRate:true},rows,fine,materialVariants,browser,materialBench:{file:"rl/runs/ground-development/soft-material-final.json",sha256:hash("rl/runs/ground-development/soft-material-final.json")}};
fs.mkdirSync(path.dirname(path.resolve(root,output)),{recursive:true});fs.writeFileSync(path.resolve(root,output),JSON.stringify(result,null,2)+"\n");
console.log(JSON.stringify({accepted:result.accepted,failures,counts,halfStepClearanceDifferenceMm,output},null,2));assert.equal(result.accepted,true);
