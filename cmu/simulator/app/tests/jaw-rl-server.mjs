// Line-delimited JSON bridge. Uses the exact audited WASM physics path.
import readline from "node:readline";
import fs from "node:fs";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {dom} from "./physics-fixture.mjs";
import {makeTablePickXml,TABLE_PICK_DEFAULTS} from "../src/game/table-pick-scene.js";
import {HeadReachController} from "../src/game/head-reach-controller.js";
import {evaluateWholeBodyTrial} from "../src/game/whole-body-acceptance.js";
import {JawActionFilter,jawObservation,jawReward,jawSafetyFailure,JAW_RL_REPEAT,JAW_RL_DT} from "../src/game/jaw-rl.js";

let r, ctl, config, filter, contact, rows, decisions, terminal, rewardTotal, startWall;
function observation(){return jawObservation(r.snapshot(contact),ctl,filter.target);}
function evidence(){
  const result={config,phase:ctl.phase,events:ctl.events,rows,summary:{...r.summary},modelAudit:r.modelAudit,
    learnedDecisions:decisions,reward:rewardTotal,wallSeconds:(performance.now()-startWall)/1000};
  result.acceptance=evaluateWholeBodyTrial(result);return result;
}
async function motorTick(action, scripted=false){
  const out=ctl.update(r.snapshot(contact));
  await r.policy("walk",out.command);r.applyTask(out);
  const jaw=scripted?out.jaw:filter.update(action);
  contact=r.step(jaw);
  const s=r.snapshot(contact);rows.push({...s,phase:ctl.phase,jawTarget:jaw,learnedAction:scripted?null:action});
  return s;
}
async function reset(options){
  if(r)await r.dispose();
  config={...TABLE_PICK_DEFAULTS,dt:JAW_RL_DT,...options};
  r=await wholeBodyRuntime({mutate:xml=>makeTablePickXml(xml,dom,config)});
  r.reset([config.x,config.y,config.z+.0005,1,0,0,0]);
  ctl=new HeadReachController({height:config.z,lift:config.lift,biteOffset:config.biteOffset});
  filter=new JawActionFilter();contact={};rows=[];decisions=0;terminal=false;rewardTotal=0;startWall=performance.now();
  // The deterministic approach is part of every complete rollout, not an
  // invisible reset into a gripped/held pose. All its physics is audited.
  while(!["clamp","failed"].includes(ctl.phase))await motorTick(1,true);
  if(ctl.phase==="failed"||jawSafetyFailure(r.summary))throw Error("Scripted approach failed: "+JSON.stringify(evidence().acceptance));
  return {observation:observation(),info:{config,approachSeconds:r.d.time}};
}
async function step(action,scripted=false){
  if(!r||terminal)throw Error("Call reset before step");
  if(!scripted&&action!==0&&action!==1)throw Error("Invalid action");
  let reward=0;
  for(let i=0;i<JAW_RL_REPEAT;i++){
    const s=await motorTick(action,scripted);
    reward+=jawReward(s,ctl)/JAW_RL_REPEAT;
    if(jawSafetyFailure(r.summary)){ctl.enter("failed");reward-=20;terminal=true;break;}
    if(["done","failed"].includes(ctl.phase)){terminal=true;break;}
  }
  decisions++;
  if(decisions>=250){terminal=true;ctl.enter("failed");}
  let info={phase:ctl.phase};
  if(terminal){const trial=evidence();reward+=trial.acceptance.accepted?10:-5;info={...info,acceptance:trial.acceptance,summary:trial.summary,modelAudit:trial.modelAudit,decisions,config};}
  rewardTotal+=reward;
  return {observation:observation(),reward,terminated:terminal,truncated:false,info};
}
const input=readline.createInterface({input:process.stdin,crlfDelay:Infinity});
for await(const line of input){
  try{
    const request=JSON.parse(line);let result;
    if(request.op==="reset")result=await reset(request.config??{});
    else if(request.op==="step")result=await step(request.action,request.scripted);
    else if(request.op==="evidence"){
      result=evidence();if(request.output){fs.writeFileSync(request.output,JSON.stringify(result)+"\n");result={output:request.output,acceptance:result.acceptance};}
    }else if(request.op==="close"){if(r)await r.dispose();console.log(JSON.stringify({ok:true}));break;}
    else throw Error("Unknown operation");
    console.log(JSON.stringify(result));
  }catch(error){console.log(JSON.stringify({error:String(error),stack:error.stack}));}
}
