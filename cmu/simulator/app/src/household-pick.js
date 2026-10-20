import {createDisplayBudget,waitForVisible} from "./game/browser-budget.js";
import {preparePhysicsXml} from './game/physics-model.js';
import {householdConfig,makeHouseholdPickXml} from './game/household-pick-scene.js';
import {HouseholdPickController} from './game/household-pick-controller.js';
import {createWholeBodyCore} from './game/whole-body-core.js';
import {JawActionFilter} from './game/jaw-rl.js';
import {POLICIES} from './game/constants.js';
import {householdView} from './household-view.js';
import {evaluateHouseholdTrial} from './game/household-pick-acceptance.js';
import {createPredictiveClient} from './household-predictive-client.js';
import {timedForward} from './household-drive.js';
const $=s=>document.querySelector(s),params=new URLSearchParams(location.search);
const predictive=params.get('controller')==='predictive';
const predictiveDefaults={dynamicsReach:true,axisGrasp:true,graspForecast:true,reachFeedback:true,stanceFeedback:true,headGravityFeedback:true,stanceLift:false,readyTolerance:.002,crouch:.14,alignTolerance:.006,settleTolerance:.008,yawTolerance:.1,settleYawTolerance:.12,navSpeed:.25,preStand:true,footNavigation:true,navBackSpeed:.35,reachTimeout:30,stopLiftClearance:.04};
const config=householdConfig({kind:params.get('object')??'pen',...(predictive?predictiveDefaults:{}),...JSON.parse(params.get('config')??'{}')});
$('#object').value=config.kind;
$('#object').onchange=e=>{const url=new URL(location.href);url.searchParams.set('object',e.target.value);url.searchParams.delete('config');location.href=url.href;};
const labels={manual:'手动行走',prestand:'停步，等待站稳',align:'自动调整位置与朝向',settle:'站稳',descend:'蹲低、低头',reach:'伸颈接近物品',recover:'站起、挪步后重新尝试',clamp:'闭嘴夹持',lift:'叼住并抬起',hold:'检查离地保持',carried:'已叼住 · 按 E 放下',release:'张嘴放下，等待自然落地',done:'物品已放下',failed:'本次失败 · 保留现场'};
async function start(){
  const [{default:loadMujoco},ort]=await Promise.all([import(/* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js'),import(/* @vite-ignore */ 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs')]);
  ort.env.wasm.wasmPaths='https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/';ort.env.wasm.numThreads=1;
  const mj=await loadMujoco(),prepared=preparePhysicsXml(await (await fetch('/robot/mjlab/robot_allcollisions.xml')).text(),window),xml=makeHouseholdPickXml(prepared.xml,window,config),vfs=new mj.MjVFS();
  try{await Promise.all(prepared.meshFiles.map(async f=>{const response=await fetch('/robot/mjlab/meshes/'+f);if(!response.ok)throw Error('模型载入失败：'+f);vfs.addBuffer('assets/'+f,new Uint8Array(await response.arrayBuffer()));}));}catch(e){vfs.delete();throw e;}
  const m=mj.MjModel.from_xml_string(xml,vfs);vfs.delete();const d=new mj.MjData(m),sessions={};
  for(const name of ['walk','groundpick'])sessions[name]=await ort.InferenceSession.create(await (await fetch(POLICIES[name])).arrayBuffer(),{executionProviders:['wasm']});
  const r=createWholeBodyCore(mj,m,d,{async infer(name,obs){return (await sessions[name].run({obs:new ort.Tensor('float32',obs,[1,61])})).actions.data;}}),view=await householdView(r,config);
  $('#scope').textContent=config.kind==='pen'?`长 ${(config.length*100).toFixed(0)} 厘米、直径 ${(config.radius*2000).toFixed(0)} 毫米、${config.mass*1000} 克的笔。`:`${r.modelAudit.flexVertices} 个可自由移动的布料节点，${config.mass*1000} 克；近似闭口袜头与开口袜筒。真实布料计算明显慢于现实时间，请看下方仿真计时。`;
  if(predictive)$('#scope').textContent+=' 当前是实验性的物理预测控制，不是新训练好的强化学习策略。按 G 后由本机预演电机动作，计算时仿真暂停，速度慢于现实时间。';
  let predictor=null,ctl,filter,contact={},rows=[],paused=false,error=null,resetRequested=false,pickRequested=false,releaseRequested=false,acceptance=null,truncated=false,driveUntil=0,inputEvents=[];
  const keys=new Set();
  async function reset(){const yaw=config.objectYaw??0;r.reset([config.x,config.y,config.z+.0005,Math.cos(yaw/2),0,0,Math.sin(yaw/2)],{rootOffset:config.rootOffset});ctl=new HouseholdPickController(config);filter=new JawActionFilter();contact={};rows=[];keys.clear();paused=false;error=null;acceptance=null;truncated=false;driveUntil=0;inputEvents=[];if(predictor)await predictor.reset();}
  function request(key){if(key==='KeyG')pickRequested=true;if(key==='KeyE')releaseRequested=true;if(key==='Space')paused=!paused;}
  addEventListener('keydown',e=>{if(e.target.matches('select,input,textarea')||e.repeat)return;if(['KeyW','KeyA','KeyS','KeyD','KeyG','KeyE','Space'].includes(e.code)){e.preventDefault();keys.add(e.code);request(e.code);}});
  addEventListener('keyup',e=>keys.delete(e.code));addEventListener('blur',()=>keys.clear());
  for(const button of document.querySelectorAll('[data-drive]')){button.onpointerdown=e=>{button.setPointerCapture(e.pointerId);keys.add(button.dataset.drive);};button.onpointerup=button.onpointercancel=()=>keys.delete(button.dataset.drive);}
  $('#pick').onclick=()=>pickRequested=true;$('#release').onclick=()=>releaseRequested=true;$('#reset').onclick=()=>resetRequested=true;$('#pause').onclick=()=>paused=!paused;
  function drive(seconds){const event=timedForward({phase:ctl.phase,paused,time:d.time},seconds);if(event){driveUntil=event.until;inputEvents.push(event);}}
  $('#step-forward').onclick=()=>drive(.5);$('#walk-forward').onclick=()=>drive(2);
  $('#wire').onchange=e=>view.wires.forEach(mesh=>mesh.visible=e.target.checked);
  const evidence=()=>({config,phase:ctl.phase,events:ctl.events,inputEvents,graspEvents:ctl.inner?.events,attempts:ctl.attempts,summary:{...r.summary},modelAudit:r.modelAudit,snapshot:r.snapshot(contact),rows,alignment:ctl.alignment,paused,error,acceptance,truncated,...(predictor?{planner:{controller:'local native-replica motor prediction, not PPO',seconds:predictor.totalSeconds,updates:predictor.plan?.audit?.update??0}}:{})});
  $('#export').onclick=()=>{const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify({...evidence(),modelXml:xml})],{type:'application/json'}));a.href=url;a.download=`microduck-${config.kind}-trial.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  await reset();if(predictive)predictor=await createPredictiveClient(r,xml,config);for(const b of document.querySelectorAll('button'))b.disabled=false;
  const display=createDisplayBudget();
  function render(now){if(!display.shouldRender(now)){requestAnimationFrame(render);return;}view.render();const s=r.snapshot(contact);$('#state').textContent=error?error:(paused?'已暂停 · ':'')+(predictor?.busy?'本机预演电机动作中 · 仿真暂停':labels[ctl.phase]??ctl.phase)+(ctl.phase==='manual'?' · '+ctl.message:'')+(acceptance&&!acceptance.accepted?' · 验收未通过：'+acceptance.failures.join('；'):'');
    $('#measure').textContent=`仿真时间 ${d.time.toFixed(2)} s\n物品最低点 ${(s.shape.minZ*1000).toFixed(1)} mm\n连续夹持 ${ctl.hold.toFixed(2)} s\n上嘴 ${(contact.upperForce??0).toFixed(3)} N · 下嘴 ${(contact.lowerForce??0).toFixed(3)} N\n最大物品穿入 ${(r.summary.maxPenetration*1000).toFixed(3)} mm${ctl.phase==='align'?`\n对准距离误差 ${(ctl.alignment?.distance*1000||0).toFixed(1)} mm`:''}`;
    $('#measure').textContent+=`\n${display.speedLabel(d.time,paused||predictor?.busy)}`;
    $('#pause').textContent=paused?'继续':'暂停';$('#pick').disabled=ctl.phase!=='manual';$('#release').disabled=!['hold','carried'].includes(ctl.phase);$('#step-forward').disabled=$('#walk-forward').disabled=ctl.phase!=='manual'||paused;requestAnimationFrame(render);}
  async function tick(){await waitForVisible();const start=performance.now();try{
    if(resetRequested){resetRequested=false;await reset();}
    if(!paused&&!error){
      const before=r.snapshot(contact);
      if(pickRequested){pickRequested=false;ctl.begin(before);keys.clear();driveUntil=0;}if(releaseRequested){releaseRequested=false;ctl.release(before);}
      if(predictor){
        if(ctl.phase==='reach'&&ctl.inner?.time>=4.38)await predictor.update('reach');
        else if(config.dynamicsLift&&['lift','hold','carried'].includes(ctl.phase))await predictor.update('transport');
      }
      // A pause or reset requested while prediction was running invalidates
      // that pending actuation. No physical step is performed in this branch.
      if(paused||resetRequested||document.hidden){setTimeout(tick,1);return;}
      const command=new Float32Array(13);command[0]=(keys.has('KeyW')||d.time<driveUntil?.25:0)-(keys.has('KeyS')?.2:0);command[2]=(keys.has('KeyA')?.7:0)-(keys.has('KeyD')?.7:0);
      const target=await ctl.actuate(r,before,command),jaw=filter.update(config.open?1:target===0?0:1);contact=config.kind==='sock'?await r.stepAsync(jaw):r.step(jaw);rows.push({...r.snapshot(contact),phase:ctl.phase,manualCommand:Array.from(command.slice(0,3)),...(predictor?{motorPlan:predictor.plan}:{} )});
      if(rows.length>10000){rows.splice(50,1);truncated=true;}
      if(['carried','done','failed'].includes(ctl.phase)&&(ctl.phase!=='carried'||rows.length%10===0))acceptance=evaluateHouseholdTrial(evidence(),{requireRelease:ctl.phase==='done'});
      if(!r.summary.finite||r.summary.solverReset){ctl.enter('failed');error='物理计算异常，已停止并保留现场';}
      if(['failed','done'].includes(ctl.phase))paused=true;
    }
  }catch(e){error='运行失败：'+e.message;paused=true;console.error(e);}setTimeout(tick,Math.max(1,20-(performance.now()-start)));}
  window.householdPick={get evidence(){const {rows,...small}=evidence();return small;},get rows(){return rows;}};render();tick();
}
start().catch(e=>{$('#state').textContent='加载失败：'+e.message;console.error(e);});
