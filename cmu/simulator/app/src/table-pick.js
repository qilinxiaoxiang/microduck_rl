import {createDisplayBudget,displayPixelRatio,waitForVisible} from "./game/browser-budget.js";
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {preparePhysicsXml} from "./game/physics-model.js";
import {makeTablePickXml,TABLE_PICK_DEFAULTS} from "./game/table-pick-scene.js";
import {makeGroundPickXml,GROUND_PICK_DEFAULTS} from "./game/ground-pick-scene.js";
import {GroundPickController} from "./game/ground-pick-controller.js";
import {evaluateGroundTrial} from "./game/ground-pick-acceptance.js";
import groundControllerSource from "./game/ground-pick-controller.js?raw";
import groundSceneSource from "./game/ground-pick-scene.js?raw";
import {createWholeBodyCore} from "./game/whole-body-core.js";
import {HeadReachController} from "./game/head-reach-controller.js";
import {JawActionFilter,jawObservation,jawActor,jawSafetyFailure,JAW_RL_REPEAT,JAW_RL_PHASES,JAW_RL_DT} from "./game/jaw-rl.js";
import {evaluateWholeBodyTrial} from "./game/whole-body-acceptance.js";
import {buildRig,loadKinematics,setJawOpen,JAW_MAX_OPEN,ROBOT_KINEMATICS} from "./game/duck.js";
import {POLICIES} from "./game/constants.js";
import controllerSource from "./game/head-reach-controller.js?raw";
import coreSource from "./game/whole-body-core.js?raw";
import sceneSource from "./game/table-pick-scene.js?raw";
import jawSource from "./game/jaw-rl.js?raw";

const state=document.querySelector("#state"),measure=document.querySelector("#measure");
const ground=location.pathname.includes("ground-pick");
const failureLabels={
  "controller did not complete":"流程未完成",
  "no continuous one-second elevated opposing-contact hold":"未离台持续夹持至少 1 秒",
  "object slipped by at least 3 mm during final held second":"保持时滑动达到 3 毫米",
  "opening did not release the object into a real fall":"未张嘴释放并自然落下",
  "externally applied force or torque detected":"检测到人工外加力",
  "non-finite simulation state":"物理计算出现异常值",
  "payload impact penetration exceeds 0.5 mm":"物体接触穿入超过 0.5 毫米",
  "robot/environment penetration exceeds 0.5 mm":"身体接触穿入超过 0.5 毫米",
  "robot braced on something other than its feet":"身体借助双脚以外部位支撑",
  "joint exceeded anatomical limit by over 0.002 rad":"关节超出允许范围",
  "payload or robot base is not free":"物体或机器人基座被固定",
  "attachment or externally prescribed body present":"检测到附着或外部位置控制",
  "no continuous elevated opposing-pad hold for 1.2 seconds":"未离地持续双面夹持至少 1.2 秒",
  "ball slipped by at least 3 mm during final held second":"保持时小球滑动达到 3 毫米",
  "opening did not release the ball into a real fall onto the ground":"小球未自然落下并回到地面",
  "non-finite state or solver reset":"物理计算异常或发生自动重置",
  "physics engine warning detected":"物理计算报告警告",
  "a soft-ball radial element collapsed or inverted":"软球内部单元发生塌陷或翻转",
  "ball did not start resting directly on the ground":"小球起始位置没有直接落在地面",
  "payload, soft nodes or robot base not free":"小球或机器人被外部固定",
  "ground is not a flat zero-height plane":"场景中存在垫高或非平地结构",
  "payload is not a deformable ball":"物体不是可变形小球",
};
async function start(){
  const mode=new URLSearchParams(location.search).get("controller")??(ground?"learned":"scripted");
  if(!["scripted","baseline","learned","untrained"].includes(mode))throw Error("未知控制器");
  const actorText=["scripted","baseline"].includes(mode)?null:await (await fetch(mode==="learned"?"/policies/jaw-ppo.json":"/policies/jaw-ppo-initial.json")).text();
  const actor=actorText?JSON.parse(actorText):null;
  if(actor){document.querySelector("h1").textContent=mode==="learned"?"强化学习 · 嘴部控制":"对照 · 未训练策略";document.querySelector("#scope").textContent="新策略决定嘴部开合；站立、伸颈与抬头沿用已有控制器。蓝色物体只受重力和真实碰撞作用。";}
  if(mode==="baseline")document.querySelector("h1").textContent="对照 · 原控制器";
  const [{default:loadMujoco},ort]=await Promise.all([
    import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js"),
    import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs")]);
  ort.env.wasm.wasmPaths="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";ort.env.wasm.numThreads=1;
  const mj=await loadMujoco(),config=ground?{...GROUND_PICK_DEFAULTS,...JSON.parse(new URLSearchParams(location.search).get("config")??"{}")}:{...TABLE_PICK_DEFAULTS,...(mode!=="scripted"?{dt:JAW_RL_DT}:{})};
  if(ground){document.querySelector("h1").textContent=mode==="learned"?"从地上叼小球":mode==="untrained"?"地面小球 · 未训练对照":"地面小球 · 规则控制对照";document.querySelector("#scope").textContent=`直径 ${(config.radius*2000).toFixed(0)} 毫米、质量 ${(config.mass*1000).toFixed(1)} 克的弹性软球，直接放在平地上。${mode==="learned"?"沿用已训练的嘴部开合策略；蹲低、抬头由已有策略与专用控制器完成。":mode==="untrained"?"使用训练前的嘴部网络，身体动作保持相同。":"嘴部按规则开合，身体动作保持相同。"}材质是未经过实物标定的近似模型。`;}
  const prepared=preparePhysicsXml(await (await fetch("/robot/mjlab/robot_allcollisions.xml")).text(),window);
  const xml=(ground?makeGroundPickXml:makeTablePickXml)(prepared.xml,window,config),vfs=new mj.MjVFS();
  try{await Promise.all(prepared.meshFiles.map(async f=>{const res=await fetch("/robot/mjlab/meshes/"+f);if(!res.ok)throw Error("模型文件载入失败："+f);vfs.addBuffer("assets/"+f,new Uint8Array(await res.arrayBuffer()));}));}catch(e){vfs.delete();throw e;}
  const model=mj.MjModel.from_xml_string(xml,vfs);vfs.delete();const data=new mj.MjData(model);
  const policyBytes=await (await fetch(POLICIES.walk)).arrayBuffer();
  const policy=await ort.InferenceSession.create(policyBytes,{executionProviders:["wasm"]});
  const sessions={walk:policy};
  const groundPolicyBytes=ground?await (await fetch(POLICIES.groundpick)).arrayBuffer():null;
  if(ground)sessions.groundpick=await ort.InferenceSession.create(groundPolicyBytes,{executionProviders:["wasm"]});
  const hash=async value=>Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",typeof value==="string"?new TextEncoder().encode(value):value)),x=>x.toString(16).padStart(2,"0")).join("");
  const provenance={engine:"MuJoCo 3.11.0",inference:"onnxruntime-web 1.27.0 wasm",modelSha256:await hash(xml),policySha256:await hash(policyBytes),controllerSha256:await hash(controllerSource),coreSha256:await hash(coreSource),sceneSha256:await hash(sceneSource)};
  Object.assign(provenance,{controlMode:mode,jawPolicySha256:actorText?await hash(actorText):null,jawAdapterSha256:await hash(jawSource)});
  if(ground)Object.assign(provenance,{controllerSha256:await hash(groundControllerSource),sceneSha256:await hash(groundSceneSource),groundPolicySha256:await hash(groundPolicyBytes),task:"ground-soft-ball",learnedScope:"frozen table-trained jaw PPO transferred to ground; body approach/lift controller is not learned here"});
  const r=createWholeBodyCore(mj,model,data,{async infer(name,obs){const out=await sessions[name].run({obs:new ort.Tensor("float32",obs,[1,61])});return out.actions.data;}});
  const rig=await buildRig(await loadKinematics(ROBOT_KINEMATICS));rig.root.rotation.set(0,0,0);
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x111e28);scene.add(rig.placer);
  // Flatten body frames and use actual MuJoCo poses, without joint-angle
  // clamping or a separate kinematic approximation in the visual readback.
  const frames=[];
  for(const [name,group] of rig.bodies){const id=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY.value,name);if(id>=0){rig.root.add(group);frames.push({id,group});}}
  const camera=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.001,10);camera.up.set(0,0,1);camera.position.set(-.15,-.60,.40);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(displayPixelRatio);renderer.setSize(innerWidth,innerHeight);document.body.prepend(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(-.49,0,.13);controls.update();
  scene.add(new THREE.HemisphereLight(0xffffff,0x5b6879,2.4));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(-.2,-.5,1);scene.add(light);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(3,3),new THREE.MeshStandardMaterial({color:0x293b48,roughness:1}));scene.add(floor);
  const grid=new THREE.GridHelper(3,60,0x577281,0x354f60);grid.rotation.x=Math.PI/2;grid.position.z=.0001;scene.add(grid);
  const objects=[],wires=[],cache=new Map(),matrix=new THREE.Matrix4();
  for(let i=0;i<model.ngeom;i++){
    const name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM.value,i)??"",target=name==="payload_geom",stand=name==="payload_pedestal_geom";
    if(ground&&target)continue;
    const mesh=model.geom_type[i]===mj.mjtGeom.mjGEOM_MESH.value;
    if(!target&&!stand&&!(mesh&&model.geom_contype[i]))continue;
    let geometry;
    if(mesh){const mid=model.geom_dataid[i];if(!cache.has(mid)){const va=model.mesh_vertadr[mid],vn=model.mesh_vertnum[mid],fa=model.mesh_faceadr[mid],fn=model.mesh_facenum[mid],g=new THREE.BufferGeometry();g.setAttribute("position",new THREE.BufferAttribute(new Float32Array(model.mesh_vert.slice(va*3,(va+vn)*3)),3));g.setIndex(new THREE.BufferAttribute(new Uint32Array(model.mesh_face.slice(fa*3,(fa+fn)*3)),1));g.computeVertexNormals();cache.set(mid,g);}geometry=cache.get(mid);}
    else geometry=new THREE.BoxGeometry(...Array.from(model.geom_size.slice(i*3,i*3+3),v=>2*v));
    const wire=!target&&!stand,object=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:target?0x45c7f7:stand?0x839aab:0x55ffaa,wireframe:wire,transparent:wire,opacity:wire?.6:1,roughness:.6}));
    object.visible=!wire;scene.add(object);objects.push({i,object});if(wire)wires.push(object);
  }
  let flexGeometry;
  if(ground&&r.payloadFlexId>=0){
    const f=r.payloadFlexId,va=model.flex_vertadr[f],vn=model.flex_vertnum[f],sa=model.flex_shelldataadr[f],sn=model.flex_shellnum[f];
    flexGeometry=new THREE.BufferGeometry();
    flexGeometry.setAttribute("position",new THREE.BufferAttribute(new Float32Array(vn*3),3));
    flexGeometry.setIndex(Array.from(model.flex_shell.slice(sa,sa+sn*3)));
    scene.add(new THREE.Mesh(flexGeometry,new THREE.MeshStandardMaterial({color:0xffa629,roughness:.85,side:THREE.DoubleSide})));
    controls.target.set(-.51,0,.14);camera.position.set(-.94,-.56,.31);controls.update();
  }
  let ctl,jawFilter,learnedTicks=0,learnedAction=1,contact={},paused=false,resetRequested=false,rows=[],framesRendered=0,lastTickMs=0,error=null,acceptance=null;
  function reset(){r.reset([config.x,config.y,config.z+.0005,1,0,0,0]);ctl=ground?new GroundPickController(config):new HeadReachController({height:config.z,lift:config.lift,biteOffset:config.biteOffset});jawFilter=new JawActionFilter();learnedTicks=0;learnedAction=1;contact={};rows=[];paused=false;error=null;acceptance=null;document.querySelector("#pause").textContent="暂停";}
  const labels={settle:"站稳",descend:"在物体后方降低嘴部",reach:"水平伸颈对准",clamp:"闭嘴夹紧",lift:"离台抬起",hold:"只靠嘴部接触保持",release:"张嘴退出，观察自然下落",done:"本次流程结束",failed:"本次失败：保留现场，不自动摆回"};
  if(ground)Object.assign(labels,{descend:"蹲低、低头",reach:"继续俯身，让嘴接近地面小球",lift:"叼住小球，站起抬头",release:"张嘴、低头，让小球自然落地"});
  const display=createDisplayBudget();
  function render(now){
    if(!display.shouldRender(now)){requestAnimationFrame(render);return;}
    if(flexGeometry){const start=model.flex_vertadr[r.payloadFlexId]*3,attr=flexGeometry.getAttribute("position");attr.array.set(data.flexvert_xpos.subarray(start,start+attr.array.length));attr.needsUpdate=true;flexGeometry.computeVertexNormals();flexGeometry.computeBoundingSphere();}
    for(const {id,group} of frames){const p=data.xpos.subarray(id*3,id*3+3),q=data.xquat.subarray(id*4,id*4+4);group.position.set(...p);group.quaternion.set(q[1],q[2],q[3],q[0]);}setJawOpen(rig,data.qpos[r.jaw]/JAW_MAX_OPEN);
    for(const {i,object} of objects){if(!object.visible)continue;const p=data.geom_xpos.subarray(i*3,i*3+3),v=data.geom_xmat.subarray(i*9,i*9+9);object.position.set(...p);matrix.set(v[0],v[1],v[2],0,v[3],v[4],v[5],0,v[6],v[7],v[8],0,0,0,0,1);object.quaternion.setFromRotationMatrix(matrix);}
    state.textContent=error?"运行失败："+error:acceptance?(acceptance.accepted?"本次接触验收通过":"未通过验收："+acceptance.failures.map(f=>failureLabels[f]??f).join("；")):(paused?"已暂停 · ":"")+(labels[ctl.phase]??ctl.phase);
    measure.textContent=`仿真时间 ${data.time.toFixed(2)} s\n物体离台高度 ${((data.qpos[r.p+2]-config.z)*100).toFixed(2)} cm\n连续夹持 ${ctl.hold.toFixed(2)} s\n上嘴 ${(contact.upperForce??0).toFixed(3)} N · 下嘴 ${(contact.lowerForce??0).toFixed(3)} N\n最大物体穿入 ${(r.summary.maxPenetration*1000).toFixed(3)} mm\n其他部位撑地/撑台：${r.summary.nonFootSupport?"有":"无"}\n外加力/力矩峰值 ${r.summary.maxExternal.toFixed(3)}`;
    if(ground)measure.textContent=measure.textContent.replace("物体离台高度","球心抬升高度").replace("撑地/撑台","撑地")+`\n球面最低点 ${(r.snapshot(contact).shape.minZ*1000).toFixed(2)} mm\n${r.modelAudit.flexVertices} 个真实弹性节点 · 无固定台`;
    measure.textContent+=`\n${display.speedLabel(data.time,paused)}`;
    controls.update();renderer.render(scene,camera);framesRendered++;requestAnimationFrame(render);
  }
  async function tick(){
    await waitForVisible();
    const began=performance.now();
    try{if(resetRequested){resetRequested=false;reset();}
      if(!paused&&!error){
        const before=r.snapshot(contact),learning=actor&&JAW_RL_PHASES.includes(ctl.phase);
        if(learning&&learnedTicks%JAW_RL_REPEAT===0)learnedAction=jawActor(actor,Array.from(new Float32Array(jawObservation(before,ctl,jawFilter.target)))).action;
        let out;
        if(ground)out={jaw:await ctl.actuate(r,before)};
        else {out=ctl.update(before);await r.policy("walk",out.command);r.applyTask(out);}
        const jaw=learning?jawFilter.update(learnedAction):ground?jawFilter.update(out.jaw===0?0:1):out.jaw;if(learning)learnedTicks++;
        contact=r.step(jaw);rows.push({...r.snapshot(contact),phase:ctl.phase,jawTarget:jaw,learnedAction:learning?learnedAction:null});
        if((ground||actor||mode==="baseline")&&jawSafetyFailure(r.summary))ctl.enter("failed");
        if(rows.length>2500)rows.shift();if(["done","failed"].includes(ctl.phase)){paused=true;acceptance=(ground?evaluateGroundTrial:evaluateWholeBodyTrial)({phase:ctl.phase,rows,summary:r.summary,config,modelAudit:r.modelAudit});document.querySelector("#pause").textContent="已结束";}}
    }catch(e){error=String(e);paused=true;console.error(e);}
    lastTickMs=performance.now()-began;
    // Always yield a real browser task, even when slower than real time.
    setTimeout(tick,Math.max(1,20-lastTickMs));
  }
  document.querySelector("#restart").disabled=false;document.querySelector("#restart").onclick=()=>{resetRequested=true;};
  document.querySelector("#pause").disabled=false;document.querySelector("#pause").onclick=()=>{if(acceptance)return;paused=!paused;document.querySelector("#pause").textContent=paused?"继续":"暂停";};
  document.querySelector("#wire").onchange=e=>wires.forEach(o=>o.visible=e.target.checked);
  document.querySelector("#export").disabled=false;document.querySelector("#export").onclick=()=>{const a=document.createElement("a"),url=URL.createObjectURL(new Blob([JSON.stringify({config,provenance,modelXml:ground?xml:undefined,events:ctl.events,phase:ctl.phase,summary:r.summary,modelAudit:r.modelAudit,acceptance,rows},null,2)],{type:"application/json"}));a.href=url;a.download=ground?"microduck-ground-ball-trial.json":"microduck-whole-body-trial.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
  reset();window.tablePick={get evidence(){return {phase:ctl.phase,time:data.time,hold:ctl.hold,events:ctl.events,summary:r.summary,modelAudit:r.modelAudit,acceptance,provenance,snapshot:r.snapshot(contact),framesRendered,lastTickMs,paused,error};},get rows(){return rows;}};
  requestAnimationFrame(render);tick();
}
start().catch(e=>{state.textContent="加载失败："+e.message;console.error(e);});
