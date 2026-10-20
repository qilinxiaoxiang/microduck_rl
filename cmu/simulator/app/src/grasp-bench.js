import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {preparePhysicsXml} from "./game/physics-model.js";
import {makeGraspBenchXml} from "./game/grasp-bench-model.js";
import {readPayloadContacts} from "./game/contact-observer.js";
import parts from "./game/beak-colliders.js";
import {TIMESTEP} from "./game/constants.js";

const state=document.querySelector("#state"),measure=document.querySelector("#measure");
async function start() {
  const {default:loadMujoco}=await import(/* @vite-ignore */ "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js");
  const mj=await loadMujoco();
  const base="/robot/mjlab/";
  const response=await fetch(base+"robot_allcollisions.xml");
  if(!response.ok)throw Error("model fetch failed");
  const prepared=preparePhysicsXml(await response.text(),window);
  const xml=makeGraspBenchXml(prepared.xml,window);
  const vfs=new mj.MjVFS();
  try {
    await Promise.all(prepared.meshFiles.map(async f=>{
      const res=await fetch(base+"meshes/"+f);if(!res.ok)throw Error(`mesh fetch failed: ${f}`);
      vfs.addBuffer("assets/"+f,new Uint8Array(await res.arrayBuffer()));
    }));
  } catch(error) {vfs.delete();throw error;}
  const model=mj.MjModel.from_xml_string(xml,vfs);vfs.delete();
  const data=new mj.MjData(model),scratch=new mj.DoubleBuffer(6);
  const id=(type,name)=>mj.mj_name2id(model,mj.mjtObj[`mjOBJ_${type}`].value,name);
  const p=model.jnt_qposadr[id("JOINT","payload_freejoint")],jaw=model.jnt_qposadr[id("JOINT","beak_joint")];
  const payload=[id("GEOM","payload_geom")],body=model.geom_bodyid[payload[0]];
  const lower=["jaw","jaw_soft"].flatMap(source=>parts[source].parts.map((_,i)=>id("GEOM",`beak_${source}_contact_${i}`)));
  const upper=parts.soft_mouth_top.parts.map((_,i)=>id("GEOM",`beak_soft_mouth_top_contact_${i}`));
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x101923);
  const camera=new THREE.PerspectiveCamera(40,innerWidth/innerHeight,.001,10);
  camera.up.set(0,0,1);camera.position.set(.30,-.32,.24);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.setSize(innerWidth,innerHeight);document.body.prepend(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(.07,0,.105);controls.update();
  scene.add(new THREE.HemisphereLight(0xffffff,0x49637b,2.5));
  const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(.2,-.3,.6);scene.add(light);
  const floor=new THREE.Mesh(new THREE.PlaneGeometry(2,2),new THREE.MeshStandardMaterial({color:0x243340,roughness:1}));scene.add(floor);
  const objects=[],wireObjects=[],cache=new Map();
  function meshGeometry(mid) {
    if(cache.has(mid))return cache.get(mid);
    const vadr=model.mesh_vertadr[mid],vnum=model.mesh_vertnum[mid],fadr=model.mesh_faceadr[mid],fnum=model.mesh_facenum[mid];
    const g=new THREE.BufferGeometry();
    g.setAttribute("position",new THREE.BufferAttribute(new Float32Array(model.mesh_vert.slice(3*vadr,3*(vadr+vnum))),3));
    g.setIndex(new THREE.BufferAttribute(new Uint32Array(model.mesh_face.slice(3*fadr,3*(fadr+fnum))),1));g.computeVertexNormals();cache.set(mid,g);return g;
  }
  for(let i=0;i<model.ngeom;i++) {
    const name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM.value,i)??"";
    const contact=name.startsWith("beak_")&&name.includes("_contact_");
    const mid=model.geom_dataid[i];let geometry;
    if(model.geom_type[i]===mj.mjtGeom.mjGEOM_MESH.value)geometry=meshGeometry(mid);
    else if(name==="payload_geom"||name==="payload_pedestal_geom")geometry=new THREE.BoxGeometry(...Array.from(model.geom_size.slice(i*3,i*3+3),x=>x*2));
    else continue;
    const meshName=mid>=0?mj.mj_id2name(model,mj.mjtObj.mjOBJ_MESH.value,mid):"";
    const color=name==="payload_geom"?0x56c8f5:name==="payload_pedestal_geom"?0x75909f:/jaw|mouth/.test(meshName)?0xeaa052:0xe8e7df;
    const material=new THREE.MeshStandardMaterial({color:contact?0x62ff9e:color,roughness:.65,wireframe:contact,transparent:contact,opacity:contact?.5:1});
    const object=new THREE.Mesh(geometry,material);object.userData.geom=i;object.visible=!contact;scene.add(object);objects.push(object);if(contact)wireObjects.push(object);
  }
  let paused=false,maxPen=0,maxExternal=0,contact={},steps=0,previous=performance.now(),accumulator=0;
  function reset(){mj.mj_resetData(model,data);data.qpos.set([.075,0,.124,1,0,0,0],p);data.qpos[jaw]=.32;mj.mj_forward(model,data);maxPen=0;maxExternal=0;steps=0;accumulator=0;previous=performance.now();paused=false;document.querySelector("#pause").textContent="暂停";}
  const matrix=new THREE.Matrix4();
  function frame(){
    const now=performance.now();if(!paused)accumulator+=Math.min(.05,(now-previous)/1000);else accumulator=0;previous=now;
    while(!paused && steps<Math.round(4/TIMESTEP) && accumulator>=TIMESTEP){
      accumulator-=TIMESTEP;
      const t=steps*TIMESTEP;data.ctrl[1]=.05*Math.min(1,Math.max(0,t-1));data.ctrl[0]=t>=3?.32:0;
      mj.mj_step(model,data);steps++;
      contact=readPayloadContacts(mj,model,data,payload,lower,upper,scratch);
      maxPen=Math.max(maxPen,contact.maxPenetration);
      for(const value of data.xfrc_applied.subarray(body*6,body*6+6))maxExternal=Math.max(maxExternal,Math.abs(value));
    }
    for(const object of objects){const i=object.userData.geom,pos=data.geom_xpos.subarray(i*3,i*3+3),r=data.geom_xmat.subarray(i*9,i*9+9);object.position.set(...pos);matrix.set(r[0],r[1],r[2],0,r[3],r[4],r[5],0,r[6],r[7],r[8],0,0,0,0,1);object.quaternion.setFromRotationMatrix(matrix);}
    const t=steps*TIMESTEP;state.textContent=t<1?"夹紧":t<2?"升降台抬起":t<3?"保持：只靠嘴部接触":t<4?"张嘴，观察自然运动":"试验结束，可重做";
    if(t>=4)accumulator=0;
    measure.textContent=`仿真时间 ${t.toFixed(2)} s\n上嘴 ${(contact.upperForce??0).toFixed(3)} N · 下嘴 ${(contact.lowerForce??0).toFixed(3)} N\n物体高度 ${(data.qpos[p+2]*100).toFixed(2)} cm\n最大穿入 ${(maxPen*1000).toFixed(3)} mm\n物体外加力/力矩峰值 ${maxExternal.toFixed(3)}`;
    controls.update();renderer.render(scene,camera);requestAnimationFrame(frame);
  }
  document.querySelector("#restart").disabled=false;document.querySelector("#restart").onclick=reset;
  document.querySelector("#pause").disabled=false;document.querySelector("#pause").onclick=()=>{paused=!paused;document.querySelector("#pause").textContent=paused?"继续":"暂停";};
  document.querySelector("#wire").onchange=e=>wireObjects.forEach(o=>o.visible=e.target.checked);
  addEventListener("resize",()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
  window.graspBench={model,data,mujoco:mj,reset,setPaused:value=>{paused=value;document.querySelector("#pause").textContent=paused?"继续":"暂停";},get evidence(){return {steps,time:data.time,maxPenetration:maxPen,maxPayloadExternal:maxExternal,contact,position:Array.from(data.qpos.slice(p,p+3))};}};
  reset();requestAnimationFrame(frame);
}
start().catch(error=>{state.textContent="加载失败："+error.message;console.error(error);});
