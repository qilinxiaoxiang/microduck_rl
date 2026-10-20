import {JOINT_NAMES,DEFAULT_POSE,CTRL_DT} from "./constants.js";
import parts from "./beak-colliders.js";
import {readPayloadContacts,createContactWindow,addContactSample,averageContactWindow} from "./contact-observer.js";
import {applyJawSafety,JAW_OPEN_THRESHOLD_RAD,POSE_JOINTS} from "./jaw-safety.js";
// Shared CPU/browser state and physics path. Only inference is injected.
export function createWholeBodyCore(mj,m,d,{infer}) {
  const id=(type,n)=>mj.mj_name2id(m,mj.mjtObj[`mjOBJ_${type}`].value,n);
  const joints=JOINT_NAMES.map(n=>id("JOINT",n));
  const qa=joints.map(j=>m.jnt_qposadr[j]),va=joints.map(j=>m.jnt_dofadr[j]);
  const trunk=id("BODY","trunk_base"),head=id("BODY","jaw_soft");
  const robotGeoms=new Set();
  for(let g=0;g<m.ngeom;g++){let b=m.geom_bodyid[g];while(b>0&&b!==trunk)b=m.body_parentid[b];if(b===trunk)robotGeoms.add(g);}
  const robotAudit={robotGeoms,feetGeoms:new Set([id("GEOM","left_foot_collision"),id("GEOM","right_foot_collision")])};
  robotAudit.head=head;
  robotAudit.mouthRimGeoms=new Set(["bottom_head_shell","top_head_shell"].flatMap(source=>parts[source].parts.map((_,i)=>id("GEOM",`beak_${source}_contact_${i}`))));
  // Shared jaw/head-pitch limits act at the last write before mj_step (jaw-safety.js).
  const headPitch=JOINT_NAMES.indexOf("head_pitch"),headPitchActuator=Array.from({length:m.nu},(_,a)=>a).find(a=>m.actuator_trnid[2*a]===joints[headPitch]);
  if(headPitchActuator!==headPitch)throw Error("Controllers assume ctrl[6] drives head_pitch");
  const headPitchKp=m.actuator_gainprm[headPitchActuator*10];
  // Jaw pose cap: neck_pitch, head_pitch, head_yaw and head_roll position actuators share their joint's index.
  const headPose=Object.fromEntries(POSE_JOINTS.map(n=>[n,JOINT_NAMES.indexOf(n)]));
  for(const [n,j] of Object.entries(headPose))if(m.actuator_trnid[2*j]!==joints[j])throw Error(`Controllers assume ctrl[${j}] drives ${n}`);
  const headPoseKp=Object.fromEntries(Object.entries(headPose).map(([n,j])=>[n,m.actuator_gainprm[j*10]]));
  const beakJoint=id("JOINT","beak_joint"),jaw=m.jnt_qposadr[beakJoint],payloadJoint=id("JOINT","payload_freejoint"),p=payloadJoint>=0?m.jnt_qposadr[payloadJoint]:null;
  const payload=[id("GEOM","payload_geom")].filter(g=>g>=0);
  const payloadFlexId=id("FLEX","payload_flex"),payloadBody=id("BODY","payload");
  const clothNodes=payloadFlexId>=0&&p===null?Array.from(m.flex_vertbodyid.subarray(m.flex_vertadr[payloadFlexId],m.flex_vertadr[payloadFlexId]+m.flex_vertnum[payloadFlexId])):[];
  // WASM 3.11 does not bind boolean memory views (jnt_limited). Our generated
  // unlimited sliders have zero/zero ranges; a limited zero range cannot compile.
  const freeCloth=clothNodes.length>0&&clothNodes.every(b=>m.body_jntnum[b]===3&&[0,1,2].every(k=>{const j=m.body_jntadr[b]+k;return m.jnt_type[j]===mj.mjtJoint.mjJNT_SLIDE.value&&m.jnt_range[2*j]===0&&m.jnt_range[2*j+1]===0;}));
  const clothAxisEnds=clothNodes.length?(()=>{
    const xs=clothNodes.map(b=>m.body_pos[b*3]),lo=Math.min(...xs),hi=Math.max(...xs);
    return [lo,hi].map(end=>xs.flatMap((x,i)=>Math.abs(x-end)<1e-8?[i]:[]));
  })():null;
  function payloadPose(){
    if(p!==null)return Array.from(d.qpos.subarray(p,p+7));
    const start=m.flex_vertadr[payloadFlexId]*3,point=Array.from(d.flexvert_xpos.subarray(start,start+3));
    // Cloth has no rigid-body quaternion. For ground approach only, estimate
    // its planar heading from the mean toe and cuff positions. Keep tracking
    // the same material point, rather than silently substituting a moving COM.
    const ends=clothAxisEnds.map(indices=>[0,1].map(axis=>indices.reduce((sum,i)=>sum+d.flexvert_xpos[start+3*i+axis],0)/indices.length));
    const yaw=Math.atan2(ends[1][1]-ends[0][1],ends[1][0]-ends[0][0]);
    return [...point,Math.cos(yaw/2),0,0,Math.sin(yaw/2)];
  }
  const radialNodes=payloadFlexId<0||m.flex_dim[payloadFlexId]!==3?[]:Array.from(m.flex_vertbodyid.subarray(m.flex_vertadr[payloadFlexId],m.flex_vertadr[payloadFlexId]+m.flex_vertnum[payloadFlexId])).filter(b=>b!==payloadBody&&m.body_jntnum[b]===1).map(b=>({q:m.jnt_qposadr[m.body_jntadr[b]],radius:Math.hypot(...m.body_pos.subarray(b*3,b*3+3))}));
  const lower=["jaw","jaw_soft"].flatMap(source=>parts[source].parts.map((_,i)=>id("GEOM",`beak_${source}_contact_${i}`)));
  const upper=parts.soft_mouth_top.parts.map((_,i)=>id("GEOM",`beak_soft_mouth_top_contact_${i}`));
  const beakGeoms=[...lower,...upper,...robotAudit.mouthRimGeoms];
  const gyro=m.sensor_adr[id("SENSOR","imu_ang_vel")],scratch=new mj.DoubleBuffer(6);
  const jacPosition=new mj.DoubleBuffer(3*m.nv),jacRotation=new mj.DoubleBuffer(3*m.nv);
  // Unlike data.contact, data.warning is bound with reference() and belongs
  // to MjData. Keep one borrowed view; deleting the vector corrupts MjData.
  const warningView=d.warning;
  const obs=new Float32Array(61),lastAction=new Float32Array(14);
  const sites=Object.fromEntries(["mouth_tip","beak_upper_grip","beak_lower_grip"].map(n=>[n,id("SITE",n)]));
  const summary={};
  const jointRanges=Array.from(m.jnt_range); // Immutable model limits: avoid repeated WASM view creation.
  const modelAudit={freePayload:freeCloth||m.jnt_type[payloadJoint]===mj.mjtJoint.mjJNT_FREE.value,
    freeBase:m.jnt_type[m.body_jntadr[trunk]]===mj.mjtJoint.mjJNT_FREE.value,
    equalities:m.neq,mocapBodies:m.nmocap,payloadMass:m.body_subtreemass[payloadBody],
    deformable:payloadFlexId>=0,flexVertices:payloadFlexId>=0?m.flex_vertnum[payloadFlexId]:0,
    pedestalPresent:id("GEOM","payload_pedestal_geom")>=0,terrainPresent:id("GEOM","terrain")>=0,
    floorHeight:m.geom_pos[id("GEOM","floor")*3+2],
    flexAllNodesFree:payloadFlexId>=0&&Array.from(m.flex_vertbodyid.subarray(m.flex_vertadr[payloadFlexId],m.flex_vertadr[payloadFlexId]+m.flex_vertnum[payloadFlexId])).every(b=>{while(b>0&&b!==payloadBody)b=m.body_parentid[b];return b===payloadBody;}),
    jawTorqueLimit:Array.from(m.actuator_forcerange.subarray(28,30)),timestep:m.opt.timestep};
  function reset(payloadPose,{rootOffset}={}) {
    if(rootOffset&&(!Array.isArray(rootOffset)||rootOffset.length<2||rootOffset.length>3||!rootOffset.every(Number.isFinite)))throw Error('Invalid initial root offset');
    mj.mj_resetDataKeyframe(m,d,id("KEY","STAND"));lastAction.fill(0);
    if(rootOffset){d.qpos[0]+=rootOffset[0];d.qpos[1]+=rootOffset[1];const yaw=rootOffset[2]??0;d.qpos.set([Math.cos(yaw/2),0,0,Math.sin(yaw/2)],3);}
    if(payloadPose&&p!==null)d.qpos.set(payloadPose,p);
    if(payloadPose&&freeCloth){
      const [x,y,z,w,qx,qy,qz]=payloadPose;
      if(Math.abs(qx)+Math.abs(qy)>1e-9)throw Error('Cloth reset supports yaw only');
      const co=1-2*qz*qz,si=2*w*qz;
      for(const b of clothNodes){const ji=m.body_jntadr[b],bx=m.body_pos[b*3],by=m.body_pos[b*3+1];
        d.qpos[m.jnt_qposadr[ji]]=x-m.body_pos[payloadBody*3]+co*bx-si*by-bx;
        d.qpos[m.jnt_qposadr[ji+1]]=y-m.body_pos[payloadBody*3+1]+si*bx+co*by-by;
        d.qpos[m.jnt_qposadr[ji+2]]=z-m.body_pos[payloadBody*3+2];
      }
    }
    mj.mj_forward(m,d);
    Object.assign(summary,{maxPenetration:0,maxExternal:0,maxRobotPenetration:0,nonFootSupport:false,maxNonFootForce:0,nonFootImpulse:0,nonFootSeconds:0,maxZ:-Infinity,steps:0,finite:true,solverReset:false,resetAt:null,warningCounts:{},maxJointLimitExcess:0,maxTargetClamp:0,minRadialScale:1,jawCapTicks:0,headPitchInterlockTicks:0,maxHeadPitchJawOpen:null,jawPoseCapTicks:0,jawClosedZoneTicks:0,jawClosedZoneFirstPoseDeg:null,minJawPoseCapDeg:null});
  }
  function observation(command=new Float32Array(13)) {
    obs.set(d.sensordata.subarray(gyro,gyro+3),0);
    const [w,x,y,z]=d.xquat.subarray(trunk*4,trunk*4+4);
    obs[3]=2*(w*y-x*z);obs[4]=-2*(w*x+y*z);obs[5]=2*(x*x+y*y)-1;
    for(let i=0;i<14;i++){obs[6+i]=d.qpos[qa[i]]-DEFAULT_POSE[i];obs[20+i]=d.qvel[va[i]];}
    obs.set(lastAction,34);obs.set(command,48);return obs;
  }
  async function policy(name,command) {
    lastAction.set(await infer(name,observation(command)));
    for(let i=0;i<14;i++)d.ctrl[i]=DEFAULT_POSE[i]+lastAction[i];
  }
  function applyTask(out) {
    for(let h=0;h<4;h++)d.ctrl[5+h]=out.head[h]+d.qfrc_bias[va[5+h]]/.55;
    d.ctrl[4]-=out.ankleLean;d.ctrl[13]+=out.ankleLean;
    for(let i=0;i<14;i++){
      const j=joints[i],raw=d.ctrl[i],bounded=Math.max(m.jnt_range[j*2],Math.min(m.jnt_range[j*2+1],raw));
      if(!Number.isFinite(raw))throw Error("Non-finite actuator command");
      summary.maxTargetClamp=Math.max(summary.maxTargetClamp,Math.abs(raw-bounded));d.ctrl[i]=bounded;
    }
  }
  function pointJacobian(point,indices=[5,6,7,8]) {
    return indices.map(j=>{
      const a=d.xaxis.subarray(joints[j]*3,joints[j]*3+3),o=d.xanchor.subarray(joints[j]*3,joints[j]*3+3),v=point.map((x,i)=>x-o[i]);
      return [a[1]*v[2]-a[2]*v[1],a[2]*v[0]-a[0]*v[2],a[0]*v[1]-a[1]*v[0]];
    });
  }
  function jointAxes(indices=[5,6,7,8]){return indices.map(j=>Array.from(d.xaxis.subarray(joints[j]*3,joints[j]*3+3)));}
  function stanceKinematics(point){
    const dofs=[0,1,2,3,4,5,...va];
    const rows=buffer=>Array.from({length:3},(_,i)=>dofs.map(j=>buffer.GetView()[i*m.nv+j]));
    mj.mj_jac(m,d,jacPosition,jacRotation,point,head);
    const grip={position:rows(jacPosition),rotation:rows(jacRotation)};
    const feet=Array.from(robotAudit.feetGeoms,g=>{
      mj.mj_jacGeom(m,d,jacPosition,jacRotation,g);
      return {point:Array.from(d.geom_xpos.subarray(3*g,3*g+3)),orientation:Array.from(d.geom_xmat.subarray(9*g,9*g+9)),position:rows(jacPosition),rotation:rows(jacRotation)};
    });
    mj.mj_jacSubtreeCom(m,d,jacPosition,trunk);
    return {grip,feet,com:{point:Array.from(d.subtree_com.subarray(3*trunk,3*trunk+3)),position:rows(jacPosition)}};
  }
  function beakClearance(){
    let height=Infinity,point=null,geom=null;
    for(const g of beakGeoms){
      const mesh=m.geom_dataid[g],start=m.mesh_vertadr[mesh],count=m.mesh_vertnum[mesh],R=d.geom_xmat.subarray(g*9,g*9+9),p=d.geom_xpos.subarray(g*3,g*3+3);
      for(let i=0;i<count;i++){
        const v=m.mesh_vert.subarray((start+i)*3,(start+i)*3+3),z=p[2]+R[6]*v[0]+R[7]*v[1]+R[8]*v[2];
        if(z<height){height=z;point=Array.from(p,(x,k)=>x+R[k*3]*v[0]+R[k*3+1]*v[1]+R[k*3+2]*v[2]);geom=g;}
      }
    }
    return {height,point,geom};
  }
  function upperSurfaceGap(){
    if(payload[0]<0)return null;
    let nearest=null;
    for(const geom of upper){
      const distance=mj.mj_geomDistance(m,d,geom,payload[0],1,scratch);
      if(!nearest||distance<nearest.distance){const points=Array.from(scratch.GetView());nearest={distance,point:points.slice(0,3),objectPoint:points.slice(3,6),geom};}
    }
    return nearest;
  }
  // Same substeps and audits in both paths. Browser cloth can yield between
  // small batches so buttons/rendering remain responsive without changing dt.
  function* stepBatches(jawTarget=.32) {
    d.ctrl[14]=jawTarget;
    const pose={indices:headPose,kp:headPoseKp,bias:{},measured:{}};
    for(const [n,j] of Object.entries(headPose)){pose.bias[n]=d.qfrc_bias[va[j]];pose.measured[n]=d.qpos[qa[j]];}
    const limited=applyJawSafety(d.ctrl,{jawIndex:14,headIndex:headPitchActuator,measuredJaw:d.qpos[jaw],headBias:d.qfrc_bias[va[headPitch]],kp:headPitchKp,pose});
    summary.jawCapTicks+=limited.jawCapped;summary.headPitchInterlockTicks+=limited.headCapped;
    summary.jawPoseCapTicks+=limited.jawPoseCapped;summary.jawClosedZoneTicks+=limited.jawClosedZone;
    summary.minJawPoseCapDeg=Math.min(summary.minJawPoseCapDeg??Infinity,limited.jawCapRad*180/Math.PI);
    if(limited.jawClosedZone&&!summary.jawClosedZoneFirstPoseDeg)summary.jawClosedZoneFirstPoseDeg=Object.fromEntries(Object.entries(pose.measured).map(([n,v])=>[n,v*180/Math.PI]));
    for(let j=0;j<14;j++)lastAction[j]=d.ctrl[j]-DEFAULT_POSE[j];
    const window=createContactWindow(),decimation=Math.round(CTRL_DT/m.opt.timestep);
    // Applied-force arrays are inputs: mj_step never changes them. Audit every
    // controller write boundary, rather than rescanning the same input 320 times.
    for(const f of d.xfrc_applied)summary.maxExternal=Math.max(summary.maxExternal,Math.abs(f));
    for(const f of d.qfrc_applied)summary.maxExternal=Math.max(summary.maxExternal,Math.abs(f));
    for(let k=0;k<decimation;k++) {
      const beforeTime=d.time;mj.mj_step(m,d);
      // MuJoCo can reset data after unstable acceleration, leaving finite
      // coordinates again. Never mistake that automatic reset for a valid run.
      if(d.time<beforeTime+m.opt.timestep*.5){summary.finite=false;summary.solverReset=true;summary.resetAt=beforeTime;break;}
      const c=readPayloadContacts(mj,m,d,payload,lower,upper,scratch,robotAudit,payloadFlexId);
      addContactSample(window,c);
      summary.maxPenetration=Math.max(summary.maxPenetration,c.maxPenetration);
      summary.maxRobotPenetration=Math.max(summary.maxRobotPenetration,c.robotMaxPenetration);
      summary.nonFootSupport||=c.nonFootSupport;
      summary.maxNonFootForce=Math.max(summary.maxNonFootForce,c.nonFootForce);
      summary.nonFootImpulse+=c.nonFootForce*m.opt.timestep;
      if(c.nonFootSupport)summary.nonFootSeconds+=m.opt.timestep;
      summary.maxZ=Math.max(summary.maxZ,p!==null?d.qpos[p+2]:d.flexvert_xpos[m.flex_vertadr[payloadFlexId]*3+2]);summary.steps++;
      const qpos=d.qpos; // Refresh after contact auditing, which can grow WASM memory.
      for(const q of qpos)if(!Number.isFinite(q))summary.finite=false;
      for(const node of radialNodes)summary.minRadialScale=Math.min(summary.minRadialScale,1+qpos[node.q]/node.radius);
      for(let i=0;i<14;i++){const j=joints[i],q=qpos[qa[i]];summary.maxJointLimitExcess=Math.max(summary.maxJointLimitExcess,jointRanges[2*j]-q,q-jointRanges[2*j+1]);}
      summary.maxJointLimitExcess=Math.max(summary.maxJointLimitExcess,jointRanges[2*beakJoint]-qpos[jaw],qpos[jaw]-jointRanges[2*beakJoint+1]);
      if(qpos[jaw]>JAW_OPEN_THRESHOLD_RAD)summary.maxHeadPitchJawOpen=Math.max(summary.maxHeadPitchJawOpen??-Infinity,qpos[qa[headPitch]]);
      if(k%8===7&&k+1<decimation)yield;
    }
    for(let i=0;i<warningView.size();i++){const w=warningView.get(i);try{if(w.number){summary.warningCounts[i]=Math.max(summary.warningCounts[i]??0,w.number);if(i>=mj.mjtWarning.mjWARN_BADQPOS.value)summary.finite=false;}}finally{w.delete();}}
    return averageContactWindow(window);
  }
  function step(jawTarget=.32){const batches=stepBatches(jawTarget);let next;do{next=batches.next();}while(!next.done);return next.value;}
  async function stepAsync(jawTarget=.32){const batches=stepBatches(jawTarget);let next;do{next=batches.next();if(!next.done)await new Promise(resolve=>setTimeout(resolve,0));}while(!next.done);return next.value;}
  function snapshot(contact) {
    const hr=d.xmat.subarray(head*9,head*9+9),hp=d.xpos.subarray(head*3,head*3+3);
    const graspPoint=Array.from(hp,(v,i)=>v-hr[i*3]*.016-hr[i*3+2]*.075);
    const headJac=[5,6].map(j=>{const a=d.xaxis.subarray(joints[j]*3,joints[j]*3+3),o=d.xanchor.subarray(joints[j]*3,joints[j]*3+3),v=graspPoint.map((x,i)=>x-o[i]);return [a[1]*v[2]-a[2]*v[1],a[0]*v[1]-a[1]*v[0]];});
    let shape;
    if(payloadFlexId>=0){
      const start=m.flex_vertadr[payloadFlexId],count=m.flex_vertnum[payloadFlexId],pr=d.xmat.subarray(payloadBody*9,payloadBody*9+9),pp=d.xpos.subarray(payloadBody*3,payloadBody*3+3);
      const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];let minZ=Infinity,maxZ=-Infinity,maxRadialDisplacement=0,maxNodeDisplacement=0;
      for(let i=0;i<count;i++){
        const v=Array.from(d.flexvert_xpos.subarray((start+i)*3,(start+i)*3+3)),delta=v.map((x,j)=>x-pp[j]);
        minZ=Math.min(minZ,v[2]);maxZ=Math.max(maxZ,v[2]);
        for(let j=0;j<3;j++){const local=delta[0]*pr[j]+delta[1]*pr[3+j]+delta[2]*pr[6+j];min[j]=Math.min(min[j],local);max[j]=Math.max(max[j],local);}
        const body=m.flex_vertbodyid[start+i],ji=m.body_jntadr[body];
        if(body!==payloadBody&&ji>=0){
          if(clothNodes.length)maxNodeDisplacement=Math.max(maxNodeDisplacement,Math.hypot(...[0,1,2].map(k=>d.qpos[m.jnt_qposadr[ji+k]])));
          else maxRadialDisplacement=Math.max(maxRadialDisplacement,Math.abs(d.qpos[m.jnt_qposadr[ji]]));
        }
      }
      shape={diameters:max.map((v,j)=>v-min[j]),minZ:minZ-m.flex_radius[payloadFlexId],maxZ:maxZ+m.flex_radius[payloadFlexId],...(clothNodes.length?{maxNodeDisplacement}:{maxRadialDisplacement})};
    }else if(payload[0]>=0){
      const g=payload[0],size=m.geom_size.subarray(g*3,g*3+3),mat=d.geom_xmat.subarray(g*9,g*9+9),center=d.geom_xpos.subarray(g*3,g*3+3),type=m.geom_type[g];
      const half=type===mj.mjtGeom.mjGEOM_CAPSULE.value?size[0]+Math.abs(mat[8])*size[1]:type===mj.mjtGeom.mjGEOM_SPHERE.value?size[0]:Math.abs(mat[6])*size[0]+Math.abs(mat[7])*size[1]+Math.abs(mat[8])*size[2];
      const diameters=type===mj.mjtGeom.mjGEOM_CAPSULE.value?[2*size[0],2*size[0],2*(size[0]+size[1])]:Array.from(size,v=>v*2);
      shape={minZ:center[2]-half,maxZ:center[2]+half,diameters};
    }
    return {time:d.time,root:Array.from(d.qpos.subarray(0,7)),payload:payloadPose(),...(shape?{shape}:{}),
      ...(clothNodes.length?{clothPoints:clothNodes.map((_,i)=>Array.from(d.flexvert_xpos.subarray((m.flex_vertadr[payloadFlexId]+i)*3,(m.flex_vertadr[payloadFlexId]+i)*3+3)))}:{}),
      graspPoint,headJac,joint:Array.from(qa,q=>d.qpos[q]),
      ctrl:Array.from(d.ctrl),jaw:d.qpos[jaw],contact,
      sites:Object.fromEntries(Object.entries(sites).map(([n,i])=>[n,Array.from(d.site_xpos.subarray(i*3,i*3+3))])),
      headQuat:Array.from(d.xquat.subarray(head*4,head*4+4))};
  }
  reset();
  return {m,d,mj,p,jaw,qa,va,sites,head,trunk,headPitchKp,payloadFlexId,summary,modelAudit,reset,policy,applyTask,pointJacobian,jointAxes,stanceKinematics,beakClearance,upperSurfaceGap,step,stepAsync,snapshot,observation,dispose(){scratch.delete();jacPosition.delete();jacRotation.delete();}};
}
