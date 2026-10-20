// Shared training/browser contract. A residual is a small correction to the
// existing motor targets, not a pose teleport, force input, or jaw attachment.
import {householdHeading} from './household-grasp-target.js';
export const RESIDUAL_PHASES=['reach','clamp','lift','hold','carried','release','done','failed'];
export const RESIDUAL_SCALES=Array.from({length:14},(_,j)=>j>=5&&j<=8?.30:.25);
export const RESIDUAL_OBS_SIZE=93;
export const RESIDUAL_CONTRACT={version:2,observationVersion:'shaft-heading-v2',observationSize:93,actionSize:14,decisionHz:10,prefixTime:3.0,startTime:3.2,scales:RESIDUAL_SCALES};
const clip=(x,a=-5,b=5)=>Math.max(a,Math.min(b,x));
export function validateResidualAction(action){
  if(!action||action.length!==14||!Array.from(action).every(Number.isFinite))throw Error('Expected 14 finite residual actions');
}
export function residualObservation(s,prefix,headOrientation,ctl,last=Array(14).fill(0)){
  const inner=ctl.inner??ctl,contact=s.contact??{},R=headOrientation;
  const relative=s.payload.slice(0,3).map((v,i)=>v-s.graspPoint[i]);
  const headLocal=Array.from({length:3},(_,j)=>relative.reduce((sum,v,i)=>sum+R[3*i+j]*v,0));
  const q=s.root.slice(3),yaw=Math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2));
  const objectYaw=householdHeading(s.payload,inner.c?.kind??ctl.c?.kind,yaw);
  const dx=s.payload[0]-s.root[0],dy=s.payload[1]-s.root[1];
  const observation=[...prefix,...headLocal.map(v=>v*20),
    (dx*Math.cos(yaw)+dy*Math.sin(yaw))*5,(-dx*Math.sin(yaw)+dy*Math.cos(yaw))*5,(s.payload[2]-s.root[2])*5,
    Math.sin(objectYaw-yaw),Math.cos(objectYaw-yaw),...R,
    (contact.lowerForce??0)*2,(contact.upperForce??0)*2,(contact.feetForce??0)/8,(contact.floorForce??0)*10,
    ...RESIDUAL_PHASES.map(phase=>Number(inner.phase===phase)),(inner.age??0)/10,s.shape.minZ*20,...last];
  if(observation.length!==RESIDUAL_OBS_SIZE||!observation.every(Number.isFinite))throw Error('Invalid household residual observation');
  return observation.map(v=>clip(v));
}
export class ReachResidual {
  constructor(){this.value=Array(14).fill(0);}
  apply(r,ctl,action){
    validateResidualAction(action);
    const inner=ctl.inner??ctl;
    if(!['reach','clamp'].includes(inner.phase)||(inner.time??0)<RESIDUAL_CONTRACT.startTime){this.value.fill(0);return;}
    for(let j=0;j<14;j++){
      this.value[j]+=.12*(clip(action[j],-1,1)-this.value[j]);
      const ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]),margin=.04;
      r.d.ctrl[j]=clip(r.d.ctrl[j]+this.value[j]*RESIDUAL_SCALES[j],r.m.jnt_range[2*ji]+margin,r.m.jnt_range[2*ji+1]-margin);
    }
  }
}
