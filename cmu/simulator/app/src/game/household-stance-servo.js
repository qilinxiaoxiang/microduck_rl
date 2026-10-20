import {solve} from './household-reach-servo.js';
import {closestPenBite,householdHeading} from './household-grasp-target.js';
import {rootAngles} from './table-pick-controller.js';
import {supportStatics} from './household-support-statics.js';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function rotationError(R,goal){
  const skew=[0,0,0];
  for(let j=0;j<3;j++){
    const a=[R[j],R[3+j],R[6+j]],b=[goal[j],goal[3+j],goal[6+j]];
    skew[0]+=(a[1]*b[2]-a[2]*b[1])*.5;
    skew[1]+=(a[2]*b[0]-a[0]*b[2])*.5;
    skew[2]+=(a[0]*b[1]-a[1]*b[0])*.5;
  }
  return skew;
}
export function solveWithLowerBound(matrix,rhs,normal=null,lower=0){
  const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
  let change=solve(matrix,rhs);
  if(normal){
    const violation=lower-dot(normal,change);
    if(violation>0){
      const response=solve(matrix,normal),denominator=dot(normal,response);
      if(denominator>1e-10)change=change.map((v,j)=>v+response[j]*violation/denominator);
    }
  }
  return change;
}
// Differential whole-body inverse kinematics. Floating-base coordinates are
// optimization variables ONLY: they are never written back to the simulator.
// Feet and COM constraints couple leg motor increments to the head task.
export class HouseholdStanceServo{
  constructor(c){this.c=c;this.stable=0;this.offset=Array(14).fill(0);}
  apply(r,s,{target,orientation}={}){
    const R=Array.from(r.d.xmat.subarray(r.head*9,r.head*9+9));
    const local=this.c.gripOffset??[-.001,-.05,.003];
    this.point=s.graspPoint.map((p,i)=>p+local.reduce((sum,v,j)=>sum+v*R[i*3+j],0));
    const objectTarget=s.taskGraspPoint?.slice()??(this.c.axisGrasp&&this.c.kind==='pen'?closestPenBite(s.payload,this.point,this.c.length):s.payload.slice(0,3));
    if(this.c.stanceApproachClearance&&!target){
      this.approachHeight??=this.c.stanceApproachClearance;
      const horizontal=Math.hypot(objectTarget[0]-this.point[0],objectTarget[1]-this.point[1]);
      // Align above the shaft before descending. The earlier open-loop neck
      // ramp could push the pen with the rigid rim before feedback even began.
      if(horizontal<.002&&Math.abs(objectTarget[2]+this.approachHeight-this.point[2])<.003){
        this.approachHeight=Math.max(0,this.approachHeight-(this.c.stanceApproachSpeed??.003)*.02);
      }
      objectTarget[2]+=this.approachHeight;
    }
    this.error=(target??objectTarget).map((v,i)=>v-this.point[i]);
    const surface=this.c.surfaceSeek&&!target?r.upperSurfaceGap():null;
    if(surface){
      this.point=surface.point;this.surfaceGap=surface.distance;
      const delta=surface.objectPoint.map((v,i)=>v-this.point[i]),length=Math.hypot(...delta);
      this.error=length>1e-9?delta.map(v=>v*(Math.max(0,surface.distance)+.00002)/length):[0,0,-.00002];
    }
    const k=r.stanceKinematics(this.point);
    if(this.c.supportGravityGain)this.updateSupport(r,k);
    this.reference??={feet:k.feet.map(f=>({point:f.point.slice(),orientation:f.orientation.slice()})),com:k.com.point.slice(),ctrl:Array.from(r.d.ctrl),bias:Array.from(r.d.qfrc_bias)};
    const A=[],b=[];
    const add=(rows,error,weight,gain=1.5,limit=.008)=>rows.forEach((row,i)=>{A.push(row.map(v=>v*weight));b.push(clamp(error[i]*gain,-limit,limit)*weight);});
    add(k.grip.position,this.error,1,1.5,this.c.stancePositionSpeed??.008);
    const roll=this.c.servoRoll??.076,pitch=this.c.servoPitch??-.098;
    const cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch);
    this.heading=householdHeading(s.payload,this.c.kind,this.heading??rootAngles(s.root.slice(3)).yaw);
    const yaw=this.heading+.036;
    const cy=Math.cos(yaw),sy=Math.sin(yaw),goal=[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr];
    this.orientationError=rotationError(R,orientation??goal);add(k.grip.rotation,this.orientationError,(orientation?this.c.liftOrientationWeight:undefined)??this.c.stanceOrientationWeight??.05,1.5,.2);
    k.feet.forEach((foot,i)=>{
      const ref=this.reference.feet[i];
      add(foot.position,ref.point.map((v,j)=>v-foot.point[j]),this.c.stanceFootPositionWeight??2,3,.01);
      add(foot.rotation,rotationError(foot.orientation,ref.orientation),this.c.stanceFootRotationWeight??.12,3,.2);
    });
    add(k.com.position.slice(0,2),this.reference.com.slice(0,2).map((v,i)=>v-k.com.point[i]),this.c.stanceComWeight??1,2,.01);
    if(this.c.stanceTrunkRollWeight){
      // MuJoCo free-joint angular velocities are expressed in the local body
      // frame. ZYX Euler roll rate is wx + sin(roll)*tan(pitch)*wy +
      // cos(roll)*tan(pitch)*wz. This is an optimization row, not a base write.
      const a=rootAngles(s.root.slice(3)),row=Array(20).fill(0);
      row[3]=1;row[4]=Math.sin(a.roll)*Math.tan(a.pitch);row[5]=Math.cos(a.roll)*Math.tan(a.pitch);
      add([row],[(this.c.stanceTrunkRollTarget??0)-a.roll],this.c.stanceTrunkRollWeight,1.5,.2);
    }
    this.clearance=r.beakClearance();
    const safeHeight=this.c.safeHeight??.002;
    const floor=this.clearance.height<safeHeight+.003?r.stanceKinematics(this.clearance.point).grip.position[2]:null;
    const floorLower=3*(safeHeight-this.clearance.height);
    const inactive=new Set();let change;
    for(let iteration=0;iteration<15;iteration++){
      const C=Array.from({length:20},(_,j)=>A.map(row=>inactive.has(j)?0:row[j]));
      const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0),lambda=this.c.stanceDamping??.005;
      const matrix=C.map((a,i)=>C.map((b,j)=>dot(a,b)+(i===j?lambda**2:0)));
      // One-sided control barrier: constrain downward velocity, never attract
      // the mouth back to the floor when the requested motion is upward.
      change=solveWithLowerBound(matrix,C.map(col=>dot(col,b)),floor?.map((v,j)=>inactive.has(j)?0:v),floorLower);
      let changed=false;
      for(let j=0;j<14;j++){
        const ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]),q=s.joint[j],dq=change[6+j];
        const target=this.reference.ctrl[j]+this.offset[j]+this.biasCorrection(r,j),margin=this.c.margin??.04;
        if(!inactive.has(6+j)&&((q<r.m.jnt_range[2*ji]+.055||target<=r.m.jnt_range[2*ji]+margin+.0001)&&dq<0||(q>r.m.jnt_range[2*ji+1]-.055||target>=r.m.jnt_range[2*ji+1]-margin-.0001)&&dq>0)){
          inactive.add(6+j);changed=true;
        }
      }
      if(!changed)break;
    }
    const speedScale=Math.max(1,...change.slice(6).map(v=>Math.abs(v)/(this.c.stanceJointSpeed??.3)));
    for(let j=0;j<14;j++){
      // Preserve the task-space direction and one-sided barrier when limiting
      // motor speed; independent clipping can turn an upward vector downward.
      const offsetLimit=this.c.stanceOffsetLimit??.6;
      this.offset[j]=clamp(this.offset[j]+change[6+j]/speedScale*.02,-offsetLimit,offsetLimit);
      const ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]),margin=this.c.margin??.04;
      const bias=this.biasCorrection(r,j);
      r.d.ctrl[j]=clamp(this.reference.ctrl[j]+this.offset[j]+bias,r.m.jnt_range[2*ji]+margin,r.m.jnt_range[2*ji+1]-margin);
      this.offset[j]=r.d.ctrl[j]-this.reference.ctrl[j]-bias;
    }
    const ready=(this.approachHeight??0)===0&&(surface?surface.distance<.0002&&(s.contact?.upperForce??0)>.01:Math.hypot(...this.error)<(this.c.readyTolerance??.002));
    this.stable=ready&&this.clearance.height>safeHeight*.5?this.stable+.02:0;
    return this.stable>=.2;
  }
  updateSupport(r,k){
    const bias=[...Array.from(r.d.qfrc_bias).slice(0,6),...Array.from(r.va,v=>r.d.qfrc_bias[v])];
    this.support=supportStatics(k.feet,bias,this.c.supportMomentLength??.015);
    this.supportReference??=this.support.torques.slice();
  }
  biasCorrection(r,j){
    if(this.c.headGravityFeedback&&j>=5&&j<=8)return (r.d.qfrc_bias[r.va[j]]-this.reference.bias[r.va[j]])/.55;
    // All 14 chosen_actuator servos have kp=0.55 in the source MJCF.
    // Preserve the initial equilibrium; only compensate estimated load change.
    return this.support?clamp(this.c.supportGravityGain*(this.support.torques[j]-this.supportReference[j])/.55,-.2,.2):0;
  }
  hold(r){if(this.c.supportGravityGain)this.updateSupport(r,r.stanceKinematics(this.point));for(let j=0;j<14;j++)r.d.ctrl[j]=this.reference.ctrl[j]+this.offset[j]+this.biasCorrection(r,j);}
}
