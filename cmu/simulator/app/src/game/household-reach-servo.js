// Experimental motor-only Cartesian feedback. Kinematics supplies the four
// head columns; the paired-knee column is a measured local stance response.
// This does not change qpos, apply an external force, or attach the object.
import {householdHeading} from './household-grasp-target.js';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function solve(a,b){
  const n=b.length;
  const rows=a.map((r,i)=>[...r,b[i]]);
  for(let i=0;i<n;i++){
    let pivot=i;for(let j=i+1;j<n;j++)if(Math.abs(rows[j][i])>Math.abs(rows[pivot][i]))pivot=j;
    [rows[i],rows[pivot]]=[rows[pivot],rows[i]];
    const d=rows[i][i];if(Math.abs(d)<1e-12)return Array(n).fill(0);
    for(let k=i;k<=n;k++)rows[i][k]/=d;
    for(let j=0;j<n;j++)if(j!==i){const v=rows[j][i];for(let k=i;k<=n;k++)rows[j][k]-=v*rows[i][k];}
  }
  return rows.map(r=>r[n]);
}
export class HouseholdReachServo{
  constructor(c){this.c=c;this.offset=[0,0,0,0,0];this.stable=0;}
  apply(r,s){
    const [w,x,y,z]=s.headQuat,R=[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];
    const local=this.c.gripOffset??[-.001,-.05,.003];
    const point=s.graspPoint.map((p,i)=>p+local.reduce((sum,v,j)=>sum+v*R[i*3+j],0));
    const target=s.payload.slice(0,3),error=target.map((p,i)=>p-point[i]);
    const yaw=Math.atan2(2*(s.root[3]*s.root[6]+s.root[4]*s.root[5]),1-2*(s.root[5]**2+s.root[6]**2));
    let columns=[...r.pointJacobian(point),[.10*Math.cos(yaw),.10*Math.sin(yaw),-.10]];
    const velocity=error.map(e=>clamp(e*(this.c.reachGain??1.5),-.008,.008));
    if(this.c.poseFeedback){
      const roll=this.c.servoRoll??.076,pitch=this.c.servoPitch??-.098;
      const cr=Math.cos(roll),sr=Math.sin(roll),cp=Math.cos(pitch),sp=Math.sin(pitch);
      const objectYaw=this.heading=householdHeading(s.payload,this.c.kind,this.heading??yaw);
      const cy=Math.cos(objectYaw+.036),sy=Math.sin(objectYaw+.036);
      const goal=[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr];
      const skew=[0,0,0];for(let j=0;j<3;j++){const a=[R[j],R[3+j],R[6+j]],b=[goal[j],goal[3+j],goal[6+j]];skew[0]+=(a[1]*b[2]-a[2]*b[1])*.5;skew[1]+=(a[2]*b[0]-a[0]*b[2])*.5;skew[2]+=(a[0]*b[1]-a[1]*b[0])*.5;}
      const length=this.c.orientationWeight??.05,axes=[...r.jointAxes(),[0,0,0]];
      columns=columns.map((col,k)=>[...col,...axes[k].map(v=>v*length)]);
      velocity.push(...skew.map(v=>clamp(v*1.5,-.2,.2)*length));this.orientationError=skew;
    }
    if(this.c.clearanceFeedback){
      this.clearance=r.beakClearance();
      if(this.clearance.height<.003){
        const floorColumns=[...r.pointJacobian(this.clearance.point).map(col=>col[2]),-.10];
        columns=columns.map((col,k)=>[...col,3*floorColumns[k]]);
        velocity.push(3*clamp((.001-this.clearance.height)*3,-.001,.008));
      }
    }
    // Damped least squares avoids explosive commands near a straight neck.
    const lambda=.025,dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
    const inactive=new Set();let change;
    for(let iteration=0;iteration<6;iteration++){
      const usable=columns.map((col,k)=>inactive.has(k)?col.map(()=>0):col);
      const matrix=usable.map((a,i)=>usable.map((b,j)=>dot(a,b)+(i===j?lambda**2:0)));
      change=solve(matrix,usable.map(col=>dot(col,velocity)));
      if(!this.c.activeSet)break;
      let changed=false;
      for(let k=0;k<5;k++){
        if(inactive.has(k))continue;
        let blocked=false;
        if(k<4){const j=5+k,ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]),q=s.joint[j];blocked=q<r.m.jnt_range[2*ji]+.05&&change[k]<0||q>r.m.jnt_range[2*ji+1]-.05&&change[k]>0;}
        else blocked=this.offset[k]>=.3499&&change[k]>0||this.offset[k]<=-.1499&&change[k]<0;
        if(blocked){inactive.add(k);changed=true;}
      }
      if(!changed)break;
    }
    for(let k=0;k<5;k++){
      let dq=clamp(change[k],-.15,.15)*.02;
      if(k<4){const j=5+k,ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]),q=s.joint[j];if(q<r.m.jnt_range[2*ji]+.05&&dq<0||q>r.m.jnt_range[2*ji+1]-.05&&dq>0)dq=0;}
      this.offset[k]=clamp(this.offset[k]+dq,k===4?-.15:-.5,k===4?.35:.5);
    }
    for(let k=0;k<4;k++)r.d.ctrl[5+k]+=this.offset[k];
    r.d.ctrl[3]+=this.offset[4];r.d.ctrl[12]-=this.offset[4];
    this.error=error;this.point=point;this.stable=Math.hypot(...error)<(this.c.readyTolerance??.0015)?this.stable+.02:0;
    return this.stable>=.2;
  }
}
