// Experimental planner interface. Predictions live in separate simulation
// data; this controller still writes only the real robot's motor targets.
import {closestPenBite} from './household-grasp-target.js';
export class HouseholdPredictiveServo{
  constructor(c){this.c=c;this.offset=Array(14).fill(0);this.stable=0;}
  apply(r,s){
    const R=Array.from(r.d.xmat.subarray(r.head*9,r.head*9+9)),local=this.c.gripOffset??[-.001,-.05,.003];
    this.point=s.graspPoint.map((p,i)=>p+local.reduce((sum,v,j)=>sum+v*R[3*i+j],0));
    const target=s.taskGraspPoint?.slice()??(this.c.axisGrasp&&this.c.kind==='pen'?closestPenBite(s.payload,this.point,this.c.length):s.payload.slice(0,3));
    this.error=target.map((v,i)=>v-this.point[i]);this.clearance=r.beakClearance();
    const plan=r.predictedReachPlan?.();
    if(!plan)throw Error('Predictive reach requires an isolated dynamics planner');
    if(plan.ctrl.length!==14||!plan.ctrl.every(Number.isFinite))throw Error('Invalid predictive motor targets');
    this.ctrl=plan.ctrl.slice();this.plan=plan.audit;this.blocked=Boolean(plan.blocked);
    if(this.blocked){this.stable=0;return false;}
    this.hold(r);
    // A predicted pinch does not by itself establish a deep enough grasp for
    // lifting. Keep the geometric requirement as well as the closure forecast.
    const ready=Math.hypot(...this.error)<(this.c.readyTolerance??.002)&&(!this.c.graspForecast||plan.graspReady===true);
    this.stable=ready&&this.clearance.height>.001?this.stable+.02:0;
    return this.stable>=.2;
  }
  hold(r){for(let j=0;j<14;j++)r.d.ctrl[j]=this.ctrl[j];}
}
