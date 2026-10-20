import {rootAngles} from "./table-pick-controller.js";
import {HouseholdReachServo} from './household-reach-servo.js';
import {HouseholdStanceServo} from './household-stance-servo.js';
import {HouseholdPredictiveServo} from './household-predictive-servo.js';
import {limitHeadPitchTarget} from './jaw-safety.js';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
// Only motor targets are written. Success is decided
// separately from this phase machine, using actual contact/shape trajectories.
export class GroundPickController {
  constructor(c={}){
    for(const [key,lo,hi] of [['liftStartPhase',.3,.5],['liftBlendSeconds',.2,3],['clampSettleSeconds',.8,2.5],['stopLiftClearance',.035,.08]])if(c[key]!==undefined&&(!Number.isFinite(c[key])||c[key]<lo||c[key]>hi))throw Error(`Invalid ${key}`);
    Object.assign(this,{phase:"settle",age:0,time:0,hold:0,stable:0,events:[],height:c.z,lift:c.lift??.06,c});
  }
  enter(phase){this.events.push({time:this.time,from:this.phase,to:phase});this.phase=phase;this.age=0;this.stable=0;}
  release(s){const a=rootAngles(s.root.slice(3));if(this.c.dynamicsLift)this.stoppedLiftCtrl=s.ctrl.slice();this.releaseHeight=s.payload[2];this.releaseNeck=s.joint[5];this.releaseTheta=a.pitch-s.joint[5]+s.joint[6]-Math.PI/2;this.enter('release');}
  async actuate(r,input){
    // Motor decisions run at 50 Hz. MuJoCo's accumulated floating clock can
    // fall either side of an exact boundary when the physics timestep changes.
    // Quantize only the controller clock, never physical state or recorded time.
    const s={...input,time:Math.round(input.time/.02)*.02};
    const previousCtrl=Array.from(r.d.ctrl);
    this.time=s.time;this.age+=.02;const c=this.c,a=rootAngles(s.root.slice(3)),contact=s.contact??{};
    if(this.phase==="settle"&&s.time>=1)this.enter(c.cartesianDescent?'reach':"descend");
    if(this.phase==="descend"&&s.time>=2.2)this.enter("reach");
    if(this.phase==='reach'&&c.touchClose&&s.time>=4.4){
      this.touchStable=(contact.upperForce??0)>.01&&!contact.nonFootSupport? (this.touchStable??0)+.02:0;
      if(this.touchStable>=.06)this.reachReady=true;
    }
    if(this.phase==="reach"&&(c.reachFeedback?this.reachReady:s.time>=(c.closeAt??4.8)))this.enter("clamp");
    const opposing=contact.lowerForce>.002&&contact.upperForce>.002;
    if(this.phase==="clamp"){
      this.stable=opposing?this.stable+.02:0;
      if(this.stable>=.4&&this.age>=(c.clampSettleSeconds??.8)){this.graspOrigin=s.payload.slice(0,3);this.lowCtrl=Array.from(r.d.ctrl);this.gripNeck=s.joint[5];this.gripTheta=a.pitch-s.joint[5]+s.joint[6]-Math.PI/2;this.gripYaw=s.joint[7]+a.yaw;this.gripRoll=s.joint[8]-a.roll;this.enter("lift");}
      else if(this.age>3){
        if(c.regrasp&&this.reachServo&&(this.regraspCount??0)<2){this.regraspCount=(this.regraspCount??0)+1;this.reopenUntil=s.time+.7;this.reachReady=false;this.reachServo.stable=0;this.enter('reach');}
        else this.enter("failed");
      }
    }
    // Picking does not require finishing the entire learned standing motion.
    // Once the whole object is clear, retain the attained motor targets and
    // measure whether real contact can sustain it in this seated posture.
    if(this.phase==='lift'&&c.stopLiftClearance!==undefined&&s.shape?.minZ>c.stopLiftClearance&&opposing&&!contact.nonMouthSupport&&!contact.nonFootSupport&&contact.feetForce>1){
      this.stoppedLiftCtrl=previousCtrl.slice();this.enter('hold');
    }
    const liftReady=this.phase==='lift'&&(c.stanceLift?s.shape?.minZ>.035&&opposing:s.payload[2]>this.height+.04);
    if(this.phase==="lift"&&s.time-this.riseStart>=3&&liftReady&&!contact.nonMouthSupport){
      if(c.stanceStandUp){
        this.uprightCtrl=Array.from(r.d.ctrl);this.stanceReleased=true;
        this.gripNeck=s.joint[5];this.gripTheta=a.pitch-s.joint[5]+s.joint[6]-Math.PI/2;this.gripYaw=s.joint[7]+a.yaw;
        this.enter('upright');
      }else this.enter('hold');
    }
    if(this.phase==='upright'){
      this.stable=this.age>(c.standDuration??3)&&Math.abs(a.pitch)<.15&&s.root[2]>.11&&opposing&&s.shape?.minZ>.035?this.stable+.02:0;
      if(this.stable>.4)this.enter('hold');else if(this.age>10)this.enter('failed');
    }
    if(this.phase==="hold"){
      this.hold=opposing&&!contact.nonMouthSupport&&!contact.nonFootSupport&&s.shape?.minZ>.025?this.hold+.02:0;
      if(this.hold>=1.24){if(c.holdUntilRelease){this.carryCtrl=Array.from(r.d.ctrl);this.enter('carried');}else this.release(s);}
      else if(this.age>3)this.enter("failed");
    }
    if(this.phase==="release"){
      const released=this.age>1.5&&s.jaw>.31&&contact.lowerForce<.002&&contact.upperForce<.002&&contact.floorForce>.001&&Math.abs(s.shape.minZ)<.0005&&s.payload[2]<this.releaseHeight-.035;
      this.stable=released?this.stable+.02:0;if(this.stable>=.3)this.enter("done");
    }
    if(this.phase==='carried'){
      this.stable=opposing&&!contact.nonMouthSupport?0:this.stable+.02;
      if(this.stable>.3)this.enter('failed');
      else if(!c.dynamicsLift){r.d.ctrl.set(this.carryCtrl);return 0;}
    }
    if(this.phase==="lift"&&this.age>(c.dynamicsLift?20:c.stanceLift?14:5)||this.time>16&&!c.holdUntilRelease)this.enter("failed");
    // A failed transition must not fall through to a different body policy.
    // Preserve the last motor targets and open the jaw; never reposition bodies.
    if(this.phase==='failed'){r.d.ctrl.set(previousCtrl);return .32;}
    if(c.dynamicsLift&&['lift','hold','carried'].includes(this.phase)){
      const plan=r.predictedReachPlan?.();
      if(plan?.stage==='transport'){
        this.reachServo.apply(r,s);
        if(this.reachServo.blocked){this.enter('failed');r.d.ctrl.set(previousCtrl);return .32;}
      }else r.d.ctrl.set(previousCtrl); // First phase-boundary tick, before the next prediction.
      return 0;
    }
    if(this.stoppedLiftCtrl&&['hold','release','done'].includes(this.phase)){
      r.d.ctrl.set(this.stoppedLiftCtrl);return this.phase==='hold'?0:.32;
    }
    const command=new Float32Array(13),lifted=["lift","upright","hold","release","done"].includes(this.phase);
    let phase=Math.min(.3,Math.max(0,(s.time-1)/4));
    if(lifted){this.riseStart??=s.time;phase=(c.liftStartPhase??.3)+Math.max(0,s.time-this.riseStart-1)/4;}
    if(c.cartesianDescent&&s.time>=1&&!lifted){this.cartesianBase??=Array.from(r.d.ctrl);r.d.ctrl.set(this.cartesianBase);}
    else if(s.time<1&&c.homePose)r.d.ctrl.set(c.homePose);
    else if(s.time<1||phase>.7)await r.policy("walk",command);
    else if(s.time<2.2||lifted){command[0]=Math.cos(2*Math.PI*phase);command[1]=Math.sin(2*Math.PI*phase);await r.policy("groundpick",command);}
    else {
      this.frozen??=Array.from(r.d.ctrl);this.frozenHead??=s.joint.slice(5,7);
      this.frozenGaze??=this.frozen.slice(7,9);
      if(c.liveBase&&s.time<(c.freezeBaseAt??Infinity)){command[0]=Math.cos(2*Math.PI*.3);command[1]=Math.sin(2*Math.PI*.3);await r.policy('groundpick',command);if(c.freezeBaseAt)this.frozen=Array.from(r.d.ctrl);if(c.isolateHead){r.d.ctrl[7]=this.frozenGaze[0];r.d.ctrl[8]=this.frozenGaze[1];}}
      else r.d.ctrl.set(this.frozen);
      const blend=clamp((s.time-2.2)/(c.ramp??2),0,1),[h,k,n]=c.crouchCoeffs??[0,1,.7];
      for(const [j,v] of [[2,h],[3,k],[4,n],[11,-h],[12,-k],[13,-n]])r.d.ctrl[j]+=v*(c.crouch??.25)*blend;
      r.d.ctrl[4]-=(c.lean??0)*blend;r.d.ctrl[13]+=(c.lean??0)*blend;
      if(c.worldAngle!==undefined){const neck=this.frozenHead[0]+((c.neckTarget??-1.5)-this.frozenHead[0])*blend;const head=neck-a.pitch+Math.PI/2+c.worldAngle*blend;r.d.ctrl[5]=neck+r.d.qfrc_bias[r.va[5]]/.55;r.d.ctrl[6]=clamp(head,-1.53,1.53)+r.d.qfrc_bias[r.va[6]]/.55;}
      if(c.balance){const correction=.8*(a.pitch-(c.pitchTarget??.55))+.08*(a.pitch-(this.lastPitch??a.pitch))/.02;r.d.ctrl[4]+=correction;r.d.ctrl[13]-=correction;}
    }
    this.lastPitch=a.pitch;
    if(c.reachFeedback&&this.phase==='reach'&&s.time>=(c.cartesianDescent?1:(c.reachFeedbackStart??4.4))&&s.time>=(this.reopenUntil??0)){
      this.reachServo??=c.dynamicsReach?new HouseholdPredictiveServo(c):c.stanceFeedback?new HouseholdStanceServo(c):new HouseholdReachServo(c);this.reachReady=this.reachServo.apply(r,s);
      if(this.reachServo.blocked)this.enter('failed');
      const reachError=Math.hypot(...this.reachServo.error);
      if(reachError<(this.reachBest??Infinity)-.0005){this.reachBest=reachError;this.reachImprovedAt=s.time;}
      if(c.verticalRetry&&this.age>5&&s.time-(this.reachImprovedAt??s.time)>2&&this.reachServo.error[2]<-.004&&s.joint[5]<-1.44)this.enter('failed');
      if(this.age>(c.reachTimeout??(c.cartesianDescent?40:14)))this.enter('failed');
    }
    if(this.phase==='failed'){r.d.ctrl.set(previousCtrl);return .32;}
    if(c.reachFeedback&&(this.phase==='clamp'||this.phase==='reach'&&s.time<(this.reopenUntil??0))&&this.reachServo){
      if(c.stanceFeedback||c.dynamicsReach){if(c.clampFeedback&&this.phase==='clamp')this.reachServo.apply(r,s);else this.reachServo.hold(r);}
      else {
      for(let k=0;k<4;k++)r.d.ctrl[5+k]+=this.reachServo.offset[k];
      r.d.ctrl[3]+=this.reachServo.offset[4];r.d.ctrl[12]-=this.reachServo.offset[4];
      }
    }
    if(lifted&&this.lowCtrl){const u=clamp((s.time-this.riseStart)/(c.liftBlendSeconds??1),0,1),v=u*u*(3-2*u);for(let j=0;j<14;j++)r.d.ctrl[j]=this.lowCtrl[j]+(r.d.ctrl[j]-this.lowCtrl[j])*v;}
    if(lifted&&c.uprightLift){r.d.ctrl[5]=this.gripNeck+r.d.qfrc_bias[r.va[5]]/.55;r.d.ctrl[6]=this.gripNeck-a.pitch+Math.PI/2+this.gripTheta+r.d.qfrc_bias[r.va[6]]/.55;r.d.ctrl[7]=this.gripYaw-a.yaw;r.d.ctrl[8]=a.roll+(c.preserveGraspRoll?this.gripRoll:0);}
    if(this.phase==="release"||this.phase==="done"){
      const u=clamp((this.age-.3)/1.5,0,1),v=u*u*(3-2*u),theta=this.releaseTheta+(-.25-this.releaseTheta)*v;
      r.d.ctrl[5]=this.releaseNeck+r.d.qfrc_bias[r.va[5]]/.55;
      // Release and done command the jaw open (.32), so head_pitch stays at or below the shared jaw-open limit (65 deg).
      r.d.ctrl[6]=limitHeadPitchTarget(clamp(this.releaseNeck-a.pitch+Math.PI/2+theta,-1.5,1.48),true)+r.d.qfrc_bias[r.va[6]]/.55;
    }
    if(c.stanceLift&&!this.stanceReleased&&this.reachServo&&lifted){
      if(['lift','hold'].includes(this.phase)){
        if(!this.stanceLiftOrigin){
          this.stanceLiftOrientation=Array.from(r.d.xmat.subarray(r.head*9,r.head*9+9));
          const local=c.gripOffset??[-.001,-.05,.003],R=this.stanceLiftOrientation;
          this.stanceLiftOrigin=s.graspPoint.map((p,i)=>p+local.reduce((sum,v,j)=>sum+v*R[i*3+j],0));
        }
        const target=this.stanceLiftOrigin.map((v,i)=>v+(i===2?.07:0));
        this.reachServo.apply(r,s,{target,orientation:this.stanceLiftOrientation});
      }else this.reachServo.hold(r);
    }
    if(this.phase==='upright'){
      const u=clamp(this.age/(c.standDuration??3),0,1),blend=u*u*(3-2*u);
      for(let j=0;j<14;j++)r.d.ctrl[j]=this.uprightCtrl[j]+blend*(r.d.ctrl[j]-this.uprightCtrl[j]);
    }
    for(let j=0;j<14;j++){const ji=Array.from(r.m.jnt_qposadr).indexOf(r.qa[j]),margin=c.margin??.04;r.d.ctrl[j]=clamp(r.d.ctrl[j],r.m.jnt_range[ji*2]+margin,r.m.jnt_range[ji*2+1]-margin);}
    return ["clamp","lift","upright","hold"].includes(this.phase)?0:.32;
  }
}
