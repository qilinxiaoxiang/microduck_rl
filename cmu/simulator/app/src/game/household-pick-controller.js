import {rootAngles} from './table-pick-controller.js';
import {GroundPickController} from './ground-pick-controller.js';
import {householdHeading} from './household-grasp-target.js';
import {selectRaisedClothNode} from './household-cloth-target.js';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
export const wrapAngle=a=>Math.atan2(Math.sin(a),Math.cos(a));
export function footNavigationPosition(feet,offset,yaw){
  if(feet.length!==2||feet.some(p=>p.length<2)||offset.length!==2||![...feet.flat(),...offset,yaw].every(Number.isFinite))throw Error('Invalid foot navigation geometry');
  const co=Math.cos(yaw),si=Math.sin(yaw);
  return [(feet[0][0]+feet[1][0])/2+co*offset[0]-si*offset[1],(feet[0][1]+feet[1][1])/2+si*offset[0]+co*offset[1]];
}
export function retryBiteTarget({distance,lateral,error,yaw,verticalStep=0}){
  if(![distance,lateral,yaw,verticalStep,...error].every(Number.isFinite)||error.length!==3)throw Error('Invalid measured retry target');
  const co=Math.cos(yaw),si=Math.sin(yaw);
  return {distance:clamp(distance-.8*(error[0]*co+error[1]*si)+verticalStep,.045,.11),
          lateral:clamp(lateral-.8*(-error[0]*si+error[1]*co),-.075,-.02)};
}

// P4 local navigation. Object poses are privileged simulator observations.
// Every command passes through the existing learned walking motor policy.
export class HouseholdPickController{
  constructor(c){if(c.raisedCloth&&(c.kind!=='sock'||!c.stanceFeedback))throw Error('Raised-cloth target requires the cloth stance controller');this.c=c;this.phase='manual';this.events=[];this.attempts=[];this.hold=0;this.time=0;this.age=0;this.message='走到物品附近，按 G 自动叼取';}
  enter(phase){this.events.push({time:this.time,from:this.phase,to:phase});this.phase=phase;this.age=0;this.stable=0;}
  begin(s){
    if(this.phase!=='manual')return false;
    if(Math.hypot(s.payload[0]-s.root[0],s.payload[1]-s.root[1])>.55){this.message='距离超过 55 厘米，请先靠近';return false;}
    this.time=s.time;this.target=s.payload.slice(0,3);
    if(this.c.raisedCloth){
      this.graspNode=selectRaisedClothNode(s,this.c.gripOffset);
      if(this.graspNode===null){this.message='袜头附近没有找到足够凸起的布边，保留现场';return false;}
      this.target=s.clothPoints[this.graspNode].slice();
    }
    const current=rootAngles(s.root.slice(3)).yaw;
    this.yaw=wrapAngle(householdHeading(s.payload,this.c.kind,current)+(this.c.bodyHeadingOffset??0));
    this.alignStage=this.c.stagingDistance?'staging':'final';
    this.enter(this.c.preStand?'prestand':'align');if(this.graspNode!==undefined)this.events.push({time:this.time,type:'cloth-grasp-node',node:this.graspNode,point:this.target.slice()});this.message=this.c.preStand?'正在停步并站稳':'正在调整脚下位置与朝向';return true;
  }
  release(s){if(this.inner&&['carried','hold'].includes(this.phase)){this.inner.release(s);this.phase=this.inner.phase;}}
  async actuate(r,s,manual=new Float32Array(13)){
    if(this.graspNode!==undefined&&this.graspNode!==null){
      const point=s.clothPoints?.[this.graspNode];if(!point||point.length!==3||!point.every(Number.isFinite))throw Error('Tracked cloth node is unavailable');
      s={...s,taskGraspPoint:point.slice()};
    }
    this.time=s.time;this.age+=.02;
    if(this.phase==='manual'){await r.policy('walk',manual);return .32;}
    if(this.phase==='prepose'){
      const u=clamp(this.age/1.5,0,1),blend=u*u*(3-2*u);
      for(let j=0;j<14;j++)r.d.ctrl[j]=this.homeStart[j]+blend*(this.c.homePose[j]-this.homeStart[j]);
      if(this.age>=(this.c.homeDuration??3)){
        this.start=s.time;this.inner=new GroundPickController({...this.c,holdUntilRelease:true});this.enter('settle');
      }
      return .32;
    }
    if(this.phase==='prestand'){
      await r.policy('walk',new Float32Array(13));
      const velocity=this.previousRoot?Math.hypot(...s.root.slice(0,3).map((v,i)=>(v-this.previousRoot[i])/.02)):Infinity;
      this.previousRoot=s.root.slice(0,3);
      this.stable=this.age>=2&&velocity<.015&&(s.contact?.feetForce??0)>1?this.stable+.02:0;
      if(this.stable>=.4){this.enter('align');this.message='正在调整脚下位置与朝向';}
      else if(this.age>8){this.enter('failed');this.message='尚未站稳，保留现场';}
      return .32;
    }
    if(this.phase==='recover'){
      const cmd=new Float32Array(13),phase=.3+Math.max(0,this.age-1)/4;
      if(phase>.7)await r.policy('walk',cmd);
      else {cmd[0]=Math.cos(2*Math.PI*phase);cmd[1]=Math.sin(2*Math.PI*phase);await r.policy('groundpick',cmd);}
      if(this.c.recoveryHeadLock){
        const a=rootAngles(s.root.slice(3)),p=this.recoveryHead;
        r.d.ctrl[5]=p.neck+r.d.qfrc_bias[r.va[5]]/.55;
        r.d.ctrl[6]=p.neck-a.pitch+Math.PI/2+p.theta+r.d.qfrc_bias[r.va[6]]/.55;
        r.d.ctrl[7]=p.yaw-a.yaw;r.d.ctrl[8]=p.roll+a.roll;
      }
      const u=clamp(this.age/1.2,0,1),blend=u*u*(3-2*u);
      for(let j=0;j<14;j++)r.d.ctrl[j]=this.recoveryCtrl[j]+(r.d.ctrl[j]-this.recoveryCtrl[j])*blend;
      if(this.age>4.5){this.target=s.payload.slice(0,3);this.braking=false;this.enter('align');}
      return .32;
    }
    if(this.phase==='align'){
      const co=Math.cos(this.yaw),si=Math.sin(this.yaw),forward=(this.biteDistance??this.c.biteDistance??.0667)+(this.alignStage==='staging'?(this.stagingDistance??this.c.stagingDistance):0),side=this.biteLateral??this.c.biteLateral??.0058;
      const goal=[this.target[0]-co*forward+si*side,this.target[1]-si*forward-co*side];
      const a=rootAngles(s.root.slice(3));let navigationPosition=s.root;
      if(this.c.footNavigation){
        const feet=r.stanceKinematics(s.graspPoint).feet.map(f=>f.point),center=footNavigationPosition(feet,[0,0],0);
        const dx=s.root[0]-center[0],dy=s.root[1]-center[1];
        this.footNavOffset??=[Math.cos(a.yaw)*dx+Math.sin(a.yaw)*dy,-Math.sin(a.yaw)*dx+Math.cos(a.yaw)*dy];
        // The walking network first leans the trunk by about a centimetre,
        // often without stepping. Stop on foot placement, not that transient
        // body lean. The post-stop check still uses the actual trunk position.
        navigationPosition=footNavigationPosition(feet,this.footNavOffset,a.yaw);
      }
      const dx=goal[0]-navigationPosition[0],dy=goal[1]-navigationPosition[1],dist=Math.hypot(dx,dy),actualDist=Math.hypot(goal[0]-s.root[0],goal[1]-s.root[1]);
      const ex=dx*Math.cos(a.yaw)+dy*Math.sin(a.yaw),ey=-dx*Math.sin(a.yaw)+dy*Math.cos(a.yaw);
      const yawError=wrapAngle(this.yaw-a.yaw),command=new Float32Array(13);
      const backSpeed=this.c.navBackSpeed??(this.attempts.length?this.c.retryBackSpeed:undefined);
      if(this.c.nonholonomic){
        // Measured walking response: sideways commands can stall, and -0.25
        // produces almost no backward travel. Steer then walk; do not assume
        // a commanded planar velocity is realized by the learned policy.
        const heading=wrapAngle(Math.atan2(dy,dx)-a.yaw),back=!this.c.forwardOnly&&Math.abs(heading)>Math.PI/2;
        const error=wrapAngle(heading-(back?Math.PI:0));
        if(dist>(this.c.alignTolerance??.009)){
          command[0]=Math.abs(error)<.35?(back?-(this.c.backSpeed??.4):.25):0;
          command[2]=Math.abs(error)>.04?Math.sign(error)*clamp(2*Math.abs(error),.8,1):0;
        }else command[2]=Math.abs(yawError)>.04?Math.sign(yawError)*clamp(2*Math.abs(yawError),.8,1):0;
        if(this.c.turnGait&&command[0]===0&&Math.abs(command[2])>0){
          command[0]=Math.floor(this.age/.6)%2===0?.25:-(this.c.backSpeed??.4);
        }
      }else if(dist>.035){
        const heading=wrapAngle(Math.atan2(dy,dx)-a.yaw),back=Math.abs(heading)>Math.PI/2;
        const error=wrapAngle(heading-(back?Math.PI:0));
        command[0]=Math.abs(error)<.5?(back?-(backSpeed??.25):.25):0;
        command[2]=Math.abs(error)>.04?Math.sign(error)*clamp(2*Math.abs(error),.8,1):0;
      }else{
        // The upstream walking policy also accepts lateral velocity. Its
        // response is verified in the approach perturbation tests, not assumed.
        command[0]=Math.abs(ex)>.002?(ex<0?-(backSpeed??this.c.navSpeed??.2):(this.c.navSpeed??.2)):0;command[1]=Math.abs(ey)>.002?Math.sign(ey)*.3:0;command[2]=Math.abs(yawError)>.03?Math.sign(yawError)*.8:0;
      }
      if(this.c.pulsed===true&&dist<.06){
        command.fill(0);
        if(this.pulseUntil&&s.time<this.pulseUntil)command.set(this.pulseCommand);
        else if(s.time>=(this.restUntil??0)){
          const pulse=new Float32Array(3),speed=this.c.pulseSpeed??.3;
          let duration=this.c.pulseDuration??.12;
          if(Math.abs(yawError)>(this.c.pulseYawTolerance??.04)){pulse[2]=Math.sign(yawError)*1;duration=.08;}
          else if(Math.abs(ex)>.002&&(this.c.pulseLongitudinalFirst||Math.abs(ex)>=Math.abs(ey)))pulse[0]=ex<0?-(this.c.pulseBackSpeed??speed):speed;
          else if(Math.abs(ey)>(this.c.pulseLateralTolerance??.002))pulse[1]=Math.sign(ey)*speed;
          this.pulseCommand=pulse;this.pulseUntil=s.time+duration;this.restUntil=this.pulseUntil+(this.c.restDuration??.6);command.set(pulse);
        }
      }
      const longitudinal=(s.root[0]-goal[0])*co+(s.root[1]-goal[1])*si;
      if(dist<(this.c.alignTolerance??.004)&&Math.abs(longitudinal)<(this.c.alignLongitudinalTolerance??Infinity)&&Math.abs(yawError)<(this.c.yawTolerance??.05))this.braking=true;
      if(this.braking){command.fill(0);this.stable+=.02;}
      if(this.braking&&this.stable>=1&&(actualDist>(this.c.settleTolerance??.006)||Math.abs(longitudinal)>(this.c.settleLongitudinalTolerance??Infinity)||Math.abs(yawError)>(this.c.settleYawTolerance??.06))){this.braking=false;this.stable=0;}
      this.alignment={distance:actualDist,navigationDistance:dist,navigationPosition:Array.from(navigationPosition.slice(0,2)),yawError,goal,stage:this.alignStage,command:Array.from(command.slice(0,3))};
      await r.policy('walk',command);
      if(this.braking&&this.stable>=1){
        if(this.alignStage==='staging'){
          this.alignStage='final';this.braking=false;this.stable=0;this.age=0;
          this.events.push({time:this.time,from:'staging',to:'final-approach'});return .32;
        }
        if(this.c.homePose){
          if(this.c.homePose.length!==14||!this.c.homePose.every(Number.isFinite))throw Error('Expected 14 finite home motor targets');
          this.homeStart=Array.from(r.d.ctrl);this.enter('prepose');this.message='正在调整拾取前的关节姿态';
        }else{
          this.start=s.time;
          this.inner=new GroundPickController({...this.c,holdUntilRelease:true});this.enter('settle');
        }
      }
      if(this.age>(this.c.alignTimeout??40)){this.message='对准超时，保留现场';this.enter('failed');}
      return .32;
    }
    if(this.phase==='failed')return .32;
    const target=await this.inner.actuate(r,{...s,time:s.time-this.start});
    this.phase=this.inner.phase;this.hold=this.inner.hold;
    if(this.phase==='failed'&&this.c.retry&&this.inner.reachServo&&this.attempts.length<2&&!r.summary.nonFootSupport&&r.summary.finite&&!r.summary.solverReset&&!(r.summary.maxExternal??0)&&Math.max(r.summary.maxPenetration??0,r.summary.maxRobotPenetration??0)<.0005&&(r.summary.maxJointLimitExcess??0)<.002){
      const error=this.inner.reachServo.error,vertical=this.c.verticalRetry&&error[2]<-.004&&s.joint[5]<-1.44;
      const next=retryBiteTarget({distance:this.biteDistance??this.c.biteDistance,lateral:this.biteLateral??this.c.biteLateral,error,yaw:this.yaw,verticalStep:vertical?(this.c.verticalRetryStep??.006):0});
      this.attempts.push({time:s.time,error:error.slice(),reason:vertical?'neck-limit vertical reach failure':'measured reach error',next,events:this.inner.events.slice(),alignment:this.alignment});
      this.biteDistance=next.distance;this.biteLateral=next.lateral;
      if(this.c.retryStagingDistance){this.stagingDistance=this.c.retryStagingDistance;this.alignStage='staging';}
      const a=rootAngles(s.root.slice(3));
      this.recoveryHead={neck:s.joint[5],theta:a.pitch-s.joint[5]+s.joint[6]-Math.PI/2,yaw:s.joint[7]+a.yaw,roll:s.joint[8]-a.roll};
      this.recoveryCtrl=Array.from(r.d.ctrl);this.enter('recover');
    }
    return target;
  }
}
