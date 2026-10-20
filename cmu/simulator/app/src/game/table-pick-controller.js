// Model-based task controller. Inputs are simulator state estimates (not
// camera detections); outputs only drive the existing leg/head/jaw motors.
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
export function rootAngles([w,x,y,z]) {
  return {roll:Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
    pitch:Math.asin(clamp(2*(w*y-z*x),-1,1)),
    yaw:Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))};
}
export class TablePickController {
  constructor({height=.18,lift=.034,dt=.02,biteOffset=.008}={}) {
    Object.assign(this,{height,lift,dt,biteOffset,phase:"settle",age:0,time:0,neck:.23,stable:0,hold:0,fineX:0,events:[]});
  }
  enter(phase){this.events.push({time:this.time,from:this.phase,to:phase});this.phase=phase;this.age=0;this.stable=0;}
  update(s) {
    this.age+=this.dt;this.time+=this.dt;
    const a=rootAngles(s.root.slice(3)),c=s.contact??{},p=s.graspPoint,t=s.payload;
    const command=new Float32Array(13);
    const liftedPhase=["lift","hold","release","done"].includes(this.phase);
    const z=this.height+(liftedPhase?this.lift:0);
    if(!["settle","position"].includes(this.phase))this.neck=clamp(this.neck+clamp(30*(z-p[2]),-.55,.55)*this.dt,-1.56,.15);
    const yaw=clamp(this.graspYaw===undefined?s.joint[7]+(t[1]-p[1])/.075:this.graspYaw-a.yaw,-.65,.65);
    const tilt=liftedPhase?-.45*Math.min(1,(this.time-this.liftStarted)/1.8):0;
    const head=[this.neck,this.neck-a.pitch+tilt,yaw,a.roll];
    let jaw=["clamp","lift","hold"].includes(this.phase)?0:.32;
    if(this.phase==="settle"&&this.age>1)this.enter("lower");
    if(this.phase==="position") {
      // Walk with a neutral head; the extreme low-neck pose is outside
      // the walking checkpoint's useful locomotion envelope.
      const dx=t[0]-.115-s.root[0];
      command[0]=Math.abs(dx)<.006?0:Math.sign(dx)*clamp(3*Math.abs(dx),.12,.18);
      command[2]=clamp(-a.yaw,-.2,.2);
      if(Math.abs(dx)<.006)this.enter("lower");
    }
    if(this.phase==="lower"&&Math.abs(z-p[2])<.0015&&this.age>2)this.enter("approach");
    if(this.phase==="approach") {
      const dx=t[0]-this.biteOffset-p[0];
      command[0]=Math.sign(dx)*.25;
      command[2]=clamp(2*Math.atan2(t[1]-s.root[1],Math.max(.08,t[0]-s.root[0]))-2*a.yaw,-.6,.6);
      if(t[0]-p[0]<.025){command[0]=0;this.enter("align");}
    }
    if(this.phase==="align") {
      const dx=t[0]-this.biteOffset-p[0];
      // Only stand-policy z/roll/pitch commands were actually trained.
      // Its x/y slots are padding, not a Cartesian translation controller.
      if(Math.abs(t[1]-p[1])<.006)this.fineX=clamp(this.fineX+clamp(5*dx,-.07,.07)*this.dt,-.50,.50);
      if(Math.abs(dx)<.002&&Math.abs(t[1]-p[1])<.003&&Math.abs(t[2]-p[2])<.003)this.stable+=this.dt;
      else this.stable=0;
      if(this.stable>=.15){this.graspYaw=s.joint[7]+a.yaw;this.enter("clamp");}
      else if(this.age>2&&dx>.04){this.fineX=0;this.enter("approach");}
    }
    const opposing=c.lowerForce>.002&&c.upperForce>.002;
    if(this.phase==="clamp") {
      this.stable=opposing?this.stable+this.dt:0;
      if(this.stable>=.2){this.graspOrigin=Array.from(t.slice(0,3));this.liftStarted=this.time;this.enter("lift");}
      else if(this.age>3)this.enter("failed");
    }
    if(this.phase==="lift"&&this.age>.1&&Math.abs(z-p[2])<.0015)this.enter("hold");
    if(this.phase==="hold") {
      const held=opposing&&!c.otherSupport&&!c.nonFootSupport&&c.feetForce>1&&t[2]>this.height+.03&&c.maxPenetration<.0005;
      this.hold=held?this.hold+this.dt:0;
      if(this.hold>=1.2)this.enter("release");
      else if(this.age>3)this.enter("failed");
    }
    if(this.phase==="release"&&this.age>1.5){
      if(s.jaw>.31&&c.upperForce<.002&&c.lowerForce<.002&&t[2]<this.height-.03&&!c.nonFootSupport)this.enter("done");
      else if(this.age>4)this.enter("failed");
    }
    if(this.time>35&&!['done','failed'].includes(this.phase))this.enter("failed");
    if(!liftedPhase&&t[2]<this.height-.005)this.enter("failed");
    if(s.root[2]<.08)this.enter("failed");
    return {head,command,jaw,ankleLean:this.fineX,phase:this.phase};
  }
}
