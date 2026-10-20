import {rootAngles} from "./table-pick-controller.js";
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
// End-point servo using measured joint axes/anchors. No state writes,
// no target attachment, and no world-frame head force is applied.
export class HeadReachController {
  constructor({height=.195,lift=.04,biteOffset=.014,dt=.02}={}){
    Object.assign(this,{height,lift,biteOffset,dt,phase:"settle",time:0,age:0,hold:0,stable:0,events:[]});
  }
  enter(phase){this.events.push({time:this.time,from:this.phase,to:phase});this.phase=phase;this.age=0;this.stable=0;}
  update(s){
    this.time+=this.dt;this.age+=this.dt;
    const a=rootAngles(s.root.slice(3)),p=s.graspPoint,t=s.payload,c=s.contact??{},phase=this.phase;
    this.angles??=[.23,.23-a.pitch];
    this.startX??=p[0];
    let goal=[this.startX,this.height];
    if(phase==="reach")goal=[t[0]-this.biteOffset,this.height];
    if(phase==="clamp")goal=this.gripPoint;
    if(["lift","hold","release","done"].includes(phase)){
      const u=Math.min(1,(this.time-this.liftStarted)/2.5),v=u*u*(3-2*u);
      goal=[this.gripPoint[0]-.015*v,this.gripPoint[1]+this.lift*v];
      // Withdraw after opening so the lower bill cannot continue to prop
      // a released object against the pedestal edge.
      if(phase==="release"||phase==="done")goal[0]-=.012*Math.min(1,Math.max(0,this.age-.4)/1.2);
    }
    if(phase!=="settle"){
      const [[jx0,jz0],[jx1,jz1]]=s.headJac;
      const vx=clamp(6*(goal[0]-p[0]),-.02,.02),vz=clamp(6*(goal[1]-p[2]),-.02,.02);
      // Damped least squares stays bounded near a straight neck.
      const xx=jx0*jx0+jx1*jx1+1e-6,xz=jx0*jz0+jx1*jz1,zz=jz0*jz0+jz1*jz1+1e-6,det=xx*zz-xz*xz;
      const ex=(zz*vx-xz*vz)/det,ez=(xx*vz-xz*vx)/det;
      const q0=jx0*ex+jz0*ez,q1=jx1*ex+jz1*ez;
      const scale=Math.max(1,Math.abs(q0)/.6,Math.abs(q1)/.6);
      // Scale together: clipping separately rotates Cartesian velocity
      // near singularity and can sweep the upper bill into the object.
      this.angles[0]=clamp(this.angles[0]+q0/scale*this.dt,-1.55,.8);
      this.angles[1]=clamp(this.angles[1]+q1/scale*this.dt,-1.55,1.5);
    }
    this.ankleLean??=0;
    // Use ankle motors for the final few millimetres near full extension.
    // The floating base is not repositioned or constrained.
    if(phase==="reach"&&s.joint[5]<-1.20&&goal[0]-p[0]>.0002)
      this.ankleLean=clamp(this.ankleLean+Math.min(.04,4*(goal[0]-p[0]))*this.dt,0,.25);
    // Turn the open bill away from the narrow pedestal to remove any
    // remaining one-sided tray support without pushing through the edge.
    const releaseYaw=phase==="release"?.6*Math.min(1,Math.max(0,this.age-.4)/1.2):0;
    const yaw=clamp(this.graspYaw===undefined?s.joint[7]+(t[1]-p[1])/.075:this.graspYaw-a.yaw+releaseYaw,-.65,.65);
    const head=[...this.angles,yaw,a.roll],jaw=phase==="clamp"?.32*Math.max(0,1-this.age/.6):["lift","hold"].includes(phase)?0:.32;
    const error=Math.hypot(goal[0]-p[0],goal[1]-p[2]);
    const opposing=c.lowerForce>.002&&c.upperForce>.002;
    if(phase==="settle"&&this.age>1){this.startX=p[0];this.angles=s.joint.slice(5,7);this.enter("descend");}
    else if(phase==="descend"&&error<.001&&this.age>1)this.enter("reach");
    else if(phase==="reach"){
      this.stable=error<.0008&&Math.abs(t[1]-p[1])<.0015?this.stable+this.dt:0;
      if(this.stable>=.3){this.gripPoint=[p[0],p[2]];this.graspYaw=s.joint[7]+a.yaw;this.enter("clamp");}
    }else if(phase==="clamp"){
      this.stable=opposing?this.stable+this.dt:0;
      if(this.stable>=.4){this.liftStarted=this.time;this.enter("lift");}
      else if(this.age>3)this.enter("failed");
    }else if(phase==="lift"){
      if(this.age>2.5&&error<.002&&t[2]>this.height+.03)this.enter("hold");
      else if(this.age>8)this.enter("failed");
    }else if(phase==="hold"){
      const held=opposing&&!c.otherSupport&&!c.nonFootSupport&&c.feetForce>1&&t[2]>this.height+.03&&c.maxPenetration<.0005;
      this.hold=held?this.hold+this.dt:0;
      if(this.hold>=1.2)this.enter("release");else if(this.age>3)this.enter("failed");
    }else if(phase==="release"&&this.age>1.5){
      if(s.jaw>.31&&c.upperForce<.002&&c.lowerForce<.002&&t[2]<this.height-.03&&!c.nonFootSupport)this.enter("done");
      else if(this.age>6)this.enter("failed");
    }
    if((!['lift','hold','release','done'].includes(phase)&&t[2]<this.height-.005)||s.root[2]<.08||this.time>35||c.maxPenetration>.001)this.enter("failed");
    return {head,jaw,command:new Float32Array(13),ankleLean:this.ankleLean,phase:this.phase};
  }
}
