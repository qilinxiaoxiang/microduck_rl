// Independent readback: controller phase labels alone never prove a grasp.
const norm=v=>Math.hypot(...v);
function relativePosition(row){
  const q=row.headQuat,[w,x,y,z]=q,p=row.object??row.payload,g=row.point??row.graspPoint;
  const [a,b,c]=p.map((v,i)=>v-g[i]).slice(0,3);
  return [
    (1-2*y*y-2*z*z)*a+2*(x*y+w*z)*b+2*(x*z-w*y)*c,
    2*(x*y-w*z)*a+(1-2*x*x-2*z*z)*b+2*(y*z+w*x)*c,
    2*(x*z+w*y)*a+2*(y*z-w*x)*b+(1-2*x*x-2*y*y)*c,
  ];
}
export function evaluateWholeBodyTrial({phase,rows,summary,config,modelAudit}){
  const height=config.z??.195,failures=[];
  let run=[],best=[],previous=-Infinity;
  for(const row of rows){
    const c=row.contact??{},p=row.object??row.payload,t=row.t??row.time;
    const held=row.phase==="hold"&&c.lowerForce>.002&&c.upperForce>.002&&!c.otherSupport&&
      !c.nonFootSupport&&c.feetForce>1&&p[2]>height+.03&&c.maxPenetration<.0005;
    if(!held||t-previous>.020001)run=[];
    if(held){run.push(row);if(run.length>best.length)best=[...run];}
    previous=t;
  }
  const stable=best.slice(-50),ref=stable.length?relativePosition(stable[0]):[0,0,0];
  const holdDrift=Math.max(0,...stable.map(r=>norm(relativePosition(r).map((v,i)=>v-ref[i]))));
  const heldSeconds=best.length*.02;
  const minHeldLift=best.length?Math.min(...best.map(r=>(r.object??r.payload)[2]-height)):0;
  const last=rows.at(-1),lastContact=last?.contact??{},lastPayload=last?.object??last?.payload;
  const released=!!last&&last.jaw>.31&&lastContact.lowerForce<.002&&lastContact.upperForce<.002&&lastPayload[2]<height-.03;
  const check=(ok,message)=>{if(!ok)failures.push(message);};
  check(phase==="done","controller did not complete");
  check(heldSeconds>=1,"no continuous one-second elevated opposing-contact hold");
  check(holdDrift<.003,"object slipped by at least 3 mm during final held second");
  check(released,"opening did not release the object into a real fall");
  check(summary.maxExternal===0,"externally applied force or torque detected");
  check(summary.finite===true,"non-finite simulation state");
  check(summary.maxPenetration<.0005,"payload impact penetration exceeds 0.5 mm");
  check(summary.maxRobotPenetration<.0005,"robot/environment penetration exceeds 0.5 mm");
  check(summary.nonFootSupport===false,"robot braced on something other than its feet");
  check(summary.maxJointLimitExcess<=.002,"joint exceeded anatomical limit by over 0.002 rad");
  check(modelAudit?.freePayload===true&&modelAudit?.freeBase===true,"payload or robot base is not free");
  check(modelAudit?.equalities===0&&modelAudit?.mocapBodies===0,"attachment or externally prescribed body present");
  return {accepted:failures.length===0,heldSeconds,minHeldLift,holdDrift,released,failures};
}
