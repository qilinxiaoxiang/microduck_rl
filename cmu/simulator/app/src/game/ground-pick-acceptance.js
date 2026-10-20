// P0 independent acceptance. A completed phase flag alone is never enough.
function relative(s){
  const [w,x,y,z]=s.headQuat,R=[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];
  const p=s.payload.slice(0,3).map((v,j)=>v-s.graspPoint[j]);
  return [0,1,2].map(j=>p[0]*R[j]+p[1]*R[j+3]+p[2]*R[j+6]-[.016,0,.075][j]);
}
export function evaluateGroundTrial({phase,rows,summary,config,modelAudit}){
  const failures=[],require=(ok,message)=>{if(!ok)failures.push(message);};
  require(phase==="done","controller did not complete");
  require(modelAudit.freePayload&&modelAudit.freeBase&&modelAudit.flexAllNodesFree,"payload, soft nodes or robot base not free");
  require(modelAudit.equalities===0&&modelAudit.mocapBodies===0,"attachment or externally prescribed body present");
  require(!modelAudit.pedestalPresent&&!modelAudit.terrainPresent&&modelAudit.floorHeight===0,"ground is not a flat zero-height plane");
  require(modelAudit.deformable&&modelAudit.flexVertices>=27,"payload is not a deformable ball");
  const start=rows.filter(s=>s.time<=1&&s.shape?.minZ<.001&&s.shape?.minZ>-.0005&&s.contact.floorForce>.001);
  require(start.length>=10,"ball did not start resting directly on the ground");
  require(summary.finite&&!summary.solverReset,"non-finite state or solver reset");
  require(Object.values(summary.warningCounts??{}).every(n=>n===0),"physics engine warning detected");
  require(summary.maxExternal===0,"externally applied force or torque detected");
  require(summary.maxPenetration<.0005,"payload impact penetration exceeds 0.5 mm");
  require(summary.maxRobotPenetration<.0005,"robot/environment penetration exceeds 0.5 mm");
  require(!summary.nonFootSupport,"robot braced on something other than its feet");
  require(summary.maxJointLimitExcess<.002,"joint exceeded anatomical limit by over 0.002 rad");
  require(summary.minRadialScale>0,"a soft-ball radial element collapsed or inverted");
  let run=[],best=[],previous=-Infinity;
  for(const s of rows){
    const c=s.contact,held=s.phase==="hold"&&c.lowerForce>.002&&c.upperForce>.002&&c.nonMouthSupport===false&&!c.nonFootSupport&&c.feetForce>1&&s.shape.minZ>.025&&s.payload[2]>config.z+.04;
    if(!held||Math.abs(s.time-previous-.02)>.000001)run=[];
    if(held)run.push(s);if(run.length>best.length)best=[...run];previous=s.time;
  }
  const heldSeconds=best.length*.02;
  require(heldSeconds>=1.2,"no continuous elevated opposing-pad hold for 1.2 seconds");
  const finalHold=best.slice(-50),origin=finalHold.length?relative(finalHold[0]):null;
  const drift=origin?Math.max(...finalHold.map(s=>Math.hypot(...relative(s).map((v,j)=>v-origin[j])))):Infinity;
  require(drift<.003,"ball slipped by at least 3 mm during final held second");
  const release=rows.filter(s=>s.phase==="release"||s.phase==="done"),last=rows.at(-1);
  const fall=release.length?release[0].payload[2]-last.payload[2]:0;
  const released=fall>.035&&last.jaw>.30&&last.contact.upperForce<.002&&last.contact.lowerForce<.002&&last.contact.floorForce>.001&&Math.abs(last.shape.minZ)<.0005;
  require(released,"opening did not release the ball into a real fall onto the ground");
  return {accepted:failures.length===0,failures,heldSeconds,heldDrift:drift,minHeldClearance:best.length?Math.min(...best.map(s=>s.shape.minZ)):0,releaseFall:fall,maximumPenetration:summary.maxPenetration,maximumJointLimitExcess:summary.maxJointLimitExcess,minRadialScale:summary.minRadialScale};
}
