// P0: independent trajectory acceptance, shared by browser and headless tests.
const relative=s=>{const [w,x,y,z]=s.headQuat,R=[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)];const p=s.payload.slice(0,3).map((v,j)=>v-s.graspPoint[j]);return [0,1,2].map(j=>p[0]*R[j]+p[1]*R[j+3]+p[2]*R[j+6]);};
export function evaluateHouseholdTrial({phase,rows,summary,modelAudit},{requireRelease=true}={}){
  const failures=[],require=(ok,msg)=>{if(!ok)failures.push(msg);};
  require(requireRelease?phase==='done':['done','carried'].includes(phase),'动作流程未完成');
  require(modelAudit.freePayload&&modelAudit.freeBase&&(!modelAudit.deformable||modelAudit.flexAllNodesFree),'物品或机器人受外部固定');
  require(modelAudit.equalities===0&&modelAudit.mocapBodies===0,'存在附着约束或外部位置控制');
  require(!modelAudit.pedestalPresent&&!modelAudit.terrainPresent&&modelAudit.floorHeight===0,'物品没有直接放在平地');
  require(rows.filter(s=>s.time<=1&&Math.abs(s.shape?.minZ)<.0005&&s.contact.floorForce>.001).length>=10,'缺少初始地面支撑证据');
  require(summary.finite&&!summary.solverReset&&Object.values(summary.warningCounts??{}).every(x=>x===0),'物理计算异常');
  require(summary.maxExternal===0,'出现人工外加力');
  require(summary.maxPenetration<.0005&&summary.maxRobotPenetration<.0005,'接触穿入达到 0.5 毫米');
  require(summary.maxJointLimitExcess<.002,'关节越过机械极限');
  require(!summary.nonFootSupport,'机器人曾借助双脚以外部位撑地');
  let runLength=0,bestLength=0,bestEnd=0,previous=-Infinity;
  for(let i=0;i<rows.length;i++){const s=rows[i],c=s.contact,held=['hold','carried'].includes(s.phase)&&c.upperForce>.002&&c.lowerForce>.002&&c.nonMouthSupport===false&&!c.nonFootSupport&&c.feetForce>1&&s.shape.minZ>.025;
    if(!held||Math.abs(s.time-previous-.02)>1e-6)runLength=0;
    if(held)runLength++;
    if(runLength>bestLength){bestLength=runLength;bestEnd=i+1;}previous=s.time;}
  const best=rows.slice(bestEnd-bestLength,bestEnd);
  const heldSeconds=best.length*.02,tail=best.slice(-50),origin=tail.length?relative(tail[0]):null;
  const heldDrift=origin?Math.max(...tail.map(s=>Math.hypot(...relative(s).map((v,j)=>v-origin[j])))):Infinity;
  require(heldSeconds>=1.2,'未持续离地双面夹持 1.2 秒');require(heldDrift<.003,'保持时相对嘴部滑动达到 3 毫米');
  const released=rows.filter(s=>['release','done'].includes(s.phase)),last=rows.at(-1),releaseFall=released.length?released[0].payload[2]-last.payload[2]:0;
  if(requireRelease)require(releaseFall>.035&&last.jaw>.30&&last.contact.upperForce<.002&&last.contact.lowerForce<.002&&last.contact.floorForce>.001&&Math.abs(last.shape.minZ)<.0005,'未张嘴释放并自然落地');
  return {accepted:!failures.length,failures,heldSeconds,heldDrift,minHeldClearance:best.length?Math.min(...best.map(s=>s.shape.minZ)):0,releaseFall,maximumPenetration:summary.maxPenetration};
}
