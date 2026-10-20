// Read-only contact evidence. No forces, poses or constraints are written here.
export function readPayloadContacts(mujoco, model, data, payloadIds, lowerIds, upperIds, scratch, robotAudit, payloadFlexId=-1) {
  const result = { lowerForce: 0, upperForce: 0, maxPenetration: 0, otherSupport: false, nonMouthSupport:false, count: 0 };
  if(robotAudit)Object.assign(result,{feetForce:0,nonFootSupport:false,nonFootForce:0,robotMaxPenetration:0});
  const buffer = scratch ?? new mujoco.DoubleBuffer(6);
  // Embind returns a copied vector, not a borrowed view. Release both the
  // vector and each contact handle, including early-continue paths.
  const contacts = data.contact;
  try {
  for (let i = 0; i < data.ncon; i++) {
    const c = contacts.get(i);
    try {
    const geom1=c.geom1,geom2=c.geom2,distance=c.dist,address=c.efc_address;
    const flex=payloadFlexId>=0?c.flex:null,flex0=flex?.[0],flex1=flex?.[1];
    const first = payloadIds.includes(geom1)||(payloadFlexId>=0&&flex0===payloadFlexId), second = payloadIds.includes(geom2)||(payloadFlexId>=0&&flex1===payloadFlexId);
    const robot=robotAudit?.robotGeoms.has(geom1)?geom1:robotAudit?.robotGeoms.has(geom2)?geom2:null;
    const environment=robot!==null&&model.geom_bodyid[robot===geom1?geom2:geom1]===0;
    if (!first && !second && !environment) continue;
    if(environment)result.robotMaxPenetration=Math.max(result.robotMaxPenetration,-distance);
    if(environment&&distance<=0&&address>=0){
      mujoco.mj_contactForce(model,data,i,buffer);
      const f=buffer.GetView()[0];
      if(robotAudit.feetGeoms.has(robot))result.feetForce+=f;
      else if(f>1e-5){result.nonFootSupport=true;result.nonFootForce+=f;result.nonFootPair=[geom1,geom2];}
    }
    if(!first&&!second)continue;
    if (-distance > result.maxPenetration) {
      result.maxPenetration = -distance;
      result.deepestPair = [geom1, geom2];
    }
    // Cloth self-contact transmits internal forces; it cannot support the
    // object against gravity or count as contact with the robot/ground.
    if(first&&second)continue;
    if (distance > 0 || address < 0) continue;
    mujoco.mj_contactForce(model, data, i, buffer);
    const force = buffer.GetView();
    if (force[0] <= 1e-5) continue;
    result.count++;
    const other = first ? geom2 : geom1;
    if (lowerIds.includes(other)) result.lowerForce += force[0];
    else if (upperIds.includes(other)) result.upperForce += force[0];
    else {
      result.otherSupport = true;result.otherGeoms??={};result.otherGeoms[other]=(result.otherGeoms[other]??0)+force[0];result.otherContactPosition??=Array.from(c.pos);
      // Keep the legacy strict pad-only flag above. For the soft-ball task,
      // separately identify the original rigid mouth rim, not the whole head.
      // The rear boundary is the measured rear edge of both pad meshes
      // (local z ~= -43.5 mm). Body/neck/ground contacts remain non-mouth support.
      let rim=false;
      if(robotAudit?.mouthRimGeoms?.has(other)){
        const h=robotAudit.head,R=data.xmat.subarray(h*9,h*9+9),p=data.xpos.subarray(h*3,h*3+3),v=Array.from(c.pos,(x,j)=>x-p[j]);
        const local=[0,1,2].map(j=>v[0]*R[j]+v[1]*R[j+3]+v[2]*R[j+6]);
        rim=local[2]<=-.0435&&Math.abs(local[1])<=.0462;
        if(rim){result.mouthRimForce=(result.mouthRimForce??0)+force[0];result.mouthRimPosition??=local;}
      }
      if(!rim)result.nonMouthSupport=true;
      if(other>=0&&model.geom_bodyid[other]===0)result.floorForce=(result.floorForce??0)+force[0];
    }
    } finally { c.delete(); }
  }
  return result;
  } finally { contacts.delete(); if (!scratch) buffer.delete(); }
}

export function createContactAudit() {
  return { steps: 0, maxPenetration: 0, maxExternalForce: 0, maxZ: -Infinity,
    simultaneousForceSteps: 0, heldSeconds: 0, maxHeldSeconds: 0, escapes: 0 };
}

// Average normal impulse over the actual 20 ms controller/sensor window.
// Raw substep evidence is still audited separately; this writes no physics.
export function createContactWindow() {
  return {samples:0,lowerForce:0,upperForce:0,maxPenetration:0,otherSupport:false,nonMouthSupport:false,count:0};
}
export function addContactSample(window, sample) {
  window.samples++;
  window.lowerForce+=sample.lowerForce;
  window.upperForce+=sample.upperForce;
  window.otherSupport ||= sample.otherSupport;
  window.nonMouthSupport ||= sample.nonMouthSupport;
  if(sample.mouthRimForce!==undefined){window.mouthRimForce=(window.mouthRimForce??0)+sample.mouthRimForce;window.mouthRimPosition??=sample.mouthRimPosition;}
  if(sample.floorForce!==undefined)window.floorForce=(window.floorForce??0)+sample.floorForce;
  if(sample.otherGeoms){window.otherGeoms??={};for(const [g,f] of Object.entries(sample.otherGeoms))window.otherGeoms[g]=(window.otherGeoms[g]??0)+f;window.otherContactPosition??=sample.otherContactPosition;}
  window.count=Math.max(window.count,sample.count);
  if(sample.maxPenetration>window.maxPenetration)window.deepestPair=sample.deepestPair;
  window.maxPenetration=Math.max(window.maxPenetration,sample.maxPenetration);
  if(sample.feetForce!==undefined){
    window.feetForce=(window.feetForce??0)+sample.feetForce;
    window.nonFootForce=(window.nonFootForce??0)+sample.nonFootForce;
    window.maxNonFootForce=Math.max(window.maxNonFootForce??0,sample.nonFootForce);
    window.nonFootSupport||=sample.nonFootSupport;
    if(sample.nonFootPair)window.nonFootPair??=sample.nonFootPair;
    window.robotMaxPenetration=Math.max(window.robotMaxPenetration??0,sample.robotMaxPenetration);
  }
}
export function averageContactWindow(window) {
  return {...window,...(window.mouthRimForce!==undefined?{mouthRimForce:window.mouthRimForce/Math.max(1,window.samples)}:{}),...(window.otherGeoms?{otherGeoms:Object.fromEntries(Object.entries(window.otherGeoms).map(([g,f])=>[g,f/Math.max(1,window.samples)]))}:{}),...(window.floorForce!==undefined?{floorForce:window.floorForce/Math.max(1,window.samples)}:{}),...(window.feetForce!==undefined?{feetForce:window.feetForce/Math.max(1,window.samples),nonFootForce:window.nonFootForce/Math.max(1,window.samples)}:{}),lowerForce:window.lowerForce/Math.max(1,window.samples),upperForce:window.upperForce/Math.max(1,window.samples)};
}

export function recordContactStep(audit, contact, data, payloadBodyId, payloadQposAdr, dt, held) {
  audit.steps++;
  if (contact.maxPenetration > audit.maxPenetration) {
    audit.worstContact = {time:data.time,pair:contact.deepestPair,
      position:Array.from(data.qpos.subarray(payloadQposAdr,payloadQposAdr+3))};
  }
  audit.maxPenetration = Math.max(audit.maxPenetration, contact.maxPenetration);
  const external = data.xfrc_applied;
  for (let i = 0; i < 6; i++) audit.maxExternalForce = Math.max(audit.maxExternalForce, Math.abs(external[payloadBodyId * 6 + i]));
  audit.maxZ = Math.max(audit.maxZ, data.qpos[payloadQposAdr + 2]);
  if (contact.lowerForce > 0 && contact.upperForce > 0) audit.simultaneousForceSteps++;
  audit.heldSeconds = held ? audit.heldSeconds + dt : 0;
  audit.maxHeldSeconds = Math.max(audit.maxHeldSeconds, audit.heldSeconds);
}
