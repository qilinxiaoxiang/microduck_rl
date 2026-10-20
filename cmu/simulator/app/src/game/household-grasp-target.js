// A pen can be grasped anywhere along its usable shaft. Constraining its
// center to one fixed edge of the bill unnecessarily couples reach to roll.
export function householdHeading(payload,kind,reference=0){
  if(payload.length!==7||![...payload,reference].every(Number.isFinite))throw Error('Invalid household heading');
  const q=payload.slice(3),norm=Math.hypot(...q);
  if(norm<1e-12)throw Error('Invalid object quaternion');
  const [w,x,y,z]=q.map(v=>v/norm);
  if(kind!=='pen')return Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));
  // The capsule's shaft is local +Y. Axial roll must not turn a horizontal
  // pen's approach direction by pi (Euler yaw becomes singular at 90 deg).
  const ax=2*(x*y-w*z),ay=1-2*(x*x+z*z);
  if(Math.hypot(ax,ay)<1e-6)return reference;
  let heading=Math.atan2(-ax,ay);
  if(Math.cos(heading-reference)<0)heading+=Math.PI;
  return reference+Math.atan2(Math.sin(heading-reference),Math.cos(heading-reference));
}
export function closestPenBite(payload, point, length, endMargin=.02){
  if(payload.length!==7||point.length!==3||![...payload,...point,length,endMargin].every(Number.isFinite)||length<=0||endMargin<0)throw Error('Invalid pen grasp geometry');
  const [w,x,y,z]=payload.slice(3),axis=[2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)];
  const half=Math.max(0,length/2-endMargin);
  const along=Math.max(-half,Math.min(half,point.reduce((sum,v,i)=>sum+(v-payload[i])*axis[i],0)));
  return payload.slice(0,3).map((v,i)=>v+along*axis[i]);
}
