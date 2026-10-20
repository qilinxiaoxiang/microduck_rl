// Select a measured raised fold near the toe, retaining its material-node ID.
// Observations only: selecting a point never moves any cloth node.
export function rotateVector(q,v){
  if(q.length!==4||v.length!==3||![...q,...v].every(Number.isFinite)||Math.hypot(...q)<1e-12)throw Error('Invalid grasp rotation');
  const n=Math.hypot(...q),[w,x,y,z]=q.map(a=>a/n);
  const t=[2*(y*v[2]-z*v[1]),2*(z*v[0]-x*v[2]),2*(x*v[1]-y*v[0])];
  return [v[0]+w*t[0]+y*t[2]-z*t[1],v[1]+w*t[1]+z*t[0]-x*t[2],v[2]+w*t[2]+x*t[1]-y*t[0]];
}
export function selectRaisedClothNode(s,offset=[-.001,-.05,.003]){
  if(!Array.isArray(s.clothPoints)||!s.clothPoints.length||s.clothPoints.some(p=>p.length!==3||!p.every(Number.isFinite)))throw Error('Expected measured cloth vertices');
  const axis=rotateVector(s.payload.slice(3),[1,0,0]);
  const local=rotateVector(s.headQuat,offset),near=s.graspPoint.map((p,i)=>p+local[i]);
  let index=null,cost=Infinity;
  for(let i=0;i<s.clothPoints.length;i++){
    const p=s.clothPoints[i],along=p.reduce((v,x,j)=>v+(x-s.payload[j])*axis[j],0);
    if(p[2]<.004||along<-.002||along>.03)continue;
    const distance=(p[0]-near[0])**2+(p[1]-near[1])**2;
    if(distance<cost){index=i;cost=distance;}
  }
  return index;
}
