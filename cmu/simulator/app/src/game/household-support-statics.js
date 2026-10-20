import {solve} from './household-reach-servo.js';

// Quasi-static support estimate, not a physics constraint or applied force.
// Foot wrench rows use world coordinates; their base columns already encode
// MuJoCo's local free-joint angular-velocity convention.
export function supportStatics(feet,bias,momentLength=.015){
  const rows=feet.flatMap(f=>[...f.position,...f.rotation]);
  if(feet.length!==2||rows.length!==12||rows.some(r=>r.length!==20)||bias.length!==20||
    ![...rows.flat(),...bias,momentLength].every(Number.isFinite)||momentLength<=0)throw Error('Invalid support statics input');
  const weights=rows.map((_,i)=>i%6<3?1:momentLength**2);
  const matrix=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>
    rows.reduce((sum,row,k)=>sum+weights[k]*row[i]*row[j],0)+(i===j?1e-10:0)));
  const dual=solve(matrix,bias.slice(0,6));
  const wrench=rows.map((row,k)=>weights[k]*row.slice(0,6).reduce((sum,v,i)=>sum+v*dual[i],0));
  const residual=bias.map((v,j)=>v-rows.reduce((sum,row,k)=>sum+row[j]*wrench[k],0));
  return {torques:residual.slice(6),baseResidual:residual.slice(0,6),wrenches:[wrench.slice(0,6),wrench.slice(6)]};
}
