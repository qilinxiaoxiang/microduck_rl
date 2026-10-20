// Deterministic PPO actor export. This evaluates action means; exploration
// noise and the critic are training-only. Input normalization lives in the
// shared observation function, not an untracked browser-specific transform.
import {RESIDUAL_CONTRACT} from './household-residual.js';
export function householdResidualActor(model,observation){
  if(Object.entries(RESIDUAL_CONTRACT).some(([key,value])=>JSON.stringify(model[key])!==JSON.stringify(value))||observation.length!==93||!Array.from(observation).every(Number.isFinite))throw Error('Incompatible household residual policy');
  let x=Array.from(observation);
  for(const layer of model.layers){
    if(!['tanh','linear'].includes(layer.activation)||layer.weight.length!==layer.bias.length||layer.weight.some(row=>row.length!==x.length))throw Error('Invalid residual layer');
    x=layer.weight.map((row,i)=>{
      const value=row.reduce((sum,w,j)=>sum+w*x[j],layer.bias[i]);
      return layer.activation==='tanh'?Math.tanh(value):value;
    });
  }
  if(x.length!==14||!x.every(Number.isFinite))throw Error('Invalid residual actor output');
  return x.map(v=>Math.max(-1,Math.min(1,v)));
}
