import {wholeBodyRuntime} from "./whole-body-runtime.mjs";
import {DEFAULT_POSE,CTRL_DT} from "../src/game/constants.js";
const config=JSON.parse(process.argv[2]??"{}");
const r=await wholeBodyRuntime();
const command=new Float32Array(13),rows=[];
for(let k=0;k<Math.round((config.seconds??4)/CTRL_DT);k++) {
  const t=k*CTRL_DT;
  if(config.policy==="fixed"){
    r.d.ctrl.set(DEFAULT_POSE);
    for(let h=0;h<4;h++)r.d.ctrl[5+h]+=(config.headDelta?.[h]??0)*Math.min(1,Math.max(0,t-1));
    if(config.gravity)for(let j=0;j<14;j++)r.d.ctrl[j]+=r.d.qfrc_bias[r.va[j]]/.55;
  } else {
    command.fill(0);
    if(t>=1){
      if(config.policy==="groundpick"){command[0]=Math.cos(2*Math.PI*(t-1)/4);command[1]=Math.sin(2*Math.PI*(t-1)/4);}
      else command.set(config.headDelta??[0,0,0,0],3);
      if(config.velocity)command.set(config.velocity,0);
    }
    await r.policy(t<1?"stand":(config.policy??"walk"),command);
    if(config.overrideHead&&t>=1)for(let h=0;h<4;h++){
      const a=Math.min(1,(t-1)/2);
      r.d.ctrl[5+h]=DEFAULT_POSE[5+h]*(1-a)+config.overrideHead[h]*a+r.d.qfrc_bias[r.va[5+h]]/.55;
    }
  }
  const c=r.step(.32);
  if(k%25===0){const s=r.snapshot(c);rows.push(config.compact?{t:s.time,root:s.root,head:s.joint.slice(5,9),point:s.graspPoint}:s);}
}
console.log(JSON.stringify({config,rows,summary:r.summary}));
await r.dispose();
