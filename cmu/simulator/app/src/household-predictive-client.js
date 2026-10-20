// Optional slow model-based controller. Physics stays in this browser;
// the local service sees observations and returns only 14 motor targets.
export async function createPredictiveClient(r,xml,config,{endpoint='http://127.0.0.1:18766'}={}){
  if(config.raisedCloth)throw Error('布料节点预测目前只接通了本地离线实验，浏览器版本尚未验证');
  const url=new URL(endpoint);
  if(url.protocol!=='http:'||url.hostname!=='127.0.0.1')throw Error('预测服务必须运行在本机');
  const mj=r.mj,spec=mj.mjtState.mjSTATE_INTEGRATION.value;
  const signature={nq:r.m.nq,nv:r.m.nv,nu:r.m.nu,nbody:r.m.nbody,ngeom:r.m.ngeom,stateSize:mj.mj_stateSize(r.m,spec)};
  function readState(){
    // mj_getState does not copy its output back into a detached JS array in
    // this binding. Use the owned WASM buffer and take an independent copy.
    const buffer=new mj.DoubleBuffer(signature.stateSize);
    try{mj.mj_getState(r.m,r.d,buffer,spec);return Float64Array.from(buffer.GetView());}
    finally{buffer.delete();}
  }
  const modelText=xml.replace(/^\s*<\?xml[^?]*\?>\s*/,'');
  const modelSha256=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(modelText))),b=>b.toString(16).padStart(2,'0')).join('');
  async function post(path,value){
    const response=await fetch(new URL(path,url),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value),signal:AbortSignal.timeout(60000)});
    const data=await response.json();if(!response.ok)throw Error(data.error??'本机预测失败');return data;
  }
  let session,sequence=0,lastTime=-Infinity,plan=null,busy=false,totalSeconds=0;
  async function reset(){
    const initialized=await post('/init',{config,signature,modelSha256,modelXml:xml});
    session=initialized.session;sequence=0;lastTime=-Infinity;plan=null;totalSeconds=0;
  }
  async function update(stage='reach'){
    if(!['reach','transport'].includes(stage)||stage==='transport'&&!config.dynamicsLift)throw Error('无效的预测阶段');
    if(plan?.stage===stage&&r.d.time-lastTime<.199)return plan;
    if(busy)throw Error('上一轮预测尚未结束');
    const state=readState();
    const started=performance.now();busy=true;
    try{
      const result=await post('/plan',{session,sequence:sequence+1,stage,state:Array.from(state)});
      const after=readState();
      if(state.some((v,i)=>!Object.is(v,after[i])))throw Error('预测期间仿真状态变化，已拒绝过期结果');
      if(result.sequence!==sequence+1||Math.abs(result.time-r.d.time)>1e-8||result.plan?.stage!==stage||result.plan?.ctrl?.length!==14||!result.plan.ctrl.every(Number.isFinite))throw Error('预测返回值不符合电机控制约定');
      sequence=result.sequence;lastTime=r.d.time;plan=result.plan;totalSeconds+=(performance.now()-started)/1000;
      return plan;
    }finally{busy=false;}
  }
  await reset();r.predictedReachPlan=()=>plan;
  return {reset,update,get busy(){return busy;},get totalSeconds(){return totalSeconds;},get plan(){return plan;}};
}
