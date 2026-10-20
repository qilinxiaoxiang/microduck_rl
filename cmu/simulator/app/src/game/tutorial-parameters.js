import {GROUND_PICK_DEFAULTS} from "./ground-pick-scene.js";

const LIMITS={
  radius:[.008,.024],mass:[.0002,.003],young:[1000,15000],poisson:[.05,.45],
  softDamping:[0,.003],friction:[.25,1.4],meshCount:[3,7],x:[-.54,-.524],dt:[.00003125,.000125],
};
const clamp=(value,[lo,hi])=>Math.max(lo,Math.min(hi,value));

export function normalizeCourseConfig(values={}){
  const config={...GROUND_PICK_DEFAULTS};
  for(const name of Object.keys(LIMITS)){
    if(values[name]===undefined)continue;
    const value=Number(values[name]);
    if(!Number.isFinite(value))throw Error(`${name} 不是有效数字`);
    config[name]=clamp(value,LIMITS[name]);
  }
  config.meshCount=[3,5,7].reduce((best,n)=>Math.abs(n-config.meshCount)<Math.abs(best-config.meshCount)?n:best,3);
  config.softBall=true;
  return config;
}

export function courseMetrics(config){
  const volume=4/3*Math.PI*config.radius**3;
  return {
    diameterMm:config.radius*2000,
    massG:config.mass*1000,
    density:config.mass/volume,
    flexVertices:config.meshCount**3-(config.meshCount-2)**3+1,
    physicsStepsPerControl:Math.round(.02/config.dt),
    knownDefault:config.meshCount===5&&Math.abs(config.x-GROUND_PICK_DEFAULTS.x)<1e-12&&Math.abs(config.dt-GROUND_PICK_DEFAULTS.dt)<1e-12,
  };
}

export function groundExperimentUrl(config,controller="learned",base="http://127.0.0.1/"){
  if(!["learned","untrained","baseline"].includes(controller))throw Error("未知控制器");
  const url=new URL("ground-pick.html",base);
  url.searchParams.set("controller",controller);
  url.searchParams.set("config",JSON.stringify(config));
  return url.href;
}

export {LIMITS};
