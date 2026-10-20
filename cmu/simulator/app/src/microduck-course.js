import {courseMetrics,groundExperimentUrl,normalizeCourseConfig} from "./game/tutorial-parameters.js";

const form=document.querySelector("#material-form");
const resetButton=document.querySelector("#reset-form");
const runLink=document.querySelector("#run-experiment");

function values(){
  const data=new FormData(form);
  return {
    radius:Number(data.get("radius"))/1000,
    mass:Number(data.get("mass"))/1000,
    young:Number(data.get("young")),
    poisson:Number(data.get("poisson")),
    softDamping:Number(data.get("softDamping")),
    friction:Number(data.get("friction")),
    meshCount:Number(data.get("meshCount")),
    dt:Number(data.get("dt")),
    x:Number(data.get("x"))/1000,
  };
}

function show(name,value,digits=1){
  document.querySelector(`[data-for="${name}"]`).textContent=Number(value).toFixed(digits);
}

function update(){
  const config=normalizeCourseConfig(values()),m=courseMetrics(config);
  show("radius",config.radius*1000);show("mass",config.mass*1000);show("young",config.young,0);
  show("poisson",config.poisson,2);show("softDamping",config.softDamping,4);show("friction",config.friction,2);show("x",config.x*1000,1);
  document.querySelector("#diameter").textContent=`${m.diameterMm.toFixed(1)} mm`;
  document.querySelector("#density").textContent=`${m.density.toFixed(1)} kg/m³`;
  document.querySelector("#vertices").textContent=String(m.flexVertices);
  document.querySelector("#steps").textContent=`${m.physicsStepsPerControl} 物理步`;
  document.querySelector("#experiment-status").textContent=m.knownDefault?"默认已验证配置":"新组合，需要重新验收";
  document.querySelector("#experiment-status").classList.toggle("changed",!m.knownDefault);
  runLink.href=groundExperimentUrl(config,new FormData(form).get("controller"),location.href);
}

form.addEventListener("input",update);
resetButton.addEventListener("click",()=>{form.reset();update();});
update();
