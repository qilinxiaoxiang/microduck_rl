import {makeTablePickXml} from "./table-pick-scene.js";

// P0: the target rests directly on z=0. No hidden pedestal or raised terrain.
export const GROUND_PICK_DEFAULTS={x:-.532,y:.006,z:.0142,mass:.0005,lift:.06,
  radius:.014,young:5000,poisson:.2,softDamping:.0002,meshCount:5,softBall:true,
  crouch:.30,worldAngle:0,neckTarget:-1.53,ramp:2,closeAt:4.6,
  jawTorque:.02,friction:.8,footStiff:true,stiffStops:true,dt:.0000625};
export function makeGroundPickXml(xml,dom,options={}){
  const c={...GROUND_PICK_DEFAULTS,...options};
  // The intermediate pedestal is removed before model compilation.
  const doc=new dom.DOMParser().parseFromString(makeTablePickXml(xml,dom,{...c,z:.195}),"text/xml");
  doc.querySelector('geom[name="payload_pedestal_geom"]').remove();
  doc.querySelector('geom[name="terrain"]').remove();
  // Resolve mechanical stops rather than allowing the default soft limit to
  // overshoot for tens of milliseconds. Bounds themselves are not widened.
  if(c.stiffStops)for(const j of doc.querySelectorAll("joint[range]")){j.setAttribute("solreflimit",".002 1");j.setAttribute("solimplimit",".99 .999 .001");}
  if(c.halfSize)doc.querySelector('geom[name="payload_geom"]').setAttribute("size",c.halfSize.join(" "));
  if(c.softBall){
    const el=(name,attrs)=>{const e=doc.createElement(name);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;};
    const body=doc.querySelector('body[name="payload"]'),g=doc.querySelector('geom[name="payload_geom"]');
    const n=c.meshCount,vertices=n**3-(n-2)**3;
    // A freely moving radial finite-element sphere, following MuJoCo's sphere_radial
    // example. Every surface vertex moves under contact/elastic forces. No weld,
    // pin to the world, scripted deformation, or invisible rigid collision sphere.
    // The central inertial mass is one of the tetrahedral mesh point masses.
    for(const [k,v] of Object.entries({type:"sphere",size:.0001,mass:c.mass/(vertices+1),contype:0,conaffinity:0,rgba:"0 0 0 0"}))g.setAttribute(k,String(v));
    const def=el("default",{class:"soft_ball_nodes"});
    def.appendChild(el("joint",{damping:0,frictionloss:0,armature:0,limited:"false",stiffness:0}));
    doc.querySelector("default").appendChild(def);body.setAttribute("childclass","soft_ball_nodes");
    const spacing=2*c.radius/(n-1),flex=el("flexcomp",{name:"payload_flex",type:"ellipsoid",dof:"radial",dim:3,count:`${n} ${n} ${n}`,spacing:`${spacing} ${spacing} ${spacing}`,radius:.0002,mass:c.mass,inertiabox:.001,rgba:"1 .55 .1 1"});
    flex.appendChild(el("elasticity",{young:c.young,poisson:c.poisson,damping:c.softDamping}));
    flex.appendChild(el("contact",{selfcollide:"none",internal:"false",contype:32,conaffinity:19,condim:3,priority:1,friction:`${c.friction} .0002 .00001`,solref:".00025 1",solimp:".99 .999 .001"}));
    body.appendChild(flex);
    const key=doc.querySelector('key[name="STAND"]');
    key.setAttribute("qpos",key.getAttribute("qpos")+" "+Array(vertices).fill(0).join(" "));
  }
  return new dom.XMLSerializer().serializeToString(doc);
}
