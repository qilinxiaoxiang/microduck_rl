import {makeGroundPickXml,GROUND_PICK_DEFAULTS} from './ground-pick-scene.js';

// P3/P4 prototypes. Dimensions/materials are explicit assumptions, not calibrated
// household specimens. The pen is a rigid capsule; the sock is a free cloth tube.
export const HOUSEHOLD_OBJECTS={
  pen:{label:'笔',radius:.006,length:.14,mass:.006,z:.0062,x:-.52,y:-.044,crouch:.28,uprightLift:true,biteDistance:.0787,biteLateral:-.0442},
  sock:{label:'薄袜子',radius:.0003,length:.10,width:.04,mass:.003,z:.008,x:-.52,y:-.044,crouch:.28,worldAngle:-.15,young:20000,rings:7,sides:6,biteDistance:.0787,biteLateral:-.0442},
};
export function householdConfig(options={}){
  const kind=options.kind??'pen';
  if(!HOUSEHOLD_OBJECTS[kind])throw Error('Unknown household object');
  const c={...GROUND_PICK_DEFAULTS,...HOUSEHOLD_OBJECTS[kind],kind,softBall:false,...options};
  for(const key of ['radius','length','mass','dt'])if(!Number.isFinite(c[key])||c[key]<=0)throw Error('Invalid '+key);
  if(kind==='pen'&&c.length<=2*c.radius)throw Error('Pen length must exceed its diameter');
  if(kind==='sock'&&(!Number.isFinite(c.width)||c.width<=0||!Number.isInteger(c.rings)||c.rings<3||c.rings>20||!Number.isInteger(c.sides)||c.sides<4||c.sides>16))throw Error('Invalid cloth dimensions or resolution');
  return c;
}
export function makeHouseholdPickXml(xml,dom,options={}){
  const c=householdConfig(options),doc=new dom.DOMParser().parseFromString(makeGroundPickXml(xml,dom,c),'text/xml');
  const el=(tag,attrs)=>{const e=doc.createElement(tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,String(v));return e;};
  const body=doc.querySelector('body[name="payload"]'),geom=doc.querySelector('geom[name="payload_geom"]');
  if(c.kind==='pen'){
    geom.setAttribute('type','capsule');geom.setAttribute('size',String(c.radius));
    geom.setAttribute('fromto',`0 ${-c.length/2+c.radius} 0 0 ${c.length/2-c.radius} 0`);
    geom.removeAttribute('pos');geom.removeAttribute('quat');
    geom.setAttribute('rgba','.1 .6 .95 1');
  }else{
    // Flattened knitted-tube approximation, closed toe and open cuff.
    // All nodes have three unconstrained sliders under a static coordinate
    // frame. MuJoCo's bending model requires this topology. There are no pins.
    const rings=c.rings,sides=c.sides,points=[],elements=[];
    doc.querySelector('option').setAttribute('solver','CG');doc.querySelector('option').setAttribute('tolerance','1e-8');
    // Start at the toe surface; the first free node supplies a tracked point.
    for(let i=0;i<rings;i++)for(let j=0;j<sides;j++){
      const a=2*Math.PI*j/sides,scale=i===0?.25:1;
      points.push(i*c.length/(rings-1),Math.sin(a)*c.width/2*scale,Math.cos(a)*.006*scale-.0015);
    }
    for(let i=0;i<rings-1;i++)for(let j=0;j<sides;j++){
      const a=i*sides+j,b=i*sides+(j+1)%sides,d=(i+1)*sides+j,e=(i+1)*sides+(j+1)%sides;
      elements.push(a,b,d,b,e,d);
    }
    for(let j=1;j<sides-1;j++)elements.push(0,j+1,j);
    const def=el('default',{class:'household_cloth_nodes'});
    def.appendChild(el('joint',{damping:0,frictionloss:0,armature:0,limited:'false',stiffness:0}));doc.querySelector('default').appendChild(def);
    body.setAttribute('childclass','household_cloth_nodes');
    body.querySelector('freejoint').remove();geom.remove();
    body.setAttribute('pos',`${c.x} ${c.y} ${c.z}`);
    const flex=el('flexcomp',{name:'payload_flex',type:'direct',dof:'full',dim:2,point:points.join(' '),element:elements.join(' '),radius:c.radius,mass:c.mass,inertiabox:.001,rgba:'.95 .5 .3 1'});
    flex.appendChild(el('elasticity',{young:c.young,poisson:.2,thickness:.0006,elastic2d:'both',damping:.002}));
    flex.appendChild(el('contact',{selfcollide:'auto',internal:'false',contype:32,conaffinity:51,condim:3,priority:1,friction:`${c.friction} .0002 .00001`,solref:'.00025 1',solimp:'.99 .999 .001'}));
    body.appendChild(flex);
    const key=doc.querySelector('key[name="STAND"]'),q=key.getAttribute('qpos').trim().split(/\s+/);key.setAttribute('qpos',q.slice(0,-7).concat(Array(rings*sides*3).fill(0)).join(' '));
  }
  return new dom.XMLSerializer().serializeToString(doc);
}
