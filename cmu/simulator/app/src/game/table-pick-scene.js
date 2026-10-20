// Jaw torque and friction are explicit, uncalibrated simulation parameters.
export const TABLE_PICK_DEFAULTS={x:-.48,y:0,z:.195,mass:.0025,lift:.04,biteOffset:.014,footStiff:true,jawTorque:.02,friction:.8,cartesian:true};
export function makeTablePickXml(xml,{DOMParser,XMLSerializer},options={}) {
  const c={...TABLE_PICK_DEFAULTS,...options},doc=new DOMParser().parseFromString(xml,"text/xml");
  const g=doc.querySelector('geom[name="payload_pedestal_geom"]');
  g.setAttribute("pos",`${c.x+.008} ${c.y} ${(c.z-.003)/2}`);
  g.setAttribute("size",`.010 .006 ${(c.z-.003)/2}`);
  doc.querySelector('geom[name="payload_geom"]').setAttribute("mass",String(c.mass));
  if(c.friction!==undefined)for(const geom of doc.querySelectorAll('geom[name^="beak_"][name*="_contact_"], geom[name="payload_geom"]'))geom.setAttribute("friction",`${c.friction} 0.0002 0.00001`);
  if(c.jawTorque)doc.querySelector('position[name="beak_actuator"]').setAttribute("forcerange",`${-c.jawTorque} ${c.jawTorque}`);
  if(c.dt)doc.querySelector("option").setAttribute("timestep",String(c.dt));
  if(c.footStiff)for(const name of ["left_foot_collision","right_foot_collision"]){
    const foot=doc.querySelector(`geom[name="${name}"]`);
    foot.setAttribute("priority","1");foot.setAttribute("solref","0.002 1");foot.setAttribute("solimp","0.99 0.999 0.001");
  }
  return new XMLSerializer().serializeToString(doc);
}
