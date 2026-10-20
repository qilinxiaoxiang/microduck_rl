// A constrained physical head-lift fixture. This is not a whole-body robot
// task: a servo carries the head vertically. The payload remains free.
export function makeGraspBenchXml(xml, {DOMParser, XMLSerializer}, prop={x:.075,y:0,z:.124}) {
  const doc=new DOMParser().parseFromString(xml,"text/xml");
  const el=(tag,attrs)=>{const e=doc.createElement(tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e;};
  const head=doc.querySelector('body[name="jaw_soft"]').cloneNode(true);
  for(const child of [...head.children])if(child.tagName==="joint")child.remove();
  head.append(el("joint",{name:"bench_lift",type:"slide",axis:"1 0 0"}));
  head.setAttribute("pos","0 0 0.14");
  head.setAttribute("quat","0.7071067811865476 0 -0.7071067811865476 0");
  doc.querySelector('body[name="trunk_base"]').remove();
  doc.querySelector("worldbody").append(head);
  doc.querySelector("sensor").remove();
  doc.querySelector("keyframe").remove();
  for(const a of [...doc.querySelector("actuator").children])if(a.getAttribute("name")!=="beak_actuator")a.remove();
  doc.querySelector("actuator").append(el("position",{name:"bench_lift_actuator",joint:"bench_lift",kp:"1000",kv:"20",forcerange:"-5 5"}));
  for(const e of [...doc.querySelectorAll("equality")])e.remove();
  const pedestal=doc.querySelector('geom[name="payload_pedestal_geom"]');
  pedestal.setAttribute("pos",`${prop.x+.015} ${prop.y??0} ${(prop.z-.003)/2}`);
  pedestal.setAttribute("size",`.002 .004 ${(prop.z-.003)/2}`);
  return new XMLSerializer().serializeToString(doc);
}
