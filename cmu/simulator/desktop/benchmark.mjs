import fs from 'node:fs';
const root=new URL('../app/',import.meta.url).href;
const {wholeBodyRuntime}=await import(root+'tests/whole-body-runtime.mjs');
const {dom}=await import(root+'tests/physics-fixture.mjs');
const {makeGroundPickXml,GROUND_PICK_DEFAULTS:c}=await import(root+'src/game/ground-pick-scene.js');
const {GroundPickController}=await import(root+'src/game/ground-pick-controller.js');
const {JawActionFilter}=await import(root+'src/game/jaw-rl.js');
const r=await wholeBodyRuntime({mutate:x=>makeGroundPickXml(x,dom)});
let times=[],result;
for(let repeat=0;repeat<3;repeat++){
 r.reset([c.x,c.y,c.z+.0005,1,0,0,0]);const ctl=new GroundPickController(c),filter=new JawActionFilter();let contact={};const start=performance.now();
 for(let i=0;i<100;i++){const target=await ctl.actuate(r,r.snapshot(contact));contact=r.step(filter.update(target===0?0:1));}
 times.push(performance.now()-start);result={qpos:Array.from(r.d.qpos),summary:r.summary,contact};
}
const out={times,simSeconds:2,...result};fs.writeFileSync(process.argv[2],JSON.stringify(out,null,2));console.log(JSON.stringify({times,simSeconds:2}));await r.dispose();
