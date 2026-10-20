import {displayPixelRatio} from "./game/browser-budget.js";
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {buildRig,loadKinematics,setJawOpen,JAW_MAX_OPEN,ROBOT_KINEMATICS} from './game/duck.js';

export async function householdView(r,c){
  const {m,d,mj}=r,rig=await buildRig(await loadKinematics(ROBOT_KINEMATICS));
  rig.root.rotation.set(0,0,0);
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x101f29);scene.add(rig.placer);
  const frames=[];
  for(const [name,group] of rig.bodies){const id=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY.value,name);if(id>=0){rig.root.add(group);frames.push({id,group});}}
  const camera=new THREE.PerspectiveCamera(38,innerWidth/innerHeight,.001,15);camera.up.set(0,0,1);camera.position.set(-.95,-.73,.47);
  const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(displayPixelRatio);renderer.setSize(innerWidth,innerHeight);document.body.prepend(renderer.domElement);
  const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(-.44,.03,.11);controls.update();
  scene.add(new THREE.HemisphereLight(0xffffff,0x5b6879,2.4));const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(-.2,-.5,1);scene.add(light);
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(3,3),new THREE.MeshStandardMaterial({color:0x293b48,roughness:1})));
  const grid=new THREE.GridHelper(3,60,0x577281,0x354f60);grid.rotation.x=Math.PI/2;grid.position.z=.0001;scene.add(grid);
  const objects=[],wires=[],matrix=new THREE.Matrix4();
  for(let g=0;g<m.ngeom;g++){
    const name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM.value,g)??'',target=name==='payload_geom';
    const wire=name.startsWith('beak_')&&name.includes('_contact_');
    if(!target&&!wire)continue;
    let geometry;
    if(wire){const mid=m.geom_dataid[g],va=m.mesh_vertadr[mid],vn=m.mesh_vertnum[mid],fa=m.mesh_faceadr[mid],fn=m.mesh_facenum[mid];geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.BufferAttribute(new Float32Array(m.mesh_vert.slice(va*3,(va+vn)*3)),3));geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(m.mesh_face.slice(fa*3,(fa+fn)*3)),1));geometry.computeVertexNormals();}
    else {geometry=new THREE.CapsuleGeometry(c.radius,c.length-2*c.radius,6,20);geometry.rotateX(Math.PI/2);}
    if(target){const pos=geometry.getAttribute('position'),colors=new Float32Array(pos.count*3);for(let i=0;i<pos.count;i++){const z=pos.getZ(i),color=new THREE.Color(z < -c.length/2+.008?0x273944:z > c.length/2-.018?0x175976:0x43bef2);colors.set(color.toArray(),i*3);}geometry.setAttribute('color',new THREE.BufferAttribute(colors,3));}
    const object=new THREE.Mesh(geometry,new THREE.MeshStandardMaterial({color:wire?0x59ffb0:0xffffff,vertexColors:target,wireframe:wire,transparent:wire,opacity:wire?.5:1,roughness:.65}));object.visible=!wire;scene.add(object);objects.push({g,object});if(wire)wires.push(object);
  }
  let flex;
  if(r.payloadFlexId>=0){const f=r.payloadFlexId,va=m.flex_vertadr[f],vn=m.flex_vertnum[f],ea=m.flex_elemdataadr[f],en=m.flex_elemnum[f];
    flex=new THREE.BufferGeometry();flex.setAttribute('position',new THREE.BufferAttribute(new Float32Array(vn*3),3));flex.setIndex(Array.from(m.flex_elem.slice(ea,ea+en*3)));
    const colors=new Float32Array(vn*3);for(let i=0;i<vn;i++){const ring=Math.floor(i/c.sides),color=new THREE.Color(ring===c.rings-2?0x71a8bc:0xe6dbce);colors.set(color.toArray(),i*3);}flex.setAttribute('color',new THREE.BufferAttribute(colors,3));
    const mesh=new THREE.Mesh(flex,new THREE.MeshStandardMaterial({color:0xffffff,vertexColors:true,roughness:1,side:THREE.DoubleSide}));scene.add(mesh);
  }
  const ring=new THREE.Mesh(new THREE.RingGeometry(.08,.082,64),new THREE.MeshBasicMaterial({color:0x91d6cf,side:THREE.DoubleSide,transparent:true,opacity:.55}));ring.position.set(c.x,c.y,.0003);scene.add(ring);
  function render(){
    for(const {id,group} of frames){const p=d.xpos.subarray(id*3,id*3+3),q=d.xquat.subarray(id*4,id*4+4);group.position.set(...p);group.quaternion.set(q[1],q[2],q[3],q[0]);}setJawOpen(rig,d.qpos[r.jaw]/JAW_MAX_OPEN);
    for(const {g,object} of objects){if(!object.visible)continue;const p=d.geom_xpos.subarray(g*3,g*3+3),v=d.geom_xmat.subarray(g*9,g*9+9);object.position.set(...p);matrix.set(v[0],v[1],v[2],0,v[3],v[4],v[5],0,v[6],v[7],v[8],0,0,0,0,1);object.quaternion.setFromRotationMatrix(matrix);}
    if(flex){const start=m.flex_vertadr[r.payloadFlexId]*3,attr=flex.getAttribute('position');attr.array.set(d.flexvert_xpos.subarray(start,start+attr.array.length));attr.needsUpdate=true;flex.computeVertexNormals();flex.computeBoundingSphere();}
    controls.update();renderer.render(scene,camera);
  }
  addEventListener('resize',()=>{camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
  return {render,wires};
}
