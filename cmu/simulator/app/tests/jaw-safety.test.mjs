import assert from "node:assert/strict";
import fs from "node:fs";
import {execFileSync} from "node:child_process";
import {JAW_LIMITS,JAW_MAX_OPEN_RAD,JAW_OPEN_THRESHOLD_RAD,HEAD_PITCH_MAX_OPEN_RAD,limitJawTarget,jawCountsAsOpen,limitHeadPitchTarget,headPitchCtrlCap,applyJawSafety,
  POSE_AXES,POSE_TABLE_MIN_CONTACT_DEG,poseContactDeg,capForContact,poseJawCapRad} from "../src/game/jaw-safety.js";
import {DEFAULT_POSE,SPAWN_X,SPAWN_Y} from "../src/game/constants.js";
import {wholeBodyRuntime} from "./whole-body-runtime.mjs";

const DEG=Math.PI/180,close=(a,b,tol=1e-12)=>Math.abs(a-b)<=tol;
const canonical=new URL("../../../guide/firmware/pi/jaw_limits.json",import.meta.url);
if(fs.existsSync(canonical))assert.equal(fs.readFileSync(new URL("../src/game/jaw-limits.json",import.meta.url),"utf8"),fs.readFileSync(canonical,"utf8"),"jaw-limits.json must stay a byte copy of the robot file");
const canonicalTable=new URL("../../../guide/firmware/pi/jaw_pose_caps.json",import.meta.url);
if(fs.existsSync(canonicalTable))assert.equal(fs.readFileSync(new URL("../src/game/jaw-pose-caps.json",import.meta.url),"utf8"),fs.readFileSync(canonicalTable,"utf8"),"jaw-pose-caps.json must stay a byte copy of the robot table");
assert.equal(JAW_LIMITS.jaw_max_open_deg,25);assert.equal(JAW_LIMITS.jaw_open_threshold_deg,2);assert.equal(JAW_LIMITS.head_pitch_max_while_jaw_open_deg,65);
assert.ok(close(JAW_MAX_OPEN_RAD,25*DEG)&&close(JAW_OPEN_THRESHOLD_RAD,2*DEG)&&close(HEAD_PITCH_MAX_OPEN_RAD,65*DEG));

// Jaw cap: at and below 25 deg unchanged, above it clamped, closing past zero untouched.
for(const t of [-.05,0,.32,25*DEG])assert.equal(limitJawTarget(t),t);
assert.equal(limitJawTarget(25*DEG+1e-6),JAW_MAX_OPEN_RAD);assert.equal(limitJawTarget(1),JAW_MAX_OPEN_RAD);
assert.throws(()=>limitJawTarget(NaN));
// Open test: larger of commanded and measured, strictly above 2 deg; unknown counts as open.
assert.equal(jawCountsAsOpen(2*DEG,2*DEG),false);assert.equal(jawCountsAsOpen(0,2*DEG+1e-6),true);
assert.equal(jawCountsAsOpen(2*DEG+1e-6,0),true);assert.equal(jawCountsAsOpen(0,null),false);assert.equal(jawCountsAsOpen(),true);
// Head interlock on the target angle.
assert.equal(limitHeadPitchTarget(65*DEG,true),65*DEG);assert.equal(limitHeadPitchTarget(65*DEG+1e-6,true),HEAD_PITCH_MAX_OPEN_RAD);
assert.equal(limitHeadPitchTarget(1.48,false),1.48);assert.equal(limitHeadPitchTarget(-1.5,true),-1.5);
// Control cap includes the gravity term a position actuator needs to settle at the limit.
assert.ok(close(headPitchCtrlCap(.011,.55),65*DEG+.02));assert.throws(()=>headPitchCtrlCap(0,0));
// Head pose for applyJawSafety: indices/kp/bias as whole-body-core passes them, measured head straight ahead.
const indices={neck_pitch:5,head_pitch:6,head_yaw:7,head_roll:8},kp={neck_pitch:.55,head_pitch:.55,head_yaw:.55,head_roll:.55};
const STRAIGHT={neck_pitch:0,head_pitch:0,head_yaw:0,head_roll:0},NOBIAS={neck_pitch:0,head_pitch:0,head_yaw:0,head_roll:0};
const pose=(measured=STRAIGHT,bias=NOBIAS)=>({indices,kp,bias,measured});
const flags=x=>({jawCapped:x.jawCapped,headCapped:x.headCapped,jawPoseCapped:x.jawPoseCapped,jawClosedZone:x.jawClosedZone});
const ctrl=new Float64Array(15);ctrl[14]=.6;ctrl[6]=1.4;
// The head command 1.4 is capped to 65 deg + bias/kp, whose equilibrium is 65 deg; the pose box spans the measured 0 deg.
const c65=poseContactDeg({...STRAIGHT,head_pitch:65*DEG},STRAIGHT);
let res=applyJawSafety(ctrl,{measuredJaw:0,headBias:.011,kp:.55,pose:pose(undefined,{...NOBIAS,head_pitch:.011})});
assert.deepEqual(flags(res),{jawCapped:true,headCapped:true,jawPoseCapped:c65<27,jawClosedZone:false});
assert.equal(ctrl[14],Math.min(JAW_MAX_OPEN_RAD,capForContact(c65).capDeg*DEG));assert.ok(close(ctrl[6],65*DEG+.02));
ctrl[14]=0;ctrl[6]=1.4;assert.deepEqual(flags(applyJawSafety(ctrl,{measuredJaw:.001,headBias:0,kp:.55,pose:pose()})),{jawCapped:false,headCapped:false,jawPoseCapped:false,jawClosedZone:false});assert.equal(ctrl[6],1.4);
ctrl[14]=0;ctrl[6]=1.4;assert.deepEqual(flags(applyJawSafety(ctrl,{measuredJaw:.2,headBias:0,kp:.55,pose:pose()})),{jawCapped:false,headCapped:true,jawPoseCapped:false,jawClosedZone:false});
console.log("PASS jaw-safety: 25 deg jaw cap, 2 deg open threshold, 65 deg head interlock, control cap");

// Pose cap: grid nodes, the cells between them, unknown joints, the closed zone and the cap formula.
const AX=["neck_pitch","head_pitch","head_yaw","head_roll"].map(n=>POSE_AXES[n]),SH=AX.map(a=>a[2]);
const TABLE=JSON.parse(fs.readFileSync(new URL("../src/game/jaw-pose-caps.json",import.meta.url),"utf8"));
const node=(i)=>{const v=TABLE.first_contact_deg[i[0]][i[1]][i[2]][i[3]];return v??TABLE.clear_to_deg;};
const at=(i,frac=[0,0,0,0])=>Object.fromEntries(["neck_pitch","head_pitch","head_yaw","head_roll"].map((n,a)=>[n,(AX[a][0]+(i[a]+frac[a])*AX[a][1])*DEG]));
const boxMin=(lo,hi)=>{let v=Infinity;for(let a=lo[0];a<=hi[0];a++)for(let b=lo[1];b<=hi[1];b++)for(let c=lo[2];c<=hi[2];c++)for(let d=lo[3];d<=hi[3];d++)v=Math.min(v,node([a,b,c,d]));return v;};
let all=Infinity,count=0;
for(let a=0;a<SH[0];a++)for(let b=0;b<SH[1];b++)for(let c=0;c<SH[2];c++)for(let d=0;d<SH[3];d++){
  const v=node([a,b,c,d]);all=Math.min(all,v);count++;
  if((a+b+c+d)%7===0)assert.equal(poseContactDeg(at([a,b,c,d]),null),v,`node ${a},${b},${c},${d}`);
}
assert.equal(POSE_TABLE_MIN_CONTACT_DEG,all);
// Inside a cell every surrounding node counts, never an interpolated value.
for(const i of [[0,0,0,0],[6,20,10,5],[9,35,33,9],[3,9,28,4],[5,29,17,9]]){
  assert.equal(poseContactDeg(at(i,[.5,.5,.5,.5]),null),boxMin(i,i.map(x=>x+1)),`cell ${i}`);
  // Commanded and measured span the box between them.
  const j=[Math.min(i[0]+1,SH[0]-1),Math.min(i[1]+3,SH[1]-1),i[2]+1,i[3]];
  assert.equal(poseContactDeg(at(i),at(j)),boxMin(i,j),`span ${i}`);
}
// Unknown joints take their whole axis; a fully unknown pose (or one off the grid) is the most conservative.
assert.equal(poseContactDeg(null,null),all);
const mid=[6,18,17,5];
assert.equal(poseContactDeg({...at(mid),head_pitch:200*DEG},null),boxMin([6,0,17,5],[6,SH[1]-1,17,5]));
assert.equal(poseContactDeg({head_yaw:60*DEG},undefined),boxMin([0,0,23,0],[SH[0]-1,SH[1]-1,23,SH[3]-1]));
// Up to 0.5 deg past an edge counts as the last cell; further out the joint is unknown.
assert.equal(poseContactDeg({...at(mid),neck_pitch:-90.3*DEG},null),boxMin([0,18,17,5],[1,18,17,5]));
assert.equal(poseContactDeg({...at(mid),head_roll:25.4*DEG},null),boxMin([6,18,17,9],[6,18,17,10]));
assert.equal(poseContactDeg({...at(mid),neck_pitch:-90.6*DEG},null),boxMin([0,18,17,5],[SH[0]-1,18,17,5]));
assert.deepEqual(capForContact(27),{capDeg:25,closed:false});assert.deepEqual(capForContact(20),{capDeg:18,closed:false});
assert.deepEqual(capForContact(7),{capDeg:5,closed:false});assert.deepEqual(capForContact(6.99),{capDeg:0,closed:true});
assert.deepEqual(capForContact(0),{capDeg:0,closed:true});
assert.deepEqual(poseJawCapRad(STRAIGHT,STRAIGHT),{capRad:25*DEG,closed:false,contactDeg:27});
// The pitch -45 / yaw 110..160 torso zone holds the jaw closed, and applyJawSafety logs it.
const zone={neck_pitch:0,head_pitch:-45*DEG,head_yaw:140*DEG,head_roll:0};
assert.equal(poseJawCapRad(zone,zone).closed,true);
ctrl[14]=.2;ctrl[5]=0;ctrl[6]=zone.head_pitch;ctrl[7]=zone.head_yaw;ctrl[8]=0;
res=applyJawSafety(ctrl,{measuredJaw:0,headBias:0,kp:.55,pose:pose(zone)});
assert.deepEqual(flags(res),{jawCapped:false,headCapped:false,jawPoseCapped:false,jawClosedZone:true});assert.equal(ctrl[14],0);
ctrl[14]=-.01;applyJawSafety(ctrl,{measuredJaw:0,headBias:0,kp:.55,pose:pose(zone)});assert.equal(ctrl[14],-.01,"squeeze past closed stays");
// Without a head pose the most conservative cap applies.
ctrl[14]=.2;ctrl[6]=0;ctrl[7]=0;applyJawSafety(ctrl,{measuredJaw:0,headBias:0,kp:.55});
assert.equal(ctrl[14],Math.min(.2,capForContact(all).capDeg*DEG));
// Same answers as the robot's jaw_limits.py on random poses (including commanded/measured boxes and unknown joints).
const pyDir=new URL("../../../guide/firmware/pi/",import.meta.url);
if(fs.existsSync(new URL("jaw_limits.py",pyDir))){
  let seed=20261013;const rnd=()=>(seed=(seed*1103515245+12345)%2147483648)/2147483648;
  const sample=()=>{const o={};for(const [n,lo,hi] of [["neck_pitch",-91,61],["head_pitch",-91,91],["head_yaw",-171,171],["head_roll",-26,26]])if(rnd()>.15)o[n]=(lo+rnd()*(hi-lo))*DEG;return o;};
  const cases=Array.from({length:400},()=>[sample(),rnd()>.3?sample():null]);
  const out=execFileSync("python3",["-c","import sys,json,jaw_limits as j\nprint(json.dumps([[j.pose_contact_deg(c,m),*j.pose_jaw_cap_deg(c,m)[:2]] for c,m in json.load(sys.stdin)]))"],
    {cwd:pyDir,input:JSON.stringify(cases)}).toString();
  JSON.parse(out).forEach(([contact,cap,closed],n)=>{
    const js=poseJawCapRad(...cases[n]);
    assert.equal(poseContactDeg(...cases[n]),contact,`python contact ${n}`);
    assert.ok(close(js.capRad,cap*DEG),`python cap ${n}`);assert.equal(js.closed,closed,`python closed ${n}`);
  });
}
console.log(`PASS jaw-safety pose cap: ${count} nodes, cells, spans, unknown pose, closed zone, Python match; table min contact ${all} deg`);

// MuJoCo check: the core enforces the interlock at its last write before mj_step, whatever the controller asks.
// The trunk is welded in place (a robot held on a stand) so the head moves only under its own actuator and gravity;
// a free robot without its balance policy falls and the ground then pushes the head.
// relpose is the world origin in the trunk frame at the STAND keyframe (trunk at SPAWN_X, SPAWN_Y, 0.12, level).
const r=await wholeBodyRuntime({mutate:x=>x.replace("</mujoco>",`<equality><weld body1="trunk_base" relpose="${-SPAWN_X} ${-SPAWN_Y} -0.12 1 0 0 0"/></equality></mujoco>`)});
assert.ok(close(r.headPitchKp,.55,1e-9),"head_pitch actuator gain");
const hold=(jaw,head,ticks)=>{for(let t=0;t<ticks;t++){for(let j=0;j<14;j++)r.d.ctrl[j]=DEFAULT_POSE[j];r.d.ctrl[6]=head+r.d.qfrc_bias[r.va[6]]/.55;r.step(jaw);}};
r.reset();hold(.32,1.48,100);
const open=r.d.qpos[r.qa[6]],openPeak=r.summary.maxHeadPitchJawOpen;
assert.ok(open<=65*DEG+.5*DEG,`jaw open: head_pitch settles at ${(open/DEG).toFixed(2)} deg`);
// The cap is a static-equilibrium command; a step from the 20 deg stand pose overshoots briefly before settling.
assert.ok(openPeak<=65*DEG+2*DEG,`jaw open: head_pitch peak ${(openPeak/DEG).toFixed(2)} deg`);
assert.ok(r.summary.headPitchInterlockTicks>90,"interlock engaged");
r.reset();hold(0,1.2,100);
const shut=r.d.qpos[r.qa[6]];
assert.equal(r.summary.headPitchInterlockTicks,0,"closed jaw is not interlocked");
assert.ok(shut>65*DEG+1*DEG,`closed jaw: head_pitch free to ${(shut/DEG).toFixed(2)} deg`);
console.log(`PASS jaw-safety MuJoCo: jaw open head_pitch settles ${(open/DEG).toFixed(2)} deg (peak ${(openPeak/DEG).toFixed(2)}), closed jaw reaches ${(shut/DEG).toFixed(2)} deg`);
await r.dispose();
