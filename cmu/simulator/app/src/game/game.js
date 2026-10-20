import {waitForVisible} from "./browser-budget.js";
// Microduck RL playground core: the REAL trained policies, not a procedural
// waddle. Framework-agnostic port of the pre-React rl.js.
//
// Physics runs in MuJoCo compiled to WebAssembly (the official
// @mujoco/mujoco bindings), stepping the same MJCF the policies were
// trained on (pollen-robotics/microduck_rl). The controller is one of the
// exported ONNX checkpoints from pollen-robotics/microduck, executed with
// onnxruntime-web at 50 Hz. Contact validation uses 0.125 ms integration
// and 160 substeps while preserving that trained policy interface.
//
// Obs layout (61D, "new-cmd-obs" flavor, from the ONNX metadata):
//   [base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
//    last_action(14), command(13)]
//
// Integration contract with the React shell:
//   - bootGame({ scene, camera, renderer }) is called once from inside the
//     R3F canvas; it loads everything, wires inputs and starts the 50 Hz
//     control loop.
//   - frame(dt) is called by R3F's useFrame every animation frame; it does
//     everything the old rAF loop did EXCEPT renderer.render (R3F renders).
//   - UI state flows out through the zustand store (throttled), UI intents
//     flow back in through gameApi.

import * as THREE from "three";
import { preparePhysicsXml } from "./physics-model.js";
import { paceControlLoop } from "./control-loop-clock.js";
import beakColliders from "./beak-colliders.js";
import { readPayloadContacts, createContactAudit, recordContactStep, createContactWindow, addContactSample, averageContactWindow } from "./contact-observer.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { signed } from "./signed.js";
import {
  POLICIES, JOINT_NAMES, DEFAULT_POSE, NUM_JOINTS, OBS_SIZE, CMD_SIZE,
  ACTION_SCALE, TIMESTEP, DECIMATION, CTRL_DT,
  VEL_FWD, VEL_BACK, VEL_ANG,
  GROUND_PICK_PERIOD_S, GROUND_PICK_END_PHASE,
  BALL_RADIUS, BALL_PARK_POS,
  PAYLOAD_CAP_HALF_X, PAYLOAD_CAP_HALF_Y, PAYLOAD_CAP_HALF_Z,
  PAYLOAD_MASS, PAYLOAD_SPAWN_DIST, PAYLOAD_LATERAL,
  PAYLOAD_PEDESTAL_HALF_XY, PAYLOAD_PEDESTAL_HALF_Z,
  PAYLOAD_REST_Z, PAYLOAD_PARK_POS,
  ARENA_HALF, SPAWN_X, SPAWN_Y,
  RELIEF_BUMPS, RELIEF_HMAX, RELIEF_GRID, RELIEF_SINK, RELIEF_RATE,
} from "./constants.js";
import { loadProps, propColliders } from "./props.js";
import {
  buildRig, cloneRig, loadKinematics, setJoint, setJawOpen, JAW_MAX_OPEN,
  MODEL_DIR, MESH_VERSION, ROBOT_KINEMATICS,
  loadGlbGeometries, geometryToBinaryStl,
} from "./duck.js";
import {
  VARIANTS, materialHookFor, DEFAULT_VARIANT, applyVariant,
} from "./variants.js";
import { Controller } from "./controls/controller.js";
import { KeyboardSource } from "./controls/keyboard.js";
import { GamepadSource } from "./controls/gamepad.js";
import { TouchSource } from "./controls/touch.js";
import * as fx from "./fx/fx-wireframe.js";
import { createCeremony, CAM_RESET_S } from "./ceremony.js";
import { createBallActor } from "./ball-actor.js";
import { initGhosts } from "./ghosts.js";
import { makeInfiniteGrid, makeArenaWalls } from "./arena.js";
import { createBallVisual } from "./ball-visual.js";
import { createPayloadVisual } from "./payload-visual.js";
import { useGame, gameApi, bootLine, bootNote, bootHalt } from "../store.js";

// Physics + inference runtimes stay on the CDN, exactly like the pre-Vite
// app: mujoco.js resolves its .wasm sidecar relative to its own URL, and
// onnxruntime fetches its wasm from wasmPaths - neither ever touches the
// bundle. @vite-ignore keeps Rollup's static analysis out of it.
const MUJOCO_URL = "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.11.0/mujoco.js";
const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.min.mjs";

let bootStarted = false;

// HMR teardown for the ghost session: invalidating this module (directly or
// via an edit to ghosts.js) used to stack a live 15 Hz broadcast interval
// plus a ghost room per reload (the historical "stale module" bug class).
const liveGhostSessions = new Set();
if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    for (const g of liveGhostSessions) g.destroy();
    liveGhostSessions.clear();
  });
}

export async function bootGame({ scene, camera, renderer }) {
  if (bootStarted) return;
  bootStarted = true;
  try {
    await boot({ scene, camera, renderer });
  } catch (err) {
    console.error("[game] boot failed", err);
    bootHalt(err?.message || String(err));
  }
}

async function boot({ scene, camera, renderer }) {
  const setStore = useGame.setState;
  const store = useGame.getState;

  bootNote("Microduck BIOS v1.0");
  bootLine("MEMORY CHECK")("640K OK");
  bootLine("DUCK FIRMWARE")("PRESENT");

  // Surface async boot failures in the BIOS halt screen. Gated on the boot
  // still being in flight: post-boot async noise (ghost relay hiccups,
  // audio autoplay rejections...) must NOT cue the halt screen.
  const bootGuard = (e, msg) => {
    if (!store().bootDone && !store().bootFailed) bootHalt(msg);
  };
  window.addEventListener("unhandledrejection", (e) => {
    console.error("[game] unhandled rejection", e.reason);
    bootGuard(e, e.reason?.message || String(e.reason));
  });
  window.addEventListener("error", (e) => {
    console.error("[game] window error", e.message);
    bootGuard(e, e.message);
  });

  // Halting at the failure site: a rejected await inside this async boot
  // would otherwise only surface through the caller's catch.
  const traced = (label, p) => {
    const done = bootLine(label);
    return p.then(
      (v) => { done("OK"); return v; },
      (err) => {
        done("FAILED");
        console.error(`[game] ${label} FAILED`, err);
        bootHalt(err?.message || String(err));
        throw err;
      },
    );
  };

  // ── Runtimes (CDN) ──────────────────────────────────────────────────
  const [{ default: loadMujocoFactory }, ort] = await traced(
    "RUNTIME MODULES",
    Promise.all([
      import(/* @vite-ignore */ MUJOCO_URL),
      import(/* @vite-ignore */ ORT_URL),
    ]),
  );
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
  ort.env.wasm.numThreads = 1; // static hosting sends no COOP/COEP headers

  // ── MJCF preparation ────────────────────────────────────────────────
  // robot_allcollisions.xml is what infer_policy.py's scene.xml includes:
  // it carries body/shell collision geoms that robot_walk.xml lacks, which
  // the sitstand policy needs (a sit rests the trunk on the ground).
  // Visual meshes are irrelevant to the dynamics: every body carries an
  // explicit <inertial>, and visual geoms have contype=0 conaffinity=0.
  // Stripping them means the MuJoCo VFS only needs the ~10 meshes
  // referenced by collision geoms. Works for both variants.
  async function buildPhysicsXml(xmlFile) {
    const src = await (await fetch(signed(`${MODEL_DIR}/${xmlFile}`))).text();
    return preparePhysicsXml(src, { DOMParser, XMLSerializer, colliders: propColliders(), JAW_MAX_OPEN });
  }

  // ── Boot physics + policy in parallel with the render rig ────────────
  const [mujoco, { xml, meshFiles }, k] = await Promise.all([
    traced("MUJOCO WASM", loadMujocoFactory()),
    traced("PHYSICS MJCF", buildPhysicsXml("robot_allcollisions.xml")),
    traced("KINEMATICS", loadKinematics(ROBOT_KINEMATICS)),
  ]);

  const doneMeshes = bootLine("MESH ASSETS");
  const vfs = new mujoco.MjVFS();
  const vfsFiles = new Set();
  async function addMeshesToVfs(files) {
    const geoms = await loadGlbGeometries();
    await Promise.all(
      files.map(async (f) => {
        if (vfsFiles.has(f)) return;
        vfsFiles.add(f);
        // Body meshes live in the visual GLB: rebuild binary STL in memory
        // so MuJoCo never triggers a second download. Files not in the GLB
        // still fetch as STL.
        const entry = geoms.get(f);
        const buf = entry
          ? geometryToBinaryStl(entry.welded)
          : await (await fetch(signed(`${MODEL_DIR}/meshes/${f}?v=${MESH_VERSION}`), { cache: "force-cache" })).arrayBuffer();
        // meshdir="assets" in the MJCF, so the compiler looks up "assets/<f>".
        vfs.addBuffer(`assets/${f}`, new Uint8Array(buf));
      }),
    );
  }
  try {
    await addMeshesToVfs(meshFiles);
  } catch (err) {
    doneMeshes("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
  doneMeshes(`${meshFiles.length} FILES`);

  const sessions = {};
  let currentVariant = DEFAULT_VARIANT;
  const rigPromise = (async () => {
    const doneRig = bootLine("RENDER RIG");
    try {
      const builtRig = await buildRig(k, { materialForMesh: materialHookFor(VARIANTS[currentVariant]) });
      doneRig("OK");
      return builtRig;
    } catch (err) {
      doneRig("FAILED");
      bootHalt(err?.message || String(err));
      throw err;
    }
  })();
  // Boot policies with a live [n/7] counter on the BIOS line.
  const donePolicies = bootLine("LOADING POLICIES");
  const sessionOpts = { executionProviders: ["wasm"] };
  let policiesLoaded = 0;
  const bootPolicy = (url) =>
    ort.InferenceSession.create(signed(url), sessionOpts).then((s) => {
      donePolicies.progress(`${++policiesLoaded}/7`);
      return s;
    });
  try {
    [sessions.walk, sessions.sitstand, sessions.roll, sessions.kickL, sessions.kickR,
     sessions.groundpick, sessions.stand] =
      await Promise.all([
        bootPolicy(POLICIES.walk),
        bootPolicy(POLICIES.sitstand),
        bootPolicy(POLICIES.roll),
        bootPolicy(POLICIES.kickL),
        bootPolicy(POLICIES.kickR),
        bootPolicy(POLICIES.groundpick),
        bootPolicy(POLICIES.stand),
      ]);
  } catch (err) {
    donePolicies("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
  donePolicies("7/7");

  const doneCompile = bootLine("COMPILING PHYSICS");
  let model, data;
  try {
    model = mujoco.MjModel.from_xml_string(xml, vfs);
    data = new mujoco.MjData(model);
  } catch (err) {
    doneCompile("FAILED");
    bootHalt(err?.message || String(err));
    throw err;
  }
  doneCompile("COMPILED");

  // Addresses resolved once for the compiled model. qpos/qvel/sensordata
  // views are re-read at each use: the WASM heap can grow and detach
  // earlier TypedArray views.
  function resolveAddrs(model) {
    return {
      qposAdr: JOINT_NAMES.map((n) => model.jnt(n).qposadr),
      dofAdr: JOINT_NAMES.map((n) => model.jnt(n).dofadr),
      gyroAdr: model.sensor("imu_ang_vel").adr,
      trunkId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "trunk_base"),
      standKeyId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY.value, "STAND"),
      ballQposAdr: model.jnt("ball_freejoint").qposadr,
      ballDofAdr: model.jnt("ball_freejoint").dofadr,
      payloadQposAdr: model.jnt("payload_freejoint").qposadr,
      payloadDofAdr: model.jnt("payload_freejoint").dofadr,
      payloadGripGeomIds: ["payload_geom"].map(
        (name) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, name),
      ),
      mouthSiteId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, "mouth_tip"),
      beakTipSiteId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE.value, "beak_tip"),
      beakUpperGripSiteId: mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE.value, "beak_upper_grip",
      ),
      beakLowerGripSiteId: mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE.value, "beak_lower_grip",
      ),
      beakQposAdr: model.jnt("beak_joint").qposadr,
      beakActuatorId: mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR.value, "beak_actuator",
      ),
      beakLowerGeomIds: ["jaw", "jaw_soft"].flatMap(source => beakColliders[source].parts.map((_, i) => `beak_${source}_contact_${i}`)).map(
        (name) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, name),
      ),
      beakUpperGeomIds: beakColliders.soft_mouth_top.parts.map((_, i) => `beak_soft_mouth_top_contact_${i}`).map(
        (name) => mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM.value, name),
      ),
    };
  }
  const {
    qposAdr, dofAdr, gyroAdr, trunkId, standKeyId,
    ballQposAdr, ballDofAdr,
    payloadQposAdr, payloadDofAdr, payloadGripGeomIds,
    mouthSiteId, beakTipSiteId, beakUpperGripSiteId, beakLowerGripSiteId,
    beakQposAdr, beakActuatorId, beakLowerGeomIds, beakUpperGeomIds,
  } = resolveAddrs(model);

  const velLims = () => [VEL_FWD, VEL_BACK, VEL_ANG];

  const lastAction = new Float32Array(NUM_JOINTS);
  const obs = new Float32Array(OBS_SIZE);
  const cmd = new Float32Array(CMD_SIZE); // [vx, vy, wz, head(4), body(6)]
  // Input controller: keyboard + gamepad + touch sources merged into one
  // continuous command + discrete action surface, in priority order.
  const kbSource = new KeyboardSource({ getVelocityLimits: () => velLims() });
  const padSource = new GamepadSource({ getVelocityLimits: () => velLims() });
  const touchSource = new TouchSource({ getVelocityLimits: () => velLims() });
  // Keyboard last: it reads zero when idle, so it doubles as the fallback.
  const controller = new Controller({ sources: [padSource, touchSource, kbSource] });
  // Right-stick camera state, read by the telemetry before the camera-orbit
  // section below has evaluated.
  let padOrbitLive = false;
  // Robot input gate: twist commands, mode changes, rolls, kicks and ball
  // spawns all stay inert until the entrance sequence has fully played out.
  let inputLocked = true;
  let ceremony = null;
  let ball = null;
  let payloadPose = null;
  let stickers = null; // comic popups, currently disabled

  let mode = "walk"; // "walk" | "stand" | "sitstand" | "roll" | "kickL" | "kickR" | "groundpick"
  let sitFlag = 0;
  const isKick = () => mode === "kickL" || mode === "kickR";

  // HEAD mode (runtime-faithful, pad Y): locomotion is zeroed and both
  // sticks drive the head command slots cmd[3..6] = [neck_pitch,
  // head_pitch, head_yaw, head_roll]. Targets are stick * HEAD_MAX,
  // EMA-smoothed at 50 Hz in buildObs like the runtime (alpha 0.2).
  // Offsets PERSIST when leaving head mode; only a sim reset zeroes them.
  let headMode = false;
  const HEAD_MAX = 2.5; // rad at full deflection (runtime head_max)
  const HEAD_ALPHA = 0.2;
  // Stick-to-joint polarity (deflections come in up/left = +1), tuned
  // visually in the sim: cmd + means UP for neck_pitch but DOWN for
  // head_pitch, LEFT for head_yaw but RIGHT-tilt for head_roll - hence
  // the mixed signs, so stick up = look up, stick left = turn/tilt left.
  // Order matches cmd[3..6] = [neck_pitch, head_pitch, head_yaw, head_roll].
  const HEAD_SIGNS = new Float32Array([1, -1, 1, -1]);
  const headTarget = new Float32Array(4);
  const headSmooth = new Float32Array(4);
  // Local-only kickable ball: false while parked at the keyframe spot
  // (mesh hidden), true once popped in front of the duck.
  let ballActive = false;
  let payloadActive = false;
  let payloadHeld = false;
  let jawTargetOpen = 0;
  let gripTargetAngle = null;
  let lowerContactAge = Infinity;
  let upperContactAge = Infinity;
  const contactScratch = new mujoco.DoubleBuffer(6);
  let contactAudit = createContactAudit();
  let payloadContact = { lowerForce: 0, upperForce: 0, maxPenetration: 0, otherSupport: false, count: 0 };
  let stableHoldSteps = 0;

  // The twist the policy actually receives. Mid-roll every movement input
  // is ignored (zero twist) until the roll hands back to walk on its own.
  // HEAD mode also zeroes it: the runtime stops the robot while the
  // sticks drive the head.
  const ZERO_CMD = new Float32Array(3);
  function effectiveCmd() {
    if (inputLocked || headMode || mode === "roll" ||
        mode === "groundpick" || isKick() || postKickLock > 0 || recovery)
      return ZERO_CMD;
    return controller.getCommand();
  }
  let rollRun = null;
  let pickRun = null;
  let pickPreparing = false;
  let kickRun = null;
  let KICK_STEPS = 25;
  // Post-kick grace: keep commands zeroed for a beat after the kick window
  // hands back to walk. Step-counted like everything else.
  const POST_KICK_LOCK_STEPS = 20; // 0.4 s at 50 Hz
  let postKickLock = 0;

  // Pending mode-transition timers (sit hand-over, stand-up hand-back).
  let sitTimer = null;
  let standTimer = null;
  let fallenSince = null;

  // ── Automatic fall recovery (legs walk mode only) ────────────────────
  // Mirrors the runtime's --fall-detect state machine (main.rs ~3658):
  // a debounced tip (gz > -0.5 for 0.2 s) freezes ctrl on the current
  // pose for a short settle (the runtime goes limp), then hands the duck
  // to the stand policy with all commands zeroed until it's been upright
  // (gz < -0.85) for a full second. If it can't get up within 6 s, fall
  // back to the old kill: resetSim + materialization. Sit/roll/kick/
  // groundpick and the entrance lock keep the plain kill.
  const FALL_DEBOUNCE_STEPS = 10; // 0.2 s of gz > -0.5 before triggering
  const FALL_SETTLE_STEPS = 15; // 0.3 s ctrl freeze once triggered
  const RECOVER_UPRIGHT_STEPS = 50; // 1 s of gz < -0.85 to declare recovered
  const RECOVER_GIVEUP_STEPS = 300; // 6 s of stand attempts before reset
  let recovery = null; // null | { state: "fallen"|"recovering", steps, uprightSteps }
  let fallDebounce = 0;

  function clearModeTimers() {
    clearTimeout(sitTimer); sitTimer = null;
    clearTimeout(standTimer); standTimer = null;
    pickPreparing = false;
  }

  // ── Mouse grab, physics side (MuJoCo-viewer-style perturbation) ───────
  // While a grab is live, EVERY PHYSICS SUBSTEP writes a spring-damper
  // force on the grabbed free body via xfrc_applied (world frame),
  // pulling it toward the cursor target. Formula and gains mirror the
  // native viewer's mjv_applyPerturbForce (engine_vis_interact.c):
  //   F = -stiffness*mass*(pos - ref) - sqrt(stiffness)*mass*vel
  // with stiffness = m->vis.map.stiffness default (100) and the damping
  // coefficient sqrt(stiffness) exactly as MuJoCo computes it. Two
  // deliberate departures: mass is the subtree mass instead of the
  // Jacobian-derived localmass (equivalent for a free body pulled at its
  // root, and the bindings expose no mj_jac), and the force acts
  // torque-free at the freejoint origin instead of at the picked point
  // (the viewer adds moment_arm x F; skipping it avoids spinning the duck
  // the walking policy would then fight). The early per-CONTROL-step
  // version of this (50 Hz zero-order hold, damping on 4-substep-stale
  // velocity) was the jitter the user felt: a stiff spring held over 20 ms
  // limit-cycles. Per-substep application is what the viewer does.
  // Pointer wiring (raycast pick, target plane, cursor) lives after the
  // camera section below; this block stays above resetSim so the control
  // loop and resets can reference it during boot.
  const GRAB_STIFFNESS = 100; // MuJoCo vis.map.stiffness default
  const GRAB_DAMPING = Math.sqrt(GRAB_STIFFNESS); // viewer's damping coefficient
  const GRAB_MAX_ACC = 200; // safety clamp only - the viewer has none
  let grab = null; // { bodyId, qAdr, dofAdr, mass, target: [x,y,z] MJCF }
  let endGrabHook = () => {}; // reassigned by the pointer wiring
  function applyGrabForce() {
    if (!grab) return;
    const qpos = data.qpos, qvel = data.qvel;
    let fx = GRAB_STIFFNESS * (grab.target[0] - qpos[grab.qAdr]) - GRAB_DAMPING * qvel[grab.dofAdr];
    let fy = GRAB_STIFFNESS * (grab.target[1] - qpos[grab.qAdr + 1]) - GRAB_DAMPING * qvel[grab.dofAdr + 1];
    let fz = GRAB_STIFFNESS * (grab.target[2] - qpos[grab.qAdr + 2]) - GRAB_DAMPING * qvel[grab.dofAdr + 2];
    const n = Math.hypot(fx, fy, fz);
    if (n > GRAB_MAX_ACC) {
      const s = GRAB_MAX_ACC / n;
      fx *= s; fy *= s; fz *= s;
    }
    // Fresh view each call: the WASM heap can grow and detach old ones.
    const xfrc = data.xfrc_applied;
    const a = grab.bodyId * 6;
    xfrc[a] = grab.mass * fx;
    xfrc[a + 1] = grab.mass * fy;
    xfrc[a + 2] = grab.mass * fz;
  }
  function releaseGrabForce() {
    if (!grab) return;
    const xfrc = data.xfrc_applied;
    const a = grab.bodyId * 6;
    xfrc[a] = 0; xfrc[a + 1] = 0; xfrc[a + 2] = 0;
    grab = null;
  }

  function resetSim() {
    // A live grab must not survive a reset: the per-step spring would
    // immediately yank the respawned duck toward the stale cursor target.
    endGrabHook();
    // Single reset path: Space, fall-kill, failed roll.
    clearModeTimers();
    rollRun = null;
    kickRun = null;
    pickRun = null;
    postKickLock = 0;
    fallenSince = null;
    recovery = null;
    fallDebounce = 0;
    mode = "walk";
    // Head mode exits and its offsets DO reset here (the one place).
    headMode = false;
    padSource.headMode = false;
    headTarget.fill(0);
    headSmooth.fill(0);
    jawTargetOpen = 0;
    gripTargetAngle = null;
    lowerContactAge = Infinity;
    upperContactAge = Infinity;
    stableHoldSteps = 0;
    mujoco.mj_resetDataKeyframe(model, data, standKeyId);
    mujoco.mj_forward(model, data);
    lastAction.fill(0);
    sitFlag = 0;
    // Park the ball in physics immediately; if it was on screen, the
    // reverse scan peels it away at its last pose. A queued B-respawn is
    // cancelled: a reset means no ball.
    ball?.despawn({ cancelQueued: true, parkPhysics: parkBallPhysics });
    ballActive = false;
    payloadActive = false;
    payloadHeld = false;
    syncButtons();
    ceremony?.playRespawn();
  }
  resetSim();

  function parkBallPhysics() {
    const qpos = data.qpos, qvel = data.qvel;
    qpos[ballQposAdr] = 50;
    qpos[ballQposAdr + 1] = 0;
    qpos[ballQposAdr + 2] = BALL_RADIUS;
    qpos[ballQposAdr + 3] = 1;
    qpos[ballQposAdr + 4] = 0;
    qpos[ballQposAdr + 5] = 0;
    qpos[ballQposAdr + 6] = 0;
    for (let i = 0; i < 6; i++) qvel[ballDofAdr + i] = 0;
    mujoco.mj_forward(model, data);
    ballActive = false;
  }

  // Pop / respawn the ball ~0.35 m in front of the duck, with a small
  // random heading + distance jitter. If the ball is already on screen,
  // peel it away first (reverse scan) and pop the new one when that
  // finishes - same appear/disappear pair as the duck's wireframe ceremony.
  function spawnBall(opts = {}) {
    if (inputLocked && !opts.fromQueue) return;
    if (!ball) return;
    if (ball.visual !== "hidden") {
      ball.queueRespawn();
      ball.despawn({ parkPhysics: parkBallPhysics });
      return;
    }
    const qpos = data.qpos, qvel = data.qvel;
    const yaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    const heading = yaw + (Math.random() - 0.5) * 0.7;
    const dist = 0.35 + (Math.random() - 0.5) * 0.1;
    const lim = ARENA_HALF - BALL_RADIUS - 0.05;
    const clamp = (v) => Math.min(lim, Math.max(-lim, v));
    qpos[ballQposAdr] = clamp(qpos[0] + Math.cos(heading) * dist);
    qpos[ballQposAdr + 1] = clamp(qpos[1] + Math.sin(heading) * dist);
    qpos[ballQposAdr + 2] = BALL_RADIUS + 0.02;
    qpos[ballQposAdr + 3] = 1;
    qpos[ballQposAdr + 4] = 0;
    qpos[ballQposAdr + 5] = 0;
    qpos[ballQposAdr + 6] = 0;
    for (let i = 0; i < 6; i++) qvel[ballDofAdr + i] = 0;
    mujoco.mj_forward(model, data);
    ballActive = true;
    // Snap the mesh to the new pose BEFORE the scan starts: the FX
    // recomputes its bbox from the live mesh.
    ball.poseFromQpos(qpos, ballQposAdr);
    ball.appear();
    stickers?.pop("spawn");
  }

  // Place the small target on the measured inner-mouth path of the
  // ground-pick policy.
  function spawnPayload(opts = {}) {
    if (inputLocked && !opts.fromQueue) return;
    const qpos = data.qpos, qvel = data.qvel;
    // Reset to the fixed pedestal, never to a robot-relative moving target.
    qpos[payloadQposAdr] = SPAWN_X + PAYLOAD_SPAWN_DIST;
    qpos[payloadQposAdr + 1] = SPAWN_Y + PAYLOAD_LATERAL;
    qpos[payloadQposAdr + 2] = PAYLOAD_REST_Z + 0.002;
    qpos[payloadQposAdr + 3] = 1;
    qpos[payloadQposAdr + 4] = 0;
    qpos[payloadQposAdr + 5] = 0;
    qpos[payloadQposAdr + 6] = 0;
    for (let i = 0; i < 6; i++) qvel[payloadDofAdr + i] = 0;
    payloadActive = true;
    payloadHeld = false;
    gripTargetAngle = null;
    lowerContactAge = Infinity;
    upperContactAge = Infinity;
    mujoco.mj_forward(model, data);
    syncButtons();
  }

  // There is deliberately no target-side spring, weld, or qpos write here.
  // The position actuator only closes the articulated lower bill. MuJoCo
  // contacts and friction on the actual mouth collision volumes decide
  // whether the free payload lifts, slips, or falls.
  function clearPayloadGrip() {
    const hadGrip = gripTargetAngle !== null || payloadHeld;
    gripTargetAngle = null;
    payloadHeld = false;
    stableHoldSteps = 0;
    return hadGrip;
  }

  function updatePayloadGripState() {
    if (!payloadActive) return;
    const lowerContact = payloadContact.lowerForce > 0.002;
    const upperContact = payloadContact.upperForce > 0.002;
    lowerContactAge = lowerContact ? 0 : lowerContactAge + 1;
    upperContactAge = upperContact ? 0 : upperContactAge + 1;
    const qpos = data.qpos;
    const gripUpper = data.site_xpos.subarray(beakUpperGripSiteId * 3, beakUpperGripSiteId * 3 + 3);
    const gripLower = data.site_xpos.subarray(beakLowerGripSiteId * 3, beakLowerGripSiteId * 3 + 3);
    const gripMx = (gripUpper[0] + gripLower[0]) * 0.5;
    const gripMy = (gripUpper[1] + gripLower[1]) * 0.5;
    const gripMz = (gripUpper[2] + gripLower[2]) * 0.5;
    const distance = Math.hypot(
      qpos[payloadQposAdr] - gripMx,
      qpos[payloadQposAdr + 1] - gripMy,
      qpos[payloadQposAdr + 2] - gripMz,
    );
    const centered =
      Math.abs(qpos[payloadQposAdr] - gripMx) < 0.018 &&
      Math.abs(qpos[payloadQposAdr + 1] - gripMy) < 0.012 &&
      Math.abs(qpos[payloadQposAdr + 2] - gripMz) < 0.015;
    if (gripTargetAngle === null && lowerContact && upperContact && centered) {
      // Keep commanding the hinge closed; opposing collision forces decide
      // the actual angle. This tactile controller latch closes as soon as
      // both pads touch; it is never a payload latch.
      gripTargetAngle = 0;
      // Pause the blind head trajectory briefly while the hinge develops
      // clamping force. The payload remains a free body; only the policy
      // phase is held, like a contact-triggered grasp controller.
      if (mode === "groundpick" && pickRun && !pickRun.contactHoldUsed) {
        pickRun.contactHold = 20;
        pickRun.contactHoldUsed = true;
      }
    }

    // Losing either side of the physical contact pair means the grasp failed.
    if (gripTargetAngle !== null &&
        (lowerContactAge > 3 || upperContactAge > 3 || distance > 0.04)) {
      clearPayloadGrip();
    }
    const lifted = qpos[payloadQposAdr + 2] > PAYLOAD_REST_Z + 0.002;
    const payloadBody = model.geom_bodyid[payloadGripGeomIds[0]];
    const unassisted = data.xfrc_applied.subarray(payloadBody * 6, payloadBody * 6 + 6)
      .every(value => Math.abs(value) < 1e-6);
    const supported = gripTargetAngle !== null && lowerContact && upperContact && lifted &&
      unassisted && !payloadContact.otherSupport && payloadContact.maxPenetration < 0.0005;
    stableHoldSteps = supported ? stableHoldSteps + 1 : 0;
    const nowHeld = stableHoldSteps * CTRL_DT >= 0.2;
    if (payloadHeld !== nowHeld) {
      payloadHeld = nowHeld;
      syncButtons();
    }
  }

  function releasePayload() {
    if (!clearPayloadGrip()) return false;
    syncButtons();
    return true;
  }

  // ── Observation ─────────────────────────────────────────────────────
  const _q = new THREE.Quaternion();
  const _g = new THREE.Vector3();

  function buildObs() {
    const qpos = data.qpos, qvel = data.qvel, sens = data.sensordata;
    let i = 0;
    for (let a = 0; a < 3; a++) obs[i++] = sens[gyroAdr + a];
    // projected gravity: world -z rotated into the trunk frame
    const xq = data.xquat.subarray(trunkId * 4, trunkId * 4 + 4); // [w, x, y, z]
    _q.set(xq[1], xq[2], xq[3], xq[0]).conjugate();
    _g.set(0, 0, -1).applyQuaternion(_q);
    obs[i++] = _g.x; obs[i++] = _g.y; obs[i++] = _g.z;
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qpos[qposAdr[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qvel[dofAdr[j]];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = lastAction[j];
    // command: walking uses the twist; sitstand uses cmd[0] as the posture
    // flag; the ground-pick one-shot carries its phase encoding in the vel
    // slots ([cos, sin, 0]).
    cmd.fill(0, 0, 3);
    if (mode === "sitstand") {
      cmd[0] = sitFlag;
    } else if (mode === "groundpick" && pickRun) {
      const a = 2 * Math.PI * pickRun.phase;
      cmd[0] = Math.cos(a);
      cmd[1] = Math.sin(a);
    } else {
      const c = effectiveCmd();
      cmd[0] = c[0]; cmd[1] = c[1]; cmd[2] = c[2];
    }
    // Head slots cmd[3..6]: EMA toward the stick targets at 50 Hz (this
    // runs once per control step), exactly the runtime's smoothing. Kept
    // filled outside head mode too - offsets persist like on the robot.
    for (let h = 0; h < 4; h++) headSmooth[h] += HEAD_ALPHA * (headTarget[h] - headSmooth[h]);
    // Ground pick parity: the runtime zero-pads the head (and body) slots
    // for its obs (mjlab's zero_command_padding), so persisted head
    // offsets must not leak into the pick policy's command buffer. Fall
    // recovery zeroes them too: the stand policy gets an all-zero command.
    const gpZero = mode === "groundpick" || mode === "stand" || recovery !== null;
    cmd[3] = gpZero ? 0 : headSmooth[0]; cmd[4] = gpZero ? 0 : headSmooth[1];
    cmd[5] = gpZero ? 0 : headSmooth[2]; cmd[6] = gpZero ? 0 : headSmooth[3];
    for (let c = 0; c < CMD_SIZE; c++) obs[i++] = cmd[c];
    return obs;
  }

  // The ONNX session for the current mode; the fall-recovery state machine
  // overrides everything with the get-up policy while it owns the duck.
  const activeSession = () => {
    if (recovery?.state === "recovering") return sessions.stand;
    return sessions[mode];
  };

  // ── Control loop (50 Hz, async because ONNX inference is async) ──────
  let ctrlHz = 0;

  // Fresh projected-gravity z straight from the trunk pose (buildObs is
  // skipped during the fall-recovery settle, so obs[5] can go stale).
  function projGravZ() {
    const xq = data.xquat.subarray(trunkId * 4, trunkId * 4 + 4); // [w, x, y, z]
    _q.set(xq[1], xq[2], xq[3], xq[0]).conjugate();
    _g.set(0, 0, -1).applyQuaternion(_q);
    return _g.z;
  }

  // Dead pose: "fallen" = trunk tilted past ~60 deg or sunk below the
  // floor. NaN/Inf is a solver explosion: no grace, reset on the spot.
  // In walk mode a debounced fall now goes to the recovery state machine
  // instead of the kill; everywhere else (sit, one-shots, entrance lock)
  // the old grace-then-reset behavior stands.
  function poseIsDead() {
    const z = data.qpos[2];
    const gz = projGravZ();
    if (!Number.isFinite(z) || !Number.isFinite(gz)) return "exploded";
    if (gz > -0.5 || z < 0.02) return "fallen";
    return null;
  }

  async function controlStep() {
    driveRelief(CTRL_DT); // kinematic terrain, written before the physics steps
    // Settle phase: ctrl frozen on the pose held at the fall (approximates
    // the runtime's limp beat), physics keeps stepping, no inference.
    if (recovery?.state !== "fallen") {
      const feeds = { obs: new ort.Tensor("float32", buildObs(), [1, OBS_SIZE]) };
      const out = await activeSession().run(feeds);
      const act = out.actions.data;
      lastAction.set(act);
      const ctrl = data.ctrl;
      for (let j = 0; j < NUM_JOINTS; j++) ctrl[j] = DEFAULT_POSE[j] + act[j] * ACTION_SCALE;
    }
    data.ctrl[beakActuatorId] = gripTargetAngle ?? (JAW_MAX_OPEN * jawTargetOpen);
    const contactWindow = createContactWindow();
    for (let s = 0; s < DECIMATION; s++) {
      applyGrabForce(); // mouse perturbation, fresh velocity every substep
      mujoco.mj_step(model, data);
      if (payloadActive) {
        payloadContact = readPayloadContacts(mujoco, model, data, payloadGripGeomIds,
          beakLowerGeomIds, beakUpperGeomIds, contactScratch);
        addContactSample(contactWindow, payloadContact);
        recordContactStep(contactAudit, payloadContact, data,
          model.geom_bodyid[payloadGripGeomIds[0]], payloadQposAdr, TIMESTEP, payloadHeld);
      }
    }
    if (contactWindow.samples) payloadContact = averageContactWindow(contactWindow);
    updatePayloadGripState();

    const death = poseIsDead();
    if (death === "exploded") {
      resetSim();
    } else if (recovery) {
      // Recovery state machine owns the duck: settle -> stand policy ->
      // hysteresis exit (upright for a full second) or 6 s give-up reset.
      recovery.steps++;
      if (recovery.state === "fallen") {
        if (recovery.steps >= FALL_SETTLE_STEPS) {
          recovery = { state: "recovering", steps: 0, uprightSteps: 0 };
          lastAction.fill(0);
          syncButtons();
        }
      } else {
        recovery.uprightSteps = projGravZ() < -0.85 ? recovery.uprightSteps + 1 : 0;
        if (recovery.uprightSteps >= RECOVER_UPRIGHT_STEPS) {
          recovery = null;
          mode = "walk";
          lastAction.fill(0);
          syncButtons();
        } else if (recovery.steps >= RECOVER_GIVEUP_STEPS) {
          resetSim();
        }
      }
    } else if (death === "fallen") {
      const recoverable = mode === "walk" &&
        !inputLocked && postKickLock === 0 && !standTimer;
      if (recoverable) {
        fallenSince = null;
        if (++fallDebounce >= FALL_DEBOUNCE_STEPS) {
          fallDebounce = 0;
          exitHeadMode();
          recovery = { state: "fallen", steps: 0 };
          syncButtons();
        }
      } else {
        fallDebounce = 0;
        const now = performance.now();
        const graceMs = mode === "roll" ? 5000 : 1000;
        fallenSince ??= now;
        if (now - fallenSince > graceMs) resetSim();
      }
    } else {
      fallDebounce = 0;
      fallenSince = null;
    }

    // Ball respawn watchdog: outside the arena bounds means "escaped
    // through a solver glitch", bring it back near the duck.
    if (ballActive) {
      const q = data.qpos;
      const escaped =
        Math.abs(q[ballQposAdr]) > ARENA_HALF + 0.1 ||
        Math.abs(q[ballQposAdr + 1]) > ARENA_HALF + 0.1;
      if (escaped) spawnBall();
    }

    // Record a solver escape without silently respawning the payload.
    if (payloadActive && !payloadHeld) {
      const q = data.qpos;
      const escaped =
        Math.abs(q[payloadQposAdr]) > ARENA_HALF + 0.1 ||
        Math.abs(q[payloadQposAdr + 1]) > ARENA_HALF + 0.1 ||
        q[payloadQposAdr + 2] < -0.2;
      if (escaped && !contactAudit.escaped) contactAudit.escapes++;
      contactAudit.escaped = escaped;
    }

    if (postKickLock > 0 && mode === "walk") postKickLock--;

    // One-shot kick: fixed 0.5 s window like the robot runtime, then
    // straight back to walking. lastAction is NOT zeroed on either swap.
    if (isKick() && kickRun) {
      kickRun.steps++;
      if (kickRun.steps >= KICK_STEPS) {
        kickRun = null;
        mode = "walk";
        postKickLock = POST_KICK_LOCK_STEPS;
        syncButtons();
      }
    }

    // Ground-pick one-shot: advance the trained phase clock, ending
    // at the runtime's cycle end (phase 0.7 of a 4 s period, ~2.8 s).
    if (mode === "groundpick" && pickRun) {
      if (pickRun.contactHold > 0) pickRun.contactHold--;
      else pickRun.phase += CTRL_DT / GROUND_PICK_PERIOD_S;
      if (pickRun.phase >= GROUND_PICK_END_PHASE) {
        pickRun = null;
        mode = "walk";
        syncButtons();
      }
    }

    // One-shot roll, step-counted like the robot runtime: hand back to
    // walking once the trunk has tipped over and is upright again, or
    // after a hard window if the roll never initiated.
    if (mode === "roll" && rollRun) {
      rollRun.steps++;
      if (obs[5] > -0.3) rollRun.tipped = true;
      const upright = obs[5] < -0.85;
      const done = rollRun.tipped && upright && rollRun.steps >= 40;
      const expired = rollRun.steps >= 150; // 3 s, roll should long be over
      if (done || expired) {
        rollRun = null;
        mode = "walk";
        lastAction.fill(0);
        // Timed out mid-roll: don't hand a tipped duck to the walking
        // policy (it has no get-up skill).
        if (!upright) resetSim();
        syncButtons();
      }
    }
  }

  // ── Relief (prototype: the level itself gains gentle slopes) ────────
  // One analytic height function (cosine bumps, RELIEF_BUMPS) drives both
  // surfaces: the MuJoCo heightfield gets it sampled into hfield_data
  // once per compiled model, and the grid floor shader displaces its
  // vertices with the same function (uTopoScale uniform). Raising or
  // sinking the terrain = ramping one scalar that scales the hfield
  // z-size and the shader uniform together, so physics and visuals stay
  // the same surface at every moment of the transition. Trigger for now:
  // window.rl.setRelief(bool) (prototype - no UI yet).
  let reliefOn = false;
  let reliefScale = 0;
  let reliefGridMat = null; // assigned at scene wiring (grid built below)
  const reliefFilled = new WeakSet();
  function topoH(x, y) {
    let H = 0;
    for (const [cx, cy, h, r] of RELIEF_BUMPS) {
      const u = Math.hypot(x - cx, y - cy) / r;
      if (u < 1) H += h * (0.5 + 0.5 * Math.cos(Math.PI * u));
    }
    return H;
  }
  function fillHfield(m) {
    if (reliefFilled.has(m)) return;
    // Re-read the view on every fill: heap growth detaches TypedArrays.
    const n = RELIEF_GRID, hdata = m.hfield_data;
    for (let r = 0; r < n; r++) {
      const y = -ARENA_HALF + (2 * ARENA_HALF * r) / (n - 1);
      for (let c = 0; c < n; c++) {
        const x = -ARENA_HALF + (2 * ARENA_HALF * c) / (n - 1);
        hdata[r * n + c] = topoH(x, y) / RELIEF_HMAX;
      }
    }
    reliefFilled.add(m);
  }
  function driveRelief(dt) {
    fillHfield(model); // no-op once the compiled model is filled
    const target = reliefOn ? 1 : 0;
    if (reliefScale !== target) {
      const d = Math.max(-RELIEF_RATE * dt, Math.min(RELIEF_RATE * dt, target - reliefScale));
      reliefScale += d;
    }
    // z-size scales every bump; the floor keeps it strictly positive and,
    // combined with the geom's RELIEF_SINK offset, fully buried when off.
    model.hfield_size[2] = Math.max(reliefScale * RELIEF_HMAX, 1e-4);
    if (reliefGridMat) reliefGridMat.uniforms.uTopoScale.value = reliefScale;
  }
  let running = true;
  let controlLoopPromise = null;
  let controlLoopErrors = 0;
  let controlLoopLastStep = -Infinity;
  async function runControlLoop() {
    let next = performance.now();
    let count = 0, hzT0 = next;
    while (running) {
      await waitForVisible();
      if (!running) break;
      try {
        await controlStep();
      } catch (err) {
        // A transient ONNX/WebAssembly rejection must not permanently kill
        // every robot action. Leave the current state intact and retry after
        // a short backoff; frame() also restarts a loop that ever exits.
        controlLoopErrors++;
        console.error("[game] control loop recovered", err);
        await new Promise((resolve) => setTimeout(resolve, 100));
        next = performance.now();
        continue;
      }
      controlLoopLastStep = performance.now();
      count++;
      const now = performance.now();
      if (now - hzT0 > 500) {
        ctrlHz = (count * 1000) / (now - hzT0);
        count = 0; hzT0 = now;
      }
      next = await paceControlLoop(next, CTRL_DT * 1000);
    }
  }
  function ensureControlLoop() {
    if (!running || controlLoopPromise) return;
    controlLoopPromise = runControlLoop().finally(() => {
      controlLoopPromise = null;
    });
  }
  ensureControlLoop();

  // ── Scene wiring (grid, walls, rig, ball, arcade row) ────────────────
  // The grid/walls carry ceremony-driven uReveal uniforms and per-frame
  // focus updates, so the game owns them; lights and environment live in
  // the R3F layer.
  const grid = makeInfiniteGrid();
  scene.add(grid);
  reliefGridMat = grid.material; // relief drive mirrors uTopoScale into it
  const { wallMats, wallMeshes } = makeArenaWalls();
  for (const m of wallMeshes) scene.add(m);

  const rig = await rigPromise;
  scene.add(rig.placer);
  const trunkGroup = rig.bodies.get("trunk_base");

  // ── Cutscenes (entrance + respawn) ──────────────────────────────────
  ceremony = createCeremony({
    THREE, scene, camera, renderer, fx,
    getRig: () => rig,
    grid, wallMats,
    syncRig, startCameraReset,
    setLocked: (v) => {
      inputLocked = v;
      controller.setLocked(v);
      // A ball is always in play: pop one the moment the entrance or a
      // respawn ceremony hands control back. The small foam pickup block
      // respawns on the same hand-off, directly under the ground-pick path.
      if (!v && ball && !ballActive) spawnBall({ fromQueue: true });
      if (!v && payloadPose && !payloadActive) spawnPayload({ fromQueue: true });
    },
    flashReset: () => {},
  });

  const { group: ballGroup, mesh: ballMesh } = createBallVisual(renderer);
  scene.add(ballGroup);
  ball = createBallActor({
    THREE, scene, camera, renderer, fxModule: fx, mesh: ballMesh, group: ballGroup,
  });

  const payloadVisual = createPayloadVisual();
  payloadPose = payloadVisual.pose;
  scene.add(payloadVisual.group);

  // ── Prop library (wall/corner dressing + entrance FX) ────────────────
  // Every enabled def in props.js: loaded, real-size scaled, floor
  // snapped, wireframe-materialized with the ceremony (staggered after
  // the duck's scan cue). Physics-side, buildPhysicsXml planted one
  // static box per declared collider.
  const propGroups = await loadProps({
    THREE, GLTFLoader, signed, scene, camera, renderer, fx, ceremony,
  });

  // ── Camera: orbit controls + chase cam + reset glide ─────────────────
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(SPAWN_X, 0, -SPAWN_Y); // orbit around the spawn cell
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 0.25;
  controls.maxDistance = 3;
  controls.maxPolarAngle = Math.PI / 2 - 0.03;

  // Chase cam (default ON): each frame the camera eases toward a point
  // behind the duck's heading at the current orbit distance, while the
  // orbit target keeps easing to the trunk in syncRig. Implemented by
  // overwriting camera.position AFTER controls.update() so we never fight
  // OrbitControls' own spherical bookkeeping.
  let chaseCam = true;
  const CHASE_PITCH = 0.42; // rad above horizontal, keeps the floor in view
  const CHASE_EASE = 0.05;
  const _chasePos = new THREE.Vector3();
  const _chaseDir = new THREE.Vector3();
  // During one-shot rolls and kicks the trunk tumbles: hold the last
  // healthy yaw for the whole one-shot.
  let chaseHeldYaw = 0;
  // Heading hysteresis (Schmitt trigger): the walking gait wiggles the
  // trunk yaw ~±14 deg per step; two-layer EMA + engage/release thresholds
  // keep the camera steady while walking straight but responsive on turns.
  let chaseYawSmooth = 0;
  let chaseYawFollow = 0;
  let chaseYawTracking = false;
  const CHASE_YAW_SMOOTH_EASE = 0.04;
  const CHASE_YAW_ENGAGE = 0.17;
  const CHASE_YAW_RELEASE = 0.03;
  const CHASE_YAW_EASE = 0.10;
  const CHASE_YAW_EASE_TURN = 0.5;
  const wrapPi = (a) => Math.atan2(Math.sin(a), Math.cos(a));
  function updateChaseCam() {
    // Reset glide: one clean tween from wherever the camera is back to the
    // home framing. Runs instead of the chase logic and hands control back
    // to it on landing.
    if (camResetT0 !== null) {
      if (!chaseCam) { camResetT0 = null; return; }
      const t = (performance.now() - camResetT0) / 1000 / CAM_RESET_S;
      const e = t >= 1 ? 1 : t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
      camera.position.lerpVectors(_camFrom, _camTo, e);
      controls.target.lerpVectors(_tgtFrom, _tgtTo, e);
      camera.lookAt(controls.target);
      if (t >= 1) camResetT0 = null;
      return;
    }
    // Head mode: the camera freezes where it is. It keeps looking at the
    // duck for free - syncRig translates camera and target by the same
    // delta, and the duck isn't walking anyway (twist zeroed).
    if (headMode) return;
    if (!chaseCam) return;
    const qpos = data.qpos;
    let rawYaw;
    if (mode === "roll" || isKick()) {
      rawYaw = chaseHeldYaw;
    } else {
      rawYaw = Math.atan2(
        2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
        1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
      );
      chaseHeldYaw = rawYaw;
    }
    // "turning" reads the raw per-source wz commands (not the locked/merged
    // view) so an intentional turn engages on the first frame.
    const turning = controller.sources.some((s) => Math.abs(s.command[2]) > 0.05);
    chaseYawSmooth = wrapPi(
      chaseYawSmooth +
        wrapPi(rawYaw - chaseYawSmooth) * (turning ? CHASE_YAW_EASE_TURN : CHASE_YAW_SMOOTH_EASE),
    );
    const yawErr = wrapPi(chaseYawSmooth - chaseYawFollow);
    if (turning || Math.abs(yawErr) > CHASE_YAW_ENGAGE) chaseYawTracking = true;
    if (chaseYawTracking) {
      chaseYawFollow = wrapPi(
        chaseYawFollow + yawErr * (turning ? CHASE_YAW_EASE_TURN : CHASE_YAW_EASE),
      );
      if (!turning && Math.abs(yawErr) < CHASE_YAW_RELEASE) chaseYawTracking = false;
    }
    const yaw = chaseYawFollow;
    const dist = camera.position.distanceTo(controls.target);
    const horiz = dist * Math.cos(CHASE_PITCH);
    const vert = dist * Math.sin(CHASE_PITCH);
    // Duck forward in MJCF is (cos yaw, sin yaw, 0); Z-up -> Y-up maps it
    // to three-space (cos yaw, 0, -sin yaw). Behind = minus that.
    _chasePos.set(
      controls.target.x - Math.cos(yaw) * horiz,
      controls.target.y + vert,
      controls.target.z + Math.sin(yaw) * horiz,
    );
    camera.position.lerp(_chasePos, CHASE_EASE);
    // Re-project onto the orbit sphere: lerping between two points at the
    // same radius cuts the chord, which would slowly zoom the camera in
    // during large swings.
    _chaseDir.copy(camera.position).sub(controls.target);
    const len = _chaseDir.length();
    if (len > 1e-6) camera.position.copy(controls.target).addScaledVector(_chaseDir, dist / len);
    camera.lookAt(controls.target);
  }
  renderer.domElement.addEventListener("pointerdown", () => { chaseCam = false; });

  // Camera reset glide (owned by the respawn ceremony): back to the
  // page-load framing - the chase cam's ideal point behind the duck's
  // spawn heading, at the boot orbit distance.
  const CAM_HOME_DIST = camera.position.distanceTo(controls.target);
  let camResetT0 = null;
  const _camFrom = new THREE.Vector3(), _camTo = new THREE.Vector3();
  const _tgtFrom = new THREE.Vector3(), _tgtTo = new THREE.Vector3();
  function startCameraReset() {
    const qpos = data.qpos;
    const yaw = Math.atan2(
      2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5]),
      1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6]),
    );
    chaseHeldYaw = yaw;
    chaseYawSmooth = yaw;
    chaseYawFollow = yaw;
    chaseYawTracking = false;
    _tgtTo.set(qpos[0], qpos[2], -qpos[1]); // trunk at spawn, MJCF -> three
    const horiz = CAM_HOME_DIST * Math.cos(CHASE_PITCH);
    const vert = CAM_HOME_DIST * Math.sin(CHASE_PITCH);
    _camTo.set(
      _tgtTo.x - Math.cos(yaw) * horiz,
      _tgtTo.y + vert,
      _tgtTo.z + Math.sin(yaw) * horiz,
    );
    _camFrom.copy(camera.position);
    _tgtFrom.copy(controls.target);
    camResetT0 = performance.now();
    chaseCam = true; // reset always re-attaches the chase cam
  }

  // ── Mouse grab, pointer side (pick + drag target + cursor) ────────────
  // Pointer-down on the duck (or the live ball) grabs it; anywhere else
  // falls through to OrbitControls untouched. The pick is a three.js
  // raycast against the render rig (the WASM bindings expose no
  // mjv_select, and the rig IS the duck's collision-accurate silhouette
  // for mouse purposes). While dragging, the cursor is projected on a
  // camera-facing plane through the grab point - horizontal AND vertical
  // drags both work, so the duck can be lifted - and the target is
  // clamped inside the arena walls and to a sane height band. Desktop
  // mouse only: the touch overlay keeps its own controls.
  const GRAB_TARGET_ZMIN = 0.02, GRAB_TARGET_ZMAX = 0.45;
  const _grabRaycaster = new THREE.Raycaster();
  const _grabNdc = new THREE.Vector2();
  const _grabPlane = new THREE.Plane();
  const _grabHit = new THREE.Vector3();
  const _grabCamDir = new THREE.Vector3();
  function grabRayFrom(e) {
    const r = renderer.domElement.getBoundingClientRect();
    _grabNdc.set(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1,
    );
    _grabRaycaster.setFromCamera(_grabNdc, camera);
  }
  function grabPick() {
    const duckHit = _grabRaycaster.intersectObject(rig.placer, true)[0];
    const ballHit = ballActive ? _grabRaycaster.intersectObject(ballMesh, true)[0] : undefined;
    if (duckHit && (!ballHit || duckHit.distance <= ballHit.distance)) {
      return { kind: "duck", point: duckHit.point };
    }
    return ballHit ? { kind: "ball", point: ballHit.point } : null;
  }
  function updateGrabTarget() {
    if (!_grabRaycaster.ray.intersectPlane(_grabPlane, _grabHit)) return;
    const lim = ARENA_HALF - 0.05;
    // three (x, y, z) -> MJCF (x, -z, y), Z-up.
    grab.target[0] = Math.min(lim, Math.max(-lim, _grabHit.x));
    grab.target[1] = Math.min(lim, Math.max(-lim, -_grabHit.z));
    grab.target[2] = Math.min(GRAB_TARGET_ZMAX, Math.max(GRAB_TARGET_ZMIN, _grabHit.y));
  }
  function endGrab() {
    if (!grab) return;
    releaseGrabForce();
    controls.enabled = true;
    renderer.domElement.style.cursor = "";
  }
  endGrabHook = endGrab;
  // Capture phase on window: runs before OrbitControls' pointerdown on the
  // canvas, so the orbit can be disabled for the whole drag. The canvas's
  // own chase-detach listener still fires afterward (grabbing detaches the
  // chase cam exactly like an orbit drag does).
  window.addEventListener("pointerdown", (e) => {
    if (e.target !== renderer.domElement || e.pointerType !== "mouse" || e.button !== 0) return;
    if (grab || inputLocked) return;
    grabRayFrom(e);
    const pick = grabPick();
    if (!pick) return;
    // Any duck mesh grabs the trunk: the freejoint root carries the whole
    // body, and pulling the CoM is what the viewer perturbation feels like.
    const g = pick.kind === "duck"
      ? { bodyId: trunkId, qAdr: 0, dofAdr: 0, mass: model.body(trunkId).subtreemass }
      : {
          bodyId: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "ball"),
          qAdr: ballQposAdr, dofAdr: ballDofAdr,
          mass: model.body(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, "ball")).mass,
        };
    camera.getWorldDirection(_grabCamDir);
    _grabPlane.setFromNormalAndCoplanarPoint(_grabCamDir.negate(), pick.point);
    grab = { ...g, target: [0, 0, 0] };
    updateGrabTarget();
    controls.enabled = false;
    renderer.domElement.style.cursor = "grabbing";
    try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* capture unsupported */ }
  }, true);
  let grabHoverAt = 0;
  window.addEventListener("pointermove", (e) => {
    if (grab) {
      grabRayFrom(e);
      updateGrabTarget();
      return;
    }
    // Hover affordance: grab cursor over anything grabbable. Throttled -
    // a full-rig raycast per mousemove event would be wasteful - and
    // skipped mid-orbit (buttons held) so the cursor doesn't flicker.
    if (e.target !== renderer.domElement || e.pointerType !== "mouse" || e.buttons || inputLocked) return;
    const now = performance.now();
    if (now - grabHoverAt < 80) return;
    grabHoverAt = now;
    grabRayFrom(e);
    renderer.domElement.style.cursor = grabPick() ? "grab" : "";
  });
  window.addEventListener("pointerup", endGrab);
  window.addEventListener("pointercancel", endGrab);

  // Pause: while the menu is up over a live game, keys belong to the menu.
  const setInputLock = (v) => { inputLocked = v; controller.setLocked(v); };
  useGame.subscribe(
    (s) => s.menuOpen,
    (open) => {
      if (!ceremony.entranceDone) return;
      if (open) setInputLock(true);
      else if (!ceremony.respawnActive) setInputLock(false);
    },
  );

  // The rig's root already applies the MJCF Z-up -> three Y-up fix, so the
  // trunk group can take the freejoint pose in raw MJCF coordinates.
  const _target = new THREE.Vector3();
  const _follow = new THREE.Vector3();
  function syncRig() {
    const qpos = data.qpos;
    trunkGroup.position.set(qpos[0], qpos[1], qpos[2]);
    trunkGroup.quaternion.set(qpos[4], qpos[5], qpos[6], qpos[3]);
    for (let j = 0; j < NUM_JOINTS; j++) setJoint(rig, JOINT_NAMES[j], qpos[qposAdr[j]]);
    // Ball: live follows qpos; ghost freeze is owned by the ball actor.
    if (ball) ball.sync(qpos, ballQposAdr, ballActive);
    if (payloadPose) {
      payloadPose.visible = payloadActive;
      if (payloadActive) {
        payloadPose.position.set(
          qpos[payloadQposAdr], qpos[payloadQposAdr + 1], qpos[payloadQposAdr + 2],
        );
        payloadPose.quaternion.set(
          qpos[payloadQposAdr + 4], qpos[payloadQposAdr + 5],
          qpos[payloadQposAdr + 6], qpos[payloadQposAdr + 3],
        );
      }
    }
    // Follow cam: ease the orbit target toward the trunk and translate the
    // camera by the same delta, so the camera-to-duck distance and viewing
    // angle stay constant while the duck walks. Paused while the reset
    // glide owns the camera.
    if (camResetT0 === null) {
      _target.set(qpos[0], qpos[2], -qpos[1]);
      _follow.copy(_target).sub(controls.target);
      // Horizontal follow at the usual rate; vertical much slower so the
      // per-step gait bob doesn't nod the frame.
      _follow.x *= 0.06;
      _follow.z *= 0.06;
      _follow.y *= 0.015;
      controls.target.add(_follow);
      camera.position.add(_follow);
    }
    // Keep the grid plane (and its fade center) under the action; the wall
    // grids share the same radial fade focus.
    grid.position.set(controls.target.x, 0, controls.target.z);
    grid.material.uniforms.uFocus.value.copy(controls.target);
    for (const m of wallMats) m.uniforms.uFocus.value.copy(controls.target);
  }

  // ── Quack: jaw + chirp ────────────────────────────────────────────────
  // The jaw isn't a MuJoCo joint (duck.js re-creates the hinge in JS), so
  // this is purely cosmetic and can't upset the policy. Voice banks from
  // the robot runtime: each colourway gets its own bank and every quack
  // draws a random chirp take from it.
  const QUACK_MS = 480;
  let quackAt = -Infinity;
  let quackTimer = null;
  let quacking = false;
  let wheeeActive = false;
  let padJaw = 0;
  const CHIRP_TAKES = "abcdefghijkl";
  const VOICE_BANK = { classic: "duck1", charcoal: "duck2", purple: "duck3", blue: "duck4" };
  const chirpCache = new Map();
  function playChirp() {
    const bank = VOICE_BANK[currentVariant] ?? "duck1";
    const take = CHIRP_TAKES[(Math.random() * CHIRP_TAKES.length) | 0];
    const url = signed(`./assets/voices/${bank}/chirp_${take}.wav`);
    let a = chirpCache.get(url);
    if (!a) {
      a = new Audio(url);
      a.volume = 0.7;
      chirpCache.set(url, a);
    }
    a.currentTime = 0;
    a.play().catch(() => {});
  }
  function syncVoiceLabel() {
    const voiceLabel = quacking ? "Quack" : wheeeActive ? "Wheee" : null;
    if (store().voiceLabel !== voiceLabel) setStore({ voiceLabel });
  }
  const quackLoud = () => {
    quackAt = performance.now();
    quacking = true;
    clearTimeout(quackTimer);
    syncVoiceLabel();
    quackTimer = setTimeout(() => {
      quacking = false;
      syncVoiceLabel();
    }, QUACK_MS);
    playChirp();
    stickers?.pop("quack");
  };
  // Ground-pick jaw: on the robot the pick policy drives the mouth itself
  // (mouth is part of its action space); the sim's ONNX exports have no
  // mouth channel (all heads are 14 actions), so the peck is re-created
  // here on the same phase clock. Keyed to the measured cycle: the beak
  // first touches the top of the block around phase 0.20-0.24; open on
  // approach, then close during that contact window before the head rises.
  const PICK_JAW_KEYS = [[0, 1], [0.10, 1], [0.17, 0]];
  function pickJawNow() {
    if (pickPreparing) return 1;
    const phase = mode === "groundpick" ? pickRun?.phase : null;
    if (phase == null) return 0;
    const K = PICK_JAW_KEYS;
    if (phase <= K[0][0]) return K[0][1];
    if (phase >= K[K.length - 1][0]) return K[K.length - 1][1];
    for (let i = 1; i < K.length; i++) {
      if (phase > K[i][0]) continue;
      const [p0, v0] = K[i - 1];
      const [p1, v1] = K[i];
      const t = (phase - p0) / (p1 - p0);
      return v0 + (v1 - v0) * (1 - Math.cos(Math.PI * t)) / 2; // eased
    }
    return 0;
  }
  function manualJawOpenNow() {
    const t = (performance.now() - quackAt) / QUACK_MS;
    const flap = t >= 0 && t < 1 ? Math.sin(Math.PI * t) : 0;
    return Math.max(flap, padJaw);
  }
  function jawOpenNow() {
    // Runtime mouth-mode rule (main.rs: motor_targets[MOUTH] += offset):
    // the policy's jaw is the BASE and the trigger/quack opening is an
    // additive offset on top, clamped - it never fights the pick motion.
    return Math.min(1, pickJawNow() + manualJawOpenNow());
  }
  function syncJaw() {
    const physicalOpen = data.qpos[beakQposAdr] / JAW_MAX_OPEN;
    setJawOpen(rig, Math.min(1, Math.max(0, physicalOpen)));
  }

  // ── Wheee: LT-held playable note (sim behavior) ───────────────────────
  // The ride plays the voice bank's LOOP segment only (crossfade-authored
  // to wrap sample-exactly), faded in over ~20 ms, and the LT analog
  // pressure PICKS ITS NOTE: major-pentatonic steps over one octave via
  // playbackRate, glided with setTargetAtTime so per-frame updates and
  // step changes never zipper or click. The runtime has no pitch feature
  // (raw PCM through aplay) - this is the sim's own instrument.
  //
  // The authored start segment is deliberately NOT played: it is 0.8-0.9 s
  // long and cannot be pitch-modulated without breaking the sample-accurate
  // start→loop handoff, so with it the first second of every squeeze was
  // stuck at base pitch - pressure read as a volume change (the attack's
  // own crescendo), not as notes.
  //
  // Release CUTS the ride and plays nothing else - the runtime kills the
  // streaming aplay on the LT falling edge (its end segment never plays on
  // the gamepad path), and the sim's old end-segment playback re-attacked
  // a note on release, which read as a retriggered sound. A short gain
  // ramp stands in for the process kill so Web Audio doesn't click. The
  // gain is otherwise CONSTANT - pressure must never track loudness.
  const WHEEE_TAKES = "ab";
  // Major pentatonic anchored one octave BELOW the sample's natural pitch:
  // full squeeze reaches the natural note, casual play sits clearly lower
  // (the natural pitch alone read as too shrill). -12 st = playbackRate 0.5.
  const WHEEE_SCALE = [-12, -10, -8, -5, -3, 0]; // semitones vs natural pitch
  const WHEEE_DEADZONE = 0.05; // squeeze below this is stick noise, maps to the root
  const WHEEE_GAIN = 0.7;
  let wheeeCtx = null;
  const wheeeBufCache = new Map();
  let wheeeRide = null; // current ride, null while the trigger is up
  function wheeeBuffer(url) {
    let p = wheeeBufCache.get(url);
    if (!p) {
      p = fetch(url)
        .then((r) => r.arrayBuffer())
        .then((ab) => wheeeCtx.decodeAudioData(ab));
      wheeeBufCache.set(url, p);
    }
    return p;
  }
  async function startWheee() {
    stopWheee({ silent: true }); // a re-press replaces the current ride
    wheeeActive = true;
    syncVoiceLabel();
    wheeeCtx ??= new (window.AudioContext ?? window.webkitAudioContext)();
    if (wheeeCtx.state === "suspended") wheeeCtx.resume().catch(() => {});
    const bank = VOICE_BANK[currentVariant] ?? "duck1";
    const take = WHEEE_TAKES[(Math.random() * WHEEE_TAKES.length) | 0];
    const ride = { loopSrc: null, gain: null };
    wheeeRide = ride;
    let loopBuf;
    try {
      loopBuf = await wheeeBuffer(signed(`./assets/voices/${bank}/wheee_loop_${take}.wav`));
    } catch {
      if (wheeeRide === ride) {
        wheeeRide = null;
        wheeeActive = false;
        syncVoiceLabel();
      }
      return; // asset missing / fetch failed: ride silently never starts
    }
    if (wheeeRide !== ride) return; // released (or replaced) during decode
    const gain = wheeeCtx.createGain();
    gain.connect(wheeeCtx.destination);
    const t0 = wheeeCtx.currentTime + 0.02;
    // The loop is steady-state audio (no authored attack): a ~20 ms fade-in
    // makes a clean note onset instead of a click. Constant gain after that.
    gain.gain.setValueAtTime(0, t0);
    gain.gain.linearRampToValueAtTime(WHEEE_GAIN, t0 + 0.02);
    const loopSrc = wheeeCtx.createBufferSource();
    loopSrc.buffer = loopBuf;
    loopSrc.loop = true;
    loopSrc.connect(gain);
    loopSrc.start(t0);
    Object.assign(ride, { loopSrc, gain });
  }
  function stopWheee({ silent = false } = {}) {
    wheeeActive = false;
    syncVoiceLabel();
    const ride = wheeeRide;
    if (!ride) return;
    wheeeRide = null;
    // Nothing audible yet (released mid-decode) or replaced by a re-press:
    // hard stop is inaudible and frees the nodes immediately.
    if (silent || !ride.gain) {
      try { ride.loopSrc?.stop(); } catch { /* already ended */ }
      ride.gain?.disconnect();
      return;
    }
    // Release: cut the ride, retrigger nothing (runtime kills its player
    // here). ~50 ms fade instead of a hard stop so the cut doesn't click.
    const t = wheeeCtx.currentTime;
    ride.gain.gain.setTargetAtTime(0, t, 0.05);
    const stopAt = t + 0.3; // > 5 time constants: fully silent by then
    try { ride.loopSrc?.stop(stopAt); } catch { /* already ended */ }
    const gain = ride.gain;
    setTimeout(() => gain.disconnect(), 400);
  }

  // Per-frame note picking: the full LT travel (above a small deadzone)
  // spans the pentatonic scale, one octave below natural pitch at rest up
  // to the natural pitch at full squeeze. Quantized to scale steps so
  // squeezing plays NOTES, not a siren; the ~40 ms setTargetAtTime glide
  // smooths both the per-frame updates and the step jumps (portamento
  // instead of clicks). Digital 0/1 triggers simply play the top note
  // (the natural pitch). Gain never tracks pressure.
  function driveWheeePitch(pressure) {
    const ride = wheeeRide;
    if (!ride?.loopSrc) return;
    const u = Math.min(1, Math.max(0, (pressure - WHEEE_DEADZONE) / (1 - WHEEE_DEADZONE)));
    const semis = WHEEE_SCALE[Math.round(u * (WHEEE_SCALE.length - 1))];
    ride.loopSrc.playbackRate.setTargetAtTime(2 ** (semis / 12), wheeeCtx.currentTime, 0.04);
  }

  // ── Telemetry (throttled into the store) ─────────────────────────────
  // FPS EMA is per-frame; the store write is 4 Hz so React re-renders
  // stay far away from frame rate. Odometer integrates horizontal trunk
  // travel; teleport-sized jumps (resets) don't count.
  let fpsEma = 60;
  let fpsLastT = performance.now();
  let odoM = 0;
  let odoX = null, odoY = null;
  let telemetryLastPush = 0;
  function renderTelemetry() {
    const now = performance.now();
    const dtF = (now - fpsLastT) / 1000;
    fpsLastT = now;
    if (dtF > 0 && dtF < 0.5) fpsEma += (1 / dtF - fpsEma) * 0.05;
    const stepD = (odoX === null) ? 0 : Math.hypot(data.qpos[0] - odoX, data.qpos[1] - odoY);
    if (stepD < 0.05) odoM += stepD; // plausible per-frame travel only
    odoX = data.qpos[0];
    odoY = data.qpos[1];
    if (now - telemetryLastPush < 250) return;
    telemetryLastPush = now;
    setStore({
      telemetry: {
        fps: Math.round(fpsEma),
        ctrlHz: Math.round(ctrlHz),
        speed: Math.hypot(data.qvel[0], data.qvel[1]),
        odo: odoM,
        peers: ghosts?.peerCount() ?? 0,
      },
    });
  }

  // ── Right-stick camera orbit (inertia downstream of the controller) ──
  // The stick steers an angular VELOCITY that eases toward the stick's
  // target rate, so pushing ramps up gently and releasing coasts to a stop
  // over ~0.3 s. Vertical is flight-style inverted.
  const PAD_ORBIT_SPEED = 2.4; // rad/s at full deflection
  const PAD_ORBIT_SMOOTH = 8; // 1/s response rate (~95% in 0.37 s)
  const padOrbitVel = { az: 0, el: 0 };
  const _padSph = new THREE.Spherical();
  const _padOff = new THREE.Vector3();
  function padOrbitStep(rx, ry, dt) {
    padOrbitLive = rx !== 0 || ry !== 0;
    if (padOrbitLive) chaseCam = false; // detach, same as a mouse grab
    const k = 1 - Math.exp(-PAD_ORBIT_SMOOTH * dt);
    padOrbitVel.az += (rx * PAD_ORBIT_SPEED - padOrbitVel.az) * k;
    padOrbitVel.el += (-ry * PAD_ORBIT_SPEED * 0.75 - padOrbitVel.el) * k;
    if (chaseCam) { padOrbitVel.az = 0; padOrbitVel.el = 0; return; }
    if (Math.abs(padOrbitVel.az) < 1e-3 && Math.abs(padOrbitVel.el) < 1e-3) return;
    _padOff.copy(camera.position).sub(controls.target);
    _padSph.setFromVector3(_padOff);
    _padSph.theta -= padOrbitVel.az * dt;
    _padSph.phi += padOrbitVel.el * dt;
    _padSph.phi = Math.min(controls.maxPolarAngle, Math.max(0.08, _padSph.phi));
    _padSph.makeSafe();
    camera.position.setFromSpherical(_padSph).add(controls.target);
    camera.lookAt(controls.target);
  }

  // Multiplayer ghosts, initialised asynchronously at the end of the boot.
  let ghosts = null;

  // ── Per-frame drive, called by R3F's useFrame ────────────────────────
  let padWasConnected = null;
  let touchWasConnected = null;
  function frame(dt) {
    // Rendering can resume before background timers in some browser
    // lifecycle transitions. If the async 50 Hz loop exited, revive it from
    // the live render surface instead of leaving inputs stuck at phase zero.
    ensureControlLoop();
    controller.update(dt);
    padJaw = controller.getAxes().jaw;
    jawTargetOpen = jawOpenNow();
    const mouthOpen = padJaw > 0.05;
    if (store().mouthOpen !== mouthOpen) setStore({ mouthOpen });
    driveWheeePitch(controller.getAxes().ride); // no-op while no ride is open
    if (padSource.connected !== padWasConnected) {
      padWasConnected = padSource.connected;
      setStore({ padConnected: padSource.connected });
    }
    if (touchSource.connected !== touchWasConnected) {
      touchWasConnected = touchSource.connected;
      setStore({ touchMode: touchSource.connected });
    }
    // Head mode: sticks steer the head targets (stick * HEAD_MAX, signed
    // per joint); the EMA toward them runs in buildObs at 50 Hz. Without
    // a pad the targets stay put (and are debug-writable via window.rl).
    if (headMode && padSource.connected) {
      const h = padSource.head;
      headTarget[0] = HEAD_SIGNS[0] * h.neckPitch * HEAD_MAX;
      headTarget[1] = HEAD_SIGNS[1] * h.pitch * HEAD_MAX;
      headTarget[2] = HEAD_SIGNS[2] * h.yaw * HEAD_MAX;
      headTarget[3] = HEAD_SIGNS[3] * h.roll * HEAD_MAX;
    }
    // Camera orbit runs every frame while a pad is present (the coasting
    // needs the zero-deflection frames too); without a pad, park the
    // state. Head mode parks it too: the right stick belongs to the head
    // and the camera must freeze in place (no leftover coasting).
    if (padSource.connected && !headMode) {
      padOrbitStep(controller.getAxes().orbitX, controller.getAxes().orbitY, dt);
    } else {
      padOrbitLive = false;
      padOrbitVel.az = 0;
      padOrbitVel.el = 0;
    }
    // User-controlled mouth opening is the drop command. Ignore the pick
    // policy's own jaw motion here or the target would release while the same
    // gesture that grabbed it is still finishing.
    if ((payloadHeld || gripTargetAngle !== null) && manualJawOpenNow() > 0.55) {
      releasePayload();
    }
    syncRig();
    syncJaw();
    ghosts?.update();
    controls.update();
    updateChaseCam();
    ceremony.drive();
    ball.drive(() => spawnBall({ fromQueue: true }));
    renderTelemetry();
  }

  // ── Input wiring: arm the controller sources, bind actions ───────────
  controller.init();

  // Keyboard F alternates kicking feet; only advance the alternation on
  // kicks that actually launched (triggerKick reports that).
  let kbKickFoot = "left";
  const srcTag = (source) => (source === "gamepad" ? "pad" : "kb");

  controller.on("reset", () => resetSim());
  controller.on("spawnBall", () => spawnBall());
  controller.on("headToggle", () => toggleHeadMode());
  controller.on("chaseToggle", () => { chaseCam = !chaseCam; });
  controller.on("roll", ({ source }) => triggerRoll(srcTag(source)));
  controller.on("groundPick", ({ source }) => triggerGroundPick(srcTag(source)));
  controller.on("kickL", ({ source }) => triggerKick("left", srcTag(source)));
  controller.on("kickR", ({ source }) => triggerKick("right", srcTag(source)));
  controller.on("alternateKick", ({ source }) => {
    if (triggerKick(kbKickFoot, srcTag(source))) {
      kbKickFoot = kbKickFoot === "left" ? "right" : "left";
    }
  });
  controller.on("sitToggle", () => {
    const sitting = mode === "sitstand" && sitFlag === 1;
    setMode(sitting ? "walk" : "sit");
  });
  // Pad DpadUp: straight back to running (ignored mid-roll: the roll hands
  // back to walk on its own).
  controller.on("walk", () => {
    if (mode !== "walk" && mode !== "roll") setMode("walk");
  });
  controller.on("quack", () => quackLoud());
  controller.on("wheeeStart", () => startWheee());
  controller.on("wheeeStop", () => stopWheee());

  // Leaving head mode keeps the head offsets (runtime behavior): only
  // resetSim zeroes headTarget/headSmooth.
  function exitHeadMode() {
    if (!headMode) return;
    headMode = false;
    padSource.headMode = false;
    syncButtons();
  }

  function toggleHeadMode() {
    if (headMode) return exitHeadMode();
    // Enterable from walk or sit only - never during one-shots (roll /
    // kick / ground pick), the post-kick grace, a stand-up hand-back, a fall
    // recovery, or while the entrance/respawn lock holds the inputs.
    if (inputLocked || (mode !== "walk" && mode !== "sitstand") || postKickLock > 0 ||
        standTimer || recovery)
      return;
    headMode = true;
    padSource.headMode = true;
    syncButtons();
  }

  function setMode(next, { force = false } = {}) {
    if (!force && inputLocked) return;
    // No policy switching mid-roll or mid-kick: both end on their own and
    // return to walk - switching now would floor the duck. Same while the
    // fall-recovery state machine owns the duck.
    if (recovery) return;
    if ((mode === "roll" && rollRun) || (isKick() && kickRun) ||
        (mode === "groundpick" && pickRun)) return;
    exitHeadMode(); // posture changes exit head mode (offsets kept)
    clearModeTimers();
    rollRun = null;
    pickRun = null;
    if (next !== "sit") {
      // Leaving a sit: let the sitstand policy stand the duck back up first.
      if (mode === "sitstand" && sitFlag === 1) {
        sitFlag = 0;
        standTimer = setTimeout(() => {
          standTimer = null;
          mode = next;
          lastAction.fill(0);
          syncButtons();
        }, 2000);
        syncButtons();
        return;
      }
      mode = next;
      lastAction.fill(0);
    } else {
      // Hand over gently: hold the stand under the sitstand policy for a
      // moment before commanding the sit, or the abrupt session switch
      // knocks the duck over.
      mode = "sitstand";
      sitFlag = 0;
      lastAction.fill(0);
      sitTimer = setTimeout(() => {
        sitTimer = null;
        if (mode === "sitstand") { sitFlag = 1; syncButtons(); }
      }, 800);
    }
    syncButtons();
  }

  // One roll, then straight back to running. lastAction is deliberately
  // NOT zeroed: the runtime keeps one continuous action history across
  // policy switches, and the roll initiates more reliably mid-gait.
  function triggerRoll(source = "kb") {
    if (inputLocked || mode !== "walk" || standTimer || recovery) return;
    exitHeadMode();
    clearModeTimers();
    mode = "roll";
    sitFlag = 0;
    rollRun = { steps: 0, tipped: false };
    syncButtons();
    stickers?.pop("roll");
  }

  // One-shot ground pick (runtime A button): peck the ground and stand
  // back up, phase-driven (cos/sin phase encoding in the command vel
  // slots). From walk only, and never during another one-shot / a stand-up
  // hand-back / the entrance lock.
  function triggerGroundPick(source = "kb") {
    if (inputLocked || mode !== "walk" || standTimer || recovery ||
        gripTargetAngle !== null) return;
    exitHeadMode();
    clearModeTimers();
    pickPreparing = true;
    // Stabilize under the stand policy first. Entering the one-shot from an
    // arbitrary left/right gait phase moves the blind mouth path by several
    // centimetres and makes a physical contact task irreproducible.
    mode = "stand";
    sitFlag = 0;
    lastAction.fill(0);
    standTimer = setTimeout(() => {
      standTimer = null;
      if (mode !== "stand") {
        pickPreparing = false;
        return;
      }
      mode = "groundpick";
      pickPreparing = false;
      pickRun = { phase: 0, contactHold: 0, contactHoldUsed: false };
      lastAction.fill(0);
      syncButtons();
    }, 800);
    syncButtons();
  }

  // One blind kick (the duck can't see any ball - it's a scripted boot).
  // Returns whether the kick actually launched so the keyboard's foot
  // alternation only advances on real kicks.
  function triggerKick(foot, source = "kb") {
    if (inputLocked || mode !== "walk" || standTimer || recovery) return false;
    exitHeadMode();
    clearModeTimers();
    mode = foot === "left" ? "kickL" : "kickR";
    sitFlag = 0;
    kickRun = { steps: 0 };
    syncButtons();
    stickers?.pop("kick");
    return true;
  }

  function syncButtons() {
    const sitting = mode === "sitstand" && sitFlag === 1;
    const label =
      recovery ? "Recovery"
      : standTimer ? "Stand up"
      : sitTimer ? "Sit prep"
      : mode === "roll" ? "Roll"
      : mode === "groundpick" ? "Pick"
      : isKick() ? "Kick"
      : headMode ? "Head"
      : sitting ? "Sit"
      : payloadHeld ? "Holding"
      : "Run";
    if (store().modeLabel !== label) setStore({ modeLabel: label });
    if (store().ballActive !== ballActive) setStore({ ballActive });
    if (store().payloadActive !== payloadActive || store().payloadHeld !== payloadHeld) {
      setStore({ payloadActive, payloadHeld });
    }
  }

  // ── Public surface for the React UI ──────────────────────────────────
  Object.assign(gameApi, {
    frame,
    setVariant: (name) => {
      if (!VARIANTS[name] || name === currentVariant) return;
      currentVariant = name;
      applyVariant(rig, name);
      setStore({ variant: name });
    },
    resetSim,
    spawnBall: () => spawnBall(),
    spawnPayload: () => spawnPayload(),
    releasePayload: () => releasePayload(),
    startEntrance: () => ceremony.startEntrance(),
  });

  // Deterministic hooks for automated verification (rAF pauses in
  // background tabs, and the control loop is async).
  window.rl = {
    get model() { return model; },
    get data() { return data; },
    mujoco, camera, controls,
    get mode() { return mode; },
    get sitFlag() { return sitFlag; },
    buildObs, cmd,
    velCmd: kbSource.command, lastAction, resetSim,
    controller, kbSource, padSource,
    spawnBall, spawnPayload, releasePayload, triggerKick, triggerRoll, sessions, ort,
    triggerGroundPick,
    get groundPickPhase() { return pickRun?.phase ?? null; },
    get kickSteps() { return KICK_STEPS; },
    set kickSteps(v) { KICK_STEPS = v; },
    get recovery() { return recovery?.state ?? null; },
    // Debug shove for fall-recovery testing: an instantaneous trunk
    // velocity kick (free-joint dofs are qvel[0..5]).
    debugPush: (vx = 0, vy = 0, vz = 0, wx = 0, wy = 0, wz = 0) => {
      const qvel = data.qvel;
      qvel[0] += vx; qvel[1] += vy; qvel[2] += vz;
      qvel[3] += wx; qvel[4] += wy; qvel[5] += wz;
    },
    get headMode() { return headMode; },
    toggleHeadMode, headTarget, headSmooth,
    get ballActive() { return ballActive; },
    get ballQposAdr() { return ballQposAdr; },
    get payloadActive() { return payloadActive; },
    get payloadHeld() { return payloadHeld; },
    get physicsDiagnostics() { return { ...contactAudit, contact: { ...payloadContact }, held: payloadHeld }; },
    resetPhysicsDiagnostics: () => { contactAudit = createContactAudit(); },
    get payloadQposAdr() { return payloadQposAdr; },
    get payloadGripGeomIds() { return payloadGripGeomIds; },
    get payloadGripActive() { return gripTargetAngle !== null; },
    get mouthSiteId() { return mouthSiteId; },
    get beakTipSiteId() { return beakTipSiteId; },
    get beakUpperGripSiteId() { return beakUpperGripSiteId; },
    get beakLowerGripSiteId() { return beakLowerGripSiteId; },
    get beakQposAdr() { return beakQposAdr; },
    get gripTargetAngle() { return gripTargetAngle; },
    get chaseCam() { return chaseCam; },
    set chaseCam(v) { chaseCam = !!v; },
    get props() { return propGroups; },
    get relief() { return reliefOn; },
    setRelief: (v) => { reliefOn = !!v; },
    get camResetActive() { return camResetT0 !== null; },
    get respawnActive() { return ceremony?.respawnActive ?? false; },
    get camPose() {
      return {
        pos: camera.position.toArray(),
        target: controls.target.toArray(),
      };
    },
    get chaseYaw() { return { follow: chaseYawFollow, smooth: chaseYawSmooth, held: chaseHeldYaw, tracking: chaseYawTracking }; },
    padOrbitStep,
    jawOpenNow,
    step: async (n = 1) => { for (let i = 0; i < n; i++) await controlStep(); },
    render: () => { syncRig(); renderer.render(scene, camera); },
    frame: (dt = 1 / 60) => frame(dt),
    get ghosts() { return ghosts; },
    get inputLocked() { return inputLocked; },
    get controlLoop() {
      return {
        running: controlLoopPromise !== null,
        errors: controlLoopErrors,
        lastStepAge: performance.now() - controlLoopLastStep,
      };
    },
    entrance: {
      start: () => ceremony.startEntrance(),
      setReveal: (floor, wall) => ceremony.setReveal(floor, wall),
      setFx: (p) => ceremony.setFx(p),
    },
  };

  if (new URLSearchParams(location.search).get("telemetry") === "1") {
    const { installPhysicsOverlay } = await import("./physics-overlay.js");
    window.physicsOverlay?.destroy();
    window.physicsOverlay = installPhysicsOverlay(() => window.rl);
  }

  // Boot complete: the sim/HUD go live immediately. The BIOS readout (if
  // the user already waddled in, or when they do) sees bootDone and closes
  // with READY. + fade on its own.
  setStore({ bootDone: true });

  // ── Multiplayer ghosts (WebRTC, serverless signaling) ────────────────
  // Broadcast this duck's pose and render up to 3 other visitors live as
  // translucent ducks. Fire-and-forget: any failure just means no ghosts.
  const r3 = (x) => Math.round(x * 1000) / 1000;
  try {
    // Ghosts only join once the entrance has fully played: the world (and
    // this duck) must stay hidden until then, translucent peers included.
    await ceremony.entranceFinished;
    ghosts = await initGhosts({
      scene, rig, cloneRig, setJoint, setJawOpen, applyVariant,
      jointNames: JOINT_NAMES,
      // Payload sanitizing: ghosts.js coerces unknown peer variants to the
      // default instead of letting applyVariant throw on a bad key.
      variantNames: Object.keys(VARIANTS),
      defaultVariant: DEFAULT_VARIANT,
      getLocalState: () => {
        const qpos = data.qpos;
        const j = new Array(NUM_JOINTS);
        for (let i = 0; i < NUM_JOINTS; i++) j[i] = r3(qpos[qposAdr[i]]);
        return {
          p: [r3(qpos[0]), r3(qpos[1]), r3(qpos[2]), r3(qpos[3]), r3(qpos[4]), r3(qpos[5]), r3(qpos[6])],
          j,
          w: r3(jawOpenNow()),
          v: currentVariant,
        };
      },
    });
    liveGhostSessions.add(ghosts);
    if (ghosts.room) ghosts.room.onPeerJoin = () => stickers?.pop("hi");
  } catch (e) {
    window.__ghostErr = String((e && e.stack) || e);
    console.warn("ghosts disabled:", e);
  }
}
