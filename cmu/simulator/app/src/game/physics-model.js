import { JOINT_NAMES, DEFAULT_POSE, TIMESTEP, BALL_RADIUS, BALL_PARK_POS, PAYLOAD_CAP_HALF_X, PAYLOAD_CAP_HALF_Y, PAYLOAD_CAP_HALF_Z, PAYLOAD_MASS, PAYLOAD_PEDESTAL_HALF_XY, PAYLOAD_PEDESTAL_HALF_Z, PAYLOAD_SPAWN_DIST, PAYLOAD_LATERAL, PAYLOAD_PARK_POS, ARENA_HALF, SPAWN_X, SPAWN_Y, RELIEF_GRID, RELIEF_HMAX, RELIEF_SINK } from "./constants.js";

import beakColliders from "./beak-colliders.js";

// Shared by the browser and deterministic CPU/WASM regression tests.
export function preparePhysicsXml(src, { DOMParser, XMLSerializer, colliders = [], JAW_MAX_OPEN = 0.32 }) {
  const doc = new DOMParser().parseFromString(src, "text/xml");

  // Promote the render-only beak parts into collision geoms before the
  // visual strip below. The lower rigid jaw + soft pad move together on a
  // real MuJoCo hinge; the upper soft pad remains fixed to the head.
  const headBody = doc.querySelector('body[name="jaw_soft"]');
  const directGeoms = [...headBody.children].filter((n) => n.tagName === "geom");
  const lowerRigid = directGeoms.find(
    (g) => g.getAttribute("mesh") === "jaw" && g.getAttribute("class") === "collision",
  );
  const lowerSoft = directGeoms.find((g) => g.getAttribute("mesh") === "jaw_soft");
  const upperSoft = directGeoms.find((g) => g.getAttribute("mesh") === "soft_mouth_top");
  const shells = ["bottom_head_shell", "top_head_shell"].map(source =>
    [source, directGeoms.find(g => g.getAttribute("mesh") === source && g.getAttribute("class") === "collision")]);
  lowerSoft.setAttribute("class", "collision");
  upperSoft.setAttribute("class", "collision");
  for (const g of [...doc.querySelectorAll('geom[class="visual"]')]) g.remove();
  const usedMeshes = new Set(
    [...doc.querySelectorAll("geom[mesh]")].map((g) => g.getAttribute("mesh")),
  );
  for (const m of [...doc.querySelectorAll("asset > mesh")]) {
    const name = m.getAttribute("name") ?? m.getAttribute("file").replace(/\.stl$/i, "");
    if (!usedMeshes.has(name)) m.remove();
  }
  const root = doc.documentElement;
  const el = (tag, attrs) => {
    const e = doc.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  };

  // Hinge location comes from the circular axle in jaw.stl (see duck.js).
  // In head-body coordinates the axle is (0.003, 0, -0.01796), axis +Y.
  // Re-express the two lower geoms relative to that new child-body origin,
  // preserving their closed-pose transform exactly.
  const beakBody = el("body", { name: "beak_lower", pos: "0.003 0 -0.01796" });
  beakBody.appendChild(el("joint", {
    name: "beak_joint", type: "hinge", axis: "0 1 0", range: `0 ${JAW_MAX_OPEN}`,
    limited: "true", damping: "0.01", frictionloss: "0.001", armature: "0.00005",
  }));
  lowerRigid.setAttribute("pos", "-0.0075 0 -0.00004");
  lowerRigid.setAttribute("name", "beak_lower_rigid_geom");
  lowerRigid.setAttribute("mass", "0.0008");
  lowerRigid.setAttribute("friction", "1.8 0.05 0.01");
  lowerRigid.setAttribute("condim", "6");
  // Keep the rigid jaw mesh for the moving body's mass, but not for
  // payload contact: MuJoCo convexifies it across the open mouth and that
  // hull would falsely block the target several millimetres from the
  // rendered surface.
  lowerRigid.setAttribute("contype", "0");
  lowerRigid.setAttribute("conaffinity", "0");
  lowerSoft.setAttribute("pos", "-0.0076 0 -0.0000396");
  lowerSoft.setAttribute("name", "beak_lower_soft_geom");
  lowerSoft.setAttribute("mass", "0.0002");
  lowerSoft.setAttribute("friction", "1.8 0.05 0.01");
  lowerSoft.setAttribute("condim", "6");
  lowerSoft.setAttribute("contype", "0");
  lowerSoft.setAttribute("conaffinity", "0");
  upperSoft.setAttribute("name", "beak_upper_soft_geom");
  upperSoft.setAttribute("friction", "1.8 0.05 0.01");
  upperSoft.setAttribute("condim", "6");
  upperSoft.setAttribute("contype", "0");
  upperSoft.setAttribute("conaffinity", "0");
  // Use convex pieces of the actual surface, keeping the original MJCF
  // transforms. A single convex hull bridges the concave open mouth.
  beakBody.appendChild(lowerRigid);
  beakBody.appendChild(lowerSoft);
  for (const [source, parent, original] of [
    ["jaw", beakBody, lowerRigid],
    ["jaw_soft", beakBody, lowerSoft],
    ["soft_mouth_top", headBody, upperSoft],
    ...shells.map(([source, original]) => [source, headBody, original]),
  ]) {
    original.setAttribute("contype", "0");
    original.setAttribute("conaffinity", "0");
    for (const [i, file] of beakColliders[source].parts.entries()) {
      const meshName = file.replace(/\.stl$/, "");
      doc.querySelector("asset").appendChild(el("mesh", {name: meshName, file}));
      parent.appendChild(el("geom", {
        name: `beak_${source}_contact_${i}`, type: "mesh", mesh: meshName,
        pos: original.getAttribute("pos"), quat: original.getAttribute("quat"),
        contype: "16", conaffinity: "33", group: "3", mass: "0",
        friction: "0.8 0.0002 0.00001", condim: "4",
        solref: "0.00025 1", solimp: "0.99 0.999 0.001", margin: "0",
      }));
    }
  }
  // At q=0 this coincides with the original fixed mouth_tip; at q>0 it
  // follows the moving lower bill and gives us an honest jaw-gap reading.
  beakBody.appendChild(el("site", {
    name: "beak_tip", pos: "-0.01109334 0 -0.0597783", size: "0.001",
  }));
  // Sites only measure the center corridor; the matching geoms above do
  // all contact and force generation.
  headBody.appendChild(el("site", {
    name: "beak_upper_grip", pos: "-0.009 0 -0.064", size: "0.001",
  }));
  beakBody.appendChild(el("site", {
    name: "beak_lower_grip", pos: "-0.011 0 -0.04604", size: "0.001",
  }));
  headBody.appendChild(beakBody);

  doc.querySelector("actuator").appendChild(el("position", {
    name: "beak_actuator", joint: "beak_joint", kp: "0.4", dampratio: "1",
    ctrlrange: `0 ${JAW_MAX_OPEN}`, forcerange: "-0.006 0.006",
  }));
  let contact = doc.querySelector("contact");
  if (!contact) {
    contact = el("contact", {});
    root.insertBefore(contact, doc.querySelector("worldbody"));
  }
  contact.appendChild(el("exclude", { body1: "jaw_soft", body2: "beak_lower" }));
  root.appendChild(el("option", { timestep: String(TIMESTEP), integrator: "implicitfast",
    solver: "Newton", cone: "elliptic", impratio: "10", iterations: "100", tolerance: "1e-10" }));
  doc.querySelector("worldbody").appendChild(
    el("geom", {
    name: "floor", type: "plane", size: "0 0 0.05", pos: "0 0 0",
    conaffinity: "3",
    }),
  );
  // Arena walls: four static boxes (no joints, so no qpos/keyframe
  // impact); default contype/conaffinity collides with ball and duck.
  const ht = 0.05 / 2, hh = 0.25 / 2;
  const off = ARENA_HALF + ht, span = ARENA_HALF + 0.05;
  const walls = [
    { name: "wall_px", pos: `${off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_nx", pos: `${-off} 0 ${hh}`, size: `${ht} ${span} ${hh}` },
    { name: "wall_py", pos: `0 ${off} ${hh}`, size: `${span} ${ht} ${hh}` },
    { name: "wall_ny", pos: `0 ${-off} ${hh}`, size: `${span} ${ht} ${hh}` },
  ];
  for (const w of walls) {
    doc.querySelector("worldbody").appendChild(
    el("geom", {
      name: w.name, type: "box", pos: w.pos, size: w.size, conaffinity: "3",
    }),
    );
  }
  // Prop library colliders: one static box per enabled prop
  // (declared in props.js next to the visual placement, optionally
  // yawed via euler to match off-axis staging) so the duck and ball
  // can't clip through the dressing.
  for (const c of colliders) {
    const attrs = { name: c.name, type: "box", pos: c.pos, size: c.size };
    if (c.euler) attrs.euler = c.euler;
    doc.querySelector("worldbody").appendChild(el("geom", attrs));
  }
  // Kickable ball: a light free sphere (beach-ball feel). MuJoCo has no
  // restitution parameter - the bounce comes from solref damping < 1, and
  // the rolling-friction term makes it come to rest. Appended AFTER the
  // robot body so the trunk freejoint stays first in qpos.
  const ballBody = el("body", { name: "ball", pos: BALL_PARK_POS });
  ballBody.appendChild(el("freejoint", { name: "ball_freejoint" }));
  // condim 6 enables the torsional + rolling friction components; with
  // the default condim 3 a rolling ball never decelerates.
  ballBody.appendChild(el("geom", {
    name: "ball_geom", type: "sphere", size: String(BALL_RADIUS),
    mass: "0.03", friction: "0.4 0.01 0.003", solref: "0.03 0.4", condim: "6",
  }));
  doc.querySelector("worldbody").appendChild(ballBody);

  // A beak-sized training tab. It stays a real free body throughout: the
  // articulated lower bill and fixed upper mouth pad must make the contacts
  // that lift it from the separate fixed pedestal.
  const payloadBody = el("body", { name: "payload", pos: PAYLOAD_PARK_POS });
  payloadBody.appendChild(el("freejoint", { name: "payload_freejoint" }));
  payloadBody.appendChild(el("geom", {
    name: "payload_geom", type: "box",
    size: `${PAYLOAD_CAP_HALF_X} ${PAYLOAD_CAP_HALF_Y} ${PAYLOAD_CAP_HALF_Z}`,
    mass: String(PAYLOAD_MASS),
    friction: "0.8 0.0002 0.00001", solref: "0.00025 1", solimp: "0.99 0.999 0.001", condim: "4",
    priority: "1", margin: "0", contype: "32", conaffinity: "19",
  }));
  doc.querySelector("worldbody").appendChild(payloadBody);
  doc.querySelector("worldbody").appendChild(el("geom", {
    name: "payload_pedestal_geom", type: "box",
    pos: `${SPAWN_X + PAYLOAD_SPAWN_DIST} ${SPAWN_Y + PAYLOAD_LATERAL} ${PAYLOAD_PEDESTAL_HALF_Z}`,
    size: `${PAYLOAD_PEDESTAL_HALF_XY} ${PAYLOAD_PEDESTAL_HALF_XY} ${PAYLOAD_PEDESTAL_HALF_Z}`,
    friction: "0.8 0.0002 0.00001", solref: "0.00025 1", condim: "4",
    contype: "1", conaffinity: "49",
  }));
  // Relief terraces: one kinematically driven box per raisable grid
  // cell, on a vertical slide joint (qpos written directly each control
  // step, like the ball). q = 0 parks the box fully below the floor;
  // q = h + RELIEF_EPS puts its top exactly at h. Appended after the
  // ball so the keyframe layout stays robot + ball + relief.
  // Relief heightfield: a static hfield over the whole arena, elevation
  // data filled at runtime from the shared analytic bump function (see
  // driveRelief). It compiles flat (no file/elevation = zeros) and has
  // no joints, so qpos and the keyframe are untouched. The geom sits
  // RELIEF_SINK below the floor so a near-zero z-size is fully buried;
  // raising the terrain = scaling model.hfield_size z at runtime.
  doc.querySelector("asset").appendChild(el("hfield", {
    name: "terrain", nrow: String(RELIEF_GRID), ncol: String(RELIEF_GRID),
    size: `${ARENA_HALF} ${ARENA_HALF} ${RELIEF_HMAX} 0.1`,
  }));
  doc.querySelector("worldbody").appendChild(el("geom", {
    name: "terrain", type: "hfield", hfield: "terrain",
    pos: `0 0 ${-RELIEF_SINK}`, conaffinity: "3",
  }));
  // STAND keyframe from mjlab's scene_walk.xml. qpos must cover every
  // joint in document order: the 14 actuated hinges take DEFAULT_POSE by
  // name, anything else starts at zero. The ball's 7 free-joint values
  // MUST be appended or nq won't match; both free props are parked 50 m
  // away = effectively absent.
  const qposFree = `${SPAWN_X} ${SPAWN_Y} 0.12 1 0 0 0`;
  const poseByName = new Map(JOINT_NAMES.map((n, i) => [n, DEFAULT_POSE[i]]));
  const qposJoints = [...doc.querySelectorAll("body > joint")]
    .map((j) => poseByName.get(j.getAttribute("name")) ?? 0)
    .join(" ");
  const pose14 = Array.from(DEFAULT_POSE).join(" ");
  const kf = doc.createElement("keyframe");
  kf.appendChild(el("key", {
    name: "STAND",
    qpos: `${qposFree} ${qposJoints} ${BALL_PARK_POS} 1 0 0 0 ${PAYLOAD_PARK_POS} 1 0 0 0`,
    ctrl: `${pose14} 0`,
  }));
  root.appendChild(kf);
  const meshFiles = [...doc.querySelectorAll("asset > mesh")].map((m) => m.getAttribute("file"));
  return { xml: new XMLSerializer().serializeToString(doc), meshFiles };
}
