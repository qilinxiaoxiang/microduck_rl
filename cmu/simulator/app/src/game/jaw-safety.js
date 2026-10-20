// Jaw and head-pitch safety limits shared with the real robot.
// jaw-limits.json and jaw-pose-caps.json are byte-identical copies of microduck-sim/guide/firmware/pi/jaw_limits.json
// and jaw_pose_caps.json; edit or rebuild those (jaw-neck-check/scripts/build_pose_caps.py) and copy them here
// (guide/firmware/pi/test_jaw_limits.py and tests/jaw-safety.test.mjs check the copies). The simulator beak hinge maps
// directly to the V1.8 jaw frame (jaw-neck-check README, "Hinge equivalence"), so beak_joint radians are jaw opening
// from closed. The pose cap lookup matches jaw_limits.py pose_contact_deg() node for node.
import LIMITS from "./jaw-limits.json" with {type: "json"};
import POSE_TABLE from "./jaw-pose-caps.json" with {type: "json"};

const DEG = Math.PI / 180, RAD_TO_DEG = 180 / Math.PI, EPS = 1e-9 * DEG;  // as Python math.radians/degrees
export const JAW_LIMITS = Object.freeze({...LIMITS});
export const JAW_MAX_OPEN_RAD = LIMITS.jaw_max_open_deg * DEG;
export const JAW_OPEN_THRESHOLD_RAD = LIMITS.jaw_open_threshold_deg * DEG;
export const HEAD_PITCH_MAX_OPEN_RAD = LIMITS.head_pitch_max_while_jaw_open_deg * DEG;
export const POSE_JOINTS = Object.freeze(["neck_pitch", "head_pitch", "head_yaw", "head_roll"]);
const EPS_DEG = 1e-9;

const finite = (x, name) => { if (!Number.isFinite(x)) throw Error(`Non-finite ${name}`); return x; };

// Jaw opening never exceeds 25 deg. Closing below 0 is the hinge stop, not a limit here.
export function limitJawTarget(target) {
  return Math.min(finite(target, "jaw target"), JAW_MAX_OPEN_RAD);
}

// Larger of commanded and measured opening; with neither known the jaw counts as open.
export function jawCountsAsOpen(commanded, measured) {
  const known = [commanded, measured].filter(x => x !== undefined && x !== null).map(x => finite(x, "jaw opening"));
  return known.length === 0 || Math.max(...known) > JAW_OPEN_THRESHOLD_RAD + EPS;
}

// Desired head_pitch angle (MJCF joint radians, positive tips the head forward).
export function limitHeadPitchTarget(angle, jawOpen) {
  finite(angle, "head_pitch target");
  return jawOpen && angle > HEAD_PITCH_MAX_OPEN_RAD + EPS ? HEAD_PITCH_MAX_OPEN_RAD : angle;
}

// Position-actuator command whose static equilibrium holds head_pitch at the limit. A position actuator settles where
// kp (ctrl - q) equals qfrc_bias, so ctrl <= limit + bias / kp keeps the settled joint angle at or below the limit,
// whether or not the caller already added gravity feed-forward.
export function headPitchCtrlCap(biasForce, kp) {
  if (!(kp > 0)) throw Error("Invalid head_pitch actuator gain");
  return HEAD_PITCH_MAX_OPEN_RAD + finite(biasForce, "head_pitch bias") / kp;
}

// Pose table: first jaw contact (deg) on the neck_pitch x head_pitch x head_yaw x head_roll grid; null = clear to
// clear_to_deg.
if (POSE_TABLE.joints.join() !== POSE_JOINTS.join()) throw Error("jaw-pose-caps.json joint order");
export const POSE_AXES = Object.freeze(Object.fromEntries(POSE_JOINTS.map(j => [j, POSE_TABLE.axes_deg[j].map(Number)])));
const SHAPE = POSE_JOINTS.map(j => POSE_AXES[j][2]);
const FLAT = [];
(function walk(node, depth) {
  if (depth === SHAPE.length) {
    const x = node === null ? POSE_TABLE.clear_to_deg : node;
    if (!Number.isFinite(x) || x < 0) throw Error("jaw-pose-caps.json value");
    FLAT.push(Math.min(x, POSE_TABLE.clear_to_deg));
    return;
  }
  if (!Array.isArray(node) || node.length !== SHAPE[depth]) throw Error(`jaw-pose-caps.json grid shape at ${POSE_JOINTS[depth]}`);
  for (const child of node) walk(child, depth + 1);
})(POSE_TABLE.first_contact_deg, 0);
// Table minimised over each set of unknown axes (bit a of the mask = axis a unknown), built from the next larger one.
const REDUCED = new Map([[0, {flat: FLAT, shape: SHAPE}]]);
for (let mask = 1; mask < 1 << SHAPE.length; mask++) {
  const last = 31 - Math.clz32(mask), parent = REDUCED.get(mask & ~(1 << last));
  let axis = 0;
  for (let a = 0; a < last; a++) if (!(mask >> a & 1)) axis++;
  const n = parent.shape[axis], inner = parent.shape.slice(axis + 1).reduce((x, y) => x * y, 1), out = [];
  for (let base = 0; base < parent.flat.length; base += n * inner)
    for (let i = 0; i < inner; i++) {
      let v = Infinity;
      for (let k = 0; k < n; k++) v = Math.min(v, parent.flat[base + k * inner + i]);
      out.push(v);
    }
  REDUCED.set(mask, {flat: out, shape: parent.shape.filter((_, a) => a !== axis)});
}
export const POSE_TABLE_MIN_CONTACT_DEG = REDUCED.get((1 << SHAPE.length) - 1).flat[0];

// A value at most jaw_pose_edge_tolerance_deg past an edge counts as lying in the last cell at that edge.
function axisNodes(name, lo, hi) {
  const [start, step, count] = POSE_AXES[name], end = start + step * (count - 1);
  const tol = LIMITS.jaw_pose_edge_tolerance_deg + 1e-9;
  if (lo === null || lo < start - tol || hi > end + tol) return null;
  let i0 = Math.floor((lo - start) / step + 1e-9), i1 = Math.ceil((hi - start) / step - 1e-9);
  if (lo < start - 1e-9) { i0 = 0; i1 = Math.max(i1, 1); }
  if (hi > end + 1e-9) { i0 = Math.min(i0, count - 2); i1 = count - 1; }
  return [Math.max(0, i0), Math.min(count - 1, i1)];
}

// Smallest first-contact angle (deg) over the grid nodes surrounding every pose between commanded and measured
// (objects of MJCF joint radians; a joint missing from both, or off the grid, takes its whole range).
export function poseContactDeg(commanded, measured) {
  const ranges = POSE_JOINTS.map(name => {
    const vals = [commanded, measured].filter(o => o && o[name] !== undefined && o[name] !== null).map(o => finite(o[name], name) * RAD_TO_DEG);
    return vals.length ? axisNodes(name, Math.min(...vals), Math.max(...vals)) : null;
  });
  const {flat, shape} = REDUCED.get(ranges.reduce((m, r, a) => m | (r === null ? 1 << a : 0), 0));
  const known = ranges.filter(r => r !== null), strides = [];
  for (let a = shape.length - 1, acc = 1; a >= 0; acc *= shape[a], a--) strides[a] = acc;
  let offsets = [0];
  known.forEach(([lo, hi], a) => {
    const next = [];
    for (const o of offsets) for (let k = lo; k <= hi; k++) next.push(o + k * strides[a]);
    offsets = next;
  });
  return Math.min(...offsets.map(o => flat[o]));
}

// min(25, contact - 2) deg, or 0 (held closed) when that leaves less than the usable opening.
export function capForContact(contactDeg) {
  const cap = Math.min(LIMITS.jaw_max_open_deg, contactDeg - LIMITS.jaw_pose_cap_margin_deg);
  return cap < LIMITS.jaw_min_usable_open_deg - EPS_DEG ? {capDeg: 0, closed: true} : {capDeg: cap, closed: false};
}

export function poseJawCapRad(commanded, measured) {
  const contactDeg = poseContactDeg(commanded, measured), {capDeg, closed} = capForContact(contactDeg);
  return {capRad: capDeg * DEG, closed, contactDeg};
}

// Last step before a control tick reaches MuJoCo: cap the jaw command at 25 deg, hold head_pitch at or below the limit
// while the commanded or measured jaw is open, then cap the jaw at the pose cap for the commanded and measured head pose.
// pose = {indices, bias, kp, measured}, each keyed by POSE_JOINTS name: the commanded joint angle is the actuator's
// static equilibrium ctrl - bias / kp (after the head_pitch limit). Without pose the head is unknown and the most
// conservative cap applies. ctrl is modified in place; returns which limits acted.
export function applyJawSafety(ctrl, {jawIndex = 14, headIndex = 6, measuredJaw, headBias, kp, pose}) {
  const result = {jawCapped: false, headCapped: false, jawPoseCapped: false, jawClosedZone: false};
  const jaw = limitJawTarget(ctrl[jawIndex]);
  if (jaw !== ctrl[jawIndex]) { ctrl[jawIndex] = jaw; result.jawCapped = true; }
  if (jawCountsAsOpen(ctrl[jawIndex], measuredJaw)) {
    const cap = headPitchCtrlCap(headBias, kp);
    if (finite(ctrl[headIndex], "head_pitch command") > cap) { ctrl[headIndex] = cap; result.headCapped = true; }
  }
  const commanded = pose ? Object.fromEntries(POSE_JOINTS.map(n => {
    if (!(pose.kp[n] > 0)) throw Error(`Invalid ${n} actuator gain`);
    return [n, finite(ctrl[pose.indices[n]], `${n} command`) - finite(pose.bias[n], `${n} bias`) / pose.kp[n]];
  })) : null;
  const {capRad, closed, contactDeg} = poseJawCapRad(commanded, pose ? pose.measured : null);
  result.jawCapRad = capRad;
  result.contactDeg = contactDeg;
  if (ctrl[jawIndex] > capRad + EPS) {
    ctrl[jawIndex] = capRad;
    if (closed) result.jawClosedZone = true; else result.jawPoseCapped = true;
  }
  return result;
}
