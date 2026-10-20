// P0 experiment: a learned jaw motor policy, with frozen balance/reach skills.
// No action can change a body pose, contact geometry, force limit or task label.
export const JAW_RL_VERSION = 1;
// Execution fidelity: coarse 0.125 ms release/foot impacts failed the
// independent depth limit. 0.0625 and 0.03125 ms were checked on that failure.
// Training can explicitly request the original 0.125 ms model for replay.
export const JAW_RL_DT = .0000625;
export const JAW_RL_REPEAT = 5; // 10 Hz decisions, 50 Hz motor/balance loop
export const JAW_RL_PHASES = ["clamp", "lift", "hold", "release"];
export const JAW_RL_OBS_SIZE = 20;
const clip = (x, lo = -5, hi = 5) => Math.max(lo, Math.min(hi, x));

export class JawActionFilter {
  constructor() { this.target = .32; }
  update(action, dt = .02) {
    if (action !== 0 && action !== 1) throw Error("Jaw action must be 0 (close) or 1 (open)");
    const desired = action * .32;
    this.target += clip(desired - this.target, -.6 * dt, .6 * dt);
    return this.target;
  }
}

export function jawObservation(s, ctl, target) {
  const c = s.contact ?? {}, p = s.payload, g = s.graspPoint;
  const obs = [
    ...p.slice(0, 3).map((v, i) => (v - g[i]) / .05),
    s.jaw / .32, target / .32,
    (c.upperForce ?? 0) / .4, (c.lowerForce ?? 0) / .4,
    Number(!!c.otherSupport), (p[2] - ctl.height) / .05,
    (s.root[2] - .117) / .03, s.root[4], s.root[5],
    s.joint[5], s.joint[6], Math.min(ctl.age / 3, 2), ctl.hold / 1.2,
    ...JAW_RL_PHASES.map(phase => Number(ctl.phase === phase)),
  ];
  if (obs.length !== JAW_RL_OBS_SIZE || obs.some(x => !Number.isFinite(x))) throw Error("Invalid jaw observation");
  return obs.map(x => clip(x));
}

// Physical progress only; no label-matching, demonstrations or target actions.
// A separate unchanged whole-body verifier decides final success.
export function jawReward(s, ctl) {
  const c = s.contact ?? {};
  const opposing = (c.upperForce ?? 0) > .002 && (c.lowerForce ?? 0) > .002;
  if (ctl.phase === "release" || ctl.phase === "done") {
    return .12 * clip(s.jaw / .32, 0, 1) + (!opposing ? .08 : 0) - .01;
  }
  const lift = clip((s.payload[2] - ctl.height) / .04, 0, 1);
  return -.01 + (opposing ? .15 + .3 * lift * Number(!c.otherSupport) : 0);
}

export function jawSafetyFailure(summary) {
  return !summary.finite || summary.maxExternal !== 0 || summary.nonFootSupport ||
    summary.maxPenetration >= .0005 || summary.maxRobotPenetration >= .0005 ||
    summary.maxJointLimitExcess > .002;
}

// Plain exported actor, identical tanh/linear evaluation in Node and browser.
export function jawActor(model, observation) {
  if (model.version !== JAW_RL_VERSION || observation.length !== JAW_RL_OBS_SIZE) throw Error("Incompatible jaw policy");
  let x = observation;
  for (const layer of model.layers) {
    x = layer.weight.map((row, i) => {
      const v = row.reduce((sum, w, j) => sum + w * x[j], layer.bias[i]);
      return layer.activation === "tanh" ? Math.tanh(v) : v;
    });
  }
  if (x.length !== 2 || x.some(v => !Number.isFinite(v))) throw Error("Invalid actor output");
  return { action: Number(x[1] > x[0]), logits: x };
}
