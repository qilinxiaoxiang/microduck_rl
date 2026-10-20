const scalar = (value, fallback = 0) => {
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  const first = value?.[0];
  return Number.isFinite(first) ? first : fallback;
};

const item = (values, index) => {
  const value = values?.[index];
  return Number.isFinite(value) ? value : null;
};

const safeName = (mujoco, model, kind, id, fallback) => {
  try {
    return mujoco?.mj_id2name?.(model, kind, id) || fallback;
  } catch {
    return fallback;
  }
};

function resolveJointIds({ mujoco, model, jointNames, maxJoints }) {
  const jointKind = mujoco?.mjtObj?.mjOBJ_JOINT?.value
    ?? mujoco?.mjtObj?.mjOBJ_JOINT
    ?? 3;
  const requested = jointNames?.length ? jointNames : null;
  if (requested && mujoco?.mj_name2id) {
    return requested
      .map((name) => {
        try {
          return { id: mujoco.mj_name2id(model, jointKind, name), name };
        } catch {
          return { id: -1, name };
        }
      })
      .filter(({ id }) => id >= 0)
      .slice(0, maxJoints);
  }

  const count = Math.min(scalar(model?.njnt), maxJoints);
  const hingeKind = mujoco?.mjtJoint?.mjJNT_HINGE?.value
    ?? mujoco?.mjtJoint?.mjJNT_HINGE
    ?? 3;
  const ids = Array.from({ length: scalar(model?.njnt) }, (_, id) => id)
    .filter((id) => !model.jnt_type || Number(model.jnt_type[id]) === Number(hingeKind))
    .slice(0, count);
  return ids.map((id) => ({
    id,
    name: safeName(mujoco, model, jointKind, id, `joint_${id}`),
  }));
}

export function createMujocoSnapshotReader({
  mujoco,
  model,
  data,
  jointNames = [],
  maxJoints = 24,
  label = "Microduck / MuJoCo WASM",
} = {}) {
  if (!model || !data) throw new TypeError("model and data are required");

  const joints = resolveJointIds({ mujoco, model, jointNames, maxJoints });
  const qposAddress = model.jnt_qposadr;
  const dofAddress = model.jnt_dofadr;

  return function readSnapshot() {
    const jointValues = joints.map(({ id, name }) => {
      const qIndex = Number(qposAddress?.[id] ?? id);
      const dofIndex = Number(dofAddress?.[id] ?? id);
      const generalized = {
        actuator: item(data.qfrc_actuator, dofIndex),
        inertia: null,
        bias: item(data.qfrc_bias, dofIndex),
        constraint: item(data.qfrc_constraint, dofIndex),
      };
      return {
        name,
        positionRad: item(data.qpos, qIndex),
        velocityRadps: item(data.qvel, dofIndex),
        actuatorTorqueNm: generalized.actuator,
        constraintTorqueNm: generalized.constraint,
        generalized,
        wrench: { forceN: [null, null, null], torqueNm: [null, null, null] },
      };
    });

    return {
      source: label,
      timeS: scalar(data.time),
      mode: "live",
      joints: jointValues,
      contacts: {
        count: scalar(data.ncon),
        normalForceN: null,
        frictionForceN: null,
      },
      availability: {
        qpos: Boolean(data.qpos),
        qvel: Boolean(data.qvel),
        actuatorTorque: Boolean(data.qfrc_actuator),
        constraintTorque: Boolean(data.qfrc_constraint),
        contactWrench: false,
      },
      notes: [
        "Fast mode only reads arrays already produced by MuJoCo.",
        "Generalized qfrc channels are not presented as a six-axis spatial wrench.",
        "Unavailable channels are shown as unavailable, never synthesized.",
      ],
    };
  };
}
