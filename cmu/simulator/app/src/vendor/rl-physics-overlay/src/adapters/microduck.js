import { createTelemetryBus } from "../core/telemetry-bus.js";
import { createMujocoSnapshotReader } from "./mujoco-snapshot.js";
import { definePhysicsOverlay } from "../components/rl-physics-overlay.js";

export function createDynamicMujocoSnapshotReader({
  getRuntime,
  mujoco,
  model,
  data,
  jointNames = [],
  maxJoints = 24,
} = {}) {
  const fixedRuntime = { mujoco, model, data };
  const resolveRuntime = getRuntime ?? (() => fixedRuntime);
  let activeModel = null;
  let activeData = null;
  let readSnapshot = null;
  let rebindings = 0;

  const read = () => {
    const runtime = resolveRuntime();
    const currentModel = runtime?.model;
    const currentData = runtime?.data;
    if (!currentModel || !currentData) return null;
    if (currentModel !== activeModel || currentData !== activeData) {
      activeModel = currentModel;
      activeData = currentData;
      rebindings += 1;
      readSnapshot = createMujocoSnapshotReader({
        mujoco: runtime.mujoco ?? mujoco,
        model: currentModel,
        data: currentData,
        jointNames,
        maxJoints,
        label: "Microduck Simulator",
      });
    }
    return readSnapshot();
  };
  read.stats = () => ({ rebindings, ready: Boolean(readSnapshot) });
  return read;
}

export function installMicroduckOverlay({
  mujoco,
  model,
  data,
  getRuntime,
  mount = globalThis.document?.body,
  jointNames = [],
  maxJoints = 24,
  sampleHz = 10,
  historySize = 240,
  initiallyOpen = true,
  layout = "overlay",
} = {}) {
  if (!mount) throw new TypeError("mount is required in a browser environment");
  definePhysicsOverlay();

  const bus = createTelemetryBus({ capacity: historySize, maxHz: sampleHz });
  const readSnapshot = createDynamicMujocoSnapshotReader({
    getRuntime,
    mujoco,
    model,
    data,
    jointNames,
    maxJoints,
  });
  const element = document.createElement("rl-physics-overlay");
  if (layout === "studio") element.setAttribute("layout", "studio");
  element.open = initiallyOpen;
  mount.append(element);
  element.connect(bus);

  let running = true;
  let frameId = null;
  let lastSampleAt = -Infinity;
  let readErrors = 0;
  const intervalMs = 1000 / Math.max(1, sampleHz);

  const tick = (now) => {
    if (!running) return;
    if (!document.hidden && now - lastSampleAt >= intervalMs) {
      try {
        const snapshot = readSnapshot();
        if (!snapshot) {
          frameId = requestAnimationFrame(tick);
          return;
        }
        bus.publish(snapshot, now);
        lastSampleAt = now;
      } catch (error) {
        readErrors += 1;
        element.setDiagnostic?.(`telemetry read failed: ${error.message}`);
      }
    }
    frameId = requestAnimationFrame(tick);
  };
  frameId = requestAnimationFrame(tick);

  return Object.freeze({
    element,
    bus,
    pause() { running = false; if (frameId) cancelAnimationFrame(frameId); },
    resume() {
      if (running) return;
      running = true;
      lastSampleAt = -Infinity;
      frameId = requestAnimationFrame(tick);
    },
    destroy() {
      running = false;
      if (frameId) cancelAnimationFrame(frameId);
      element.disconnect?.();
      element.remove();
    },
    stats() { return { ...bus.stats(), ...readSnapshot.stats(), readErrors }; },
  });
}
