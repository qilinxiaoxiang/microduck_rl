import { installMicroduckOverlay } from "../vendor/rl-physics-overlay/src/adapters/microduck.js";

// Optional, read-only, local-only instrumentation. No network or force writes.
export function installPhysicsOverlay(getRuntime) {
  const overlay = installMicroduckOverlay({getRuntime, jointNames:["beak_joint","head_pitch","left_knee","right_knee"],sampleHz:5,historySize:240});
  const unsubscribe = overlay.bus.subscribe(() => {
    const a = getRuntime()?.physicsDiagnostics;
    if (!a) return;
    overlay.element.setDiagnostic(`物体接触：上嘴 ${a.contact.upperForce.toFixed(3)} N，下嘴 ${a.contact.lowerForce.toFixed(3)} N；最大穿入 ${(a.maxPenetration*1000).toFixed(3)} mm；外加力/力矩峰值 ${a.maxExternalForce.toFixed(3)}；稳定夹持 ${a.maxHeldSeconds.toFixed(2)} s。仿真参数尚未实机标定。`);
  });
  return {...overlay,destroy(){unsubscribe();overlay.destroy();}};
}
