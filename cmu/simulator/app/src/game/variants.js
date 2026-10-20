// One robot, one look. Every scene draws the CMU MicroDuck as built: its
// printed parts carry their own filament colours in robot/cmu/kinematics.json
// (generated from microduck_rl/cmu/model, see duck.js ROBOT_KINEMATICS), so
// there are no colourways to choose. This module keeps the small API the
// game, the ghosts and the HUD were written against.

export const VARIANTS = { cmu: {} };
export const VARIANT_NAMES = Object.keys(VARIANTS);
export const DEFAULT_VARIANT = "cmu";
export const VARIANT_LABELS = { cmu: "CMU MicroDuck" };
export const randomVariantName = () => DEFAULT_VARIANT;

// buildRig materialForMesh hook: null keeps the model's own colour.
export const materialHookFor = () => () => null;

// Nothing to re-skin: materials come from the model.
export function applyVariant() {}
