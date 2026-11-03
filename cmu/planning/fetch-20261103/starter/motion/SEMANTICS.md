# Observation, action and evidence contracts

All supplied graphs accept float32 `[1,61]`, emit `[1,14]`, and contain their own observation normalization. Do not normalize a second time. Action scale is 1: output is a joint-position offset in radians from the ONNX metadata's default. Runtime then adds the default and clamps goals. A 4.978-rad logged action is a neural output, not evidence of that joint actually moving 4.978 rad or an unclamped motor command.

The policy joint order is in `data/model-metadata.json`: left hip yaw/roll/pitch, left knee/ankle, neck pitch, head pitch/yaw/roll, right hip yaw/roll/pitch, right knee/ankle. Raw robot-log q/dq order is different: right ankle/knee/hip pitch/roll/yaw, left ankle/knee/hip pitch/roll/yaw, head pitch, neck pitch, head yaw/roll. The runner maps by name.

Zero-based observation slots:

- 0–2: trunk-frame angular velocity, rad/s.
- 3–5: gravity unit vector in the trunk frame; upright is `[0,0,-1]`.
- 6–19: measured joint angles minus model default, policy joint order, rad.
- 20–33: joint velocities in policy order, rad/s.
- 34–47: previous **applied** action in policy order. Candidate inference always uses the same recorded history, not its own imagined closed-loop history.
- 48–50: velocity graphs use `[vx m/s, vy m/s, turn rad/s]`; pixel-pick instead uses `[cos(2π phase),sin(2π phase),0]`. Stand admission uses zero commands.
- 51–54: walking head-pose deltas in neck pitch/head pitch/yaw/roll order, after runtime smoothing. Pick uses zero. This is the actual concatenation in `reference/walk_brain.py`; an older comment incorrectly names slots 55–58.
- 55–60: velocity/stand use zero body commands. Pixel-pick uses `[u,v,width,age,holding,visible]`: u/v normalized to −1…1 across 640×360, width/640, age seconds capped at 2, holding bit and visibility bit. Invisible retains last box and increases age. These are not floor coordinates; old seeing-pick coordinates are incompatible.

The runner explicitly switches phase/pixel body fields to zero stand commands/body fields at the handoff and preserves prior action. It never feeds a pixel vector to a stand graph merely because both have 61 slots.

## Real data and limits

The four `*-real.jsonl` files contain selected fields from actual previously copied **local** robot logs, with original 1-based source lines. Full original file hashes and extraction fields are in `PROVENANCE.json`. No SSH, remote-log parsing, new motion capture, personal audio or images are involved.

There are no captured raw61 observation vectors in these logs. Reconstructed vectors use rounded logged q/dq/gyro/gravity, logged smoothed head command/turn, and prior applied action. Discontinuities skip the first unknown-history row. These are approximate inputs derived from real data, not exact actor-input recordings. The current source's `walk_wired.py` enables prediction only with `--compensation`; the recorded t2b launch omits it. `age_used` alone does not mean prediction was applied. The runner includes the incorrect predicted-state calculation as an explicit diagnostic comparison.

Walking reconstruction retains a maximum action discrepancy of 0.130/0.133 rad despite mean errors 0.00173/0.00139. Exact historical runtime/command timing and unrounded inputs were not captured; those residuals are unresolved, and this is not export parity. Pick and stand maxima are 0.00259/0.00242 rad. All error denominators are reported, not filtered away. Logged jaw-holding is a proxy; no independent object-lift truth is fabricated. Logs have no forward odometry.

`synthetic-inference-fixtures.npz` is separately marked synthetic: deterministic nominal states generated from graph metadata and per-policy command semantics, with expected outputs from the supplied exports under the pinned CPU runtime. No `.pt` actor/optimizer checkpoint is packaged, so true actor-versus-export parity remains untested.
