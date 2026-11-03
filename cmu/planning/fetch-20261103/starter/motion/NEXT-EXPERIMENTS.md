# The next bounded comparisons

The supplied CPU commands are ready now and already yield failure evidence. The following are **future training experiments**, not ready-to-launch commands. They do not require another researcher to rediscover source/weights. Shared prerequisites are explicit below; no new-body simulation/training is authorized by this package.

## C: does anchor coverage preserve head-down control without suppressing gaze?

Measured gap: the same t2b candidate has 1.67 versus 14.04 action jumps/s on the supplied default/head-down real walks. A similar local gyro Jacobian does not explain this gap. Hypothesis: anchoring only default-head states leaves a harmful off-reference leg-action region when the head is commanded. This is a testable controller-design hypothesis, not an established physical cause.

Code to change: `reference/finetune_worker.py` has the actual `DELTA_T2B`, `anchor_mask`, `anchor_targets`, and `anchor_loss_w` implementation. The team main's older worker is not substituted for it. Once the model is reconciled and reproduces a relevant head-down failure signature, compare unchanged `DELTA_T2B` to the same recipe with only head-command **leg** anchoring enabled (`beta_legs=1.0`), retaining default-head anchor beta 2.0. Do not add the T3 friction/noise/smoothness changes at the same time.

Use walk61 ONNX initialization, the same frozen normalizer, body, solver, command sampler, learning rate and training step budget. Use paired seeds 1 and 2 and a frozen test script containing the supplied default/−0.4-rad neck commands, turns and stop. Budget is determined by the exact-task smoke benchmark first; run no comparison if two variants × two seeds exceed available quota/credit. The exported graphs permit inference and fresh warm-start initialization; they do not provide exact optimizer resumption.

Read back task-relevant head tracking, forward motion, stopping, action jumps and falls on paired held-out cases. A smooth policy that refuses to walk or lower its head fails the hypothesis. Improvement across matched seeds without these losses supports extending anchor coverage. No improvement refutes it as the next repair and justifies investigating a history/dynamics intervention. Physical transfer still requires the same-release guarded test; a simulation result alone cannot close it.

## D: can training the held rise remove the loaded exit mismatch?

Measured gap: the supplied real T2 trace exits bowed by 1.4903 rad relative to stand's nominal neck, followed by a fall. The fixed-state holding-bit intervention changes the neck action by 0.4993 rad. Hypothesis: the learned terminal state conditioned on holding is incompatible with the downstream stand/carry controller.

Code to change: `reference/cmu_seeing_pick_env_cfg.py` defines descent-gated return-pose/upright terms; `reference/cmu_seeing_pix_env_cfg.py` supplies pixel/holding observations; `reference/grasp_worker.py` is the actual bounded worker. `reference/mdp-grasp-lines-7884-7944.txt` documents simulator-held truth. `reference/walk_brain.py` exposes previous-action preservation and the handoff. Source snapshots are for locating the modification, not a complete installable training checkout.

After body/target reconciliation, compare unchanged T2 pixel training against the identical task with the episode extended through **loaded rise plus downstream stand admission** and terminal reward/termination based on actual retained payload and supported downstream posture. Initialize both from the same supplied `spx-b_1800.onnx` actor/normalizer using the verified warm-start path, or explicitly implement/check that path before launch; no optimizer state is implied. Keep sensing, jaw timeline, object/start distribution, solver, PPO settings and total environment steps matched, paired seeds 1 and 2. Do not simultaneously add memory or change camera calibration randomization.

Evaluate from held-out actual upstream exits, preserving joint state, residual velocity, previous action and visual history; do not reset between pick/rise/stand. Read lift, retention, exit posture, first-second action discontinuity/gyro, falls and downstream settling. If both variants leave the same bowed state, terminal training did not address the gap. If the extended variant retains the ball and enters stand/carry compatibly across held-out starts, it supports replacing timer-only admission. This remains a conditional experiment until a proper transfer envelope is defined; the current replay does not establish one.

## Shared training prerequisites and source ownership

- Coordinator supplies the changed battery body's scan, weighed/placed parts and model reconciliation; validates static poses and the relevant dynamic signature before new-body studies. Current readiness is owned only by `microduck-sim/CURRENT-STATE.json`, not the packaged snapshot.
- Coordinator and capability owners freeze target ball geometry/mass, actuator/contact configuration, observation timing and supported entry/exit ranges. Real sensor/power probes must be fresh before physical work.
- Source integrator brings the exact hashed snapshots into a clean training branch with the existing pinned project dependencies, verifies ONNX warm-start semantics and checks export against the training actor. Missing dock-e/f artifacts are not a dependency of these increments and remain unresolved.
- Compute owner measures setup, steps/s, checkpoint/export/resume overhead and peak memory for that exact branch before selecting finite step counts. Kaggle access, $54 user-reported Modal credit and unknown teammate compute are access inputs, not a completed budget feasibility check.

These dependencies prevent a training launch; they do not block either supplied CPU increment.
