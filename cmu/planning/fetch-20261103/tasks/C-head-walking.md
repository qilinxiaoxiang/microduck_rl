# C — Walk, turn and stop while actually looking down

**Where/why:** looking at a ball changes whole-body balance. A smooth policy that barely moves or refuses to lower its head does not solve the demo. Later the same controller must carry the ball. This advances P5/P6. **Difficulty 5/5; project novelty 3/5.** CPU baseline/diagnostic work starts now; new-body training waits for F0.

**Build:** improve the existing walker to follow head/neck and velocity requests together. Start with unloaded motion, then extend to D's real loaded exit. Preserve a named supported range for B/E.

### Start with supplied material

Use `starter/motion/README.md` to install its pinned CPU environment. Run `python replay.py locomotion --out out/gait.json` with that environment. The package contains real walk61 and t2b exports, bounded historical sensor/action traces, source snapshots and hashes. It reproduces the head-down jitter comparison. Raw 61-value observations were not logged: reconstructed-input inference is explicitly approximate, not exact controller parity. Historical models are not accepted for the new backpack.

### Suggested operations

1. Reproduce the default-head versus lowered-neck trace diagnostic. Inspect action jumps alongside head error, speed/distance and tilt. Check compensation/filtering/observation semantics; the package documents the opt-in compensation trap.
2. Inspect the supplied finetune/environment source. Prepare a two-condition evaluation using identical velocity commands, head commands and initial states. After F0, run it unchanged on the matched model. If it cannot reproduce the relevant failure, first fix model/actuator/timing agreement; do not hide mismatch by changing reward weights.
3. Compare **one** change: retain velocity/head tracking while adjusting the original-policy leg reference strength. Keep head joints outside an anchor that would force the old head pose. Reject changes that reduce jitter by not walking or not looking down. A short observation/action history is a later comparison if measured timing uncertainty is the unresolved limitation; its benefit alone does not prove the physical cause.
4. Evaluate B's recorded requests, then D's supported loaded exit. Export through the pinned input/action adapter and replay the same tests. Do not infer compatibility from equal vector length.

### Deliver and accept

Week 1: baseline readback, evaluation code/config and one precisely specified comparison; a CPU diagnostic is already possible. Supply action-jump rate, head error, travel/velocity tracking, tilt and falls together. Preserve failed traces. No new-body training claim before F0.

After F0: freeze a physically reachable head/velocity range and trial starts before tuning. Candidate must improve the preselected failure metric while retaining required head/velocity tracking and stable stop across the range. Record real-condition and stress results separately. Compare at least two training seeds if compute permits; one seed is exploratory, not established robustness. Only a matched exported version passing protected robot trials advances to live B/D integration.

**Handoff:** model/source/config hashes, units/joint mapping, supported head/velocity/payload range, previous-action/history initialization, all seeds and failures. D does not wait for this new model to begin isolated development: both start from the shared historical baseline after F0. Loaded walking is the next increment after D supplies a validated exit and payload.
