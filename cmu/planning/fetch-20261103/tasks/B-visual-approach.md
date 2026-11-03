# B — Keep the selected ball or original caller in view and approach it

**Where/why:** outward trip and return use the same visual following behavior. Existing orange detection/gaze code does not prove “ball” identity, continuous tracking near the mouth, or return to the right person. This task advances P0/P1/P6. **Difficulty 4/5; project novelty 3/5.** Image/protocol work starts now; movement calibration waits for F0/C/D.

**Build:** target selection, temporal tracking and gaze/velocity requests for ball and person modes. Output intentions and observation age; C handles balanced joint motion. E decides which target/phase is active.

### Start with supplied material

From `starter/vision`, follow `README.md` to create the pinned environment, then run `.venv/bin/python run.py --out out`. It includes the actual orange detector, a composite positive image, no-orange negative and orange non-ball challenge. Inspect the annotated outputs and `tracks.jsonl`: the rectangle is a preserved false positive, not a solved benchmark. Eight explicit track cases exercise identity/loss handling; person identities are fixtures, not recognized people.

### Suggested operations

1. Keep the baseline output. Add ball/non-ball and self-orange-part negatives before tuning; record lighting/viewpoint split. Begin with size/shape/temporal consistency around the existing detector; add a compact semantic detector only if this baseline fails the frozen distractors. Never special-case a supplied filename or pixel position.
2. Maintain one selected target across frames. Use the supplied current gaze/controller source reference to turn image-center error into head requests. Add a genuine person-observation adapter and a reidentification cue shared with A; a tracker ID that resets after looking away is insufficient.
3. Keep capture time and freshness through occlusion. Use short history to search/reacquire; stale/unknown identity requests a stop, not continued forward motion. Show selected target, age, gaze error and request overlaid on video.
4. Add approach speed/turn requests constrained by C's supported range. Use a configurable D entry profile and E wait zone. First exercise them with supplied protocol fixtures; after F0 calibrate them with actual camera/body readback and continuous B/C → D entries.

### Deliver and accept

Week 1: code plus a replay command that produces overlays and track/request JSON from the frozen clips. Compare to the original detector; report ball recall, distractor false selections, identity switches, stale-command duration and latency with counts. Every wrong/uncertain/stale target must stop approach in the contract tests. Improving one published rectangle does not establish generalization.

Week 2, conditional: jointly demonstrate B/C reaches D's published entry range from varied starts, then B/C/E returns to the original caller after that person moves beside a distractor. Preserve complete failures, not selected clips. Before trials, freeze the test starts and measured wait/pick ranges; success means arriving inside those ranges without manual target/pose correction. The first gate is range compatibility, not an invented universal centimeter threshold.

**Handoff:** target adapter, timestamps/identity semantics, selected-target recording, movement requests, supported camera/head range. B/C own gaze-versus-balance; B/D own entry; A/B/E own original-person return. If D's region is not yet known, software work proceeds with a fixture profile and physical entry remains pending.
