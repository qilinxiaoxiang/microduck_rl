# MicroDuck: one fetch behavior, five claimable tasks

**Goal:** call “Mochi” → look at the caller → hear “fetch the orange ball” → approach while watching → bite and rise → carry back to **that same person** → keep holding until the person takes the ball.

We are five people including Shawn. These are tasks to claim, split and hand over, not permanent person assignments. The package supplies actual starting code, models, inputs and offline commands. It does **not** claim that audio recognition, autonomous grasping or new-backpack walking are already solved.

## Live task Issues

- [[Fetch A] Caller-bound voice interaction](https://github.com/qilinxiaoxiang/microduck_rl/issues/2)
- [[Fetch B] Continuous vision and approach](https://github.com/qilinxiaoxiang/microduck_rl/issues/3)
- [[Fetch C] Head-aware walking and carrying](https://github.com/qilinxiaoxiang/microduck_rl/issues/4)
- [[Fetch D] Visual pickup through a stable loaded rise](https://github.com/qilinxiaoxiang/microduck_rl/issues/5)
- [[Fetch E] Full fetch execution, return and handover](https://github.com/qilinxiaoxiang/microduck_rl/issues/6)

All five were published unassigned. Claim or transfer an increment in the corresponding Issue comments; see the handover instructions below.

## Read this first

- **A — Interaction:** hear a call/request and retain who asked. Produces task events; never controls joints.
- **B — Vision and approach:** keep the selected ball/person in view and request gaze/speed. Reused on the outward and return trip.
- **C — Head-aware walking:** execute gaze and velocity together while balancing; later carry a payload.
- **D — Pick and rise:** take over from the actual approach state, lift, and produce a stable loaded exit.
- **E — Fetch execution and recovery:** connect the events/controllers, return to the original caller, wait for take, and handle failures. This is an explicit implementation task, not unnamed “core work.”

[English task page](index.html) · [Discord draft](DISCORD.md) · [Start locally](starter/README.md) · [Interfaces](CONTRACT.md) · [Inputs and provenance](INPUTS.md)

## What can start now

A/B/E can implement software using the supplied transcripts, images and protocol fixtures. C/D can reproduce actual historical policy inference and failure diagnostics on a CPU now. Their **new-body training and robot acceptance wait for F0** below. ONNX exports support inference/reference supervision; they are not optimizer checkpoints for exact PPO resumption.

The included coordinator consumes actual A parser output and B detector output. Its physical acknowledgements are visibly labelled fixtures. This checks that the proposed software boundaries connect; it cannot certify a robot movement. The current detector deliberately exposes an orange non-ball false positive, and the parser exposes two missed utterances.

## Critical path, with no circular start condition

**F0: match the new body → C: supported head/velocity range → B/C: approach → D: pick through loaded rise → C: loaded walk → B/E: original-person return and human take.** A develops alongside this path.

F0 is owned by Shawn with assistant support: Shawn supplies the new scan, component weights/locations and physical observations; the assistant reconciles the mass/model manifest, limits, camera geometry and source bundle. The assistant supplies the files and runnable model check, rather than asking a teammate to discover the history. See [BODY-READINESS.md](BODY-READINESS.md).

To start C and D independently, first use the same frozen historical models and trace package. After F0, D can develop from measured stable entry states using the existing baseline controller; it does not wait for a new C model. B develops against a configurable entry profile. D publishes its tested entry/exit region; B/C then must reach it. C first targets unloaded walking; it adds the payload/pose supplied by D next. Development resets and fixtures are allowed only in isolated development; final adjacent-controller acceptance runs continuously without resetting the robot.

If the regions do not overlap, the B/C/D claimants revise the entry maneuver or trained region together. Do not label an incompatible downstream controller “finished.”

## Two-week working rhythm

**First meeting, 30 minutes:** run the starter, demonstrate its known errors, choose available task increments, and agree who reviews each adjacent handoff. Claim only what your time permits.

**Week 1 checkpoint:** A/B/E deliver one working improvement with saved inputs/outputs; C/D deliver a reproduced failure plus one bounded change/comparison ready for the matched model. F0 supplies measurements when available. Every claimant leaves a branch and a one-command reproduction within two days, so anyone can take over.

**Week 2 checkpoint, conditional on F0:** join A/B/E to live sensor readback first; test C in the measured range; test D from continuous B/C exits, then loaded C and the return. If a physical gate fails, keep the failed clip/trace and change the relevant controller; do not bypass it to make a demo video. One to two weeks is a checkpoint for useful increments, **not a promise of complete autonomous fetching**.

With low teammate availability, Shawn and the assistant follow F0 → C/D → B/E while others can land A/B/E increments independently. The source package and shared acceptance inputs make handover possible without repeating private setup.

## Claim and transfer

To claim work, comment on its GitHub Issue with the task or week-one increment, your available days and first deliverable. All five Issues start unassigned. A comment records the current claimant; it is not a permanent person assignment. Work in a `codex/` or team-agreed feature branch. Before transferring, leave a handover comment on the same Issue with: branch/revision, exact command, input/config hashes, current output, unresolved failure, next operation. If you become unavailable, post the handover and mark the increment open to takeover. The next person replies on the same Issue with what they are picking up.

A change to units, action meaning, model identity or entry/exit profiles must include the producer and consumer updates plus the composed check in one review. Results record both the source hash and physical configuration. A single motor runtime writes joint commands; E arbitrates intent, C/D implement motion. No two controllers write motors concurrently.

Difficulty and project novelty use 1–5 judgments: difficulty includes implementation and transfer uncertainty; novelty means new **for this project**, not scientific originality. Problems advanced: P0 grounded success, P1 continuous observations/actions, P4 recovery, P5 model fidelity, P6 integration.
