# E — Complete the fetch, return and human handover in one runtime

**Where/why:** independently good modules do not make a fetch robot. This task explicitly owns sequencing, return, waiting, authority and recovery so those steps do not disappear into “integration later.” It advances P0/P4/P6. **Difficulty 4/5; project novelty 3/5.** Motor-free composition starts now; live readback depends on the producing tasks/F0.

**Build:** connect A/B/C/D to the existing single motor executor; retain original caller and payload through the entire job. A complete result is return to that caller and wait for the person to take the ball. A safe stop/help is a valid failure outcome, not a successful delivery.

### Start with supplied material

From `starter/`, run `python3 rehearse.py`. Standard library only. It imports the actual A parser, reads a saved B detector result and composes 22 scenarios. Read `out/rehearsal.json`, `coordinator.py` and `CONTRACT.md`. Reproduce with fresh B output via `python3 rehearse.py --tracks vision/out/tracks.jsonl`. All physical acknowledgements are labelled fixtures; this is not a simulated or real fetch success.

### Suggested operations

1. Trace the complete scenario and its failure branches. Add a small runtime adapter that maps named intents to the supplied existing executor source in `runtime-reference/`. In observation-only mode, log intended commands and actual body/sensor readback without sending motor commands.
2. Replace fixture acknowledgements one at a time: C provides settled; B/D provide entry; D provides lift/hold/supported exit; B/C provide wait-zone arrival. E implements verified human-take detection with D's payload evidence and labelled take/drop clips. Unknown must stay unknown.
3. Bind each command/result to the active task epoch and one motion owner. Cancel requests a supported exit; manual/emergency revokes all old work. Calls during walking wait for stop; calls during picking do not interrupt the body controller. Protect loaded grip from quacks/expressions.
4. Integrate original-caller search, approach, wait and take. On target loss, stop/reacquire; on empty bite, return to a validated search state; on unsupported exit or drop, stop/help. Add autonomous retries only from states the adjacent controller can accept.

### Deliver and accept

Week 1: observation-only adapter, complete transition log and positive/negative readback examples. All 22 supplied cases must retain their expected outcomes; add actual producer-output replays as they arrive. Test expired camera data, wrong person, cancel during bite and old queued commands after manual takeover. A timer or a Boolean written by the coordinator cannot stand in for physical success.

Week 2, conditional: one continuous A→B/C→D→B/C→wait/take trial, plus wrong-person, loss/drop and cancellation branches. Bind video and logs to one source/model version. No reset or manual target correction between segments. Count interventions and failure stages. Freeze the physical trial set before testing; this plan does not certify a success rate in advance.

**Handoff:** adapter, event log, authority/readback tests and a restart procedure that cannot resume an old job. E shares each transition check with its producer. If a sensor cannot distinguish take from drop, mark the handover gate incomplete and request help; do not relabel object disappearance as delivery.
