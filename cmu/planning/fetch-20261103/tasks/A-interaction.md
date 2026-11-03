# A — Turn a call and spoken request into one caller-bound fetch job

**Where/why:** the start of the demo, and the identity needed at its end. Existing name response and speech components do not yet prove one shared audio path or persistent caller binding. This task advances P1/P4/P6. **Difficulty 3/5; project novelty 2/5.** Week-one software increment is ready; live caller/gaze acceptance needs B/E and a robot window.

**Build:** an audio-to-event adapter that emits call, orange-ball request and cancel for the correct caller. Reuse existing wake/ASR components; keep interpretation narrow first. Do not write joints or a timed action sequence. Walking calls request stop through E; seated calls preserve seated posture.

### Start with supplied material

From this package: `cd starter/interaction` then `python3 run.py`. No installation, credentials or hardware. The supplied fabricated transcripts produce 21 regression checks and expose two unsupported utterances. `python3 run.py --events call_then_fetch` prints the actual event stream consumed by the coordinator. Read `ADAPTER.md`, `cases.json` and `reference/` for existing source and its dependencies.

### Suggested operations

1. Run the baseline and inspect the typo/paraphrase misses. Add a small separately held-out phrase set, including the actual Chinese request “去把那个橙色的球捡回来”, unrelated conversation, cancellation and a second speaker. Implement bounded normalization/intent handling and keep ambiguous requests unresolved.
2. Implement **one** audio producer that fans time-stamped chunks to wake detection and transcription. Start from supplied source references. Use local WAV input first; then your laptop microphone. The same bytes must reach both consumers; neither opens a second microphone. No teammate needs Shawn's private CLI subscription.
3. Add the transcript/event adapter and replay the same recordings. For real caller binding, consume B's person candidates and associate the call with a visible person; retain a re-matchable cue through the trip. Until that evidence exists, emit unknown identity instead of choosing anyone nearby.
4. Feed the events into E. Test a call while walking, a call while seated, a second person, and cancel during approach/pick. E owns when movement may stop/resume; A supplies intent.

### Deliver and accept

Week 1: code, one command, the local shared-audio replay, labelled transcripts and the resulting events. Freeze negative utterances before tuning; all expected positive requests and cancellations in the stated set must match their caller, and no negative may start a fetch. Report misses/false starts with denominators, not a single “accuracy.” Demonstrate both consumers saw identical chunk IDs/times. Retain the original baseline results for comparison.

Week 2, with B/E: show original caller → look → fetch event → original caller retained when another person speaks or the caller moves. Do not count fabricated identities as passing this gate. A/B jointly own identity; A/E jointly own call/cancel transitions. If identity is uncertain, the expected output is no fetch and a clarification request.

**Handoff:** source revision, recording/label provenance, command and JSON events. The starter's 10-second request window is a changeable development choice; record the chosen value. This task is complete only at its claimed software or live-input level, clearly distinguished.
