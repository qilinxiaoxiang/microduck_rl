# Voice-to-dock integration handoff

## Scope and status

The robot agent reports the immediate goal as foreground-armed voice control:
Mochi, then Fetch, finds the orange ball and docks. **Dock-only does not pick
up or return the ball.** The agent's handoff describes separate manual and
voice modes, with dashboard Stop and Torque Off available in voice mode; this
differs from an earlier request for ordinary manual commands to override voice.
Shawn and the operator should confirm the final intended behavior before
integration. Voice Stop is not a safety stop; keep the independent cutoff.

The public handoff branch is based on `voice/mochi-fetch-stop` at
`3b4431564161e5e90fefe7bdbf14fb0675fceb86`. The branch contains the Mac
shadow prototype and this handoff, but not the robot agent's implementation.
No operational dashboard or robot process was restarted or modified here.

## Voice interaction: the intended experience

The voice work has three spoken keywords: **Mochi** is the wake word;
**Fetch** and **Stop** are the two command intents after wake.

1. Listen for Mochi through the existing name-call microphone owner. A wake
   opens a short command window; keep earlier audio so "Mochi, fetch" in one
   utterance can be understood. The Mac shadow prototype uses a 1.5-second
   window and up to one second of pre-roll; these are not robot-validated
   timing thresholds.
2. After wake, Fetch requests the existing orange-ball find-and-dock workflow
   described below. A repeated Fetch must not start a second active job.
3. After wake, Stop requests cancellation of the current job. A **new Mochi
   wake is required before every voice Stop, including during Fetch**. The
   exact standalone transcript "stopped" after wake is also accepted by the
   Mac rules. Wake alone, speech without wake, and ambiguous or stale results
   must not start or cancel a job. Manual controls and the independent physical
   cutoff do not depend on voice recognition.

The intent labels are not motor commands. A future supervisor must check wake
and capture timing, job/session generation and current control authority, then
send only through the reviewed shared boundary. A late recognition result or
cancelled job must not overwrite a later manual command. The Stop path is not a
reliable emergency stop, and the proper docked-pose cancellation transition
remains to be established.

## Recognition and GPT work

The user's experimental pipeline explored complementary ways to understand
speech; they are not all connected or validated on the robot:

```text
shared microphone -> Mochi wake -> buffered command audio
                                      |-> experimental Fetch/Stop keyword scores
                                      `-> local Whisper ASR -> transcript
                                           |-> conservative local intent rules
                                           `-> optional GPT semantic comparison
                                                        -> fetch / stop / none
```

- **Wake and keywords:** the portable bundle contains experimental
  openWakeWord Mochi, Fetch and Stop heads. The robot agent separately reports
  `mochi_v6.onnx` on the robot; confirm the actual active detector rather than
  assuming it is the bundled Mochi model. Fetch and Stop scores are experimental,
  not proof that a command was understood. No keyword head independently enables
  motion or replaces the wake, authority and timing checks.
- **ASR:** Whisper tiny.en converts captured speech to English text locally.
  The Mac prototype accepts supplied audio and wake timestamps; it does not
  own the microphone or invoke a real wake detector. Its exact-phrase rules
  recognize `fetch`, `stop`, `stopped` and a few polite variants. They do
  **not** currently recognize the longer phrase "fetch the orange ball".
- **GPT/AI:** the desired optional semantic layer interprets a transcript as
  `fetch`, `stop`, or `none`, helping with natural requests such as "fetch the
  orange ball" and avoiding false commands from negation or conversation.
  It does not itself detect wake, infer speaker identity or drive the robot.
  When disabled or unavailable, use conservative local rules and report the
  cloud error explicitly rather than fabricating a result. Broader semantics,
  timing and false-positive behavior need separate validation before use.

The Mac `voice_fixture_compare.py` is a mocked comparison, not a GPT client
or live decision path. The earlier OpenAI experiment and local keyword/audio
evaluators are separate offline shadow tools; the robot agent reports another
shadow loop whose current code and configuration still need review. The
approved Mac experiment permits cloud requests only with fabricated fixture
text: do not send private audio, transcripts, code or credentials externally.
Any live cloud path needs separate permission and privacy review; `store:
false` is not a zero-retention guarantee.

One user-run synthetic comparison with 18 isolated clips produced 5/6 Fetch
and 2/6 Stop classifications from OpenAI after local ASR. The experimental
Fetch and Stop heads each detected 0/6 of their corresponding clips in that
same run. This small, synthetic test did not exercise a live wake gate or
robot and does not establish real-room accuracy or safe cancellation.

For transfer, `microduck-voice-openai-pipeline-20261120.zip` on the Mac
contains the experimental keyword models, local Whisper weights, evaluators,
fixtures, provenance and documentation. It is not in this Git branch, contains
no API key or live robot integration, and needs its licenses and permissions
reviewed before redistribution. Some older bundle documents describe Stop
without a wake; the **wake-before-every-Stop rule above supersedes them**.
See [VOICE-MODELS-AND-DESIGN.md](VOICE-MODELS-AND-DESIGN.md) for exact model
hashes, how to obtain the bundle from Ryan, test results and architecture
decisions.

## Existing workflow

The locally available `origin/pick-and-rise` snapshot was `b789de58` during
this review. Its `cmu/mac/dockpick.py`, reached via `seepick.py`, implements
find, aim and dock; `--dock-only` ends in a small-step hold. Full `--dock`
adds a learned kneel/pick and rise, and is a different scope. The robot agent
reports the staged dock-only route does not inherently depend on Task C.
Do not substitute the separate Task C Fetch path without team review.

The 2026-11-21 branch report records a fall during a dock scan and another
during a small step on a different physical configuration than the simulated
body. These are historical outcomes, not a current hardware assessment.
The branch source also reports that changing the docked hold to `monitor`
fell in simulation; do not assume `monitor` is safe in that pose.

## Observed robot state

On 2026-11-25, read-only SSH reached the A7Z. The local SSH alias `a7z`
did not resolve in this environment. A running `walk_wired.py` process used
the `bw-s3-wh2-skills` policy set and had no `--arbiter-socket` in
its command line. This is a snapshot, not proof of the current state at
handoff. `git` was unavailable in the noninteractive robot shell, so the
robot checkout revision was not verified here.

The other agent reported unfenced legacy writers, a brain lacking dock mode,
and dock models in staging but not loaded by that brain. Verify these claims
against the active process, its loaded model metadata and all launchers before
changing anything. Do not treat files merely present on disk as loaded or
compatible. The other agent's protocol and maintenance document is
`SIM_CONTROL_BOUNDARY.md` in the robot's firmware checkout.

**Concurrent-edit warning:** while the isolated dashboard tests were running,
the robot's `control_boundary.py` changed; a later import failed with
`IndentationError` at line 1, whose displayed beginning was ` e"""`. The
file's modification time was 2026-11-25 14:03:41 -0500. We did not edit it.
It may be an in-progress edit on the shared machine. Coordinate with its owner,
inspect its current contents, and re-run syntax and tests before using it.
Do not overwrite or restore another person's changes.

## Mac dashboard review patch

The other agent's `dashboard-boundary-review-only.patch` was created on the
A7Z. A working copy was transferred to the Mac session's `files/` directory,
not to the repository. SHA-256:
`918ec31b08d36bb21b04f62551aaca7a27868c37278f9f331df2ddb98c70a937`.
Session storage is not a durable team distribution mechanism; transfer the
patch and tests through an agreed team channel.

The patch is based on `867edf0598cc401763b23f44e7b640ebaac2e0c8`. It applies
and its three files pass syntax checks in a disposable copy of that exact base.
It does **not** apply cleanly to this branch's newer dashboard: the bridge and
browser files differ from the patch base. It must be deliberately ported and
reviewed with those changes preserved, not forced onto the live checkout.
It is review-only and not a production or physical-safety qualification.

The patched files in the disposable copy matched the robot agent's staged
helper, bridge and browser files by SHA-256. Against the agent's test setup on
the robot, four isolated helper tests passed, including missing-boundary
refusal. The fake-SSH bridge test failed to start its fake server; a subsequent
import identified the concurrent `control_boundary.py` syntax error above.
Thus the bridge test was **not verified** here. Tests are located in the robot
agent's session directory `files/dashboard-boundary/` and use fake sockets and
fake SSH; their test-only verified capability flags are not evidence of real
writer fencing. Browser UI testing remains open.

The dashboard sends its command helper from the Mac over SSH; editing only the
robot's copy does not migrate dashboard writers. Existing open writer sessions
continue running the code they started with.

## Mac voice shadow

`VOICE-SHADOW.md` documents the current motor-free prototype and exact local
test command. On this branch, 22 tests passed using the existing Mac Python
environment. It does not open a microphone or issue commands. The robot agent
describes a separate OpenWakeWord/Whisper/OpenAI shadow loop; its current
source and configuration were not inspected here. Do not assume the Mac
prototype is connected to that loop.

The earlier approved Mac plan allows cloud experiments only on fabricated
fixtures. The robot agent reports a separate live-transcript-to-OpenAI path;
its privacy permissions and configuration must be reviewed separately, rather
than assumed authorized by the Mac prototype.

## Next work for the team

1. Coordinate access to the shared robot checkout; resolve the concurrent edit
   and preserve the robot agent's working tree. Obtain exact revisions and
   the latest test results for boundary, supervisor, staged dock and dashboard.
2. Port the review-only dashboard patch onto the current Mac branch without
   dropping existing features; exercise the real browser and fake-server
   failure and sequence cases. Agree on mode behavior first.
3. Integrate the existing dock-capable runtime and compatible model set into
   the current brain without replacing newer changes. Test command validation,
   deadlines, guards and the docked final hold and cancellation transition;
   do not enable capability flags merely to pass a handshake.
4. Migrate or exclude every legacy writer, startup path and old SSH session.
   Implement foreground arming, disarming and process-death handling. Test
   takeover, cancellation, stale commands, disconnect, failures and restart
   against the actual integrated brain with fake devices.
5. With Shawn, establish a coordinated maintenance procedure and secured
   no-motion end-to-end checks before any separately agreed supervised motion.
   Record robot body configuration, installed policies, camera calibration,
   free disk, test results and operator/cutoff arrangements.
