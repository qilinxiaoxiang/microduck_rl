# MicroDuck pick-up with re-aim, re-approach and retry: literature pass

Verification date: 2026-10-26. Sources were read on arXiv (HTML or PDF text). Quotes are under 15 words. "Inference" marks
my reading for MicroDuck; everything else is what the paper states.

Earlier pass, not repeated here: `cmu/aim_pick/README.md` ("Published recipe" section) summarises the 2026-10-21 note
`aim-crouch-literature-20261021.md` (that file was not found under `~/code` or `/tmp` on 2026-10-26; only the README
summary survives). It covers goal frame (UMI on Legs Table 1, Pedipulate Sec. II), exp kernel and sigma curriculum,
late goal reward plus progress term (2409.16048 Sec. III-E), error-gated goal range, scripted gripper (DWBC App. F,
DoggyBot Sec. V-C), beak pushes objects (VBC App. I), goal-ignoring check (AMO Sec. IV-A), knee term (HOMIE Eq. 4).

## Bottom line

1. Every published real legged or humanoid pick-up system that retries is hierarchical: a learned low-level body
   policy under a higher layer that re-issues targets. The retry itself came from two routes: (a) a learned
   high-level policy that re-plans every step from fresh perception and was trained with episodes long enough to
   fail and try again (VBC, ASC, Skill Transformer, QT-Opt); (b) a state machine reading an explicit "holding"
   signal (DextrAH-G, DribbleBot's recovery switch). No paper found trains one monolithic joint-level policy for
   walk + aim + grasp + search + retry on a real legged robot.
2. Measuring the object once and freezing it is what failed in DoggyBot's ablation; systems that work re-estimate
   at 10-30 Hz and train with delay and dropped frames, plus a short history (4 frames in VBC, 15 steps in
   DribbleBot, LSTM in VIRAL and DextrAH-G).
3. Retry behaviour needs training episodes that contain failures: random object relocation mid-episode (VBC 10 %,
   DribbleBot teleport every 7 s, DextrAH-G random wrenches p = 0.1 per step) and mid-task start states (DeepMimic
   RSI, PhysHSI hybrid RSI, VIRAL demo-state starts).
4. Grasp state enters as a low-dimensional observation (QT-Opt binary gripper status: 53 % -> 58 % -> 70 % with
   height; Skill Transformer binary is_holding). No RL paper found uses gripper motor current as the slip signal;
   a scripted current/position monitor feeding a binary "holding" bit is the closest published pattern.

## Q1. Monolithic vs hierarchical, and who actually retries

- **Visual Whole-Body Control (VBC, arXiv 2403.16967).** Low-level whole-body tracker plus a high-level
  task-planning policy fed segmented depth and proprioception (Sec. 3). The high level outputs the gripper bit
  itself (Sec. 3.2.1). Retry is learned: "allows for autonomous retrying behaviors when one pick-up trial fails"
  (Sec. 4.2); "emerges to learn a retrying behavior in simulation training" (Sec. 4.2). Their trick is App. C.2
  item 4: with 10 % probability the object position and pose are randomly changed, which "helps the robot to learn
  the retrying behaviors". Scripted guard: base velocity forced to 0 while the gripper closes (App. C.2 item 5).
- **Helpful DoggyBot (2410.00231).** RL low level, VLM/vision pipeline on top; grasp fires when coordinates are
  within a threshold. No retry or failure detection described; reports first-attempt rates (Ground + Ball 70 %).
  Its "No Tracking" ablation fails because it cannot compensate navigation drift (see Q2).
- **VIRAL (2511.15200).** Teacher outputs commands to a pretrained WBC (HOMIE) (Sec. 2.1); student distilled from
  RGB. No retry behaviour or failure injection described; episodes start from 200 teleop demo states.
- **Adaptive Skill Coordination (ASC, 2304.00410), Spot.** Skill library (navigation, pick, place) + learned
  coordination policy + corrective policy. Real world 59/60 (98 %) vs 44/60 (73 %) for sequential skills
  (abstract). On a failed grasp "ASC can re-activate the pick policy" (Sec. VIII-D); robust to the target being
  moved because it does not take object position as input (Sec. VI-C).
- **Skill Transformer (2308.09873).** One transformer that predicts a skill id and low-level actions; "capable of
  detecting and retrying failed skills" while "neither modular baseline re-executes skills" (results). Under the
  drawer-closing perturbation, pick success ST 26.75 -> 19.50 vs modular oracle 24.00 -> 0.50 (Table 2). It is a
  monolithic network with explicit skill structure, trained by imitation in Habitat, not on a real legged robot.
- **QT-Opt (1806.10293).** End-to-end closed-loop grasping; learns "regrasping strategies" and "responds
  dynamically to disturbances" (abstract); "can open and close the gripper at any time" (Regrasping paragraph, p. 7).
- **DribbleBot (2304.01159).** Single dribbling MLP plus a separate recovery policy, switched by a finite state
  machine on roll/pitch > 1.0 rad, back below 0.5 rad (Sec. III-B5).
- **UMI on Legs (2407.10353).** Diffusion policy at 1-5 Hz over a 50 Hz RL WBC (Sec. 3); no retry: after the
  kettlebell topples "we halt the system".
- **Whole-body dynamic grasping (2508.08328).** Low-level policy + high-level policy (Sec. 4.1); a failed grasp
  simply resets the scene (App. A.2).

Inference for MicroDuck: keep the walking policy and the crouch as separate learned skills under a high level that
re-reads the camera and the jaw every step. The high level can be scripted first (ASC's ablation shows plain
sequencing loses about 25 points, mostly from hand-off errors), and a learned coordinator is the published upgrade.
A single 61-D actor that must also learn search and stepping would be an unpublished design.

## Q2. Closed-loop vision during approach and descent

- **Re-estimate every frame.** DoggyBot tracks the target at 10 Hz with Cutie between 0.2 Hz detections for
  "continuous updating of the object's position" (system section; number not recorded); the no-tracking variant fails from drift. VBC's high
  level reads 4 stacked segmented depth frames each step (App. C.2) and is "capable of recovering from short-term
  tracking loss" (Sec. 4.2). DribbleBot detects at 30 Hz with a 15-step observation history (Sec. III-A3, III-B1).
- **Teacher-student.** VBC teacher sees object pose plus a PointNet++ shape latent; student from depth by DAgger
  (Sec. 3.2). 2508.08328 teacher sees ground-truth pose and velocity, student sees masks + depth "delayed by four
  frames" (Sec. 5.1). DextrAH-G teacher sees noisy object pose; depth student also predicts object position as an
  auxiliary head (beta = 0.1) used by the deployment state machine (Sec. 3.3). VIRAL: privileged teacher, RGB
  student; history models (feed-forward history, LSTM) beat single-step (Sec. 4.5).
- **Delay and dropouts randomised.** DribbleBot: Poisson frame arrival, mean delay 20-60 ms, frame arrival rate
  0.3-0.7 (Sec. VIII, Table IV). VBC: camera latency randomised for the student (App. D). PhysHSI (2510.11072):
  "retaining the last observed object pose" when detection drops, and pose masked after grasp so the policy relies
  on proprioception (Sec. IV-A); offsets, Gaussian noise and delays added in training (Sec. IV-B).
- **Memory.** Short histories (4-15 steps) or LSTMs (DextrAH-G teacher MLP+LSTM, App. E.6; VIRAL LSTM ablation).

Inference: the aim-crouch env's "measured once at reset" input is the DoggyBot no-tracking configuration. The
published fix is a per-step object input in the stance frame, trained with a hold-last-value dropout model
(MicroDuck: 10 fps, 0.1-0.3 s delay, multi-second stalls), a few steps of history or an age-of-measurement scalar,
and masking once the ball is under the beak (PhysHSI). A teacher with true ball position then distilled to the
noisy input is the VBC/DextrAH-G route; with only 3 numbers of object input, training the actor directly on the
noisy, delayed value (asymmetric critic with the true value) may be enough. That last point is inference.

## Q3. Training recovery: start states and failure injection

- Mid-task starts: DeepMimic RSI, "before mastering the earlier phases, little progress" on later ones (Sec. 6.1);
  removing RSI cut backflip return 0.791 -> 0.730 (ET only), walk barely changed (Table 5). Reverse curriculum
  (1707.05300): start near the goal, expand start states as performance grows (abstract). PhysHSI hybrid RSI plus a
  subset from the default pose (Sec. III-B3). VIRAL starts from demo states (Sec. 2.1).
- Failure injection: VBC 10 % object relocation (App. C.2). DribbleBot ball teleport within 1.0 m every 7 s,
  velocity kicks 0-0.3 m/s, drag 0-1.5 (Table IV). DextrAH-G random wrenches, p = 0.1 per timestep (App. E.3).
  2508.08328: moving targets up to about 30 cm/s (Sec. 3.3).
- Rolling objects: DribbleBot randomises ball mass and drag, not rolling friction (Table IV). No paper found trains
  pick-up of a lumpy, rolling, 1-1.5 cm object.

Inference: episodes should start from a mix of standing far, standing near with offset, mid-crouch, and
just-after-a-slip states, and the ball should be nudged or relocated with a small probability during approach and
descent, so that "step again" and "look again" receive reward.

## Q4. Grasp-failure detection

- QT-Opt: binary gripper open/closed status plus gripper height in the state; image-only 53 %, + status 58 %,
  + height 70 % (Table 2). Scripted termination uses gripper closed + height (App., Algorithm 2). Real success label
  by image subtraction after dropping (Appendix D, Fig. 12).
- Skill Transformer: "A binary indicator if the robot is holding an object" in proprioception (Sec. 4.1).
- ASC: failure seen as the grasp API's error return (Sec. VIII-D).
- DextrAH-G: drop not observed explicitly; predicted object position drives the state machine (Sec. 3.3).
- VBC: no onboard grasp verification described.
- Motor current as a learned slip observation: not found in RL papers (search 2026-10-26); found only as a safety
  limit or in supervised/tactile slip detectors.

Inference: a scripted jaw monitor (XL330 present current and present position against an empty-close position)
that outputs one "holding" bit, debounced over a few ticks, matches the published pattern; the bit then gates the
high level (lift vs re-search) and can be an actor input in training, simulated by the jaw-closed angle in sim.

## Q5. Budgets and small-object notes

- VBC high level: teacher 10240 envs about 36 h, student 240 envs about 48 h, one RTX 4090/3090 (App. C.2).
- DoggyBot: oracle 6144 robots, 20k iterations, about 10 h on one 4090; distilled 384 robots, 5k iterations, 6 h.
- UMI on Legs WBC: 4000 iterations (Sec. 10.2).
- 2508.08328: teacher 6000 envs, rollout 24, one RTX 4090 (Sec. 5.3).
- DribbleBot: 4096 envs, one RTX 3090; total timesteps stated as 77 B (text figures garbled).
- DextrAH-G teacher: 8192 envs/GPU on 4 V100, 4.7 B frames, about 68 h (App. E.11).
- ASC: pick/place skills 18 M steps, 22 h on 1 GPU; coordinator RL 150 M steps, about 45 h (Sec. VIII-C).
- VIRAL: 1-2 GPUs plateau "far below the desired performance range" (Sec. 4.7); runs used 16-64 GPUs.
- Small objects: VBC beak gripper "tends to push objects away" (App. I); DoggyBot ground ball 70 %.

Inference: a learned high-level coordinator over frozen MicroDuck skills is the cheaper training (ASC-scale
coordinator, VBC-scale teacher on one GPU); retraining walking and crouch jointly with search would be closer to
VIRAL's multi-GPU budget.
