# Sim-to-real on the CMU MicroDuck: the measured gap and what has been tried

The policies walk in the as-built simulation without falling. On the robot the same policies walk 3 to 4 s at
0.25 m/s and then wobble and fall more often than not. This page records what has been measured about that gap,
what was tried and failed, and the hypotheses that are still open, each with a test. Dates are the measurement dates;
the current robot condition is in [robot-condition.json](robot-condition.json). Fine-tune run details and the
literature behind each change are in [finetune/modal/FINETUNE-DELTA.md](finetune/modal/FINETUNE-DELTA.md).

## How sim and real are compared

- The CPU check `mac/brain_sim_check.py` runs the exact Mac-side control code (`mac/walk_brain.py`: mode logic,
  tick conversions, guards) against the as-built MuJoCo model with BAM's XL330 model at the real supply voltage,
  with the real fraction of dropped servo reads injected. `--tick-log` writes one row per 20 ms tick in the same
  shape as the robot's `walk_wired.py` log, so the two are compared on identical statistics
  (`real-logs/README.md` has the row format and the commands).
- `simcheck/sim_suite.py` is the acceptance tool: eight cases, of which the "real" ones take supply, IMU rate, delay
  and drop fraction from `robot-condition.json`; the "stress" ones (6.3 V, 8 % drops with a 35 Hz IMU hold) are
  harsher than the robot and are reported, never used to accept or reject.
- Real reference runs: four 2026-10-18 logs in `real-logs/robot-runs-20261018/`. walk61 at 0.25 m/s stepped
  cleanly through its 3 to 4 s in all three walks; afterwards one walked into a laptop at the stop and one tripped
  the action guard 8 s after the command while standing. ft3 at 0.20 m/s lost balance in three of three walks.
  Earlier logs (24 walk61 walks on 2026-10-16, 6 ft3 walks on 2026-10-17) were analysed but not archived; their
  statistics are quoted below.

## What is measured (2026-10-15 to 2026-10-18)

- The wobble is real motion, not sensor noise. At a quiet stand the gyro tick-to-tick change has a standard deviation
  of 0.01 to 0.05 rad/s; while stepping it is 1.0 to 1.3 rad/s on the robot against 0.17 to 0.30 in the sim.
- The extra shake does not come through the joints. Leg joint velocity jerk on the robot is 1.5 to 2 times the sim
  (tick-change std 1.0 to 1.7 against 0.5 to 1.2 rad/s) and the joint velocity spectra match, but the trunk gyro
  jitters 4 to 6 times the sim and the gravity vector 2 to 4 times, with broadband power at 5 to 25 Hz (72 % of roll
  power above 5 Hz; the sim sits under 5 Hz). Candidates that fit this: foot impacts and slip, play between horn and
  link, IMU mount vibration.
- The servos move less than commanded. Real goal-to-position lag is 4 to 5 ticks (sim 7); the real amplitude ratio
  of position to goal is 0.17 to 0.45 against 0.24 to 0.75 in the sim. The policy answers with 2 to 4 times larger
  actions (max 2 to 4 rad against 0.9 to 1.1), and real walks show action jumps over 0.5 rad on 35 to 46 % of ticks
  against 1.5 to 2.7 % in the sim.
- Deadband under load. Open-loop step tests on the standing robot at gain 400: ankles realised 0 to 15 % of 0.03 to
  0.05 rad steps, knees 0 %, hips 8 to 19 %; at gain 1200 every joint tracks 92 to 107 % with a 60 ms rise, so the
  servo is fast and the missing part is output authority at low gain. Lifted, the same joints track at gain 400
  (load-induced stiction), except the right ankle, which binds on its own (a replacement servo on that joint was as
  bad; the foot bracket is suspected). In walk logs at gain 200 the leg joints realised 6 to 19 % of the policy's
  0.02 to 0.04 rad corrections and 30 to 50 % of the stride amplitude, 80 to 100 ms late.
- Falls begin with a yaw-rate spike of 6 to 10 rad/s; the trained pushes are linear only.
- The IMU is not the problem. End-to-end latency is about 15 ms plus half a tick; a raw-gyro stream coincides with
  the calibrated gyro to 0 ms. The 40 to 60 ms trunk-versus-encoder lag seen in stance tests is plant softness (the
  rigid sim shows 18 to 22 ms for the same chain). Full-rate 50 Hz IMU since the A7Z head board (2026-10-18).
- Dropped servo reads (about 4 % of stepping ticks) do not raise the sim's jitter by themselves.
- Supply: the bench rail is 7.0 V while walking; in the sim 7.0 against 7.4 V changes distance by a few percent and
  no outcome.

## What the sim reproduces and what it does not

- Reproduces: the static sag at gain 400 (0.041 rad sim against 0.05 to 0.067 real), the stride itself, the
  stand-to-walk sequence, the ground-pick crouch and rise, the head command response, the effect of supply voltage
  and centre-of-mass shifts.
- Does not reproduce: the stiction band at gain 200 (BAM's XL330 fit, from Rhoban's bench at 7.5 V, underestimates
  friction under load), the trunk's broadband 5 to 25 Hz shake, the yaw-rate spikes, and therefore the falls after
  4 s.
- The friction scale in BAM is not the knob: scaling it by 1.5 makes walk61 stop stepping at 0.25 m/s in the sim and
  2.0 falls at 2.3 s, while the real robot steps at 0.25 and 0.20 despite its lower amplitude ratio. A faithful
  actuator fit needs the measured stiction fractions above, not a scalar.

## What was tried and failed

- Runtime action smoothing (a 0.5 EMA on the goals, 2026-10-16): removed; the literature says smoothing must be
  trained in, and the filtered loop was not better.
- Higher gain for the deployed policy: in the sim walk61 falls at 1.98 to 2.18 s at gains 800 and 1200 (actions
  saturate during the stand policy). The policy needs the compliance it was trained with.
- Integral gain on the ankles: no reach gained at I 3000, a limit cycle at I 10000 and above, and the XL330 keeps its
  integrator state until a reboot. Ruled out.
- Adaptive breakaway push (feed-forward goal offset while a joint is stuck, `walk_brain.Breakaway`): brings every leg
  joint to 83 to 114 % tracking when standing, but switching to the stand policy at gain 200 with the push on diverged
  in 0.4 s and fell. The sim did not predict this (no stiction there, so the push rarely engages). It is now applied
  in stand mode only.
- Fine-tune ft1 (yaw pushes, wider friction and supply randomisation, higher observation noise, doubled action-rate
  penalty): the policy learned to stand still; a reward audit showed standing already collects 80 % of walking's
  reward in Pollen's env. ft2 (half-way randomisation, learning-rate clamp, tracking guard): drifted to standing
  within 25 to 50 iterations. ft3 (`lowcmd`: velocity tracking on a 0.5 s EMA of the base velocity with a tighter
  Gaussian): no drift, steps at 0.20 m/s, 17 to 69 % more distance in every real-condition sim case, but on the
  robot it lost balance at 2.5 to 4 s in three of three walks at 0.20 m/s and showed the same jitter as walk61.
  Sim improvements that do not address the measured failure mode do not transfer.

## Differences from Pollen's own runtime (verified at the source, 2026-10-17)

Pollen's robot daemon scales actions by 0.9 while walking and by 7.4 V over the measured battery voltage, applies a
first-order low-pass on targets (legs 0.7, head 0.5) that its config says the alpha policies were trained with, coasts
on the last good sample for up to 3 dropped ticks, and stands at gain 160. The public training repository (this one)
has no low-pass term (tried and reverted in July), and the official web simulator runs the same policies unfiltered.
Our runtime applies none of these. Which of them matter for a robot with our stiction is untested; each is a cheap
experiment (below).

## Open hypotheses, each with a test

1. Foot impact and slip excite the trunk at 5 to 25 Hz. Test: record the same walk with the robot held in the air
   (feet stepping without contact) and on the floor; compare the gyro spectra. Air-only shake points at the links and
   mount, floor-only at the feet and sole.
2. Horn and link play under alternating load. Test: static torque-reversal on each leg joint with the encoder and an
   external angle reference; measure the lost motion. Model it as backlash (Pollen's `-Backlash` task variants add
   plus or minus 1 degree of play) if it is of that order.
3. Stiction at gain 200 is the dominant plant difference. Test: fit BAM's friction terms to the measured step
   responses (the gain sweep fractions above), rerun the suite; the fitted model must reproduce 6 to 19 % tracking of
   small steps and still walk. Then train or fine-tune against it. This is the only honest basis for a retrain.
4. Pollen's target low-pass and 0.9 scale are part of the policy's trained plant. Test: run walk61 with those two
   runtime changes on the robot for the usual 0.25 m/s walks and compare jitter and fall rate with the 2026-10-18
   rows; also run them in the sim suite.
5. Coasting instead of holding on dropped reads. Test: implement Pollen's 3-tick coast in `walk_brain.py`, replay in
   the sim (drop 0.04 and 0.08), then on the robot.
6. Mechanical: free the right ankle (foot bracket alignment, idler, horn screws) and recheck the gain-400 step test;
   the left knee is the next stiffest joint.

Rules for the next study, learned the hard way: name the measured failure mode and its robot metric before any
training run; audit the reward for degenerate policies (standing) before launch; run the sim suite at the real
condition, not at stress settings; and check the published recipe for the same failure mode before queuing compute.
