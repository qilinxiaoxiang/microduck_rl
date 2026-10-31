# walk61 fine-tune: what changes in the simulator and why

Location note (2026-10-19): this pipeline now lives in the `qilinxiaoxiang/microduck_rl` fork under `cmu/finetune/`,
with its assets in `cmu/whole/`, and it fine-tunes from the deployed walk61 ONNX (`cmu/whole/runtime-overlay/models/walk61.onnx`).
The as-built variant of a full training run is the registered task `Mjlab-Velocity-Flat-MicroDuck-CMU-Plugged` /
`-Battery` (being added by another agent); the worker's `--cmu-config plugged|battery` selects the same mass manifests
(`cmu/model/manifests/`) for a fine-tune.

Date: 2026-10-16. Run config: `cmu/finetune/modal/finetune.py` + `cmu/finetune/modal/finetune_worker.py` (this directory). Problem advanced: the
real MicroDuck walks 4 s reliably but wobbles and falls more at 6 s; replaying the real logs showed the gap is not
sensor smoothing but the closed loop on the real actuators and body (main session, 2026-10-16).

## Source policy

`walk61.onnx` (cmu/whole/runtime-overlay/models, SHA256 e36332d3…) is byte-identical to `BEST_alpha_walking.onnx` in
pollen-robotics/microduck-simulator (commit 836975a, 2026-08-25). No `.pt` for it is public (HF mirrors carry only
the ONNX; Pollen's wandb project is private). `velocity.pt` in Microduck-build-tutorial is a different policy
(63 inputs, 18 outputs, weights differ by up to 1.5) and is not used. The actor is rebuilt from the ONNX
initializers (normalizer mean/std + 4 ELU layers): on 32,242 observations rebuilt from the 2026-10-16 walk logs the
torch rebuild matches onnxruntime to 1.2e-6, and `model_0.onnx` exported from the rebuilt runner is checked against
`walk61.onnx` at the start of every run (assert < 1e-4). The critic, optimizer and action std do not exist in the
ONNX; the critic is trained from scratch for the first `--warmup-iters` (default 100) iterations with the actor
frozen and a fixed learning rate (PIRL-style critic-first warm start; Fan et al. 2025 report warm-starting only one
network is unstable, so the actor is not updated until the critic has converged on its returns), then PPO continues
with Pollen's hyperparameters (lr 1e-3 adaptive, desired_kl 0.01, entropy 0.01). Action std starts at 0.25
(`--init-std`); the converged value of the original run is unknown, 1.0 would make the rollouts fall at once.
All iteration-indexed curricula are collapsed to their final stage (`common_step_counter` = 2000 x 24 + 24), as the
microduck_rl AGENTS.md warm-start rule requires, so the loaded policy trains under the conditions it was trained under.

## Robot model

As-built mass manifest `cmu/whole/dynamics/asbuilt-20261015/asbuilt-mass-manifest.json` (V1.8 prints, V1.9 rear pack, D-007
bench backpack on the 7.4 V rail, Pi Zero W head, audio) is applied to the official MJCF bodies as explicit
inertials, with the NP-F970 battery (90 g) removed, exactly as `cmu/whole/dynamics/evaluate.py` does for the as-built sim
check. Build target confirmed with Shawn on 2026-10-16 ("目标机型 A"): today's robot (Pi Zero W, bench backpack), not the A7Z + battery backpack build.
KP_fw stays 200, the 61-input observation layout and `default_joint_pos` are unchanged, so walk_brain.py loads
the result without changes.

## Delta (measured gap -> change -> published basis)

1. **Yaw-rate pushes.** Every real fall begins with a 6-10 rad/s yaw-rate spike; the trained pushes are linear
   x/y +-0.3 m/s only. Change: `push_robot.velocity_range["yaw"] = (-1.0, 1.0)` rad/s on the same 3-6 s interval.
   Basis: random base-velocity pushes are the standard robustness term (legged_gym `push_robot`, ABS Table II,
   RMA); mjlab's `push_by_setting_velocity` already takes a `yaw` key. +-1.0 is a kick of the same order as the
   linear one (0.3 m/s over a 0.1 m stance), not the fall signature itself.
2. **Joint-friction DR widened.** The left hip_yaw tracks 0.05-0.08 rad short unloaded, loaded joints ignore
   steps under about 0.05 rad at KP 200, while the sim randomizes BAM's friction budget only by (0.9, 1.1).
   Change: `randomize_joint_friction.scale_range = (0.8, 2.0)`. Basis: the BAM friction budget (Coulomb +
   Stribeck + load) is the sim's stiction term; walk-these-ways and RMA randomize motor strength/friction by
   factors of about 2; the deadband itself is left to the mechanical fix (no actuator refit in this run).
3. **Supply sag widened.** The servo rail sags to 4.9-6.3 V in stop struggles; trained `vin_range` (6.5, 8.2),
   drop gain (0, 0.2) V/Nm, floor 6.0 V. Change: `vin_range = (6.0, 8.2)`, `vin_drop_gain_range = (0.0, 0.4)`,
   `vin_min = 5.0`. Basis: BAM's voltage control law makes torque fall with vin; the measured floor must be inside
   the range (same argument as the friction note for mu, Walking-on-Ice baseline at the range edge).
4. **Observation noise raised on the fast channels.** Real body gyro tick-to-tick change std 1.0-1.6 rad/s vs
   0.08-0.18 in the as-built sim; real action jumps > 0.5 rad 66-137 per 4 s walk vs 2-3 in sim. Change:
   `base_ang_vel` noise +-0.03 -> +-0.15 rad/s, `joint_vel` noise +-0.25 -> +-0.75 rad/s. Basis: Humanoid-Gym and
   DreamWaQ train with gyro noise of this order (DreamWaQ Table II: ang vel noise 0.2); the policy should stop
   acting on single-tick gyro content it cannot trust on the real IMU/servo bus.
5. **Action-rate penalty doubled.** Same action-jitter gap. Change: `action_rate_l2` final weight -1.0 -> -2.0
   (curriculum last stage and the live weight). Basis: ABS/DreamWaQ regularize |a_t - a_{t-1}|^2; the literature
   note (section 4) says smoothing must be trained in, never added at deployment, and the runtime stays
   unfiltered.

Kept: foot friction (0.7, 1.3) (rubber-band soles are grippy; the cork slip note is a separate problem), command
delay 3-6 ticks (measured 80 ms goal-to-measured lag is inside it), KP_fw 200, mass DR +-5 %, IMU misalignment 6 deg,
encoder bias, all rewards except action_rate.

## Budget and acceptance

Smoke (64 envs) and a 40-iteration bench at 4096 envs measure it/s on the chosen GPU before the detached run;
the caps (`--max-hours`, `--max-dollars`) are enforced by the worker's save hook every 25 iterations and
by the Modal function timeout, and a reward guard stops the run when the mean episode reward falls under half its
post-warm-up peak on two consecutive saves (`model_best.pt` is kept). Acceptance: the exported ONNX runs in the
as-built MuJoCo check (`cmu/mac/brain_sim_check.py --model asbuilt`) against walk61 on three seeds:
distance, falls, max tilt and action-jump count; then the real walk in Shawn's main session after 扶好了.

## Run ft1 (2026-10-16): delta "default" produced a standing policy

Modal app `ap-K5AexZpPzN7XaskCaq8mJy`, L4, 4096 envs, 999 iterations, 56 min, $1.05. Training reward rose 27.6 -> 98.8
and mean episode length 592 -> 943 steps, but the gain came from survival, upright, pose and head terms: the
command-manager metric `Metrics/twist/error_vel_xy` went from 0.28 (frozen walk61, iterations 0-100) to 0.46, and
`Episode_Reward/action_rate_l2` fell from -3.0 to -1.4. In the as-built MuJoCo check the exported policy does not
step at any command (10 s distance 0.003-0.009 m in all seven suite cases, max |action| 0.35 rad); checkpoints exported
with `cmu/finetune/modal/ckpt2onnx.py` still walk at iterations 100-200 (1.42-1.68 m) and stop from iteration 225 on. Neither
ck150 nor ck200 beats walk61 on the suite (more tilt, or shorter distance under dropped reads and 6.3 V).

Cause: item 5 doubled the action-rate tax and items 2-4 made stepping fall-prone (walk61 fell 1.45 per env-minute in
the delta env against 0.15 in the trained env); standing pays neither, so the PPO update after the warm-up moved
to standing. Pollen's own note in `microduck_velocity_env_cfg.py` (head droop fix, run 5yay13u4) records the same
failure when a tight head tolerance taxed stepping: "Standing still scored higher, so it stood still."

## Run ft2 plan: delta "staged"

- `action_rate` back to -1.0 (Pollen's final weight); smoothing is left to the measured-noise items.
- Stage the randomization half-way: joint friction (0.8, 1.5), yaw push +-0.6 rad/s, vin drop gain (0, 0.3), floor
  5.5 V, gyro noise +-0.10, joint-vel noise +-0.50. Staged/annealed domain randomization is the standard remedy
  when the policy cannot survive the target range at once (ABS and RMA widen ranges over training; legged_gym
  curricula).
- PPO learning rate clamped at 3e-4 (`--max-lr`) so the first updates after the warm-up stay close to walk61; the
  adaptive KL schedule still lowers it.
- Tracking guard (`--track-guard 1.3`): the worker averages `Metrics/twist/error_vel_xy` between saves, takes the
  warm-up (frozen walk61) value as the baseline, excludes saves above 1.3x from `model_best.pt`, and stops after two
  such saves. On ft1's curve this would have stopped the run at iteration 110-125 (0.43-0.44 against 0.28) for about
  $0.15 instead of $1.05.
- Acceptance unchanged: the as-built suite against walk61; the in-container eval now also prints the tracking error
  and the speed along the command for every variant.

## Reward review gate (added after ft1)

`modal run cmu/finetune/modal/finetune.py::audit --variants <env variants>` rolls out walk61 and stand61 deterministically in each env
(2048 envs, 15 s) and writes `cmu/finetune/modal/reward-audit.json`: reward per simulated second by term for both policies, the
stand/walk ratio and a verdict. `finetune.py::run` refuses to spawn without a PASS for the run's variant (`none` ->
`asbuilt`, `default` -> `delta_asbuilt`, `staged` -> `staged_asbuilt`); unregistered deltas cannot be launched.
First audit (as-built masses, Pollen's reward, 2026-10-16 18:40 EDT):

    term                    walk61   stand61   stand-walk
    air_time                 1.036     0.025     -1.011   <- the only term that pays stepping
    head_pose_bias          -0.226    -0.633     -0.408
    head_pose_tracking       1.841     1.470     -0.371
    track_linear_velocity    1.329     1.324     -0.005   <- pays standing as much as walking (std^2 0.1, walk61 at ~0.11 m/s)
    upright                  1.852     1.848     -0.004
    action_rate_l2          -0.109    -0.062     +0.047
    pose                     0.705     0.769     +0.065
    track_angular_velocity   1.083     1.305     +0.222
    total                    7.477     6.008     ratio 0.804

Standing already collects 80 % of walking's reward in the env that produced walk61; ft1's delta only had to tax
stepping by about 1 reward/s to flip the optimum. The threshold in `reward-audit.json` is calibrated against the
known-good (as-built, walk61 trained) and known-bad (delta_asbuilt, ft1 stood) variants, see the audit file.

Calibration (audit-1016-1940, 2048 envs x 15 s, as-built masses in all three):

    variant          stand/walk   walk61 falls per env-min   margin walk-stand   verdict
    asbuilt            0.804            0.158                    1.469/s          PASS   (env that produced walk61)
    delta_asbuilt      0.849            1.430                    1.036/s          FAIL   (env that produced ft1, stood still)
    staged_asbuilt     0.813            0.338                    1.371/s          PASS   (ft2)

Gate in `finetune.py`: `AUDIT_MAX_RATIO = 0.83` and `AUDIT_MAX_WALK_FALLS = 0.5` (per env-minute). The fall-rate limit carries
most of the discrimination; the ratio limit lies between the two measured values and is thin, so it is re-calibrated on
the next run with a known outcome. The audit is a necessary check, not a guarantee: in every variant the standing policy
collects over 80 % of the walking reward, so the tracking guard and the learning-rate clamp stay on during training.

## Run ft2 (2026-10-16): staged delta, stopped at iteration 150

Resumed under the per-step guard; stopped by hand when the parent session's re-check showed the iteration-125 policy no
longer steps at 0.25 m/s. Local as-built checks (7.4 V, 10 s): walk61 1.15 m at 0.25 and 1.57 m at 0.30; ft2@100
(frozen) 1.17 / 1.57; ft2@125 0.003 / 1.74; ft2@150 0.002 / 0.002. With Pollen's reward the actor drifts toward
standing within 25-50 iterations even with the action-rate weight at -1.0, half-way randomization and the learning
rate clamped at 3e-4: the audit margin (air_time about 1 reward/s, velocity tracking indifferent between standing and
walk61's 0.1 m/s) is too thin for PPO to keep stepping at low commands. The guard now also watches the speed along
0.15-0.3 m/s commands with an absolute floor (`--low-speed-floor 0.03`), and the sim suite carries a 0.25 m/s case.
Any further attempt needs a reward change that pays stepping at low commands (a velocity term that distinguishes
0.1 m/s from standing, or fewer standing envs during fine-tuning) and its own literature pass before any launch.
Cost: ft2 $0.33 (7.6 min to it 125, then 10 min of the resume).

## Run ft3 plan (2026-10-16): delta "lowcmd", the velocity term is filtered and tightened

### Why the reward paid standing (measured)

Pollen's `track_linear_velocity` is `exp(-(|cmd_xy - v_xy|^2 + v_z^2) / 0.1)` on the INSTANTANEOUS base velocity. In the
as-built CPU model (`brain_sim_check.py --model asbuilt`, 7.4 V, scratch `veltrace/trace_vel.py`) walk61 under a
0.25 m/s command moves at 0.123 m/s forward with a stride oscillation of 0.17 m/s (lateral) and 0.08 m/s (vertical)
standard deviation; under 0.15 and 0.20 m/s commands it does not step at all (distance 0.0 in the suite). Scoring the
recorded traces with the term (weight 2, reward per second, "s" = a robot standing under the same command):

    std^2   filter      walk61 @0.25 cmd     walk61 @0.30 cmd
    0.1     none        w 1.23  s 1.07  +0.16   w 1.18  s 0.81  +0.36    <- Pollen's term: nearly indifferent
    0.1     EMA 0.5 s   w 1.65  s 1.07  +0.58   w 1.59  s 0.81  +0.77
    0.05    EMA 0.5 s   w 1.37  s 0.57  +0.79   w 1.27  s 0.33  +0.93    <- chosen
    0.03    EMA 0.5 s   w 1.07  s 0.25  +0.82   w 0.94  s 0.10  +0.84
    0.02    EMA 0.5 s   w 0.78  s 0.09  +0.70   w 0.66  s 0.02  +0.64
    (EMA 0.25 s and 1.0 s give the same margins within 0.05/s)

The oscillation, not the slow forward speed, is what the Gaussian taxes: filtering alone lifts the margin at 0.25 m/s
from 0.16/s to 0.58/s. It is the same structure as the head-tracking tax Pollen documented in the env cfg (run
5yay13u4, fixed by `head_pose_bias`: price the escapable DC error on an EMA, let the oscillation cancel). The margin
over standing peaks at std^2 0.03-0.05 for walk61's current speeds; 0.05 keeps the walking policy at 1.37 of 2.0 so
the gradient toward tracking the command stays visible (AGENTS.md: "pick stds wide enough that the CURRENT policy
scores visibly").

### Literature pass (sources read 2026-10-16; the research-lookup API keys are not set on this machine, so the pass used
web search and the primary files)

1. legged_gym, `legged_robot_config.py` (Rudin et al. 2022): `tracking_sigma = 0.25`, reward `exp(-error^2/sigma)`,
   commands +-1.0 m/s. Standing at the top of the range pays exp(-4) = 0.02. Pollen: std^2 0.1 at +-0.4 m/s, standing
   at 0.4 pays exp(-1.6) = 0.20 and at 0.25 pays 0.54, i.e. cmd_max^2/std^2 = 1.6 against legged_gym's 4.0.
   `feet_air_time` 1.0 gated on |cmd| > 0.2 is the stepping term there too.
2. Open Duck Mini v2 (`Open_Duck_Playground/playground/open_duck_mini_v2/joystick.py`, the duck family's previous
   MJX recipe): `tracking_sigma = 0.01` ("was working at 0.01") at lin_vel_x +-0.15 m/s (cmd_max^2/std^2 = 2.25),
   plus an imitation reward (reference gait, weight 1.0), alive 20.0, `cost_stand_still` gated on |cmd| < 0.01 and a
   10 % all-zero command draw. Stepping there is held by the imitation term; the Gaussian is 10x tighter than mjlab's.
3. Humanoid-Gym (`humanoid_config.py`, Gu et al. 2024): `tracking_sigma = 5` (exp(-5 err^2), std^2 0.2) at
   lin_vel_x [-0.3, 0.6], but stepping is enforced by gait-phase terms (feet_contact_number 1.2, feet_clearance 1.0,
   joint_pos 1.6 against a reference gait with cycle_time 0.64 s), not by the velocity term.
4. Siekmann et al. 2021 (ICRA, "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition",
   Cassie): standing and walking are separated by the clock's swing ratio (omega = sigmoid(50 (r_swing - 0.15)));
   the velocity term 1 - exp(-2 omega |x_des - x|) is switched off while standing; instantaneous pelvis velocity.
5. van Marum et al. 2024 ("Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking", Digit):
   tracking exp(-5 (v - c)^2) per axis, standing is the zero command; the paper states that most reward terms are
   greater when a policy stands still with both feet on the ground, and holds stepping with an air-time term that
   charges 0.4 per touchdown (constant when standing) plus a single-foot-contact term; no clock.
6. Pollen's own note in `microduck_velocity_env_cfg.py` (head droop fix) and the microduck_rl AGENTS.md reward rules:
   an instantaneous tight tolerance on a quantity that must oscillate while stepping is an unescapable tax on
   walking; price only the escapable (DC) part on an EMA.

Recipe in the published work: either stepping is paid by a term independent of velocity tracking (gait clock,
imitation, air time with a touchdown charge) or the velocity Gaussian is 2.5-10x tighter relative to the command
range than Pollen's. A clock input would change the 61-input layout and is excluded; `air_time` stays at 3.0. No
source was found that filters the base velocity inside the tracking term (one search snippet showed
exp(-|v_bar - v_d|^2) with an averaged velocity but without a verifiable attribution); the filter is justified by the
measured oscillation above and by Pollen's own EMA fix for the head.

### The delta

`DELTA_LOWCMD = DELTA_STAGED + track_lin_vel = {tau_s: 0.5, std: sqrt(0.05)}` (`finetune_worker.py`): the staged
randomization of ft2 (audit PASS 0.813 / 0.338 falls) and `track_linear_velocity` replaced by
`track_linear_velocity_ema` (same weight 2.0, same Gaussian on an EMA of `root_link_lin_vel_b` with time constant
0.5 s, reset at episode start). Everything else (61-input layout, KP 200, default_joint_pos, air_time 3.0,
rel_standing_envs 0.25, action_rate -1.0) is unchanged. Run settings as ft2: warm-up 100, `--max-lr 3e-4`,
`--track-guard 1.3`, low-speed floor 0.03 m/s. Acceptance: 8-case as-built suite (incl. vx0.25 and vx0.2) against
walk61; a checkpoint is handed over only if it steps at 0.25 m/s and matches or beats walk61.

### Audit and smoke (audit-1016-2046, smoke-1016-2046; 2048 envs x 15 s, as-built masses)

    variant          stand/walk   walk61 falls/env-min   margin   track_linear_velocity walk61 / stand61   low-cmd speed walk61 / stand61
    staged_asbuilt     0.813            0.338              1.371/s        1.320 / 1.334                           (not recorded)
    lowcmd_asbuilt     0.798            0.361              1.457/s        1.202 / 1.112                           0.073 / 0.004 m/s   PASS

The gate passes (ratio under 0.83 and under the worker's 0.8, falls under 0.5). The aggregate margin over stand61 grows
by only 0.09/s, not the 0.6-0.8/s the single-command traces show, because the audit averages over the whole command
distribution: 25 % standing envs and the turn-in-place envs pay both policies the full term, and walk61 tracks
lateral and diagonal commands poorly (0.10 m/s along commands above 0.15 m/s), so the tightened Gaussian lowers its
own score too. The per-command margin at 0.25-0.30 m/s (the band where ft2 stopped stepping first) is where the
change acts; the per-step tracking guard with the 0.03 m/s low-speed floor stays on and bounds a drift at about
$0.3. walk61's low-command speed of 0.073 m/s confirms that it already stands at part of the 0.15-0.3 band.
Smoke (64 envs, 6 + 3 iterations, resume with `--rebaseline-saves 1`): reward term runs, model_0.onnx == walk61,
export and resume OK, $0.07. Audit $0.11. No live Modal app afterwards.

## Run ft3 (2026-10-16/17): delta lowcmd, 999 iterations, no guard stop

`modal run --detach cmu/finetune/modal/finetune.py::run --run-name ft3 --delta lowcmd --max-iters 1000 --max-hours 2 --max-dollars 3
--max-lr 0.0003 --track-guard 1.3` (app ap-xa8A5RNwW2qTAk1FmSdvRJ, call fc-01M53Y09M8JK9AWN8C70Z2PG8V, L4, 4096 envs,
0.31-0.32 it/s). Finished by max_iters after 55.7 min, $1.04 (cap 2 h / $3 approved by Shawn in the side session).
Per-step task metrics from progress.csv (warm-up baseline = frozen walk61 under the lowcmd env):

    iter   track_err   speed_along_cmd   low_cmd_speed (0.15-0.3 m/s)   mean_reward   ep_len   action_std
    100     0.280        0.099             0.080                          68.7         785      0.250  (baseline)
    200     0.270        0.116             0.085                          95.1         920      0.212
    400     0.263        0.130             0.093                         103.5         943      0.198
    600     0.265        0.130             0.092                         106.1         930      0.196
    800     0.264        0.135             0.105                         105.2         938      0.201
    999     0.258        0.144             0.111                         107.7         962      0.203

Unlike ft1 (speed fell to 0 by it 225) and ft2 (0.25 m/s abandoned by it 125), every task metric moved the right
way for the whole run: speed along the command +46 %, low-command speed +39 %, tracking error -8 %; the guard never
fired and model_best is the last best-reward save with tracking OK. The as-built suite result follows below.

### ft3 as-built suite (full table in `cmu/whole/runtime-overlay/models-ft3-it999/README.md` and `suite_asbuilt_vs_walk61.jsonl`; the conclusion below was superseded by the corrections at the end of this file)

    case             walk61                      ft3 it750                     ft3 it999
    nominal          1.57 m, tilt 5.9, jumps 1   2.37 m, 8.7, 10               2.17 m, 6.9, 4
    vx0.25           1.15 m                      1.99 m                        1.89 m
    vx0.2            0.00 m (no stepping)        1.51 m                        1.56 m
    volt6.3          1.43 m                      -                             2.03 m
    imu35_only       1.55 m, tilt 7.2, jumps 22, yaw 1.5    2.20, 11.5, 83, yaw 8.8    2.07, 7.4, 65, yaw -16
    drop8_imu35 x3   1.47-1.56 m, tilt 6.7-9.2, jumps 23-31, yaw within 5 deg
                                                 1.77-2.18 m, 11.6-13.5, 88-105, yaw 4-13   1.66-1.91 m, 8.8-13.6, 85-138, yaw -12/-43/-156
    it500 and it900: fall in two of the three drop8_imu35 seeds.

Conclusion: the reward change did what it was for (no standing drift, stepping at 0.20-0.25 m/s, every task metric
improved over 900 actor iterations, more distance in every case, no falls for it750/it999), but the fine-tuned
policies lost the jitter margin walk61 has under a 32-40 Hz IMU hold [superseded: a stress case; the robot runs the IMU at full 50 Hz, current condition in microduck-sim/CURRENT-STATE.json] (3-4x the action jumps, higher tilt,
heading veer or falls depending on checkpoint and seed). The run's acceptance rule ("matches or beats walk61") is not
met; [superseded: these cases are harsher than the robot, see the correction below]. The training
env has no observation hold/delay randomization; adding the IMU hold and read drops to the observation path is the
next delta, to be audited and run under its own cap. Cost this round: audit $0.11, smoke $0.07, ft3 $1.04 (run
records; verify on the usage page).

### Correction (2026-10-17): the real sensor condition is the nominal case; ft3 it999 is the robot-comparison candidate

Shawn: the IMU runs at the full 50 Hz on the robot (the dropout problem was fixed and the delay study found no delay).
The 10/16 walk logs also show 0 % stale servo reads while the legs move. The 35 Hz / 45 Hz hold and 8 % drop cases come
from the 10/15 state (Pi CPU saturated) and are stress cases, not the robot's condition; the earlier rejection of ft3
rested on them and is withdrawn, as is the proposed IMU-hold delta. At the real condition (nominal, vx0.25, vx0.2,
com+5, imu_pitch4, volt6.3) it999 never falls, walks 24-43 % further, steps at 0.20 m/s where walk61 cannot, with about
1 deg more peak tilt and 1-9 vs 1-2 action jumps per 10 s. Delivered as `cmu/whole/runtime-overlay/models-ft3-it999/`
(walk61.onnx = ft3.onnx, iteration 999; ft3_it750.onnx for reference); not yet run on the robot.

## Run plan "tether" (2026-10-30): walk61 with the charger lead, for walking and in-place turns

Problem: on the robot (bench supply, 3 m braided USB-C lead to a power bank on the floor) in-place turns stall after
40-90 deg and the duck steps in place (`cmu/mac/turn-inplace-20261030/README.md` in the robotics tree). The as-built
sim reproduces the stall only with a restoring yaw torque of about 0.010 N m/rad, a lead twisted by the turn; the hip
yaws then use 1-2 % of the XL330 torque. walk61 met no yaw disturbance in training (pushes are x/y velocity only), so
it learned to turn with almost no hip-yaw torque and does not push harder when a turn is blocked.

The delta: `DELTA_TETHER` = Pollen's recipe + ft3's EMA velocity tracking (`track_lin_vel` tau 0.5 s, std sqrt(0.05),
without which ft1/ft2 drifted to standing) + the step event `tether_lead`, which puts the lead's wrench
(`cmu/mac/tether.py`) on `trunk_base` every env step in 80 % of the envs, parameters drawn per episode: hanging weight,
floor drag on the length dragged so far (it grows with the distance walked), random snags (drag x2-4 for 1-4 s),
a taut pull toward the power bank capped by the bank sliding (0.5-1.5 N), and a twist torque of 0-0.02 N m/rad that
slips past 0.005-0.03 N m (Shawn: keep it moderate, no lead that never lets go). Exit point at the backpack's lower
rear corner, side drawn left or right. Nothing else changes: the actor still sees the same 61 inputs (no lead state,
no heading error); it can only notice the lead through its gyro, joint readings and the command it is not meeting.

### Audit (2026-10-30, `tether_audit.py`, CPU, 128 envs x 15 s, as-built masses, `reward-audit-tether.json`)

    variant          walk61 reward/s   stand61 reward/s   stand/walk   walk61 falls/env-min   walk61 turn in place (yaw rate along the command / |command|)
    lowcmd_asbuilt       7.521              5.722            0.761            0.188                 0.248 / 0.709 rad/s
    tether_asbuilt       7.459              5.876            0.788            0.156                 0.165 / 0.671 rad/s

Standing still collects 79 % of walking's reward with the lead, under the worker's 0.8 gate, so the delta does not pay
the policy to stand; the margin is thinner than without the lead (0.76), so the tracking guard stays on. The lead
takes a third off walk61's in-place turning in the training sim (0.25 to 0.17 rad/s), the effect the robot shows,
and leaves walking and falls about where they were. stand61 falls 1.8-1.9 times per env-minute under walking commands.

### Run settings and acceptance

Kaggle T4 (free, about 80 min per 1000 iterations), `--cmu-config plugged` (the robot runs on the lead, no battery),
4096 envs, 1000 iterations, warm-up 100, `--max-lr 3e-4`, `--track-guard 1.3`, `--max-hours 2`; two seeds as a paired
job. Acceptance against walk61 in the as-built eval sim (`cmu/simcheck/sim_suite.py`, real cases only) plus
`brain_sim_check.py --tether` seeds 0-4: in-place turns faster than walk61's 12-23 deg/s with the lead and no slower
than its 28 deg/s without; walking distance and falls no worse with or without the lead; action jumps and peak tilt
no worse than walk61 in the real cases (ft3 lost balance on the robot while beating walk61 in sim, so the jitter rows
decide). Started 2026-10-30 18:40 EDT on Shawn's go: smoke `tether-smoke`/`tether-smoke-b` (64 envs, 6 iterations, source digest 7c061ff19b546ae5) COMPLETE with the `tether_lead` event active and ONNX exported; run `tether-a` (seed 1) / `tether-b` (seed 2) as one paired Kaggle kernel `microduck-tether-a`.

### Result (2026-10-30): turns fixed, jitter rejects both seeds

Both jobs COMPLETE, 1000 iterations each in 83 and 82 min (0.20-0.21 it/s on one T4 each). Training-sim tracking at
iteration 999: speed along the command 0.17 m/s against walk61's 0.074, track error 0.246 against 0.293; the
training sim's own action_rate_l2 mass fell from -1.44 to -1.19 per episode, so the training sim saw no jitter.

Acceptance (`cmu/finetune/kaggle/tether_accept.sh`, summary `tether_accept_summary.py`; plugged manifest,
`--volt 7.0 --drop 0.04` from `cmu/robot-condition.json`, walk61 rerun in the same sim as the baseline, because
`reference-walk61-plugged.jsonl` predates the 2026-10-26 plugged model and walk61 now walks 0.9 m less in it):

- **In-place turns with the lead:** walk61 7-25 deg/s over seeds 0-4 and both directions; tether-a 23-29, tether-b 26-29
  (all four checkpoints 21-29). Without the lead all at 28-29 deg/s. Passes.
- **Walking 12 s at 0.25 m/s:** walk61 1.1-1.25 m with or without the lead; tether-b 2.6-3.2 m, tether-a 2.3-2.8 m.
  No falls in the 18 lead cases for any checkpoint. Passes.
- **Jitter, which decides:** action jumps (ticks moving more than 0.5 rad) in the five real suite cases: walk61 0-5;
  tether-a 19-58, tether-b 8-82, the best checkpoints 9-80; ft3, which lost balance on the robot, had 4-13. Peak tilt
  8-22 deg against walk61's 2-11. tether-b_best fell at 11.2 s in `nominal`. Over the 18 lead cases walk61 totals 55
  jumps, the candidates 261-394. **Fails on every checkpoint.**

Not for the robot. The training sim does not show the jitter the eval sim shows (the gap the jitter sim-ID measured:
BAM friction about 1.4 and gyro vibration proportional to leg speed). Run t3 (branch `cmu/grasp`, commits
1bb0560c..675add5a, its plan in that branch's FINETUNE-DELTA.md) puts that gap and an `action_acc_l2` penalty into
training; the lead belongs on top of t3 once t3 passes its jitter gates, not as another
run on the current recipe.
