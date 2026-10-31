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

## Run t1 plan (2026-10-30): delta "t1", the body follows the head (PLAN-continuous-vision.md section 5)

Problem advanced: final-goal steps 2 and 3 (walk to the item, bend with the beak open). The pet-style loop needs the
walking policy to carry the neck and head where the gaze loop (`mac/gaze.py`) points them and to keep stepping while it
does, at any speed from a creep to the robot-tested 0.25 m/s. Source policy: the deployed `walk61.onnx` (CURRENT-STATE
`walking_policy`), not the ft3 candidate (fell on the robot three of three times at 0.20 m/s).

### What walk61 does today (measured)

- Robot (PICKUP.md, stand61): the neck lifts to about -0.09 rad absolute whatever is commanded (-0.4 to -0.8 asked);
  walking bursts with the head commanded down fell twice.
- As-built CPU sim, plugged manifest (`cmu/mac/brain_sim_check.py --model asbuilt --manifest cmu/model/manifests/plugged.json
  --volt 7.0 --walk-at 3 --walk-s 6 --neck-pitch N`, new flag, 2026-10-30; absolute neck targets, finals from the model zero):

      vx     neck cmd   neck reached   head_pitch   path in 6 s   max tilt
      0.00    -0.1        +0.002         0.336        0.003 m       1.9 deg
      0.00    -0.4        -0.125         0.255        0.005 m       2.0
      0.00    -0.8        -0.262         0.162        0.007 m       3.0
      0.25    -0.1        -0.054         0.341        0.009 m       1.9    <- no stepping: a head command stops walk61
      0.25    -0.4        -0.286         0.294        0.007 m       2.2
      0.25    -0.8        -0.557         0.262        0.006 m       3.5

  The sim reproduces both robot findings (the neck reaches a third of the command standing; any head command stops the
  stepping), so this is the policy's behaviour, not the sim-to-real gap, and the sim can measure the fix. walk61 was
  trained with head commands resampled every 2-5 s at the full ranges, yet the audit shows it pays head_pose_bias only
  -0.23/s: with the 0.5 rad Gaussian and weight-3 L1 bias the cheapest answer to "neck down" was a partial neck and a
  dropped head_pitch, and standing collected most of the reward while the head was moving (the ft1 structure again).

### Delta t1 (`finetune_worker.py::DELTA_T1`, every other setting is walk61's training recipe)

- Twist command: `lin_vel_x` uniform 0..0.30 (forward only; the robot-tested 0.25 inside the range), `lin_vel_y`
  +-0.10, `ang_vel_z` +-1.0 as trained; 15 % of envs hold the exact zero command (the robot's idle), turn-in-place 15 %.
- Head command (slots 55-58, deltas from HOME): neck_pitch -1.10..+0.40, head_pitch -0.30..+1.00 (the gaze loop's
  0.35..1.35 absolute), head_yaw +-0.90, head_roll +-0.05; each sample held 3-8 s. All inside the caps walk61 was
  trained to (+-1.10/1.10/1.40/0.31), so the baked normalizer sees nothing new.
- Rewards: `track_linear_velocity` -> ft3's EMA form (std^2 0.05, tau 0.5 s, weight 2.0) so stepping at 0.15-0.30 m/s
  beats standing by 0.7-1.5/s before air_time; `head_pose_bias` weight 3 -> 5, tau 1.0 -> 0.5 s (the acceptance is the
  DC head error 0.5 s after a command); `head_pose_tracking` unchanged (std 0.5, weight 2; a tight instantaneous std
  killed walking in Pollen's run 5yay13u4); `air_time` 3.0, `action_rate_l2` final -1.0 unchanged.
- Actor input: the 61-number contract as deployed (48 proprioception: gravity, gyro, joint positions and velocities,
  last action, feet; then twist 3, head 4, body 6). The actor does not see the camera; the gaze loop writes the head
  slots and the walk command, as `walk_brain.py` does now. Critic as in walk61's recipe (adds the true base velocity).
- Robot model: `--cmu-config plugged` (A7Z on the roof saddle, bench supply; `cmu/model/README.md`), not the 2026-10-15
  as-built manifest with the Pi Zero W head. Not modelled there: saddle lid, U20CAM cable (recorded in that README).

### Degenerate-policy audit (tests/test_t1_delta.py, from the cfg constants; the GPU audit confirms it before the run)

    policy                                   head reward /s    stepping reward /s        verdict
    tracks neck and head (DC error 0.08)       1.52             air_time 1.0 + EMA term 1.4-1.9 at 0.15-0.30 m/s
    ignores the neck (walk61 today)            0.86 (mean over the neck range)            loses 0.66/s
    stands still under 0.15..0.30 m/s          same head terms   EMA term 0.1-1.3 (loses 0.7-1.9/s) and no air_time
    creeps at 0.05 m/s vs standing             indistinguishable (0.1/s): the lowest trained speed is 0.10 m/s

    sim done criterion: DC head error 0.10 rad (not the plan's 0.05: a 0.05 rad error costs under 0.1/s in any term and the
    robot's stand61 already shows 0.11 rad on the head at a 0.3 command)

### Jobs (Kaggle T4, free, `cmu/finetune/kaggle/kaggle_job.py`; 15.9 h of the weekly 30 h left on 2026-10-30, reset 10-31 00:00)

    python3 cmu/finetune/kaggle/kaggle_job.py push run --run-name audit-t1 -- --eval-variants t1_asbuilt \
        --onnx /root/assets/walk61.onnx,/root/assets/stand61.onnx --cmu-config plugged --num-envs 2048 --eval-steps 750 --max-hours 0.5
    python3 cmu/finetune/kaggle/kaggle_job.py push smoke --run-name smoke-t1 -- --delta t1 --cmu-config plugged --max-lr 0.0003 --track-guard 1.3
    python3 cmu/finetune/kaggle/kaggle_job.py push run --run-name t1 -- --delta t1 --cmu-config plugged --max-iters 1000 --max-hours 3 \
        --max-lr 0.0003 --track-guard 1.3

The audit (walk61 and stand61 rolled out in the t1 env, 2048 envs x 15 s, about 5 min) must show stand/walk under 0.83
and walk61 falls under 0.5 per env-minute, and it records walk61's head error per joint under the t1 commands (the
`head_err_rad` field) as the baseline the run must beat. Smoke: 64 envs, 6 iterations, model_0.onnx == walk61, ONNX
export (about 4 min). Run: 100 critic-only warm-up iterations, then PPO at lr <= 3e-4 under the reward and tracking
guards, 1000 iterations (about 80 min on a T4). Note for the guard: its low-speed baseline is walk61 under t1 commands,
which includes head-command episodes where walk61 does not step, so the baseline is lower than in ft3.

### Audit and smoke (2026-10-30 04:15-04:22 EDT, Kaggle T4, both PASS; Shawn's nod 04:0x)

    variant       stand/walk   walk61 falls/env-min   margin   walk61 head error (rad, mean |err|)          walk61 speed along / low-cmd
    t1_asbuilt      0.748           0.193            1.92/s   neck 0.119, head_pitch 0.059, yaw 0.072       0.072 / 0.062 m/s

stand61 in the t1 env: falls 0.97/env-min, neck error 0.346 rad, head_pose_bias -0.86/s. Gate: ratio under 0.83 and falls
under 0.5, PASS (`reward-audit.json`, key `t1_asbuilt`). The walk61 head error (0.12 rad on the neck) is the baseline the
run must beat; the sim mass is the plugged variant at 901.6 g (saddle, 6530B, re-run leads; `cmu/model/README.md`).
Smoke-t1: `model_0.onnx` equals walk61 (max diff 0.00e+00), 6 iterations at 64 envs in 2.3 min, ONNX exported, exit 0.
Run `t1` pushed 04:3x (`microduck-t1`, 4096 envs, 1000 iterations, 3 h cap, lr <= 3e-4, tracking guard 1.3).

### Acceptance

- Sim (before the robot): the six-row table above rerun with the exported policy: at vx 0.25 with neck -0.4 and -0.8 the
  duck walks (path over 0.6 m in 6 s, as walk61 does without a head command) and the neck ends within 0.10 rad of the
  command; standing, the same neck accuracy; plus the as-built suite against walk61 (`cmu/simcheck/sim_suite.sh`,
  real-condition cases) with no new falls.
- Robot (Shawn runs the procedure): standing, neck -0.8 then back to the default; then a 0.25 m/s walk with the neck at
  -0.4; then the gaze loop (`mac/gaze.py`) on a hand-held ball with the body following.
- Not covered by t1: speeds below 0.10 m/s (the reward cannot separate them from standing), turning under the head
  (section 5's yaw-from-head-yaw is a brain-side mapping on top of this policy), and the bend (T2).

## Run t1 (2026-10-30): 1000 iterations on one Kaggle T4, no guard stop

`microduck-t1` kernel, 4096 envs, 0.21 it/s, 81 min, stop reason max_iters, best reward 129.7 at iteration 800
(`cmu/whole/runtime-overlay/models-t1-best/t1_progress.csv`, `t1_run_record.json`). Per-save task metrics (warm-up
baseline = frozen walk61 under t1 commands):

    iter   track_err   speed_along_cmd   low_cmd_speed   mean_reward   ep_len   action_std
    100     0.270        0.035             0.026            71.2        828      0.250  (baseline)
    200     0.220        0.119             0.107           115.8        953      0.202
    500     0.205        0.176             0.160           123.5        965      0.193
    800     0.208        0.177             0.159           129.7        967      0.187  (model_best)
    999     0.207        0.182             0.165           127.5        966      0.186

Every task metric moved the right way for the whole run (speed along the command x5, low-command speed x6, tracking
error -23 %), the opposite of ft1/ft2. The head error was not logged per save (only the eval has `head_err_rad`); the
CPU acceptance below measures it.

### Acceptance in the as-built sim (plugged, 7.0 V)

`t1_sim_accept.py` on iteration 800 (`t1_accept_best_it800.json`) and 999 (`t1_accept_it999.json`), walk61 baseline in
`t1_accept_walk61_baseline.json`:

    row                               it800 neck / path        it999 neck / path        walk61 neck / path
    walk 0.25, neck -0.1               -0.139 / 1.16 m          -0.176 / 1.26 m          -0.054 / 0.009 m
    walk 0.25, neck -0.4               -0.393 / 1.09 m          -0.456 / 1.21 m          -0.286 / 0.007 m
    walk 0.25, neck -0.8               -0.711 / 0.95 m          -0.803 / 1.11 m          -0.557 / 0.006 m
    walk 2 s then stand, neck -0.4     -0.379                   -0.466                   (no step)
    walk 2 s then stand, neck -0.8     -0.740                   -0.772                   (no step)
    walk 0.25, head_pitch 1.2          head 1.23 / 1.30 m       head 1.23 / 1.27 m       head 1.17 / 0.010 m
    peak tilt in these rows            7.7-8.4 deg              7.4-8.8 deg              1.9-3.5 deg (standing still)

Sim done criterion met by both: walking at 0.25 with the neck anywhere in -0.8..-0.1, neck within 0.09 rad (it800) or
0.06 rad (it999, which overshoots the small commands), standing after a step with the neck held, walking with the gaze
head pitch. The three standing-before-any-step rows run Pollen's stand61 and still fail (-0.4 -> -0.125); with the t1
policy copied over stand61 too they pass from the open-loop stand entry (`t1_accept_both.json`: -0.4 -> -0.358,
-0.8 -> -0.618, tilt 4.2 deg, no fall), so the robot test can either step first or run t1 as both policies.

### Sim suite (plugged, real-condition cases; `cmu/simcheck/reference-t1-best.jsonl` against `reference-walk61-plugged.jsonl`)

    case          it800 dist / tilt / jumps      walk61 dist / tilt / jumps     verdict rule
    nominal        2.40 m / 10.3 deg / 25         1.59 m / 6.5 / 3               worse (tilt +3.8, jumps)
    vx0.25         2.06 m /  9.6 / 14             1.17 m / 6.3 / 1               worse (tilt +3.3, jumps)
    vx0.2          1.76 m /  8.9 / 5              0.00 m / 4.0 / 0               worse by the rule; walk61 does not step
    com+5          3.02 m / 12.4 / 13             2.14 m / 14.0 / 10             worse (jumps)
    imu_pitch4     1.73 m /  6.7 / 27             1.46 m / 5.5 / 2               worse (jumps)
    stress (report only): volt6.3 2.43 m no fall; drop8_imu35 falls left at 7-9 s in both seeds (walk61 walks);
    floor_mu_bare falls forward at 15 s (the robot has soles; floor_mu_sole walks 2.0 m).

No falls and more distance in every real case, but 3-4 deg more peak tilt and 5-27 action jumps per 10 s against
walk61's 0-10: the same jitter pattern ft3 had, and ft3 lost balance on the robot at 0.20 m/s (three of three) while
walk61 did not. The suite's verdict is "worse than the reference" on those two measures. What the run was for (walk
with the neck down, keep stepping under head commands, stand with the neck held) it does; whether the extra tilt
transfers as a fall is the robot test's question. Candidate delivered as `cmu/whole/runtime-overlay/models-t1-best/`
(README there has the robot procedure); walk61 stays deployed.

### Robot test (2026-10-30 10:27-10:29; robotics repo `microduck-sim/guide/firmware/mac/robot-runs-20261030-t1/`)

The extra tilt transfers as a fall. The neck part of the goal holds on the robot; the gait does not.

- Neck: command -0.4 -> -0.38..-0.42 while walking at 0.25 m/s; -0.8 -> -0.70..-0.92 walking, then -0.816 held for
  10 s standing at 1.9 deg tilt (walk61 on the robot: -0.09).
- Plain walk at 0.25 m/s, no head command: fell sideways 3.4 s in (tilt 6.5 -> 12 deg over 3 s, then 29 deg while
  still walking, guard at 36 deg). The two head-command walks stayed up but tilted 18-20 deg at the stop.
- Against walk61's five 0.25 m/s walks on the same robot the same day: peak tilt 13.8-29.3 deg vs 7.4-8.6, action p95
  1.77-2.06 vs 0.91-1.19, tick-to-tick action jumps over 0.5 rad 17-22 per s vs 1.4-2.7, servo read misses alike
  (3-5 % vs 1-6 %).

The sim suite ranked the jitter the right way round (14 vs 1 jumps per 10 s at vx 0.25) but no sim case fell, so a
sim pass on tilt and falls does not clear a fine-tune for the robot while its action jumps exceed walk61's. That is
the second fine-tune (after ft3) to fail on the robot in this way. Not deployable; walk61 stays deployed.

### Why T1 jitters, and what published work does about it (2026-10-30, after the robot test)

**Speed-matched check first** (`cmu/simcheck/speed_matched.py`, plugged, the suite's base flags; rows in
`cmu/simcheck/speed_matched_t1*.jsonl`). At the same command T1 walks almost twice as fast as walk61 (vx 0.25: 2.06 m
against 1.17 m in 10 s), because walk61 under-tracks. At the same actual speed most of the gap disappears:

    T1 command      neck    dist 10 s   peak tilt   max|a|   jumps > 0.5 rad / 10 s
    0.10            -       1.16 m      7.3 deg     1.03     3
    0.12            -       1.33        8.2         1.02     4
    0.15            -       1.47        8.5         1.03     3
    0.10            -0.4    0.79        5.9         1.28     5      neck reached -0.46
    0.15            -0.4    1.25        7.0         1.15     3      neck -0.46
    0.10            -0.8    0.59        5.7         1.46     5      neck -0.85
    0.15            -0.8    1.08        6.5         1.28     5      neck -0.74
    walk61 0.25     -       1.17        6.3         0.91     1      (reference-walk61-plugged.jsonl)
    T1 0.25         -       2.06        9.6         1.08     14     (reference-t1-best.jsonl)

Jumps are the same measure as the robot log. On the robot both policies jumped about 15 times more often than in the sim
(walk61 1.4-2.7 per s against 0.1; T1 17-22 against 1.4), so the sim ranks the jitter correctly and the robot amplifies
it. Most of T1's jitter is the cost of actually walking at the commanded 0.25 m/s; at walk61's own speed T1 is within
about 1 deg tilt and 2-4 jumps per 10 s of it, and it keeps stepping with the neck down, which walk61 cannot.

**Published recipes and our gap** (sources read 2026-10-30):

- Lipschitz-constrained policies (Chen et al., IROS 2025, arXiv 2410.11825, https://lipschitz-constrained-policy.github.io):
  a gradient penalty lambda * E||grad_s log pi(a|s)||^2 on the whole observation, lambda 0.002 (ablation 0.001-0.01).
  Sim action jitter: 3.21 with it, 5.74 with smoothness rewards, 7.86 with a low-pass filter, 42.19 with nothing; the
  filter lowered task return and "dampen[s] or limit[s] exploration". Real humanoids (GR1, H1, Berkeley) run LCP
  alone. Our recipe has only the smoothness-reward kind (action_rate_l2, final weight -1.0); no policy-side constraint.
- CAPS (Mysore et al., ICRA 2021, arXiv 2012.06644): temporal (consecutive actions) and spatial (nearby states) losses
  added to the policy objective; about 80 % less power on a real quadrotor. Same family as LCP.
- humanoid-gym (RobotEra, `humanoid_env.py::_reward_action_smoothness`): first difference + second difference of the
  actions + 0.05 x L1 of the actions. We penalise the first difference only.
- Fine-tuning forgets (Wolczyk et al., ICML 2024, arXiv 2402.02868): fine-tuning erodes the pretrained behaviour in
  states not visited early; behaviour cloning (KL(pi_pretrained || pi) on pretraining states) worked best, vanilla
  fine-tuning was no better than training from scratch. Our fine-tunes have no anchor to walk61.
- Step-wise Gaussian exploration gives shaky motion on hardware (Raffin et al., CoRL 2021, gSDE). Our fine-tunes restart
  the exploration std at 0.25 i.i.d. per tick (ended at 0.19).
- Runtime low-pass filters: used with smoothing rewards, not instead of them (a Butterworth 4 Hz on a biped, arXiv
  2401.16889; Disney's Olaf low-passes upsampled targets; both from search summaries, papers not read in full); trained-unfiltered policies deployed filtered break transfer
  (AGENTS.md), and our 2026-10-16 EMA 0.5 trial on the goals did not reduce the walking jumps (walk_wired.py note).

**Next, in cost order.** (1) No training: T1 on the robot at vx 0.10-0.15 with the neck at -0.4 and -0.8
(`guide/firmware/mac/t1_robot_test.py walk_at 0.10 -0.4`); the sim says it walks at walk61's speed with walk61's
smoothness and the neck down. (2) If a faster T1 is needed, a fine-tune with the LCP penalty (lambda 0.002) and a
second-difference term, anchored to walk61 by a KL term on walk61's own states, judged by the sim jump count at
matched speed before any robot time. Training needs Shawn's nod after the plan.

**Robot check of the speed explanation (2026-10-30 11:00, robotics `guide/firmware/mac/robot-runs-20261030-t1/`,
"Follow-up"): it failed.** T1 at 0.10 m/s with the neck at -0.4 stayed up but made 13.5 action jumps per s (walk61 at
0.25: 1.4-2.7; T1 at 0.25: 17-22) and tilted 15.5 deg at the stop; at 0.15 m/s with the neck at -0.8 it fell at 2.7 s
after holding about 11 deg of roll to one side. Slowing down removed about a third of the jitter. The matched-speed sim
rows (3-5 jumps per 10 s) do not carry over: the robot's amplification of T1's jitter is far larger than of walk61's,
so the jitter is a property of the fine-tuned policy, and the remaining route is the training-side one above (LCP
gradient penalty, a second-difference term, a KL anchor to walk61), with the acceptance judged on the robot's jump
rate, since the sim's jump count ranked T1 correctly but underestimated it by an amount that differs per policy.

## Run t2 plan (2026-10-30, needs Shawn's nod): T1's head following, anchored to walk61's gait

### What failed and what the robot data say (robotics `guide/firmware/mac/robot-runs-20261030-t1/`)

T1 made the neck follow on the robot but fell in 2 of 5 walks and made 13.5-22 action jumps per s against walk61's
1.4-2.7. Three offline checks on the robot's own logged states (`sensitivity.py` there, `sensitivity.json`) decide
what to change:

- **Not sensor-noise gain.** The Jacobian of the actions with respect to the 61 inputs is only about 10 % larger for
  T1 (median Frobenius norm 11.9-13.3 against 10.7-12.3; gyro 0.77-0.89 against 0.73-0.85, gravity 7.2-8.2 against
  6.4-7.8, joint velocity equal). The large difference is the command input (2.9-3.5 against 1.3-1.5), which is what
  T1 was trained to use. A Lipschitz penalty alone (LCP) therefore does not target the cause; it is left out.
- **T1 rewrote walk61's gait where walk61 already worked.** On walk61's own robot walks (0.25 m/s, head at default,
  649 ticks) T1's actions differ from walk61's by 0.71 rad on average (vector norm), 73 % of walk61's own action size;
  0.10-0.27 rad per leg joint. That is the fine-tuning forgetting of Wolczyk et al. (ICML 2024): the pretrained
  behaviour is lost even in states the new task does not need to change.
- **The jumps are lateral balance corrections.** T1's jumps sit in hip roll, hip pitch and knee of both legs (left hip
  roll on 14 % of ticks at 0.10 m/s; walk61 at most 2.4 % on any joint). On the robot T1 walks leaning 2-5 deg to the
  same side as both falls (mean roll -2.1..-4.8 deg, walk61 -0.9) and up to 5 deg further forward than walk61 (mean
  pitch +0.6..+3.8, walk61 -1.1); in the sim it does not (sim pitch +0.5 deg at 0.25, robot +3.8), so the fine-tune lost walk61's tolerance of the real robot's differences from the model.

### Delta t2 = T1 + an anchor to walk61 (rewards unchanged)

- **Anchor (behaviour-cloning form of Wolczyk et al.)**: during the 100 critic warm-up iterations the actor is walk61,
  frozen; keep the actor inputs and walk61's mean actions from envs whose head command is exactly the default and whose
  forward command is 0 or 0.25-0.30 (walk61's robot-proven regime; below 0.25 walk61 does not step, so those states
  would teach not stepping). Subsample to 200k states. After every PPO update, one supervised step on a 4096-state
  minibatch of that buffer: loss = beta x mean sum over joints (mu(s) - mu_walk61(s))^2, same optimizer and learning
  rate. States with a head command are never anchored, so the neck reward keeps its full pull there.
- **Default-head envs**: 20 % of envs get the exact default head command (as `rel_standing_envs` does for the twist),
  so the buffer fills (~570k candidate states in 100 iterations) and the deployment's most common head state stays
  trained. Everything else is T1's delta.
- **Two variants on one paired Kaggle job** (two T4s, same quota as one): t2a beta 0.5, t2b beta 2.0. Logged per
  iteration: anchor distance on the buffer, neck error, speed along the command.
- **What the actor sees**: unchanged, the same 61 numbers walk61 sees (gyro 3, gravity 3, joint offsets 14, joint
  velocities 14, last action 14, twist 3, head command 4, body 6 at zero). No camera, no new input; the runtime
  needs no change.

### Degenerate audit

The reward terms and weights are T1's, so T1's audit holds (stand/walk 0.748, ignoring the neck costs 0.66/s,
standing under 0.15-0.30 costs 0.7-1.9/s). The anchor is a loss on default-head states only, where walk61's own
behaviour (walking at 0.25-0.30, bringing the stride to rest at 0) is the target, so it cannot pay for standing under
a walking command or for ignoring a head command. Risk it does carry: shared weights pulled toward walk61 could weaken
neck tracking everywhere; gate G2 measures that, and the two betas bracket it.

### Gates, in order (each must pass before the next costs anything)

- **G1, offline, robot states**: action distance from walk61 on `walk61_see_walks.jsonl` <= 0.25 (T1: 0.71), with
  `sensitivity.py`. No sim, no robot.
- **G2, sim acceptance** (`cmu/finetune/t1_sim_accept.py`, plugged): T1's rows: walking at 0.25 with the neck at -0.4
  and -0.8 path >= 0.6 m and neck within 0.10 of the command; walk 2 s then stand with the neck held.
- **G3, sim suite** (`sim_suite.py --variant plugged`): real cases within the suite rule against walk61 (tilt +2 deg,
  jumps +2, distance >= 0.9 x); and drop8_imu35, both seeds, must not fall, since walk61 does not and T1 did. Using a
  stress case as a gate is a change to the suite's rule for this run only; Shawn's word is needed for it.
- **G4, robot** (Shawn runs `guide/firmware/mac/t1_robot_test.py`): stand, walk 0.25 with no head command, walk_neck
  -0.4, neck_hold -0.8. Stop on a fall, walking tilt above 9 deg or more than 5 jumps per s.

### Cost and code

Kaggle free T4s: 14.3 h of the weekly 30 h left on 2026-10-30, reset 2026-10-31 00:00 UTC. Smoke about 10 min, the
paired 1000-iteration job about 90 min (T1 took 81 min; one extra minibatch step per update). No Modal, no MacBook
training. Code before any push: buffer collection and the anchor step in `finetune_worker.py` (wrapped around
`alg.update` like the learning-rate clamp), a `t2` delta with `default_head_envs`, `AUDIT_KEY` t2 -> t1_asbuilt,
CPU tests (buffer filter keeps only default-head and 0 / 0.25-0.30 states; the anchor loss is zero for walk61 itself
and falls under one step on a perturbed actor), then the smoke.

## Run t2 (2026-10-30): implementation, smoke and launch

Shawn approved the t2 plan, including drop8_imu35 as a G3 gate for this run ("点头，跑", 2026-10-30).

- **Code** (`finetune_worker.py`): `DELTA_T2A` / `DELTA_T2B` = `DELTA_T1` + `default_head_envs` 0.2 (the head command's
  `zero_command_prob`) + `anchor` (beta 0.5 / 2.0, 200k states, minibatch 4096, vx 0 or 0.25-0.30). While the actor is
  frozen in the warm-up, `collect_anchor_states` keeps up to 2000 filtered actor inputs per iteration from the rollout
  storage with walk61's mean actions; at the end of the warm-up they are written to `anchor_states.pt`;
  `anchor_step` runs one optimizer step on `anchor_loss` after every PPO update (after the learning-rate clamp) and
  `progress.csv` gains a last column `anchor_dist` (mean vector-norm distance on the minibatch). For anchored runs the
  actor's observation normalizer is frozen at walk61's (`until = 1`); without that rsl_rl keeps updating it, and T1's
  exported normalizer had drifted 0.025 in mean and 0.042 in divisor. At start the worker checks that actor columns
  48-50 and 51-54 equal the twist and head commands. `finetune.py` AUDIT_KEY maps t2a/t2b to t1_asbuilt (same
  rewards).
- **Tests**: `tests/test_t2_anchor.py` (delta = T1 + anchor, default-head share applied, the filter keeps exactly the
  default-head vx 0 / 0.25-0.30 rows, `actor_mean` on walk61's weights reproduces walk61.onnx within 1e-4, the anchor
  loss is 0 for walk61 and falls after one Adam step on a perturbed copy); with `test_t1_delta.py` 12 passed.
- **G1 tool**: robotics `guide/firmware/mac/robot-runs-20261030-t1/action_distance.py`; it gives T1 0.711 rad and
  walk61 0.000 on the 649 states of `walk61_see_walks.jsonl` (gate 0.25, exit 1 on a miss).
- **Smoke** (Kaggle, paired smoke-t2a / smoke-t2b, 64 envs, 6 iterations, 2.2 min each): both exit 0 and export ONNX;
  model_0 == walk61 (0.0); the input-column check passed; 679 anchor states kept (331 at vx 0); anchor_dist 0.065 /
  0.062 at iteration 5; the exported normalizers equal walk61's exactly. smoke-t2a was stopped by the tracking guard at
  iteration 5 because 64-env episodes are too short to step (walk61's own low-command speed there is 0.03 m/s, the
  floor); the guard is T1's, unchanged.
- **Launch**: paired Kaggle job `microduck-t2a` (t2a GPU 0, t2b GPU 1), 4096 envs, 1000 iterations, 3 h cap,
  `--cmu-config plugged --max-lr 0.0003 --track-guard 1.3`, the T1 flags otherwise:

      python3 cmu/finetune/kaggle/kaggle_job.py push run --run-name t2a --pair-run-name t2b -- --delta t2a --cmu-config plugged \
          --max-iters 1000 --max-hours 3 --max-lr 0.0003 --track-guard 1.3 -- --delta t2b --cmu-config plugged \
          --max-iters 1000 --max-hours 3 --max-lr 0.0003 --track-guard 1.3

### t2 results (2026-10-30): the anchor keeps walk61's gait; the neck reaches less far; not robot-tested

Both jobs ran 1000 iterations without a guard stop (t2a 78.5 min, t2b 81.6 min). Warm-up kept 200k walk61 states each
(about 106k at vx 0). Anchor distance on the buffer settled at 0.30 (t2a) and 0.20 (t2b); speed along the command
rose 0.035 -> 0.12 (t2a) and -> 0.093 (t2b), against T1's 0.18. Best saves: t2a iteration with peak reward 128.5,
t2b iteration 850. Candidate directory `cmu/whole/runtime-overlay/models-t2b-best` (README there).

- **G1** (649 states of walk61's robot walks, gate 0.25): t2b 0.208 final / 0.221 best, PASS; t2a 0.320 / 0.303,
  FAIL; T1 0.711. No t2b joint differs by more than 0.07 rad on average (T1 up to 0.27).
- **G2** (`t1_sim_accept.py`; `cmu/finetune/t2b_accept_*.json`, `t2a_accept_best.json`), rows the plan names:

      row                             t2b best       t2b it999      t2a best       T1 it800
      walk 0.25, neck -0.4            -0.367/0.59 m  -0.356/0.54 m  -0.380/0.93 m  -0.393/1.09 m
      walk 0.25, neck -0.8            -0.686/0.56 m  -0.623/0.53 m  -0.670/0.86 m  -0.711/0.95 m
      walk 2 s then stand, neck -0.4  -0.367         -0.354         -0.387         -0.379
      walk 2 s then stand, neck -0.8  -0.682         -0.619         -0.736         -0.740

  Every variant misses the neck -0.8 tolerance (0.10) by 0.01-0.08 while walking. t2b also misses the 0.6 m path
  threshold under a neck command; that threshold was set from T1's speed. Plain walks at vx 0.25 in the same sim
  (seed 0, 6 s): walk61 0.45 m, t2b 0.66 m, t2b best 0.70 m, T1 1.24 m, so t2b under a neck command still steps
  faster than walk61 without one.
- **G3** (`sim_suite.py --variant plugged`, t2b best, `cmu/simcheck/reference-t2b-best.jsonl`): real cases nominal,
  com+5 and imu_pitch4 match or beat walk61; vx0.25 has 4 action jumps per 10 s against the allowed 3 (walk61 1,
  T1 14); vx0.2 fails the tilt and jump limits because walk61 stands still there (0 m) and t2b walks (1.02 m, 7.5 deg).
  Action jumps on the real cases 2-4 per 10 s against T1's 5-27. drop8_imu35 over seeds 0-5
  (`cmu/simcheck/drop8_seeds_t2.jsonl`): falls walk61 0/6, T1 3/6, t2b best 1/6 (seed 0), t2a best 1/6 (seed 3).

Verdict under the plan: G1 passes for t2b, G2 and G3 fail, so t2b does not go to the robot without Shawn's waiver.
The anchor did what it was for in the sim (T1's 0.71 rad rewrite of walk61's gait down to 0.22, action jumps back to
walk61's range, drop8 falls 3/6 -> 1/6), and cost neck depth at -0.8 (0.69 against T1's 0.71 and the 0.70 needed).

## Run t3 plan (2026-10-30 night, needs Shawn's nod): the anchor's calm in every head state

### What t2b showed on the robot (robotics `guide/firmware/mac/robot-runs-20261030-t1/t2b/README.md`)

t2b walked like walk61 with the default head (8.7 deg peak tilt, 1.7 action jumps per s; walk61 the same hour 6.4-7.5
and 2.0) and like T1 as soon as the neck was commanded (neck -0.4: 16.4 deg, 14.0 jumps per s; neck -0.8: 10.8 deg,
13.6 per s, and it did not walk forward; standing it held the neck at -0.64 for a -0.8 command). The anchor covered
default-head states only (`anchor_mask`), so this was the design's own prediction and should have been stated before
the robot run; both t2 sim gates passed or nearly passed because the sim did not show the head-down shaking (0-2 jumps
per 10 s against about 140 on the robot).

The jitter study (`guide/firmware/mac/jitter-20261018/README.md`, "2026-10-30 targets", 58 settings) then located the
sim gap and found one setting that reproduces five of today's six robot rows within 2x with the ordering kept:
`--gyro-noise-dq 0.25 --bam-friction-scale 1.4`, gyro noise whose rms is 0.25 x the rms leg joint speed that tick (foot
strikes and play shake the trunk in proportion to leg motion; the robot's extra gyro power is above 5 Hz and grows with
the shaking) plus BAM friction x1.4. The shaking is the same loop on both sides: both legs' hip pitch and knee in phase
at 1.7-2.2 Hz, a trunk-pitch bounce, which the base sim damps to half the robot's amplitude. The one row the setting
misses is t2b with the default head (robot 1.7, sim 8.1 jumps per s): the anchor's protection has no sim counterpart,
so the calibrated sim is pessimistic exactly where the anchor acts, and it cannot falsely pass a jittery policy.

### Delta t3 = t2b + the calibrated gap in training + a smoothness term that reaches every state

Three changes, each tied to a measured fact; rewards otherwise T1's, anchor otherwise t2b's:

1. **Train in the identified gap, not beside it.** The training DR gets the two calibrated mechanisms so the policy
   learns under the trunk shake the robot really has: (a) joint-friction DR range (0.9, 1.1) -> (1.0, 1.5) on BAM's
   `friction_scale` (covers the 1.4 the study needed; 1.5 alone stopped walk61 in the study, so the range is per env, not
   a fixed 1.5); (b) a new gyro observation-noise model, rms per axis = K x rms leg joint speed this tick, K drawn per
   env in 0.15-0.35 (study value 0.25), replacing the constant gyro noise for the actor only (the critic keeps clean
   observations, as the recipe's `_safe` critic terms do today). Published basis: the asymmetric-critic / noisy-actor
   recipe the 10/13 literature pass recorded, applied to a noise model fitted to this robot's logs rather than a guess.
2. **Second-difference action term on all states** (humanoid-gym `_reward_action_smoothness`: ||a_t - 2 a_{t-1} +
   a_{t-2}||^2, the term our recipe lacks; the robot's bounce is an alternating pattern that a first difference prices
   at the same rate as a smooth stride). New mdp function `action_acc_l2`, >= 0, negative weight; weight set by reward
   mass, not by copying: on walk61's own rollouts in the calibrated sim the new term's mass is matched to
   `action_rate_l2`'s at its final weight -1.0 (the warm-up iterations print both; the weight is fixed before iteration
   100, then constant). LCP's gradient penalty stays out, as the 10/30 Jacobian check showed the input gain is not the
   cause.
3. **Anchor unchanged for default-head states; a counterfactual leg anchor for head-command states.** For states with a
   non-default head command the target is walk61's own leg actions on the same state with the head columns zeroed
   (what walk61's gait would do if the head were at default), applied to the ten leg outputs only, at beta_legs; the
   four head outputs are free there so the neck reward keeps its full pull. walk61 does not step with a head command,
   so anchoring to walk61(s) itself would teach standing (why t2 excluded those states); the counterfactual keeps the
   gait and lets the legs deviate where the head-down balance needs it. Two variants on one paired Kaggle job:
   t3a beta_legs 0 (changes 1 and 2 only), t3b beta_legs 1.0 (half of the default-head beta 2.0).

What the actor sees: unchanged, the same 61 inputs as walk61; the runtime needs no change. Policies stay unfiltered.

### Degenerate audit (incentive change: rules literature-before-experiments.md and the audit gate in finetune.py::run)

Changes 1 and 2 make walking cost more, so standing under a walking command gets relatively cheaper. Before the GPU
run the CPU audit (`tests/test_t3_delta.py`, from the cfg constants) and the Kaggle audit job must show, under the
calibrated DR: stand-still earns less than walking at 0.25 by at least T1's margin (0.748 ratio, 0.7-1.9/s), the new
term's mass on walk61's own rollouts is within 2x of `action_rate_l2`'s, every penalty logs <= 0, and ignoring the neck
still costs 0.66/s. The tracking guard stays on tracking error (1.3), never on total reward.

### Gates, in order (each must pass before the next costs anything)

- **G1, offline, robot states**: action distance from walk61 on `walk61_see_walks.jsonl` <= 0.25 (`action_distance.py`;
  t2b 0.221, T1 0.711).
- **G2, calibrated sim** (new: `jitter_sim.py --gyro-noise-dq 0.25 --bam-friction-scale 1.4`, three seeds, 10 s walks at
  vx 0.25): with the neck at -0.4 and -0.8, jumps per s <= 2 x walk61's default-head row in the same setting (walk61
  2.7 -> limit 5.4; t2b scores 11.4 and 13.3 there, T1 15.7), peak tilt <= 12 deg, no falls; neck within 0.10 of -0.4
  and 0.15 of -0.8 while walking (t2b -0.37 / -0.69), path >= 0.6 m in 6 s. The default-head row is reported, not gated
  (the known false alarm).
- **G3, sim suite** (`sim_suite.py --variant plugged`): real cases within the suite rule against walk61; drop8_imu35 over
  seeds 0-5 no falls (walk61 0 of 6, T1 3, t2b 1).
- **G4, robot**, only after G1-G3, and only with the two pre-robot lines written in the plan record: (1) the states the
  change does not cover and what the policy does there; (2) whether the sim has been shown blind to the failure being
  tested. Steps as t2's (`t1_robot_test.py restart t3x`, stand, walk, walk_neck, neck_hold, back), stop on a fall,
  walking tilt above 9 deg or more than 5 jumps per s; plus a 200 Hz gyro capture during the plain walk to settle
  whether the >5 Hz trunk power is motion or IMU-mount vibration.

### Cost and code

Kaggle free T4s: 12.69 h of the weekly 30 h left at 2026-10-30 17:20 EDT, reset 2026-10-31 00:00 UTC (20:00 EDT
tonight). Audit and smoke about 15 min, the paired 1000-iteration job about 90 min (one extra reward term, one extra
anchor target per state). No Modal, no MacBook training. Code before any push: friction range and the gyro noise model
in `apply_delta` (new keys `friction_range`, `gyro_noise_dq`), `action_acc_l2` in mdp.py with its weight-by-mass step
in the worker, the counterfactual leg anchor (`anchor_mask` keeps all states; per-state target = walk61 on the state
with head columns zeroed, leg columns only, beta_legs), `DELTA_T3A/B`, `AUDIT_KEY` t3 -> t1_asbuilt, CPU tests
(`test_t3_delta.py`: ranges applied, noise rms scales with leg speed, acc term zero on a constant action and positive
on an alternating one, counterfactual target equals walk61's leg actions on the zero-head state, anchor loss zero for
walk61 itself), then the Kaggle audit and smoke, then the paired run. Nothing starts before Shawn's nod.

### Run t3 progress (2026-10-30 evening, Shawn's nod "点头，跑")

- **Code** (fork commits 1bb0560c, 5cdee5d5, 912a596c): `base_ang_vel_imu_misaligned_vibration` and `action_acc_l2` in
  `mdp.py`; `apply_delta` keys `gyro_vibration` (actor base_ang_vel only) and `action_acc`; `anchor_targets` /
  `anchor_loss_w` (per-state weight vector: beta on all 14 outputs for default-head rows, beta_legs on the 10 leg outputs
  and 0 on the head outputs for head-command rows, target walk61 on the head-zeroed state); `DELTA_T3A/B`;
  `tests/test_t3_delta.py` 6 tests, with `test_t1_delta.py` and `test_t2_anchor.py` 18 passed. `cmu/finetune/g2_jitter.py`
  runs G2 (walk61 reference plus each candidate's default / neck -0.4 / neck -0.8 rows under
  `--gyro-noise-dq 0.25 --bam-friction-scale 1.4`, three seeds, 10 s walks).
- **Audit 1** (`audit-t3`, Kaggle T4, 6 min, `cmu/finetune/kaggle/out/audit-t3/runs/audit-t3/eval.json`, friction
  1.0-1.5, vibration k 0.15-0.35, action_acc at -1.0): walk61 reward mass per second action_rate_l2 -0.200 (t1 env
  -0.086: the vibration noise makes walk61's own actions 2.3x rougher, the robot's mechanism), action_acc_l2 -0.341 ->
  mass-matched action_acc weight -0.59; stand/walk 0.817 measured, 0.807 at -0.59 (max 0.83); **walk61 falls 0.527 per
  env-minute, over the 0.5 limit** (t1 env 0.193, stand61 1.46). Friction 1.5 was my own extension past the study's
  calibrated 1.4, and the 10/17 study already found x1.5 stops stepping, so the range is now 1.0-1.4 (commit 5cdee5d5).
- **Audit 2** (`audit-t3b`): `t3a_asbuilt` at friction 1.0-1.4 plus two audit-only diagnostics, `t3fric_asbuilt` (t2b +
  friction 1.0-1.5 only) and `t3vib_asbuilt` (t2b + vibration only), to attribute the fall rate. Result below.
  Audit 2 result (`audit-t3b`, 11 min, `cmu/finetune/kaggle/out/audit-t3b/runs/audit-t3b/eval.json`), walk61 falls per
  env-minute: t2b env 0.193 (audit-t1) -> friction 1.0-1.5 alone 0.328, vibration alone 0.344, t3a (friction 1.0-1.4 +
  vibration) 0.432; the two gaps add about 0.14 each. stand61 falls 1.25-1.40 in every variant. t3a_asbuilt: stand/walk
  0.815 at action_acc -1.0, 0.804 at the mass-matched **-0.58** (walk61 action_rate_l2 -0.199/s, action_acc_l2 -0.342/s
  at -1.0), falls 0.432 <= 0.5, **PASS**; `reward-audit.json` key `t3a_asbuilt` written from this run (the entry's
  action_acc_l2 column is rescaled to -0.58). `DELTA_T3A/B` action_acc weight -0.58.
- **Smoke** (`smoke-t3a` / `smoke-t3b` paired, 4 min, both rc 0): model_0.onnx vs walk61.onnx max abs diff 0 on 256
  random observations; anchor buffer t3a 1206 states (901 at vx 0, 0 with a head command: beta_legs 0 drops them),
  t3b 4006 states (2261 at vx 0, 2800 with a head command), `anchor_states.pt` weight rows: default-head 2.0 on all
  outputs, head rows 1.0 on the leg outputs and 0 on the head outputs; ONNX exported.
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

## Run t3+tether plan (2026-10-30, Shawn: "t3 过了就训")

Branch `cmu/t3-tether` = `cmu/grasp` (t3) plus the tether commits. Deltas `t3a_tether` / `t3b_tether` are t3a / t3b
unchanged plus `tether=dict(fraction=0.8)` (t3 already carries ft3's EMA velocity tracking). Only the variant that
passes t3's gates G1-G3 is trained; if both pass, both run as one paired job (seeds 1). What the actor sees: unchanged,
61 inputs.

- **Audit** (`t3a_tether_asbuilt`, plugged, 2048 envs, 750 steps; the anchor does not enter the audit, so one key
  covers both): stand/walk at most 0.83 and walk61 falls at most 0.5 per env-minute. The lead raised stand/walk by
  0.027 on the lowcmd recipe (0.761 -> 0.788) and t3a sits at 0.804, so the margin is thin; a FAIL stops here.
  Result (`audit-t3tether`, Kaggle, 4 min): stand/walk **0.803** (t3a 0.804), walk61 falls **0.465** per env-minute
  (t3a 0.432, limit 0.5), walk61 7.09/s, stand61 5.70/s: **PASS** at finetune.py's gate (the worker's own print uses
  its 0.8 default and says FAIL, as it did for t3a). `reward-audit.json` key `t3a_tether_asbuilt`. On t3's DR the lead
  leaves the standing incentive where it was and adds 0.03 falls per env-minute.
- **Smoke** `t3b_tether` paired with the audit (64 envs, 6 iterations): builds, the tether event registered, ONNX out.
  Result: rc 0, `tether_lead` in the event table, ONNX exported.
- **t3's gates (cmu/grasp 030407f0, d98f30ac, 2026-10-30 21:10)**: G1 all four t3 candidates pass. G2 (calibrated jitter
  sim, `cmu/finetune/g2_t3.json`) fails all eight neck rows: with the neck at -0.4 or -0.8 the neck reaches -0.24 to
  -0.60, tilt reaches 12-27 deg and t3a falls in 1-2 of 3 seeds. The default-head rows, which are not gated and are how
  the duck walks and turns on the lead, are smoother than walk61 there: jumps 0.47-1.5 per s against 3.38, peak tilt
  6.9-8.1 against 9.1 deg, no falls (t3b_final 0.47 per s, 6.9 deg). t3 did not pass, so the t3+tether run is not
  started; whether to train t3b+tether for walking and turning with the default head is Shawn's call.
- **Run** (Shawn 2026-10-30: "用 t3b 训"): paired Kaggle kernel `microduck-t3btether-a`, `t3btether-a` seed 1 and
  `t3btether-b` seed 2, delta `t3b_tether`, settings as above, pushed 21:17 EDT. Scope: walking and turning with the
  default head; the result inherits t3b's neck-row failure, so it is not a candidate for head-down walks. Acceptance:
  G2's default-head rows (jumps per s and tilt no worse than walk61's, no falls), `tether_accept.sh` lead cases, and the
  sim_suite real cases with the same rule against walk61.
- **Result (2026-10-30 22:51-23:06)**: both jobs rc 0, 1000 iterations, 85 min. No candidate passes every gate;
  `t3bt_a` (seed 1, final) misses one real case by a small margin. Files: `cmu/finetune/g2_t3btether.json`,
  `t3btether_accept_lead.json`, `t3btether_accept_seeds.json` (seeds 0-4 of the suite's real cases,
  `kaggle/seed_sweep.sh` + `seed_summary.py`).
  - G2 default head (calibrated jitter sim, 3 seeds): jumps per s walk61 3.38, `t3bt_a` 0.87, `t3bt_a_best` 0.73,
    `t3bt_b` 1.27, `t3bt_b_best` 1.37; peak tilt 7.2-7.9 against 9.1 deg; no falls; all walk further. Pass for all.
  - Lead cases (`tether_accept.sh`, seeds 0-4): turn with the lead, mean deg/s right/left, walk61 17.1/16.1 (worst seed
    12.2/7.2), `t3bt_a` 26.1/26.5 (worst 20.3/19.4), `t3bt_a_best` 26.0/26.4, `t3bt_b` 22.6/26.2, `t3bt_b_best`
    23.6/26.1; walks with the lead 1.37 m against 1.20; no falls. The turn problem the tether delta targets is fixed.
  - Suite real cases, five seeds against this run's walk61 (the stored `reference-walk61-plugged.jsonl` from 10/19 no
    longer matches the current sim: walk61 itself reads "worse" against it, so it was not used):
    `t3bt_a` passes nominal, vx0.25 (tilt +2.0 deg, at the limit), com+5, imu_pitch4 and both no-lead turns; it fails
    vx0.2 on tilt, 4.6 against 2.4 deg (both stand still at that command). `t3bt_b` is slower on nominal and com+5
    and stopped in one imu_pitch4 seed; `t3bt_a_best` and `t3bt_b_best` stand still with the 4 deg IMU pitch offset
    (0.00 m in every seed): rejected.
  - Stress (report only): `t3bt_a` fell in 1 of 5 seeds of drop8_imu35 (8 % dropped reads, 35 Hz IMU; tilt 49 deg),
    walk61 in none. The robot runs 4 % drops and a 50 Hz IMU (`microduck-sim/CURRENT-STATE.json`).
  - Open: `servo_rail_v` in CURRENT-STATE (7.0 V) is stale after the move to the battery backpack (8.0-8.2 V idle),
    and every case here ran at 7.0 V.
  - Decision for Shawn: robot G4 for `t3bt_a` with the vx0.2 tilt miss accepted, or no robot run.
- **Run** after t3 passes: Kaggle T4, `--cmu-config plugged --max-lr 3e-4 --track-guard 1.3`, 1000 iterations, warm-up
  100, `--max-hours 2`.
- **Acceptance**: t3's G1-G3 unchanged, plus `tether_accept.sh` (lead cases seeds 0-4, both turn directions):
  in-place turns with the lead faster than walk61, no slower without; walks and falls no worse; jumps and tilt within
  the suite rule against walk61 in the lead cases too. Robot only after these, under t3's G4 procedure.
