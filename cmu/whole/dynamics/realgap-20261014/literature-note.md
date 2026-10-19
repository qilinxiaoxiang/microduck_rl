# MicroDuck real-gap literature note (L1, verified 2026-10-14)

Problem: community walk/stand policies hold only on the nominal model; the real robot leans about 10 deg more forward than sim. Tags: **[S]** sourced fact, **[I]** project inference.

## Bottom line

- **[S]** The community MicroDuck recipe trains with much smaller calibration errors than our failure points: encoder bias ±0.015 rad (±0.86 deg), zero-centred IMU misalignment up to 6 deg, trunk-body COM up to ±15 mm. Its own playbook says a systematic IMU mounting bias "is a runtime calibration" that domain randomization cannot cover ([pollen-robotics/microduck_rl](https://github.com/pollen-robotics/microduck_rl), `microduck_velocity_env_cfg.py`, `AGENTS.md`, local commit 5946fd9, 2026-09-01).
- **[I]** Our failures at 8 deg hip, 4 deg ankle and 10 deg IMU pitch are outside that trained band, so a fall within 1 s is expected behaviour. Calibrate first; widen randomization second.

## 1. Sim-to-real recipes for small bipeds

- **Community MicroDuck, mjlab + BAM, 2026 [S]** ([microduck_rl](https://github.com/pollen-robotics/microduck_rl))
  - Actuator: BAM M6 XL330 model with backlash encoder, firmware kp 200, `vin_range` 6.5–8.2 V with load-dependent sag, actuator lag 3–6 steps.
  - Randomization: trunk COM curriculum 3→5→10→15 mm (capped because ±30 mm exceeded the 20 mm heel support), head COM to ±10 mm, mass and inertia ±5%, BAM friction and armature ±10%, Kp randomization off, velocity pushes ±0.3 m/s every 3–6 s.
  - Noise and delay: IMU observation delay 0–1 step (≤20 ms); noise gyro ±0.03, gravity ±0.01, joint position ±0.001 rad, joint velocity ±0.25.
  - Network and budget: MLP 512-256-128; velocity runs 50,000 iterations × 24 steps × 4096 envs (about 4.9e9 steps).
  - **[I]** The lag is probably counted in 5 ms physics steps (15–30 ms). Our robot runs at 5 V, below the trained 6.5–8.2 V band. The XL330 is rated 3.7–6.0 V (0.52 N·m stall at 5 V) ([ROBOTIS e-manual](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)), so check which `vin` the as-built sim uses.
  - **[I]** Trunk COM randomization moves only the trunk body. The whole-robot shift is m_trunk/M × 15 mm, so our 10 mm whole-robot test (`policy_test.py`) may exceed what the policy saw.
- **Open Duck Mini v2, MuJoCo Playground, 2025 [S]** ([Open_Duck_Playground](https://github.com/apirrone/Open_Duck_Playground), `randomize.py`, `joystick.py`)
  - Randomization: torso COM ±5 cm, link mass ×0.9–1.1, torso mass ±0.1 kg, `qpos0` ±0.03 rad, Kp ×0.9–1.1, floor friction 0.5–1.0, frictionloss ×0.9–1.1.
  - Delay: action delay 0–3 and IMU delay 0–3 steps at 50 Hz.
  - Noise hip 0.03, knee 0.05, ankle 0.08 rad; budget 300M steps.
  - **[I]** In MuJoCo, jittering `qpos0` changes the reset pose, not the joint reading, so it does not model encoder offset.
- **Disney BD-X, RSS 2024 [S]** ([arXiv 2501.05204](https://arxiv.org/abs/2501.05204))
  - Randomization: encoder offset U(±0.02 rad); backlash 0.002–0.005 rad on Dynamixel XH540; armature up to 20%.
  - Pelvis pushes of 90–150 N; joint noise grows with joint speed.
  - Policy 50 Hz; 3×512 ELU; 100k PPO iterations (about 22 days on one RTX 4090). No COM ranges or latency given.
- **DeepMind OP3 soccer, Science Robotics 2024 [S]** ([arXiv 2304.13653](https://arxiv.org/abs/2304.13653))
  - Randomization: joint offsets ±2.9 deg, IMU orientation 2 deg and position 5 mm, external mass up to 0.5 kg at a random torso point, floor friction 0.5–1.0.
  - Delay: observation delay 10–50 ms.
  - 40 Hz control; actor 256-256-128.
  - Calibration: hip joints loosened and encoders "became miscalibrated over time", so regular maintenance was needed.
  - Without randomization the robots fell "with every one or two steps".
- **Berkeley Humanoid, 2024 [S]** ([arXiv 2407.21781](https://arxiv.org/abs/2407.21781))
  - Randomization: default joint position ±0.05 rad, base mass ±1.0, link mass ×0.9–1.1, friction 0.2–1.25.
  - Noise: IMU 0.05.
  - Network 512-256-128; gains deliberately not randomized; latency not simulated.
- **MuJoCo Playground, Zakka et al. 2025 [S]** ([arXiv 2502.08844](https://arxiv.org/abs/2502.08844))
  - Randomization: link COM, armature and "joint calibration offsets" (no numbers given).
  - Network: asymmetric actor-critic, 512-256-128.
  - Budget: Berkeley Humanoid flat terrain trains in under 15 min on 2×RTX 4090.

## 2. Getting up from the floor

- **OP3, learned [S]**
  - Method: the get-up teacher tracks targets interpolated between three keyframes taken from the scripted ROBOTIS get-up, rewarded on pose and gravity error.
  - Budget: 2.4e8 steps, 14 h.
  - Result: 63% faster than the scripted get-up.
- **Playground Go1 get-up [S]:** 400 W power termination plus a joint-velocity cost; transfers zero-shot.
- **HumanUP, RSS 2025 [S]** ([arXiv 2502.12152](https://arxiv.org/abs/2502.12152))
  - Method: Stage I discovers a motion freely; Stage II refines it to be slow and smooth.
  - Transfer: needs accurate collision geometry.
  - Result: beats the G1 scripted controller on slopes.
- **HoST, RSS 2025 [S]** ([arXiv 2502.08378](https://arxiv.org/abs/2502.08378)): motion-speed bound plus smoothness regularization.
- **Community MicroDuck [S]:** `microduck_standup_env_cfg.py` mixes face-up and face-down starts through a `ground_state_mix` curriculum (to iteration 2500); 15,000 iterations.
- **[I]** Transfer came from capped power or speed plus smoothness, and from starting the policy in the real fallen states. A scripted keyframe get-up remains the reference baseline and the seed motion.

## 3. Calibration before deployment

- **Joint zeros [S]:** Open Duck Mini runs `find_soft_offsets.py` before policies and stores offsets in `duck_config.json` ([Open_Duck_Mini_Runtime](https://github.com/apirrone/Open_Duck_Mini_Runtime)). OP3 recalibrated as routine maintenance. BD-X and the community recipe randomize only the residual (±0.02 and ±0.015 rad).
- **IMU [S]:** Open Duck Mini checks the IMU frame by streaming it. The community runtime corrects the board's systematic ~5 deg pitch offset at the source (`imu-pitch-offset`) and randomizes only the residual.
- **COM [S]:** In a static pose the ground projection of the COM equals the centre of pressure, and static-pose least squares identifies mass and COM parameters ([arXiv 2202.13064](https://arxiv.org/abs/2202.13064); [Disney practical calibration](https://la.disneyresearch.com/publication/practical-kinematic-and-dynamic-calibration-methods-for-force-controlled-humanoid-robots/)). No small-robot paper uses a tipping test. **[I]** A scale-free tipping test is plain statics.

## 4. Head gaze and person following

- **Head pose is a policy input [S].** BD-X standing and walking policies take head height and orientation commands. Open Duck Mini's 7-D command includes neck pitch and head pitch, yaw and roll. The community MicroDuck observations include `head_command`. Pollen lists "gaze IK" as deferred ([robotd-design.md](https://github.com/pollen-robotics/microduck/blob/main/docs/design/robotd-design.md)).
- **Following is a layered stack [S].** ANYmal leader following ([arXiv 2410.00572](https://arxiv.org/abs/2410.00572), ICRA 2025):
  - Person detections are matched to a 5 Hz radio bearing within a 20 deg window.
  - A 10 Hz constant-velocity EKF tracks the leader.
  - A 50 Hz planner sends a reference velocity to the locomotion controller.
- **[I] MicroDuck:** detector on the Pi camera, then bearing, then a head-yaw gaze command plus a yaw-rate command proportional to head yaw. The policy stays unchanged.

## 5. Fall safety layer

- **NimbRo-OP [S]:** an attitude threshold triggers servo relaxation, followed by a scripted prone or supine get-up ([arXiv 1809.11051](https://arxiv.org/abs/1809.11051)).
- **BD-X and OP3 [S]:** BD-X "motion stop" moves to standing, then freezes stiff after 0.5 s; OP3 adds torso bumpers.
- **Learned safe falling [S]:** Kumar et al. 2017 minimize peak impulse ([arXiv 1703.02905](https://arxiv.org/abs/1703.02905)).
- **Pollen MicroDuck runtime `limp_fall` [S]**
  - Detection: a gyro-rate detector predicts gravity about 0.3 s ahead and fires at about 26 deg tilt when the robot is still tipping, debounced over 3 ticks.
  - Response: go limp following the joints down, wait for the gyro to settle, ramp to the standing pose over about 1 s, then hand over to the standing policy.
  - **[I]** Going limp conflicts with the torque-hold rule; needs Shawn's go.

## What this implies for MicroDuck tomorrow

Minimal calibration, in order:

1. **IMU pitch and roll offset.** Hold the trunk level on a reference surface and average projected gravity for 5 s. Then rotate it out in the runtime so it is zero. Target residual ≤2 deg, which is OP3's range and inside the community's 6 deg. *[S: community playbook, OP3. I: procedure]*
2. **Joint zeros.** Put each leg joint in a printed jig or a known mechanical stop, then write the offset. Target ≤0.86 deg, the community's trained band, and in no case above OP3's 2.9 deg. Hip pitch and ankle matter most. *[S: ODM `find_soft_offsets`, community ±0.015 rad. I: priority from our sweep]*
3. **COM check without a scale.** Torque on, at the policy default pose, tilt the support board slowly forward until the robot tips. With tip angle θ, sim COM height h and toe-edge distance d_toe, x_COM = d_toe − h·tanθ, and the sim model's x_COM gives the gap. Fix the MJCF mass placement before changing the policy. *[I: statics. S: CoP = COM projection, 2202.13064]*
4. **Re-run `policy_test.py` on the corrected model**, and confirm it reproduces the real lean. *[I]*

If retraining is needed (needs Shawn's go on compute):

- **Encoder bias:** ±0.026 rad (1.5 deg), between community 0.86 and OP3 2.9. *[S ranges, I choice]*
- **IMU misalignment:** 3 deg after calibration. *[S: OP3 2 deg, community 6 deg]*
- **Whole-robot COM:** ±10 mm, centred on the measured offset, applied as a whole-body shift instead of trunk-only. *[S: community 15 mm cap. I: whole-body conversion]*
- **Latency:** observation and action delay 10–50 ms, because the MacBook-over-USB loop is longer than the community's onboard loop. *[S: OP3. I: our path]*
- **Supply voltage:** centred on 5 V (4.75–5.25 V, as already used in this folder's perturbation runs). *[S: XL330 spec. I: range]*
- **Pushes and noise:** keep the community values. *[S]*
