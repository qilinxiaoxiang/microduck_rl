# Beak grasp of a small printed ball (final-goal step 4)

Goal (Shawn, 2026-10-26): the robot picks up a 3D-printed orange ball, 1.0-1.5 cm across and not quite round, from
the floor with its beak. This directory trains that in simulation on our robot before the robot tries it.

## Published recipe this follows

Pollen's own grasp work in `pollen-robotics/microduck_rl` (read 2026-10-26):

- `ground-grasp` branch, `docs/superpowers/specs/2026-10-10-ground-grasp-design.md`: the deployed ground pick
  (`alpha_ground_pick.onnx`, our `groundpick61`) never grasps: lips 2.7 cm above the floor, beak straight down. A
  model with a real jaw joint, a scripted mouth timeline, sticky friction (elliptic cone, impratio 10) and a blind
  policy warm-started from the deployed move reached 41/48 placements at the spot (run 8, about 3000 iterations at
  4096 envs). Weak cases: round or flat objects (lying AA battery 2/8, coin).
- `static-grasp` branch, `docs/static_grasp.md` (2026-10-26): a statically stable crouch with the lips 1-2 mm above
  the floor exists (head pitch about 67 deg, jaw 22 deg). `Mjlab-StaticGrasp-Hover-FromPick-MicroDuck` (run 4) goes
  down, hovers, bites and stands up: Lego, block, die, pawn 3/3, AA 1/3, coin 0/3.

## Where our case differs

- Masses: the CMU plugged robot is 859 g against Pollen's 737 g, with the A7Z and saddle on the back
  (`cmu/model/manifests/plugged.json`).
- Object: a 1.06-1.44 cm ellipsoid (independent radii), 0.5-2.5 g. Pollen trained 1.4-2.6 cm boxes of 5-25 g.
  A ball rolls when touched; Pollen's round object (AA) was its weakest.
- Friction: our lips are printed PLA, Pollen assumed soft TPU pads (friction 1.1-1.6). The ball's friction is drawn
  from 0.35 (PLA on PLA) to 1.1 (beak lined with bike inner tube).
- Jaw: our software cap (`guide/firmware/pi/jaw_limits.py`) allows 25 deg at the static-grasp head pose (first
  contact 27 deg), so the 22 deg opening is inside it.

## Files

- `src/mjlab_microduck/tasks/cmu_grasp_env_cfg.py`: the CMU masses and the ball on Pollen's grasp envs.
  Registered: `Mjlab-{GroundGrasp-FromPick,StaticGrasp-Hover-FromPick}-MicroDuck-CMU-Plugged[-Ball]`.
- `sg_search_cmu.py`: Pollen's static pose search on a CMU task's model; results in `poses/`.
- `grasp_worker.py`: train one task with the `train` CLI under a wall-time cap and export every checkpoint to ONNX.
  On Kaggle: `python3 cmu/finetune/kaggle/kaggle_job.py push <smoke|bench|run> --worker grasp --run-name <name> -- <worker flags>`.
