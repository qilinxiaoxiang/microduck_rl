# Static grasp (2026-10-26)

Question (Rémi, Antoine): can Microduck hold a grasp pose with the beak about 1 mm above the floor, statically stable,
without touching it? Page with videos and numbers: https://claude.ai/artifact/6kkBqY8Z5V3yrdbcbNbZG2 (private).

## Findings

- **The pose exists.** `scripts/static_grasp/sg_search.py` on the exact ground-grasp training model: head pitch 67°,
  jaw open 22°, lowest lip 1.0 mm (other lip 1.9 mm), centre of mass 1.37 cm inside the support polygon (standing:
  1.67 cm), 0.19 N.m worst holding torque, joints in range, no self-collision. Margin barely depends on height
  (0.5 mm: 1.20 cm, 10 mm: 1.53 cm). Keeping 0.10 rad from every joint limit: 0.92 cm at 1 mm.
- **It holds with no policy** (`sg_hold.py`, BAM XL330 at kP 200, constant targets = pose + torque / 0.554 N.m/rad):
  20 s, beak 0.3-1.7 mm, never touching; survives 0.2 s trunk pushes of 2.8-4.4 N.
- **The earlier claim** (ground-grasp page, 10 Oct) only covered a level beak. With the beak straight down the front
  of the head shell, 6 mm past the lips, touches first; a 67° head pitch makes the lips the lowest points.
- **Getting there with fixed targets fails** (`sg_stiffness.py`, `sg_motion.py`): a held pose is a set of servo
  springs; at kP 200 the half-crouch between standing and the grasp pose is elastically unstable (-0.6 N.m/rad),
  standing is marginal (+0.04), the grasp pose stable (+0.54). Every fixed-target descent from standing falls or
  lands the beak; robotd's `robot.poseJoints` ramps from the measured positions, so a switch drops the holding
  torques for an instant.
- **Trained move** (`Mjlab-StaticGrasp-*` tasks, from the default grasp's weights): run 4
  (`Mjlab-StaticGrasp-Hover-FromPick-MicroDuck`, resumed from run 3 · 500, wandb neoy2a9d) goes down, holds a
  statically stable crouch (frozen mid-hold on its own targets: stays up, beak 2-5 mm, CoM 1.0-1.3 cm inside), bites,
  stands up: Lego / block / die / pawn 3/3, AA 1/3, coin 0/3; CoM >= 0.84 cm inside; a 0.02-0.1 s tap as the jaw
  closes; active hover 10-17 mm; targets <= 0.25 rad past a range.

## Tasks

| task | what |
|---|---|
| `Mjlab-StaticGrasp-FromPick-MicroDuck` | ground grasp + pull to the static pose, lips -> 1.5 mm, CoM over the feet, contact priced (run 1: leaned on the beak) |
| `-Strict-` | contact and CoM-past-toes priced 5x (run 2: balanced, never hovered) |
| `-Retimed-` | arrive 0.24, hold 0.24-0.34, bite 0.34-0.40, up by 0.66 (run 3: rested a beak corner) |
| `-Hover-` | + level head, episode ends on beak contact > 0.5 N in the hold (run 4: candidate) |

Evaluate a checkpoint the way robotd plays it: `PYTHONPATH=src python scripts/static_grasp/sg_eval.py POLICY.onnx
--task <task> --name X --seeds 3 --freeze`. The beak timeline is the task's mouth action (client-played on the robot).
