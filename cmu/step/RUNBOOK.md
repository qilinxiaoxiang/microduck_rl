# Step-look-step: robot test sheet (2026-11-22)

One burst, stop, look, repeat. Two people are not needed at the laptop, but **someone spots the duck on every run**.
Everything goes to `~/dock-stage` on the A7Z; the live `~/rt` code and `~/duck-policy` are never written.

| Arm | What | May run? |
|---|---|---|
| `walk61` | the deployed walk policy (baseline) | yes |
| `step005` | round 1, old body | retired |
| `step015_v2m5` | round 3: walk61 (trimmed 6 deg) + at most 0.15 rad, in both slots | replaced by step045n_v2m5 (2026-11-23) |
| `step045n_v2m5` | round 4: walk61 (trimmed 6 deg) + at most 0.45 rad, trained under servo read noise, in both slots, on the 1035 g body | replaced by rcn030_w918 (2026-11-23) |
| `rcn030_w918` | round 7: walk61 (trimmed 6 deg) + at most 0.30 rad, trained on the 918 g body under the robot's sensor noise, in both slots (`EXPERIMENT-rcn030.md`) | **not until Michael waives its gate failure** |

**Do not press N (dock) or F (small step) on the floor.** Both fell on 2026-11-21.

## 0. Before anything

1. **Disk.** The A7Z was 100 % full. Copy the 2026-11-21 brain logs off before deleting anything (the fall's log is
   there), then free at least 1 GB. The script refuses to run below that.
2. **Right turn.** The duck could not turn right. Fix that, then reset or relearn `~/.duck_yaw_trim.json` (it holds
   -0.196 rad/s and will steer right once fixed). The bursts here do not teach the trim.
3. **Power.** Write down what is on the duck (pack and weight, or tether and where the cord leaves the body).
   The batteries are broken: a tethered, no-battery duck is not the trained body, so label those runs as such.

## 1. Install (from the repo root)

    cmu/step/step.sh check        # every line ok
    cmu/step/step.sh install

## 2. Start the baseline

    cmu/step/step.sh arm walk61   # HOLD the duck: the brain restarts limp

Press Balance stand in the app, set the duck down, ball 35-45 cm straight ahead.

    cmu/step/step.sh dry          # must print a plausible distance to the ball; the legs do not step

Stop here if that distance is more than 3 cm from a tape measure.

## 3. Single bursts (no ruler: the servos measure it)

After each, `cmu/step/odom_report.py <brain log>` gives the distance (in sim it reads 8-15 % short, right direction).
**Start every run from Balance stand, never Fixed stand**: with the new pack the Fixed stand tips back in sim if the
centre of mass is 3 mm further back than estimated. Check `step.sh check` prints no pitch trim file for the step arm
(it carries its own 6 deg).

One burst, look at the printed camera distance, put the duck back.

    cmu/step/step.sh single 0.15  # 5 times
    cmu/step/step.sh single 0.3   # 5 times
    cmu/step/step.sh single 0.6   # 5 times
    cmu/step/step.sh single 1.0   # 5 times, ball 45 cm or more ahead


These logs replace the burst tables in `arms.json` (odom_report.py prints the table).

## 4. The loop (only after 3)

    cmu/step/step.sh run          # 5 times, ball 30-45 cm, within 15 deg of straight ahead

It should end with the ball 18 cm ahead (+-2.5). It stops there on purpose: nearer, the ball leaves the picture
unless the head goes past the 1.1 rad look-down that held on the robot. This does not yet reach the pickup spot (9.5 cm).

## Stop

    cmu/step/step.sh stop

Stop the session after **one fall, two catches, any servo fault, or visible rocking that does not settle**.

## Afterwards

    cmu/step/step.sh fetch        # traces and brain logs into cmu/step/robot-runs/
    cmu/step/step.sh live         # HOLD the duck: back to the live code

Send the filled table, the videos and that folder.
