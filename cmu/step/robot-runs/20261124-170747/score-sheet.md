# rcn030_w918 robot session, 2026-11-24 (Shawn, carpet)

Robot: 918 g body, 3000 mAh battery backpack, battery 7.6-7.7 V. Floor: office carpet. Ball: 12 mm orange.
Code: pick-and-rise 4f3e695f, installed with `step.sh install` into ~/dock-stage (stage brain, live ~/rt untouched).
rcn030_w918 gate is FAIL; Shawn waived it on site ("直接装上试", 2026-11-24). The waiver lives only in the local
arms.json used for the install; arms.json in the repo is unchanged.
Videos: posted by Shawn in the team Discord.

## walk61 arm (baseline): stopped after one run

| arm | block | run | result |
|---|---|---|---|
| walk61 | V1 (Balance stand from Fixed stand) | 1 | fell forward within 4 s of Balance stand (pitch to about 65 deg in 2 s); log duck_wired_log_stepwalk61-1124-165843.jsonl. Not repeated: the 11-21 stand61 is known not to hold this body. |

## rcn030_w918: loop (`steplook.py`), started straight from walk mode at zero command

| run | start (camera) | bursts (T, closer) | end | fell | caught | notes |
|---|---|---|---|---|---|---|
| 1 | 46 cm, 6 cm left | 1.3 s +16.6 cm; 1.3 s | look after burst 2 failed; a re-look read 15.9 cm (inside 18 +- 2.5) | no | no | trace steplook_1124-170124 |
| 2 | 66 cm, 12 cm right | 1.3 s +22.6; 1.3 s +21.4; 1.3 s | ball not seen after burst 3; Shawn's photo: ball about 8-10 cm ahead of the toes, 3-4 cm right | no | no | trace steplook_1124-170222 |
| 3 | 73 cm (over the 70 cm limit) | manual 1.0 s straight walk -> 55 cm, 29 cm right; loop 1.3 s +23.0 (-> 31 cm, 37 deg right, loop stopped); manual turn in place vtheta -0.3 for 2 s -> 29 cm, 9 deg right; loop 1.3 s | ball not seen after the last burst (expected 6-12 cm ahead) | no | no | the straight burst drifted left |

## Findings

- No falls and no catches over 3 loops (8 bursts plus 2 manual commands); stands at pitch 2-3 deg between bursts.
- Real burst length on carpet at 1.3 s: 16.6-23.0 cm, against 6.5 cm typical (8.7 furthest) in the burst table.
  The planner still picks 1.3 s at 22-31 cm, so the last burst ends 6-16 cm from the ball, closer than the 18 cm stop.
- Every loop ended with "ball not seen": the ball is below the frame at the 1.1 rad look-down once it is that close.
  On the robot the duck had arrived (near the 9.5 cm pickup spot), so the planner reports a failure where it succeeded.
- Straight walking drifts left (run 3).
