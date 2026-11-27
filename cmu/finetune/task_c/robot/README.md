# Fetch C: ball approach

Finds the orange ball with the head camera, walks to it and stops about 12 cm away (from the body centre), facing it.
Walking policy: Shawn's S2 b2 (`cmu/whole/runtime-overlay/models-bw-s2c/bw-s2c-b2.onnx`) in the walk and stand
slots, with the robot's normal brain. Issue #4.

## Files

- `run_approach.py`: the launcher (run this).
- `taskc_approach.py`: the approach loop: steering and stop rule, head follows the ball, start scan, body-turn
  search, close-ball fix, final line-up.

## Run on the robot

Copy both files into the team folder (next to `approach.py` and `ball_detect.py`), start the brain as usual, then:

    ~/.venvs/duck-policy/bin/python run_approach.py --final --hold-s 5

`--final` turns on the tested setting: stop at 12 cm, head tilt at most 1.30 rad, head follows the ball, start
scan, body-turn search, close-ball fix, final line-up.

## Simulator result

918 g battery body, 1.3 cm ball, balls 4-50 cm away, 40 fixed cases (front, beside, behind):

- 36 of 40 reached (front 22 of 24, beside 9 of 10, behind 5 of 6), 0 falls.
- Stops 10-17 cm from the ball, every angle within 7.5 deg.

## Not done yet

- Not tested on the real robot.
- Balls touching the feet are not found.
- If the ball is moved out of view after the robot has seen it, the robot goes to where it last saw the ball and
  does not search again.
