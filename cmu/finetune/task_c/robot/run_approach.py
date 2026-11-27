"""Task C approach launcher (packed from ball_loop_team.py, commit 9db2fb9a; the loop is taskc_approach.py).

    ~/.venvs/duck-policy/bin/python run_approach.py --final --hold-s 5

Put this file and taskc_approach.py in the team folder on the robot (beside approach.py, ball_detect.py).
The text below is ball_loop_team.py's.
Run ball_loop_v2.py (the robot loop) with the team's ball detector instead of the plain colour filter.

The team's ball_detect.BallDetect (Shawn, 2026-11-12, used by seepick.py): the orange colour filter proposes, the
learned detector (learned_target, prompt "small orange ball") confirms; a confirmed ball is then tracked by the colour
filter at camera rate. Written because the colour filter alone took the robot's own foot, a far wall and the floor for
the ball. On 2026-11-20 our walk61-approach run saw "a ball" 0.9-1.0 m away and 55-82 deg left 6 times while the real
ball lay about 0.55 m straight ahead.

Nothing in ball_loop_v2.py or the team's files changes: this file replaces orange_target.detect in memory with a
BallDetect instance before running ball_loop_v2.py's main. Same options as ball_loop_v2.py, e.g.
    ~/.venvs/duck-policy/bin/python ball_loop_team.py --rule walker --no-look --hold-s 5
The learned pass costs about 1 s per picture on the A7Z; while it runs the loop does not send new commands.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import taskc_approach  # noqa: E402,F401  (registers goal_inputs, ball_loop, ... from the packed file)

# --final (2026-11-26): exactly the setting of the simulator baseline (random_check.py tag "all12": 40 cases, 4-50 cm,
# 1.3 cm ball, 918 g body, S2 b2 in the walk and stand slots, stock brain: reached 22/9/5 of 24/10/6, 0 falls; balls
# from 20 cm on end 10-17 cm from the body centre, every angle within 7.5 deg). Stop 12 cm, head pitch cap 1.30.
# Robot side: the brain is the stock one (walk_wired.py / duck-run.sh, NOT brain_taskc.py) with bw-s2c-b2 in both slots.
FINAL = ["--background", "--rule", "straight", "--no-look", "--odo-gain", "0.35", "--head-servo", "--stop-d", "0.12", "--max-head-pitch", "1.30",
         "--plus", "steer_max=0.3,slow_gap=0.15,early=0.04,near_d=0.15,arrive_b=0.15", "--body-search", "--final-align",
         "--near-view", "--any-brain"]
if "--final" in sys.argv:
    i = sys.argv.index("--final")
    sys.argv[i:i + 1] = FINAL
    print("ball_loop_team: --final = " + " ".join(FINAL), flush=True)
PRINT_CONFIG = "--print-config" in sys.argv     # laptop check: set everything up, print it, exit before the loop
if PRINT_CONFIG:
    sys.argv.remove("--print-config")
import ball_detect  # noqa: E402  (the team's; imports learned_target and orange_target)
import orange_target  # noqa: E402

_colour = orange_target.detect
_bd = {}

# --background (2026-11-21): ball_detect_bg.py, the learned pass in a worker thread and colour tracking only near the
# confirmed ball, so the loop keeps the camera's rate (robot 2026-11-20: one picture per 1.0-1.3 s with BallDetect).
BACKGROUND = "--background" in sys.argv
if BACKGROUND:
    sys.argv.remove("--background")


def detect(frame, hsv_lo=orange_target.HSV_LO, hsv_hi=orange_target.HSV_HI, min_area=None, mask=False, accept=None,
           **kw):
    if "bd" not in _bd:
        ma = min_area if min_area is not None else 60
        if BACKGROUND:
            import ball_detect_bg
            _bd["bd"] = ball_detect_bg.BackgroundBallDetect(hsv_lo, hsv_hi, ma, learned=ball_detect.learned_target.detect,
                                                            colour=_colour, log=lambda s: print(s, flush=True))
        else:
            _bd["bd"] = ball_detect.BallDetect(hsv_lo, hsv_hi, ma, colour=_colour)
        print(f"ball_loop_team: team detector on (colour + learned{', learned in the background' if BACKGROUND else ''}, "
              f"{ball_detect.learned_target.MODEL})", flush=True)
    return _bd["bd"](frame, accept=accept)


orange_target.detect = detect


@__import__("atexit").register
def _stats():
    if "bd" in _bd:
        print(f"ball_loop_team: detector counts {_bd['bd'].stats}", flush=True)

# Head-down limit (2026-11-20, run 19:11:50 with walk61-approach): close to the ball the loop asked head pitch 1.30
# (measured 1.34-1.35 rad) while walking at 0.25 m/s; outputs jumped to 1.4-1.9 rad and it fell. walk61-approach was
# fitted up to about 1.25, and on the robot it was already shakier at 1.25 after stopping. --max-head-pitch caps the
# absolute head pitch the loop asks for (default 1.15 rad); a ball very close may then leave the bottom of the picture,
# and the loop walks to its last known place.
MAX_HEAD_PITCH = 1.15
if "--max-head-pitch" in sys.argv:
    i = sys.argv.index("--max-head-pitch")
    MAX_HEAD_PITCH = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
import goal_inputs  # noqa: E402
import approach  # noqa: E402  (the team's; its start scan changed on 2026-11-22 to straight-ahead poses only)
import scan_poses  # noqa: E402
scan_poses.install(approach)        # our loop scans once without body turns: keep the head-yaw poses

# --odo-gain (2026-11-21): the rule's distance factor, used while the ball is not seen (distance walked / speed input x
# time). walker rule: 0.5 measured for t3b, walk61-approach about 0.28; rule straight with r6 / r8 about 0.35.
ODO = None
if "--odo-gain" in sys.argv:
    i = sys.argv.index("--odo-gain")
    ODO = float(sys.argv[i + 1])
    import walker_rule
    walker_rule.ODO_GAIN = ODO
    del sys.argv[i:i + 2]
    print(f"ball_loop_team: distance factor {ODO}", flush=True)

# --rule straight (2026-11-21): goal_inputs.StraightEncoder, the input rule recipes r6 / r8 were trained with (no
# sideways input, aim within 0.05 rad, walk 0.25-0.30 m/s with a gentle steer, re-aim beyond 0.5 rad). ball_loop_v2.py
# does not know it, so this file runs ball_loop_v2's main with its "trained" slot handing out a StraightEncoder.
# For r8, use the stock brain's heading hold against the robot's left drift (simulator: path within 5-8 cm of the
# line with it, 16-20 cm without): start the brain with walk_wired.py / duck-run.sh, and add --any-brain here.
# Stop distance for r8: pass --stop-d 0.20 (ball_loop_v2's option). At its default 0.145 the camera loses the ball at
# 13-15 cm and the robot creeps on to 3-12 cm (simulator, 2026-11-21); at 0.20 the ball stays in the picture.
# --head-servo (2026-11-21, with --rule straight): head_servo.py tilts the head so the ball sits in the middle of the
# picture (simulator, r8: the ball sat 11-18 deg below the centre on average; with the servo 3-6 deg in most cases).
HEAD_SERVO = "--head-servo" in sys.argv
if HEAD_SERVO:
    sys.argv.remove("--head-servo")
    import head_servo
    head_servo.PITCH_MAX = MAX_HEAD_PITCH

# --plus K=V,... (2026-11-22, with --rule straight): loop options of goal_inputs.StraightPlus. The tested setting ("v5",
# random_check.py, 24 front cases: OK 16 of 24, median 7.6 deg / 2.7 cm vs 10 of 24 before):
#     --plus steer_max=0.3,slow_gap=0.15,early=0.04,near_d=0.15
# --body-search: body_search.py turns the body when the start scan finds nothing (beside 10 of 10, behind 5 of 6 found).
PLUS = None
if "--plus" in sys.argv:
    i = sys.argv.index("--plus")
    PLUS = sys.argv[i + 1]
    del sys.argv[i:i + 2]
BODY_SEARCH = "--body-search" in sys.argv
if BODY_SEARCH:
    sys.argv.remove("--body-search")
# --final-align (2026-11-26): final_align.py turns in place to face the ball after the stop (front 15-50 cm: over 8 deg
# 5 -> 1 of 19). --near-view: near_view.py, the horizon test that works in the lowest head look (balls 8-11 cm found).
FINAL_ALIGN = "--final-align" in sys.argv
if FINAL_ALIGN:
    sys.argv.remove("--final-align")
NEAR_VIEW = "--near-view" in sys.argv
if NEAR_VIEW:
    sys.argv.remove("--near-view")
    import near_view
    near_view.install(approach)
    print("ball_loop_team: near view on (horizon test by the ray, balls close to the feet)", flush=True)

STRAIGHT = "--rule" in sys.argv and sys.argv[sys.argv.index("--rule") + 1] == "straight"
if STRAIGHT:
    sys.argv[sys.argv.index("--rule") + 1] = "trained"

_cap = MAX_HEAD_PITCH - 0.349          # as a change from the default head pitch (ball_loop.DEFAULT_HEAD)
_pitch_delta = goal_inputs.GoalEncoder.pitch_delta
goal_inputs.GoalEncoder.pitch_delta = lambda self, d: min(_pitch_delta(self, d), _cap)
print(f"ball_loop_team: head pitch asked at most {MAX_HEAD_PITCH:.2f} rad", flush=True)

# 2026-11-26: the loop is ball_loop.py, the same file the simulator runs (sim_check.py imports it). Was ball_loop_v2.py,
# a robot-only copy that is not in the repo.
sys.argv[0] = os.path.join(HERE, "run_approach.py")
if not STRAIGHT:
    sys.modules["ball_loop"].main()
else:
    blv2 = sys.modules["ball_loop"]     # the packed ball_loop (taskc_approach.py)
    _make_rule = blv2.make_rule

    def make_rule(name):
        if name != "trained":
            return _make_rule(name)
        if PLUS:
            for kv in PLUS.split(","):
                k, v = kv.split("=")
                if k == "arrive_b":       # the arrival window's bearing (rad) and, 0.15 rad wider, the leave window
                    goal_inputs.S_ARRIVE = (goal_inputs.S_ARRIVE[0], float(v))
                    goal_inputs.S_LEAVE = (goal_inputs.S_LEAVE[0], float(v) + 0.15)
                else:
                    goal_inputs.PLUS[k] = float(v)
            enc = goal_inputs.StraightPlus()
        else:
            enc = goal_inputs.StraightEncoder()
        if ODO is not None:
            enc.odo_gain = ODO          # ball_loop uses the rule's own odo_gain while the ball is not seen
        print("ball_loop_team: input rule straight (goal_inputs.StraightEncoder)", flush=True)
        return enc
    blv2.make_rule = make_rule
    if BODY_SEARCH:
        import body_search
        body_search.install(blv2.BallLoop, log=lambda m: print(m, flush=True))
        print("ball_loop_team: body search on (start scan found nothing -> turn the body 90 deg, scan again)", flush=True)
    if HEAD_SERVO:
        head_servo.install(blv2.BallLoop)
        print(f"ball_loop_team: head servo on (tilt only, head pitch at most {head_servo.PITCH_MAX:.2f} rad)", flush=True)
    if FINAL_ALIGN:
        import final_align
        final_align.install(blv2.BallLoop, log=lambda m: print(m, flush=True))
        print("ball_loop_team: final line-up on (turn in place to face the ball after the stop)", flush=True)
    if PRINT_CONFIG:
        import body_search as _bs
        enc = blv2.make_rule("trained")
        print("CONFIG", dict(argv=sys.argv[1:], S_ARRIVE=goal_inputs.S_ARRIVE, S_LEAVE=goal_inputs.S_LEAVE,
                             PLUS=goal_inputs.PLUS, encoder=type(enc).__name__, odo_gain=getattr(enc, "odo_gain", None),
                             scan=approach.SCAN, dwell=approach.SCAN_DWELL_S, below_horizon=approach.below_horizon.__module__,
                             body_search_turns=_bs.TURNS if BODY_SEARCH else None,
                             head_pitch_max=head_servo.PITCH_MAX if HEAD_SERVO else None), flush=True)
        sys.exit(0)
    blv2.main()
