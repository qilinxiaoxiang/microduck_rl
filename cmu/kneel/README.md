# cmu/kneel: dock + kneel-rise (the split pick, 2026-11-18)

The fetch pick is split in two (plan: `PLAN.md`):

1. **Dock (closed loop)**: from wherever the approach leaves the robot (ball 10–35 cm ahead, ±45°), stand still
   with the ball at the static pose's landing spot (`OBJECT_NOMINAL_XY` = 8.7 cm ahead of the foot-frame origin,
   centred). Task `Mjlab-BallDock-Walk61-MicroDuck-CMU-<body>`, `src/mjlab_microduck/tasks/cmu_ball_dock_env_cfg.py`.
2. **Kneel-rise (open loop, blind)**: from that stance kneel, bite, rise and keep the ball; no camera input, only
   the jaw's holding bit. Task `Mjlab-KneelRise-FromRise-MicroDuck-CMU-<body>`,
   `src/mjlab_microduck/tasks/cmu_kneel_rise_env_cfg.py`.

The window the dock has to hit is the window over which kneel-rise succeeds (`kr_eval.py` bins success by the
ball's start offset).

## Files

| File | What |
|---|---|
| `kr_eval.py` | kneel-rise battery: one 6.1 s episode per env; fell / bitten / held at hand-over / held at end / drops after hand-over / end rms and tilt / neck peak while held / success by start offset; `w3_lat55` = issue 5's 55 ms latency |
| `dock_check.py` | dock env checks: `walk` / `stand` (walk61 inside the commander env), `pctrl` (scripted P-controller on the TRUE ball position, the best case of a scripted dock), `policy` (`--onnx`, the trained commander) |
| `results/` | JSON outputs |
| `runs/` | training stdout logs (checkpoints are in `logs/rsl_rl/<experiment>/`) |

## Commands

```bash
uv run train Mjlab-KneelRise-FromRise-MicroDuck-CMU-Plugged --env.scene.num-envs 4096 --agent.logger tensorboard
uv run train Mjlab-BallDock-Walk61-MicroDuck-CMU-Plugged --env.scene.num-envs 4096 --agent.logger tensorboard
uv run scripts/export.py <task> --checkpoint-file logs/rsl_rl/<exp>/<run>/model_N.pt     # writes ./output.onnx
uv run python cmu/kneel/kr_eval.py --onnx <kneel.onnx> --cases w1,w3,w5,w3_lat55 --num-envs 512
uv run python cmu/kneel/dock_check.py --mode policy --onnx <dock.onnx> --num-envs 256
uv run play <task> --checkpoint-file <model_N.pt>                                        # watch it
```

Note on deploying the blind kneel-rise: the runtime must feed **zeros** in body slots 55–58 and 60 (u, v, width,
age, visible) and the delayed pinch bit in 59. The first-layer columns of those slots were zeroed at init, but the
normaliser means are non-zero, so a runtime that feeds the camera there would not be "blind".
