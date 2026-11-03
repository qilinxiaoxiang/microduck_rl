Hey team — let’s build the full MicroDuck fetch demo: call Mochi → it looks at you → ask for the orange ball → it approaches while watching → picks it up → returns to the same person → waits for you to take it.

[Starting materials: code, models, samples and setup](https://github.com/qilinxiaoxiang/microduck_rl/blob/main/cmu/planning/fetch-20261103/README.md)

Five tasks are open for claiming:

- [A · Voice and caller identity](https://github.com/qilinxiaoxiang/microduck_rl/issues/2) — call/fetch/cancel from one shared audio input. Difficulty 3/5.
- [B · Continuous vision and approach](https://github.com/qilinxiaoxiang/microduck_rl/issues/3) — track the selected ball/person and request gaze/speed. Difficulty 4/5.
- [C · Head-aware walking](https://github.com/qilinxiaoxiang/microduck_rl/issues/4) — walk, turn and stop while looking down, then carry. Difficulty 5/5.
- [D · Pick and rise](https://github.com/qilinxiaoxiang/microduck_rl/issues/5) — visual pickup through a stable loaded handoff. Difficulty 5/5.
- [E · Full fetch execution](https://github.com/qilinxiaoxiang/microduck_rl/issues/6) — connect the skills, return, wait for take and handle failures. Difficulty 4/5.

All five start unassigned. Comment on an Issue with the task or small increment you want to claim, your availability and first deliverable. Tasks can be shared or handed over: leave the branch, runnable command, current result and next step on the same Issue.

A/B/E software and C/D historical CPU replay can start now. New-body training waits for the matched backpack scan/model. I’ll coordinate physical access and measurements.

Can we meet for 30 minutes this week to run the starter and choose increments? Please reply with availability and any compute you can contribute. Aim for a reproducible checkpoint in one week, then joint testing in week two if the physical prerequisites are ready.
