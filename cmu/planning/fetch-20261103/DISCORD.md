Hey everyone — I’d like us to build one complete MicroDuck demo:

**Call “Mochi” → it looks at you → ask for the orange ball → it approaches while watching → picks it up → carries it back to the same person → waits for you to take it from its mouth.**

I’ve prepared five **claimable tasks**, with starting code/data/models, tested local commands, suggested steps and acceptance criteria in the attached task pack:

- **A · Interaction:** one shared audio path, call/fetch/cancel events, and keeping track of who asked. Start with the supplied transcript parser; extend to local audio replay. Difficulty 3/5.
- **B · Vision and approach:** track the selected ball/person, handle lost views, and request gaze/speed. Start with the detector/images and its known false positive. Difficulty 4/5.
- **C · Head-aware walking:** walk/turn/stop while genuinely looking down, then extend to carrying. Start with supplied policy exports and the actual head-down failure trace. Difficulty 5/5.
- **D · Pick and rise:** maintain vision through the bite and hand over in a stable loaded pose. Start with the recorded pick→stand failure and model replay. Difficulty 5/5.
- **E · Full fetch execution:** connect the modules, return to the original person, wait for take, and handle cancel/drop/loss. Start with the runnable full-sequence rehearsal. Difficulty 4/5.

These aren’t permanent assignments. Please claim a task **or one small increment**, mention your available time, and leave a runnable branch so someone else can take over. There are five of us including me; I’ll coordinate physical access and the new-backpack measurements.

A/B/E software and C/D historical CPU replay can start now. Training for the changed body waits for the matched scan/model. We’ll use one agreed interface and one motor executor, and test adjacent skills without resetting the robot between them.

Could we have a **30-minute meeting this week** to run the starter and choose increments? Please reply with availability. Let’s aim for a useful, reproducible checkpoint in **one week**, and joint testing in **week two if the physical prerequisites are ready**. Complete autonomous fetching is not guaranteed in that window.

Open `index.html` in the attached package for the cards; begin with `starter/README.md`. No robot, paid cloud account or private credentials are required for the first local checks. If you have AWS or other compute available, please mention it—we’ll allocate specific pinned comparisons after a short throughput/cost check.
