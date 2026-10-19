# Foot sole material, 2026-10-18

Official reference: Pollen's maintainer notes that the real Microduck has a grippy polyurethane (PU) sole, while training randomised foot friction mu 0.7-1.3 (`microduck_rl/scripts/infer_policy.py`, `--foot-friction` help, commit bfc0ed4, 2026-08-19). The runtime repo (pollen-robotics/microduck, read 2026-10-18) names no sole material.

Our feet ran on sandpaper over 0.5 mm 3M foam tape. RoboClub had only a 2 mm rubber roll. Withdrawn: the 0.8 mm 60A silicone sheet (Amazon B0B4X47N19, $14.99) was removed from the cart at the user's word; nothing ordered. Free routes first: sandpaper stuck directly with thin double-sided tape (no foam layer), or a discarded bicycle inner tube (butyl, about 1 mm) ([offers](offers.json), [authorization](authorization.json)).

Cutting: the foot bottom is flat over about 39 x 27 mm and curves up 1 mm at 3.5 mm and 2 mm at 5 mm past that edge. In the sim walk at 0.25 m/s about 90 % of floor contacts sit on that curve (foot tilt 6-10 deg), so the sheet covers it: trace the foot outline on the backing paper, cut 2 mm inside the line (about 50 x 38 mm), cut a small V notch at each corner, and press the sheet over the curve from the centre outward.

Contact study: `~/.venvs/microduck-dyn/bin/python sole_contact_sim.py out.json --model asbuilt --volt 7.0 --drop 0.04 --vx 0.25 --walk-at 4 --walk-s 6` (walk61, as-built model, real-condition flags). Standing: all contacts on the flat. Walking: contact height above the flat median 0.5 mm, 90th percentile 1.1 mm, max 1.6 mm.

## Sole stack thickness (sim sweep, 2026-10-19)

`sole_thickness_sim.py` inflates the two foot collision meshes by the stack thickness (MuJoCo geom margin, rigid; compliance not modelled) and runs walk61 at 0.25 m/s for 8 s on the as-built model with the real-condition flags (`--volt 7.0 --drop 0.04`). Results in `sweep/summary.json` (27 runs):

- 0, 1, 1.5, 2, 3 and 5 mm, seeds 0-2: no fall in any run; max tilt 5.3-6.7 deg, distance 0.83-0.90 m, max action 0.85-0.91 rad, all within the seed-to-seed spread at 0 mm. Swing-foot peak height is 6 mm (median) and 13.5 mm (90th pct) at every thickness: both feet carry the same sole, so the stance leg lifts the body by exactly what the swing sole adds, and clearance does not change.
- Head board sensitivity: the as-built manifest still carries the Pi Zero W (add-pi, 9.3 g) although the A7Z is fitted since 2026-10-17. Radxa publishes no mass for the Cubie A7Z (product brief section 7 gives dimensions only; docs and ameridroid pages likewise, read 2026-10-19), so the A7Z is bounded as 15 g and 20 g at the Pi position (`--head-swap --a7z-g`): 0, 1 and 2 mm soles still never fall, tilt 6.2-6.7 deg. Open item: weigh the A7Z (with header) when the head is next open and replace add-pi in `asbuilt-mass-manifest.json`.
- The 0.15 m/s rows stand in place (walk61 does not step at that command) and test nothing.

Conclusion: a 1 mm inner tube plus thin tape needs no foot reprint. Up to 5 mm the sim shows no change, so the practical limit is set by the wrap, not by walking: a sheet thicker than about 1.5 mm will not follow the 7-8 mm edge radius and lifts at the edges. Tape: use a thin PET-carrier or transfer double-sided tape (0.05-0.2 mm); the 0.5 mm foam tape used until 2026-10-18 adds a layer whose shear compliance is 10-50 x that of the rubber (acrylic foam G about 0.1 MPa, 50A butyl about 0.8 MPa) and whose edges peel.

Template: `sole_template.py` writes `sole_template.pdf` (US Letter, 1:1, 50 mm check bar, two per foot): the printed foot bottom developed to 2.5 mm above the flat (sim contacts reach 1.6 mm), with a 2 mm dart at each corner (the doubly curved corner has about 1 mm of excess at 45 deg of tilt).

Official sole (read 2026-10-19): Open Duck Mini v2, the open design the MicroDuck descends from, prints `foot_bottom_tpu.stl` in TPU at 40 % infill and screws it to `foot_bottom_pla.stl` with two M3x6 (apirrone/Open_Duck_Mini v2 print guide and assembly guide); the BOM lists no separate rubber or foam. Pollen's public CAD shows the sole as a separate yellow layer under the foot body (teardown article, 2026); Pollen's training script calls the real sole "grippy polyurethane" (microduck_rl infer_policy.py). A TPU print is a polyurethane; whether Pollen's kit part is printed or moulded is not published. Printed TPU is slippery when hard (95A) and printed smooth; the grip of a PU sole comes from low hardness (30-60A) and a matte surface, which is why cast PU (two-part urethane rubber poured in a printed mould) or sheet rubber grips better than a 95A print.

## Hot-glue sole cast in a printed mold (2026-10-19)

RoboClub's suggestion: coat the foot bottom with hot glue. Free-hand coats grip well but the face is uneven. `hotglue_mold.py` builds a mold from each foot STL (left `community-44`, right `community-46`):

- Cavity floor = the foot bottom offset outward by 1.0 mm along the normal, up to 3.0 mm above the flat (sim contacts reach 1.6 mm), so the coat is 1.0 mm on the flat and on the curve.
- Outside the cavity the mold top is the plane 3.0 mm up; the foot touches it along its 3 mm contour, which closes the cavity and gives the coat a clean edge.
- Three 1.6 mm pins with their tops on the flat set the depth (the coat gets three 1.6 mm dimples).
- Eight vents lead excess glue to a moat; four tabs 0.4 mm outside the foot outline, up to 12 mm, centre the foot.
- Check (`hotglue-mold-check.json`): no overlap between the seated foot and the mold, coat 1.00 mm along the normal (1st-99th pct), coat volume 1.95 cm3 per foot, cavity 1859 mm2.
- Print: `mold-process.json` (X1C) and `mold-process-elegoo.json` (Centauri Carbon), PLA, 0.12 mm layers, top ironing, 4 walls, 40 % infill, no supports, cavity up. Both molds on one plate: X1C 2 h 17 min / 38.9 g, Centauri 2 h 29 min / 36.8 g. Delivery: `~/Downloads/zhengxia_MicroDuck_HotGlueMold/` (X1C, Elegoo, STL, preview).

Process:
1. Peel the old glue off. Sand the foot bottom with the sandpaper, wipe with alcohol. Nothing oily on the foot.
2. Wipe a thin film of release on the mold cavity, pins and top (cooking oil or petroleum jelly; silicone spray if at hand), wipe off the excess so no pools stay. A paper or PTFE liner wrinkles in the curved cavity and prints the wrinkles into the sole.
3. Fill the cavity from the centre outward with the glue gun, about 2.5 cm3 (6 cm of 11 mm stick), slightly domed.
4. If the surface has skinned, warm the glue in the mold with the heat gun on low, nozzle moving 5 cm away, a few seconds, until it is glossy again. Heat the glue in the mold, never the foot: PLA softens at about 60 C.
5. Press the foot straight down between the tabs until it stops on the pins; hold one minute. Glue should come out of the vents.
6. Cool 5 minutes (2 minutes in a freezer helps release), lift the foot out, trim the vent flash and the edge with a knife.

A PLA mold softens where hot air hits its bare surfaces; if it deforms after a few casts, print it again or print it in PETG.
