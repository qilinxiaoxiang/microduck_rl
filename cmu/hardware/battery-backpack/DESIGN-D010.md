# MicroDuck battery backpack (D-010)

Status 2026-10-17: the switch in hand is a Swann 39 (SPST, 15 A 125 VAC). The hole is now its 12.3 x 27.5 mm cutout and the box 0.7 mm deeper than V1.18 (frame and cap rebuilt; they print in V1.20 with the unprinted V1.19 rear lid). Status 2026-10-15: print geometry built at the Tattu listing size (57 x 31 x 12 mm) plus clearance. Shawn kept the upright battery and released it: the frame and cap are the whole V1.18 release, one orange plate of about 2 h (`guide/print/release-V1.18/`; V1.17 is printing), and the assembly guide has a chapter 11 for the backpack and the swap. Renders: `renders/BP-sheet.png` (six views); why the battery stands upright: `renders/lead-bend-explainer.png`.

## Change from the two drafts

The battery now stands upright. In both drafts it lay across the box, but its leads leave the 31 x 12 mm end, and only 2.4 mm was left there to bend them. Standing up, the leads leave the top into a 7 mm space under the box roof. The box gets deeper: 27.45 mm behind the front plate (drafts 24.0, bench box 20.1). It keeps the bench box width, top height and straight step, so nothing grows above the shells.

## How it sits on the robot

The backpack bolts on where the bench box does: the same front plate outline and the same four fixing holes, M2 x 6 into the pack posts. Two parts are printed: the frame (front plate, walls, holders) and the cap (the rounded back).

## What is inside and how each part is held

- **Switch (Swann 39, the RoboClub switch).** Rocker down, flush in a pocket in the box floor, under the narrow lower part. You press it with a finger from below. Its body goes up through a 12.3 x 27.5 mm hole in a 3 mm floor panel, and its snap wings grip the panel. The switch has bare solder lugs, so it is wired as a short switch lead (BATTERY-WIRING.html). Sources: the switch markings (swan logo, "39", 15A 125V AC, T85, Singapore); Swann's catalogue cutout for the same family, 27.4 x 12.3 mm ([Globelink Swann catalogue](https://www.globelink.co.nz/wp-content/uploads/product_images/ABMProductDocuments/Switching%20&%20Controls/Swann%20Rockers%20&%20Indicators/Swann%20Rocker%20&%20Indicator%20Catalogue.pdf), series 40 page); a Swann 39 listing gives 27.3 x 12.2 mm and a 30 x 13.8 mm flange ([eBay 296007287322](https://www.ebay.com/itm/296007287322), [eBay 125944093531](https://www.ebay.com/itm/125944093531)); Shawn measured the body 11.5 mm thick (2026-10-17). By photo scale the rocker and flange stand about 8 to 9 mm proud, so the rocker sits within about 0.5 mm of the floor face in the 8.49 mm pocket; the lugs end about 21 mm above the panel, inside the REV envelope the CAD checks use. The pocket is 14.4 mm wide (flange + 0.3 a side). The battery pocket starts 0.3 mm behind the measured 11.5 mm body, so the box is 0.7 mm deeper than V1.18 and the cap moves back with it.
- **Battery.** Stands upright behind the switch, on the floor panel and a shelf. Two fences on the cap keep it centred. A foam pad stuck inside the cap presses it forward.
- **DFR1202 5 V board.** Stands against the inside of the front plate with its power pads at the bottom edge. Two ribs hold its back 2.2 mm off the plate, two pairs of flexible lips clip its two pad-free long edges, and one small stop sits under its VIN end. The pads stay open from below and from behind. A 2 mm foam pad on its big inductor rests against the battery.
- **Cap.** Three M2 x 8 screws go in from the back through holes in the cap into the frame: two high on the sides and one low in the middle. No latches, no hooks.

## Plugs, in plain words

- **Plug 1: battery to switch (XT30).** It hangs loose on the robot's right side of the battery, in the space between the battery and the side wall. The balance plug sits at the top on the same side and stays on the battery.
- **Plug 2: switch to the 5 V board feed (XT30).** It is the mirror of plug 1, on the robot's left side. It stays plugged. It exists only so the switch can be replaced.
- **Plug 3: servo swap plug (XT30).** It stands in the empty pack channel in front of the box, behind the OpenRB, with the robot half on top. The robot half is the FIT0586 XT30 male lead from OpenRB J4.
- **Plug 4: A7Z swap plug (MR30).** It stands beside plug 3 in the same channel. The robot half is the MR30 male on the A7Z lead from the neck.

The backpack halves of plugs 3 and 4 come out of a window in the front plate under the 5 V board. The robot does not need a printed holder for its two plug halves: both stand in the channel on their own leads.

## Charging

Take out the three cap screws, take off the cap, slide the battery back out of the box and unplug plug 1 outside the box. The battery is never charged on the robot.

## Swapping to the bench box

With the cap and battery out, the two upper fixing screws are reached straight through the empty battery space. The two lower ones sit under the box floor and are reached from behind, under the box. Pull the box about 4 cm back, unplug plugs 3 and 4, plug in the other box and screw it on. The shells stay closed.

## Mass and balance (`bp_mass.json`, method of `layout-F/mass_estimate.py`)

The backpack adds about 91 g (printed frame 16 g, cap 8 g, battery 43 g) and moves the whole-robot balance point 6.5 mm back. The bench box adds 57 g and moves it 3.9 mm back. Only the battery mass is a listed value; the switch, board, plug and wire masses are estimates.

## Checks (2026-10-15, listing-size geometry)

- **Fit** (`build/build.json`): the listing battery sits 0.5 mm inside the cavity at its closest point, with a 7.3 mm lead space above it. Battery, plugs and board do not touch any wall. The only contacts are designed: the switch snap wings in the panel hole (31.9 mm3) and the four clip lips on the board's long edges (7.8 mm3).
- **Assembly** (`bp_checks.json`): the battery slides out the back and the cap goes on without touching anything. The switch goes up from below and the DFR1202 in from behind, each touching only its panel or its clip lips.
- **Outline:** nothing behind the front plate pokes outside the box outline. No wire centreline passes through a wall.
- **Screws:** each cap screw gets 5.0 mm of thread in the frame, with a clear head seat and a clear 4 mm driver lane from the cap back. All four fixing screws have a clear 4 mm driver lane: the upper two through the empty battery space, the lower two under the box floor past every robot part.
- **Hip sweep** (`bp_hip_sweep.json`, hip pitch ±30°, roll ±10°): no leg part comes within 6 mm of the box, the plugs or the wires.
- **Print** (`bp_checks.json` printPoses): both parts are single watertight solids. The frame prints rim-down like the bench box frame: 713 mm2 bed contact, 22.05 mm tall, tree support under the front plate. The cap prints on its back face: 2691 mm2 contact, 7.4 mm tall.

## Open

- Measure the battery. The pocket is 58 x 32 x 13 mm; the cap foam pad takes up the last 1 mm behind the pack.
- Screws needed: 4 x M2 x 6 (fixing, same as the bench box) and 3 x M2 x 8 (cap). The cap's screw-head floors are 3.0 mm (V1.19; 1.0 mm in V1.18, the same detail that tore on the V1.17 box cap).

## Files

- `build_bp.py`: print geometry. Writes `build/` (frame, cap, bought parts, `wires.json`, `build.json` with fit and overlap checks).
- `render_bp.py`: the render sheet. `bp_mass.py`: mass and balance.
- `bp_checks.py`: print poses, assembly paths, cap path, outline, wires and screws (`bp_checks.json`, `build/*-print.stl`). `bp_hip_sweep.py`: the hip sweep (dyn venv).
- `to_guide_world.py`: copies the frame, cap and bought parts into `guide/print/world/` (`guide-world.json`). In the guide data the backpack and the rear converter box are two `alternates`: the print gate skips part pairs and cables across them, and the guide's swap step removes the box.
- `release_rows.py`: the release-named STLs `battery_backpack_{frame,cap}-{world,print}.stl` and `release-rows-orange.json`, the two Orange rows for the next release manifest.
- `concepts.py`, `render_concepts.py`, `concept_mass.py`: the two earlier drafts (superseded).
- `probe_front.py`, `probe_channel.py`: free space behind the robot and in the pack channel.
- `parts/rev_31_1387_switch-local.stl`: switch envelope from REV's STEP file (same flange as the Swann 39, deeper body; the Swann's 11.5 mm body thickness is applied in build_bp.py).
