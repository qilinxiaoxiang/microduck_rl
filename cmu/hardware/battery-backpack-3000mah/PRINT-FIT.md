# Print fit practice (FDM PLA)

Size every printed fit from this file before export: screw pilots, clearance holes, and every male feature that enters a female one. The numbers live in `print_fit.py` (`pilot(screw, orientation, kind, printer)`, `clear(screw, orientation, printer)`, `pilot_depth(screw, clamped)`, `clearance(fit_class, orientation, at_bed)`), and `fit_check.py FITS.json` fails a release whose declared fits are under-toleranced or whose teardrops point the wrong way. Robotics, RIG and MicroDuck part scripts call these; they do not keep their own copies of the numbers.

## Scope

- **Printers.** Bambu X1C, Bambu H2S, Elegoo Centauri Carbon. One table serves all three because the fit-relevant profile settings match; a measured per-printer offset is added on top (see Per-printer offsets).
- **Material.** PLA (Bambu PLA Basic, Elegoo/Generic PLA), 0.4 mm nozzle, 0.2 mm layers, 4 walls.
- **Profile settings the numbers assume.** X-Y hole compensation 0, X-Y contour compensation 0, Bambu auto circle compensation off, elephant-foot compensation 0.1 to 0.15 mm, classic wall generator, 0.42 mm outer wall. A profile that changes any of these needs a test coupon before the table is trusted.
- **Diameters are CAD values.** The slicer reproduces them exactly (RIG 1.3 H2S gcode: a 1.75 pilot gets a 2.165 mm outer-wall path, which is 1.745 + 0.42). All shrinkage happens in the physical print, so the allowance must be in the model.

## Orientation decides the allowance

Read every hole axis in the print pose, never in the assembly frame. `print_fit.orientation(axis, print_rot)` rotates the part-frame axis with the same `print_rot` the STL export uses.

- **Vertical (axis within 30 deg of Z).** The hole is a stack of loops. It prints about 0.1 to 0.2 mm under its CAD diameter in PLA (Hydraraptor "Polyholes"; rigid-prints design guide: add 0.2 mm to PLA holes).
- **Horizontal (axis beyond 30 deg).** Layer steps flatten the sides and the top sags. Add 0.2 mm on the diameter over the vertical value and cut a teardrop.
- **Bambu's own compensation covers vertical circles only.** Its circle compensation says it handles "complete circles on the horizontal plane"; vertical-plane circles are left uncompensated, and the release profiles turn it off anyway.

## Screw pilots (CAD diameter)

Shawn's rule: "screw kind is never prescribed: pilots take machine or thread-forming screws of that size, pilots through or deep enough for common lengths, seats for pan and button heads". A pilot is sized so a machine screw (ISO 7380, DIN 912, ISO 7045) forms its thread without a drill and a thread-forming screw (TAP, PT, ROBOTIS PTS) still bites. The guide gives the size and length ("M2 x 6"), never a kind.

- **Vertical:** M2 1.85, M2.5 2.30, M2.6 2.35, M3 2.80, M5 4.60 (M5 has no bench result yet: Rigid Prints OD - 0.4, first used on the Scout Mini PETG rail blocks 2026-11-13; write the bench result here when they are fitted).
- **Horizontal (teardrop):** M2 2.05, M2.5 2.50, M2.6 2.55, M3 3.00, M5 4.80.
- **Printer offset:** add `PRINTER_OFFSET[printer]` (H2S +0.10, provisional; see Per-printer offsets). `pilot(screw, orientation, printer=...)` does it.
- **Accepted band:** 0.10 under to 0.15 over the table value. Under binds a machine screw; over strips a thread-forming one.
- **Depth.** A blind pilot reaches the longest common length of its size plus 1 mm below the clamped stack: `pilot_depth(screw, clamped)`, with common lengths M2 4 to 10, M2.5 and M2.6 6 to 8, M3 6 to 16, M5 10 to 20. Where the part is thinner than that, the pilot runs through. `fit_check.py` fails a declared `depth` shorter than that.
- **Engagement:** 2.5 d or less over the length that threads; a longer bite in PLA needs more torque than a 2 mm hex or PH0 driver transmits.
- **Head seats:** a counterbore or flat that takes both a pan head (ISO 7045, ISO 14583) and a button head (ISO 7380): M2 4.6 x 1.7, M2.5 and M2.6 5.6 x 1.9, M3 6.3 x 2.1 (`HEAD_SEAT`). The head clearance rule in Access still applies. Where a head bears on a thin or highly loaded printed face, put an ISO 7089 washer under it and seat it at washer OD + 0.4 (M2: washer 5.0 x 0.3, seat 5.4; `WASHER_SEAT`); count the washer in the clamped stack.
- **Pilot mouth on the first layer:** add 0.1 to the diameter or cut a 0.4 x 45 deg mouth chamfer; the first layer flares into the hole even with compensation on.
- **Smallest hole:** 1.5 mm. Rigid-prints puts the reliable minimum at 2 mm; the M2 rows sit under it.
- **Old TAP-only values** (`kind="tap"`: M2 1.7, M2.5 2.2, M2.6 2.1, M3 2.6 vertical) are the MicroDuck as-built record. `fit_check.py` checks old parts against them with a warning; new parts never use them.

## Clearance holes (screw passes freely)

- **Vertical:** M2 2.4, M2.5 2.9, M2.6 3.0, M3 3.4, M4 4.5, M5 5.5, M6 6.6.
- **Horizontal (teardrop):** M2 2.6, M2.5 3.1, M2.6 3.2, M3 3.6, M4 4.7, M5 5.8, M6 6.9.
- Two horizontal holes in two parts that one screw must pass in line (a hinge ear and its lug) take the horizontal value in both parts and a 0.5 x 45 deg lead-in on the entry mouth.

## Per-printer offsets

The H2S joined the club printers on 2026-10-24 and has been the default since 2026-11-01. Two recent club plates needed their M2 pilots drilled: the RIG 1.3 reprints (printed on the H2S, 2026-10-31) and the MicroDuck V1.40 battery pack (2026-11-01/02; an H2S file of that plate exists, the machine it ran on is not recorded). No earlier X1C or Elegoo duck plate did.

- **The slicer did not change.** All Bambu plates (RIG 1.3 H2S, V1.40 battery pack H2S and X1C files, V1.8 X1C of 2026-10-12) were sliced in Bambu Studio 02.08.02.61; the Elegoo plates (V1.6 of 10-09, V1.35 of 10-25) in OrcaSlicer 2.4.2. Every fit setting matches across them: classic wall generator, outer and inner wall 0.42 and 0.45, `xy_hole_compensation` 0, `xy_contour_compensation` 0, `enable_circle_compensation` off (hole_coef 0.23415), `precise_outer_wall` off, `slice_closing_radius` 0.049, `resolution` 0.012, arc fitting on, flow ratio 0.98. Elephant-foot compensation is 0.1 (V1.40 both files, Elegoo) or 0.15 (V1.8 X1C, RIG H2S); it acts on the first layer only.
- **The toolpaths are identical.** `gcode_holes.py` on the V1.40 battery-backpack frame: the H2S and the X1C files give the same outer-wall paths to the thousandth (1.6 CAD pilots: 1.598 hole, open on the first layer from z 0.2 to 6.2; 1.75 pilots: 1.749). The RIG 1.3 H2S plate: 1.747 STL, 1.745 path hole.
- **What differs is physical.** H2S machine, Textured PEI plate, "Generic PLA @BBL H2S" or "Bambu PLA Basic @BBL H2S" filament presets whose differences are cooling, retraction, accelerations, max speeds and max volumetric speed (12 against 21 on the X1C preset). None of them sets extrusion width or flow, so the printed hole on the H2S comes out smaller for a reason the files cannot measure: machine calibration (flow, pressure advance, XY steps), or slower, cooler outer walls laying a wider bead.
- **Countermeasure.** `PRINTER_OFFSET = {"X1C": -0.20, "Elegoo-CC": -0.20, "H2S": +0.10}`, added to every pilot and clearance hole of a plate sliced for that printer. All three values are provisional and marked so in `PRINTER_OFFSET_STATUS`.
- **Bench report 2026-11-21, FlightPack-D1 (X1C Cool Plate and Elegoo CC, PLA 0.2 mm, 100 % infill on Orange).** The plates were designed with the H2S offset, so every M2 pilot was 2.15 (horizontal teardrops: nacelle split pilots, tray pilots, MTF-01 shelf pilots) or 1.95 (vertical): printed on the X1C and Elegoo, the M2 screws went in loose and fell out (Shawn: "说是M2的 结果洞太大了 螺丝直接掉出来"). A 2.15 pilot is wider than an M2 screw's 2.0 major diameter. X1C and Elegoo now carry -0.20 (M2 vertical 1.65, horizontal 1.85), the middle of the as-built record below; confirm with the pilot coupon on the next X1C or Elegoo plate. A FITS.json names its printer at the top level (`"printer": "H2S"`) and `fit_check.py` applies the offset.
- **Acceptance.** Print `coupons/pilot-coupon.stl` (63 x 7 x 6 mm strip, about 2.9 g, small enough to ride on any plate: vertical through pilots 1.5 to 2.2 labelled on the front face, horizontal teardrop pilots 1.7 to 2.3 labelled on the top face, "15" = 1.5) from one .3mf on the H2S and on the X1C. The smallest pilot each printer lets an M2 x 8 machine screw and an M2 x 8 thread-forming screw enter by hand driver gives that printer's offset against the 1.85 table value; write both results here and into `PRINTER_OFFSET` the same day.

## Male and female fits (gap per side, CAD)

Gap per side = (female size - male size) / 2, for round features and for slots, tongues, grooves and ribs alike. The class sets the gap for a vertical female feature away from the bed; the adders below apply on top.

- **Press (pins tapped in):** -0.05 to 0.00. Vertical features only, never on the first layer.
- **Crush ribs:** rib tip -0.10 to -0.15 into the bore. Triangular ribs with a base of 0.8 mm or more, vertical axis only. A wide keel or flat band at that interference is a press fit over a large area, not a rib.
- **Locating (cups over bosses, ribs in grooves, dowels):** 0.15 to 0.20.
- **Sliding (pivot pins, hinges, tongues into slots, lids):** 0.20 to 0.30.
- **Loose (drop-in, bought parts with their own tolerance):** 0.40 to 0.60.
- **Adders.** Female feature horizontal in print: +0.10 per side. Either mating surface in the first 0.4 mm without a 0.4 x 45 deg chamfer: +0.10 per side.
- **Snaps** (a bulb passing a narrower mouth) are designed as two fits: the bulb in the slot behind the mouth is a sliding fit, and the snap interference at the mouth is 0.10 to 0.15 per side with a 45 deg lead-in.
- **Bought parts** (fans, boards, servo holders, cable) carry their own tolerance; take their measured or published size and do not apply print adders to them.

## Teardrops

`cadlib.teardrop(d, p0, axis, length, up)` in the RIG scripts and its MicroDuck equivalents cut the full circle of diameter d plus a 45 deg tip, so a teardrop never narrows the hole; the tip only removes the sagging roof.

- Cut a teardrop on every horizontal hole of 2 mm or more that a screw, pin or cable passes, and on no vertical hole.
- The tip vector is perpendicular to the hole axis and points to +Z of the print pose. Derive it from `print_rot`; a part printed upside down (`((1, 0, 0), 180)`) takes `up = (0, 0, -1)` in its part frame.
- A horizontal male cylinder (standoff, pin, axle) takes the tip pointing to -Z of the print pose, so its underside is the self-supporting face.
- `fit_check.py` rotates both the axis and the tip with the declared pose and fails a tip more than 15 deg off, a tip not perpendicular to the axis, and a teardrop on a vertical axis.
- The teardrop gives no thread in its tip. A horizontal pilot keeps the horizontal pilot diameter; the tip does not count toward it.

## First layer and elephant foot

The first layer spreads outward and pulls holes inward at the bed. Compensation (0.1 or 0.15) removes most of it; up to 0.1 mm per side remains on the Textured PEI plate (Bambu: on snaps, rails and shaft holes "the flared base directly affects fit").

- A male feature that enters a mating part with 0.3 mm per side or less does not touch the bed: print it with that feature up, or put a 0.4 x 45 deg chamfer on its bed edge.
- A female feature whose mouth is on the bed (cup, pilot, groove) gets a 0.4 x 45 deg mouth chamfer, or takes the +0.10 bed adder.
- Fit surfaces start 0.4 mm or more above the bed unless chamfered.
- Every outer bottom edge of a part that sits in a pocket or against a wall gets a 0.4 to 0.8 mm chamfer.

## Declaring fits for the checker

Each release keeps one `FITS.json` next to its part scripts (format in the `fit_check.py` docstring; examples in `examples/`). Declare every pilot with its screw and kind, every clearance hole a screw passes, and every male/female pair with its fit class. `poses` holds each part's `print_rot` exactly as passed to the export. Run `python3 fit_check.py FITS.json` in the release gate; exit 1 blocks the release.

## Access

Every screw must be drivable on a real bench at the moment the build order drives it, with only the parts present then. A screw whose axis passes a wall closer than the numbers below is a design defect, fixed in CAD before the guide is written.

- **Driver corridor.** Free radius around the screw axis, from the seated head top outward, at least 4.0 mm over the first 60 mm. That fits a 3 mm shaft precision driver with 2.5 mm left for wobble, print error and a view of the head. Under 4 mm the hole counts as "against a wall".
- **Handle zone.** Free radius at least 15 mm from 60 to 120 mm along the axis: a 10 mm handle plus the fingertips that turn it (a 20 mm handle keeps 5 mm).
- **Finger start.** At least 8 mm free radius around the head over 10 mm, at the height where the head sits when the tip first enters the pilot (head lifted by the screw length minus the clearance-hole length). Below that the guide must say to start the screw on the driver tip; the design accepts this only where a bought board's own parts cause it.
- **Shawn's rule:** "no screw hole whose axis runs parallel to a nearby wall closer than the driver clearance". The driver clearance is the 4.0 mm corridor above; a wall face parallel to the axis inside it fails, at any height the driver passes.
- **Head seat.** The seated head clears every solid radially; counterbores are at least head diameter + 0.6 mm.
- **Straight only.** The design target is a straight driver along the axis. A tilt is never an instruction; the checker tolerates at most 10 deg for a hex socket and 5 deg for a Phillips head before it calls the screw blocked.
- **No table tricks.** The assembly sits flat on the table with its underside on the surface. Nothing passes below the underside or over the table edge. A horizontal screw needs its axis 15 mm or more above the table for the handle. Sub-assemblies count as held freely only on a bench step of the guide.
- **Build order.** Check each screw with exactly the parts in place when it is driven, and screws loosened in use (tilt friction screws) with the finished assembly at every tilt. Arm links that can be swung away and flexible cables are reported but do not decide.
- **Fixes, in order of preference.** Move the hole so its axis is 4 mm or more from every wall face over 60 mm; lower or notch the wall (a driver slot at least 8 mm wide); turn the screw to come in from the open side; change the fastener (locating pin plus snap, a cup over an existing boss); change the build order or the pose the part is held in. A tilted driver, a stubby driver or an L-key is not a fix.
- **Gate.** `access_check.py SCREWS.json` (format in its docstring) reports, per screw, the free cylinder, the tools that reach straight (3 x 60 mm hex driver with a 20 mm handle, 3 x 100 mm driver with a 10 mm handle, PH0 3 x 50 mm, 2 mm L-key both ways round, a stubby with a 25 mm handle), the tilt the hex driver needs, finger start and a verdict. Exit 1 on any blocked screw; `--strict` also fails awkward ones. `examples/access-selftest/selftest.screws.json` must give OK, awkward and blocked for its three screws.

## Evidence

- **MicroDuck as built (X1C and Elegoo CC, PLA, 0.2 mm, the profile settings above with elephant foot 0.1).** M2 x 6 and x 8 TAP screws drove without drilling into vertical blind pilots of 1.7 mm (A7Z face camera bosses: 4.8 mm bosses, 2.2 mm deep, measured on `microduck-sim/guide/print/face-a7z/face_panel_a7z-print.stl`; eye trim: 5.4 mm deep, `eye_trim_a7z-print.stl`) and 1.6 mm (rear plate OpenRB and 5991 bosses, 4.35 mm engagement, `guide/print/back-plate/build_rear_plate.py`). ROBOTIS M2.6 TAP took 2.0 to 2.2 mm pilots (`guide/content/fasteners.json`, V2.0 head). The 0.3 mm rim gap of the rear converter box cap is the clearance that kept the rim from seating first (D-034). Every MicroDuck M2 screw line in the guide names the screw "TAP". `examples/microduck-as-built.fits.json` (printer X1C) passes with TAP-table warnings.
- **MicroDuck V1.40 battery pack, printed at the club 2026-11-01/02 (H2S file on Textured PEI and an X1C file exist; which machine ran it is not recorded).** The M2 pilots of the printed pack had to be drilled before screws went in. The frame carries 1.6 mm vertical pilots open on the bed (z 0.2 to 6.2) and 1.75 pilots higher up; the H2S and X1C files of the same plate give identical hole paths. The pilots are under the current table (1.85 + 0.10 H2S) and the 1.6 ones also take the bed mouth flare.
- **RIG 1.3 reprint plate (H2S, Bambu PLA Basic, 0.2 mm, Textured PEI, elephant foot 0.15), 2026-10-31, bench report 2026-11-01.** M2 pilots needed drilling, male fitted features were too large at their first-layer ridge, and round parts did not enter teardrop holes. The geometry was exported and sliced exactly: every M2 pilot is 1.747 inscribed in the STL and 1.745 in the gcode, all teardrops point to +Z of their print pose, and the slicer fit settings equal the MicroDuck X1C profile except elephant foot 0.15 against 0.1. What differs from MicroDuck is what the table above now covers: the radar carrier's two M2 pilots are horizontal and round (1.75, no teardrop); the guide line for the Zero, CM3, radar and BMP388 screws reads "M2 x 6（手上有）" without a kind, while MicroDuck's says TAP; every M3 joint drives ISO 7380 machine screws into 2.6 pilots, five of them horizontal teardrops; the fan head's F5 pilots open on the first layer with 7 mm of PTS engagement; the Zero carrier cups are 6.1 over 6.0 bosses (0.05 per side) with their mouths on the bed; the cable-holder clip's tongue (bulb 3.5 in a 3.6 slot) prints its first layer on the bed; the arm and SMPS2DXL plugs carry 0.25 mm radial interference, the SMPS2DXL one over a 48 deg keel band on its bed face. `examples/rig13-as-printed.fits.json` (printer H2S) fails 25 of 30 fits against the any-kind pilots and the H2S offset.
- **RIG 1.3 screw access, bench report 2026-11-01.** The builder could not get the driver into several screws although `access_check_13.py` passed. It passed any tool cylinder with more than 0 mm of air, ignored the table (the arm-bolt driver handle hung 2.5 mm below the tray bottom over the bench edge), never checked the MOSFET, fan, radar-board or BMP388 screws, and had no finger or handle-zone rule. `access_check.py` with the rules above, on the frame B meshes (`Assembly-Guide/access_spec_13.py`, `data/access-real-13.json`), finds: the MOSFET +Y screw (tray boss and mosfet_riser alike) 0.22 mm from the tray's mast-screw block, which overhangs its axis 14 mm above the head; the MOSFET board's +Y screw on the riser 1.86 mm from the next block; the console arm bolts horizontal 7.5 mm above the table under the arm's own leg 5.5 mm above the axis.
- **Sources.** Bambu Studio auto circle contour compensation (wiki.bambulab.com/en/software/bambu-studio/manual/auto-circle-contour-compensation): horizontal-plane circles only. Bambu elephant-foot compensation (wiki.bambulab.com/en/software/bambu-studio/parameter/elephant-foot): first-layer flare affects snaps, rails and shaft holes; measure and compensate. Prusa elephant-foot compensation (help.prusa3d.com/article/elephant-foot-compensation_114487): about 0.2 mm for a 0.4 nozzle. Hydraraptor "Polyholes" (hydraraptor.blogspot.com/2011/02/polyholes.html): printed holes come out undersized from faceting and arc shrinkage. Rigid Prints design guide (rigid-prints.pages.dev/design-guide/holes-bosses-threads, /tolerances-and-fits): pilot = screw OD - 0.4 for self-tapping screws, PLA holes +0.2, minimum hole 2 mm, fit classes press -0.05, location +0.05, sliding +0.15, loose +0.3 per face. Where this file's numbers are larger than Rigid Prints', the MicroDuck and RIG evidence set them; the horizontal +0.2 and the bed +0.1 adders are project inference from that evidence, not a published value.
