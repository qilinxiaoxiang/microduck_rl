# Battery backpack for the 2S 3000 mAh battery

This is the backpack on the robot from 2026-11-22. It carries a Crazepony 2S 3000 mAh LiPo (Amazon B08LZ8L4YD, 110 g)
standing upright in a printed frame, with the old rocker switch clipped into the printed cap. It replaces the
battery-on-the-cap backpack in [../battery-redesign/](../battery-redesign/README.md) (Tattu 2S 650 mAh). The simulator
already uses it: `cmu/model/current.json` names the variant `battery-v2m5-w918-20261123` (since 2026-11-23; before
that `battery-v2m5-20261121`), so every `-CMU-Current` task,
`apply_masses.py --variant current` and `brain_sim_check.py --manifest current` load this backpack.

Two printed parts, frame and cap, screwed to the four posts of the OpenRB box with the four M2 x 6 screws of the old
backpack. The switch, the DFR1202 5 V board and every plug lead come over from the old backpack unchanged. Charging
needs no opening: switch off and plug the battery's white 3-pin balance plug, which sits on top of the pack, into the
balance charger.

## Print it

Print one file, for whichever printer is free. Each file is the whole plate: frame, cap and a small M2 hole test strip,
100 % infill, about 46.5 g of PLA.

- Bambu H2S, textured PEI: [print/H2S/](print/H2S/), 1 h 39 min, Bambu PLA Silk+ preset.
- Bambu X1C, textured PEI: [print/X1C-TexturedPEI/](print/X1C-TexturedPEI/), 1 h 33 min.
- Bambu X1C, Cool Plate (smooth): [print/X1C-CoolPlate/](print/X1C-CoolPlate/), 1 h 32 min.
- Elegoo Centauri Carbon: [print/Elegoo/](print/Elegoo/), 1 h 40 min.

The screw holes are sized for each printer: the X1C and the Elegoo print holes about 0.3 mm larger than the H2S, so
their files have smaller holes (M2 pilot 1.65 mm, clearance 2.2 mm; the H2S file has 1.95 and 2.5). Do not print one
printer's file on another. The X1C and Elegoo files use generic PLA presets; pick your filament in the slicer.
Nothing needs trimming, drilling or support removal. `print/four-files.json` holds each file's sha256.

The test strip has a row of vertical holes marked 15 to 22 (1.5 to 2.2 mm) and a row of horizontal holes marked 17 to
23. Try an M2 x 8 self-tapping screw from the smallest hole up: the smallest one it threads into is what that printer
needs. The fit numbers come from [PRINT-FIT.md](PRINT-FIT.md); tell us your result so it goes into that table.

## Build it

[BATTERY-PACK-BUILD.html](BATTERY-PACK-BUILD.html) is the step-by-step assembly page (Chinese, open it in a browser;
it is self-contained and shows each step in 3D). The first page lists what to print, the one part to buy, what comes
over from the old backpack and the tools. [BOM.json](BOM.json) is the parts list.

## Redesign it

- `stl/H2S-holes/` and `stl/X1C-Elegoo-holes/`: frame and cap in the robot frame (`-world`) and in print orientation
  (`-print`), millimetres, x forward, y left, z up (the guide frame of the robot; see [../README.md](../README.md)).
  `build.json` in each records the build. The two builds differ only in hole sizes.
- `stl/pilot-coupon.stl`: the hole test strip; `cad-source/build_pilot_coupon.py` makes it.
- `envelopes/`: the bought and reused parts the backpack is drawn around (battery, switch, DFR1202 and its leads,
  plugs 1 and 2, balance and DC5.5 plugs, foam pad, tape), in the robot frame.
- [DESIGN-CONTRACT.json](DESIGN-CONTRACT.json): the approved concept's load-bearing facts (orientation, which wall each
  connector faces, sizes) that the CAD is checked against. [FITS.json](FITS.json) and `checks/FITS-X1C.json`: every
  printed fit with its screw and orientation; `checks/fit_check_*.json`: the fit check per printer.
  [SCREWS.json](SCREWS.json): every screw with its axis and driver access.
- `checks/`: interference and access checks (`checks.json`, `access.json`) and mass and centre of mass
  (`mass_com.json`).
- `cad-source/build_v2.py`, `checks.py`, `mass_com.py`: the scripts that made the parts and the numbers. They import
  the robot's private CAD library, so they are a record of how the parts were made rather than something that runs
  from this repository.

## In the simulator

The mass model is `cmu/model/manifests/battery-v2m5-20261121.json` (from `cmu/model/variants/battery-v2m5-20261121.json`):
the 2026-11-04 robot with the old backpack's 18 parts (103.1 g) replaced by this backpack's 11 parts (177.5 g). Whole
robot 1034.8 g, centre of mass at the stand pose x -11.2 mm in the trunk frame (the old backpack: 960.5 g, x -7.6 mm).
The robot weighed 918 g on 2026-11-23, so the current model is `battery-v2m5-w918-20261123` (from
`cmu/model/variants/battery-v2m5-w918-20261123.json`): the same parts and positions with the total matched to 918.00 g
(Pollen CAD mass our robot does not carry and slicer grams first, the rest spread over parts still estimated); centre
of mass at the stand pose x -12.2 mm, z +18.7 mm (the 1034.8 g estimate: x -11.2, z +22.8).
The masses are estimates: printed parts from CAD volume, the battery from its listing, the reused parts at their
earlier masses. The visual layer is `cmu/model/visual-20261122/` (built by `make_visual.py --fitted fitted-20261122.json`).

The open-loop (Fixed) stand pose in `cmu/pi/stand_pose.py` is retuned for this backpack: `ANKLE_BACK = -0.045` (both
ankles tilt the body 2.6 deg forward of the policies' start pose; the charger backpack needed +0.060, which tips this
robot backward at 0.7 s). The study is `guide/firmware/mac/stand-ankle-20261122/` in the private tree: set down in the
pose, the sim robot stands over centre-of-mass offsets 0 to +2 mm, at floor friction 0.4, and under 0.1 m/s forward and
0.05 m/s backward pushes. Switching from the balanced stand (stand61) to the Fixed stand still fails on this
backpack, because stand61 itself shuffles about 35 cm backward in 8 s at zero command before the switch.

What the deployed walking policy (walk61) does on it in `brain_sim_check.py --model asbuilt --manifest current
--walk-s 8`, vx 0.3, with the new stand pose: after the default 1 s of Fixed stand it walks 0.60 m with 6.6 deg peak
tilt and 4 action jumps; started from the policy's first tick (`--stand-s 0`) 0.66 m, 7.2 deg, 4 jumps (old backpack:
0.92 m, 6.2 deg, 1 jump). Policies trained on `-CMU-Current` from now on learn the heavier, further-back robot.

Everything in this folder except this README is a copy of the private robotics tree, refreshed by
`python3 cmu/sync_from_robotics.py`.
