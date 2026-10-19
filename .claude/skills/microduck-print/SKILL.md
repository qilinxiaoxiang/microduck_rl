---
name: microduck-print
description: Prepare a MicroDuck part for the RoboClub printers (Bambu Lab X1 Carbon, Elegoo Centauri Carbon): print pose, brim rule, profiles, slicer CLI, Centauri preview block, delivery names. Use when asked to slice, print, release or hand over a printed part.
---

# MicroDuck print preparation

The whole procedure is in `cmu/hardware/printing/README.md`; follow it, do not improvise settings. In short:

1. Pose: apply R1 to R7 of `cmu/hardware/printing/PRINT-ORIENTATION-PRACTICE.md`; score candidates with
   `cmu/hardware/printing/scripts/orientation_check.py` on a manifest shaped like `cmu/hardware/printing/ready/V1.20/manifest-V1.20.json`.
2. Profile: PLA, 0.4 mm nozzle, 0.2 mm layers, 4 walls, 30 % gyroid, tree support 30 degrees, 0.2 mm top gap,
   elephant foot 0.1 mm (0.15 for Yellow and White). X1C on Cool Plate with Bambu PLA Basic; Centauri on Textured
   PEI with generic PLA. Start from `cmu/hardware/printing/ready/V1.20/*_editable.3mf` or `cmu/hardware/printing/profiles/`.
3. Brim per object, never per plate: outer-only 4 mm with 0.1 mm gap only when the pose's tip angle is under
   10 degrees; otherwise no brim. 12 mm between parts, plate centred on (128, 128) within 1 mm, corners excluded.
4. Slice with the CLI lines in the README (`--allow-newer-file --arrange 0 --orient 0 --slice 0`, Bambu adds
   `--min-save` and the `filament_self_index` fix; Elegoo loads `profiles/elegoo-{machine,process}.json` and
   `--load-filaments profiles/elegoo-filament.json`).
5. Centauri G-code: insert the 144 x 144 thumbnail block right after `; HEADER_BLOCK_END`, before the first G/M line.
6. Deliver one folder per printer (`X1C/*.gcode.3mf`, `Elegoo/*.gcode`), file names
   `<who>_<version>_<Colour>_<Bambu-X1C|Elegoo-Centauri-Carbon>_<HhMMm>`, plus one `<Colour>-plate.png` per plate.
   Report what is on the plate and what each part replaces; never call a plate released without the preview read-back.

Known failures to avoid are listed in `cmu/hardware/printing/PRINT-LESSONS.md`.
