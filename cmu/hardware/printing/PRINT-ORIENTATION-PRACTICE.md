# MicroDuck print orientation practice

Use this guide to choose the print pose of a MicroDuck part and to prove the choice with the study scripts. It was built and tested on the 13 parts of the V1.6 release. All 13 current poses stay, and each part's evidence and residual risks are in `results.json`.

Working directory for every command: `microduck-sim/guide/print/orientation-study/`. Sources were read on 2026-10-09. A statement marked **Inference** is a project judgement that no official source states.

## Decision rules, highest priority first

Apply the rules in order. A lower rule only decides between poses that tie on every rule above it.

### R1. Stand on a real flat base

The part must sit on true flat contact, with its centre of mass well inside the first-layer outline. A pose that needs support or a brim to stay upright is rejected.

- **Sources.** Bambu's warping page: "Consider reorienting the model to lie flat." ([link](https://wiki.bambulab.com/en/filament-acc/filament/print-quality/warping-falling-off-collapsing)). Bambu auto orientation scores a larger bottom area and a larger bottom convex hull for stability ([link](https://wiki.bambulab.com/en/software/bambu-studio/auto-orientation)). Bambu auto brim: "A taller part with a smaller footprint usually needs a wider brim" ([link](https://wiki.bambulab.com/en/software/bambu-studio/auto-brim)). Prusa: "use at least a 3 mm brim" for tall parts on a small base ([link](https://help.prusa3d.com/article/skirt-and-brim_133969)). Elegoo lists a small bottom contact area as a warping cause ([link](https://wiki.elegoo.com/en/troubleshooting/fdm-printer/edge-warping)).
- **Measure.** `orientation_check.py` reports `contactMm2` (triangles within 0.01 mm of the bed, the release pipeline's own flat-contact rule), `firstLayerSectionMm2`, `tipMarginMm` and `tipAngleDeg`. A tip margin at or below zero is a hard failure. The native readback gives the slicer's first-layer model area.
- **Inference.** Below a 10 degree tip angle, add a 3 to 5 mm brim on that object. Auto brim chose no brim for the two V1.6 parts below that line (head cap 9.0 degrees, IMU clamp 6.9 degrees).

### R2. Keep functional faces off support

Screw-head seats, sliding and bearing seats, PCB pads, pin tips and mating faces should face up, sit on the bed or form vertical walls.

- **Sources.** Bambu auto orientation avoids support on surfaces because "supports can damage the interface surface more or less" ([link](https://wiki.bambulab.com/en/software/bambu-studio/auto-orientation)). Bambu's support page explains that the top Z gap and interface layers trade easy removal against surface quality ([link](https://wiki.bambulab.com/en/software/bambu-studio/support)). Project precedent: the 10-SURFACE-QUALITY pack flipped ankle community-66 so its four screw-head seats face up.
- **Measure.** The screen counts `holeEndsFacingDown` and `bossFacesFacingDown`. After a native slice, `scripts/seat_support.py` measures support toolpath under each down-facing hole end and boss face, plus support trapped inside hole cores. `scripts/support_map.py` plots every face that rests on support. The checker cannot tell which curved faces are sliding fits, so name them first and probe them, as `scripts/jaw_bearing_seat.py` does for the jaw seat.
- **Inference.** When two functional faces compete, a moving fit outranks a static clamp seat. A rough seat under a screw head still clamps, while a supported sliding seat with 0.2 mm clearance can bind or rattle.
- **Inference.** A down-facing end lower than 0.6 mm above the bed is bridged by the first layers. The mouth ring's 0.3 mm step sliced with zero support.

### R3. Print holes vertical where roundness or pitch matter

Screw pilots, horn hole patterns, bores and press fits are most accurate with a vertical axis.

- **Sources.** Bambu's hole compensation corrects only circles in the horizontal plane: "semicircles or circles in the vertical direction will not be compensated" ([link](https://wiki.bambulab.com/en/software/bambu-studio/manual/auto-circle-contour-compensation)). Hubs (non-official) reports that vertical cylinders are smooth and accurate while horizontal ones facet ([link](https://www.hubs.com/knowledge-base/how-does-part-orientation-affect-3d-print/)).
- **Gap.** No official Bambu, Prusa, Orca or Elegoo page covers horizontal-hole droop or teardrop holes.
- **Measure.** `holesVertical`, `holesInclined` and `holesHorizontal`, with each hole's `tiltDeg` (vertical up to 15 degrees, horizontal from 75 degrees).

### R4. Keep the main load in the layer plane

Layers bond more weakly than lines within a layer, so a thin arm, pin or clip that carries real bending or tension should lie in the layer plane.

- **Sources.** Bambu: "avoid having the layers bear the main load" ([link](https://wiki.bambulab.com/en/knowledge-sharing/common-print-quality-problem)). Prusa: "FFF prints are stronger in one direction than another" ([link](https://help.prusa3d.com/article/modeling-with-3d-printing-in-mind_164135)). Hubs (non-official): "Z tension increases the risk of delamination and fracture."
- **Measure.** Name the main load and the weakest section by hand, then estimate its stress. The screen reports `pinsVertical` and `pinsHorizontal` for small round bosses.
- **Inference.** The neck pitch ears carry about 1.5 MPa at 0.25 N m per ear across an 18 by 3 mm root, far below PLA's strength across layers. A small load like this does not justify breaking R1 to R3.

### R5. Place cosmetic faces deliberately

The bed face prints flattest, takes the plate texture and flares slightly. Faces that rest on support print rough.

- **Sources.** Bambu: "the flared base directly affects fit" ([link](https://wiki.bambulab.com/en/software/bambu-studio/parameter/elephant-foot)). Bambu Textured PEI "Adds a unique textured finish to the bottom surface of printed parts." ([link](https://wiki.bambulab.com/en/filament-acc/acc/plates)). Prusa puts elephant foot at about 0.2 mm ([link](https://help.prusa3d.com/article/elephant-foot-compensation_114487)).
- **Project profiles.** Elephant-foot compensation is 0.15 mm (Yellow, White) or 0.1 mm (Orange, Black). The Bambu profiles use Cool Plate and the Elegoo profiles in this study use Textured PEI.
- **Measure.** `support_map.py` reports `supportTouchedFaceMm2` by face direction and draws the supported faces.

### R6. Minimise support and keep it removable

Prefer less support, none trapped in cavities, slots or hole cores.

- **Sources.** Bambu: "The default threshold angle is 30 degrees." ([link](https://wiki.bambulab.com/en/software/bambu-studio/support)). Bambu: below 45 degrees from the bed "it is recommended to add supports" ([link](https://wiki.bambulab.com/en/filament-acc/filament/print-quality/overhang)). Prusa advises reorienting to reduce supports ([link](https://help.prusa3d.com/article/support-material_1698)). All MicroDuck profiles use tree(auto) support, a 30 degree threshold and a 0.2 mm top Z gap.
- **Measure.** Screen: `overhang60Mm2`, `supportProxyMm3`, `supportOnModelMm2`. Native: `supportGrams` and `supportInHoleCoresMm`. Protect slots and pilots with support blockers in the manifest; the pipeline carries them with the part through any pose.

### R7. Use height, time and filament as tie-breakers

- **Measure.** Native `etaMinutes`, `totalGrams` and `layers` from `slice_readback.py`.
- **Gap.** No official FDM source gives a numeric rule linking orientation to time, so the native slice is the measure.

## How to check a part

1. **Screen every pose.** Run the checker under the memory guard. It loads one mesh at a time and reports the current pose next to the best alternative.

   ```
   G=microduck-sim/build-pack-20260913/redesign-20260913/head-redesign-20261008/run_bounded_cad.py
   python3 $G --max-mib 1200 --seconds 300 --wait-lock-seconds 60 --report evidence/check-resource.json -- \
     /tmp/microduck-cad-venv/bin/python orientation_check.py ../release-V1.6-final/final-V1.6.json \
     --extra-poses candidates/extra-poses.json --out evidence/check-v16-screen.json
   ```

   Add `--key KEY` for one part. `candidates/extra-poses.json` holds named poses such as the 10/1 pose and the V1.4 on-edge pose.

2. **Choose two or three candidates.** Always take the current pose, the best-screened alternative and any functional pose the screen undervalues, such as the robot-upright pose.

3. **Slice them natively.** Add rows `{key, slug, name, contactMm2, sourceToPrint}` to `candidates/slice-plan.json`, then run `python3 scripts/slice_candidates.py candidates/slice-plan.json --machines Bambu --only KEY/SLUG`. The driver writes a single-part STUDY manifest once, then runs prepare, slice and readback one job at a time under the guard. Add `Elegoo` for parts that matter on both printers.

4. **Read the functional evidence.** Run `scripts/seat_support.py RUN COLOR MACHINE OUT.json` and `scripts/support_map.py RUN COLOR MACHINE OUT.png OUT.json` for each slice, each under the guard. Write a small probe for any sliding or bearing seat.

5. **Decide and record.** Walk R1 to R7 in order. Write the verdict in `candidates/verdicts.json` and run `/tmp/microduck-cad-venv/bin/python scripts/build_results.py`. The builder fails if a part lacks candidates, a recommendation, evidence paths or, for a changed pose, a native slice.

## Known failure cases

- **V1.4-V1.5 on-edge regression.** The lower head shell stood 115.7 mm tall on a 332 mm2 edge with its centre of mass outside the first-layer outline (tip angle -9.5 degrees). On Bambu it took 42.5 min and 11.9 g more than the flat pose, with 21.44 g of support against 8.71 g and 578 layers against 100. The head cap stood 91.3 mm tall at -31 degrees and cost 48 min and 12.9 g more. Both poses maximised one flat facet without checking R1's tip margin. V1.6 restored the flat poses.
- **A screen favourite that breaks a sliding fit.** The jaw support's robot-upright flip frees the four screw-head seats and screens best. The native slice shows 83.8 of the 240.5 mm2 mouth-ring seat resting on support, so the current pose stays.
- **Hidden down-facing pads.** Laying the IMU clamp on its plate makes its pilots vertical and saves 2.7 min, but all four PCB pads and their locating pins then face down and print on 1147 mm of support.
- **Seats on support accepted as a cost.** The lower head shell's three cap-screw tabs and the jaw support's four screw seats print on support because every alternative breaks R1 or a higher R2 face. Clean those seats after printing.
- **A membrane inside a bore.** The neck bracket's 6 mm centre bore keeps a 0.25 mm membrane 1.7 mm above the bed, and the recess under it fills with support. Clear it and confirm whether the membrane should stay.
- **No brim on the lowest-margin parts.** The released Orange plates contain no brim on either printer, although the head cap and IMU clamp have the smallest tip angles.

## Patterns by part type

- **Shells** (lower head shell, head cap). Put the open side up, or the dome on the bed. Accept support under the dome when that keeps every post and pilot end clean. Never stand a shell on its edge.
- **Brackets** (neck pitch, jaw support). Put the mating face (servo horn face, flange) on the bed or facing up, with the horn pattern and bores vertical. Find any sliding seat before comparing screw seats. Tall ears are fine when their root stress is small.
- **Clips and holders** (IMU clamp, camera retainer, mouth ring). Keep PCB pads and pins as vertical walls or facing up. Put the largest flat face down. Print thin rings flat.
- **Trays** (OpenRB tray). Put the base on the bed and block support out of PCB slots.
- **Panels and plates** (face panel, eye trim, IMU cover, speaker cradle and keeper). Put the large or visible face on the bed; the holes then print vertical by default.

## Counterexamples checked

- **Less support, but horizontal screw holes.** The neck bracket on its side saves 8.1 min and 0.41 g of support, but makes the horn hole pattern and 6 mm bore horizontal and lifts the horn face off the bed. Kept current.
- **Clean screw seats, but a supported sliding seat.** Jaw support flip. Kept current.
- **Vertical pilots and fewer layers, but supported PCB pads.** IMU clamp lying. Kept current.
- **A better visible dome, but every post supported.** Head cap rim-down puts 14 hole ends and 21 boss faces on support and costs 79 min more. Kept current.
- **A flagged step that costs nothing.** The mouth ring's 0.3 mm step slices with zero support, which led to the near-bed rule in R2.

## V1.6 verdicts

- **head_bottom**: keep upright. On-edge costs 42.5 min; rim-down has a 19 mm2 first layer.
- **face_panel**: keep outer face down. The flip adds 19.1 min and 4.13 g of support.
- **head_cap**: keep top-down. Rim-down adds 79 min and puts every post on support; consider a brim.
- **speaker_cradle**: keep flat. Both faces slice identically.
- **speaker_keeper**: keep flat. Both faces slice identically.
- **eye_trim**: keep flange down. The flip leaves 6 mm2 of first layer.
- **camera_retainer**: keep the larger face down. The flip puts three screw ends on support.
- **jaw_support_original**: keep. The flip supports the mouth-ring seat; clean the four screw seats.
- **mouth_sliding_ring**: keep. The flip adds support under the ring faces.
- **neck_pitch_integrated_1mm_seats**: keep. The side pose makes the horn holes horizontal; clear the 6 mm membrane recess.
- **imu_cover**: keep flat back down. The flip leaves 36 mm2 of contact.
- **imu_clamp**: keep standing. Lying supports the PCB pads; consider a brim.
- **openrb_tray**: keep base down. The flip leaves a 20 mm2 first layer and adds 69 min.

## V1.8 verdicts

- **rear_pack**: keep upright (floor on the bed). Lying on the open back (the only alternative with real flat contact) puts the four OpenRB pad pilots and 13 boss faces on support (Bambu: 871.5 mm of support under pilot ends, 1077.9 mm under boss faces, against 0 and 0 upright) and adds 36.8 min and 6.95 g of support on Bambu (36.4 min and 5.40 g on Elegoo). Upright costs: eight 1.6 mm pilots print horizontal, and 82.8 mm stands on a 31.5 mm wide base (tip angle 14.2 degrees, above the brim line). Evidence: `slices/rear_pack/`, plan `candidates/slice-plan-v18-rear-pack.json`, run with `--final ../release-V1.8/manifest-V1.8.json`.

## Limits

- No candidate was printed. Surface finish, support removal and fits are predicted from native slices.
- Strength judgements are estimates; no part was load tested.
- The checker's penalty weights are heuristic and only order candidates for slicing and review.
- BambuStudio hung once at start-up on the mouth ring (170 s guard timeout); the retry sliced normally.
