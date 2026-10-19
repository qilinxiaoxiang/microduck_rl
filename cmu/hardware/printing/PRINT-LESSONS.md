# Print lessons

## Brim puddles on small parts (2026-10-14)

**What happened.** On the V1.9 Orange plate the four cable clips and the speaker retainer came off the bed sitting in a
thin PLA sheet with nothing printed on top of it. The sheet on the speaker retainer was very hard to peel.

**What the sheet is.** It is the slicer brim: a 4 mm outer brim that the release put on every object of the Orange plate.
It is not a raft (`raft_layers = 0`), not a skirt (`skirt_loops = 0`), not support interface and not elephant foot
(`elefant_foot_compensation = 0.1` shrinks the first layer).

**Cause.** `release-V1.9/manifest-V1.9.json` sets a plate-wide process override:

```json
"processOverrides": {
  "Orange_Bambu":  {"brim_type": "outer_only", "brim_width": "4"},
  "Orange_Elegoo": {"brim_type": "outer_only", "brim_width": "4"}
}
```

Its stated reason covers one part only: "Head cap tip angle 7.7 degrees ... is below the 10 degree line in
guide/print/orientation-study/PRINT-ORIENTATION-PRACTICE.md". The pipeline applies an override to the whole plate's
`project_settings.config` (`release_pipeline.py` line 144) and writes only `name` and `extruder` per object (line 157),
so all seven objects got the brim. In V1.8 the Orange plate held only the head cap and the rear pack, two large parts,
so the problem did not show. V1.9 added five small parts to the same plate. V1.10 carries the same override; V1.11 sets
`brim_type = no_brim` for the whole Orange plate.

**Evidence.** Both delivered files of both V1.9 runs (`release-V1.9/run/` and `run-r2/`) contain `brim_type = outer_only`,
`brim_width = 4`, `brim_object_gap = 0.1`, `raft_layers = 0`, `skirt_loops = 0` (Bambu `Metadata/project_settings.config`;
Elegoo G-code footer), and no per-object setting in `Metadata/model_settings.config`. First layer of
`run-r2/delivery/zhengxia_V1.9_Orange_Bambu-X1C_3h38m.gcode.3mf`, extrusion paths buffered to 0.5 mm line width:

- speaker_retainer_v19: brim 687 mm2 against 347 mm2 of part (2.0x). 286 mm2 of that brim lies inside the ring.
- clip_cap_v19: brim 271 mm2 against 52 mm2 (5.3x), 66 mm2 of it inside the hook pockets.
- clip_torso_v19: 167 against 55 mm2 (3.1x). clip_bracket_v19: 173 against 63 mm2 (2.7x). clip_head_v19: 193 against 111 mm2 (1.7x).
- head_cap_v19: 449 against 451 mm2 (1.0x). rear_pack: 287 against 641 mm2 (0.4x). Here the brim is a strip around a large
  base, so it looks like a normal brim.

The retainer is a 1.6 mm frame shaped like a C: the speaker opening is open to one side, and only the two screw eyes are
closed holes (mesh Euler number -2). `outer_only` treats the open speaker opening as outside, so the brim runs into it and
covers most of the opening. The thin band is bonded to brim along both its outer and inner edges, 0.1 mm away. A 1.6 mm
frame bends before it levers off, which is why that sheet was the hard one. The brim gap was 0.1 mm, not 0.

The V1.9 review did not catch this. `release-V1.9/scripts/review_v19.py` lines 139 to 140 only require some brim on a colour
that has an override (`brimExtrusionMm > 0`), and `run/native-review.json` records "4 mm outer brim present on both machines"
as a pass.

**Which parts need a brim.** In this repository the brim rule is R1 of `PRINT-ORIENTATION-PRACTICE.md`: add a 3 to 5 mm brim
below a 10 degree tip angle. `flatContactMinimumMm2` in `scripts/rows.json` is the pipeline's check that the chosen face really
lies flat on the bed. It does not decide adhesion. Using both (tip angles from `ext-v19/checks/orientation-ext-v19.json` and
`rear-pack-v19/orientation-check.json`):

- head_cap_v19: keep the brim. Tip angle 7.7 degrees, 47 mm tall, true flat contact 104 mm2 (minimum 93) under a 464 mm2 first layer.
- rear_pack: no brim needed. Tip angle 13.0 degrees, 661 mm2 contact (minimum 600).
- clips: no brim. Tip angles 50.5 to 57.8 degrees, 6 to 11 mm tall, contacts 35 to 120 mm2 (minimums 31 to 107).
- speaker_retainer_v19: no brim. Tip angle 86.8 degrees, 1.6 mm tall, 368 mm2 contact (minimum 330).

**Fix: brim per part (pipeline v2).** Each part now carries its own brim decision. New releases use
`build-pack-20260913/redesign-20260913/printing/13-RELEASE-PIPELINE-20261009/release_pipeline_v2.py`, `verify_release_v2.py`
and `review_brim.py`. `release_pipeline.py` and `verify_release.py` stay unchanged because every V1.4 to V1.11 receipt binds
the old pipeline's hash.

- **Default.** Leave the brim out of `rows.json` and the pipeline applies R1 from the part's tip angle in its print pose,
  computed like `orientation_check.py` (on V1.9 the same values to 0.1 degree). Below 10 degrees the part gets
  `brim_type = outer_only`, `brim_width = 4`, `brim_object_gap = 0.1`; every other part gets `brim_type = no_brim`.
  On the V1.9 Orange plate that gives the brim to head_cap_v19 (7.7 degrees) only.
- **Override.** To decide differently, put both fields on the part's row in `rows.json`:

  ```json
  "objectSettings": {"brim_type": "outer_only", "brim_width": "4", "brim_object_gap": "0.1"},
  "objectSettingsReason": "why R1 does not fit this part"
  ```

  `brim_type` is `no_brim` or `outer_only`; `outer_only` also needs `brim_width` and `brim_object_gap`; values are strings.
  The release's `make_manifest.py` copies the two fields into the manifest part, for example
  `parts.append(dict(..., **{k: r[k] for k in ('objectSettings', 'objectSettingsReason') if k in r}))`
  (`13-RELEASE-PIPELINE-20261009/tests/brim-per-object-V1.9-Orange/make_manifest.py` shows it).
  Brim ears, inner brims, auto brim and painted brims are refused, because the review has no rule for them. Brim ears also
  hold corners down without stopping a part from tipping.
- **Plate overrides.** `processOverrides` must not set `brim_type`, `brim_width` or `brim_object_gap`; the pipeline refuses
  them, so a V1.10- or V1.11-style plate brim line must be deleted. Other process overrides still apply to the whole plate.
  The donor default (`auto_brim`, 5 mm) no longer reaches any part, because every object carries its own `brim_type`.
- **Where it lands.** `prepare` writes the settings as `<metadata key="brim_type" value="..."/>` entries on each object in
  `Metadata/model_settings.config`. `preparation.json` records `tipAngleDeg`, `objectSettings` and `objectSettingsSource`
  (R1 default or the manifest reason) for each part.
- **Review.** `first_layer()` measures the brim of each object. OrcaSlicer prints each brim just before the object's
  label, so the pipeline gives unlabelled brim to the nearest part. `brim_review()` fails a part with a brim it should not
  have, a planned brim that is missing, a brim whose outer edge is more than 0.5 mm off gap plus width, or more than 2 mm2
  of brim inside the part's own convex first-layer outline. `package` stops on a failure, and `verify_release_v2.py`
  recomputes the review from each native file.
- **Review script for the next release.** Copy `release-V1.11/scripts/review_v111.py`, import `release_pipeline_v2` as `r2`,
  and replace the plate-wide check (the `brim_mm` call on line 95 and the `ov` / `brim_needed` / `brim_ok` lines 116 to 118)
  with `rows, _ = r2.first_layer(g, project, machine)` and `brims = r2.brim_review(rows, parts, machine)`. Use
  `brims['passed']` in `ok` and store `brims` in the review JSON. `review_brim.py RUN Color Machine` gives the same result
  from the command line.

**Acceptance (2026-10-14).** A copy of the V1.9 Orange plate without the plate brim is in
`13-RELEASE-PIPELINE-20261009/tests/brim-per-object-V1.9-Orange/` (`acceptance.json`), sliced on both printers. Bambu Studio
and OrcaSlicer both honour the per-object `brim_type` and `brim_width`. First-layer brim on Bambu, delivered `run-r2` before,
new slice after:

- head_cap_v19: 449 mm2 before and after (planned; outer edge 4.29 mm out for 4.1 planned).
- speaker_retainer_v19: 687 mm2 before (286 of it inside its outline), 0 after.
- clip_cap_v19: 271 before (66 inside), 0 after. clip_torso_v19: 167 before, 0 after. clip_bracket_v19: 173 before, 0 after.
  clip_head_v19: 193 before, 0 after.
- rear_pack: 287 before, 0 after.

Elegoo shows the same result (rear_pack 331, clip_torso 134, clip_head 155 mm2 before; 0 after; head cap 449 both times).
The raw G-code agrees: before, all seven objects had brim extrusion; after, only the head cap (971 mm on both printers).
The new review fails both delivered V1.9 files and passes both new ones.

## Previewing delivered files on the MacBook (2026-10-14)

- **X1C `.gcode.3mf` in Bambu Studio 02.08.02.61:** a CLI-sliced file used to open with "Invalid configuration file" and the default yellow. Cause (Bambu Studio source, PresetBundle.cpp L3794-3810): the loader requires one `filament_self_index` entry per `filament_extruder_variant` row; the donor projects carry no `filament_self_index`, so the CLI wrote one entry against two rows (Direct Drive Standard, High Flow). Not a user-config problem: the user preset folder is empty.
- **Fix (release pipeline v2, from V1.14):** `bambu_self_index()` writes one `filament_self_index` entry per variant row, and the Bambu slice adds `--min-save` so the archive carries no model objects (Plater.cpp L9579-9614 reuses the stored plate G-code only when the project has no objects; padding the index without `--min-save` opens on Prepare with no toolpaths). `check_bambu_archive()` enforces both plus a stored plate G-code at slice and package time. Shawn confirmed on V1.14: no dialog, supports shown, correct colour.
- **Never open the X1C `.gcode.3mf` in OrcaSlicer:** Orca 2.4.2 rejects the Bambu 02.08 config (four dialogs) and shows no toolpaths. Installing Orca into /Applications silently made it the .3mf handler; .3mf was set back to Bambu Studio explicitly (NSWorkspace setDefaultApplication), .gcode to Orca. Re-check both handlers after installing or updating either slicer.
- **Elegoo `.gcode`:** opens in OrcaSlicer (installed in /Applications, default handler for .gcode since 2026-10-14) with the full preview including support; the plate title reads Untitled because a bare G-code file has no project name.
- **Delivery:** `make_delivery_folder.py guide/print/release-<version>` writes ~/Downloads/zhengxia_MicroDuck_<version>/ with X1C/ and Elegoo/ (no zip) and one <Colour>-plate.png per plate, so nobody has to open a slicer to see what is on a plate.
