# Incremental dual-printer release pipeline

This directory is a reusable preparation tool, not a new MicroDuck print release. V1.4 is read-only. The only native smoke-test geometry is one existing 6 × 6 × 2 mm Pi spacer; its output is marked `FIXTURE_ONLY_NOT_RELEASED` and must not be distributed as a robot package.

## Slicers

Bambu Studio runs from `/Applications/BambuStudio.app`. The Elegoo slice runs OrcaSlicer 2.4.2 from `/Volumes/OrcaSlicer`, the official disk image kept at `~/Library/Caches/microduck/OrcaSlicer-2.4.2.dmg` (GitHub release asset `OrcaSlicer_Mac_universal_V2.4.2.dmg`, sha256 `e15e7bb1b66214ec6e96b169b388004179c4f5f705effcdaf8c80d4992ee0366`). Keep the image out of `/tmp`, which macOS clears. When `slice ... Elegoo` fails with `No such file or directory: '/Volumes/OrcaSlicer/...'`, mount it:

    hdiutil attach -readonly -nobrowse -mountpoint /Volumes/OrcaSlicer ~/Library/Caches/microduck/OrcaSlicer-2.4.2.dmg

## Pipeline versions

New releases use `release_pipeline_v2.py`, `verify_release_v2.py` and `review_brim.py`. `release_pipeline.py` (v1) and `verify_release.py` stay byte-identical: every V1.4 to V1.11 `preparation.json` binds the v1 pipeline's sha256, so an edit would break their verification. Verify a v1 run with `verify_release.py` and a v2 run with `verify_release_v2.py`.

v2 sets the brim per part. Each `parts` entry may carry `objectSettings` with `brim_type` (`no_brim` or `outer_only`), and for `outer_only` also `brim_width` and `brim_object_gap`, all as strings, plus a non-empty `objectSettingsReason`. Without `objectSettings` the pipeline applies R1 of `guide/print/orientation-study/PRINT-ORIENTATION-PRACTICE.md` from the part's tip angle in its print pose (computed like `orientation_check.py`): below 10 degrees `{"brim_type": "outer_only", "brim_width": "4", "brim_object_gap": "0.1"}`, otherwise `{"brim_type": "no_brim"}`. `prepare` writes the settings as `<metadata key=... value=.../>` on each object in `Metadata/model_settings.config`; `preparation.json` records `tipAngleDeg`, `objectSettings` and `objectSettingsSource` per part. Brim keys in `processOverrides` are rejected; other process overrides still apply plate-wide.

`first_layer` measures brim per object: Bambu prints it inside the object label, OrcaSlicer just before `EXCLUDE_OBJECT_START`, so unlabelled brim is assigned to the nearest first-layer model. `brim_review` fails a part with an unplanned brim, a planned brim that is missing or whose outer reach differs from `brim_object_gap + brim_width` by more than 0.5 mm, or more than 2 mm² of brim inside the part's own convex first-layer outline (an open opening filled with brim, as on the C-shaped V1.9 speaker retainer). `package` stops on a failure and records the review in `native-delivery-checks.json`; `verify_release_v2.py` recomputes it from each native archive. A release review script calls `release_pipeline_v2.brim_review(rows, parts)` or runs `review_brim.py RUN Color Machine`. Both slicers honour per-object `brim_type` and `brim_width` (`tests/brim-per-object-V1.9-Orange/acceptance.json`). `test_pipeline_v2.py` holds the v2 tests.

## Final input

Supply a JSON manifest with `schemaVersion: 1`, `releaseKind: "incremental"`, a new `version`, `baselineStock: {path, sha256}` for `printing/copy-manifest.json`, and `profiles` containing the eight color/machine donor-project references. `profile-probe.json` enumerates the current donors. Use the actual final adopted geometry, never a head-layout candidate.

Each `parts` entry contains:

- `key`, `color` (`Yellow`, `White`, `Orange`, `Black`), `quantity`.
- Absolute STL `path` and exact `sha256`.
- `sourceToPrint`: rigid 4 × 4 matrix in millimetres. This is the selected functional orientation, not a request to find the largest face. The pipeline translates the lowest point onto the bed without changing rotation.
- `flatContact.minimumAreaMm2`: the actual intended planar-contact lower bound for this part. Contact triangles must lie within 0.01 mm of the bed. There is no universal minimum area.
- `decision`: `replacement` or `addition`; `replacesStockIds`: actual `copy-manifest.json` object IDs, not guessed community part numbers.
- `changeReason` and `incrementalEvidence: {path, sha256}`: the specific source-bound change/stock comparison. Different file bytes alone are not evidence that an unchanged part needs reprinting.
- Optional `supportBlockers`: `{path, sha256, machines: ["Bambu", "Elegoo"]}` entries. These are in the **same source frame as the owner STL**, receive the same rigid transform, and are nonprinting modifiers.

Optional `placements` maps each color to `{key, copy, translation: [x,y,0]}` for every instance. Otherwise a centered row layout uses 12 mm model gaps within the common central bed region. Explicit placements must also centre the plate: prepare refuses a plate whose combined model footprint centre is more than 1 mm (`PLATE_CENTRE_TOL_MM`) from the bed centre (128, 128) and records `plateCentreOffsetMm` per row; compute translations from the printed part sizes, never hand-typed corner coordinates (V1.17 and V1.18 printed 11 to 63 mm off centre). Optional `processOverrides` maps `Color_Machine` to deliberate slicer process settings; protected printer/material/nozzle/bed identities are checked afterwards. Explicit pose/placement/modifier changes are recorded in preparation and require a new native slice. Nozzle is 0.4 mm, material is PLA, Bambu plate is Cool Plate and Elegoo plate is Textured PEI.

`fixture-manifest.json` is an executable schema example. It deliberately has only Black and is **not** a final four-color manifest.

## Run

Use `/tmp/microduck-cad-venv/bin/python` (NumPy, Trimesh, Shapely and Pillow are already present). No new software or purchase is needed.

```sh
python release_pipeline.py probe profile-probe.json
python release_pipeline.py prepare final-release.json NEW_RUN_DIRECTORY
```

For each generated row, run one native slice under the existing **exclusive** process-tree resource guard. Start with 512 MiB / 120 seconds for these measured small jobs; do not bypass a timeout or resource rejection. A full plate may need its established measured budget, chosen before launch. Do not use `--parallel-light` for slicing.

```sh
python ../../head-redesign-20261008/run_bounded_cad.py \
  --max-mib 512 --seconds 120 --wait-lock-seconds 30 \
  --report NEW_RUN_DIRECTORY/Black-Bambu-resource.json -- \
  /tmp/microduck-cad-venv/bin/python release_pipeline.py slice \
  NEW_RUN_DIRECTORY/preparation.json Black Bambu
```

Repeat for Elegoo and the other present colors. Preparation does not slice anything. The slicer runs `--arrange 0 --orient 0`, never `--no-check`; native material IDs are populated. Failed native export/crash remains a failure. Existing `V1.4` inputs and outputs are never edited.

Native readback checks actual model first-layer area, layer height, all-layer extrusion footprint including supports/brim, printer identity and embedded preview. The parser supports both Bambu legacy object labels and current `OBJECT_ID` labels, plus Orca object labels and circular arcs. Bambu extruder offset is added back before bed measurement. The old first-layer helper omitted Z from its token parser and assumed 0.2 mm; it is deliberately not reused here.

Inspect the native slice's functional holes/seats and support access using the existing per-part probes or viewer. This is the ordinary print review, not another robot/CAD acceptance gate. Record a JSON receipt:

```json
{
  "passed": true,
  "preparationSha256": "actual preparation file hash",
  "nativeArchives": {"/absolute/path/to/native/Black_Bambu/sliced.3mf": "actual archive hash"},
  "reviewNotes": "Actual flat model first layers, complete support footprints and relevant functional holes reviewed."
}
```

The dictionary must include **every** generated archive. The review is separate from the release input to avoid a circular hash dependency. Then:

```sh
python release_pipeline.py package NEW_RUN_DIRECTORY/preparation.json \
  --native-review NEW_RUN_DIRECTORY/native-review.json
```

Without that ordinary review, output status is `PREPARED_NATIVE_REVIEW_PENDING`, not `RELEASED`. Fixture status can never become `RELEASED`. Packaging uses a new empty delivery directory; it does not silently replace previous machine files.

## Binding and handoff

`preparation.json` binds the manifest, pipeline, all final STL sources, stock evidence, exact source-to-print transforms, donor profiles and modifier files. Every actual source and copied baseline STL is checked against the 89-instance manifest. Exact unchanged stock files and duplicate replacement IDs are rejected; a source-bound incremental decision covers geometric change rather than mistaking a new encoding for a new part.

Each native result binds preparation and project bytes, command and filament profile, native archive, and G-code bytes. Delivery ETA comes from that machine's real G-code header, rounded **up** to the filename minute. Names are `zhengxia_VERSION_Color_Bambu-X1C_TIME.gcode.3mf` or `zhengxia_VERSION_Color_Elegoo-Centauri-Carbon_TIME.gcode`.

Bambu exports retain their native archive PNGs. Orca CLI omits thumbnails, so the Elegoo file carries the Bambu native preview of the identical model layout, resized to the profile's `thumbnails` size (144x144). The block goes right after `; HEADER_BLOCK_END`, wrapped in `; THUMBNAIL_BLOCK_START`/`END` as in Elegoo's own sample files. The Centauri Carbon firmware (`elegooofficial/CentauriCarbon`, `firmware/core/klippy/gcode_preview.c`) reads the preview, ETA and grams only when `OrcaSlicer` appears in the first 30 lines, and stops looking at the first G/M command. A preview prepended above the header (every Elegoo file through V1.5) shows nothing on the printer. `check_cc_preview` replays that parser in `package` and `verify_release.py`, and proves the command stream is byte-for-byte the native one apart from the inserted block. The preview is cross-machine provenance, not an Orca GUI export, and does not assert identical supports across machines.

`delivery-index.json`, `incremental-audit.json`, `native-delivery-checks.json` and `package-receipt.json` bind the source-to-ZIP chain. The ZIP contains only the two native machine files for each present color: eight files when all four colors are present. Metadata and source STLs remain outside the SD ZIP. No printing, upload, website publication, or physical-fit claim is performed by this tool.

## Tested readback

The existing spacer fixture passed both native exports: Bambu 1.42 s / 204 MiB sampled peak and Elegoo 1.12 s / 195 MiB, each under the exclusive 512 MiB / 120 s guard. Actual native first-layer model areas are 19.335 and 19.334 mm², one connected island, zero support. Native ETA is 552 s (filename `10m`) for Bambu and 109 s (`2m`) for Elegoo; startup/calibration explains why a tiny part still has a longer Bambu estimate.

Run the read-only binding verifier before root packages or publishes:

```sh
python verify_release.py NEW_RUN_DIRECTORY --output NEW_RUN_DIRECTORY/final-binding-check.json
```

`test_pipeline.py` contains fifteen preparation/source/profile/pose/stock/bed/arc tests, including three Centauri Carbon preview tests: the preview follows the header, the old prepended layout is rejected, and an altered command stream is rejected. `test_binding.py` contains four real-native-output tests, including altered bytes, falsified ETA and an attempt to relabel the fixture as released. All nineteen pass. `fixture-acceptance.json` binds the actual test archive. This smoke run verifies tool compatibility and binding, not the pending final head geometry.
