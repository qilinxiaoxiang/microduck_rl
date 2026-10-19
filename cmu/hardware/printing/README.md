# Printing MicroDuck parts on the RoboClub printers

Two printers, one material: PLA, 0.4 mm nozzle, 0.2 mm layers, 4 walls, 30 % gyroid infill, tree supports from
30 degrees, 0.2 mm support top gap, 0.1 mm elephant-foot compensation, no prime tower, aligned seam. Our profiles for
both machines are in `profiles/`; the names below are what you see inside the slicer.

- **Bambu Lab X1 Carbon**: Cool Plate, Bambu PLA Basic (filament id GFA00), 220 C nozzle, 35 C plate. Slicer: Bambu
  Studio 02.08.02.61. The printer takes a `.gcode.3mf` on its SD card (or over the network from the slicer).
- **Elegoo Centauri Carbon**: Textured PEI plate, generic PLA (EPLAB00), 220 C nozzle, 55 C plate. Slicer: OrcaSlicer
  2.4.2 (ElegooSlicer is a fork of it; the Orca profile works in both). The printer takes a plain `.gcode` on a USB
  stick; its screen shows a preview only if the file follows the rule in "The Centauri preview" below.

Both beds are 256 x 256 mm; keep parts out of the X1C's front-left 18 x 28 mm corner and the Centauri's front-right
10 x 20 mm corner (the profiles' `bed_exclude_area`). Delivered plates are centred on (128, 128) within 1 mm.

## Ready to print today: the V1.20 Orange plate

`ready/V1.20/` is the released plate with the battery backpack frame, its cap and the bench box cap (`rear_lid`):

- `X1C/zhengxia_V1.20_Orange_Bambu-X1C_2h20m.gcode.3mf`: copy onto the X1C SD card, print from the printer's screen.
- `Elegoo/zhengxia_V1.20_Orange_Elegoo-Centauri-Carbon_2h22m.gcode`: copy onto a USB stick for the Centauri.
- `Orange-plate.png`: what is on the plate and where, without opening a slicer. `PARTS-SHEET-V1.20.html`: every part
  with its purpose, screws and what it replaces.
- `Orange_Bambu_editable.3mf` and `Orange_Elegoo_editable.3mf`: the same plate as a project you can open, move parts
  on, add a part to and re-slice. **Open the Bambu one in Bambu Studio and the Elegoo one in OrcaSlicer**, never the
  other way round (Orca rejects the Bambu 02.08 config, Bambu Studio mangles the Orca one). Never open the sliced
  `.gcode.3mf` in OrcaSlicer either.
- `manifest-V1.20.json`: the release input: for each part its source STL, its print pose as a 4 x 4 matrix, the flat
  contact it must make with the bed, what it replaces and why. Use it as the template for your own release.

## Slicing a new or changed part by hand (the quick route)

1. Pick the print pose with the rules in [PRINT-ORIENTATION-PRACTICE.md](PRINT-ORIENTATION-PRACTICE.md) (R1 to R7,
   highest first): stand the part on a real flat face with its centre of mass well inside that face; keep screw seats,
   sliding fits and PCB pads off support; print holes vertical where roundness matters; keep the main load in the layer
   plane; place cosmetic faces deliberately; minimise support and keep it out of cavities; height, time and filament
   decide ties. `scripts/orientation_check.py` scores poses for you (NumPy, Trimesh, Shapely; `uv run --with shapely
   python cmu/hardware/printing/scripts/orientation_check.py MANIFEST.json`, manifest format as `ready/V1.20/manifest-V1.20.json`).
2. Open the editable project for the colour and machine (`ready/V1.20/*_editable.3mf` carry the Orange profiles;
   for another colour the only differences are the filament colour and, for Yellow and White, 0.15 mm elephant-foot
   compensation), delete the plate's parts, import your STL, put it in its pose, 12 mm or more between parts, centred
   on the bed.
3. Brim per part, never for the whole plate (the V1.9 lesson in [PRINT-LESSONS.md](PRINT-LESSONS.md)): a part whose
   tip angle in its pose is under 10 degrees gets an outer-only brim of 4 mm with a 0.1 mm gap, set on that object;
   every other part gets no brim. Leave tree support on (30 degrees, auto).
4. Slice, read the preview: the first layer should be the flat face you chose, supports only where you expected them,
   no support inside a screw pilot or a sliding seat.
5. Export: Bambu Studio "Export plate sliced file" gives the `.gcode.3mf`; OrcaSlicer "Export G-code" gives the
   `.gcode`. For the Centauri, add the preview block (next section) or the printer shows a blank tile.
6. Name the files `<who>_<version>_<Colour>_Bambu-X1C_<time>.gcode.3mf` and `<who>_<version>_<Colour>_Elegoo-Centauri-Carbon_<time>.gcode`
   with the slicer's estimated time rounded up to the minute, and hand them over as one folder per printer
   (`X1C/`, `Elegoo/`) plus one `<Colour>-plate.png` top view per plate, so nobody needs a slicer to see what is on it.

## Slicing from the command line (what the release pipeline does)

Both slicers have a CLI. Given a project `.3mf` whose plate already holds the parts in their poses with the profile
inside (the editable projects above are such files), the exact commands we use are:

```bash
# Bambu Lab X1 Carbon (Bambu Studio 02.08.02.61, macOS path)
/Applications/BambuStudio.app/Contents/MacOS/BambuStudio --allow-newer-file --arrange 0 --orient 0 --slice 0 \
  --min-save --export-3mf sliced.3mf --outputdir OUT_DIR PROJECT.3mf
```

```bash
# Elegoo Centauri Carbon (OrcaSlicer 2.4.2; on our Mac it runs from the mounted release image)
/Volumes/OrcaSlicer/OrcaSlicer.app/Contents/MacOS/OrcaSlicer --allow-newer-file --arrange 0 --orient 0 --slice 0 \
  --load-settings profiles/elegoo-machine.json;profiles/elegoo-process.json \
  --load-filaments profiles/elegoo-filament.json \
  --export-3mf sliced.3mf --outputdir OUT_DIR PROJECT.3mf
```

`--arrange 0 --orient 0` keep your poses and placements (never let the CLI rearrange a plate you checked);
`--slice 0` slices every plate. The Bambu `--min-save` matters: without it the exported file opens in Bambu Studio on
the Prepare tab with no toolpaths. The Bambu output also needs one `filament_self_index` entry per extruder variant
row in its `Metadata/project_settings.config`, or Bambu Studio 02.08 reports "Invalid configuration file"; the
pipeline script writes it (`bambu_self_index()` in `scripts/release_pipeline_v2.py`). The Elegoo `sliced.3mf`
contains `Metadata/plate_1.gcode`; that is the file for the USB stick after the preview block is added. The CLI
accepts JSON profiles exported from the slicer's preset menus; `profiles/elegoo-*.json` are ours, and the two
`*-project_settings.json` files are the complete settings of the Orange plates for reference.

## The Centauri preview

The Centauri Carbon firmware reads the thumbnail, time and grams from the G-code only when the word `OrcaSlicer`
(or `ElegooSlicer`) appears in the first 30 lines, and it stops reading at the first `G` or `M` command. Orca's CLI
writes no thumbnail, so our files carry a 144 x 144 PNG, base64, as

```
; THUMBNAIL_BLOCK_START
; thumbnail begin 144x144 <bytes>
; <base64 lines>
; thumbnail end
; THUMBNAIL_BLOCK_END
```

placed right after the `; HEADER_BLOCK_END` line and before the first command. A preview put above the header shows
nothing (every file through V1.5 had that fault). The GUI export of OrcaSlicer writes the block itself.
`cc_preview()` and `check_cc_preview()` in `scripts/release_pipeline_v2.py` insert and verify it.

## The release pipeline (reference)

For the robot we do not slice by hand: `scripts/release_pipeline_v2.py` takes a manifest like
`ready/V1.20/manifest-V1.20.json`, places every part by its recorded pose and placement, sets the per-part brim from
the tip angle (R1), runs both slicers with the commands above, reads the native output back (first-layer area, layer
height, printer identity, brim per object, the preview block) and packages the delivery with a hash chain
(`verify_release_v2.py` recomputes it; `review_brim.py RUN Colour Machine` reviews one plate). How it is run, and what
each check means, is in [RELEASE-PIPELINE.md](RELEASE-PIPELINE.md). The script is here unchanged apart from the path
of the Elegoo filament profile, but as is it is bound to things that are not in this repository: the donor projects
of the eight colour and machine combinations, the 89-part baseline stock manifest it audits against, and the bounded
CAD runner (it refuses to start unless `MICRODUCK_BOUNDED_CAD` is set in the environment). To reuse it, point the
manifest's `profiles` at your own project files (the two editable projects here are valid donors for Orange) and its
`baselineStock` at your own stock list, and set the environment variable yourself.

## Lessons that cost prints

[PRINT-LESSONS.md](PRINT-LESSONS.md) has the record. The short version: a plate-wide brim glued thin parts into a
sheet (brim per part now); 1.0 mm screw-head floors tore through on the box caps (3.0 mm now); plates placed by
hand-typed corner coordinates printed up to 63 mm off centre (centre the plate, check the offset); a Centauri preview
above the header never shows; a Bambu CLI export without `--min-save` and the self-index fix opens with no toolpaths.
