"""Draw the A7Z on the D-022 roof saddle that is on the robot (fitted.json extra_parts).

The robot carries the saddle printed on the V1.33 White plate: the D-022 layout, the board on four posts with its
40-pin header at the rear (board x -42.8..-12.8, PCB z 47..48.6, header housings x -41.8..-36.8, antenna on the rear
wall inner face at x -43.25; `guide/print/a7z-saddle/build/build.json` at robotics commit 2c179f0). The guide data has
moved on to the V3 saddle (D-028), whose board is the same A7Z turned 180 deg about z with its PCB 1.0 mm higher and
its centre 1.1 mm further forward. This script takes the guide's V3 board parts (sd_*: PCB, chips, sockets, microSD,
Dupont housings) and maps them onto the D-022 pose, x' = -(x + 26.7) - 27.8, y' = -y, z' = z - 1.0 (a rotation, so the
winding is kept), and draws the antenna strip as a 0.3 x 42 x 12 mm plate at its D-022 place. The 6530B heatsink, its
pad and the camera's right-angle plug are left out: none is fitted (CURRENT-STATE.json parts_a7z_extensions).

  /tmp/microduck-cad-venv/bin/python cmu/model/saddle_d022.py [--guide PATH]   -> visual/saddle-d022/*.stl (mm, robot frame)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

import make_visual

OUT = make_visual.MODEL_DIR / "visual/saddle-d022"
BOARD = ["sd_a7z_pcb", "sd_soc_a733", "sd_ram", "sd_pmic_area", "sd_wifi_module_fcu760k", "sd_usb_c_otg_dp",
         "sd_usb_c_power", "sd_micro_hdmi", "sd_fan_connector", "sd_ipex_socket", "sd_microsd_socket", "sd_microsd_card",
         "sd_csi_connector", "sd_pcie_fpc", "sd_uboot_button", "sd_underside_smd", "sd_dupont_housings"]
ANTENNA_BOUNDS = [[-43.4, -21.0, 49.5], [-43.1, 21.0, 61.5]]  # D-022 build.json antenna


def to_d022(v: np.ndarray) -> np.ndarray:
    return np.column_stack([-(v[:, 0] + 26.7) - 27.8, -v[:, 1], v[:, 2] - 1.0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--guide", default=str(make_visual.DEFAULT_GUIDE))
    args = ap.parse_args()
    guide = json.loads(Path(args.guide).read_text())
    parts = {p["key"]: p for p in guide["parts"]}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in BOARD:
        v, f = make_visual.decode(parts[key])
        trimesh.Trimesh(to_d022(v), f, process=False).export(OUT / f"{key}.stl")
        rows.append((key, parts[key]["name"], parts[key]["color"]))
    lo, hi = np.array(ANTENNA_BOUNDS)
    box = trimesh.creation.box(extents=hi - lo)
    box.apply_translation((lo + hi) / 2)
    box.export(OUT / "sd_antenna.stl")
    rows.append(("sd_antenna", "AC019 Wi-Fi antenna strip on the saddle rear wall (inner face)", "#c9a227"))
    for key, name, colour in rows:
        print(f"{key}\t{colour}\t{name[:70]}")


if __name__ == "__main__":
    main()
