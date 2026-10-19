#!/usr/bin/env python3
"""Orange-blob detector for the MicroDuck head camera (same recipe as omx-local/locate.py orange_mask).

    from orange_target import detect
    hit = detect(frame_bgr)                   # None when no blob >= min_area px, else a dict
    hit["cx"], hit["cy"]                       # centroid in pixels (x right, y down)
    hit["area"]                                # pixels in the largest connected component
    hit["bbox"]                                # (x, y, w, h)
    hit = detect(frame_bgr, mask=True)         # also returns hit["mask"], a uint8 0/255 image of that component

Recipe: BGR -> HSV, inRange(hsv_lo, hsv_hi), morphological open 5x5, largest connected component.

CLI (run with ~/.venvs/microduck-walk/bin/python):
    orange_target.py image.jpg [--out annotated.jpg] [--lo 5,120,120 --hi 22,255,255] [--min-area 80] [--hist]
    --hist prints the H/S/V percentiles of the whole image, to see what the floor/desk produces.
"""
import argparse

import cv2
import numpy as np

HSV_LO = (5, 120, 120)
HSV_HI = (22, 255, 255)
MIN_AREA = 80
KERNEL = np.ones((5, 5), np.uint8)


def detect(frame_bgr, hsv_lo=HSV_LO, hsv_hi=HSV_HI, min_area=MIN_AREA, mask=False):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mk = cv2.inRange(hsv, tuple(hsv_lo), tuple(hsv_hi))
    mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, KERNEL)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mk)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[k, cv2.CC_STAT_AREA])
    if area < min_area:
        return None
    x, y, w, h = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    # base_cx: centre column of the blob's lowest rows (the object's footprint); the whole-blob centroid of a tall
    # object seen from above sits outward of its base, which put a 9 cm box 4 cm off in y at 15 cm (sim, 2026-10-16)
    foot = lab[y + h - max(1, h // 10):y + h, x:x + w] == k
    cols = np.nonzero(foot.any(axis=0))[0]
    base_cx = float(x + (cols.min() + cols.max()) / 2.0) if cols.size else float(cents[k, 0])
    out = {"cx": float(cents[k, 0]), "cy": float(cents[k, 1]), "base_cx": base_cx, "area": area, "bbox": (x, y, w, h),
           "n_blobs": int(n - 1)}
    if mask:
        out["mask"] = (lab == k).astype(np.uint8) * 255
    return out


def annotate(frame_bgr, hit):
    img = frame_bgr.copy()
    if hit is None:
        cv2.putText(img, "no orange", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    x, y, w, h = hit["bbox"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.drawMarker(img, (int(hit["cx"]), int(hit["cy"])), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    cv2.putText(img, f"cx {hit['cx']:.0f} cy {hit['cy']:.0f} area {hit['area']}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def hsv_summary(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    out = {}
    for name, col in zip("HSV", hsv.T):
        out[name] = [int(v) for v in np.percentile(col, [5, 25, 50, 75, 95])]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="write an annotated copy here")
    ap.add_argument("--lo", default=",".join(map(str, HSV_LO)))
    ap.add_argument("--hi", default=",".join(map(str, HSV_HI)))
    ap.add_argument("--min-area", type=int, default=MIN_AREA)
    ap.add_argument("--hist", action="store_true", help="print H/S/V percentiles (5/25/50/75/95) of the image")
    ap.add_argument("--mask-out", default=None, help="write the in-range mask (before component selection) here")
    args = ap.parse_args()
    lo = tuple(int(v) for v in args.lo.split(","))
    hi = tuple(int(v) for v in args.hi.split(","))
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    hit = detect(img, lo, hi, args.min_area, mask=True)
    if hit is None:
        print("no orange blob (largest component below min_area or none in range)")
    else:
        print(f"cx {hit['cx']:.1f} cy {hit['cy']:.1f} area {hit['area']} bbox {hit['bbox']} "
              f"blobs {hit['n_blobs']} image {img.shape[1]}x{img.shape[0]}")
    if args.hist:
        print("HSV percentiles 5/25/50/75/95:", hsv_summary(img))
    if args.mask_out:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imwrite(args.mask_out, cv2.morphologyEx(cv2.inRange(hsv, lo, hi), cv2.MORPH_OPEN, KERNEL))
    if args.out:
        cv2.imwrite(args.out, annotate(img, hit))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
