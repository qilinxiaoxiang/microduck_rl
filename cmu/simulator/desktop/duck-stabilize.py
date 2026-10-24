#!/usr/bin/env python3
"""Digital stabilization of the MicroDuck head-camera stream, run on the Mac by app/a7z-bridge.mjs.

stdin and stdout carry frames as a 4-byte big-endian length followed by one JPEG. For every input frame one
stabilized JPEG comes out, in order. stderr gets one JSON line per second:
  {"stab": {"frames": n, "raw_deg": .., "out_deg": .., "raw_px": .., "out_px": .., "lost": n}}
raw_* is the rms frame-to-frame camera motion seen in the input (rotation in degrees, translation in output pixels),
out_* the same motion left in the output, lost the frames whose motion could not be measured (shown uncorrected).

Method: features tracked with pyramidal Lucas-Kanade from the previous frame give a similarity transform (shift and
roll about the image centre, RANSAC); the transforms add up to the camera path, a causal exponential average of the path
is the wanted smooth path (no delay: a live view cannot wait for later frames), and each frame is warped from its own
path point to the smooth one, then zoomed by ZOOM so the warped borders stay outside the picture. The correction is
clamped to what the zoom margin can hide, so a deliberate turn passes through with a short lag.

The duck's walking shake is rotation of the head (trunk roll and pitch at the 1.8 Hz stride, plus 5-8 Hz jitter on the
real robot, 2026-10-23); motion blur inside one exposure is not removed.
"""
import json
import math
import struct
import sys
import time

import cv2
import numpy as np

ALPHA = 0.2          # smooth path = path low-passed by this factor per frame (8 fps: about 0.6 s time constant)
ZOOM = 1.15          # crop that hides the warped borders
TRACK_W = 240        # width of the grey image the features are tracked in
MIN_POINTS = 12


def read_frame(stream):
    head = stream.read(4)
    if len(head) < 4:
        return None
    n = struct.unpack(">I", head)[0]
    data = stream.read(n)
    return data if len(data) == n else None


def similarity(prev, cur):
    """(dx, dy, da) moving prev onto cur in tracking-image pixels and radians, or None."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=6, blockSize=5)
    if pts is None or len(pts) < MIN_POINTS:
        return None
    nxt, ok, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    good = ok.ravel() == 1
    if good.sum() < MIN_POINTS:
        return None
    m, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if m is None or inl is None or inl.sum() < MIN_POINTS:
        return None
    da = math.atan2(m[1, 0], m[0, 0])
    # express the shift about the image centre so rotation and shift separate
    h, w = prev.shape
    c = np.array([w / 2, h / 2])
    shift = m[:, :2] @ c + m[:, 2] - c
    return float(shift[0]), float(shift[1]), da


class Stabilizer:
    def __init__(self):
        self.prev = None
        self.path = np.zeros(3)     # x, y in full-image pixels, angle in radians
        self.smooth = np.zeros(3)
        self.last_corr = np.zeros(3)
        self.stats = {"frames": 0, "raw": [], "out": [], "lost": 0}

    def step(self, img):
        h, w = img.shape[:2]
        scale = w / TRACK_W
        grey = cv2.cvtColor(cv2.resize(img, (TRACK_W, round(h / scale)), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        motion = similarity(self.prev, grey) if self.prev is not None else (0.0, 0.0, 0.0)
        self.prev = grey
        if motion is None:
            self.stats["lost"] += 1
            motion = (0.0, 0.0, 0.0)
        d = np.array([motion[0] * scale, motion[1] * scale, motion[2]])
        self.path += d
        self.smooth += ALPHA * (self.path - self.smooth)
        corr = self.smooth - self.path
        # the zoom hides a shift of (1 - 1/ZOOM) / 2 of the picture and about as much rotation at the corners
        lim = np.array([w * (1 - 1 / ZOOM) / 2, h * (1 - 1 / ZOOM) / 2, (1 - 1 / ZOOM) * 0.9])
        clipped = np.clip(corr, -lim, lim)
        self.smooth += clipped - corr          # a clamped correction drags the smooth path along: no runaway
        corr = clipped
        out_motion = d + corr - self.last_corr
        self.last_corr = corr
        self.stats["frames"] += 1
        self.stats["raw"].append(d)
        self.stats["out"].append(out_motion)
        # da follows image axes (y down, + turns +x towards +y); getRotationMatrix2D's + angle is the opposite way
        m = cv2.getRotationMatrix2D((w / 2, h / 2), -math.degrees(corr[2]), ZOOM)
        m[:, 2] += corr[:2] * ZOOM
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def report(self):
        s = self.stats
        def rms(rows, cols):
            if not rows:
                return 0.0
            a = np.array(rows)[:, cols]
            return float(np.sqrt((a ** 2).sum(axis=-1).mean())) if a.ndim > 1 else float(np.sqrt((a ** 2).mean()))
        out = {"frames": s["frames"], "lost": s["lost"],
               "raw_deg": round(math.degrees(rms(s["raw"], 2)), 2), "out_deg": round(math.degrees(rms(s["out"], 2)), 2),
               "raw_px": round(rms(s["raw"], slice(0, 2)), 1), "out_px": round(rms(s["out"], slice(0, 2)), 1)}
        self.stats = {"frames": 0, "raw": [], "out": [], "lost": 0}
        return out


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    stab, last = Stabilizer(), time.monotonic()
    while True:
        data = read_frame(inp)
        if data is None:
            return
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            jpeg = data                                   # undecodable: pass it through untouched
        else:
            ok, enc = cv2.imencode(".jpg", stab.step(img), [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg = enc.tobytes() if ok else data
        out.write(struct.pack(">I", len(jpeg)) + jpeg)
        out.flush()
        if time.monotonic() - last >= 1.0:
            last = time.monotonic()
            sys.stderr.write(json.dumps({"stab": stab.report()}) + "\n")
            sys.stderr.flush()


if __name__ == "__main__":
    main()
