#!/usr/bin/env python3
"""Text-prompted object detector for the MicroDuck head camera: the drop-in for orange_target.detect.

    from learned_target import detect
    hit = detect(frame_bgr)                    # None, or the dict orange_target.detect returns plus "score" and "label"
    hit["bbox"]                                # (x, y, w, h) of the best box, frame pixels
    hit["cx"], hit["cy"], hit["base_cx"]       # box centre; base_cx equals cx (a box has no footprint row)
    hit["area"]                                # box area in px (w * h), not a blob pixel count
    hit["n_blobs"]                             # boxes above the threshold after NMS
    hit["score"], hit["label"]                 # confidence of the chosen box and the baked prompt that scored it

The model is an open-vocabulary detector (Ultralytics YOLOE-11s, 384x640 input) exported to ONNX with its text prompts
baked in as classes (export_learned_target.py, Mac mini, torch); this module needs onnxruntime, numpy and OpenCV only,
so it runs in the robot's venv. A YOLO-World v2 export loads the same way. The prompts are read from the ONNX metadata
("names"); `prompts` picks which of them count.

In approach.py / seepick.py main() the swap is one line:
    detect = lambda f, accept=None: learned_target.detect(f, accept=accept)

Model file: $LEARNED_TARGET_MODEL, else runtime-overlay/models/learned_target.onnx (next to walk61.onnx; that folder
is git-ignored).

CLI (run with ~/.venvs/microduck-walk/bin/python):
    learned_target.py image.png [--out annotated.jpg] [--model x.onnx] [--conf 0.1] [--prompts "small orange ball"]
"""
import argparse
import ast
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("LEARNED_TARGET_MODEL") or os.path.join(
    HERE, "..", "..", "..", "hardware-design-20260912", "whole-robot-20260912", "runtime-overlay", "models",
    "learned_target.onnx")
PROMPTS = ("small orange ball",)   # which baked prompts count; None = all of them
CONF = 0.10                        # 2026-11-12, 1360 robot frames: the false boxes score <= 0.054, the office balls
#                                    0.57-0.81 (learned-target-20261112/frames.json holds the test subset)
IOU = 0.5
MIN_AREA = 16                      # box area, px
MAX_SIDE = 0.5                     # a box wider or taller than this fraction of the frame is not a small object
THREADS = 1                        # the brain shares the A7Z's cores; more threads is the caller's decision

_cache = {}


class Model:
    """One ONNX session plus what its metadata says: input size, prompt names."""

    def __init__(self, path=None, threads=THREADS):
        import onnxruntime as ort
        self.path = os.path.abspath(path or MODEL)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # idle pool threads must not spin
        self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.h, self.w = int(inp.shape[2]), int(inp.shape[3])
        self.half = "float16" in inp.type
        meta = self.sess.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta["names"]) if "names" in meta else {}
        self.names = [names[i] for i in sorted(names)]
        self.nc = len(self.names) or None

    def infer(self, frame_bgr):
        """-> (boxes xyxy in frame pixels [N, 4], scores [N, nc]) for every anchor, before any threshold."""
        fh, fw = frame_bgr.shape[:2]
        r = min(self.h / fh, self.w / fw)
        nw, nh = int(round(fw * r)), int(round(fh * r))
        img = frame_bgr if (nw, nh) == (fw, fh) else cv2.resize(
            frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
        px, py = (self.w - nw) // 2, (self.h - nh) // 2
        canvas = np.full((self.h, self.w, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = img
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float16 if self.half else np.float32) / 255.0
        out = self.sess.run(None, {self.input: np.ascontiguousarray(x)})[0][0].astype(np.float32)   # [4 + nc (+ k), N]
        nc = self.nc or out.shape[0] - 4
        cx, cy, w, h = out[0], out[1], out[2], out[3]
        boxes = np.stack([cx - w / 2 - px, cy - h / 2 - py, cx + w / 2 - px, cy + h / 2 - py], axis=1) / r
        return boxes, out[4:4 + nc].T


def load(path=None, threads=THREADS):
    key = (os.path.abspath(path or MODEL), int(threads))
    if key not in _cache:
        _cache[key] = Model(key[0], threads)
    return _cache[key]


def _nms(boxes, scores, iou):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        order = rest[inter / (a + b - inter + 1e-9) <= iou]
    return keep


def candidates(frame_bgr, model=None, conf=CONF, prompts=PROMPTS, iou=IOU, threads=THREADS, raw=None):
    """Every box above conf after NMS, best first: [{"bbox": (x, y, w, h), "score": s, "label": prompt}, ...].
    raw: the (boxes, scores) of an earlier Model.infer on this frame, to try other prompts without a second run."""
    m = model if isinstance(model, Model) else load(model, threads)
    boxes, scores = raw if raw is not None else m.infer(frame_bgr)
    if prompts is not None and m.names:
        missing = [p for p in prompts if p not in m.names]
        if missing:
            raise ValueError(f"prompts {missing} are not baked into {m.path}; it has {m.names}")
        cols = [m.names.index(p) for p in prompts]
    else:
        cols = list(range(scores.shape[1]))
    sub = scores[:, cols]
    best = sub.argmax(axis=1)
    sc = sub[np.arange(sub.shape[0]), best]
    idx = np.nonzero(sc >= conf)[0]
    if not idx.size:
        return []
    fh, fw = frame_bgr.shape[:2]
    b = boxes[idx].copy()
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, fw)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, fh)
    out = []
    for k in _nms(b, sc[idx], iou):
        x0, y0, x1, y1 = b[k]
        x, y = int(round(x0)), int(round(y0))
        w, h = int(round(x1)) - x, int(round(y1)) - y
        if w < 1 or h < 1:
            continue
        col = cols[int(best[idx[k]])]
        out.append({"bbox": (x, y, w, h), "score": float(sc[idx[k]]),
                    "label": m.names[col] if m.names else str(col)})
    return out


def detect(frame_bgr, model=None, conf=CONF, min_area=MIN_AREA, mask=False, accept=None, prompts=PROMPTS,
           max_side=MAX_SIDE, threads=THREADS):
    """Best-scoring box; accept(x, y, w, h) -> bool, when given, skips boxes it rejects (e.g. above the horizon),
    exactly as orange_target.detect does. Returns None when no box passes."""
    fh, fw = frame_bgr.shape[:2]
    cands = [c for c in candidates(frame_bgr, model, conf, prompts, threads=threads)
             if c["bbox"][2] * c["bbox"][3] >= min_area
             and c["bbox"][2] <= max_side * fw and c["bbox"][3] <= max_side * fh]
    for c in cands:
        x, y, w, h = c["bbox"]
        if accept is None or accept(x, y, w, h):
            cx = x + w / 2.0
            out = {"cx": float(cx), "cy": float(y + h / 2.0), "base_cx": float(cx), "area": int(w * h),
                   "bbox": (x, y, w, h), "n_blobs": len(cands), "score": c["score"], "label": c["label"]}
            if mask:
                mk = np.zeros((fh, fw), np.uint8)
                mk[y:y + h, x:x + w] = 255
                out["mask"] = mk
            return out
    return None


def annotate(frame_bgr, hit, others=()):
    img = frame_bgr.copy()
    for c in others:
        x, y, w, h = c["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 1)
    if hit is None:
        cv2.putText(img, "nothing found", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    x, y, w, h = hit["bbox"]
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.putText(img, f"{hit['label']} {hit['score']:.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default=None, help="write an annotated copy here")
    ap.add_argument("--model", default=None)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--prompts", default="|".join(PROMPTS), help="baked prompts to use, '|' separated; 'all' for all")
    ap.add_argument("--threads", type=int, default=THREADS)
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    prompts = None if args.prompts == "all" else tuple(args.prompts.split("|"))
    m = load(args.model, args.threads)
    print(f"model {m.path} input {m.h}x{m.w} prompts {m.names}")
    t0 = time.perf_counter()
    hit = detect(img, m, args.conf, prompts=prompts)
    dt = time.perf_counter() - t0
    if hit is None:
        print(f"nothing found ({dt * 1000:.0f} ms)")
    else:
        print(f"bbox {hit['bbox']} score {hit['score']:.3f} label {hit['label']!r} boxes {hit['n_blobs']} "
              f"({dt * 1000:.0f} ms)")
    if args.out:
        cv2.imwrite(args.out, annotate(img, hit, candidates(img, m, args.conf, prompts)))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
