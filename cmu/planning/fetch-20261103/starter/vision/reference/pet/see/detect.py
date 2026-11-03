#!/usr/bin/env python3
"""Open-vocabulary object detector for the MicroDuck pickup chain, a drop-in for `orange_target.detect`.

    from pet.see.detect import OpenVocabDetector, DetectorClient
    det = OpenVocabDetector(["orange block"])              # local model (Mac), YOLO-World by default
    det = DetectorClient("http://mac-mini:8765", ["keys"])  # same call, model runs on the Mac mini
    hit = det.detect(frame_bgr, accept)                     # None, or the dict orange_target.detect returns:
    hit["bbox"] = (x, y, w, h); hit["cx"], hit["cy"], hit["base_cx"], hit["area"], hit["n_blobs"]
    hit["score"], hit["label"]                              # added: detector confidence and the matched prompt

`accept(x, y, w, h) -> bool` is what approach.py / seepick.py pass (`approach.below_horizon`): a detection whose box
reaches above the horizon is not an object on the floor. An int is accepted too and means "reject boxes whose top
row is at or above this horizon row". The duck's own orange parts are rejected the way approach.py does it: its
`project` drops anything that lands under the body (x < 0.03 m); on top of that this detector drops boxes that lie
mostly inside an optional `body_mask` (the feet at the bottom of a head-down frame) and boxes labelled with one of
the `negatives` prompts (for example "robot foot"), which gives the matcher something else to call the duck's parts.

Three backends, same interface (`--backend`):
  yolo-world  ultralytics yolov8s-worldv2 + CLIP ViT-B/32 text tower (chosen; see README.md)
  owlv2       google/owlv2-base-patch16-ensemble (HF transformers)
  gdino       IDEA-Research/grounding-dino-tiny (HF transformers)
Weights live in pet/see/models/ (git-ignored): see README.md for the fetch commands.

CLI (run with ~/.venvs/duck-see/bin/python):
  detect.py image.jpg --prompts "orange block,keys" [--backend yolo-world] [--out annotated.jpg] [--horizon ROW]
  detect.py --serve --port 8765 --backend yolo-world         HTTP service: POST /detect (JPEG body) -> JSON
  detect.py --client http://mac-mini:8765 image.jpg --prompts keys --repeat 20     round-trip probe
  detect.py --build-bank "orange block,keys,small ball,remote control" --out models/prompt_bank.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
BACKENDS = ("yolo-world", "owlv2", "gdino")
DEFAULT_BACKEND = "yolo-world"
MIN_AREA = 60          # px, approach.MIN_AREA
DEFAULT_CONF = 0.10    # YOLO-World score threshold; OWLv2 and Grounding DINO use their own defaults below
CONF = {"yolo-world": 0.10, "owlv2": 0.15, "gdino": 0.30}
JPEG_QUALITY = 85
# A prompt may list alternatives separated by "|": every alternative is matched, the first one is the label.
# The 3D-printed orange part scores as a "toy" or "object" with YOLO-World's CLIP vocabulary and not as a "block"
# (out/prompt_study_yolo-world.json), so the pickup chain's object name gets these alternatives by default.
SYNONYMS = {"orange block": "orange block|orange toy|orange object",
            "orange object": "orange object|orange toy|orange block",
            "orange part": "orange part|orange toy|orange object"}
YOLO_WEIGHTS = "yolov8s-worldv2.pt"
CLIP_TEXT_DIR = "clip-text-b32"          # HF CLIPTextModelWithProjection, fp16, ViT-B/32 text tower only
PROMPT_BANK = "prompt_bank.npz"           # precomputed prompt embeddings for a machine without the text tower
OWLV2_DIR = "owlv2-base-patch16-ensemble"
GDINO_DIR = "grounding-dino-tiny"


# ----------------------------------------------------------------------------------------------- selection

def expand_prompts(prompts, synonyms=SYNONYMS):
    """["orange block", "keys|house keys"] -> (["orange block", "orange toy", "orange object", "keys", "house keys"],
    {"orange block": "orange block", "orange toy": "orange block", ..., "house keys": "keys"})."""
    flat, label_of = [], {}
    for p in prompts:
        p = str(p).strip()
        if not p:
            continue
        alts = [a.strip() for a in synonyms.get(p, p).split("|") if a.strip()]
        for a in alts:
            if a not in label_of:
                flat.append(a)
                label_of[a] = alts[0]
    return flat, label_of


def as_accept(accept):
    """Normalise approach's accept callable or a horizon row into a callable (or None)."""
    if accept is None or callable(accept):
        return accept
    row = int(accept)
    return lambda x, y, w, h: y > row


def box_in_mask(bbox, mask, frac=0.5):
    """True when at least `frac` of the box lies on non-zero mask pixels."""
    x, y, w, h = bbox
    if mask is None or w <= 0 or h <= 0:
        return False
    sub = mask[max(0, y):y + h, max(0, x):x + w]
    return sub.size > 0 and float((sub > 0).mean()) >= frac


def select(dets, accept=None, min_area=MIN_AREA, body_mask=None, negatives=()):
    """orange_target's rule, scored: the best accepted detection, or None.

    dets: list of {"bbox": (x, y, w, h), "score": float, "label": str}. Order of preference: highest score; a box
    whose area is under min_area, that `accept` rejects, that lies mostly in body_mask or carries a negative label
    is skipped (orange_target skips a rejected blob and tries the next one the same way).
    """
    acc = as_accept(accept)
    best = None
    for d in sorted(dets, key=lambda d: (-d["score"], -d["bbox"][2] * d["bbox"][3])):
        x, y, w, h = (int(v) for v in d["bbox"])
        if w * h < min_area or d.get("label") in negatives:
            continue
        if acc is not None and not acc(x, y, w, h):
            continue
        if box_in_mask((x, y, w, h), body_mask):
            continue
        best = d
        break
    if best is None:
        return None
    x, y, w, h = (int(v) for v in best["bbox"])
    # orange_target: cx, cy = blob centroid; base_cx = centre column of the blob's lowest rows. A box has no mask,
    # so the footprint centre is the centre of the bottom edge, which is the box centre column.
    return {"cx": x + w / 2.0, "cy": y + h / 2.0, "base_cx": x + w / 2.0, "area": int(w * h), "bbox": (x, y, w, h),
            "n_blobs": len(dets), "score": float(best["score"]), "label": str(best.get("label", ""))}


# ----------------------------------------------------------------------------------------------- text encoder

class TextEncoder:
    """CLIP ViT-B/32 text embeddings for YOLO-World, unit length, in this order of availability:
    1. HF CLIPTextModelWithProjection in models/clip-text-b32 (same weights as OpenAI's clip ViT-B/32 text tower),
    2. the `clip` package ultralytics uses (downloads the full 338 MB ViT-B/32 on first use),
    3. models/prompt_bank.npz, precomputed with --build-bank on a machine that has 1 or 2.
    """

    def __init__(self, models_dir=MODELS_DIR, device="cpu"):
        self.models_dir, self.device = Path(models_dir), device
        self.kind, self.model, self.tok = None, None, None
        self.bank = {}
        bank = self.models_dir / PROMPT_BANK
        if bank.exists():
            z = np.load(bank, allow_pickle=False)
            self.bank = {str(p): f.astype(np.float32) for p, f in zip(z["prompts"], z["feats"])}

    def _load(self):
        if self.kind is not None:
            return
        import torch
        d = self.models_dir / CLIP_TEXT_DIR
        if d.exists():
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
            self.tok = CLIPTokenizer.from_pretrained(d)
            self.model = CLIPTextModelWithProjection.from_pretrained(d, dtype=torch.float32).eval()
            self.kind = "hf"
            return
        try:
            import clip  # noqa: F401
            from ultralytics.nn.text_model import build_text_model
            self.model = build_text_model("clip:ViT-B/32", device=torch.device(self.device))
            self.kind = "clip"
        except Exception:
            self.kind = "bank"

    def encode(self, prompts):
        """-> float32 [n, 512], each row unit length."""
        prompts = list(prompts)
        if all(p in self.bank for p in prompts):
            return np.stack([self.bank[p] for p in prompts])
        self._load()
        import torch
        if self.kind == "hf":
            with torch.no_grad():
                t = self.tok(prompts, padding=True, truncation=True, max_length=77, return_tensors="pt")
                e = self.model(**t).text_embeds
                e = e / e.norm(dim=-1, keepdim=True)
            feats = e.cpu().numpy().astype(np.float32)
        elif self.kind == "clip":
            feats = self.model.encode_text(self.model.tokenize(prompts)).cpu().numpy().astype(np.float32)
        else:
            missing = [p for p in prompts if p not in self.bank]
            raise RuntimeError(f"no CLIP text tower in {self.models_dir} and prompts not in the prompt bank: "
                               f"{missing}; bank has {sorted(self.bank)}")
        for p, f in zip(prompts, feats):
            self.bank[p] = f
        return feats

    def save_bank(self, path):
        prompts = sorted(self.bank)
        np.savez(path, prompts=np.array(prompts), feats=np.stack([self.bank[p] for p in prompts]))
        return prompts


# ----------------------------------------------------------------------------------------------- local detector

class OpenVocabDetector:
    """detect(frame_bgr, accept=None) -> orange_target-style dict or None. `prompts` are the object names; the best
    accepted detection of any prompt wins (one prompt for the pickup chain, several for the evidence runs)."""

    def __init__(self, prompts, device="cpu", backend=DEFAULT_BACKEND, conf=None, models_dir=MODELS_DIR,
                 min_area=MIN_AREA, body_mask=None, negatives=(), imgsz=640, synonyms=SYNONYMS):
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r} not in {BACKENDS}")
        self.backend, self.device, self.models_dir = backend, device, Path(models_dir)
        self.conf = CONF[backend] if conf is None else float(conf)
        self.min_area, self.body_mask, self.negatives, self.imgsz = min_area, body_mask, tuple(negatives), imgsz
        self.synonyms = dict(synonyms)
        self.prompts, self.queries, self.label_of = [], [], {}   # prompts as given; queries = expanded alternatives
        self.last, self.last_ms, self._lock = [], None, threading.Lock()
        self._model = self._proc = self._text = None
        t = time.perf_counter()
        getattr(self, "_load_" + backend.replace("-", "_"))()
        self.load_s = time.perf_counter() - t
        self.set_prompts(list(prompts) + [n for n in self.negatives if n not in prompts])

    # -- loading
    def _load_yolo_world(self):
        from ultralytics import YOLO
        self._model = YOLO(str(self.models_dir / YOLO_WEIGHTS))
        self._text = TextEncoder(self.models_dir, self.device)

    def _load_owlv2(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        d = self.models_dir / OWLV2_DIR
        self._proc = Owlv2Processor.from_pretrained(d)
        self._model = Owlv2ForObjectDetection.from_pretrained(d).to(self.device).eval()

    def _load_gdino(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        d = self.models_dir / GDINO_DIR
        self._proc = AutoProcessor.from_pretrained(d)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(d).to(self.device).eval()

    def set_prompts(self, prompts):
        prompts = [str(p).strip() for p in prompts if str(p).strip()]
        if not prompts:
            raise ValueError("at least one prompt")
        queries, label_of = expand_prompts(prompts, self.synonyms)
        with self._lock:
            self.prompts, self.queries, self.label_of = prompts, queries, label_of
            if self.backend == "yolo-world":
                import torch
                feats = torch.from_numpy(self._text.encode(queries))
                names = {i: p for i, p in enumerate(queries)}
                models = [self._model.model]                           # ultralytics WorldModel
                pred = self._model.predictor
                if pred is not None:                                   # its AutoBackend holds a fused copy
                    models.append(pred.model.model)
                    pred.model.names = names
                for wm in models:
                    wm.txt_feats = feats.reshape(1, len(queries), feats.shape[-1]).to(next(wm.parameters()).device)
                    wm.model[-1].nc = len(queries)
                    wm.names = names

    # -- inference
    def all(self, frame_bgr):
        """Every detection above the threshold: [{"bbox": (x, y, w, h), "score", "label"}], frame pixels."""
        t = time.perf_counter()
        with self._lock:
            dets = getattr(self, "_all_" + self.backend.replace("-", "_"))(frame_bgr)
        self.last_ms = (time.perf_counter() - t) * 1000.0
        self.last = dets
        return dets

    def _all_yolo_world(self, frame):
        r = self._model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy, sc, cl = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        for (x0, y0, x1, y1), s, c in zip(xyxy, sc, cl):
            out.append(self._det(x0, y0, x1, y1, s, self.label_of[self.queries[int(c)]], frame.shape))
        return out

    def _all_owlv2(self, frame):
        import torch
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self._proc(text=[self.queries], images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self._model(**inp)
        # the processor pads the picture to a square (bottom and right), so boxes come back on that square
        s = max(h, w)
        res = self._proc.post_process_grounded_object_detection(o, threshold=self.conf, target_sizes=[(s, s)],
                                                                text_labels=[self.queries])[0]
        out = []
        for (x0, y0, x1, y1), sc, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                                             res["text_labels"]):
            out.append(self._det(x0, y0, x1, y1, sc, self.label_of.get(lab, lab), frame.shape))
        return out

    def _all_gdino(self, frame):
        import torch
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        text = ". ".join(p.lower() for p in self.queries) + "."
        inp = self._proc(images=rgb, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            o = self._model(**inp)
        res = self._proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=self.conf,
                                                                text_threshold=0.25, target_sizes=[(h, w)])[0]
        labels = res.get("text_labels") or res.get("labels")
        out = []
        for (x0, y0, x1, y1), sc, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), labels):
            out.append(self._det(x0, y0, x1, y1, sc, self._nearest_prompt(str(lab)), frame.shape))
        return out

    def _nearest_prompt(self, label):
        """Grounding DINO returns the matched phrase (sometimes merged or partial); map it to one of the prompts."""
        lab = label.lower().strip()
        for p in self.queries:
            if p.lower() == lab:
                return self.label_of[p]
        for p in self.queries:
            if p.lower() in lab or lab in p.lower():
                return self.label_of[p]
        return label

    @staticmethod
    def _det(x0, y0, x1, y1, score, label, shape):
        h, w = shape[:2]
        x0, y0 = max(0, int(round(float(x0)))), max(0, int(round(float(y0))))
        x1, y1 = min(w, int(round(float(x1)))), min(h, int(round(float(y1))))
        return {"bbox": (x0, y0, max(0, x1 - x0), max(0, y1 - y0)), "score": float(score), "label": str(label)}

    def detect(self, frame_bgr, accept=None):
        return select(self.all(frame_bgr), accept, self.min_area, self.body_mask, self.negatives)


# ----------------------------------------------------------------------------------------------- HTTP service

def serve(det, host="0.0.0.0", port=8765, log=print):
    """POST /detect?conf=F&prompts=a|b (JPEG body) -> {"dets": [...], "ms": float, "shape": [h, w]}; GET /health."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json(200, {"backend": det.backend, "prompts": det.prompts, "device": det.device,
                             "conf": det.conf, "load_s": round(det.load_s, 2)})

        def do_POST(self):
            try:
                q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                n = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(n)
                frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    return self._json(400, {"error": "not a decodable image"})
                if "prompts" in q:
                    prompts = [p for p in q["prompts"][0].split("|") if p.strip()]
                    if prompts != det.prompts:
                        det.set_prompts(prompts)
                if "conf" in q:
                    det.conf = float(q["conf"][0])
                t = time.perf_counter()
                dets = det.all(frame)
                self._json(200, {"dets": [{"bbox": list(d["bbox"]), "score": round(d["score"], 4), "label": d["label"]}
                                          for d in dets], "ms": round((time.perf_counter() - t) * 1000, 1),
                                 "shape": list(frame.shape[:2]), "backend": det.backend, "prompts": det.prompts})
            except Exception as e:  # noqa: BLE001  (the robot side must see the reason, not a dropped socket)
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            log(f"{self.address_string()} {fmt % args}")

    srv = ThreadingHTTPServer((host, port), H)
    log(f"pet.see service: {det.backend} on {det.device}, prompts {det.prompts}, http://{host}:{port}/detect")
    return srv


class DetectorClient:
    """Same detect(frame, accept) as OpenVocabDetector; the model runs behind `url` (serve() above). The horizon
    rule, body mask and negatives are applied here, on the robot, where approach.py's kinematics live."""

    def __init__(self, url, prompts, conf=None, timeout=1.0, min_area=MIN_AREA, body_mask=None, negatives=(),
                 jpeg_quality=JPEG_QUALITY):
        self.url, self.timeout, self.conf = url.rstrip("/"), timeout, conf
        self.prompts = list(prompts) + [n for n in negatives if n not in prompts]
        self.min_area, self.body_mask, self.negatives, self.q = min_area, body_mask, tuple(negatives), jpeg_quality
        self.last, self.last_ms, self.last_server_ms, self.errors = [], None, None, 0

    def health(self):
        with urllib.request.urlopen(self.url + "/health", timeout=self.timeout) as r:
            return json.loads(r.read())

    def all(self, frame_bgr):
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        if not ok:
            raise RuntimeError("jpeg encode failed")
        q = {"prompts": "|".join(self.prompts)}
        if self.conf is not None:
            q["conf"] = str(self.conf)
        req = urllib.request.Request(self.url + "/detect?" + urllib.parse.urlencode(q), data=jpg.tobytes(),
                                     headers={"Content-Type": "image/jpeg"}, method="POST")
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                rep = json.loads(r.read())
        except Exception:
            self.errors += 1
            raise
        self.last_ms = (time.perf_counter() - t) * 1000.0
        self.last_server_ms = rep.get("ms")
        self.last = [{"bbox": tuple(d["bbox"]), "score": d["score"], "label": d["label"]} for d in rep["dets"]]
        return self.last

    def detect(self, frame_bgr, accept=None):
        try:
            dets = self.all(frame_bgr)
        except Exception:
            return None   # a lost frame: approach.py treats None as "nothing seen" and keeps its estimate
        return select(dets, accept, self.min_area, self.body_mask, self.negatives)


# ----------------------------------------------------------------------------------------------- drawing, CLI

def annotate(frame_bgr, dets, hit=None, horizon=None, title=None):
    img = frame_bgr.copy()
    if horizon is not None:
        cv2.line(img, (0, int(horizon)), (img.shape[1], int(horizon)), (200, 200, 0), 1)
    for d in dets:
        x, y, w, h = d["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 165, 255), 1)
        cv2.putText(img, f"{d['label']} {d['score']:.2f}", (x, max(12, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 165, 255), 1)
    if hit is not None:
        x, y, w, h = hit["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.drawMarker(img, (int(hit["base_cx"]), y + h), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(img, title or (f"{hit['label']} {hit['score']:.2f}" if hit else "no object"), (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if hit else (0, 0, 255), 2)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?")
    ap.add_argument("--prompts", default="orange block", help="comma separated object names")
    ap.add_argument("--negatives", default="", help="comma separated names whose hits are dropped (robot foot)")
    ap.add_argument("--backend", default=DEFAULT_BACKEND, choices=BACKENDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--horizon", type=int, default=None, help="reject boxes whose top is at or above this row")
    ap.add_argument("--out", default=None)
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--client", default=None, help="service URL: send the image there instead of a local model")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--build-bank", default=None, help="comma separated prompts to embed into --out (.npz)")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    a = ap.parse_args(argv)
    prompts = [p.strip() for p in a.prompts.split(",") if p.strip()]
    negatives = [p.strip() for p in a.negatives.split(",") if p.strip()]
    if a.build_bank:
        enc = TextEncoder(a.models_dir, a.device)
        enc.encode([p.strip() for p in a.build_bank.split(",") if p.strip()])
        out = a.out or str(Path(a.models_dir) / PROMPT_BANK)
        print("bank", out, enc.save_bank(out), "via", enc.kind)
        return
    if a.serve:
        det = OpenVocabDetector(prompts, a.device, a.backend, a.conf, a.models_dir, negatives=negatives)
        srv = serve(det, a.host, a.port, log=lambda s: print(s, flush=True))
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
        return
    if not a.image:
        ap.error("an image, --serve or --build-bank")
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    if a.client:
        det = DetectorClient(a.client, prompts, a.conf, negatives=negatives)
        print("service:", det.health())
    else:
        det = OpenVocabDetector(prompts, a.device, a.backend, a.conf, a.models_dir, negatives=negatives)
        print(f"loaded {a.backend} in {det.load_s:.1f} s")
    hit, ms = None, []
    for _ in range(max(1, a.repeat)):
        hit = det.detect(img, a.horizon)
        ms.append(det.last_ms)
    extra = f", server {det.last_server_ms} ms" if a.client else ""
    print(f"{len(det.last)} detections, {np.median(ms):.0f} ms median over {len(ms)} (first {ms[0]:.0f}){extra}")
    for d in det.last:
        print(f"  {d['label']:20s} {d['score']:.2f} bbox {d['bbox']}")
    print("hit:", None if hit is None else {k: v for k, v in hit.items()})
    if a.out:
        cv2.imwrite(a.out, annotate(img, det.last, hit, a.horizon))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
