"""Layer A2 (PLAN-continuous-vision.md section 3): an open-vocabulary detector seeds a video tracker that keeps the id
across frames (the Helpful DoggyBot recipe, arXiv 2410.00231). Runs on the Mac mini, consumes the MicroDuck head camera,
emits track_iface.py JSON lines, one per processed frame.

  tracker.py --query "orange ball" --source ssh-a7z --out tracks.jsonl
  tracker.py --query "ruler,basket" --source dir:testdata/bend --realtime --out /tmp/t.jsonl
  tracker.py --query person --source mjpeg-url:http://host:port/stream --out -

Sources
  ssh-a7z              ssh to the robot, pipe a7z_frames.py into the brain venv's python; capture time and frame index
                       survive; the clock offset is measured over UDP (a7z_frames.TIME_PORT) and t_cap is reported in
                       this machine's clock so age = t_out - t_cap is meaningful. Stale frames are dropped: the newest
                       frame in the pipe is processed, skipped ones are counted.
  mjpeg-url:<url>      a multipart JPEG stream (a7z_cam.py --mjpeg behind an http bridge); capture time is lost, so
                       t_cap = arrival - a7z_cam.LATENCY_S and the frame index counts arrivals.
  dir:<path>           recorded frames (png/jpg, sorted by name); --realtime paces them at --fps and stamps t_cap as
                       the pace clock, otherwise frames are processed as fast as possible with t_cap = read time.

Backends (--backend)
  yoloworld  ultralytics YOLO-World v2 (default weights/yolov8m-worldv2.pt: 31 ms on the M4's MPS and the only size
             that sees the 8 px orange ball in every frame, bench.py 2026-10-30), text prompts via CLIP, ByteTrack or
             BoT-SORT from ultralytics (persist=True) keep the id. The default.
  yoloe      ultralytics YOLOE (weights/yoloe-11s-seg.pt), text prompts via MobileCLIP, the same trackers.
  florence   Florence-2 open-vocabulary detection (transformers, slow) seeding ultralytics' BYTETracker directly.

Statistics (fps, age p50/p90, drops) print to stderr every --report seconds and at the end; --stats FILE writes them
as JSON for ACCEPTANCE.md.
"""
import argparse
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_iface import Frame, Track, W, H  # noqa: E402

A7Z_VENV_PY = "/home/radxa/.venvs/duck-policy/bin/python"
A7Z_LATENCY_S = 0.10          # a7z_cam.LATENCY_S: capture to arrival at the A7Z, assumed
PKT = struct.Struct("<4sQddI")
MAGIC = b"DKFR"
TIME_PORT = 47123


# ----------------------------------------------------------------------------------------------- sources
class Latest:
    """A one-slot mailbox: the producer overwrites, the consumer takes the newest and learns how many were skipped."""

    def __init__(self):
        self.cv = threading.Condition()
        self.item, self.skipped, self.closed = None, 0, False

    def put(self, item):
        with self.cv:
            if self.item is not None:
                self.skipped += 1
            self.item = item
            self.cv.notify()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def take(self, timeout=None):
        with self.cv:
            if not self.cv.wait_for(lambda: self.item is not None or self.closed, timeout):
                return None
            item, self.item = self.item, None
            return item


class SshA7zSource:
    """Frames from the robot over ssh (a7z_frames.py piped into the brain venv's python)."""

    def __init__(self, host="a7z", venv_py=A7Z_VENV_PY):
        self.host = host
        self.box = Latest()
        self.offset = 0.0        # A7Z clock minus this clock (seconds); t_cap_local = t_cap_a7z - offset
        self.offset_rtt = None
        self.n_in = 0
        script = open(os.path.join(HERE, "a7z_frames.py"), "rb")
        self.proc = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, f"{venv_py} -u -"],
                                     stdin=script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        threading.Thread(target=self._stderr, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()
        self.ip = self._resolve()
        threading.Thread(target=self._clock, daemon=True).start()

    def _resolve(self):
        try:
            cfg = subprocess.run(["ssh", "-G", self.host], capture_output=True, text=True, timeout=5).stdout
            m = re.search(r"^hostname (\S+)$", cfg, re.M)
            return socket.gethostbyname(m.group(1)) if m else self.host
        except Exception:
            return self.host

    def _stderr(self):
        for line in self.proc.stderr:
            print(f"[a7z] {line.decode(errors='replace').rstrip()}", file=sys.stderr, flush=True)

    def _reader(self):
        f = self.proc.stdout
        buf = b""
        try:
            while True:
                while len(buf) < PKT.size:
                    chunk = f.read(65536)
                    if not chunk:
                        raise EOFError
                    buf += chunk
                magic, idx, t_cap, t_send, n = PKT.unpack_from(buf)
                if magic != MAGIC:
                    j = buf.find(MAGIC, 1)
                    buf = buf[j:] if j >= 0 else b""
                    continue
                need = PKT.size + n
                while len(buf) < need:
                    chunk = f.read(max(65536, need - len(buf)))
                    if not chunk:
                        raise EOFError
                    buf += chunk
                jpg, buf = buf[PKT.size:need], buf[need:]
                self.n_in += 1
                self.box.put((idx, t_cap, t_send, time.time(), jpg))
        except EOFError:
            pass
        finally:
            self.box.close()

    def _clock(self):
        """Clock offset by UDP echo: offset = t_remote - (t0 + t1) / 2 at the smallest round trip, refreshed every 10 s."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        while self.proc.poll() is None:
            best = None
            for _ in range(10):
                t0 = time.time()
                try:
                    s.sendto(b"t", (self.ip, TIME_PORT))
                    data, _ = s.recvfrom(64)
                except (socket.timeout, OSError):
                    continue
                t1 = time.time()
                rtt = t1 - t0
                if best is None or rtt < best[0]:
                    best = (rtt, float(data) - 0.5 * (t0 + t1))
                time.sleep(0.05)
            if best is not None:
                self.offset_rtt, self.offset = best[0], best[1]
            time.sleep(10.0)

    def frames(self):
        """Yields (bgr, t_cap_local, idx, t_arr, skipped_before_this)."""
        while True:
            item = self.box.take(timeout=5.0)
            if item is None:
                if self.box.closed:
                    return
                print("tracker: no frame for 5 s", file=sys.stderr, flush=True)
                continue
            idx, t_cap, t_send, t_arr, jpg = item
            skipped, self.box.skipped = self.box.skipped, 0
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            yield img, t_cap - self.offset, idx, t_arr, skipped

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class MjpegUrlSource:
    """A multipart JPEG stream over http (boundary from the Content-Type or found in the body)."""

    def __init__(self, url):
        import urllib.request
        self.resp = urllib.request.urlopen(url, timeout=10)
        self.box = Latest()
        self.n_in = 0
        self.offset, self.offset_rtt = 0.0, None
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        buf, n = b"", 0
        try:
            while True:
                chunk = self.resp.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
                    if a < 0 or b < 0:
                        break
                    jpg, buf = buf[a:b + 2], buf[b + 2:]
                    n += 1
                    self.n_in = n
                    t = time.time()
                    self.box.put((n, t - A7Z_LATENCY_S, t, t, jpg))
        finally:
            self.box.close()

    frames = SshA7zSource.frames

    def close(self):
        self.resp.close()


class DirSource:
    def __init__(self, path, fps=10.0, realtime=False, start=0, count=None, blank=None, sway=0.0):
        names = sorted(f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        self.files = [os.path.join(path, f) for f in names][start:None if count is None else start + count]
        self.fps, self.realtime = fps, realtime
        self.blank = blank   # (first_i, n, x1, y1, x2, y2): paint a grey block over that box in n frames from first_i
        self.sway = sway     # px: translate frame i by (sway sin(2 pi i / 30), sway/2 cos(2 pi i / 30)), a moving head
        self.n_in = 0
        self.offset, self.offset_rtt = 0.0, None

    def frames(self):
        t0 = time.time()
        for i, f in enumerate(self.files):
            m = re.search(r"(\d+)", os.path.basename(f))
            idx = int(m.group(1)) if m else i
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if img.shape[1] != W or img.shape[0] != H:
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            if self.blank and self.blank[0] <= i < self.blank[0] + self.blank[1]:
                x1, y1, x2, y2 = [int(round(v)) for v in self.blank[2:]]
                img[max(0, y1):y2, max(0, x1):x2] = 128
            if self.sway:
                dx, dy = self.sway * np.sin(2 * np.pi * i / 30), 0.5 * self.sway * np.cos(2 * np.pi * i / 30)
                img = cv2.warpAffine(img, np.array([[1, 0, dx], [0, 1, dy]], np.float32), (W, H), borderValue=(128, 128, 128))
            if self.realtime:
                t_cap = t0 + i / self.fps
                while time.time() < t_cap:
                    time.sleep(0.002)
            else:
                t_cap = time.time()
            self.n_in += 1
            yield img, t_cap, idx, time.time(), 0

    def close(self):
        pass


def open_source(spec, args):
    if spec == "ssh-a7z":
        return SshA7zSource(args.host, args.venv_py)
    if spec.startswith("mjpeg-url:"):
        return MjpegUrlSource(spec[len("mjpeg-url:"):])
    if spec.startswith("dir:"):
        blank = None
        if args.blank:
            blank = tuple(float(v) for v in args.blank.split(","))
            blank = (int(blank[0]), int(blank[1])) + blank[2:]
        return DirSource(spec[4:], fps=args.fps, realtime=args.realtime, start=args.start, count=args.count,
                         blank=blank, sway=args.sway)
    raise SystemExit(f"unknown source {spec!r} (ssh-a7z | mjpeg-url:<url> | dir:<path>)")


# ----------------------------------------------------------------------------------------------- backends
def parse_queries(q):
    return [s.strip() for s in q.split(",") if s.strip()]


class UltralyticsBackend:
    """YOLO-World / YOLOE detection with the ultralytics tracker (ByteTrack or BoT-SORT), ids persisted."""

    def __init__(self, weights, queries, device, imgsz, conf, tracker_yaml, half, kind):
        from ultralytics import YOLO, YOLOE
        self.queries = queries
        if kind == "yoloe":
            self.model = YOLOE(weights)
            self.model.set_classes(queries, self.model.get_text_pe(queries))
        else:
            self.model = YOLO(weights)
            self.model.set_classes(queries)
        self.kw = dict(persist=True, tracker=tracker_yaml, conf=conf, imgsz=imgsz, device=device, verbose=False)

    def __call__(self, img):
        r = self.model.track(img, **self.kw)[0]
        out = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(xyxy)
        for (x1, y1, x2, y2), c, k, i in zip(xyxy, confs, cls, ids):
            out.append(Track.from_xyxy(self.queries[k] if k < len(self.queries) else str(k), x1, y1, x2, y2, i, c))
        return out


class DetBoxes:
    """Detections in the shape ultralytics' standalone trackers read: xyxy/xywh/conf/cls arrays, boolean indexing."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.asarray(xyxy, np.float32).reshape(-1, 4), np.asarray(conf, np.float32), np.asarray(cls, np.float32)

    @property
    def xywh(self):
        b = self.xyxy
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)

    def __getitem__(self, i):
        return DetBoxes(self.xyxy[i], self.conf[i], self.cls[i])

    def __len__(self):
        return len(self.conf)


class FlorenceBackend:
    """Florence-2 <OPEN_VOCABULARY_DETECTION> per query, seeding ultralytics' BYTETracker."""

    def __init__(self, model_id, queries, device, conf, tracker_yaml):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from ultralytics.trackers import BYTETracker, BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace
        self.torch = torch
        self.queries = queries
        self.device = device
        self.dtype = torch.float16 if device != "cpu" else torch.float32
        # transformers >= 4.57 carries Florence-2 natively under the florence-community checkpoints
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=self.dtype).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        cfg = IterableSimpleNamespace(**YAML.load(tracker_yaml))
        self.tracker = (BOTSORT if cfg.tracker_type == "botsort" else BYTETracker)(args=cfg)

    def detect(self, img):
        from PIL import Image
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dets = []
        task = "<OPEN_VOCABULARY_DETECTION>"
        for k, q in enumerate(self.queries):
            inputs = self.proc(text=task + q, images=pil, return_tensors="pt").to(self.device, self.dtype)
            with self.torch.no_grad():
                ids = self.model.generate(**inputs, max_new_tokens=64, num_beams=1, do_sample=False)
            text = self.proc.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self.proc.post_process_generation(text, task=task, image_size=(pil.width, pil.height))[task]
            for box in parsed.get("bboxes", []):
                dets.append((*box, 0.5, k))   # Florence gives no confidence
        return dets

    def __call__(self, img):
        dets = self.detect(img)
        arr = np.array(dets, dtype=np.float32).reshape(-1, 6)
        tr = self.tracker.update(DetBoxes(arr[:, :4], arr[:, 4], arr[:, 5]), img)   # rows: x1 y1 x2 y2 id conf cls idx
        out = []
        for row in tr:
            x1, y1, x2, y2, i, c, k = row[:7]
            out.append(Track.from_xyxy(self.queries[int(k)], x1, y1, x2, y2, int(i), c))
        return out


def open_backend(args, queries):
    dev = args.device
    if dev == "auto":
        import torch
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tracker_yaml = args.tracker if os.path.exists(args.tracker) else os.path.join(HERE, args.tracker)
    if args.backend in ("yoloworld", "yoloe"):
        w = args.weights or (os.path.join(HERE, "weights", "yolov8m-worldv2.pt") if args.backend == "yoloworld"
                             else os.path.join(HERE, "weights", "yoloe-11s-seg.pt"))
        return UltralyticsBackend(w, queries, dev, args.imgsz, args.conf, tracker_yaml, args.half, args.backend), dev
    if args.backend == "florence":
        return FlorenceBackend(args.weights or "florence-community/Florence-2-base", queries, dev, args.conf, tracker_yaml), dev
    raise SystemExit(f"unknown backend {args.backend}")


# ----------------------------------------------------------------------------------------------- main loop
def save_annotated(img, fr, path):
    im = img.copy()
    for t in fr.tracks:
        x1, y1, x2, y2 = [int(round(v)) for v in t.xyxy()]
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, f"{t.name} #{t.id} {t.conf:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(im, f"frame {fr.frame} age {fr.age * 1000:.0f} ms", (4, H - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 200, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, im)


def pct(v, p):
    return float(np.percentile(v, p)) if len(v) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", required=True, help='comma-separated names, e.g. "orange ball,person"')
    ap.add_argument("--source", default="ssh-a7z")
    ap.add_argument("--out", default="-", help="JSON lines file; - = stdout only (stdout always gets the lines)")
    ap.add_argument("--quiet", action="store_true", help="no lines on stdout (only --out)")
    ap.add_argument("--backend", default="yoloworld", choices=["yoloworld", "yoloe", "florence"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="auto", help="auto | mps | cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--tracker", default="botsort_duck.yaml", help="ultralytics tracker yaml (in this dir or a path); "
                    "botsort_duck.yaml (camera motion compensation) holds the id under head motion, bytetrack_duck.yaml does not")
    ap.add_argument("--host", default="a7z")
    ap.add_argument("--venv-py", default=A7Z_VENV_PY)
    ap.add_argument("--fps", type=float, default=10.0, help="dir: pace / stamp rate")
    ap.add_argument("--realtime", action="store_true", help="dir: pace frames at --fps instead of as fast as possible")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--blank", default=None, help="dir: first,n,x1,y1,x2,y2 paint a grey block there in n frames")
    ap.add_argument("--sway", type=float, default=0.0, help="dir: synthetic camera sway amplitude in px (period 3 s)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--report", type=float, default=10.0, help="stderr statistics every N seconds")
    ap.add_argument("--stats", default=None, help="write the final statistics as JSON here")
    ap.add_argument("--warmup", type=int, default=3, help="frames excluded from the statistics")
    ap.add_argument("--save-dir", default=None, help="write every --save-every-th processed frame, boxes drawn, here")
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    queries = parse_queries(args.query)
    t_load = time.time()
    backend, dev = open_backend(args, queries)
    print(f"tracker: backend {args.backend} on {dev}, queries {queries}, tracker {args.tracker}, "
          f"loaded in {time.time() - t_load:.1f} s", file=sys.stderr, flush=True)
    src = open_source(args.source, args)
    out = open(args.out, "w") if args.out and args.out != "-" else None

    ages, lats, gaps, n, n_found, drops = [], [], [], 0, 0, 0
    ids_seen = {}
    t_start, t_last_cap, t_rep = time.time(), None, time.time()
    try:
        for img, t_cap, idx, t_arr, skipped in src.frames():
            t0 = time.time()
            tracks = backend(img)
            fr = Frame(t_cap, idx, tracks)
            line = fr.line()
            if out:
                out.write(line + "\n")
                out.flush()
            if not args.quiet:
                print(line, flush=True)
            n += 1
            if args.save_dir and n % args.save_every == 1:
                save_annotated(img, fr, os.path.join(args.save_dir, f"f{idx:06d}.jpg"))
            if n > args.warmup:
                ages.append(fr.t_out - t_cap)
                lats.append(fr.t_out - t0)
                drops += skipped
                if t_last_cap is not None:
                    gaps.append(t_cap - t_last_cap)
                n_found += bool(tracks)
                for t in tracks:
                    ids_seen[(t.name, t.id)] = ids_seen.get((t.name, t.id), 0) + 1
            t_last_cap = t_cap
            now = time.time()
            if now - t_rep >= args.report:
                t_rep = now
                print(f"tracker: {n} frames, {len(ages) / max(1e-6, (now - t_start)):.1f} fps, age p50 {pct(ages, 50):.3f} "
                      f"p90 {pct(ages, 90):.3f} s, infer p50 {pct(lats, 50):.3f} s, drops {drops}, found {n_found}/{len(ages)}, "
                      f"clock offset {src.offset:+.3f} s (rtt {src.offset_rtt})", file=sys.stderr, flush=True)
            if args.duration and now - t_start >= args.duration:
                break
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if out:
            out.close()
    wall = time.time() - t_start
    stats = {"backend": args.backend, "device": dev, "weights": args.weights, "queries": queries, "tracker": args.tracker,
             "imgsz": args.imgsz, "source": args.source, "frames": n, "measured": len(ages), "wall_s": round(wall, 2),
             "fps": round(len(ages) / max(1e-6, wall), 2),
             "infer_ms": {"p50": round(1000 * pct(lats, 50), 1), "p90": round(1000 * pct(lats, 90), 1), "max": round(1000 * max(lats), 1) if lats else None},
             "age_s": {"p50": round(pct(ages, 50), 3), "p90": round(pct(ages, 90), 3), "max": round(max(ages), 3) if ages else None},
             "cap_gap_s": {"p50": round(pct(gaps, 50), 3), "p90": round(pct(gaps, 90), 3), "max": round(max(gaps), 3) if gaps else None},
             "drops": drops, "received": getattr(src, "n_in", None), "found": n_found,
             "clock_offset_s": round(src.offset, 4), "clock_rtt_s": src.offset_rtt,
             "ids": {f"{k[0]}#{k[1]}": v for k, v in sorted(ids_seen.items(), key=lambda kv: -kv[1])[:12]}}
    print("tracker: " + json.dumps(stats), file=sys.stderr, flush=True)
    if args.stats:
        with open(args.stats, "w") as f:
            json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
