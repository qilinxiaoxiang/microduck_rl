#!/usr/bin/env python3
"""Latest-frame reader for the MicroDuck head camera stream (pi/duck_cam.sh, duck-cam.service).

The Pi runs rpicam-vid as a TCP listener (raw H.264 Annex B, 640x480, 10 fps). OpenCV's FFmpeg backend opens
"tcp://host:port" directly. A background thread reads continuously and keeps only the newest decoded frame, so a
controller that calls latest() at any rate never sees a stale backlog.

    from duck_cam import DuckCam
    cam = DuckCam().start()
    frame, t_arrival, idx = cam.latest()     # (None, None, 0) until the first frame
    cam.fps()                                 # frames received in the last 2 s window
    cam.stop()

CLI (run with ~/.venvs/microduck-walk/bin/python):
    duck_cam.py --seconds 30 --save DIR       # fps, inter-frame gaps, every 10th frame as JPEG, then a backlog test
    duck_cam.py --seconds 30 --no-backlog     # skip the pause-and-drain measurement
    duck_cam.py --seconds 60 --arrivals a.txt # per-frame arrival times, to line stalls up with the walk_wired log

The Pi listener accepts one client; when the client disconnects rpicam-vid exits and systemd restarts it within
about a second, so reconnecting right after a stop can take 2-4 s (open + first frame).
"""
import argparse
import os
import threading
import time

# FFmpeg demuxer options: no internal buffering, minimal probing, so the first frame arrives quickly and the reader
# does not sit on a queue of old frames. Must be set before cv2 is imported.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0",
)
import cv2  # noqa: E402

DEFAULT_HOST = "DUCK_IP"
DEFAULT_PORT = 8554
NOMINAL_FPS = 10.0


class DuckCam:
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, url=None, open_timeout=15.0):
        self.url = url or f"tcp://{host}:{port}"
        self.open_timeout = open_timeout
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._t = None
        self._idx = 0
        self._arrivals = []  # arrival times of recent frames, for fps()
        self._stop = threading.Event()
        self._pause = threading.Event()  # set -> reader thread stops calling read()
        self._thread = None
        self.t_start = None
        self.t_open = None  # seconds from start() to VideoCapture opened
        self.t_first = None  # seconds from start() to first decoded frame
        self.read_failures = 0

    # ---- lifecycle -------------------------------------------------------------------------------------------
    def start(self):
        self.t_start = time.time()
        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open {self.url}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # ignored by FFmpeg, harmless
        self.t_open = time.time() - self.t_start
        self._thread = threading.Thread(target=self._loop, name="duck-cam-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _loop(self):
        while not self._stop.is_set():
            if self._pause.is_set():
                time.sleep(0.002)
                continue
            ok, frame = self._cap.read()
            now = time.time()
            if not ok:
                self.read_failures += 1
                if self.read_failures > 50:  # stream ended (Pi restarted the listener); leave latest() as is
                    break
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._t = now
                self._idx += 1
                if self.t_first is None:
                    self.t_first = now - self.t_start
                self._arrivals.append(now)
                cutoff = now - 2.0
                while self._arrivals and self._arrivals[0] < cutoff:
                    self._arrivals.pop(0)

    # ---- API -------------------------------------------------------------------------------------------------
    def latest(self):
        """Newest frame as (frame_bgr, t_arrival_unix, frame_index); (None, None, 0) before the first frame."""
        with self._lock:
            return self._frame, self._t, self._idx

    def fps(self):
        """Frames received in the last 2 s, divided by 2."""
        with self._lock:
            return len(self._arrivals) / 2.0

    def wait_first(self, timeout=None):
        timeout = self.open_timeout if timeout is None else timeout
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._idx > 0:
                return True
            time.sleep(0.01)
        return False

    def alive(self):
        return self._thread is not None and self._thread.is_alive()

    # ---- latency probe ---------------------------------------------------------------------------------------
    def measure_backlog(self, pause_s=1.0, immediate_gap_s=0.03):
        """Pause the reader for pause_s, resume, and count how many frames drain at once.

        Frames that arrive back-to-back (gap < immediate_gap_s) were queued somewhere between the Pi encoder and
        this process. The stream produces NOMINAL_FPS * pause_s frames during the pause; anything above that was
        already in flight at steady state. Returns dict(drained, expected_from_pause, in_flight_est, drain_s).
        """
        self._pause.set()
        time.sleep(pause_s)
        _, _, idx0 = self.latest()
        # Drain by reading directly here (reader thread is parked) and timing each read.
        t_last = time.time()
        t0 = t_last
        drained = 0
        while True:
            ok, _ = self._cap.read()
            now = time.time()
            if not ok:
                break
            gap = now - t_last
            t_last = now
            if drained > 0 and gap > immediate_gap_s:
                break  # this frame was a live one, not from the backlog
            drained += 1
        self._pause.clear()
        expected = NOMINAL_FPS * pause_s
        return {
            "drained": drained,
            "expected_from_pause": expected,
            "in_flight_est": drained - expected,
            "drain_s": t_last - t0,
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--url", default=None, help="override tcp://host:port")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--save", default=None, help="directory; every 10th frame saved as JPEG")
    ap.add_argument("--no-backlog", action="store_true", help="skip the pause-and-drain backlog measurement")
    ap.add_argument("--arrivals", default=None, help="write one 'unix_time frame_index' line per frame to this file")
    args = ap.parse_args()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
    cam = DuckCam(args.host, args.port, url=args.url).start()
    print(f"opened {cam.url} in {cam.t_open:.2f} s")
    if not cam.wait_first():
        cam.stop()
        raise SystemExit("no frame within the open timeout")
    print(f"first frame {cam.t_first:.2f} s after start()")

    gaps = []
    arrivals = open(args.arrivals, "w") if args.arrivals else None
    last_idx = 0
    last_t = None
    saved = 0
    t_end = time.time() + args.seconds
    while time.time() < t_end and cam.alive():
        frame, t, idx = cam.latest()
        if idx != last_idx:
            if last_t is not None:
                gaps.append(t - last_t)
            last_t, last_idx = t, idx
            if arrivals:
                arrivals.write(f"{t:.3f} {idx}\n")
            if args.save and idx % 10 == 0:
                cv2.imwrite(os.path.join(args.save, f"frame_{idx:05d}.jpg"), frame)
                saved += 1
        time.sleep(0.005)
    if arrivals:
        arrivals.close()
    n = len(gaps) + 1
    span = gaps and sum(gaps) or 0.0
    print(f"frames {n} in {span:.1f} s -> fps {n / span if span else 0:.2f}; "
          f"gap mean {sum(gaps) / len(gaps) * 1e3 if gaps else 0:.1f} ms max {max(gaps) * 1e3 if gaps else 0:.1f} ms; "
          f"gaps >200 ms {sum(g > 0.2 for g in gaps)}, >500 ms {sum(g > 0.5 for g in gaps)}; "
          f"read failures {cam.read_failures}; saved {saved} JPEGs")
    if not args.no_backlog and cam.alive():
        r = cam.measure_backlog()
        print(f"backlog after 1 s pause: drained {r['drained']} frames in {r['drain_s'] * 1e3:.0f} ms, "
              f"expected {r['expected_from_pause']:.0f} from the pause -> in flight ~{r['in_flight_est']:.0f}")
    cam.stop()


if __name__ == "__main__":
    main()
