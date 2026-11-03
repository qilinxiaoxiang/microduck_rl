"""Frame sender, run ON THE A7Z but kept only in this directory: the Mac mini pipes this file into the brain venv's
python over ssh (`ssh a7z '<venv>/bin/python -u -' < a7z_frames.py`), so nothing is installed on the robot.

Reads a7z_cam's shared frame (/dev/shm/a7z_cam.bgr: 16-byte header = frame index uint64 + capture time float64, then
640 x 360 BGR) each time its index changes, JPEG-encodes it and writes one packet per frame to stdout:

  magic b"DKFR" | frame index uint64 | capture time float64 | send time float64 | jpeg length uint32 | jpeg bytes

Capture time and frame index therefore survive the trip (the plain --mjpeg stream of a7z_cam.py loses both). It holds
the camera the way a7z_cam.py --mjpeg --hold does: starts the daemon only if nothing runs it, stops it again on exit
only in that case (the boot service duck-camera keeps it otherwise). A UDP time echo (port TIME_PORT) lets the receiver
measure the clock offset between the two machines: any datagram is answered with this host's time.time() as an ASCII
float, so the receiver can turn its arrival clock into the A7Z's clock when it computes track age.

No servo, no command file: this reads the camera only.
"""
import os
import signal
import socket
import struct
import sys
import threading
import time

A7Z_MAC_DIR = os.path.expanduser("~/rt/microduck-sim/guide/firmware/mac")
sys.path.insert(0, A7Z_MAC_DIR)
import a7z_cam  # noqa: E402  (the robot's own module; read_frame, hold, running)

PKT = struct.Struct("<4sQddI")
MAGIC = b"DKFR"
TIME_PORT = 47123
QUALITY = int(os.environ.get("DUCK_JPEG_QUALITY", "80"))


def time_echo():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("0.0.0.0", TIME_PORT))
    except OSError as e:
        print(f"a7z_frames: time echo not bound ({e})", file=sys.stderr, flush=True)
        return
    while True:
        try:
            _, addr = s.recvfrom(64)
            s.sendto(repr(time.time()).encode(), addr)
        except OSError:
            return


def main():
    import cv2
    import numpy as np
    threading.Thread(target=time_echo, daemon=True).start()
    release = a7z_cam.hold()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGHUP, lambda *_: sys.exit(0))
    out = sys.stdout.buffer
    last, idle_t, n = None, 0.0, 0
    try:
        while True:
            fr = a7z_cam.read_frame() if a7z_cam.running() else None
            if fr is None or time.time() - fr[1] >= a7z_cam.FRESH_S:
                if time.time() - idle_t >= 1.0:
                    print("a7z_frames: idle (no fresh frame)", file=sys.stderr, flush=True)
                    idle_t = time.time()
                time.sleep(0.2)
                continue
            pic, t_cap, idx = fr
            if idx == last:
                time.sleep(0.01)   # the daemon publishes every 100 ms; poll at 10 ms
                continue
            last = idx
            ok, jpg = cv2.imencode(".jpg", np.frombuffer(pic, np.uint8).reshape(a7z_cam.H, a7z_cam.W, 3),
                                   [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
            if not ok:
                continue
            b = jpg.tobytes()
            out.write(PKT.pack(MAGIC, idx, t_cap, time.time(), len(b)) + b)
            out.flush()
            n += 1
    except BrokenPipeError:
        pass
    finally:
        release()
        print(f"a7z_frames: sent {n} frames", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
