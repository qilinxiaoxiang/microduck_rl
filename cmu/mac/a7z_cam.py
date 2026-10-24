"""The head camera, always on: one process owns /dev/video1 and keeps the newest frame in shared memory.

Opening the camera per script costs about 3 s of near-black frames while the ISP auto-exposure settles (robot
2026-10-24: frame 4 black facing a lit room, frame 17 normal; in the dark frames the reddish wood floor passed as
orange), and a second opener of /dev/video0 or /dev/video1 wedges the vin driver until a reboot. So the camera stays
open in this daemon and every reader (approach.py, the app's camera window via --mjpeg, snapshots) reads
/dev/shm/a7z_cam.bgr: a 16-byte header (frame index uint64, capture time float64, little endian) and the upright
480 x 640 BGR picture, replaced atomically per frame.

  python a7z_cam.py              the daemon (one instance, a lock in /dev/shm); restarts gst if it exits
  python a7z_cam.py --ensure     start the daemon detached if it is not running, wait for a fresh frame
  python a7z_cam.py --snap F     save the newest frame as an image file
  python a7z_cam.py --mjpeg      multipart JPEG on stdout from the shared frame (boundary duckframe, quality 70),
                                  the stream a7z-bridge.mjs used to get from gst-launch directly
~/duck-run.sh starts it with the brain; it keeps running across brain restarts.
"""
import fcntl
import os
import struct
import subprocess
import sys
import time

GST = ("gst-launch-1.0 -q v4l2src device=/dev/video1 en-awisp=1 en-largemode=1"
       " ! video/x-raw,format=NV12,width=4208,height=3120,framerate=24/1 ! videorate drop-only=true"
       " ! video/x-raw,framerate=8/1 ! rawvideoparse use-sink-caps=false format=nv12 width=4208 height=3120 framerate=8/1"
       " ! videoscale ! video/x-raw,width=640,height=480 ! videoflip method=clockwise ! videoconvert"
       " ! video/x-raw,format=BGR ! fdsink fd=1")
W, H = 480, 640
NBYTES = W * H * 3
HEADER = struct.Struct("<Qd")
SHM = "/dev/shm/a7z_cam.bgr"
LOCK = "/dev/shm/a7z_cam.lock"
LATENCY_S = 0.12    # capture to arrival, assumed; frames are stamped arrival minus this
WARMUP = 24         # 3 s at 8 fps: frames before the auto-exposure settles are not published
FRESH_S = 1.0


def others_on_camera():
    """pids of camera openers that are not this daemon's gst (the app's old pipeline, a stray script)."""
    out = subprocess.run(["pgrep", "-f", "v4l2src device=/dev/video"], capture_output=True, text=True).stdout.split()
    mine = set(subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout.split())
    return [p for p in out if p not in mine]


def daemon():
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("a7z_cam is already running")
    lock.write(str(os.getpid()))
    lock.flush()
    idx = 0
    while True:
        busy = others_on_camera()
        if busy:
            print(f"camera held by pid {' '.join(busy)}: waiting (a second opener wedges the driver)", flush=True)
            time.sleep(2.0)
            continue
        p = subprocess.Popen(GST, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        n_run = 0
        print(f"gst started (pid {p.pid})", flush=True)
        while True:
            buf = bytearray()
            while len(buf) < NBYTES:
                chunk = p.stdout.read(NBYTES - len(buf))
                if not chunk:
                    break
                buf += chunk
            if len(buf) < NBYTES:
                break
            n_run += 1
            if n_run <= WARMUP:
                continue
            idx += 1
            tmp = SHM + ".tmp"
            with open(tmp, "wb") as f:
                f.write(HEADER.pack(idx, time.time() - LATENCY_S))
                f.write(buf)
            os.replace(tmp, SHM)
        p.kill()
        p.wait()
        print(f"gst exited after {n_run} frames: restarting in 2 s", flush=True)
        time.sleep(2.0)


def read_frame():
    """(picture bytes, capture time, index) of the newest published frame, or None."""
    try:
        with open(SHM, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    if len(data) != HEADER.size + NBYTES:
        return None
    idx, t = HEADER.unpack_from(data)
    return data[HEADER.size:], t, idx


def running():
    try:
        with open(LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            return False
    except BlockingIOError:
        return True


def ensure(timeout_s=12.0):
    """Start the daemon detached if needed; True once a frame newer than FRESH_S is published."""
    if not running():
        log = os.path.expanduser("~/duck-logs/a7z_cam.out")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        subprocess.Popen([sys.executable, os.path.abspath(__file__)], stdout=open(log, "a"), stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        fr = read_frame()
        if fr is not None and time.time() - fr[1] < FRESH_S:
            return True
        time.sleep(0.2)
    return False


def mjpeg(fps=8.0, quality=70):
    import cv2
    import numpy as np
    if not ensure():
        sys.exit("a7z_cam: no fresh frame (see ~/duck-logs/a7z_cam.out)")
    out, last = sys.stdout.buffer, None
    while True:
        fr = read_frame()
        if fr is not None and fr[2] != last and time.time() - fr[1] < FRESH_S:
            last = fr[2]
            ok, jpg = cv2.imencode(".jpg", np.frombuffer(fr[0], np.uint8).reshape(H, W, 3),
                                   [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                b = jpg.tobytes()
                out.write(b"--duckframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b) + b + b"\r\n")
                out.flush()
        time.sleep(0.5 / fps)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--ensure":
        sys.exit(0 if ensure() else "a7z_cam: no fresh frame (see ~/duck-logs/a7z_cam.out)")
    if len(sys.argv) > 2 and sys.argv[1] == "--snap":
        import cv2
        import numpy as np
        if not ensure():
            sys.exit("a7z_cam: no fresh frame")
        cv2.imwrite(sys.argv[2], np.frombuffer(read_frame()[0], np.uint8).reshape(H, W, 3))
        print(sys.argv[2])
    elif len(sys.argv) > 1 and sys.argv[1] == "--mjpeg":
        try:
            mjpeg()
        except BrokenPipeError:
            pass
    else:
        daemon()
