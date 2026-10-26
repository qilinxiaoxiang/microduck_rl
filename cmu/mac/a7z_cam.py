"""The head camera: one process owns the Innomaker U20CAM-720P (USB, UVC) and keeps the newest frame in shared memory,
only while a task needs it.

The U20CAM sits on the face with its plug edge up (D-023; CURRENT-STATE parts_a7z_extensions innomaker_u20cam) and
streams MJPEG 1280 x 720 at 30 fps on the A7Z's port 2. Radxa's gst v4l2src cannot open it (its vendor
VIDIOC_SET_DMA_MERGE call fails on a UVC device, 2026-10-26), so OpenCV's V4L2 backend reads it. The node is found by
name, since /dev/videoN moves with the boot order. Every third frame is decoded (10 fps); the picture is shrunk to the
upright 640 x 360 BGR that every reader (approach.py, the app's camera window via --mjpeg, snapshots) reads from
/dev/shm/a7z_cam.bgr: a 16-byte header (frame index uint64, capture time float64, little endian) and the picture,
replaced atomically per frame. Frames from the first WARMUP_S are not published while the auto-exposure settles.

The camera is not started with the brain (Shawn 2026-10-24, when the CSI capture heated the closed head): the
find-and-walk task (approach.py) starts it and stops it when it ends, and the app only watches (--mjpeg shows the
picture while the daemon runs and never starts it).

  python a7z_cam.py              the daemon (one instance, a lock in /dev/shm); reopens the camera if it stops
  python a7z_cam.py --ensure     start the daemon detached if it is not running, wait for a fresh frame
  python a7z_cam.py --stop       stop the daemon (SIGTERM), wait until the camera is closed
  python a7z_cam.py --snap F     save the newest frame as an image file (stops the daemon again if it started it)
  python a7z_cam.py --mjpeg      multipart JPEG on stdout from the shared frame (boundary duckframe, quality 70) while
                                  the daemon runs; "a7z_cam: idle" on stderr once a second while it does not
"""
import fcntl
import os
import signal
import struct
import subprocess
import sys
import time

CAMERA_NAME = "U20CAM"
CAP_W, CAP_H = 1280, 720
DECODE_EVERY = 3    # 30 fps from the camera, 10 fps published
W, H = 640, 360
NBYTES = W * H * 3
HEADER = struct.Struct("<Qd")
SHM = "/dev/shm/a7z_cam.bgr"
LOCK = "/dev/shm/a7z_cam.lock"
LATENCY_S = 0.10    # capture to arrival, assumed; frames are stamped arrival minus this
WARMUP_S = 1.0      # auto-exposure settling: frames before this are not published
FRESH_S = 1.0


def camera_node():
    """/dev/videoN of the U20CAM's capture node (index 0; index 1 is its metadata node), or None."""
    base = "/sys/class/video4linux"
    for d in sorted(os.listdir(base)):
        try:
            name = open(f"{base}/{d}/name").read()
            index = open(f"{base}/{d}/index").read().strip()
        except OSError:
            continue
        if CAMERA_NAME in name and index == "0":
            return f"/dev/{d}"
    return None


def daemon():
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("a7z_cam is already running")
    lock.write(str(os.getpid()))
    lock.flush()
    cap = []

    def quit_(*_):
        for c in cap:
            c.release()
        try:
            os.remove(SHM)
        except FileNotFoundError:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, quit_)
    import cv2
    idx = 0
    while True:
        node = camera_node()
        c = cv2.VideoCapture(node, cv2.CAP_V4L2) if node else None
        if c is None or not c.isOpened():
            print(f"no {CAMERA_NAME} capture node open ({node}): retrying in 2 s", flush=True)
            time.sleep(2.0)
            continue
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        c.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap[:] = [c]
        t_open, n_run = time.time(), 0
        print(f"{node} open", flush=True)
        while True:
            if not c.grab():
                break
            n_run += 1
            if n_run % DECODE_EVERY or time.time() - t_open < WARMUP_S:
                continue
            t_cap = time.time() - LATENCY_S
            ok, frame = c.retrieve()
            if not ok:
                break
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            idx += 1
            tmp = SHM + ".tmp"
            with open(tmp, "wb") as f:
                f.write(HEADER.pack(idx, t_cap))
                f.write(frame.tobytes())
            os.replace(tmp, SHM)
        c.release()
        print(f"camera stopped after {n_run} frames: reopening in 2 s", flush=True)
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
        with open(LOCK, "a") as lock:  # "a": "w" would truncate the daemon's pid, which stop() reads
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


def stop(timeout_s=5.0):
    """SIGTERM to the daemon (it releases the camera first); True once the lock is free."""
    if not running():
        return True
    try:
        os.kill(int(open(LOCK).read().strip()), signal.SIGTERM)
    except (OSError, ValueError):
        return False
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not running():
            return True
        time.sleep(0.1)
    return False


def mjpeg(fps=8.0, quality=70):
    """Watch only: JPEG parts while the daemon publishes fresh frames, an idle line on stderr once a second otherwise."""
    import cv2
    import numpy as np
    out, last, idle_t = sys.stdout.buffer, None, 0.0
    while True:
        fr = read_frame() if running() else None
        if fr is None or time.time() - fr[1] >= FRESH_S:
            if time.time() - idle_t >= 1.0:
                print("a7z_cam: idle", file=sys.stderr, flush=True)
                idle_t = time.time()
            time.sleep(0.5)
            continue
        if fr[2] != last:
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
    if len(sys.argv) > 1 and sys.argv[1] == "--stop":
        sys.exit(0 if stop() else "a7z_cam: the daemon did not stop")
    if len(sys.argv) > 2 and sys.argv[1] == "--snap":
        import cv2
        import numpy as np
        started = not running()
        try:
            if not ensure():
                sys.exit("a7z_cam: no fresh frame")
            cv2.imwrite(sys.argv[2], np.frombuffer(read_frame()[0], np.uint8).reshape(H, W, 3))
            print(sys.argv[2])
        finally:
            if started:
                stop()
    elif len(sys.argv) > 1 and sys.argv[1] == "--mjpeg":
        try:
            mjpeg()
        except BrokenPipeError:
            pass
    else:
        daemon()
