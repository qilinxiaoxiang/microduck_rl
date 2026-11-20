"""The detector the pickup uses: the colour filter proposes at camera rate, the learned detector decides in the background.

Shawn 2026-11-12: "该下yolo下yolo"; "检测不到是不是因为没有用YOLO，还在用颜色滤镜". The colour filter (orange_target, a few
ms) took the orange inside of the robot's own foot, a far wall and the floor for the ball, and misses a far ball whose
blob is under its area limit. The learned detector (learned_target) does not make those mistakes, but one frame costs
1.05 s on the A7Z with 1 thread, 0.74 s with 2, 0.68 s with 4 (measured 2026-11-12, 8 cores).

The learned pass never runs in the caller's loop. Robot 2026-11-12 (seepick_1112m): with the pass inside the 50 Hz
loop, each look stopped the loop for a second, so (1) the ball was in the picture 0.7 s and 1.7 s in the first two
scans and no pass looked at those frames, (2) the turn toward the ball that was finally seen never started, because
the walk command (good for 0.6 s) ran out during every pass (command at zero in 111 of 226 brain rows), and the robot
went back to turning the body. Shawn: "球都出现在视野里面了，还是不看球".

  detect(frame, accept, ctx, late_ok)
    colour candidate near a confirmed sighting (TRACK_PX, TRACK_S)  -> the colour box, at camera rate
    otherwise the newest frame goes to the worker, and the call returns at once
    a finished learned pass that found the ball                    -> a confirmed sighting; the box itself is
                                                                       returned with the ctx of ITS frame when late_ok
  pending: a colour candidate is in the picture and no confirmed sighting covers it yet (the search waits on this).
"""
import math
import threading
import time

import learned_target
import orange_target

TRACK_PX = 110          # px: a colour box this near the last confirmed or tracked centre is the same object
TRACK_S = 1.5           # s: how long a sighting stays good without a new one


class BallDetect:
    def __init__(self, hsv_lo, hsv_hi, min_area, now=time.time, learned=learned_target.detect,
                 colour=orange_target.detect, threaded=True):
        self.lo, self.hi, self.min_area, self.now = hsv_lo, hsv_hi, min_area, now
        self.learned, self.colour, self.threaded = learned, colour, threaded
        self.track = None        # (cx, cy, t) of the last confirmed or tracked sighting
        self.pending = False     # a colour candidate is in the picture and no confirmed sighting covers it yet
        self.error = None        # the last exception of a learned pass
        self.stats = dict(colour=0, tracked=0, learned=0, learned_hits=0, rejected=0)
        self._cv = threading.Condition()
        self._job = None         # newest (frame, accept, ctx, had_colour) for the worker
        self._res = None         # a finished pass the caller has not taken yet: (hit or None, ctx, had_colour)
        if threaded:
            threading.Thread(target=self._run, daemon=True).start()

    def _near(self, c, t):
        return (self.track is not None and t - self.track[2] <= TRACK_S
                and math.hypot(c["cx"] - self.track[0], c["cy"] - self.track[1]) <= TRACK_PX)

    def _pass(self, job):
        try:
            return self.learned(job[0], accept=job[1])
        except Exception as e:   # a failed pass is a miss, and says so
            self.error = repr(e)
            return None

    def _run(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job, self._job = self._job, None
            y = self._pass(job)
            with self._cv:
                self._res = (y, job[2], job[3])

    def _take(self, res, t):
        """Book a finished pass; its box (with the ctx of its frame) when it found the ball."""
        y, ctx, had_colour = res
        self.stats["learned"] += 1
        if y is None:
            self.stats["rejected"] += bool(had_colour)
            return None
        self.stats["learned_hits"] += 1
        self.track = (y["cx"], y["cy"], t)
        return dict(y, ctx=ctx)

    def __call__(self, frame, accept=None, ctx=None, late_ok=True):
        t = self.now()
        c = self.colour(frame, self.lo, self.hi, self.min_area, accept=accept)
        late = None
        with self._cv:
            res, self._res = self._res, None
        if res is not None:
            late = self._take(res, t)
        if c is not None:
            self.stats["colour"] += 1
            if self._near(c, t):
                self.track = (c["cx"], c["cy"], t)
                self.stats["tracked"] += 1
                self.pending = False
                return dict(c, ctx=ctx)
        self.pending = c is not None
        job = (frame.copy() if hasattr(frame, "copy") else frame, accept, ctx, c is not None)
        if not self.threaded:   # tests: the pass runs here and rules on this very frame
            y = self._take((self._pass(job), ctx, c is not None), t)
            if y is not None:
                self.pending = False
            return y
        with self._cv:
            self._job = job
            self._cv.notify()
        return late if late_ok else None
