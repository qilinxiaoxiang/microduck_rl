"""Layer A track interface (PLAN-continuous-vision.md section 3): what the perception layer emits per frame.

Pure python, no heavy dependencies, so the A7Z side (layer B gaze, the policies) and the Mac mini side (tracker.py)
share one definition.

One JSON line per processed frame, also when nothing is found (empty "tracks"), so a consumer can measure age:

  {"t_cap": float, "frame": int,
   "tracks": [{"name": str, "u": float, "v": float, "w": float, "h": float, "id": int, "conf": float}, ...],
   "t_out": float}

t_cap    capture time of the frame (unix seconds, the a7z_cam header's stamp: arrival at the A7Z minus LATENCY_S)
frame    a7z_cam's frame index (monotonic per daemon run), or the file index for a recorded directory
u, v     box centre in continuous pixel coordinates of the 640 x 360 picture (duck_kin convention: the centre of
         pixel column j, row i is (j + 0.5, i + 0.5)); u right, v down
w, h     box width and height in pixels
id       tracker id, held across frames and through short occlusions; -1 if the detector ran without a tracker
conf     detector confidence 0..1 of the last detection that updated the track
name     the query word the detection matched ("orange ball", "person"); several queries may be active at once
t_out    unix time when the line was written; t_out - t_cap is the track age the plan bounds at 0.3 s
"""
from dataclasses import dataclass, field, asdict
import json
import time

W, H = 640, 360


@dataclass
class Track:
    name: str
    u: float
    v: float
    w: float
    h: float
    id: int = -1
    conf: float = 0.0

    @classmethod
    def from_xyxy(cls, name, x1, y1, x2, y2, id=-1, conf=0.0):
        """From a box in continuous pixel coordinates (left, top, right, bottom)."""
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        return cls(name, 0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1, int(id), float(conf))

    def xyxy(self):
        return (self.u - 0.5 * self.w, self.v - 0.5 * self.h, self.u + 0.5 * self.w, self.v + 0.5 * self.h)


@dataclass
class Frame:
    t_cap: float
    frame: int
    tracks: list = field(default_factory=list)
    t_out: float = 0.0

    def line(self, t_out=None):
        """The JSON line (no newline); stamps t_out now unless given."""
        self.t_out = time.time() if t_out is None else t_out
        d = {"t_cap": round(self.t_cap, 4), "frame": int(self.frame),
             "tracks": [{"name": t.name, "u": round(t.u, 1), "v": round(t.v, 1), "w": round(t.w, 1),
                         "h": round(t.h, 1), "id": int(t.id), "conf": round(float(t.conf), 3)} for t in self.tracks],
             "t_out": round(self.t_out, 4)}
        return json.dumps(d, separators=(",", ":"))

    @property
    def age(self):
        return self.t_out - self.t_cap


def parse_line(s):
    """A Frame from one JSON line (a consumer's side)."""
    d = json.loads(s)
    return Frame(d["t_cap"], d["frame"], [Track(**t) for t in d["tracks"]], d.get("t_out", 0.0))


def best(frame, name=None):
    """The highest-confidence track (of one name, if given), or None."""
    cands = [t for t in frame.tracks if name is None or t.name == name]
    return max(cands, key=lambda t: t.conf) if cands else None


if __name__ == "__main__":
    f = Frame(1.0, 7, [Track.from_xyxy("orange ball", 300, 200, 340, 240, id=3, conf=0.8)])
    s = f.line(1.1)
    print(s)
    g = parse_line(s)
    assert g.tracks[0].u == 320 and g.tracks[0].v == 220 and g.tracks[0].w == 40 and g.tracks[0].id == 3
    assert abs(g.age - 0.1) < 1e-6
    print(Frame(2.0, 8).line(2.05))
    print("ok")
