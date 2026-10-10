"""Object permanence: detections in each frame -> objects with a persistent id and a world position, kept up to
date while they can't be seen.

    objects, table_z = scan_objects(scan)            # the scan's objects, refined into digital twins
    tracker = Tracker(objects, table_z)
    tracker.update(frame, detections)                # every frame: an xray.frames.Frame and its xray.detect Detections
    msg = tracker.message(frame)                     # the state message (the docs/sample_state_G8.jsonl shape)

    python -m xray.track <track folder> --scan <scan folder> [--ball | --target-pixel U V] [--out runs/track_<name>]
replays a recorded track folder (--ball: with the shell game, xray.shell) (the scan folder layout) and writes states.jsonl, which
replay_server.py --track plays back over the same recording. Detections come from the track folder's
detections_<labels>.npz when there is one (xray.detect.detect_scan writes it), else the detector runs on each frame.

Each frame:
  1. measure: a detection -> a 3D position. Its mask eroded 3 px, the nearest layer of the depth in it (3 cm in
     front of its 15th-percentile depth to 8 cm behind), confident and not flying, above the table. The camera
     sees the near side of an object, so the centre is the median of those points pushed back along the
     viewing ray by 0.75 x the object's radius. A detection that shows only part of its object (touching the
     image edge, or partly behind a nearer detection) keeps the track's height, and its position unless the
     two are more than 6 cm apart; one partly behind another and farther than that is not used at all.
     Detections of one label within 6 cm of each other are one object (the best score stays); one whose top
     is under 0.65 x the height of every object of its label is something smaller seen as that label.
  2. associate, per label: distance sqrt(dxy^2 + (0.3 dz)^2) (height is the noisiest axis), +3 cm for a hidden
     object (one in view comes first), as a share of the track's gate; the assignment with the least total
     (Hungarian), where leaving a track unmatched costs 1, so no detection beyond the gate takes it. Gate:
     0.12 m (0.15 m hidden) + 1.5 x speed x dt x (1 + frames unseen), at most 0.4 m; nothing was seen moving
     an object out of view, so its gate stays 0.15 m. A share, not metres: an uncertain track (wide gate)
     doesn't take the detection right at a well-placed one.
  3. filter: alpha-beta on position and velocity. A still object gains 0.15 / 0.02 (steady), a moving one
     0.6 / 0.25 (it keeps up). Never below the table. Without a detection it stays where it was last seen.
  4. an object without a detection is explained by the depth image where it should be:
       behind the camera or off the image                  OUT_OF_VIEW
       something nearer than its front (a hand, a box)     OCCLUDED
       otherwise (it should be seen and isn't)             LOST once 0.6 s pass without a detection or one of
                                                           the two explanations above
     A state changes once 3 frames in a row agree. A hidden object comes back (reacquired) after 3 detections
     in a row scoring >= 0.5, or 2 scoring >= 0.3 coming back into view (detections at the image edge score
     low), any score within 5 cm of where it was; until then it keeps its position.
  5. a detection near no object, in 5 frames in a row scoring >= 0.7, >= 15 cm from every object seen now and
     >= 8 cm from any object of another label (a ball seen as a cup): the nearest object of that label unseen
     for 5 frames or more, found again (it was moved while hidden, or the track slipped), or else a new object.
"""
import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray import frames as fr  # noqa: E402
from ground3d import refine as rf  # noqa: E402

VISIBLE, OCCLUDED, OUT_OF_VIEW, LOST = "VISIBLE", "OCCLUDED", "OUT_OF_VIEW", "LOST"
HIDDEN = (OCCLUDED, OUT_OF_VIEW, LOST)
RADIUS = {"cup": 0.045, "ball": 0.032, "can": 0.033}           # m, when the scan gives no size
FADE_S = {OUT_OF_VIEW: 60.0, OCCLUDED: 30.0, LOST: 5.0}        # how fast confidence fades in each hidden state
HISTORY_S = 3.0
HIDDEN_COST = 0.03                                              # m added to a hidden object's distances


def r4(v):
    return np.round(np.asarray(v, float), 4).tolist()


# ---------------------------------------------------------------------------------------- measuring
@dataclass
class Measurement:
    label: str
    score: float
    position: np.ndarray            # world centre estimate
    edge: bool                      # the box touches the image edge: part of the object is cut off
    box: tuple
    depth: float                    # distance of the centre from the camera
    top: float                      # height of its top (95th percentile of its points)
    bottom: float                   # height of its bottom (5th percentile)
    mask: np.ndarray = field(repr=False)
    n_points: int = 0
    behind: bool = False            # partly hidden behind a nearer detection


def measure(frame, det, clean, table_z=None, radius=None, erode_px=3, edge_px=3):
    """A detection -> Measurement (None if its mask holds too little depth). clean: rf.clean_pixels of the frame."""
    m = rf.nearest_layer(frame.depth, rf.erode(det.mask_at(frame.depth.shape), erode_px) & clean)
    p = fr.backproject(frame.depth, frame.K, frame.T, m)
    if table_z is not None:
        p = p[p[:, 2] > table_z + 0.005]
    if len(p) < 12:
        return None
    c = np.median(p, axis=0)
    ray = c - frame.position
    c = c + ray / max(np.linalg.norm(ray), 1e-6) * 0.75 * (radius or RADIUS.get(rf.kind(det.label), 0.04))
    h, w = frame.rgb.shape[:2]
    x0, y0, x1, y1 = det.box
    edge = x0 <= edge_px or y0 <= edge_px or x1 >= w - 1 - edge_px or y1 >= h - 1 - edge_px
    return Measurement(det.label, float(det.score), c, bool(edge), det.box, float(np.linalg.norm(ray)),
                       float(np.percentile(p[:, 2], 95)), float(np.percentile(p[:, 2], 5)), det.mask, int(len(p)))


def one_per_place(ms, apart=0.06):
    """Detections of one label within `apart` of each other are one object: the best score stays."""
    out = []
    for m in sorted(ms, key=lambda m: -m.score):
        if all(o.label != m.label or np.linalg.norm(o.position - m.position) >= apart for o in out):
            out.append(m)
    return out


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def drop_cups_on_balls(dets, overlap=0.6):
    """A cup (or can) detection whose box mostly covers a ball detection's is that ball, seen as a cup."""
    balls = [d.box for d in dets if rf.kind(d.label) == "ball"]
    return [d for d in dets if rf.kind(d.label) == "ball" or all(iou(d.box, b) <= overlap for b in balls)]


def mark_behind(ms, share=0.15, gap=0.03):
    """A detection whose box is covered `share` or more by the box of one `gap` m nearer is partly behind it."""
    for m in ms:
        x0, y0, x1, y1 = m.box
        area = max((x1 - x0) * (y1 - y0), 1.0)
        for o in ms:
            if o is not m and o.depth < m.depth - gap:
                w = min(x1, o.box[2]) - max(x0, o.box[0])
                h = min(y1, o.box[3]) - max(y0, o.box[1])
                if w > 0 and h > 0 and w * h >= share * area:
                    m.behind = True
                    break
    return ms


def distance(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(math.sqrt(d[0] * d[0] + d[1] * d[1] + (0.3 * d[2]) ** 2))


# ---------------------------------------------------------------------------------------- assignment
def hungarian(c):
    """The column for each row with the least total cost (rows <= columns): Kuhn-Munkres with potentials."""
    n, m = c.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, int), np.zeros(m + 1, int)         # p[j]: row (1-based) holding column j
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, np.inf), np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = c[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            cand = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(cand)) + 1
            delta = cand[j1 - 1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def assign(cost, gates):
    """Rows (tracks) -> columns (detections) with the least total cost, where leaving a row unmatched costs its
    gate (so nothing beyond the gate is matched): {row: column}."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return {}
    big = 1e6
    c = np.full((n, m + n), big)
    c[:, :m] = np.where(cost <= gates[:, None], cost, big)
    c[np.arange(n), m + np.arange(n)] = gates
    return {i: j for i, j in enumerate(hungarian(c)) if j < m and c[i, j] < big}


# ---------------------------------------------------------------------------------------- tracks
class Track:
    def __init__(self, tid, label, position, size=None, points=None, t=None, score=None):
        self.id, self.label = tid, label
        self.position = np.asarray(position, float).copy()
        self.origin = self.position.copy()              # where the twin cloud was made (the overlay shifts it)
        self.predicted = self.position.copy()           # where it should be this frame (for matching)
        self.velocity = np.zeros(3)
        r = RADIUS.get(rf.kind(label), 0.04)
        self.size = np.asarray(size, float) if size is not None else np.array([2 * r, 2 * r, 2 * r])
        self.radius = float(max(self.size[0], self.size[1]) / 2)
        self.points = points                            # the digital twin (world points at scan time), or None
        self.state = VISIBLE
        self.last_seen_t = self.ok_t = t                # ok_t: last time it was seen or its absence explained
        self.unseen = 0
        self.score = score
        self.pending, self.pending_n = None, 0
        self.back = 0                                   # detections in a row while coming back (hidden -> visible)
        self.reacquired_t = None
        self.top_seen = self.bottom_seen = None         # the latest detection's top and bottom height (None: only part seen)
        self.history = deque()                          # (t, position) over the last HISTORY_S
        self.target = False
        self.extra = {}                                 # fields other parts add to the message (the shell game)

    @property
    def speed(self):
        return float(math.hypot(self.velocity[0], self.velocity[1]))

    def gate(self, dt):
        if self.state == OUT_OF_VIEW:
            return 0.15
        base = 0.15 if self.state in HIDDEN else 0.12
        return min(0.4, base + 1.5 * self.speed * dt * (1 + self.unseen))

    def predict(self, dt):
        self.predicted = self.position.copy()
        if self.state == VISIBLE:
            self.predicted[:2] += self.velocity[:2] * dt

    def seen(self, m, t, dt, table_z=None):
        self.score = m.score
        part = m.edge or m.behind
        self.top_seen, self.bottom_seen = (None, None) if part else (m.top, m.bottom)
        if self.state in HIDDEN:                        # coming back: a few good detections in a row first
            into_view = self.state == OUT_OF_VIEW
            low = 0.0 if distance(self.predicted, m.position) < 0.05 else 0.3 if into_view else 0.5
            self.back = self.back + 1 if m.score >= low and not m.behind else 0
            if self.back < (2 if into_view else 3):
                return
            self.jump(m.position if not part else np.r_[m.position[:2], self.position[2]], t)
        else:
            p = self.predicted
            z = m.position.copy()
            if part:                                    # only part of it is seen: trust what we had
                z[2] = p[2]
                if math.hypot(*(z[:2] - p[:2])) <= 0.06:
                    z[:2] = p[:2]
            r = z - p
            moving = math.hypot(r[0], r[1]) > 0.02 or self.speed > 0.05
            a, b = (0.6, 0.25) if moving else (0.15, 0.02)
            self.position = p + a * r
            self.velocity[:2] += b * r[:2] / dt
            s = self.speed
            if s > 1.5:
                self.velocity[:2] *= 1.5 / s
        if table_z is not None:
            self.position[2] = max(self.position[2], table_z + 0.01)
        self.last_seen_t = self.ok_t = t
        self.unseen = 0
        self.pending, self.pending_n = None, 0

    def jump(self, position, t, score=None):
        """Found again (reacquired): straight to where it was seen."""
        self.position = np.asarray(position, float).copy()
        self.predicted = self.position.copy()
        self.velocity[:] = 0
        self.state, self.back = VISIBLE, 0
        self.reacquired_t = self.last_seen_t = self.ok_t = t
        self.unseen = 0
        self.pending, self.pending_n = None, 0
        if score is not None:
            self.score = score

    def missed(self, why, t):
        """No detection this frame; why = 'away' / 'covered' / None (see Tracker.explain)."""
        self.unseen += 1
        self.back = 0
        self.velocity[:2] *= 0.7 if self.state == VISIBLE else 0.9     # coast a little through detector flicker
        if why == "away":
            want = OUT_OF_VIEW
        elif why == "covered":
            want = OCCLUDED
        else:
            want = LOST if t - self.ok_t > 0.6 else self.state
        if why is not None and self.state == want:
            self.ok_t = t
        self.change(want)

    def change(self, want, frames=3):
        if want == self.state:
            self.pending, self.pending_n = None, 0
            return
        if want == self.pending:
            self.pending_n += 1
        else:
            self.pending, self.pending_n = want, 1
        if self.pending_n >= frames:
            self.state, self.pending, self.pending_n = want, None, 0

    def remember(self, t):
        self.history.append((t, self.position.copy()))
        while self.history and t - self.history[0][0] > HISTORY_S:
            self.history.popleft()

    def as_dict(self, t):
        since = 0.0 if self.state == VISIBLE else round(t - self.last_seen_t, 2)
        out = dict(id=self.id, label=self.label, target=self.target, state=self.state, position=r4(self.position),
                   velocity=np.round(self.velocity, 3).tolist(), size=np.round(self.size, 3).tolist(),
                   last_seen_t=round(self.last_seen_t, 3), since_seen_s=since,
                   confidence=1.0 if self.state == VISIBLE else round(math.exp(-since / FADE_S[self.state]), 3),
                   detection_score=None if self.score is None else round(self.score, 3),
                   reacquired=self.reacquired_t is not None and t - self.reacquired_t < 1.0,
                   has_ball=False, balls_inside=[])
        out.update(self.extra)
        return out


class Tracker:
    """labels: which detection labels it follows (default: every label)."""

    def __init__(self, objects=(), table_z=None, labels=None, spawn=True):
        self.table_z, self.spawn = table_z, spawn
        self.labels = set(labels) if labels else None
        self.tracks, self.candidates = [], []
        self.next_id, self.target_id = 0, None
        self.t, self.n = None, 0
        self.measurements = []                          # this frame's, after one_per_place (for the overlay)
        self.others = []                                # (label, position) of objects followed elsewhere (balls)
        for o in objects:
            self.add(o["label"], o["position"], o.get("size"), o.get("points"))

    def add(self, label, position, size=None, points=None, t=None, score=None):
        tr = Track(self.next_id, label, position, size, points, t, score)
        self.next_id += 1
        self.tracks.append(tr)
        return tr

    def get(self, tid):
        return next((tr for tr in self.tracks if tr.id == tid), None)

    def follows(self, label):
        return self.labels is None or label in self.labels

    def update(self, frame, detections):
        t = frame.t
        dt = 1 / 15 if self.t is None else max(1e-3, t - self.t)
        self.t, self.n = t, self.n + 1
        for tr in self.tracks:
            if tr.last_seen_t is None:                  # objects from the scan: seen at the start
                tr.last_seen_t = tr.ok_t = t
            tr.predict(dt)
        clean = frame.clean = rf.clean_pixels(frame.depth, frame.conf, stride=2)      # (the shell game reuses it)
        radius = {tr.label: tr.radius for tr in self.tracks}
        ms = [measure(frame, d, clean, self.table_z, radius.get(d.label)) for d in drop_cups_on_balls(detections)
              if self.follows(d.label)]
        ms = mark_behind([m for m in ms if m is not None])
        ms = one_per_place([m for m in ms if not self.too_short(m)])
        matched = {}
        for label in {tr.label for tr in self.tracks}:
            rows = [i for i, tr in enumerate(self.tracks) if tr.label == label]
            cols = [j for j, m in enumerate(ms) if m.label == label]
            if not cols:
                continue
            gates = np.array([self.tracks[i].gate(dt) for i in rows])
            cost = np.array([[distance(self.tracks[i].predicted, ms[j].position) + HIDDEN_COST * (self.tracks[i].state in HIDDEN)
                              for j in cols] for i in rows]) / gates[:, None]
            for a, b in assign(cost, np.ones(len(rows))).items():
                matched[rows[a]] = cols[b]
        for i, tr in enumerate(self.tracks):
            m = ms[matched[i]] if i in matched else None
            if m is not None and m.behind and distance(tr.predicted, m.position) > 0.06:
                m = None                                # mostly hidden by a nearer object: not a position to trust
            if m is not None:
                tr.seen(m, t, dt, self.table_z)
            else:
                tr.missed(self.explain(tr, frame), t)
        used = set(matched.values())
        self.new_objects([m for j, m in enumerate(ms) if j not in used], t)
        for tr in self.tracks:
            tr.target = tr.id == self.target_id
            tr.remember(t)
        self.measurements = ms
        return self.tracks

    def too_short(self, m):
        """A whole detection whose top is under 0.65 x the height of every object of its label: something smaller
        seen as that label (a ball as a cup)."""
        heights = [tr.size[2] for tr in self.tracks if tr.label == m.label]
        return (self.table_z is not None and not m.edge and not m.behind and bool(heights)
                and m.top - self.table_z < 0.65 * min(heights))

    def explain(self, tr, frame, margin=8):
        """Why a track has no detection: 'away' (behind the camera or off the image), 'covered' (the depth where it
        should be is nearer than its front by 3 cm or more), or None (it should be seen)."""
        u, v, z = fr.project(tr.position[None], frame.K, frame.T)
        u, v, z = float(u[0]), float(v[0]), float(z[0])
        h, w = frame.depth.shape
        if z <= 0.05 or not (margin <= u < w - margin and margin <= v < h - margin):
            return "away"
        r = max(2, int(0.5 * tr.radius * frame.K[0, 0] / z))
        ui, vi = int(u), int(v)
        win = frame.depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
        d = win[win > 0]
        if len(d) < 4:
            return None
        return "covered" if float(np.median(d)) < z - tr.radius - 0.03 else None

    def new_objects(self, ms, t):
        """Unmatched detections: 5 frames in a row (one missed frame allowed) scoring >= 0.7, >= 15 cm from every
        object seen now and >= 8 cm from other labels' -> the nearest object of that label unseen for 5 frames or
        more, or a new object."""
        if not self.spawn:
            return
        for m in ms:
            if m.score < 0.7:
                continue
            c = next((c for c in self.candidates if c["label"] == m.label
                      and np.linalg.norm(c["position"] - m.position) < 0.1), None)
            if c is None:
                c = dict(label=m.label, n=0, score=0.0)
                self.candidates.append(c)
            c.update(position=m.position, n=c["n"] + 1, score=max(c["score"], m.score), frame=self.n)
        self.candidates = [c for c in self.candidates if self.n - c["frame"] <= 1]
        for c in [c for c in self.candidates if c["n"] >= 5]:
            self.candidates.remove(c)
            near = [(tr.label, tr.unseen, math.hypot(*(tr.position[:2] - c["position"][:2]))) for tr in self.tracks]
            near += [(label, 0, math.hypot(*(np.asarray(p)[:2] - c["position"][:2]))) for label, p in self.others]
            if any(d < 0.15 and unseen == 0 or d < 0.08 and label != c["label"] for label, unseen, d in near):
                continue                                # a tracked object, or another kind of object seen as this one
            gone = [tr for tr in self.tracks if tr.label == c["label"] and tr.unseen >= 5]
            if gone:
                min(gone, key=lambda tr: np.linalg.norm(tr.position - c["position"])).jump(c["position"], t, c["score"])
            else:
                self.add(c["label"], c["position"], t=t, score=c["score"])

    def message(self, frame, fps=None):
        T = frame.T
        return dict(t=round(float(frame.t), 3), frame=int(frame.index), target_id=self.target_id, phase="track",
                    fps=None if fps is None else round(float(fps), 1),
                    camera=dict(position=r4(T[:3, 3]), forward=r4(T[:3, 2]),
                                T_world_cam=np.round(T[:3].reshape(-1), 5).tolist()),
                    objects=[tr.as_dict(frame.t) for tr in self.tracks])


# ---------------------------------------------------------------------------------------- the scene
def support_height(keyframes, seeds):
    """The height of the surface the objects stand on: the table (the most common height between 0.4 and 1.2 m), or
    the floor (the most common height below 0.3 m, about 0 in the scan's world) when the objects are lower than the
    table: on the floor, with a table somewhere else in the room."""
    table = rf.table_height(keyframes)
    low = float(np.median(np.concatenate(seeds)[:, 2])) if seeds else None
    if low is not None and (table is None or low < table - 0.05):
        floor = rf.table_height(keyframes, lo=-0.15, hi=0.3)
        return 0.0 if floor is None else floor
    return table


def scan_objects(scan, prompt=rf.DEFAULT_PROMPT, use_detector=True, labels=None, min_score=0.3, min_views=3,
                 log=print):
    """The objects the tracker starts with: every object the detector finds (score >= min_score) in >= min_views
    of the scan's keyframes, refined into its digital twin, standing on a table or on the floor (support_height). One object seen from opposite sides can come out of
    the detections as two (each view's centre is on its near side): twins of one label within 6 cm are one.
    Returns ([dict(label, position, size, points, views, score)], table height)."""
    keyframes = rf.load_keyframes(scan)
    dets = rf.scan_detections(scan, prompt, use_detector, log)
    if dets is None:
        raise SystemExit(f"no detections for {scan}: the detector for \"{prompt}\" is needed (python -m xray.detect)")
    dets = [drop_cups_on_balls(ds) for ds in dets]
    found = rf.objects_from_detections(keyframes, dets, min_score=min_score, min_views=1)
    table = support_height(keyframes, [seed for _, seed, _, _ in found])
    out = []
    for label, seed, views, score in found:
        if labels and label not in labels:
            continue
        r = rf.refine(seed, keyframes, table, label=label, dets=dets)
        if not r["n_points"]:
            continue
        same = next((o for o in out if o["label"] == label and np.linalg.norm(np.subtract(o["position"], r["center"])) < 0.06), None)
        if same is not None:
            same["views"] += views
            same["score"] = max(same["score"], round(float(score), 3))
            continue
        out.append(dict(label=label, position=r["center"], size=r["size"], points=r["points"], views=views,
                        score=round(float(score), 3)))
    balls = [o for o in out if rf.kind(o["label"]) == "ball"]
    out = [o for o in out if o["views"] >= min_views and (rf.kind(o["label"]) == "ball" or all(     # a ball seen as a cup
           math.hypot(*np.subtract(o["position"][:2], b["position"][:2])) > 0.05 for b in balls))]
    return out, table


def point_at_pixel(frame, u, v, r=3):
    """The world point under pixel (u, v) of a frame (median depth around it), or None."""
    d = frame.depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    d = d[d > 0]
    if not len(d):
        return None
    z = float(np.median(d))
    cam = np.array([(u - frame.K[0, 2]) * z / frame.K[0, 0], (v - frame.K[1, 2]) * z / frame.K[1, 1], z])
    return frame.T[:3, :3] @ cam + frame.T[:3, 3]


def nearest(tracks, point, labels=None):
    pool = [tr for tr in tracks if labels is None or tr.label in labels]
    return min(pool, key=lambda tr: np.linalg.norm(tr.position[:2] - np.asarray(point)[:2])) if pool else None


# ---------------------------------------------------------------------------------------- replay
def timeline(states, key):
    """[(t0, t1, value)] runs of key(state) over the messages."""
    out = []
    for s in states:
        v = key(s)
        if out and out[-1][2] == v:
            out[-1][1] = s["t"]
        else:
            out.append([s["t"], s["t"], v])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay a recorded track folder through the tracker (see the top of this file)")
    ap.add_argument("track", help="a track folder: color, depth, conf, intrinsic, pose, meta.json")
    ap.add_argument("--scan", required=True, help="the scan folder of the same session (same world)")
    ap.add_argument("--prompt", default=rf.DEFAULT_PROMPT, help=f"detector labels (default \"{rf.DEFAULT_PROMPT}\")")
    ap.add_argument("--target-pixel", type=int, nargs=2, metavar=("U", "V"), help="the target: the object under this pixel of the first frame")
    ap.add_argument("--target-point", type=float, nargs=3, metavar=("X", "Y", "Z"), help="the target: the object nearest this world point")
    ap.add_argument("--ball", action="store_true", help="the shell game: balls under cups (xray.shell)")
    ap.add_argument("--no-spawn", action="store_true", help="only the scan's objects")
    ap.add_argument("--out", help="output folder (default runs/track_<track folder's parent>)")
    args = ap.parse_args(argv)
    from xray import detect
    track_dir = Path(args.track).expanduser()
    out = Path(args.out or f"runs/track_{track_dir.resolve().parent.name}").expanduser()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    objects, table = scan_objects(args.scan, args.prompt)
    print(f"scan: {len(objects)} objects, table at {table if table is None else round(table, 3)} m ({time.time() - t0:.1f} s)")
    for i, o in enumerate(objects):
        print(f"  {i}  {o['label']:10s} at {r4(o['position'])}, size {rf.size_cm(o['size'])}, in {o['views']} keyframes")
    labels = detect.parse_prompt(args.prompt)
    cached = detect.load_detections(detect.cache_path(track_dir, labels))
    detector = None if cached is not None else detect.Detector(labels)
    src = fr.FolderSource(track_dir)
    game = None
    if args.ball:
        from xray import shell
        cups, balls = shell.split(objects)
        tracker = Tracker(cups, table, labels={c["label"] for c in cups}, spawn=not args.no_spawn)
        game = shell.ShellGame(tracker, balls, table, args.scan)
    else:
        tracker = Tracker(objects, table, spawn=not args.no_spawn)
    states, ms = [], []
    for f in src:
        dets = cached.get(src.ids[f.index], []) if detector is None else detector(f.rgb)
        t = time.perf_counter()
        tracker.update(f, dets)
        if game is not None:
            game.update(f, dets)
        ms.append(time.perf_counter() - t)
        if f.index == 0 and (args.target_pixel or args.target_point):
            p = point_at_pixel(f, *args.target_pixel) if args.target_pixel else np.array(args.target_point)
            tr = None if p is None else nearest(tracker.tracks, p)
            tracker.target_id = None if tr is None else tr.id
            print(f"target: {'none (no depth there)' if tr is None else f'{tr.label} {tr.id} at {r4(tr.position)}'}")
        fps = None if f.index == 0 else 1 / max(1e-3, src.times[f.index] - src.times[f.index - 1])
        msg = tracker.message(f, fps)
        states.append(game.message(msg) if game is not None else msg)
    with open(out / "states.jsonl", "w") as fh:
        for s in states:
            fh.write(json.dumps(s, separators=(",", ":")) + "\n")
    print(f"{len(states)} frames, t {states[0]['t']:.1f}-{states[-1]['t']:.1f} s -> {out}/states.jsonl; tracker "
          f"{1000 * np.median(ms):.1f} ms a frame (median, without the detector)")
    for tr in tracker.tracks:
        runs = timeline(states, lambda s: next((o["state"] for o in s["objects"] if o["id"] == tr.id), None))
        runs = [r for r in runs if r[2] is not None]
        print(f"  {tr.id}{'*' if tr.target else ' '} {tr.label:10s} end {r4(tr.position)} {tr.state}: "
              + ", ".join(f"{a:.1f}-{b:.1f} {v}" for a, b, v in runs if b - a >= 0.3 or v != VISIBLE))
    if game is not None and game.balls:
        for b in game.balls:
            runs = timeline(states, lambda s: next((x["state"] + (f" (cup {x['inside']})" if x["inside"] is not None else "")
                                                    for x in s["ball"]["balls"] if x["id"] == b.id), None))
            print(f"  {b.id}{'*' if b is game.target else ''}: " + ", ".join(f"{a:.1f}-{c:.1f} {v}" for a, c, v in runs if c - a >= 0.3))
        print("events:\n" + "\n".join(f"  {t:6.2f} s  {e}" for t, e in game.events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
