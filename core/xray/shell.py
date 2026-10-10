"""The shell game: balls hidden under cups. Which cup holds each ball through the shuffle, and the verdict when
a cup is lifted.

    cups, balls = split(objects)                          # the scan's objects (xray.track.scan_objects)
    tracker = Tracker(cups, table_z, labels={c["label"] for c in cups})
    game = ShellGame(tracker, balls, table_z, scan)
    tracker.update(frame, detections); game.update(frame, detections)       # every frame, in this order
    msg = game.message(tracker.message(frame))           # adds the cups' balls and the "ball" block

Balls are told apart by colour, not by position: the hue of their saturated pixels (OpenCV hue, 0-180) against
red 2, yellow 25, green 51, blue 108, purple 140; the scan's views give each ball its colour. The target is the
yellow ball (else the first one); target_id is the cup holding it, null after it is revealed, and before it is
first hidden the object chosen beforehand (Qwen-3D's answer), if any. Per ball:
  VISIBLE       a detection of its colour near where it was (with one ball, any colour)
  INSIDE        it went under a cup: unseen 2 frames while fully in view, with a cup's centre within the cup's
                radius + 1.2 cm of it, that cup resting on the table (not lifted) and slower than 0.15 m/s. It
                rides with that cup from then on, wherever the cup goes and whether or not the cup is seen.
  OCCLUDED / OUT_OF_VIEW / LOST   otherwise, explained like any tracked object (xray.track)
A cup is lifted when its top is 3.5 cm and its bottom 2 cm above where they rest, 2 frames in a row. A ball
INSIDE for 1.5 s or more that is seen again, not under a resting cup, is revealed. The verdict compares the cup
we said with the cup that stood over the place it shows up within the last 3 s (a lifted one first).
"""
import math

import cv2
import numpy as np

from . import frames as fr
from . import track as tk

PALETTE = {"red": 2, "yellow": 25, "green": 51, "blue": 108, "purple": 140}     # OpenCV hue (0-180)
TARGET_COLOUR = "yellow"
INSIDE = "INSIDE"
BALL_RADIUS = 0.032


def split(objects):
    """The scan's objects -> (cups and other containers, balls)."""
    balls = [o for o in objects if tk.rf.kind(o["label"]) == "ball"]
    return [o for o in objects if o not in balls], balls


def colour_of(rgb_pixels, min_sat=80, min_val=50, min_share=0.15):
    """The palette colour of some RGB pixels (N x 3 uint8): the circular mean hue of the saturated ones, or None
    if fewer than min_share of them are saturated."""
    if not len(rgb_pixels):
        return None
    hsv = cv2.cvtColor(np.asarray(rgb_pixels, np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[(hsv[:, 1] >= min_sat) & (hsv[:, 2] >= min_val)]
    if len(sat) < max(5, min_share * len(hsv)):
        return None
    ang = sat[:, 0].astype(float) * (np.pi / 90)                     # hue 0-180 -> angle 0-2 pi
    hue = (math.degrees(math.atan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2) % 180

    def gap(h):
        d = abs(hue - h) % 180
        return min(d, 180 - d)
    return min(PALETTE, key=lambda c: gap(PALETTE[c]))


def scan_colour(scan, points, views=8):
    """A ball's colour from the scan: its twin's points projected into the keyframes that see it, sampled."""
    pts = np.asarray(points, float)[:: max(1, len(points) // 300)]
    votes = {}
    for k in fr.frame_ids(scan):
        rgb, depth, conf, K, T = fr.read_frame(scan, k)
        u, v, z = fr.project(pts, K, T)
        h, w = depth.shape
        ok = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if ok.sum() < 20:
            continue
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        front = np.abs(depth[vi, ui] - z[ok]) < 0.02                  # the ball itself, not something before it
        if front.sum() < 10:
            continue
        c = colour_of(rgb[vi[front], ui[front]])
        if c:
            votes[c] = votes.get(c, 0) + 1
        if sum(votes.values()) >= views:
            break
    return max(votes, key=votes.get) if votes else None


class Ball:
    def __init__(self, obj, colour):
        self.colour = colour
        self.id = f"{colour} ball" if colour else "ball"
        self.track = tk.Track(self.id, obj["label"], obj["position"], obj.get("size"), obj.get("points"))
        self.inside = None                          # the cup's id while INSIDE
        self.hidden_t = None                        # when it went under
        self.verdict = None

    @property
    def state(self):
        return INSIDE if self.inside is not None else self.track.state

    def as_dict(self, t):
        tr = self.track
        seen = self.state == tk.VISIBLE
        return dict(id=self.id, label="ball", color=self.colour, state=self.state, inside=self.inside,
                    position=tk.r4(tr.position), since_seen_s=0.0 if seen else round(t - tr.last_seen_t, 2),
                    detection_score=round(tr.score, 3) if seen and tr.score is not None else None,
                    verdict=self.verdict)


class ShellGame:
    def __init__(self, tracker, balls, table_z=None, scan=None, log=print):
        self.tracker, self.table_z, self.log = tracker, table_z, log
        self.balls = []
        for o in balls:
            c = scan_colour(scan, o["points"]) if scan is not None and o.get("points") is not None else None
            self.balls.append(Ball(o, c))
        names = [b.colour for b in self.balls]
        self.target = next((b for b in self.balls if b.colour == TARGET_COLOUR), self.balls[0] if self.balls else None)
        self.rest = {}                              # cup id -> (recent tops, recent bottoms) while resting
        self.up = {}                                # cup id -> frames in a row lifted
        self.lifted_t = {}                          # cup id -> last time it was lifted
        self.events = []
        self.measurements = []
        self.t = None
        self.chosen = None                          # a target picked before the game (Qwen-3D's answer), if any
        self.hidden_once = False
        log(f"shell game: {len(self.balls)} balls ({', '.join(str(n) for n in names) or 'none'}), "
            f"target {self.target.id if self.target else 'none'}")

    def event(self, t, text):
        self.events.append((round(float(t), 2), text))
        self.log(f"  {t:6.2f} s  {text}")

    def cups(self):
        return [tr for tr in self.tracker.tracks if tk.rf.kind(tr.label) != "ball"]

    def lifted(self, cup):
        return self.up.get(cup.id, 0) >= 2

    def resting(self, cup):
        """On the table for sure: its latest detection showed all of it, not lifted."""
        return cup.bottom_seen is not None and not self.lifted(cup)

    def watch_lifts(self, t):
        """A cup is lifted when its top is 3.5 cm or more above where it rests and its bottom 2 cm or more, 2
        frames in a row: a hand on top of a cup moves only its top, a ragged mask edge only its bottom. Where they
        rest: the medians of the cup's recent tops and bottoms while not lifted that fit it (the top within 3 cm
        of the table + its height, the bottom within 8 cm of the table); until there are 3, those guesses."""
        floor = self.table_z if self.table_z is not None else 0.0
        for cup in self.cups():
            top, bottom = cup.top_seen, cup.bottom_seen
            if top is None or cup.unseen:
                continue
            tops, bottoms = self.rest.setdefault(cup.id, ([], []))
            expect = floor + float(cup.size[2]) + 0.01
            base_top = float(np.median(tops)) if len(tops) >= 3 else expect
            base_bottom = float(np.median(bottoms)) if len(bottoms) >= 3 else floor + 0.05
            if top - base_top > 0.035 and bottom - base_bottom > 0.02:
                self.up[cup.id] = self.up.get(cup.id, 0) + 1
                self.lifted_t[cup.id] = t
                if self.up[cup.id] == 2:
                    held = [b for b in self.balls if b.inside == cup.id]
                    self.event(t, f"cup {cup.id} lifted" + (f": the {' and '.join(b.id for b in held)} should be on the table"
                                                            if held else ""))
            else:
                self.up[cup.id] = 0
                if abs(top - expect) < 0.03:
                    tops.append(top)
                    del tops[:-30]
                if -0.01 < bottom - floor < 0.08:
                    bottoms.append(bottom)
                    del bottoms[:-30]

    def measure_balls(self, frame, detections, clean):
        out = []
        for d in detections:
            if tk.rf.kind(d.label) != "ball":
                continue
            m = tk.measure(frame, d, clean, self.table_z, BALL_RADIUS)
            if m is None:
                continue
            mask = tk.rf.erode(d.mask_at(frame.rgb.shape[:2]), 2)
            m.colour = colour_of(frame.rgb[mask])
            out.append(m)
        return tk.one_per_place(out, 0.04)

    def candidates(self, ball, ms):
        if len(self.balls) == 1:
            return ms
        return [m for m in ms if m.colour == ball.colour]

    def update(self, frame, detections):
        t = frame.t
        dt = 1 / 15 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        clean = getattr(frame, "clean", None)                       # the tracker's, for this frame
        if clean is None:
            clean = tk.rf.clean_pixels(frame.depth, frame.conf, stride=2)
        ms = self.measure_balls(frame, detections, clean)
        self.measurements = ms
        self.watch_lifts(t)
        cups = {c.id: c for c in self.cups()}
        used = set()
        for b in self.balls:
            tr = b.track
            if tr.last_seen_t is None:
                tr.last_seen_t = tr.ok_t = t
            tr.predict(dt)
            pool = [m for m in self.candidates(b, ms) if id(m) not in used]
            if b.inside is not None:
                cup = cups.get(b.inside)
                m = self.revealed(b, cup, pool, t)
                if m is None:
                    if cup is not None:                 # rides with its cup
                        tr.position = np.r_[cup.position[:2], tr.position[2]]
                    continue
                used.add(id(m))
                self.reveal(b, cup, m, t)
                continue
            m = min(pool, key=lambda m: tk.distance(tr.predicted, m.position), default=None)
            if m is not None and tk.distance(tr.predicted, m.position) <= max(tr.gate(dt), 0.15):
                used.add(id(m))
                tr.seen(m, t, dt, self.table_z)
                continue
            tr.missed(self.tracker.explain(tr, frame), t)
            cup = self.went_under(b, frame)
            if cup is not None:
                b.inside, b.hidden_t, b.verdict = cup.id, t, None
                tr.position = np.r_[cup.position[:2], tr.position[2]]
                self.event(t, f"the {b.id} went under cup {cup.id}")
        held = {}
        for b in self.balls:
            if b.inside is not None:
                held.setdefault(b.inside, []).append(b.id)
        for cup in self.cups():
            cup.extra = dict(has_ball=cup.id in held, balls_inside=sorted(held.get(cup.id, [])))
        self.tracker.others = [("ball", b.track.position) for b in self.balls if b.inside is None]
        tgt = self.target                               # its cup; before it is first hidden, the chosen object
        if tgt is not None and tgt.inside is not None:
            self.hidden_once = True
            self.tracker.target_id = tgt.inside
        else:
            self.tracker.target_id = None if self.hidden_once else self.chosen
        for cup in self.cups():
            cup.target = cup.id == self.tracker.target_id

    def went_under(self, b, frame, margin=10):
        """The cup the ball just went under, or None: unseen 2 frames while fully in view, a resting, slow cup
        over it."""
        tr = b.track
        if tr.unseen < 2:
            return None
        u, v, z = fr.project(tr.position[None], frame.K, frame.T)
        h, w = frame.depth.shape
        if not (z[0] > 0.05 and margin <= u[0] < w - margin and margin <= v[0] < h - margin):
            return None
        best = None
        for cup in self.cups():
            d = math.hypot(*(cup.position[:2] - tr.position[:2]))
            if d <= cup.radius + 0.012 and self.resting(cup) and cup.speed < 0.15 and (best is None or d < best[0]):
                best = (d, cup)
        return best[1] if best else None

    def revealed(self, b, cup, pool, t, near=0.15):
        """A detection that shows an INSIDE ball out again, or None: hidden 1.5 s or more, seen near where its cup
        stood in the last 3 s, and not under a resting cup."""
        if t - b.hidden_t < 1.5 or not pool:
            return None
        spots = [p for _, p in cup.history] if cup is not None else []
        spots.append(b.track.position)
        for m in sorted(pool, key=lambda m: -m.score):
            if min(math.hypot(*(p[:2] - m.position[:2])) for p in spots) > near:
                continue
            if any(not self.lifted(c) and math.hypot(*(c.position[:2] - m.position[:2])) < c.radius for c in self.cups()):
                continue                                # under a resting cup: that's the cup, not the ball
            return m
        return None

    def reveal(self, b, cup, m, t):
        """The verdict: the cup over the place the ball shows up, within the last 3 s (one lifted then first)."""
        def closest(c):
            return min((math.hypot(*(p[:2] - m.position[:2])) for _, p in c.history), default=math.inf)
        cups = self.cups()
        lifted = [c for c in cups if t - self.lifted_t.get(c.id, -math.inf) <= tk.HISTORY_S and closest(c) <= 0.12]
        actual = min(lifted or cups, key=closest) if cups else None
        cm = round(100 * closest(actual), 1) if actual is not None else None
        b.verdict = dict(believed=b.inside, actual=None if actual is None else actual.id,
                         right=actual is not None and actual.id == b.inside, cm=cm)
        self.event(t, f"reveal: the {b.id} was under cup {b.verdict['actual']} (it stood {cm} cm away); we said cup "
                      f"{b.inside}: {'RIGHT' if b.verdict['right'] else 'WRONG'}")
        b.inside = None
        b.track.jump(m.position, t, m.score)

    def message(self, msg):
        """The tracker's message with the ball block added (the target ball, and every ball in `balls`)."""
        if not self.balls:
            return msg
        t = msg["t"]
        msg["target_id"] = self.tracker.target_id
        for o in msg["objects"]:
            o["target"] = o["id"] == self.tracker.target_id
        msg["ball"] = dict(self.target.as_dict(t), balls=[b.as_dict(t) for b in sorted(self.balls, key=lambda b: b.id)])
        return msg
