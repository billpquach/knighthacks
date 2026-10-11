"""The live x-ray: the phone's frames -> detector -> tracker (-> shell game) -> the x-ray view and the state stream,
served to the glasses, the rover and the viewer.

    python -m xray.live --scan-seconds 20 [--ball] [--port 8080]                  # scan the table, then track
    python -m xray.live --scan-seconds 20 --ask [--ball]      # Qwen-3D loads during the scan; ask, then "go" to track
    python -m xray.live --scan <scan folder> [--ball | --results <ground3d output>]  # an earlier scan, same session
    python -m xray.live --scan-seconds 25 --ball --rover --rover-scan   # the rover scans by itself; "drive" fetches the target
    python -m xray.live --scan <scan> --replay <track folder> [--ball] [--speed 1]   # a recording instead of the phone
                                                                     # (--loop: over and over, a fresh game each time)

Endpoints, CORS on all (the same as replay_server.py, so the viewer and every client work unchanged):
    /state          server-sent events: the newest state message, up to 10 a second; on a slow link fewer, never
                    old ones (a small send buffer, so nothing queues up); ?cloud=0 leaves out the point clouds
    /state.json     the newest state message (?cloud=0 as above)
    /frame.jpg      the newest x-ray image (JPEG); ?after=SEQ waits up to 2 s for a newer one; X-Seq, X-Age-Ms headers
    /stream.mjpg    the x-ray view as an MJPEG stream (<img src=".../stream.mjpg">): no round trip per frame, so it
                    keeps up over a slow, far link; like /state, fewer frames there, never old ones
    /scene          the scan: table, coloured cloud, and each object's twin (for the viewer's 3D view)
    /scene/heal     the scan's holes where objects stood, filled in from live depth as the camera sees the bare
                    surface there: {version, points, age_s} (heal_version in each message says when it grew)
    /metrics        the Jetson's resource use (core/jetson)
    /viewer-info    {"mode": "live"}
    POST /prompt    a question for Qwen-3D, or "go" (JSON {"text": ...} or plain text): the dashboard, a voice front end
    /               the viewer, once built (viewer/dist)
Every message has scene_version, which goes up each time /scene changes (fetch it again then). Until tracking
starts the messages are {"phase": "scan"} with no objects, then with --ask {"phase": "ask"}: each
answer as an object (id q1, q2, ..., its query, score and cloud; the newest is the target; "candidates": what Qwen-3D
weighed and passed over, each with its score and a few points) and a "qwen" block (status, ready, thinking, answers
with top5 scores, seconds, tokens, memory; "sweep" for the newest answer's first seconds: each point it weighed with its
confidence), then {"phase": "starting"} while YOLOE loads, then {"phase": "track"}. The camera view shows the same:
the sweep, the runners-up with their scores, and a card with the question, what it picked and how sure. Each message carries about 1500 points of
the current depth frame (scene.point_cloud, for the viewer's live 3D view; --live-cloud 0: none). Every message also
goes to <out>/states.jsonl (without the cloud); with --video the x-ray view to <out>/xray.mp4; with --record the raw frames to
<out>/track, which --replay (or python -m xray.track) plays back.

Separate threads, so the frame rate is set by the slowest stage, not by all of them together: frames and detections;
tracking, the game and the x-ray view; JPEG and serving (always the newest frame, so the feed never lags); the log,
video and recording (live, a frame they can't keep up with is skipped there, never in the feed).

Record3D at 15 fps (Settings in the app): faster than the Jetson processes, the frames queue up before they reach
this program and the view falls further behind every second (60 fps: about 3 s more delay every second).

The scan and the tracking must share the phone's ARKit session (don't restart Record3D in between): live frames are
put in the scan's world with its meta.json world_from_arkit. --results picks the target from Qwen-3D's first answer
(python -m ground3d.ground --out); --ball plays the shell game (the yellow ball's cup is the target).

--ask, one continuous run: Qwen-3D starts loading in its own process (ground3d.ground --serve) before the scan, so
it is ready about when the scan ends. Questions come from POST /prompt or the console (typed, one per line); each
answer's cloud is drawn on the camera view as it comes. "go" stops Qwen-3D (all its memory is freed), then YOLOE
loads and tracking starts, the last answer's object as the target (a ball: its cup once it is hidden).

The x-ray view: every object's digital twin, moved to where the tracker has it, drawn by xray.render (the target
brighter; a hidden object glowing through whatever covers it); a ball under a cup drawn as a disc of its colour
through the cup; a tag on each object; status lines at the top, and the verdict when a ball is revealed.
"""
import argparse
import json
import math
import mimetypes
import os
import queue
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import cv2
import numpy as np

from . import detect, render
from . import frames as fr
from . import rover as rv
from . import track as tk

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIST = ROOT / "viewer" / "dist"
BALL_BGR = {"red": (40, 40, 230), "yellow": (40, 215, 250), "green": (60, 200, 60), "blue": (230, 130, 40),
            "purple": (200, 60, 170), None: (230, 230, 230)}
TEXT = (235, 245, 250)
CYAN = (255, 220, 0)
AMBER = (40, 190, 255)
PHONE_TIP = ("Keep Record3D open and streaming, and the iPhone awake (Settings > Display & Brightness > Auto-Lock > "
             "Never); don't restart Record3D between the scan and tracking")
# Record3D's frame rate must not exceed what the Jetson processes (about 15 a second while tracking): frames it
# can't keep up with queue up before they reach us (in the USB stream and record3d's decoder), and the view falls
# further behind every second. Set the app to 15 fps.


# ---------------------------------------------------------------------------------------- serving
class Server:
    """The newest state message and x-ray JPEG, served over HTTP from a background thread."""

    def __init__(self, host="0.0.0.0", port=8080, metrics=None):
        self.cond = threading.Condition()
        self.state, self.jpg, self.seq, self.stamp = {"phase": "scan"}, None, 0, time.time()
        self.scene, self.scene_version, self.metrics = None, 0, metrics
        self.healer = None                              # /scene/heal: the scan's holes filled in from live depth
        self.prompts = queue.Queue()                    # questions and commands: POST /prompt, or typed in the console
        self.httpd = ThreadingHTTPServer((host, port), handler(self))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def publish(self, img=None, state=None, quality=75):
        jpg = None
        if img is not None:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            jpg = buf.tobytes() if ok else None
        with self.cond:
            if jpg is not None:
                self.jpg, self.seq, self.stamp = jpg, self.seq + 1, time.time()
            if state is not None:                       # (scene_version first: "scene" stays the last key)
                self.state = {"scene_version": self.scene_version, **state}
            self.cond.notify_all()

    def set_scene(self, doc):
        """A new /scene document; each message's scene_version tells clients to fetch it again."""
        with self.cond:
            self.scene, self.scene_version = doc, self.scene_version + 1

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def handler(server):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "X-Seq, X-Age-Ms")
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def body(self, data, kind, cache="no-store", extra=()):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", cache)
            self.send_header("Content-Length", str(len(data)))
            for k, v in extra:
                self.send_header(k, v)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            """POST /prompt {"text": "..."} (or the plain text): a question for Qwen-3D, or "go"."""
            if urlsplit(self.path).path != "/prompt":
                self.send_error(404, "Not found")
                return
            raw = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 10000)).decode("utf-8", "replace")
            try:
                text = str(json.loads(raw).get("text", ""))
            except (ValueError, AttributeError):
                text = raw
            text = text.strip()
            if text:
                server.prompts.put(text)
            with server.cond:
                phase = server.state.get("phase")
            self.body(json.dumps({"ok": bool(text), "phase": phase}).encode(), "application/json")

        def do_GET(self):
            url = urlsplit(self.path)
            path = unquote(url.path)
            clouds = parse_qs(url.query).get("cloud", ["1"])[0].lower() not in ("0", "false", "no")
            if path == "/state":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:                                    # a small send buffer: a slow link gets fewer messages, never
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 32 * 1024)   # old ones
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
                last = None
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.state is not last, timeout=1.0)
                            state = last = server.state
                        state = state if clouds else without_clouds(state)
                        self.wfile.write(("data: " + json.dumps(state, separators=(",", ":")) + "\n\n").encode())
                        self.wfile.flush()                  # (waits while the link is full; then the newest state)
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return
            elif path == "/state.json":
                with server.cond:
                    state = server.state
                state = state if clouds else without_clouds(state)
                self.body(json.dumps(state, separators=(",", ":")).encode(), "application/json")
            elif path == "/frame.jpg":
                try:
                    after = int(parse_qs(url.query).get("after", ["-1"])[0])
                except ValueError:
                    after = -1
                with server.cond:
                    server.cond.wait_for(lambda: server.jpg is not None and server.seq > after, timeout=2.0)
                    jpg, seq, stamp = server.jpg, server.seq, server.stamp
                if jpg is None:
                    self.send_error(503, "No frame yet")
                    return
                self.body(jpg, "image/jpeg", extra=[("X-Seq", str(seq)), ("X-Age-Ms", str(int(1000 * (time.time() - stamp))))])
            elif path == "/stream.mjpg":                # the x-ray view as a video: no round trip per frame
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 64 * 1024)   # never old frames
                    self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
                seq = None
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.jpg is not None and server.seq != seq, timeout=2.0)
                            jpg, seq = server.jpg, server.seq
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg)
                                         + jpg + b"\r\n")
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    return
            elif path == "/scene":
                if server.scene is None:
                    self.send_error(404, "No scene yet")
                    return
                self.body(server.scene, "application/json", cache="no-cache")
            elif path == "/scene/heal":
                h = server.healer
                self.body(json.dumps(h.document() if h else dict(version=0, points=[], age_s=[])).encode(),
                          "application/json", cache="no-cache")
            elif path == "/metrics":
                latest = server.metrics.latest() if server.metrics else dict(
                    version=1, device=None, memory_kind="shared", available=False, reason="not_jetson", metrics={},
                    age_ms=None, sample_id=None)
                self.body(json.dumps(latest).encode(), "application/json")
            elif path == "/favicon.ico":                  # (browsers ask; nothing to send)
                self.send_response(204)
                self.end_headers()
            elif path == "/viewer-info":
                self.body(json.dumps({"mode": "live"}).encode(), "application/json")
            else:
                file = (VIEWER_DIST / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
                if not file.is_relative_to(VIEWER_DIST.resolve()) or not file.is_file():
                    self.send_error(404, "Not found (the viewer: cd viewer && npm ci && npm run build)")
                    return
                self.body(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream", "no-cache")
    return Handler


def without_clouds(state):
    """The state without its point clouds (?cloud=0, for clients that don't draw them, such as the glasses)."""
    out = {}
    for key, v in state.items():                        # same keys in the same order (scene stays last)
        if key == "objects":
            v = [{k: x for k, x in o.items() if k != "point_cloud"} for o in v]
        elif key == "scene":
            v = {k: x for k, x in v.items() if k != "point_cloud"}
        out[key] = v
    return out


def drive_target(tracker, game, track_id=None):
    """The track to drive to: `track_id`'s, else the tracker's target (the hidden ball's cup, Qwen-3D's answer), else
    the game's target ball while it is in the open."""
    tracks = list(tracker.tracks) + ([b.track for b in game.balls] if game is not None else [])
    want = track_id if track_id is not None else tracker.target_id
    if want is None and game is not None and game.target is not None and game.target.inside is None:
        want = game.target.track.id
    return next((tr for tr in tracks if tr.id == want), None)


def drive_command(text):
    """A typed or posted line -> (True, the named target or None for "the target") for a drive command, else
    (False, None). "drive", "drive to cup 2", "go to the yellow ball", "touch 1", "fetch the green ball"."""
    t = text.lower().strip(" .!?")
    if t in DRIVE_WORDS:
        return True, None
    m = DRIVE_VERBS.match(t)
    if not m or (m.group(1) == "go" and not m.group(2)):            # ("go" alone ends the questions)
        return False, None
    rest = re.sub(r"^(the|a)\s+", "", m.group(3).strip())
    return True, (None if rest in ("", "it", "target", "the target") else rest)


def named_target(name, tracker, game):
    """"cup 2" / "2" / "two" / "yellow ball" / "yellow" / "ball" -> (track, note) or (None, why not). A ball under a
    cup is that cup."""
    words = [str(NUMBERS.get(w, w)) for w in re.findall(r"[a-z]+|\d+", name)]
    numbers = [int(w) for w in words if w.isdigit()]
    if numbers:
        tr = next((tr for tr in tracker.tracks if tr.id == numbers[0]), None)
        return (tr, "") if tr is not None else (None, f"no cup {numbers[0]} (cups: "
                                                      f"{', '.join(str(t.id) for t in tracker.tracks) or 'none'})")
    balls = list(game.balls) if game is not None else []
    colours = [w for w in words if w in BALL_BGR]
    ball = next((b for b in balls if colours and b.colour == colours[0]), None) if colours else \
        (balls[0] if "ball" in words and len(balls) == 1 else None)
    if ball is None:
        known = [f"cup {t.id}" for t in tracker.tracks] + [b.id for b in balls]
        return None, f"don't know \"{name}\" (try: {', '.join(known) or 'nothing tracked yet'})"
    if ball.inside is not None:
        cup = next((tr for tr in tracker.tracks if tr.id == ball.inside), None)
        return (cup, f"the {ball.id} is under cup {ball.inside}") if cup is not None else (None, f"cup {ball.inside} is gone")
    return ball.track, ""


def obstacles_besides(tracker, game, target_id):
    """[(position, radius)] of everything tracked on the floor except the target (and balls inside cups)."""
    out = [(tr.position, tr.radius) for tr in tracker.tracks if tr.id != target_id]
    if game is not None:
        out += [(b.track.position, b.track.radius) for b in game.balls if b.inside is None and b.track.id != target_id]
    return out


class LateQwen:
    """--qwen-stay: Qwen-3D kept loaded through tracking, asked in the background. Its answers are about the scan
    (where things stood then); answer() turns one into the tracked object that stood there, wherever it is now."""

    def __init__(self, qwen, scan):
        self.qwen, self.scan, self.answers, self.busy = qwen, scan, queue.Queue(), None

    def ask(self, text):
        if self.busy:
            return False
        self.busy = text

        def run():
            try:
                res = self.qwen.ask(self.scan, text)
            except Exception as e:
                res = {"error": str(e)}
            self.answers.put((text, res))
            self.busy = None
        threading.Thread(target=run, daemon=True).start()
        return True

    @staticmethod
    def answer(res, tracker, game):
        """(track, note) for Qwen-3D's answer, or (None, why)."""
        a = (res.get("answers") or [{}])[0]
        point = (a.get("refined") or {}).get("center") or a.get("center")
        if res.get("error") or point is None:
            return None, res.get("error") or "nothing found"
        tracks = list(tracker.tracks) + ([b.track for b in game.balls if b.inside is None] if game is not None else [])
        point = np.asarray(point, float)
        near = min(tracks, key=lambda tr: float(np.linalg.norm(tr.origin[:2] - point[:2])), default=None)
        if near is None or np.linalg.norm(near.origin[:2] - point[:2]) > 0.15:
            return None, f"no tracked object where it pointed ({tk.r4(point)})"
        return near, f"score {a.get('score', 0):.2f}, {a.get('seconds', 0):.1f} s"


def drive_step(server, rover, nav, tracker, game, f, log=print, late=None):
    """One tracking frame of the rover: the prompts that start or stop it, then the wheel speeds toward the target
    (its remembered position when unseen: the last stretch is out of the phone's view), around the other objects.
    A cup is touched; a ball (it would roll away) is stopped just short of."""
    while True:
        try:
            text = server.prompts.get_nowait()
        except queue.Empty:
            break
        is_drive, name = drive_command(text)
        if is_drive:
            tr, note = named_target(name, tracker, game) if name else (drive_target(tracker, game), "")
            if tr is None and late is not None and name and not re.search(r"\d", name):
                if late.ask(name):                      # a description: Qwen-3D, still loaded (--qwen-stay)
                    log(f"rover: asking Qwen-3D for '{name}'")
                else:
                    log(f"rover: Qwen-3D is still on '{late.busy}'")
            elif tr is None:
                log(f"rover: {note or 'no target to drive to yet'}")
            else:
                cup = any(t is tr for t in tracker.tracks)
                nav.start(tr.id, touch=cup)
                log(f"rover: {note + ': ' if note else ''}driving to {object_name(tr, game) if cup else tr.id}"
                    + (" (to touch it)" if cup else ""))
        elif text.lower().strip(" .!?") in STOP_WORDS:
            nav.stop()
            rover.stop()
            log("rover: stopped")
    while late is not None:                             # Qwen-3D's answers, as they come
        try:
            text, res = late.answers.get_nowait()
        except queue.Empty:
            break
        tr, note = late.answer(res, tracker, game)
        if res.get("ram_peak_gb") or res.get("ram_gb"):
            note += f"; RAM peak {res.get('ram_peak_gb') or res.get('ram_gb')} GB"
        if tr is None:
            log(f"rover: Qwen-3D on '{text}': {note}")
        else:
            cup = any(t is tr for t in tracker.tracks)
            nav.start(tr.id, touch=cup)
            log(f"rover: Qwen-3D: '{text}' -> {object_name(tr, game) if cup else tr.id} ({note}); driving to it")
    if not nav.active:
        return
    tr = drive_target(tracker, game, nav.target_id)
    if tr is None:
        nav.stop("lost")
    else:
        was = nav.approach
        rover.drive(*nav.step(f.T, tr.position, tr.radius, obstacles_besides(tracker, game, tr.id)), yaw=rv.heading(f.T))
        if nav.approach is not None and was is None:
            log(f"rover: the way is blocked, going around via ({nav.approach[0]:+.2f}, {nav.approach[1]:+.2f})")
    if not nav.active:
        rover.stop()
        log(f"rover: {nav.state}" + (f", front edge {nav.info.get('gap_m', 0) * 100:+.0f} cm from {nav.target_id}'s edge"
                                     if nav.state in ("arrived", "touched") else ""))


SWEEP_S = 2.4                                           # an answer's confidence sweep, then the answer alone


def thin(points, n):
    """At most n of the points, evenly through them."""
    return points[np.linspace(0, len(points) - 1, min(len(points), n)).astype(int)] if len(points) else points


def heat_bgr(h):
    """Confidence 0..1 -> BGR: deep blue, cyan, amber, white."""
    stops = np.array([[0.0, 120, 40, 10], [0.35, 230, 200, 40], [0.7, 40, 190, 255], [1.0, 235, 250, 255]])
    return np.stack([np.interp(h, stops[:, 0], stops[:, k]) for k in (1, 2, 3)], -1)


def draw_heat(img, heat, K, T, fade=1.0):
    """Qwen-3D's per-point confidence glowing on the view (the strongest points biggest, drawn last)."""
    u, v, z = render.project(heat[:, :3], K, T)
    h, (hh, ww) = heat[:, 3], img.shape[:2]
    ok = (z > 0.05) & (u >= 0) & (u < ww) & (v >= 0) & (v < hh)
    if not ok.any() or fade <= 0:
        return
    layer, col = img.copy(), heat_bgr(h[ok])
    for i in np.argsort(h[ok]):
        cv2.circle(layer, (int(u[ok][i]), int(v[ok][i])), 3 if h[ok][i] > 0.5 else 2, tuple(int(c) for c in col[i]), -1)
    cv2.addWeighted(layer, 0.85 * fade, img, 1 - 0.85 * fade, 0, img)


def draw_candidates(img, candidates, K, T):
    """The answers Qwen-3D weighed and passed over: faint dots and their scores."""
    hh, ww = img.shape[:2]
    for c in candidates:
        u, v, z = render.project(thin(c["points"], 200), K, T)
        ok = (z > 0.05) & (u >= 0) & (u < ww) & (v >= 0) & (v < hh)
        if not ok.any():
            continue
        layer = img.copy()
        for x, y in zip(u[ok].astype(int), v[ok].astype(int)):
            cv2.circle(layer, (x, y), 1, (215, 215, 215), -1)
        cv2.addWeighted(layer, 0.5, img, 0.5, 0, img)
        top = int(np.argmin(v[ok]))
        tag(img, f"{c['score']:.2f}", (int(u[ok][top]), int(v[ok][top]) - 8), (200, 200, 200), 0.4)


def decision_card(img, a):
    """The newest answer at the bottom of the view: the question, what it picked, how sure, how fast, on what; and
    its top scores as bars (the first one picked)."""
    hh, ww = img.shape[:2]
    x0, y0, x1, y1 = 8, hh - 104, ww - 8, hh - 8
    box = img[y0:y1, x0:x1]
    box[:] = (box * 0.35).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "QWEN-3D", (x0 + 8, y0 + 18), font, 0.42, AMBER, 1, cv2.LINE_AA)
    cv2.putText(img, f"'{a['query']}'"[:44], (x0 + 76, y0 + 18), font, 0.42, TEXT, 1, cv2.LINE_AA)
    cv2.putText(img, f"-> {a.get('target') or 'object'}  {100 * a['score']:.0f}%", (x0 + 8, y0 + 44), font, 0.6, AMBER, 1, cv2.LINE_AA)
    info = f"{a['seconds']:.1f} s on the Jetson, 8-bit" + (f", {a['ram_gb']:.1f} GB" if a.get("ram_gb") else "")
    cv2.putText(img, info, (x0 + 8, y0 + 64), font, 0.4, TEXT, 1, cv2.LINE_AA)
    top = (a.get("top5") or [a["score"]])[:5]
    bw = (x1 - x0 - 16) // max(1, len(top))
    for j, sc in enumerate(top):
        bx = x0 + 8 + j * bw
        cv2.rectangle(img, (bx, y1 - 22), (bx + bw - 6, y1 - 10), (70, 70, 70), -1)
        cv2.rectangle(img, (bx, y1 - 22), (bx + int((bw - 6) * min(1.0, sc)), y1 - 10), AMBER if j == 0 else (170, 170, 170), -1)
        cv2.putText(img, f"{sc:.2f}", (bx, y1 - 26), font, 0.35, AMBER if j == 0 else TEXT, 1, cv2.LINE_AA)


class Healer:
    """The scan's holes, where objects stood (cut from /scene: they move), filled in from the live depth wherever the
    camera now sees the bare surface there: one point per 6 mm cell, each with when it came, for /scene/heal (the
    viewer shows them arriving: the LiDAR repairing the map). heal_version in each message says when it grew."""

    def __init__(self, tracks, table_z, cell=0.006, band=0.012, cap=6000):
        self.holes = np.array([[*tr.origin[:2], tr.radius + 0.02] for tr in tracks], float).reshape(-1, 3)
        self.table_z, self.cell, self.band, self.cap = table_z, cell, band, cap
        self.cells, self.points, self.born = set(), [], []
        self.version, self.now, self.lock = 0, 0.0, threading.Lock()

    def add(self, f, standing):
        """One frame's depth; standing: [(xy, radius)] of the objects there now (not surface)."""
        self.now = float(f.t)
        if not len(self.holes) or self.table_z is None or len(self.points) >= self.cap:
            return
        p = fr.backproject(f.depth, f.K, f.T, (f.conf == 2) & (f.depth > 0) & (f.depth < 2.0), 3)
        p = p[np.abs(p[:, 2] - self.table_z) < self.band]       # the surface only (not hands, not objects)
        if len(p):
            inside = np.hypot(p[:, None, 0] - self.holes[None, :, 0], p[:, None, 1] - self.holes[None, :, 1]) < self.holes[None, :, 2]
            p = p[inside.any(1)]
        for xy, r in standing:
            if len(p):
                p = p[np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) > r + 0.015]
        if not len(p):
            return
        keys, first = np.unique(np.floor(p[:, :2] / self.cell).astype(np.int64), axis=0, return_index=True)
        with self.lock:
            n = len(self.points)
            for k, i in zip(map(tuple, keys), first):
                if k not in self.cells and len(self.points) < self.cap:
                    self.cells.add(k)
                    self.points.append(np.round(p[i], 3).tolist())
                    self.born.append(self.now)
            if len(self.points) > n:
                self.version += 1

    def set_cell(self, cell):
        """Fill as densely as the scan's cloud is (its spacing on the surface); points so far are thinned to it."""
        with self.lock:
            self.cell, self.cells, kept, born = float(np.clip(cell, 0.006, 0.03)), set(), [], []
            for q, b in zip(self.points, self.born):
                k = tuple(np.floor(np.asarray(q[:2]) / self.cell).astype(np.int64))
                if k not in self.cells:
                    self.cells.add(k)
                    kept.append(q)
                    born.append(b)
            self.points, self.born, self.version = kept, born, self.version + 1

    def document(self):
        with self.lock:
            return dict(version=self.version, points=list(self.points), age_s=[round(self.now - b, 2) for b in self.born])


def scene_json(scan, tracks, cut=(), table_z=None, info=None):
    """The /scene document: the scan (viewer/tools/scene_export.py) with each object's twin as offsets from it. The
    scan's own copies of the objects (tracks and `cut`, where they stood when scanned) are taken out of the
    environment cloud: objects move, and a copy left behind would be a ghost. info (a dict) gets the cloud's point
    spacing on the surface around them (the healer fills the holes as densely)."""
    sys.path.insert(0, str(ROOT / "viewer" / "tools"))
    from scene_export import export_scene
    scene = export_scene(scan, table_z=table_z)
    env, colours = np.asarray(scene["cloud"]["points"], float).reshape(-1, 3), scene["cloud"]["colors"]
    keep = np.ones(len(env), bool)
    for tr in list(tracks) + list(cut):
        reach = np.hypot(*(env[:, :2] - tr.origin[:2]).T) < tr.radius + 0.02
        top = tr.origin[2] + 0.6 * float(tr.size[2]) + 0.02
        keep &= ~(reach & (env[:, 2] < top))
    scene["cloud"] = dict(points=np.round(env[keep], 5).tolist(), colors=[c for c, k in zip(colours, keep) if k])
    objs = list(tracks) + list(cut)
    if info is not None and objs and table_z is not None:     # surface points 10-25 cm from the objects
        c = np.array([tr.origin[:2] for tr in objs])
        surf = env[keep & (np.abs(env[:, 2] - table_z) < 0.012)]
        d = np.hypot(surf[:, None, 0] - c[None, :, 0], surf[:, None, 1] - c[None, :, 1]).min(1) if len(surf) else np.zeros(0)
        n = int(((d > 0.10) & (d < 0.25)).sum())
        if n > 50:
            info["surface_spacing"] = float(np.sqrt(len(c) * np.pi * (0.25 ** 2 - 0.10 ** 2) / n))
    scene["objects"] = []
    for tr in list(tracks) + list(cut):                # cups (numeric ids) and balls ("yellow ball")
        pts = np.zeros((0, 3)) if tr.points is None else \
            tr.points[np.linspace(0, len(tr.points) - 1, min(len(tr.points), 2000)).astype(int)] - tr.origin
        scene["objects"].append(dict(id=tr.id, label=tr.label, position=tk.r4(tr.origin), points=np.round(pts, 4).tolist()))
    return json.dumps(scene, separators=(",", ":"), allow_nan=False).encode()


# ---------------------------------------------------------------------------------------- drawing
def tag(img, text, xy, col=TEXT, scale=0.42):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(min(max(2, xy[0] - w // 2), img.shape[1] - w - 4))
    y = int(min(max(h + 4, xy[1]), img.shape[0] - base - 2))
    cv2.rectangle(img, (x - 3, y - h - 3), (x + w + 3, y + base + 2), (24, 22, 18), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def object_name(tr, game):
    inside = tr.extra.get("balls_inside") if tr.extra else None
    return f"{tr.label} {tr.id}" + (f" ({' + '.join(inside)} inside)" if inside else "")


def ball_line(b):
    if b.state == "INSIDE":
        return f"{b.id}: under cup {b.inside}"
    return f"{b.id}: {b.state}"


def draw_view(img, frame, tracker, game=None, fps=None, verdict=None):
    """The x-ray view, drawn on img (BGR) in place."""
    K, T, depth = frame.K, frame.T, frame.depth
    h, w = img.shape[:2]
    tags = []
    for tr in sorted(tracker.tracks, key=lambda tr: tr.target):          # the target last, on top
        if tr.points is None:
            continue
        pts = tr.points + (tr.position - tr.origin)
        info = render.draw_cloud(img, pts, K, T, depth, centre=tr.position, radius=tr.radius, target=tr.target,
                                 glow=tr.state in tk.HIDDEN, order_seed=tr.id)
        if info["label_xy"]:
            state = "" if tr.state == tk.VISIBLE else f": {tr.state}"
            tags.append((object_name(tr, game) + state, (info["label_xy"][0], info["label_xy"][1] - 8),
                         AMBER if tr.target else TEXT))
    if game is not None:
        for b in game.balls:
            if b.state != "INSIDE":
                if b.track.points is not None:
                    tb = b.track
                    render.draw_cloud(img, tb.points + (tb.position - tb.origin), K, T, depth, centre=tb.position,
                                      radius=tb.radius, target=b is game.target, glow=tb.state in tk.HIDDEN,
                                      order_seed=100 + game.balls.index(b))
                continue
            u, v, z = fr.project(b.track.position[None], K, T)
            if z[0] <= 0.05 or not (0 <= u[0] < w and 0 <= v[0] < h):
                continue
            r = max(4, int(K[0, 0] * 0.032 / z[0]))
            c = (int(u[0]), int(v[0]))
            pad = r + 24                                        # the glow: blurred disc, in a patch around it
            x0, y0, x1, y1 = max(0, c[0] - pad), max(0, c[1] - pad), min(w, c[0] + pad), min(h, c[1] + pad)
            glow = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            cv2.circle(glow, (c[0] - x0, c[1] - y0), r + 4, BALL_BGR.get(b.colour, BALL_BGR[None]), -1, cv2.LINE_AA)
            img[y0:y1, x0:x1] = cv2.add(img[y0:y1, x0:x1], cv2.GaussianBlur(glow, (0, 0), 6))
            cv2.circle(img, c, r, BALL_BGR.get(b.colour, BALL_BGR[None]), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, (255, 255, 255), 1, cv2.LINE_AA)
            tags.append((f"{b.id}: under cup {b.inside}", (c[0], c[1] + r + 16), BALL_BGR.get(b.colour, TEXT)))
    for text, xy, col in tags:
        tag(img, text, xy, col)
    lines = []
    if game is not None:
        lines += [(ball_line(b) + ("  <- target" if b is game.target else ""), BALL_BGR.get(b.colour, TEXT))
                  for b in game.balls]
    for tr in tracker.tracks:
        since = "" if tr.state == tk.VISIBLE else f" ({frame.t - tr.last_seen_t:.0f} s)"
        lines.append((f"{object_name(tr, game)}: {tr.state}{since}" + ("  <- target" if tr.target and game is None else ""),
                      AMBER if tr.target else TEXT))
    for i, (text, col) in enumerate(lines[:9]):
        y = 16 + 17 * i
        cv2.rectangle(img, (2, y - 12), (min(w - 2, 10 + 7 * len(text)), y + 4), (24, 22, 18), -1)
        cv2.circle(img, (9, y - 4), 3, col, -1, cv2.LINE_AA)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    if verdict:
        tag(img, verdict, (w // 2, h - 40), (90, 230, 90) if "RIGHT" in verdict else (80, 80, 240), 0.6)
    tag(img, f"t={frame.t:.1f}s" + (f"  {fps:.0f} fps" if fps else ""), (40, h - 8), TEXT, 0.38)
    return img


# ---------------------------------------------------------------------------------------- running
class StillPicker:
    """xray.scan's KeyframePicker, taking one view each time the rover stands still (ScanRoutine.still): the first
    sharp one of that stop."""

    def __init__(self, picker, routine):
        self.picker, self.routine = picker, routine
        routine.wants_views = True                      # (each stop lasts until its view, up to PAUSE_MAX)

    def offer(self, f):
        stop = self.routine.paused_from
        if not self.routine.still or stop == self.routine.viewed or not self.picker.offer(f):
            return False
        self.routine.viewed = stop
        return True

    def __getattr__(self, name):
        return getattr(self.picker, name)


def scan_with_phone(phone, seconds, out, server, min_keyframes=20, rover=None, routine=None, log=print):
    """xray.scan's capture, in this process (the phone stays connected for tracking). It goes on past `seconds`
    (up to twice as long) until it has `min_keyframes` views: the objects' twins need them from all sides. With a
    rover and a routine (rover.ScanRoutine), the rover turns and rolls through the scan by itself, stopping for each
    view, and the scan lasts as long as its moves."""
    from . import scan as sc
    shown, more = [0.0], [False]
    roving = rover is not None and routine is not None

    def progress(f, n, left):
        if roving and not routine.done:
            rover.drive(*routine.step(f.T), yaw=rv.heading(f.T))      # (the phone's own pose steers it)
            if routine.done:
                rover.stop()
                log(f"rover: scan moves done, {n} views")
        if time.monotonic() - shown[0] > 0.2:
            shown[0] = time.monotonic()
            img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
            if roving:
                text = f"SCANNING: the rover looks around, {n} views" + ("  (still: a view)" if routine.still else "")
            elif more[0]:
                text = f"SCANNING: keep circling the cups, {n}/{min_keyframes} views"
            else:
                text = f"SCANNING: circle the cups slowly  {max(0.0, left):4.1f} s, {n}/{min_keyframes} views"
            tag(img, text, (img.shape[1] // 2, 24))
            server.publish(img, {"phase": "scan", "scan": {"views": n, "views_wanted": min_keyframes}})
    if roving:
        picker = StillPicker(sc.KeyframePicker(move=0.03, turn=5.0), routine)      # (one view per stop)
        n_read, t0 = 0, time.monotonic()
        keyframes = []
        while not routine.done and time.monotonic() - t0 < max(60.0, 3 * seconds):
            keyframes, n, blurry = sc.capture(phone, 0.5, picker, progress)
            n_read += n
    else:
        picker = sc.KeyframePicker()
        keyframes, n_read, blurry = sc.capture(phone, seconds, picker, progress)
        extra = 0.0
        while len(keyframes) < min_keyframes and extra < seconds:      # too few views yet: keep going
            more[0] = True
            keyframes, n, blurry = sc.capture(phone, 1.0, picker, progress)
            n_read, extra = n_read + n, extra + 1.0
    if rover is not None:
        rover.stop()
    if len(keyframes) < 8:
        raise SystemExit(f"only {len(keyframes)} keyframes: move around more (10 cm or 15 deg between keyframes)")
    shift, world = sc.fit_world(keyframes)
    sc.apply(shift, keyframes)
    folder = Path(out) / "scan"
    if folder.exists():                                 # a reused --out: no old keyframes or detections mixed in
        shutil.rmtree(folder)
    sc.write_scan(folder, keyframes, shift, source="record3d", stream=phone.info, world=world)
    log(f"scan: {len(keyframes)} keyframes of {n_read} frames -> {folder}")
    return folder


def detect_keyframes(scan, detector, labels, log=print):
    """The scan's detections with the detector already loaded (one model in memory), cached for refine."""
    path = detect.cache_path(scan, labels)
    cached = detect.load_detections(path)
    if cached is None or set(cached) != set(fr.frame_ids(scan)):      # (none yet, or another scan's)
        out, shape = {}, None
        for k in fr.frame_ids(scan):
            rgb = fr.read_frame(scan, k)[0]
            shape = rgb.shape[:2]
            out[k] = detector(rgb)
        detect.save_detections(path, out, labels, shape or (640, 480))
        log(f"detections on {len(out)} keyframes -> {path}")


def qwen_target(results):
    """The world point of Qwen-3D's first answer (refined if it was), or None."""
    res = json.loads((Path(results) / "results.json").read_text())
    for a in res.get("answers", []):
        c = (a.get("refined") or {}).get("center") or a.get("center")
        if c:
            return np.asarray(c, float), a.get("query")
    return None, None


def world_from_phone(scan):
    """Phone frames (z-up ARKit world) -> the scan's world, from the scan's meta.json."""
    from .phone import Y_UP_TO_Z_UP
    meta = json.loads((Path(scan) / "meta.json").read_text())
    if "world_from_arkit" not in meta:
        raise SystemExit(f"{scan}/meta.json has no world_from_arkit: scan with python -m xray.scan or xray.live")
    return np.asarray(meta["world_from_arkit"], float) @ np.linalg.inv(Y_UP_TO_Z_UP)


def frames_from(source, start=None, speed=1.0):
    """Frames of a FolderSource paced at `speed` (0: as fast as they come), or of the phone."""
    if isinstance(source, fr.FolderSource):
        wall0 = rec0 = None
        for f in source:
            if start is not None and f.t < start:
                continue
            if speed > 0:
                if wall0 is None:
                    wall0, rec0 = time.time(), f.t
                lag = (f.t - rec0) / speed - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
            yield f
    else:
        while True:
            yield source.read()


# ---------------------------------------------------------------------------------------- Qwen-3D, before tracking
def qwen_command(jobs, python=None):
    """The Qwen-3D server (ground3d.ground --serve): loads at once, answers the questions put in jobs/, exits on stop."""
    return [python or sys.executable, "-m", "ground3d.ground", "--serve", str(jobs), "--parent", str(os.getpid())]


class Qwen:
    """Qwen-3D in its own process, so that all its memory is freed when it exits (before YOLOE loads)."""

    def __init__(self, folder, extra=(), python=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        for name in ("job.json", "result.json", "ready", "stop"):
            (self.folder / name).unlink(missing_ok=True)
        self.log = open(self.folder / "qwen.log", "w")
        self.proc = subprocess.Popen(qwen_command(self.folder, python) + list(extra), cwd=str(ROOT / "core"),
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.t0, self.n = time.time(), 0
        self.lock, self.preparing = threading.Lock(), None

    def prepare(self, scan):
        """Encode the scan now, in the background (the vision encoder, ~20 s on the Jetson), so the first question
        is answered in seconds; questions asked meanwhile wait for it."""
        def run():
            while not self.ready() and self.alive():
                time.sleep(0.3)
            self.preparing = time.time()
            res = self.ask(scan, None)
            self.preparing = None
            print("Qwen-3D has read the scan" + (f": {res['error']}" if res.get("error") else
                  f" ({res.get('seconds_after_job', '?')} s, RAM peak {res.get('ram_peak_gb', '?')} GB)"), flush=True)
        threading.Thread(target=run, daemon=True).start()

    def ready(self):
        return (self.folder / "ready").exists()

    def note(self, *starts, last=1):
        """The server's last log line(s) starting with one of `starts` (all lines if none), joined by " | "."""
        try:
            lines = (self.folder / "qwen.log").read_text(errors="ignore").splitlines()
        except OSError:
            return ""
        return " | ".join([ln for ln in lines if not starts or ln.startswith(starts)][-last:])

    def alive(self):
        return self.proc.poll() is None

    def ask(self, scan, text, timeout=300):
        """One question about the scan -> the server's result (answers[0] is the answer; "error" if it failed).
        text None: only encode the scan (prepare)."""
        with self.lock:                                 # one job at a time
            self.n += 1
            tmp = self.folder / "job.tmp"
            tmp.write_text(json.dumps(dict(scan=str(scan), queries=[] if text is None else [text], more=True)))
            tmp.rename(self.folder / "job.json")
            result, end = self.folder / "result.json", time.time() + timeout
            while not result.exists():
                if not self.alive():
                    return dict(error=f"Qwen-3D stopped (see {self.folder / 'qwen.log'})")
                if time.time() > end:
                    return dict(error=f"no answer in {timeout} s")
                time.sleep(0.1)
            res = json.loads(result.read_text())
            result.rename(self.folder / f"result_{self.n}.json")
            return res

    def stop(self, timeout=30):
        """Tell the server to exit, and wait: its memory must be free before YOLOE loads."""
        (self.folder / "stop").touch()
        try:
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.log.close()


def ram_in_use_gb():
    """The whole board's memory in use (GB): on the Jetson the CPU and GPU share it."""
    try:
        info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
        kb = lambda k: float(info[k].split()[0])
        return (kb("MemTotal") - kb("MemAvailable")) / 1024 ** 2
    except (OSError, KeyError, ValueError):
        return None


RAM_LIMIT_GB = 6.3                                      # the 8 GB Jetson's practical ceiling (its GPU allocator fails above)
OOM_SIGNS = ("NVML_SUCCESS", "out of memory", "OUT OF MEMORY", "NvMap", "Killed")


def leftover_qwen():
    """Qwen-3D servers already running: [(pid, its runner's pid, whether that runner is still an xray.live)]."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
            if not any(b"ground3d.ground" in c for c in cmd) or b"--serve" not in cmd:
                continue
            ppid = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            runner = (Path("/proc") / str(ppid) / "cmdline").read_bytes() if ppid > 1 else b""
            out.append((int(d.name), ppid, b"xray.live" in runner))
        except (OSError, ValueError, IndexError):
            pass
    return out


def free_for_qwen(log=print):
    """Before Qwen-3D loads: stop servers left over from earlier runs (their runner is gone, their memory isn't),
    and warn if memory is already short (it needs ~4-5 GB of the board's ~6.3)."""
    for pid, ppid, alive in leftover_qwen():
        if alive:
            log(f"warning: another run's Qwen-3D is running (pid {pid}, run pid {ppid}): two won't fit; stop that run")
            continue
        try:
            os.kill(pid, 9)
            log(f"stopped a Qwen-3D left over from an earlier run (pid {pid}): its memory is free again")
            time.sleep(2)
        except OSError:
            pass
    ram = ram_in_use_gb()
    if ram is not None and ram > 1.8:
        log(f"warning: {ram:.2f} GB of memory already in use before Qwen-3D loads (a fresh boot without the desktop: "
            "~1 GB). It may run out: free other programs (ps aux --sort=-rss | head), or reboot and "
            "'sudo systemctl isolate multi-user.target'")


def camera_block(T):
    return dict(position=tk.r4(T[:3, 3]), forward=tk.r4(T[:3, 2]), T_world_cam=np.round(T[:3].reshape(-1), 5).tolist())


def answer_points(folder, a, min_points=30):
    """An answer's cloud and which one: its dense cloud (<slug>_refined.ply), else Qwen-3D's own (<slug>.ply),
    else just its centre (both written by the server; a cloud of fewer than min_points doesn't count)."""
    from ground3d.show import read_ply, slug
    for name, which in ((f"{slug(a['query'])}_refined.ply", "refined"), (f"{slug(a['query'])}.ply", "raw")):
        ply = Path(folder) / name
        pts = read_ply(ply) if ply.exists() else ()
        if len(pts) >= min_points:
            return pts, which
    return np.asarray([a["center"]], float), "centre"


DRIVE_WORDS = {"drive", "fetch", "get it", "go get it", "go to it", "go to the target", "drive to the target"}
STOP_WORDS = {"stop", "halt", "stop driving"}
DRIVE_VERBS = re.compile(r"^(drive|fetch|get|find|touch|go)(?:\s+(?:over\s+)?(to|towards?|for))?\b\s*(.*)$")
NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}


def read_console(server):
    """Lines typed in the console become prompts, like POST /prompt (a "| tee" keeps the keyboard)."""
    if sys.stdin is not None and sys.stdin.isatty():
        def typed():
            for line in sys.stdin:
                server.prompts.put(line.strip())        # ("" is Enter: --wait starts the scan on it)
        threading.Thread(target=typed, daemon=True).start()


def wait_for_start(server, phone, qwen, yoloe, log=print):
    """--wait: the camera view says what is still loading (Qwen-3D, YOLOE); once everything is, Enter in the console
    (or POST /prompt "scan") starts the scan. yoloe: the Event set once YOLOE is loaded, or None."""
    said, shown = False, 0.0
    while True:
        loading = []
        if qwen is not None and not qwen.ready():
            if not qwen.alive():
                tail = qwen.note(last=8)
                hint = ("\nIt ran out of memory. Stop leftovers (pkill -f ground3d.ground; pkill -f xray.live), "
                        "check 'free -h' (used ~1 GB), or reboot without the desktop; then start again"
                        if any(s in tail for s in OOM_SIGNS) else "")
                raise SystemExit(f"Qwen-3D stopped while loading; the end of {qwen.folder / 'qwen.log'}:\n  "
                                 + tail.replace(" | ", "\n  ") + hint)
            loading.append(f"Qwen-3D {time.time() - qwen.t0:.0f} s")
        if yoloe is not None and not yoloe.is_set():
            loading.append("YOLOE")
        if not loading and not said:
            ram = ram_in_use_gb()
            log("READY: everything is loaded" + (f" (RAM in use {ram:.2f} GB)" if ram else "")
                + ". Press Enter (or POST /prompt 'scan') to start the scan")
            said = True
        while True:
            try:
                text = server.prompts.get_nowait().lower().strip(" .!")
            except queue.Empty:
                break
            if not loading and text in ("", "scan", "start", "go"):
                return
            if loading:
                log(f"still loading: {', '.join(loading)}")
        if phone is None:
            time.sleep(0.2)
            continue
        f = phone.read()
        if time.monotonic() - shown < 0.2:
            continue
        shown = time.monotonic()
        img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        tag(img, ("LOADING: " + ", ".join(loading)) if loading else "READY: press Enter to scan",
            (img.shape[1] // 2, 24), AMBER, 0.55)
        server.publish(img, {"phase": "loading" if loading else "ready", "loading": loading})


DRIVE_DELAY = 3.5                                       # s from Qwen-3D's answer to driving (say "stop" to cancel)
CANCEL_WORDS = {"stop", "no", "cancel", "wait", "don't"}


def ask_phase(qwen, scan, server, frames, log=print, hold=None):
    """Questions for Qwen-3D until "go": from POST /prompt (the dashboard, voice) or typed in the console. Meanwhile
    the camera view shows every answer's cloud (the newest one as the target), and the state stream lists the answers
    as objects, each with its cloud. "drive to <a description>" asks Qwen-3D too, and DRIVE_DELAY s after its answer
    (unless "stop") tracking starts with that answer as the target and the rover drives to it.
    hold: an Event the questions wait for (YOLOE loading beside Qwen-3D: one memory peak at a time).
    Returns (the answers [dict(query, score, band, seconds, point, points, ...)], the index of the one to drive to
    or None)."""
    answers, asked, busy, pending = [], queue.Queue(), [None], [None]
    requests = dict(sent=0, cancelled=0)               # drive requests: a "stop" cancels those still being answered

    def work():                                         # one question at a time, beside the camera view
        while True:
            job = asked.get()
            if job is None:
                return
            text, seq = job                             # seq: the drive request's number (0: a question)
            wants_drive = seq > requests["cancelled"]
            while not qwen.ready() and qwen.alive():
                time.sleep(0.2)
            if hold is not None:
                hold.wait(180)
            busy[0] = text
            res = qwen.ask(scan, text)
            a = (res.get("answers") or [{}])[0]
            if res.get("error") or not a.get("center"):
                log(f"Qwen-3D: '{text}': {res.get('error') or 'nothing found'}" + ("; not driving" if wants_drive else ""))
            else:
                ref = a.get("refined") or {}
                pts, which = answer_points(qwen.folder, a)
                good = which == "refined"                   # (the dense cloud's centre and size, if it is one)
                point = ref.get("center") if good else a["center"]
                answers.append(dict(a, point=point, points=pts, which=which, size=ref.get("size") if good else a.get("size"),
                                    ram_gb=res.get("ram_gb"), arrived=time.monotonic(),
                                    candidates=[dict(c, points=np.asarray(c["points"], float)) for c in a.get("candidates") or []],
                                    heat=np.asarray(a.get("heat") or np.zeros((0, 4)), float).reshape(-1, 4)))
                if seq > requests["cancelled"]:             # (not cancelled meanwhile) the countdown to the handoff
                    pending[0] = (len(answers) - 1, time.monotonic() + DRIVE_DELAY)
                why = ref.get("error") or f"{ref.get('n_points', 0)} points near the answer"
                weighed = ", ".join(f"{c['score']:.2f}" for c in a.get("candidates") or [])
                log(f"Qwen-3D: '{text}' -> score {a['score']:.2f} ({a['band']}) at {tk.r4(point)}, {a['seconds']:.1f} s; "
                    f"{which} cloud, {len(pts)} points" + (f" (dense cloud: {why})" if which != "refined" and why else "")
                    + (f"; RAM {res['ram_gb']} GB" if res.get("ram_gb") else "")
                    + (f", peak {res['ram_peak_gb']} GB" if res.get("ram_peak_gb") else "")
                    + (f"; also weighed {weighed}" if weighed else ""))
            busy[0] = None

    def countdown(img):
        if pending[0] is not None:
            a, left = answers[pending[0][0]], max(0.0, pending[0][1] - time.monotonic())
            tag(img, f"DRIVING TO '{a['query']}' IN {left:.1f} s  (say stop to cancel)",
                (img.shape[1] // 2, img.shape[0] - 120), AMBER, 0.5)
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    log('ASK: type a question (or POST /prompt {"text": ...}); "go" starts tracking; '
        '"drive to <description>" asks, then hands over and drives')
    go, last, said, shown, drive = False, None, None, 0.0, None
    for f in frames:
        if said is None and qwen.ready():               # the console says so too, with the load time and memory
            said = True
            log(f"Qwen-3D ready, {time.time() - qwen.t0:.0f} s after the start ({qwen.note('loaded in', 'READY', last=2)})")
        elif said is None and not qwen.alive():
            said = False
            log(f"Qwen-3D stopped before it was ready; the end of {qwen.folder / 'qwen.log'}:\n  "
                + qwen.note(last=8).replace(" | ", "\n  "))
        while not go:
            try:
                text = server.prompts.get_nowait()
            except queue.Empty:
                break
            word = text.lower().strip(" .!?")
            is_drive, name = drive_command(text)
            if not word:
                continue
            if word in ("go", "start", "track"):
                go = True
            elif word in CANCEL_WORDS:
                if pending[0] is not None or requests["sent"] > requests["cancelled"]:
                    pending[0], requests["cancelled"] = None, requests["sent"]
                    log("drive cancelled: ask again, or say go")
            elif is_drive and name is None:             # "drive": to the newest answer
                if answers:
                    pending[0] = (len(answers) - 1, time.monotonic() + DRIVE_DELAY)
                else:
                    log("nothing to drive to yet: say what, e.g. 'drive to the cup next to the green ball'")
            elif is_drive and re.search(r"\d", name):
                log(f"'{name}': cup numbers come with tracking. Describe it (e.g. 'drive to the cup on the left'), "
                    "or say go, then 'drive to cup N'")
            elif is_drive:
                log(f"Qwen-3D: finding '{name}' to drive to")
                requests["sent"] += 1
                asked.put((name, requests["sent"]))
            else:
                asked.put((text, 0))
        if pending[0] is not None and time.monotonic() >= pending[0][1]:
            drive, go = pending[0][0], True             # the handoff: Qwen-3D's answer becomes the target
            log(f"driving to Qwen-3D's answer to '{answers[drive]['query']}'")
        if go:
            if last is not None:                        # the view says what happens next while YOLOE loads
                img, msg = last
                tag(img, "go: Qwen-3D exits, YOLOE tracking starts", (img.shape[1] // 2, img.shape[0] - 24), AMBER, 0.6)
                server.publish(img, dict(msg, phase="starting", qwen=dict(msg["qwen"], status="done")))
            break
        if time.monotonic() - shown < 1 / 15:           # the phone sends 60 a second: 15 leave the CPU to Qwen-3D
            continue
        shown = time.monotonic()
        img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
        objects = []
        for i, a in enumerate(list(answers)):
            newest = i == len(answers) - 1
            info = {"label_xy": None}
            pts = a["points"]
            if a["which"] == "centre":                  # no cloud: a ring where the answer is
                u, v, z = render.project(np.asarray([a["point"]], float), f.K, f.T)
                if z[0] > 0.05:
                    r = int(max(10, 0.05 * f.K[0, 0] / z[0]))
                    cv2.circle(img, (int(u[0]), int(v[0])), r, AMBER if newest else TEXT, 2, cv2.LINE_AA)
                    info = {"label_xy": (int(u[0]), int(v[0]) - r)}
            elif len(pts):
                age = time.monotonic() - a["arrived"]
                if newest and age < SWEEP_S and len(a["heat"]):   # what it weighed: the sweep, then the answer
                    draw_heat(img, a["heat"], f.K, f.T, fade=min(1.0, 2 * (1 - age / SWEEP_S)))
                if newest:
                    draw_candidates(img, a["candidates"], f.K, f.T)
                if not newest or age > 0.4 * SWEEP_S:
                    info = render.draw_cloud(img, pts, f.K, f.T, f.depth, centre=a["point"], target=newest, order_seed=200 + i)
            if info["label_xy"]:
                tag(img, f"'{a['query']}' {a['score']:.2f}", (info["label_xy"][0], info["label_xy"][1] - 8),
                    AMBER if newest else TEXT)
            few = pts[np.linspace(0, len(pts) - 1, min(len(pts), 800)).astype(int)] if len(pts) else pts
            objects.append(dict(id=f"q{i + 1}", label="Qwen-3D answer", query=a["query"], target=newest, state="VISIBLE",
                                position=tk.r4(a["point"]), size=a.get("size"), score=a["score"], band=a["band"],
                                seconds=a["seconds"], point_cloud=dict(frame="world", points=np.round(few, 3).tolist()),
                                candidates=[dict(score=c["score"], position=c["center"], size=c.get("size"),
                                                 point_cloud=dict(frame="world", points=np.round(thin(c["points"], 120), 3).tolist()))
                                            for c in a["candidates"]]))
        if not qwen.ready():
            status = f"loading Qwen-3D ... {time.time() - qwen.t0:.0f} s" if qwen.alive() else "Qwen-3D failed (see qwen.log)"
        elif hold is not None and not hold.is_set():
            status = "Qwen-3D ready; loading YOLOE beside it ..."
        elif qwen.preparing and not busy[0]:
            status = f"Qwen-3D reading the scan ... {time.time() - qwen.preparing:.0f} s (ask away: it answers next)"
        elif busy[0]:
            status = f"Qwen-3D thinking: '{busy[0]}'"
        else:
            status = 'Qwen-3D ready: ask a question, or say "go"'
        tag(img, status, (img.shape[1] // 2, 22), AMBER, 0.5)
        if answers:
            decision_card(img, answers[-1])
        countdown(img)
        sweep = None
        if answers and time.monotonic() - answers[-1]["arrived"] < SWEEP_S and len(answers[-1]["heat"]):
            a = answers[-1]
            sweep = dict(answer=f"q{len(answers)}", age_s=round(time.monotonic() - a["arrived"], 2), duration_s=SWEEP_S,
                         heat=np.round(thin(a["heat"], 800), 3).tolist())
        msg = dict(t=round(float(f.t), 3), frame=int(f.index), phase="ask", target_id=None,
                   camera=camera_block(f.T), objects=objects,
                   qwen=dict(status=status, ready=qwen.ready(), thinking=busy[0], sweep=sweep,
                             driving_in=None if pending[0] is None else round(max(0.0, pending[0][1] - time.monotonic()), 1),
                             answers=[{k: a.get(k) for k in ("query", "target", "score", "band", "seconds", "point", "top5",
                                                             "llm_seconds", "tokens", "gpu_peak_gb", "ram_gb")} for a in answers]))
        server.publish(img, msg)
        last = img, msg
    asked.put(None)
    worker.join(timeout=300)                            # a question still being answered finishes first
    return answers, drive


def live_cloud(f, n=1500, max_depth=3.0):
    """About n of the frame's confident depth points (world, mm-rounded), for the viewer's live 3D view."""
    ok = (f.conf == 2) & (f.depth > 0) & (f.depth < max_depth)
    stride = max(1, int(math.sqrt(ok.sum() / max(n, 1))))
    p = fr.backproject(f.depth, f.K, f.T, ok, stride)
    if len(p) > n:
        p = p[np.linspace(0, len(p) - 1, n).astype(int)]
    return np.round(p, 3).tolist()


class Recorder:
    """Raw frames -> a track folder (the scan folder layout, in the scan's world) that xray.track / --replay read."""

    def __init__(self, folder):
        self.folder, self.n, self.times, self.size = Path(folder), 0, [], None

    def add(self, f):
        """xray.frames.write_frame's layout, written faster (light PNG compression): it runs beside a live loop."""
        k = self.n
        if k == 0:
            if self.folder.exists():                    # a reused --out: this run's frames only
                shutil.rmtree(self.folder)
            for d in ("color", "depth", "conf", "intrinsic", "pose"):
                (self.folder / d).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.folder / "color" / f"{k}.jpg"), cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        png = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        cv2.imwrite(str(self.folder / "depth" / f"{k}.png"), np.clip(np.round(f.depth * 1000), 0, 65535).astype(np.uint16), png)
        cv2.imwrite(str(self.folder / "conf" / f"{k}.png"), f.conf.astype(np.uint8), png)
        np.savetxt(self.folder / "intrinsic" / f"{k}.txt", f.K, fmt="%.6f")
        np.savetxt(self.folder / "pose" / f"{k}.txt", f.T, fmt="%.8f")
        self.times.append(round(float(f.t), 4))
        self.size = list(f.size)
        self.n += 1

    def close(self):
        if self.n:
            (self.folder / "meta.json").write_text(json.dumps(dict(
                source="xray.live --record", num_frames=self.n, frame_times_s=self.times, image_size_wh=self.size,
                keyframes=False, pose="camera-to-world, OpenCV camera (x right, y down, z forward), the scan's world")))


def main(argv=None):
    ap = argparse.ArgumentParser(description="The live x-ray (see the top of this file)")
    ap.add_argument("--scan", help="a scan folder of this phone session (else --scan-seconds scans first)")
    ap.add_argument("--scan-seconds", type=float, default=20, help="scan for this long first (without --scan)")
    ap.add_argument("--rover", nargs="?", const="auto", metavar="PORT",
                    help="drive the rover (rover/rover.ino on an Arduino): its serial port, auto (default) or dry (log only). "
                         "While tracking, \"drive\" (POST /prompt or typed) drives it to the target, \"stop\" stops it")
    ap.add_argument("--rover-scan", action="store_true", help="the rover turns and rolls through the scan by itself")
    ap.add_argument("--rover-power", type=int, nargs=3, default=[115, 120, 100], metavar=("TURN", "DRIVE", "MIN"),
                    help="starting wheel power (of 255) to turn in place and to drive, and the least it ever uses; from there it finds by itself what starts the wheels and keeps a slow, steady pace")
    ap.add_argument("--rover-size", type=float, nargs=2, default=[0.11, 0.15], metavar=("FRONT", "WIDTH"),
                    help="metres from the phone to the rover's front edge, and the rover's width (default 0.11 0.15)")
    ap.add_argument("--rover-forward-offset", type=float, metavar="DEG",
                    help="the rover's forward direction minus the phone's, in degrees (default: measured during "
                         "--rover-scan's forward roll, else 0)")
    ap.add_argument("--min-keyframes", type=int, default=20,
                    help="the scan goes on (up to twice --scan-seconds) until it has this many views")
    ap.add_argument("--replay", help="a recorded track folder instead of the phone (same world as --scan)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (0: as fast as possible)")
    ap.add_argument("--start", type=float, help="replay from this recording time (s)")
    ap.add_argument("--loop", action="store_true", help="replay over and over (a fresh game each time), until Ctrl+C")
    ap.add_argument("--prompt", default=tk.rf.DEFAULT_PROMPT, help=f"detector labels (default \"{tk.rf.DEFAULT_PROMPT}\")")
    ap.add_argument("--ball", action="store_true", help="the shell game: balls under cups")
    ap.add_argument("--ask", action="store_true", help="Qwen-3D: load it at the start, ask questions after the scan, "
                    "'go' to track (the last answer is the target)")
    ap.add_argument("--qwen-args", default="", help="extra arguments for the Qwen-3D server, e.g. \"--models ~/models/qwen3d\"")
    ap.add_argument("--qwen-python", help="the Python (venv) with torch and transformers for Qwen-3D (default: this one)")
    ap.add_argument("--wait", action="store_true",
                    help="load the models first (Qwen-3D with --ask; YOLOE too where memory allows), then start the "
                         "scan on Enter (or POST /prompt 'scan'): for a demo, everything ready before the rover moves")
    ap.add_argument("--qwen-stay", action="store_true",
                    help="with --ask: load YOLOE beside Qwen-3D right after the scan (go starts at once) and keep "
                         "Qwen-3D loaded through tracking, so 'drive to <description>' works then too "
                         "(it needs --qwen-args \"--bits 4\" to fit beside YOLOE on the 8 GB Jetson; its answers are about "
                         "the scan, and go to the object that stood there, wherever it is now)")
    ap.add_argument("--results", help="target = Qwen-3D's first answer (a python -m ground3d.ground --out folder)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default=f"runs/live_{time.strftime('%H%M%S')}")
    ap.add_argument("--video", action="store_true", help="also write the x-ray view to <out>/xray.mp4")
    ap.add_argument("--record", action="store_true", help="also save the raw frames to <out>/track (replayable)")
    ap.add_argument("--live-cloud", type=int, default=1500, metavar="N",
                    help="points of each depth frame in the state messages, for the viewer's 3D view (0: none)")
    args = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(line_buffering=True)     # each line at once, in order, through a "| tee"
    except (AttributeError, ValueError):
        pass
    from xray import shell
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    metrics = None
    try:
        sys.path.insert(0, str(ROOT))
        from core.jetson.metrics_server import Metrics
        metrics = Metrics()
    except Exception as e:                              # metrics are optional
        print(f"(no /metrics: {e})")
    server = Server(args.host, args.port, metrics)
    print(f"serving http://{args.host}:{args.port}/  (/state, /state.json, /frame.jpg, /scene, /metrics)", flush=True)
    if args.replay and not args.scan:
        ap.error("--replay needs the --scan of the same recording")
    if args.loop and not args.replay:
        ap.error("--loop is for --replay")
    if args.rover_scan and (not args.rover or args.scan):
        ap.error("--rover-scan needs --rover, and a scan made now (no --scan)")
    rover = rv.Rover(args.rover) if args.rover else None
    nav = rv.Navigator(*args.rover_power, offset=math.radians(args.rover_forward_offset or 0.0),
                       front=args.rover_size[0], width=args.rover_size[1])
    read_console(server)
    labels = detect.parse_prompt(args.prompt)
    if args.ask:
        free_for_qwen()
    qwen = Qwen(out / "qwen", shlex.split(args.qwen_args), args.qwen_python and str(Path(args.qwen_python).expanduser())) \
        if args.ask else None                           # loads while the scan runs
    if qwen is not None:
        print(f"Qwen-3D loading in its own process (log: {qwen.folder / 'qwen.log'})", flush=True)
    cached = detect.load_detections(detect.cache_path(args.replay, labels)) if args.replay else None
    pre = None                                          # YOLOE's engine, loaded ahead where memory allows: without
    if qwen is None or args.qwen_stay:                  # Qwen-3D, or beside it (after its own load: one peak at a time)
        pre = {"done": threading.Event()}
        need = not (cached is not None and args.scan and detect.load_detections(detect.cache_path(args.scan, labels)))

        def preload_yoloe():
            try:
                while qwen is not None and not qwen.ready() and qwen.alive():
                    time.sleep(0.5)
                pre["detector"] = detect.Detector(labels) if need else None
            except Exception as e:
                pre["error"] = e
            finally:
                pre["done"].set()
        threading.Thread(target=preload_yoloe, daemon=True).start()
    phone = None
    if not args.replay:
        from .phone import Phone
        phone = Phone()
    routine = rv.ScanRoutine(*args.rover_power) \
        if args.rover_scan else None
    if args.wait:
        try:
            wait_for_start(server, phone, qwen, pre["done"] if pre is not None else None)
        except BaseException:
            if rover is not None:
                rover.close()
            if qwen is not None:
                qwen.stop()
            raise
    try:
        scan = Path(args.scan) if args.scan else scan_with_phone(phone, args.scan_seconds, out, server, args.min_keyframes,
                                                                 rover, routine)
    except BaseException:
        if rover is not None:
            rover.close()
        raise
    if routine is not None and args.rover_forward_offset is None:
        if routine.forward_offset is not None:          # the phone's mount angle, from the scan's forward roll
            nav.offset = routine.forward_offset
            print(f"rover: the phone faces {math.degrees(nav.offset):+.1f} deg off the rover's forward direction", flush=True)
        else:
            print("rover: no forward roll measured; assuming the phone faces straight ahead", flush=True)
    if qwen is not None and not args.scan:              # Qwen-3D reads the new scan now: the first answer comes quick
        qwen.prepare(scan)
    source = fr.FolderSource(args.replay) if args.replay else phone
    to_world = None if args.replay else world_from_phone(scan)

    def in_world(frames):
        for f in frames:
            if to_world is not None:
                f.T = to_world @ f.T
            yield f
    answers, drive_answer = [], None
    early = None
    if qwen is not None and args.qwen_stay:            # both stay loaded: YOLOE loads now, beside Qwen-3D, so "go" is instant
        early = {}

        def load_yoloe():                              # (preloaded after Qwen-3D's own load: one memory peak at a time)
            try:
                pre["done"].wait()
                if "error" in pre:
                    raise pre["error"]
                scan_cached = detect.load_detections(detect.cache_path(scan, labels)) is not None
                early["detector"] = None if cached is not None and scan_cached else \
                    pre["detector"] or detect.Detector(labels)
                if early["detector"] is not None:
                    detect_keyframes(scan, early["detector"], labels)
                early["scene"] = tk.scan_objects(scan, args.prompt, use_detector=False)
                ram = ram_in_use_gb()
                print("YOLOE ready beside Qwen-3D" + (f", RAM in use {ram:.2f} GB" if ram else "")
                      + (f" (near the {RAM_LIMIT_GB} GB limit: try --qwen-args \"--bits 4\")" if ram and ram > RAM_LIMIT_GB - 0.5 else ""),
                      flush=True)
            except Exception as e:
                early["error"] = e
            finally:
                early["done"].set()
        early["done"] = threading.Event()
        early["thread"] = threading.Thread(target=load_yoloe, daemon=True)
        early["thread"].start()
    if qwen is not None:                                # understand, then free the memory for YOLOE
        def scan_scene():                               # the scan's cloud for the viewer while questions are asked
            try:                                        # (no objects yet: tracking replaces it with the twins)
                kf = tk.rf.load_keyframes(scan)
                z = tk.rf.table_height(kf)
                server.set_scene(scene_json(scan, (), table_z=z if z is not None else tk.rf.table_height(kf, lo=-0.15, hi=0.3)))
            except Exception as e:
                print(f"(no /scene while asking: {e})", flush=True)
        threading.Thread(target=scan_scene, daemon=True).start()
        try:
            answers, drive_answer = ask_phase(qwen, scan, server, in_world(frames_from(source, args.start, args.speed)),
                                              hold=early["done"] if early is not None else None)
        except KeyboardInterrupt:
            answers = None
        except RuntimeError as e:                       # the phone stopped
            print(f"stopped: {e}. {PHONE_TIP}", flush=True)
            answers = None
        finally:
            if not args.qwen_stay or answers is None:
                qwen.stop()
        if answers is None:                             # Ctrl+C before "go"
            if rover is not None:
                rover.close()
            server.close()
            if metrics is not None:
                metrics.stop()
            if phone is not None:
                os._exit(0)
            return 0
        print(f"Qwen-3D done: {len(answers)} answers; " + ("it stays loaded (--qwen-stay)" if args.qwen_stay else
              "its memory is free; loading YOLOE"), flush=True)
    if early is not None:
        early["thread"].join()
        if "error" in early:
            raise early["error"]
        detector, (objects, table) = early["detector"], early["scene"]
        ram = ram_in_use_gb()
        if ram is not None and ram > RAM_LIMIT_GB - 0.7 and qwen.alive():  # no room for a question's peak (~0.7 GB)
            qwen.stop()
            print(f"RAM in use {ram:.2f} GB: no room to keep Qwen-3D through tracking (a question needs ~0.7 GB more); "
                  "freed it", flush=True)
    else:
        scan_cached = detect.load_detections(detect.cache_path(scan, labels)) is not None
        if pre is not None:                             # (loaded ahead)
            pre["done"].wait()
            if "error" in pre:
                raise pre["error"]
        preloaded = pre.get("detector") if pre is not None else None
        detector = None if cached is not None and scan_cached else preloaded or detect.Detector(labels)
        if detector is not None:
            detect_keyframes(scan, detector, labels)
        objects, table = tk.scan_objects(scan, args.prompt, use_detector=False)
    print(f"scene: {len(objects)} objects, table at {table if table is None else round(table, 3)} m: "
          + ", ".join(f"{o['label']} at {tk.r4(o['position'])}" for o in objects), flush=True)
    pick = [answers[drive_answer]] if drive_answer is not None else \
        ([a for a in answers if a.get("band") == "high"] or answers)[-1:]   # the one to drive to, or the last confident
    point, query = (pick[0]["point"], pick[0]["query"]) if pick else \
        qwen_target(args.results) if args.results else (None, None)

    def new_game():                                     # the tracker (and the game) as the scan left them
        game = None
        if args.ball:
            cups, balls = shell.split(objects)
            tracker = tk.Tracker(cups, table, labels={c["label"] for c in cups})
            game = shell.ShellGame(tracker, balls, table, scan)
        else:
            tracker = tk.Tracker(objects, table)
        if point is not None:                           # the object the answer picked, tracked from now on
            tr = tk.nearest(tracker.tracks, point)
            ball = min(game.balls, key=lambda b: tk.distance(b.track.position, point)) if game is not None and game.balls else None
            if ball is not None and (tr is None or tk.distance(ball.track.position, point) < tk.distance(tr.position, point)):
                game.target, picked = ball, ball.id         # a ball: its cup is the target once it is hidden
            else:
                tracker.target_id = None if tr is None else tr.id
                if game is not None:
                    game.chosen = tracker.target_id
                picked = "none" if tr is None else f"{tr.label} {tr.id}"
            print(f"target: '{query}' -> {picked}", flush=True)
        return tracker, game
    tracker, game = new_game()
    late = LateQwen(qwen, scan) if args.qwen_stay and qwen is not None and qwen.alive() else None
    if rover is not None and drive_answer is not None:  # "drive to <description>": Qwen-3D picked it, now go there
        tr = drive_target(tracker, game)
        if tr is not None:
            cup = any(t is tr for t in tracker.tracks)
            nav.start(tr.id, touch=cup)
            print(f"rover: driving to {object_name(tr, game) if cup else tr.id}, Qwen-3D's answer", flush=True)
    def make_scene():                                   # beside the tracking: it takes seconds, and /scene can wait
        try:
            info = {}
            server.set_scene(scene_json(scan, list(tracker.tracks), cut=[b.track for b in game.balls] if game else (),
                                        table_z=table, info=info))
            if "surface_spacing" in info and server.healer is not None:
                server.healer.set_cell(info["surface_spacing"])
        except Exception as e:
            print(f"(no /scene: {e})", flush=True)
    server.healer = Healer(list(tracker.tracks) + ([b.track for b in game.balls] if game else []), table)
    threading.Thread(target=make_scene, daemon=True).start()
    floor = 0.0                                         # the scan's world has the floor at z = 0
    frames_q, out_q, stop = queue.Queue(maxsize=2), queue.Queue(maxsize=1), threading.Event()

    def offer(q, item, newest=False):
        """Put item on q; with newest, drop an item not yet taken instead of waiting (live: never fall behind)."""
        while not stop.is_set():
            if newest:
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put(item, timeout=0.5)
                return
            except queue.Full:
                pass

    def produce():                                      # thread 1: frames and their detections
        try:
            while True:
                for f in frames_from(source, args.start, args.speed):
                    if stop.is_set():
                        break
                    if to_world is not None:
                        f.T = to_world @ f.T
                    dets = cached.get(source.ids[f.index], []) if cached is not None else detector(f.rgb)
                    offer(frames_q, (f, dets), newest=phone is not None)
                if not args.loop or stop.is_set():
                    break
                offer(frames_q, "again")                # --loop: the recording from the start, a fresh game
        except Exception as e:                          # handed to the main thread
            offer(frames_q, e)
        offer(frames_q, None)

    record = Recorder(out / "track") if args.record else None
    files_q, dropped = queue.Queue(maxsize=64), [0]

    def publish():                                      # thread 3: JPEG and the stream, always the newest frame
        while True:
            item = out_q.get()
            if item is None:
                break
            server.publish(*item)

    def write():                                        # thread 4: the log, the video, the recording (may lag)
        video = None
        with open(out / "states.jsonl", "w") as log:
            while True:
                item = files_q.get()
                if item is None:
                    break
                img, msg, f = item
                log.write(json.dumps({k: v for k, v in msg.items() if k != "scene"}, separators=(",", ":")) + "\n")
                if args.video:
                    if video is None:
                        video = cv2.VideoWriter(str(out / "xray.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                                (img.shape[1], img.shape[0]))
                    video.write(img)
                if record is not None:
                    record.add(f)
        if video is not None:
            video.release()

    threads = [threading.Thread(target=produce, daemon=True), threading.Thread(target=publish, daemon=True),
               threading.Thread(target=write, daemon=True)]
    for th in threads:
        th.start()
    verdict, verdict_t, seen_verdicts = None, 0.0, set()
    fps, last = None, None
    print("tracking (Ctrl+C stops)", flush=True)
    try:                                                # main thread: tracking, the game, the view
        while True:
            item = frames_q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if isinstance(item, str):                   # "again"
                tracker, game = new_game()
                server.healer = Healer(list(tracker.tracks) + ([b.track for b in game.balls] if game else []), table)
                verdict, last = None, None
                continue
            f, dets = item
            tracker.update(f, dets)
            if game is not None:
                game.update(f, dets)
            now = time.time()
            if last is not None:
                fps = 1 / max(1e-3, now - last) if fps is None else 0.9 * fps + 0.1 / max(1e-3, now - last)
            last = now
            if server.healer is not None and f.index % 4 == 0:     # the bare surface where objects stood
                standing = [(tr.position[:2], tr.radius) for tr in tracker.tracks]
                if game is not None:
                    standing += [(b.track.position[:2], 0.035) for b in game.balls if b.inside is None]
                server.healer.add(f, standing)
            msg = tracker.message(f, fps)
            msg["heal_version"] = server.healer.version if server.healer is not None else 0
            if rover is not None:                       # "drive" / "stop" (POST /prompt or typed), then steer
                drive_step(server, rover, nav, tracker, game, f, late=late)
                msg["drive"] = nav.status()
            if game is not None:
                msg = game.message(msg)
                for b in game.balls:
                    if b.verdict and id(b.verdict) not in seen_verdicts:
                        seen_verdicts.add(id(b.verdict))
                        v = b.verdict
                        verdict = (f"REVEAL {b.id}: cup {v['actual']}, we said cup {v['believed']}: "
                                   f"{'RIGHT' if v['right'] else 'WRONG'}")
                        verdict_t = f.t
            if args.live_cloud:
                msg["scene"] = dict(floor_z=floor, point_cloud=dict(frame="world", points=live_cloud(f, args.live_cloud)))
            img = draw_view(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), f, tracker, game, fps,
                            verdict if verdict and f.t - verdict_t < 5 else None)
            if late is not None and late.busy:
                tag(img, f"Qwen-3D thinking: '{late.busy}'", (img.shape[1] // 2, 22), AMBER, 0.5)
            if rover is not None and nav.state not in ("idle",):
                d = nav.info.get("gap_m")
                tag(img, f"ROVER: {nav.state} {'to ' + str(nav.target_id) if nav.target_id is not None else ''}"
                    + (f", {d:.2f} m" if d is not None and nav.state in ("turning", "driving", "detour") else ""),
                    (img.shape[1] // 2, img.shape[0] - 40), AMBER, 0.55)
            offer(out_q, (img, msg), newest=True)        # the stream never waits behind the files
            if phone is None:
                files_q.put((img, msg, f))              # a replay keeps every frame
            else:
                try:
                    files_q.put_nowait((img, msg, f))
                except queue.Full:                      # the files fell behind: skip this frame there
                    dropped[0] += 1
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:                           # the phone stopped
        print(f"stopped: {e}. {PHONE_TIP}", flush=True)
    finally:                                            # every step guarded: the files must close, the process exit
        if rover is not None:
            rover.close()                               # first: the motors stop
        if late is not None:
            late.qwen.stop()
        for q, th in ((out_q, threads[1]), (files_q, threads[2])):
            try:
                q.put(None, timeout=5)
            except queue.Full:
                pass
            th.join(timeout=60)                         # the log, video and recording finish
        stop.set()
        try:
            if record is not None:
                record.close()
            if game is not None and game.events:
                print("events:\n" + "\n".join(f"  {t:6.2f} s  {e}" for t, e in game.events))
            print(f"-> {out}/states.jsonl" + (" and xray.mp4" if args.video else "")
                  + (f", {record.n} frames in track/" if record is not None else "")
                  + (f" ({dropped[0]} frames not saved: the files fell behind)" if dropped[0] else ""), flush=True)
        except (BrokenPipeError, OSError):              # the console went away (Ctrl+C also stops a | tee)
            pass
        finally:
            try:
                server.close()
                if metrics is not None:
                    metrics.stop()
            finally:
                if phone is not None:
                    os._exit(0)     # record3d's teardown at a normal exit hangs or crashes (as in xray.scan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
