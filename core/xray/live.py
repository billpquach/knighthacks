"""The live x-ray: the phone's frames -> detector -> tracker (-> shell game) -> the x-ray view and the state stream,
served to the glasses, the rover and the viewer.

    python -m xray.live --scan-seconds 20 [--ball] [--port 8080]                  # scan the table, then track
    python -m xray.live --scan <scan folder> [--ball | --results <ground3d output>]  # an earlier scan, same session
    python -m xray.live --scan <scan> --replay <track folder> [--ball] [--speed 1]   # a recording instead of the phone

Endpoints, CORS on all (the same as replay_server.py, so the viewer and every client work unchanged):
    /state          server-sent events: the newest state message, up to 10 a second
    /state.json     the newest state message
    /frame.jpg      the newest x-ray image (JPEG); ?after=SEQ waits up to 2 s for a newer one; X-Seq, X-Age-Ms headers
    /scene          the scan: table, coloured cloud, and each object's twin (for the viewer's 3D view)
    /metrics        the Jetson's resource use (core/jetson)
    /viewer-info    {"mode": "live"}
    /               the viewer, once built (viewer/dist)
Until tracking starts the messages are {"phase": "scan"} with no objects. Every message also goes to
<out>/states.jsonl, and with --video the x-ray view to <out>/xray.mp4.

The scan and the tracking must share the phone's ARKit session (don't restart Record3D in between): live frames are
put in the scan's world with its meta.json world_from_arkit. --results picks the target from Qwen-3D's first answer
(python -m ground3d.ground --out); --ball plays the shell game (the yellow ball's cup is the target).

The x-ray view: every object's digital twin, moved to where the tracker has it, drawn by xray.render (the target
brighter; a hidden object glowing through whatever covers it); a ball under a cup drawn as a disc of its colour
through the cup; a tag on each object; status lines at the top, and the verdict when a ball is revealed.
"""
import argparse
import json
import mimetypes
import os
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
from . import track as tk

ROOT = Path(__file__).resolve().parents[2]
VIEWER_DIST = ROOT / "viewer" / "dist"
BALL_BGR = {"red": (40, 40, 230), "yellow": (40, 215, 250), "green": (60, 200, 60), "blue": (230, 130, 40),
            "purple": (200, 60, 170), None: (230, 230, 230)}
TEXT = (235, 245, 250)
CYAN = (255, 220, 0)
AMBER = (40, 190, 255)


# ---------------------------------------------------------------------------------------- serving
class Server:
    """The newest state message and x-ray JPEG, served over HTTP from a background thread."""

    def __init__(self, host="0.0.0.0", port=8080, metrics=None):
        self.cond = threading.Condition()
        self.state, self.jpg, self.seq, self.stamp = {"phase": "scan"}, None, 0, time.time()
        self.scene, self.metrics = None, metrics
        self.httpd = ThreadingHTTPServer((host, port), handler(self))
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def publish(self, img=None, state=None, quality=80):
        jpg = None
        if img is not None:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            jpg = buf.tobytes() if ok else None
        with self.cond:
            if jpg is not None:
                self.jpg, self.seq, self.stamp = jpg, self.seq + 1, time.time()
            if state is not None:
                self.state = state
            self.cond.notify_all()

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
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
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

        def do_GET(self):
            url = urlsplit(self.path)
            path = unquote(url.path)
            if path == "/state":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                last = None
                try:
                    while True:
                        with server.cond:
                            server.cond.wait_for(lambda: server.state is not last, timeout=1.0)
                            state = last = server.state
                        self.wfile.write(("data: " + json.dumps(state, separators=(",", ":")) + "\n\n").encode())
                        self.wfile.flush()
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    return
            elif path == "/state.json":
                with server.cond:
                    state = server.state
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
            elif path == "/scene":
                if server.scene is None:
                    self.send_error(404, "No scene yet")
                    return
                self.body(server.scene, "application/json", cache="no-cache")
            elif path == "/metrics":
                latest = server.metrics.latest() if server.metrics else dict(
                    version=1, device=None, memory_kind="shared", available=False, reason="not_jetson", metrics={},
                    age_ms=None, sample_id=None)
                self.body(json.dumps(latest).encode(), "application/json")
            elif path == "/viewer-info":
                self.body(json.dumps({"mode": "live"}).encode(), "application/json")
            else:
                file = (VIEWER_DIST / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
                if not file.is_relative_to(VIEWER_DIST.resolve()) or not file.is_file():
                    self.send_error(404, "Not found (the viewer: cd viewer && npm ci && npm run build)")
                    return
                self.body(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream", "no-cache")
    return Handler


def scene_json(scan, tracks):
    """The /scene document: the scan (viewer/tools/scene_export.py) with each object's twin as offsets from it."""
    sys.path.insert(0, str(ROOT / "viewer" / "tools"))
    from scene_export import export_scene
    scene = export_scene(scan)
    scene["objects"] = []
    for tr in tracks:
        pts = np.zeros((0, 3)) if tr.points is None else \
            tr.points[np.linspace(0, len(tr.points) - 1, min(len(tr.points), 2000)).astype(int)] - tr.origin
        scene["objects"].append(dict(id=tr.id, label=tr.label, position=tk.r4(tr.position), points=np.round(pts, 4).tolist()))
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
            glow = np.zeros_like(img)
            cv2.circle(glow, c, r + 4, BALL_BGR.get(b.colour, BALL_BGR[None]), -1, cv2.LINE_AA)
            img[:] = cv2.add(img, cv2.GaussianBlur(glow, (0, 0), 6))
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
def scan_with_phone(phone, seconds, out, server, log=print):
    """xray.scan's capture, in this process (the phone stays connected for tracking)."""
    from . import scan as sc
    shown = [0.0]

    def progress(f, n, left):
        if time.monotonic() - shown[0] > 0.2:
            shown[0] = time.monotonic()
            img = cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR)
            tag(img, f"SCANNING: walk slowly around the table  {max(0.0, left):4.1f} s, {n} keyframes", (img.shape[1] // 2, 24))
            server.publish(img, {"phase": "scan"})
    keyframes, n_read, blurry = sc.capture(phone, seconds, sc.KeyframePicker(), progress)
    if len(keyframes) < 8:
        raise SystemExit(f"only {len(keyframes)} keyframes: move around more (10 cm or 15 deg between keyframes)")
    shift, world = sc.fit_world(keyframes)
    sc.apply(shift, keyframes)
    folder = Path(out) / "scan"
    sc.write_scan(folder, keyframes, shift, source="record3d", stream=phone.info, world=world)
    log(f"scan: {len(keyframes)} keyframes of {n_read} frames -> {folder}")
    return folder


def detect_keyframes(scan, detector, labels, log=print):
    """The scan's detections with the detector already loaded (one model in memory), cached for refine."""
    path = detect.cache_path(scan, labels)
    if detect.load_detections(path) is None:
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


def main(argv=None):
    ap = argparse.ArgumentParser(description="The live x-ray (see the top of this file)")
    ap.add_argument("--scan", help="a scan folder of this phone session (else --scan-seconds scans first)")
    ap.add_argument("--scan-seconds", type=float, default=20, help="scan for this long first (without --scan)")
    ap.add_argument("--replay", help="a recorded track folder instead of the phone (same world as --scan)")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (0: as fast as possible)")
    ap.add_argument("--start", type=float, help="replay from this recording time (s)")
    ap.add_argument("--prompt", default=tk.rf.DEFAULT_PROMPT, help=f"detector labels (default \"{tk.rf.DEFAULT_PROMPT}\")")
    ap.add_argument("--ball", action="store_true", help="the shell game: balls under cups")
    ap.add_argument("--results", help="target = Qwen-3D's first answer (a python -m ground3d.ground --out folder)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default=f"runs/live_{time.strftime('%H%M%S')}")
    ap.add_argument("--video", action="store_true", help="also write the x-ray view to <out>/xray.mp4")
    args = ap.parse_args(argv)
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
    labels = detect.parse_prompt(args.prompt)
    cached = detect.load_detections(detect.cache_path(args.replay, labels)) if args.replay else None
    scan_cached = args.scan and detect.load_detections(detect.cache_path(args.scan, labels)) is not None
    detector = None if cached is not None and scan_cached else detect.Detector(labels)
    phone = None
    if not args.replay:
        from .phone import Phone
        phone = Phone()
    scan = Path(args.scan) if args.scan else scan_with_phone(phone, args.scan_seconds, out, server)
    if detector is not None:
        detect_keyframes(scan, detector, labels)
    objects, table = tk.scan_objects(scan, args.prompt, use_detector=False)
    print(f"scene: {len(objects)} objects, table at {table if table is None else round(table, 3)} m: "
          + ", ".join(f"{o['label']} at {tk.r4(o['position'])}" for o in objects), flush=True)
    game = None
    if args.ball:
        cups, balls = shell.split(objects)
        tracker = tk.Tracker(cups, table, labels={c["label"] for c in cups})
        game = shell.ShellGame(tracker, balls, table, scan)
    else:
        tracker = tk.Tracker(objects, table)
        if args.results:
            point, query = qwen_target(args.results)
            tr = None if point is None else tk.nearest(tracker.tracks, point)
            tracker.target_id = None if tr is None else tr.id
            print(f"target: '{query}' -> {'none' if tr is None else f'{tr.label} {tr.id}'}", flush=True)
    try:
        server.scene = scene_json(scan, tracker.tracks)
    except Exception as e:
        print(f"(no /scene: {e})")
    source = fr.FolderSource(args.replay) if args.replay else phone
    to_world = None if args.replay else world_from_phone(scan)
    log = open(out / "states.jsonl", "w")
    video, verdict, verdict_t, seen_verdicts = None, None, 0.0, set()
    fps, last = None, None
    print("tracking (Ctrl+C stops)", flush=True)
    try:
        for f in frames_from(source, args.start, args.speed):
            if to_world is not None:
                f.T = to_world @ f.T
            dets = cached.get(source.ids[f.index], []) if cached is not None else detector(f.rgb)
            tracker.update(f, dets)
            if game is not None:
                game.update(f, dets)
            now = time.time()
            if last is not None:
                fps = 1 / max(1e-3, now - last) if fps is None else 0.9 * fps + 0.1 / max(1e-3, now - last)
            last = now
            msg = tracker.message(f, fps)
            if game is not None:
                msg = game.message(msg)
                for b in game.balls:
                    if b.verdict and id(b.verdict) not in seen_verdicts:
                        seen_verdicts.add(id(b.verdict))
                        v = b.verdict
                        verdict = (f"REVEAL {b.id}: cup {v['actual']}, we said cup {v['believed']}: "
                                   f"{'RIGHT' if v['right'] else 'WRONG'}")
                        verdict_t = f.t
            img = draw_view(cv2.cvtColor(f.rgb, cv2.COLOR_RGB2BGR), f, tracker, game, fps,
                            verdict if verdict and f.t - verdict_t < 5 else None)
            server.publish(img, msg)
            log.write(json.dumps(msg, separators=(",", ":")) + "\n")
            if args.video:
                if video is None:
                    video = cv2.VideoWriter(str(out / "xray.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (img.shape[1], img.shape[0]))
                video.write(img)
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        if video is not None:
            video.release()
        if game is not None and game.events:
            print("events:\n" + "\n".join(f"  {t:6.2f} s  {e}" for t, e in game.events))
        print(f"-> {out}/states.jsonl" + (" and xray.mp4" if video is not None else ""), flush=True)
        server.close()
        if metrics is not None:
            metrics.stop()
        if phone is not None:
            sys.stdout.flush()
            os._exit(0)             # record3d's teardown at a normal exit crashes (as in xray.scan); all is written
    return 0


if __name__ == "__main__":
    sys.exit(main())
