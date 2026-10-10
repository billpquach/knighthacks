"""Stand-in for the Jetson viewer, for building the glasses app and the rover without the Jetson.

python replay_server.py                                # synthetic states in the contract shape
python replay_server.py docs/sample_state_G8.jsonl     # replay a recorded state stream at 5 Hz, looped

GET /state        server-sent events, one JSON message per event (SSE)
GET /state.json   the newest state, once (for clients that poll instead of streaming)
GET /frame.jpg    one JPEG per request, with X-Seq and X-Age-Ms headers
Every response allows any origin (CORS), so a page served from elsewhere can read all of them.
Only needs numpy and opencv-python (already installed). Port 8080, or pass --port.
"""
import argparse
import json
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

import cv2
from viewer_demo import synthetic as demo_state, render_frame

HZ = 5.0
T0 = time.time()
STATES = []
SCENE = None
TRACK = None
VIEWER_DIST = Path(__file__).resolve().parent / "viewer" / "dist"


def synthetic(i):
    return demo_state(i, HZ)


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                STATES.append(json.loads(line))
    print(f"Loaded {len(STATES)} states from {path}")


def state_at(n):
    if STATES:
        index = n % len(STATES)
        state = {**STATES[index], "source": "replay"}
        return TRACK.enrich(state, index) if TRACK else state
    return synthetic(n)


def current_index():
    elapsed = time.time() - T0
    return TRACK.state_index(elapsed) if TRACK else int(elapsed * HZ)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        # CORS on every response, errors included, sent once here (a repeated Allow-Origin header is rejected)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Seq, X-Age-Ms, X-State-Index, X-Frame-Time")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if path == "/state":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    n = current_index()
                    self.wfile.write(("data: " + json.dumps(state_at(n), separators=(",", ":")) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(1.0 / HZ)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif path == "/state.json":
            body = json.dumps(state_at(current_index()), separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/scene":
            if SCENE is None:
                self.send_error(404, "No scene supplied; use --scene scene.json")
                return
            body = json.dumps(SCENE, separators=(",", ":"), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path in ("/frame.jpg", "/frame-raw.jpg"):
            if STATES and not TRACK:
                self.send_error(404, "No camera recording supplied with state-only replay")
                return
            index = current_index()
            if TRACK:
                try:
                    requested = parse_qs(urlsplit(self.path).query).get('index', [str(index)])[0]
                    index = int(requested)
                    if not 0 <= index < len(STATES):
                        raise ValueError('Index out of range')
                except ValueError:
                    self.send_error(400, 'Invalid state index')
                    return
                state = state_at(index)
                seq = state['replay']['frame']
                try:
                    body = TRACK.image(state, index, raw=path == '/frame-raw.jpg')
                except (OSError, ValueError):
                    self.send_error(404, 'Recorded frame unavailable')
                    return
            else:
                seq = index
                img = render_frame(state_at(seq))
                ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                body = jpg.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Seq", str(seq))
            self.send_header("X-Age-Ms", "0")
            if TRACK:
                self.send_header("X-State-Index", str(index))
                self.send_header("X-Frame-Time", str(state['replay']['frame_t']))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path == "/viewer-info":
            body = json.dumps({"mode": "replay" if STATES else "synthetic", "hz": HZ,
                               "camera": "recorded-overlay" if TRACK else "none" if STATES else "synthetic"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            file = (VIEWER_DIST / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
            if not file.is_relative_to(VIEWER_DIST.resolve()) or not file.is_file():
                self.send_error(404, "Viewer not found. Build with: cd viewer && npm install && npm run build")
                return
            body = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 explicitly for LAN clients")
    ap.add_argument("--scene", type=Path, help="Static scene.json exported from a scan")
    ap.add_argument("--track", type=Path, help="Timestamped track folder for recorded RGB and telemetry overlay")
    ap.add_argument("--hz", type=float, default=5, help="Replay state cadence (5 Hz sample, 15 Hz full G8)")
    a = ap.parse_args()
    if not 0 < a.hz <= 120:
        ap.error("--hz must be between 0 and 120")
    HZ = a.hz
    if a.scene:
        SCENE = json.loads(a.scene.read_text())
        if not isinstance(SCENE, dict) or any(SCENE.get(k) != v for k, v in
                dict(version=1, frame="world", up="z", units="m").items()):
            ap.error("--scene requires version 1, world frame, z up, metres")
        json.dumps(SCENE, allow_nan=False)
    if a.jsonl:
        load(a.jsonl)
    else:
        print("No file given, serving synthetic states")
    if a.track:
        if not STATES:
            ap.error('--track requires a state JSONL recording')
        from recorded_track import RecordedTrack
        TRACK = RecordedTrack(a.track, STATES)
    T0 = time.time()
    srv = ThreadingHTTPServer((a.host, a.port), H)
    print(f"Serving on http://{a.host}:{a.port}  (/state, /state.json, /frame.jpg, /scene)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
