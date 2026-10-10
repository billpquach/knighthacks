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
from urllib.parse import unquote, urlsplit

import cv2
from viewer_demo import synthetic as demo_state, render_frame

HZ = 5.0
T0 = time.time()
STATES = []
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
        return {**STATES[n % len(STATES)], "source": "replay"}
    return synthetic(n)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        # CORS on every response, errors included, sent once here (a repeated Allow-Origin header is rejected)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "X-Seq, X-Age-Ms")
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
                    n = int((time.time() - T0) * HZ)
                    self.wfile.write(("data: " + json.dumps(state_at(n), separators=(",", ":")) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(1.0 / HZ)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif path == "/state.json":
            body = json.dumps(state_at(int((time.time() - T0) * HZ)), separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/frame.jpg":
            if STATES:
                self.send_error(404, "No camera recording supplied with state-only replay")
                return
            seq = int((time.time() - T0) * HZ)
            img = render_frame(state_at(seq))
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            body = jpg.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Seq", str(seq))
            self.send_header("X-Age-Ms", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif path == "/viewer-info":
            body = json.dumps({"mode": "replay" if STATES else "synthetic", "hz": HZ}).encode()
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
    a = ap.parse_args()
    if a.jsonl:
        load(a.jsonl)
    else:
        print("No file given, serving synthetic states")
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), H)
    print(f"Serving on http://0.0.0.0:{a.port}  (/state, /state.json, /frame.jpg)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
