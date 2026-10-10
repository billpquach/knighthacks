"""Stand-in for the Jetson viewer, for building the glasses app and the rover without the Jetson.

python replay_server.py                          # synthetic states in the contract shape
python replay_server.py sample_state_G8.jsonl    # replay Bill's real recording at 5 Hz, looped

GET /state       server-sent events, one JSON message per event (SSE)
GET /frame.jpg   one JPEG per request, with X-Seq and X-Age-Ms headers
Only needs numpy and opencv-python (already installed). Port 8080, or pass --port.
"""
import argparse
import json
import math
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

HZ = 5.0
T0 = time.time()
STATES = []


def synthetic(i):
    t = i / HZ
    cups = []
    for k, (x, y) in enumerate([(0.30, 1.00), (0.49, 1.05), (0.68, 1.00)]):
        # cup 2 goes out of view for a while to exercise the OCCLUDED / OUT_OF_VIEW paths
        phase = (t % 30)
        state = "VISIBLE"
        if k == 2 and 10 < phase < 20:
            state = "OCCLUDED"
        if k == 0 and 20 < phase < 28:
            state = "OUT_OF_VIEW"
        cups.append({
            "id": k, "label": "solo cup", "target": k == 1, "state": state,
            "position": [x, y, 0.72], "velocity": [0, 0, 0], "size": [0.10, 0.12, 0.11],
            "last_seen_t": t if state == "VISIBLE" else max(0.0, t - 5),
            "since_seen_s": 0.0 if state == "VISIBLE" else 5.0,
            "confidence": 1.0, "detection_score": 0.9, "reacquired": False,
            "has_ball": k == 1, "balls_inside": ["yellow ball"] if k == 1 else [],
        })
    ang = 0.4 * math.sin(t / 6)
    fwd = [-0.55 * math.cos(ang), -0.18, -0.81]
    ball_state = "INSIDE" if (t % 30) > 5 else "VISIBLE"
    ball = {
        "id": "yellow ball", "label": "ball", "color": "yellow", "state": ball_state, "inside": 1,
        "position": [0.49, 1.05, 0.705], "since_seen_s": 0.0 if ball_state == "VISIBLE" else (t % 30) - 5,
        "detection_score": None, "verdict": None, "balls": [],
    }
    return {
        "t": round(t, 3), "frame": i, "target_id": 1, "phase": "track", "fps": 22.0,
        "camera": {"position": [0.89, 1.05, 1.36], "forward": fwd,
                   "T_world_cam": [1, 0, 0, 0.89, 0, 1, 0, 1.05, 0, 0, 1, 1.36]},
        "objects": cups, "ball": ball,
    }


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                STATES.append(json.loads(line))
    print(f"Loaded {len(STATES)} states from {path}")


def state_at(n):
    if STATES:
        return STATES[n % len(STATES)]
    return synthetic(n)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            n = 0
            try:
                while True:
                    self.wfile.write(("data: " + json.dumps(state_at(n)) + "\n\n").encode())
                    self.wfile.flush()
                    n += 1
                    time.sleep(1.0 / HZ)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith("/frame.jpg"):
            seq = int((time.time() - T0) * 10)
            img = np.full((480, 640, 3), 28, np.uint8)
            cv2.putText(img, "DUMMY X-RAY FRAME", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (230, 220, 60), 2)
            cv2.putText(img, f"seq {seq}", (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
            cv2.circle(img, (320 + int(80 * math.sin(seq / 10)), 360), 24, (60, 200, 230), -1)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            body = jpg.tobytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Seq", str(seq))
            self.send_header("X-Age-Ms", "0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            body = b"/state (SSE) and /frame.jpg"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
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
    print(f"Serving on http://0.0.0.0:{a.port}  (/state, /frame.jpg)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)