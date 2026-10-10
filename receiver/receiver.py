"""Record3D -> Jetson frame forwarder.

pip install record3d numpy opencv-python msgpack websockets

Record only (no Jetson yet):
    python receiver.py --record recordings/test1
Forward to the Jetson:
    python receiver.py --url ws://JETSON_IP:8765/frames --fps 10
Both at once works too. Ctrl+C to stop.

Frame message (msgpack map):
    frame_id    int
    timestamp   float, seconds (laptop clock)
    rgb_jpeg    bytes, JPEG (RGB scaled to --rgb-width)
    depth_mm    bytes, uint16 little-endian, row-major
    depth_shape [h, w]
    conf        bytes, uint8 row-major (same shape as depth) or None
    K_color     [fx, fy, cx, cy] in sent-RGB pixels
    K_depth     [fx, fy, cx, cy] in depth pixels
    rgb_shape   [h, w] of the JPEG
    pose        16 floats, row-major 4x4 ARKit camera pose (camera -> ARKit world)
"""
import argparse
import os
import time
from threading import Event

import cv2
import msgpack
import numpy as np
from record3d import Record3DStream


def pose_to_4x4(p):
    qx, qy, qz, qw = p.qx, p.qy, p.qz, p.qw
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5 or 1.0
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [p.tx, p.ty, p.tz]
    return T


class Receiver:
    def __init__(self, args):
        self.args = args
        self.event = Event()
        self.session = None
        self.frame_id = 0
        self.ws = None
        if args.record:
            os.makedirs(args.record, exist_ok=True)

    # record3d callbacks
    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect_phone(self):
        devs = Record3DStream.get_connected_devices()
        if not devs:
            raise SystemExit("No iPhone found. Plug in, unlock, Record3D in Live + USB mode.")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[0])
        print("Phone connected")

    def connect_ws(self):
        if not self.args.url:
            return
        from websockets.sync.client import connect
        try:
            self.ws = connect(self.args.url, max_size=None, open_timeout=3)
            print("Connected to", self.args.url)
        except Exception as e:
            self.ws = None
            print("WebSocket connect failed:", e)

    def build_message(self):
        s = self.session
        depth = s.get_depth_frame()
        rgb = s.get_rgb_frame()
        conf = s.get_confidence_frame()
        K = s.get_intrinsic_mat()
        pose = s.get_camera_pose()

        dh, dw = depth.shape[:2]
        rh, rw = rgb.shape[:2]
        scale = min(1.0, self.args.rgb_width / rw)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(rw * scale), int(rh * scale)), interpolation=cv2.INTER_AREA)
        sh, sw = rgb.shape[:2]
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality])
        if not ok:
            return None

        d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth_mm = np.clip(d * 1000.0, 0, 65535).astype("<u2")

        # intrinsics are for the full-res RGB; depth is the same view at lower res
        K_full = [K.fx, K.fy, K.tx, K.ty]
        k_color = [v * scale for v in K_full]
        ratio = dw / rw
        k_depth = [v * ratio for v in K_full]

        self.frame_id += 1
        return {
            "frame_id": self.frame_id,
            "timestamp": time.time(),
            "rgb_jpeg": jpg.tobytes(),
            "rgb_shape": [sh, sw],
            "depth_mm": depth_mm.tobytes(),
            "depth_shape": [dh, dw],
            "conf": None if conf is None or conf.size == 0 else np.asarray(conf, dtype=np.uint8).tobytes(),
            "K_color": k_color,
            "K_depth": k_depth,
            "pose": pose_to_4x4(pose).flatten().tolist(),
        }

    def run(self):
        self.connect_phone()
        self.connect_ws()
        period = 1.0 / self.args.fps
        last_sent = 0.0
        last_retry = time.time()
        sent = 0
        t0 = time.time()
        try:
            while True:
                self.event.wait()
                self.event.clear()
                now = time.time()
                if now - last_sent < period:
                    continue
                last_sent = now

                msg = self.build_message()
                if msg is None:
                    continue
                payload = msgpack.packb(msg, use_bin_type=True)

                if self.args.record:
                    with open(os.path.join(self.args.record, f"{msg['frame_id']:06d}.msgpack"), "wb") as f:
                        f.write(payload)

                if self.args.url:
                    if self.ws is None and now - last_retry > 2.0:
                        last_retry = now
                        self.connect_ws()
                    if self.ws is not None:
                        try:
                            self.ws.send(payload)
                        except Exception as e:
                            print("Send failed, will retry:", e)
                            self.ws = None

                sent += 1
                if sent % 30 == 0:
                    print(f"frame {msg['frame_id']}  {sent / (time.time() - t0):.1f} fps  {len(payload) / 1024:.0f} KB/frame")
        except KeyboardInterrupt:
            print("Stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="ws://JETSON_IP:PORT/path")
    ap.add_argument("--record", help="folder to save frames as .msgpack")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--rgb-width", type=int, default=960)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    Receiver(ap.parse_args()).run()