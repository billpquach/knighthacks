"""Record3D USB smoke test. Run with the iPhone plugged in and Record3D in USB streaming mode.

pip install record3d numpy opencv-python
python test_record3d.py
Press q in the window to quit.
"""
import time
from threading import Event

import cv2
import numpy as np
from record3d import Record3DStream


class Probe:
    def __init__(self):
        self.event = Event()
        self.session = None
        self.frames = 0
        self.t0 = time.time()

    def on_new_frame(self):
        self.event.set()

    def on_stream_stopped(self):
        print("Stream stopped")

    def connect(self, idx=0):
        devs = Record3DStream.get_connected_devices()
        print(f"Found {len(devs)} device(s)")
        if not devs:
            raise SystemExit("No device. Check cable, trust prompt, Apple Mobile Device Support, Record3D USB mode.")
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[idx])

    def run(self):
        printed = False
        while True:
            self.event.wait()
            depth = self.session.get_depth_frame()      # float32 meters
            rgb = self.session.get_rgb_frame()
            conf = self.session.get_confidence_frame()
            K = self.session.get_intrinsic_mat()
            pose = self.session.get_camera_pose()
            self.event.clear()
            self.frames += 1

            if not printed:
                print("depth", depth.shape, depth.dtype, "min/max m:", np.nanmin(depth), np.nanmax(depth))
                print("rgb", rgb.shape, "conf", None if conf is None else conf.shape)
                print("intrinsics fx fy cx cy:", K.fx, K.fy, K.tx, K.ty)
                print("pose q(xyzw) t:", pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz)
                printed = True

            if self.frames % 30 == 0:
                print(f"{self.frames / (time.time() - self.t0):.1f} fps")

            d = np.clip(depth / 3.0, 0, 1)
            d = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            cv2.imshow("depth", d)
            cv2.imshow("rgb", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    p = Probe()
    p.connect()
    p.run()
