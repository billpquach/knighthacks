"""The phone on the Jetson's USB port: Record3D (LiDAR camera) frames as xray Frames.

Per frame:
  * pose: ARKit's camera-to-world. Its camera has x right, y up, z backwards; an OpenCV camera has y down,
    z forward, so y and z flip. Its world has y up (gravity); ours has z up, so the world turns +90 deg
    about x. The scan later shifts this world so the floor is at z = 0 and the scan is centred in xy.
  * images: RGB resized to 480 x 640 (portrait; 640 x 480 if the stream is landscape), depth and
    confidence upsampled to that size by nearest neighbour, intrinsics scaled to it.

Needs the record3d package (pip install record3d) and the Record3D app streaming over USB.
"""
import threading
import time

import cv2
import numpy as np

from .frames import Frame

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])              # camera axes: ARKit -> OpenCV
Y_UP_TO_Z_UP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # world: y up -> z up


def quat_to_matrix(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def arkit_pose(qx, qy, qz, qw, tx, ty, tz):
    """ARKit camera-to-world (ARKit camera, y-up world) as a 4 x 4."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def zup_from_arkit(T_arkit):
    """ARKit camera-to-world -> OpenCV camera in the z-up world."""
    return Y_UP_TO_Z_UP @ T_arkit @ ARKIT_TO_CV


def output_size(w, h):
    return (480, 640) if w < h else (640, 480)


def intrinsics_reference(K4, rgb_wh, depth_wh):
    """Which image the stream's intrinsics belong to: the one whose centre is nearest the principal point."""
    fx, fy, cx, cy = K4
    d_rgb = abs(cx - rgb_wh[0] / 2) + abs(cy - rgb_wh[1] / 2)
    d_depth = abs(cx - depth_wh[0] / 2) + abs(cy - depth_wh[1] / 2)
    return rgb_wh if d_rgb <= d_depth else depth_wh


def convert(rgb, depth, conf, K4, pose7, t, index=0, ref_wh=None):
    """Raw Record3D arrays -> Frame (z-up world, before the scan's shift). K4 = (fx, fy, cx, cy) for the
    image of size ref_wh (found from the principal point if not given); pose7 = (qx, qy, qz, qw, tx, ty, tz)."""
    h, w = rgb.shape[:2]
    dh, dw = depth.shape[:2]
    ow, oh = output_size(w, h)
    ref_wh = ref_wh or intrinsics_reference(K4, (w, h), (dw, dh))
    sx, sy = ow / ref_wh[0], oh / ref_wh[1]
    fx, fy, cx, cy = K4
    K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
    rgb_o = cv2.resize(np.ascontiguousarray(rgb[..., :3]), (ow, oh), interpolation=cv2.INTER_AREA)
    depth = np.nan_to_num(np.asarray(depth, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth < 0] = 0
    depth_o = cv2.resize(depth, (ow, oh), interpolation=cv2.INTER_NEAREST)
    if conf is not None and np.size(conf) == dh * dw:
        conf_o = cv2.resize(np.asarray(conf, np.uint8).reshape(dh, dw), (ow, oh), interpolation=cv2.INTER_NEAREST)
    else:                                                   # no confidence from this device: depth means high
        conf_o = np.where(depth_o > 0, 2, 0).astype(np.uint8)
    return Frame(t, rgb_o, depth_o, conf_o, K, zup_from_arkit(arkit_pose(*pose7)), index)


class Phone:
    """The Record3D USB stream. read() waits for the next frame (the newest one; older ones are dropped)."""

    def __init__(self, device=0):
        from record3d import Record3DStream
        devs = Record3DStream.get_connected_devices()
        if not devs:
            raise SystemExit("no phone found: check the cable, tap 'Trust This Computer', and start USB streaming "
                             "in the Record3D app")
        self.stream = Record3DStream()
        self.new, self.stopped = threading.Event(), threading.Event()
        self.stream.on_new_frame = self.new.set
        self.stream.on_stream_stopped = self.stopped.set
        self.stream.connect(devs[device])
        self.t0 = time.monotonic()
        self.count = 0
        self.ref_wh = None
        self.info = {}

    def read(self, timeout=5.0):
        if self.stopped.is_set():
            raise RuntimeError("the phone stopped streaming")
        if not self.new.wait(timeout):
            raise RuntimeError(f"no frame from the phone for {timeout:.0f} s (is USB streaming on in Record3D?)")
        self.new.clear()
        s = self.stream
        rgb, depth, conf = np.asarray(s.get_rgb_frame()), np.asarray(s.get_depth_frame()), np.asarray(s.get_confidence_frame())
        k, p = s.get_intrinsic_mat(), s.get_camera_pose()
        K4 = (k.fx, k.fy, k.tx, k.ty)
        if self.ref_wh is None:
            h, w = rgb.shape[:2]
            dh, dw = depth.shape[:2]
            self.ref_wh = intrinsics_reference(K4, (w, h), (dw, dh))
            ow, oh = output_size(w, h)
            if dw * dh < 192 * 256:                          # (the rear LiDAR gives 192 x 256 depth)
                print(f"note: {dw}x{dh} depth is not the rear LiDAR's 192x256: is Record3D on the rear camera?")
            self.info = dict(rgb=f"{w}x{h}", depth=f"{dw}x{dh}", out=f"{ow}x{oh}",
                             intrinsics_for=f"{self.ref_wh[0]}x{self.ref_wh[1]}",
                             confidence="yes" if conf.size == depth.size else "no (depth > 0 counts as high)")
        f = convert(rgb, depth, conf, K4, (p.qx, p.qy, p.qz, p.qw, p.tx, p.ty, p.tz), time.monotonic() - self.t0,
                    self.count, self.ref_wh)
        self.count += 1
        return f
