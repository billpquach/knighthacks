"""python3 -m unittest discover -s tests

S1 without a phone: a stand-in for the record3d package streams a synthetic scene in ARKit's conventions
(y-up world, camera looking down its -z, 192 x 256 LiDAR depth, 720 x 960 RGB, intrinsics for the RGB),
seen from an orbit around a box on the floor. The real Phone, keyframe picker, world fit, consistency
check and scan writer run on it.
"""
import json
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
try:
    import cv2  # noqa: F401
    import numpy as np

    from xray import frames as fr, phone as ph, scan
except ImportError:
    np = None

FLOOR_Y = -1.0                                       # ARKit's origin is where the session started
BOX = ((0.15, -1.0, -1.30), (0.27, -0.88, -1.18))    # a cup-sized box on the floor (ARKit x, y, z)


def look_at(cam, target):
    """ARKit camera-to-world rotation: camera x right, y up, looking down its -z at target."""
    f = np.asarray(target, float) - cam
    f /= np.linalg.norm(f)
    z = -f
    x = np.cross([0.0, 1, 0], z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def matrix_to_quat(R):
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)


def render_depth(cam, R, K, w, h):
    """Camera-axis distance to the floor plane or the box, per pixel (0 = nothing)."""
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) / K[1, 1], -np.ones_like(u, float)], -1)   # ARKit camera
    ray = d @ R.T
    t = np.where(ray[..., 1] < -1e-6, (FLOOR_Y - cam[1]) / np.where(ray[..., 1] < -1e-6, ray[..., 1], -1), np.inf)
    lo, hi = np.array(BOX[0]), np.array(BOX[1])
    inv = 1 / np.where(np.abs(ray) < 1e-9, 1e-9, ray)
    t1, t2 = (lo - cam) * inv, (hi - cam) * inv
    tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
    box = (tf >= tn) & (tn > 0)
    t = np.where(box & (tn < t), tn, t)
    return np.where(np.isfinite(t) & (t > 0) & (t < 6), t, 0).astype(np.float32)


def make_frames(n=48):
    centre = np.array([0.21, FLOOR_Y, -1.24])
    K_rgb = np.array([[700.0, 0, 360], [0, 700.0, 480], [0, 0, 1]])
    K_d = K_rgb * np.array([[192 / 720], [256 / 960], [1]])
    rng = np.random.default_rng(1)
    texture = (rng.random((960, 720, 3)) * 255).astype(np.uint8)
    out = []
    for i in range(n):
        a = np.radians(-60 + 120 * i / (n - 1))
        cam = centre + np.array([1.0 * np.sin(a), 0.8, 1.0 * np.cos(a)])
        R = look_at(cam, centre + [0, 0.05, 0])
        depth = render_depth(cam, R, K_d, 192, 256)
        out.append((texture, depth, np.full((256, 192), 2, np.uint8), (700.0, 700.0, 360.0, 480.0),
                    (*matrix_to_quat(R), *cam)))
    return out


def fake_record3d(frames):
    """A record3d module whose stream plays `frames` once, one per read, then repeats the last."""
    class Stream:
        def __init__(self):
            self.cur, self.read_done = frames[0], threading.Event()
            self.on_new_frame = self.on_stream_stopped = None

        @staticmethod
        def get_connected_devices():
            return ["phone"]

        def connect(self, dev):
            def run():
                for f in frames + [frames[-1]] * 100000:
                    self.cur = f
                    self.read_done.clear()
                    self.on_new_frame()
                    self.read_done.wait(1.0)
            threading.Thread(target=run, daemon=True).start()
            return True

        def get_device_type(self):
            return 1

        def get_rgb_frame(self):
            return self.cur[0]

        def get_depth_frame(self):
            return self.cur[1]

        def get_confidence_frame(self):
            return self.cur[2]

        def get_intrinsic_mat(self):
            fx, fy, cx, cy = self.cur[3]
            return types.SimpleNamespace(fx=fx, fy=fy, tx=cx, ty=cy)

        def get_camera_pose(self):
            qx, qy, qz, qw, tx, ty, tz = self.cur[4]
            self.read_done.set()
            return types.SimpleNamespace(qx=qx, qy=qy, qz=qz, qw=qw, tx=tx, ty=ty, tz=tz)
    return types.SimpleNamespace(Record3DStream=Stream)


@unittest.skipIf(np is None, "needs numpy and opencv")
class TestConventions(unittest.TestCase):
    def test_axes(self):
        T = ph.zup_from_arkit(np.eye(4))                 # ARKit's starting camera: looking down ARKit -z
        np.testing.assert_allclose(T[:3, 2], [0, 1, 0], atol=1e-12)      # forward: horizontal
        np.testing.assert_allclose(T[:3, 1], [0, 0, -1], atol=1e-12)     # image down: gravity
        np.testing.assert_allclose(T[:3, 0], [1, 0, 0], atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(T[:3, :3]), 1.0)

    def test_convert(self):
        rgb = np.zeros((960, 720, 3), np.uint8)
        depth = np.ones((256, 192), np.float32)
        depth[0, 0] = np.nan
        f = ph.convert(rgb, depth, np.full((256, 192), 1, np.uint8), (700.0, 700.0, 360.0, 480.0), (0, 0, 0, 1, 0, 0, 0), 0.0)
        self.assertEqual((f.rgb.shape, f.depth.shape, f.conf.shape), ((640, 480, 3), (640, 480), (640, 480)))
        np.testing.assert_allclose(f.K, [[700 * 2 / 3, 0, 240], [0, 700 * 2 / 3, 320], [0, 0, 1]])
        self.assertEqual(f.depth[0, 0], 0)               # NaN depth -> none
        self.assertEqual(ph.intrinsics_reference((180.0, 180.0, 96.0, 128.0), (720, 960), (192, 256)), (192, 256))
        f = ph.convert(rgb, depth, np.zeros(0, np.uint8), (700.0, 700.0, 360.0, 480.0), (0, 0, 0, 1, 0, 0, 0), 0.0)
        self.assertEqual(int(f.conf.max()), 2)           # no confidence from the device: depth counts as high

    def test_picker(self):
        p = scan.KeyframePicker()
        rgb = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
        T = np.eye(4)
        mk = lambda T: fr.Frame(0, rgb, np.zeros((64, 64), np.float32), np.zeros((64, 64), np.uint8), np.eye(3), T)  # noqa: E731
        self.assertTrue(p.offer(mk(T)))
        T2 = T.copy()
        T2[0, 3] = 0.05
        self.assertFalse(p.offer(mk(T2)))                # 5 cm: not enough
        T2[0, 3] = 0.12
        self.assertTrue(p.offer(mk(T2)))                 # 12 cm
        T3 = T2.copy()
        c, s = np.cos(np.radians(20)), np.sin(np.radians(20))
        T3[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        self.assertTrue(p.offer(mk(T3)))                 # turned 20 deg


@unittest.skipIf(np is None, "needs numpy and opencv")
class TestScanFromStream(unittest.TestCase):
    def test_scan(self):
        frames = make_frames()
        sys.modules["record3d"] = fake_record3d(frames)
        try:
            phone = ph.Phone()
            keyframes, n, blurry = scan.capture(phone, 3.0)
        finally:
            sys.modules.pop("record3d", None)
        self.assertEqual(phone.info["intrinsics_for"], "720x960")
        self.assertGreaterEqual(len(keyframes), 8)
        self.assertGreaterEqual(n, len(frames))
        shift, world = scan.fit_world(keyframes)
        self.assertAlmostEqual(world["floor_z"], FLOOR_Y, delta=0.01)      # ARKit y -> world z
        self.assertAlmostEqual(world["camera_height_m"], 0.8, delta=0.05)
        self.assertEqual(world["image_down_share"], 1.0)
        import copy                                       # the camera flip undone: the floor lands above the phone
        flipped = [copy.copy(f) for f in keyframes]
        for f in flipped:
            f.T = f.T @ ph.ARKIT_TO_CV
        _, bad = scan.fit_world(flipped)
        self.assertTrue(bad["camera_height_m"] < 0.1 or bad["image_down_share"] < 0.8, bad)
        scan.apply(shift, keyframes)
        checks = scan.consistency(keyframes)
        self.assertTrue(checks)
        for a, b, cm, px in checks:
            self.assertIsNotNone(cm, (a, b, px))
            self.assertLess(cm, 2.0, (a, b))
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "scan"
            scan.write_scan(out, keyframes, shift, source="test")
            meta = json.loads((out / "meta.json").read_text())
            self.assertEqual(meta["num_frames"], len(keyframes))
            W = np.array(meta["world_from_arkit"])
            floor_pt = W @ np.array([0.5, FLOOR_Y, -1.0, 1.0])
            self.assertAlmostEqual(floor_pt[2], 0.0, delta=0.01)              # the floor at z = 0
            top = W @ np.array([0.21, BOX[1][1], -1.24, 1.0])
            self.assertAlmostEqual(top[2], 0.12, delta=0.01)                  # the box is 12 cm tall
            rgb, depth, conf, K, T = fr.read_frame(out, 0)                     # the folder reads back
            self.assertEqual(depth.shape, (640, 480))
            u, v, z = fr.project(top[None, :3], K, T)                          # the box top projects onto itself
            self.assertTrue(0 <= u[0] < 480 and 0 <= v[0] < 640)
            self.assertAlmostEqual(float(depth[int(round(v[0])), int(round(u[0]))]), float(z[0]), delta=0.03)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
            try:
                from ground3d import scene
                self.assertEqual(len(scene.load_frame(out, 0)), 5)            # what ground3d reads
            except ImportError:
                pass


if __name__ == "__main__":
    unittest.main()
