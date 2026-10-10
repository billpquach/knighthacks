"""python3 -m unittest discover -s tests   (ground3d.refine and xray.render on a synthetic cup)

A table at z = 0.75 m with an upside-down cup (radius 4.5 cm, 12 cm tall) and the floor below, seen from six
keyframes around it at ~0.7 m. The seed is what a Qwen-3D answer looks like: sparse 2 cm points on the cup,
bleeding onto the table and with a few stray points.
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
try:
    import cv2  # noqa: F401
    import numpy as np

    from ground3d import refine
    from xray import frames as fr, render
except ImportError:
    np = None

TABLE, R, H = 0.75, 0.045, 0.12
K = np.array([[500.0, 0, 240], [0, 500.0, 320], [0, 0, 1]]) if np is not None else None


def look_at(cam, target):
    f = np.asarray(target, float) - cam
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, np.cross(f, x), f], 1)          # OpenCV camera: x right, y down, z forward
    T[:3, 3] = cam
    return T


def render_depth(T, w=480, h=640):
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1) @ T[:3, :3].T
    c = T[:3, 3]
    t = np.full((h, w), np.inf)

    def hit(tt, ok):
        nonlocal t
        tt = np.where(ok & (tt > 0), tt, np.inf)
        t = np.minimum(t, tt)
    dz = np.where(np.abs(d[..., 2]) < 1e-9, -1e-9, d[..., 2])
    tf = (0 - c[2]) / dz                                      # floor
    hit(tf, d[..., 2] < 0)
    tt = (TABLE - c[2]) / dz                                  # table top, 1.2 x 1.2 m
    p = c + tt[..., None] * d
    hit(tt, (np.abs(p[..., 0]) < 0.6) & (np.abs(p[..., 1]) < 0.6))
    a = d[..., 0] ** 2 + d[..., 1] ** 2                       # cup side
    b = 2 * (c[0] * d[..., 0] + c[1] * d[..., 1])
    cc = c[0] ** 2 + c[1] ** 2 - R * R
    disc = b * b - 4 * a * cc
    ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    z = c[2] + ts * d[..., 2]
    hit(ts, (disc > 0) & (z > TABLE) & (z < TABLE + H))
    ttop = (TABLE + H - c[2]) / dz                            # cup bottom (it is upside down): a disc on top
    p = c + ttop[..., None] * d
    hit(ttop, p[..., 0] ** 2 + p[..., 1] ** 2 < R * R)
    return np.where(np.isfinite(t) & (t < 4), t, 0).astype(np.float32)


def make_scan(folder, n=6):
    rng = np.random.default_rng(0)
    for k in range(n):
        a = np.radians(-70 + 140 * k / (n - 1))
        T = look_at(np.array([0.7 * np.sin(a), -0.7 * np.cos(a), 1.15]), [0, 0, TABLE + 0.06])
        depth = render_depth(T)
        f = fr.Frame(float(k), (rng.random((640, 480, 3)) * 255).astype(np.uint8), depth,
                     np.where(depth > 0, 2, 0).astype(np.uint8), K, T)
        fr.write_frame(folder, k, f)


def seed_points():
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 2 * np.pi, 120)
    side = np.stack([R * np.cos(ang), R * np.sin(ang), TABLE + rng.uniform(0, H, 120)], 1)
    table = np.stack([rng.uniform(-0.07, 0.07, 30), rng.uniform(-0.07, 0.07, 30), np.full(30, TABLE)], 1)
    table = table[np.hypot(table[:, 0], table[:, 1]) > R]
    stray = np.array([[0.25, 0.2, TABLE + 0.02], [0.26, 0.2, TABLE + 0.03]])
    return np.concatenate([side, table, stray]) + rng.normal(0, 0.004, (len(side) + len(table) + 2, 3))


@unittest.skipIf(np is None, "needs numpy and opencv")
class TestRefine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.scan = Path(cls.tmp.name) / "scan"
        make_scan(cls.scan)
        cls.keyframes = refine.load_keyframes(cls.scan)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_table_height(self):
        self.assertAlmostEqual(refine.table_height(self.keyframes), TABLE, delta=0.011)

    def test_refine_cup(self):
        table = refine.table_height(self.keyframes)
        t = time.time()
        r = refine.refine(seed_points(), self.keyframes, table)
        secs = time.time() - t
        p = r["points"]
        self.assertGreaterEqual(r["n_points"], 800, r["n_points"])
        self.assertGreater(p[:, 2].min(), table + 0.01 - 1e-6)                 # nothing of the table
        radial = np.hypot(p[:, 0], p[:, 1])
        self.assertTrue((radial < R + 0.01).all(), radial.max())               # all of it on the cup
        self.assertTrue((p[:, 2] < TABLE + H + 0.01).all())
        self.assertLess(np.hypot(*r["center"][:2]), 0.02)                      # centre within 2 cm sideways
        self.assertEqual(r["views"], 6)
        self.assertLess(secs, 2.0)

    def test_largest_component(self):
        a = np.random.default_rng(2).uniform(-0.02, 0.02, (300, 3))       # one dense blob
        b = a[:40] + [0.3, 0, 0]
        keep = refine.largest_component(np.concatenate([a, b]), 0.005, 0.015)
        self.assertEqual(keep[:300].sum(), 300)
        self.assertEqual(keep[300:].sum(), 0)

    def test_flying_pixels(self):
        d = np.full((40, 40), 1.0, np.float32)
        d[:, 20:] = 2.0                                       # a depth edge
        ok = refine.clean_pixels(d, np.full((40, 40), 2, np.uint8), stride=1)
        self.assertFalse(ok[:, 19:21].any())                  # the edge pixels go
        self.assertTrue(ok[:, :18].all() and ok[:, 22:].all())

    def test_render_cup(self):
        table = refine.table_height(self.keyframes)
        r = refine.refine(seed_points(), self.keyframes, table)
        depth, conf, K_, T = self.keyframes[2]
        img = np.zeros((640, 480, 3), np.uint8)
        info = render.draw_cloud(img, r["points"], K_, T, depth, centre=r["center"])
        self.assertGreater(info["drawn"], 200)
        drawn = img.reshape(-1, 3)[img.reshape(-1, 3).any(1)].astype(int)
        self.assertTrue((drawn[:, 0] >= drawn[:, 2]).all())   # blues and cyans only (BGR: blue >= red)
        self.assertGreater(drawn[:, 1].max() - drawn[:, 1].min(), 60)            # near pale, far deep blue


if __name__ == "__main__":
    unittest.main()
