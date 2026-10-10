"""python3 -m unittest discover -s tests   (ground3d.refine and xray.render on a synthetic cup)

A table at z = 0.75 m with an upside-down cup (radius 4.5 cm, 12 cm tall) and the floor below, seen from eight
keyframes around it at ~0.7 m. The seed is what a Qwen-3D answer looks like: sparse 2 cm points on the cup,
bleeding onto the table, with a few stray points. The detector masks are the cup's silhouette dilated 6 px
(real masks spill over the edges).
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
try:
    import cv2
    import numpy as np

    from ground3d import refine
    from xray import detect, frames as fr, render
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


def render_scene(T, w=480, h=640, wall=None):
    """(depth, cup mask): floor, table top, the cup (side + closed top); optional wall plane y = wall."""
    v, u = np.mgrid[0:h, 0:w]
    d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones((h, w))], -1) @ T[:3, :3].T
    c = T[:3, 3]
    t = np.full((h, w), np.inf)
    cup = np.zeros((h, w), bool)

    def hit(tt, ok, is_cup=False):
        nonlocal t, cup
        tt = np.where(ok & (tt > 0), tt, np.inf)
        closer = tt < t
        t = np.where(closer, tt, t)
        cup = np.where(closer, is_cup, cup)
    dz = np.where(np.abs(d[..., 2]) < 1e-9, -1e-9, d[..., 2])
    hit((0 - c[2]) / dz, d[..., 2] < 0)                       # floor
    tt = (TABLE - c[2]) / dz                                  # table top, 1.2 x 1.2 m
    p = c + tt[..., None] * d
    hit(tt, (np.abs(p[..., 0]) < 0.6) & (np.abs(p[..., 1]) < 0.6))
    a = d[..., 0] ** 2 + d[..., 1] ** 2                       # cup side
    b = 2 * (c[0] * d[..., 0] + c[1] * d[..., 1])
    cc = c[0] ** 2 + c[1] ** 2 - R * R
    disc = b * b - 4 * a * cc
    ts = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * np.maximum(a, 1e-12))
    z = c[2] + ts * d[..., 2]
    hit(ts, (disc > 0) & (z > TABLE) & (z < TABLE + H), True)
    ttop = (TABLE + H - c[2]) / dz                            # the cup's (closed) top
    p = c + ttop[..., None] * d
    hit(ttop, p[..., 0] ** 2 + p[..., 1] ** 2 < R * R, True)
    if wall is not None:                                      # something between the camera and the cup
        dy = np.where(np.abs(d[..., 1]) < 1e-9, 1e-9, d[..., 1])
        tw = (wall - c[1]) / dy
        hit(tw, np.ones_like(tw, bool))
    return np.where(np.isfinite(t) & (t < 4), t, 0).astype(np.float32), cup


def poses(n=8):
    """Walking most of the way around the table, as a scan does."""
    out = []
    for k in range(n):
        a = np.radians(-150 + 300 * k / (n - 1))
        out.append(look_at(np.array([0.7 * np.sin(a), -0.7 * np.cos(a), 1.15]), [0, 0, TABLE + 0.06]))
    return out


def make_scan(folder):
    rng = np.random.default_rng(0)
    dets = {}
    for k, T in enumerate(poses()):
        depth, cup = render_scene(T)
        f = fr.Frame(float(k), (rng.random((640, 480, 3)) * 255).astype(np.uint8), depth,
                     np.where(depth > 0, 2, 0).astype(np.uint8), K, T)
        fr.write_frame(folder, k, f)
        mask = cv2.dilate(cup.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
        ys, xs = np.nonzero(mask)
        dets[k] = [detect.Detection("solo cup", 0.85, (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())), mask)]
    return dets


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
        by_frame = make_scan(cls.scan)
        detect.save_detections(detect.cache_path(cls.scan, ["solo cup", "ball"]), by_frame, ["solo cup", "ball"], (640, 480))
        cls.keyframes = refine.load_keyframes(cls.scan)
        cls.dets = refine.scan_detections(cls.scan, "solo cup,ball", use_detector=False)
        cls.table = refine.table_height(cls.keyframes)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_cup(self, r, min_points=800):
        p = r["points"]
        self.assertGreaterEqual(r["n_points"], min_points, r["n_points"])
        self.assertGreater(p[:, 2].min(), self.table + 0.01 - 1e-6)            # nothing of the table
        radial = np.hypot(p[:, 0], p[:, 1])
        self.assertTrue((radial < R + 0.01).all(), radial.max())               # all of it on the cup
        self.assertTrue((p[:, 2] < TABLE + H + 0.01).all())
        self.assertLess(np.hypot(*r["center"][:2]), 0.02)                      # centre within 2 cm sideways

    def test_table_height(self):
        self.assertAlmostEqual(self.table, TABLE, delta=0.011)

    def test_with_masks(self):
        t = time.time()
        r = refine.refine(seed_points(), self.keyframes, self.table, label="cup", dets=self.dets)
        self.assertLess(time.time() - t, 2.0)
        self.assertEqual((r["masked_views"], r["views"], r["kind"]), (8, 8, "cup"))
        self.check_cup(r)
        sx, sy, sz = r["size"]
        self.assertLess(abs(sx - 2 * R), 0.02)                                  # ~9 cm across
        self.assertLess(abs(sy - 2 * R), 0.02)
        self.assertGreater(sz, 0.09)                                           # ~12 cm tall (minus the table cut)

    def test_without_masks(self):
        r = refine.refine(seed_points(), self.keyframes, self.table, label="cup", dets=None)
        self.assertEqual(r["masked_views"], 0)
        self.check_cup(r, min_points=500)                                      # seed radius 1.5 cm: fewer points

    def test_objects_from_detections(self):
        found = refine.objects_from_detections(self.keyframes, self.dets)
        self.assertEqual([(lab, views) for lab, _, views, _ in found], [("solo cup", 8)])
        r = refine.refine(found[0][1], self.keyframes, self.table, label="solo cup", dets=self.dets)
        self.check_cup(r)

    def test_cache_roundtrip(self):
        cached = detect.load_detections(detect.cache_path(self.scan, ["solo cup", "ball"]))
        self.assertEqual(sorted(cached), list(range(8)))
        d = cached[2][0]
        self.assertEqual((d.label, round(d.score, 2), d.mask.shape), ("solo cup", 0.85, (640, 480)))
        self.assertGreater(d.mask.sum(), 500)
        self.assertIsNone(detect.load_detections(self.scan / "nothing.npz"))

    def test_pieces(self):
        self.assertEqual([refine.kind(x) for x in ("solo cup", "the red cup", "ball", "can", "box")], ["cup", "cup", "ball", "can", None])
        a = np.random.default_rng(2).uniform(-0.02, 0.02, (300, 3))
        keep = refine.largest_component(np.concatenate([a, a[:40] + [0.3, 0, 0]]), 0.005, 0.015)
        self.assertEqual((keep[:300].sum(), keep[300:].sum()), (300, 0))
        d = np.full((40, 40), 1.0, np.float32)
        d[:, 20:] = 2.0                                       # a depth edge
        ok = refine.clean_pixels(d, np.full((40, 40), 2, np.uint8), stride=1)
        self.assertFalse(ok[:, 19:21].any())
        self.assertTrue(ok[:, :18].all() and ok[:, 22:].all())
        layer = refine.nearest_layer(d, np.ones((40, 40), bool))
        self.assertTrue(layer[:, :20].all() and not layer[:, 20:].any())     # the near surface only
        pts = np.concatenate([np.zeros((50, 3)), [[0.2, 0, 0]]])
        self.assertEqual(refine.in_cylinder(pts, 0.06).sum(), 50)


@unittest.skipIf(np is None, "needs numpy and opencv")
class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ang = np.random.default_rng(3).uniform(0, 2 * np.pi, 6000)
        hz = np.random.default_rng(4).uniform(0, H, 6000)
        cls.cup = np.stack([R * np.cos(ang), R * np.sin(ang), TABLE + hz], 1)  # a full shell, front and back
        cls.T = poses()[2]
        cls.depth, _ = render_scene(cls.T)

    def draw(self, **kw):
        img = np.full((640, 480, 3), 60, np.uint8)
        info = render.draw_cloud(img, self.cup, K, self.T, kw.pop("depth", self.depth), centre=[0, 0, TABLE + 0.06],
                                 radius=R, **kw)
        return img, info

    def test_sparse_and_steady(self):
        img1, info = self.draw(target=True, order_seed=7)
        img2, _ = self.draw(target=True, order_seed=7)
        self.assertTrue(np.array_equal(img1, img2))                            # same dots every frame
        self.assertLessEqual(info["shown"], 1800)
        self.assertLessEqual(self.draw(order_seed=7)[1]["shown"], 900)          # others: 900
        self.assertEqual(info["hidden"], 0)                                    # its own back wall is not hidden
        changed = np.abs(img1.astype(int) - 60).sum(-1) > 20
        ys, xs = np.nonzero(changed)
        box = changed[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        self.assertLess(box.mean(), 0.8)                                       # see-through: gaps between dots
        px = img1[changed].astype(int)
        self.assertTrue((px[:, 0] >= px[:, 2]).all())                         # the cyan-azure family

    def test_hidden_and_glow(self):
        wall_depth, _ = render_scene(self.T, wall=self.T[1, 3] / 2)            # a wall halfway to the cup
        img, info = self.draw(depth=wall_depth)
        self.assertGreater(info["hidden"], 0)
        plain, _ = self.draw(depth=wall_depth)
        glow, _ = self.draw(depth=wall_depth, glow=True)
        self.assertGreater(glow.astype(int).sum(), plain.astype(int).sum())    # the halo brightens the image

    def test_budget(self):
        small = render.subsample(self.cup, 100, 5)
        self.assertTrue(np.array_equal(small, render.subsample(self.cup, 200, 5)[:100]))   # grows, never swaps
        near = render.screen_budget(self.cup, K, self.T, (640, 480))
        far = self.T.copy()
        far[:3, 3] = self.T[:3, 3] * 2 - [0, 0, TABLE + 0.06]                  # twice as far from the cup
        self.assertLess(render.screen_budget(self.cup, K, far, (640, 480)), near / 2)

    def test_ramp(self):
        c = render.ramp(np.array([0.0, 0.5, 1.0]))
        np.testing.assert_allclose(c, [[255, 255, 200], [255, 220, 0], [255, 130, 20]])


if __name__ == "__main__":
    unittest.main()
