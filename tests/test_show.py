"""python3 -m unittest discover -s tests   (ground3d.show on a one-frame synthetic scan)"""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
try:
    import cv2
    import numpy as np

    from ground3d import show
    from xray import frames as fr
except ImportError:
    np = None


@unittest.skipIf(np is None, "needs numpy and opencv")
class TestShow(unittest.TestCase):
    def test_draws_the_answer(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            K = np.array([[400.0, 0, 240], [0, 400.0, 320], [0, 0, 1]])
            T = np.eye(4)
            T[:3, :3] = np.diag([1.0, -1, -1])           # looking straight down from 1 m
            T[2, 3] = 1.0
            f = fr.Frame(0.0, np.full((640, 480, 3), 90, np.uint8), np.ones((640, 480), np.float32),
                         np.full((640, 480), 2, np.uint8), K, T)
            fr.write_frame(t / "scan", 0, f)
            res = t / "res"
            res.mkdir()
            pts = np.array([[x, y, 0.0] for x in np.linspace(-0.05, 0.05, 6) for y in np.linspace(-0.05, 0.05, 6)])
            q = "the cup closest to the can"
            (res / f"{show.slug(q)}.ply").write_text("ply\nformat ascii 1.0\nelement vertex 36\nproperty float x\n"
                                                    "property float y\nproperty float z\nend_header\n"
                                                    + "\n".join(" ".join(f"{c:.4f}" for c in p) for p in pts) + "\n")
            (res / "results.json").write_text(json.dumps({"answers": [dict(query=q, score=0.61, band="high", center=[0, 0, 0])]}))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(show.main([str(t / "scan"), str(res), "--out", str(t / "out")]), 0)
            img = cv2.imread(str(t / "out" / "view_0.jpg")).astype(int)
            drawn = np.abs(img - 90).sum(-1) > 30                 # dots on the grey frame (the colours are checked
            self.assertGreater(drawn[300:341, 220:261].sum(), 10)  # in test_refine: JPEG blurs small coloured dots)
            self.assertFalse(drawn[400:].any())                   # and nowhere else

    def test_slug_matches_the_runner(self):
        try:
            from ground3d import ground
        except ImportError:
            self.skipTest("needs torch")
        for q in ("the cup closest to the can", "The RED cup, left!", "x" * 80):
            self.assertEqual(show.slug(q), ground.slug(q))


if __name__ == "__main__":
    unittest.main()
