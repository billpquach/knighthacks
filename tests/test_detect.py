"""python3 -m unittest discover -s tests   (the parts of xray.detect that don't need ultralytics)"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core"))
try:
    import cv2  # noqa: F401
    import numpy as np

    from xray import detect
except ImportError:                                   # (the system python on a dev box may have neither)
    np = detect = None


@unittest.skipIf(np is None, "needs numpy and opencv")
class TestDetect(unittest.TestCase):
    def test_prompt_and_names(self):
        labels = detect.parse_prompt(" solo cup, ball ")
        self.assertEqual(labels, ["solo cup", "ball"])
        self.assertEqual(detect.pt_path(labels).name, "yoloe-11s-seg_solo-cup_ball.pt")
        self.assertEqual(detect.engine_path(labels).name, "yoloe-11s-seg_solo-cup_ball_640x480.engine")
        for bad in ("", " , ", "cup,cup"):
            with self.assertRaises(SystemExit):
                detect.parse_prompt(bad)

    def test_mask_at(self):
        m = np.zeros((640, 480), bool)
        m[100:300, 200:280] = True
        d = detect.Detection("solo cup", 0.8, (200.0, 100.0, 280.0, 300.0), m)
        small = d.mask_at((256, 192))                       # the LiDAR depth size
        self.assertEqual(small.shape, (256, 192))
        self.assertAlmostEqual(small.mean(), m.mean(), delta=0.01)
        self.assertIs(d.mask_at((640, 480)), m)
        self.assertIs(d.mask_at((256, 192)), small)         # cached
        self.assertEqual(d.center, (240.0, 200.0))

    def test_frames_in(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            c = Path(t) / "color"
            c.mkdir()
            for n in ("10.jpg", "2.jpg", "1.jpg", "notes.txt"):
                (c / n).write_text("")
            self.assertEqual([f.name for f in detect.frames_in(t)], ["1.jpg", "2.jpg", "10.jpg"])


if __name__ == "__main__":
    unittest.main()
