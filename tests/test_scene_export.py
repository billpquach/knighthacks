import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from viewer.tools.scene_export import export_scene
from xray.frames import Frame, write_frame


class SceneExportTests(unittest.TestCase):
    def test_real_frame_reader_confidence_depth_color_and_table(self):
        with tempfile.TemporaryDirectory() as folder:
            depth = np.full((12, 12), .67, dtype=np.float32)
            depth[0, 0] = 0
            depth[0, 3] = 3
            conf = np.full((12, 12), 2, dtype=np.uint8)
            conf[3, 0] = 1
            rgb = np.full((12, 12, 3), [120, 180, 60], dtype=np.uint8)
            K = np.diag([50., 50., 1.])
            frame = Frame(0, rgb, depth, conf, K, np.eye(4))
            write_frame(folder, 0, frame)
            write_frame(folder, 1, frame)
            scene = export_scene(Path(folder), [{"id": 42, "position": [0, 0, .73]}])
            self.assertEqual(len(scene["cloud"]["points"]), 13)
            self.assertAlmostEqual(scene["table"]["z"], .67)
            self.assertGreaterEqual(len(scene["table"]["outline"]), 3)
            self.assertEqual(scene["objects"][0]["id"], 42)
            self.assertEqual(scene["objects"][0]["points"], [])
            np.testing.assert_allclose(scene["cloud"]["colors"][0], rgb[0, 0], atol=2)
            self.assertEqual(scene["camera"]["fx"], 50)

    def test_point_budget(self):
        y, x = np.mgrid[:720, :720]
        points = np.column_stack([x.ravel() * .02, y.ravel() * .02, np.full(x.size, .67)])
        frame = (np.zeros((2160, 2160, 3), np.uint8), np.ones((2160, 2160)),
                 np.full((2160, 2160), 2), np.eye(3), np.eye(4))
        with patch('viewer.tools.scene_export.frame_ids', return_value=[0]), \
             patch('viewer.tools.scene_export.read_frame', return_value=frame), \
             patch('viewer.tools.scene_export.backproject', return_value=points):
            scene = export_scene('unused')
        self.assertEqual(len(scene['cloud']['points']), 50000)
        self.assertEqual(len(scene['cloud']['colors']), 50000)

    def test_empty_scan_fails_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'no color'):
                export_scene(folder)
