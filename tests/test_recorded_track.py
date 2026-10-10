import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from recorded_track import RecordedTrack
from xray.frames import Frame, write_frame, write_meta


class RecordedTrackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        frames = [Frame(t, np.full((48, 64, 3), 80 + i * 20, np.uint8), np.ones((48, 64)),
                        np.full((48, 64), 2), np.array([[50, 0, 32], [0, 50, 24], [0, 0, 1.]]), np.eye(4))
                  for i, t in enumerate([10, 10.1, 10.3])]
        for i, f in enumerate(frames):
            write_frame(self.folder, i, f)
        write_meta(self.folder, frames)
        self.states = [dict(t=t, camera={}, objects=[dict(id=42, position=[0, 0, 1], state='OCCLUDED')], target_id=42)
                       for t in [10, 10.3]]

    def tearDown(self):
        self.tmp.cleanup()

    def test_sparse_states_match_timestamps_and_loop(self):
        track = RecordedTrack(self.folder, self.states)
        self.assertEqual(track.matches, [0, 2])
        self.assertEqual(track.state_index(.29), 0)
        self.assertEqual(track.state_index(.31), 1)
        self.assertEqual(track.state_index(track.duration + .01), 0)
        state = track.enrich(self.states[1], 1)
        self.assertEqual(state['replay']['frame'], 2)
        self.assertEqual(state['camera']['intrinsics']['width'], 64)
        self.assertEqual(self.states[1]['camera'], {})
        raw = track.image(state, 1, raw=True)
        self.assertEqual(raw, (self.folder / 'color/2.jpg').read_bytes())
        self.assertNotEqual(track.image(state, 1), raw)

    def test_bad_timing_and_missing_images_rejected(self):
        with self.assertRaisesRegex(ValueError, '50 ms'):
            RecordedTrack(self.folder, [dict(t=100)])
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            RecordedTrack(self.folder, [dict(t=10), dict(t=10)])
        (self.folder / 'color/1.jpg').unlink()
        with self.assertRaisesRegex(ValueError, 'frame count'):
            RecordedTrack(self.folder, self.states)
