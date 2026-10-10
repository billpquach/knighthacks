"""Viewer contract and replay server HTTP checks, without Jetson hardware."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np

import replay_server as server


class ReplayViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.http.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def test_demo_transform_and_occlusion(self):
        state = server.synthetic(60)
        matrix = np.asarray(state["camera"]["T_world_cam"]).reshape(3, 4)
        np.testing.assert_allclose(matrix[:, 2], state["camera"]["forward"])
        np.testing.assert_allclose(matrix[:, :3].T @ matrix[:, :3], np.eye(3), atol=1e-10)
        self.assertEqual(state["objects"][2]["state"], "OCCLUDED")
        self.assertEqual(state["ball"]["state"], "INSIDE")
        self.assertEqual(state["ball"]["inside"], 1)
        self.assertEqual(len(state["occluders"]), 2)

    def test_viewer_info_and_image(self):
        with urlopen(self.base + "/viewer-info") as response:
            self.assertEqual(json.load(response)["mode"], "synthetic")
        with urlopen(self.base + "/frame.jpg?x=1") as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            self.assertIn("X-Seq", response.headers["Access-Control-Expose-Headers"])
            self.assertTrue(response.read().startswith(b"\xff\xd8"))

    def test_stream_and_unknown_routes(self):
        with urlopen(self.base + "/state", timeout=3) as response:
            line = response.readline().decode()
            self.assertTrue(line.startswith("data: "))
            self.assertEqual(json.loads(line[6:])["source"], "synthetic")
        for path in ["/state-wrong", "/%2e%2e/replay_server.py"]:
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + path)
            self.assertEqual(error.exception.code, 404)

    def test_state_json_and_cors_everywhere(self):
        with urlopen(self.base + "/state.json") as response:
            self.assertEqual(response.headers["Content-Type"], "application/json")
            self.assertEqual(response.headers.get_all("Access-Control-Allow-Origin"), ["*"])
            self.assertIn("objects", json.load(response))
        for path in ["/viewer-info", "/frame.jpg"]:
            with urlopen(self.base + path) as response:
                self.assertEqual(response.headers.get_all("Access-Control-Allow-Origin"), ["*"])
        with self.assertRaises(HTTPError) as error:
            urlopen(self.base + "/no-such-page")
        self.assertEqual(error.exception.headers.get_all("Access-Control-Allow-Origin"), ["*"])
        with urlopen(self.base + "/state", timeout=3) as response:
            self.assertEqual(response.headers.get_all("Access-Control-Allow-Origin"), ["*"])

    def test_recorded_states_are_not_paired_with_fake_camera(self):
        original = server.STATES
        try:
            server.STATES = [{"objects": [], "t": 1}]
            self.assertEqual(server.state_at(0)["source"], "replay")
            self.assertNotIn("source", server.STATES[0])
            with self.assertRaises(HTTPError) as error:
                urlopen(self.base + "/frame.jpg")
            self.assertEqual(error.exception.code, 404)
        finally:
            server.STATES = original


if __name__ == "__main__":
    unittest.main()
