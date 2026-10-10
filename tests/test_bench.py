"""python3 -m unittest discover -s tests   (from the repo root; no Jetson needed: a stand-in runner
answers in the real runner's output format)"""
import ast
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core" / "qwen3d_jetson"))
import bench  # noqa: E402

FAKE = Path(__file__).resolve().parent / "fake_runner" / "jetson_init.py"


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class TestParse(unittest.TestCase):
    def fold(self, *lines):
        res = dict(queries=[], oom_retries=[], oom_fatal=None, nvmap_warnings=0, error=None)
        marks = []
        for i, ln in enumerate(lines):
            bench.parse_line(ln, float(i), res, {}, marks.append)
        return res, marks

    def test_runner_lines(self):
        res, marks = self.fold(
            "torch 2.11.0, Orin, RAM in use 0.85 of 7.43 GB",
            "model loaded in 93 s, RAM in use 4.90 GB",
            "out of memory during the vision encoder with 15 frames (RAM peak 7.10 GB); retrying with 10 frames. RAM now: x",
            "scene: 10 frames -> 3123 point tokens, 4567 2 cm voxels; prep 18.4 s (load 0.5, 3d 0.3, preprocess 0.8, "
            "tokens 0.2, dense 1.0, vit 15.6), RAM in use 6.10 GB",
            "  'the cup next to the can' (target 'cup'): score 0.461 (low), 1234 points at [1.268, -0.442, 0.862], 4.60 s"
            " | laptop 0.470 (low), centre 0.8 cm apart  MATCH",
            "  'the kid's cup' (target 'kid's cup'): score 0.112 (none), 0 points, 4.20 s | laptop 0.100 (none)  MATCH (absent on both)",
            "  'the cup next to the ball' (target 'cup'): score 0.548 (high), 800 points at [0.95, -0.3, 0.86], 4.21 s"
            " | laptop 0.600 (high), centre 5.0 cm apart  DIFFERENT",
            "=== SUMMARY (send me this) ===",
            "device: Orin, torch 2.11.0, w8 + vit8 + emb8, ViT unloaded after the scan",
            "RAM in use: idle 0.85 GB, peak 6.40 GB of 7.43 GB",
            "'the cup next to the can': score 0.461 (low), 4.60 s/query, centre [1.268, -0.442, 0.862]",
            "NvMapMemAllocInternalTagged: 1075072515 error 12")
        self.assertEqual(marks, ["model load", "scene prep", "queries", "exit"])
        self.assertEqual((res["torch"], res["device"], res["ram_total_gb"]), ("2.11.0", "Orin", 7.43))
        self.assertEqual(res["model_load_s"], 93)
        self.assertEqual(res["oom_retries"][0]["retry_frames"], 10)
        self.assertEqual((res["frames_used"], res["point_tokens"], res["voxels"], res["scene_prep_s"]), (10, 3123, 4567, 18.4))
        self.assertEqual(res["scene_parts"]["vit"], 15.6)
        self.assertEqual(len(res["queries"]), 3)            # the SUMMARY's answer lines aren't counted twice
        q0, q1, q2 = res["queries"]
        self.assertEqual(q0["center"], [1.268, -0.442, 0.862])
        self.assertTrue(q0["laptop_match"])
        self.assertEqual((q1["query"], q1["center"], q1["laptop_match"]), ("the kid's cup", None, True))
        self.assertFalse(q2["laptop_match"])
        self.assertEqual(res["ram_peak_gb"], 6.40)
        self.assertEqual(res["nvmap_warnings"], 1)

    def test_fatal_oom(self):
        res, _ = self.fold("", "OUT OF MEMORY during the queries: RAM peak 7.31 / 7.43 GB (GPU: ...)")
        self.assertEqual(res["oom_fatal"], "the queries")

    def test_check_answer(self):
        q = dict(score=0.47, center=[1.27, -0.44, 0.86])
        c = bench.check_answer(q, {"score": 0.461, "center": [1.268, -0.442, 0.862]})
        self.assertTrue(c["ok"] and c["band"] and c["center"])
        self.assertLess(c["center_cm"], 1)
        self.assertFalse(bench.check_answer(q, {"score": 0.55})["ok"])                 # high expected, low given
        self.assertFalse(bench.check_answer(q, {"center": [1.5, -0.44, 0.86]})["ok"])  # 23 cm away
        self.assertTrue(bench.check_answer(q, {"center": [1.3, -0.44, 0.86], "tol_cm": 5})["ok"])
        self.assertFalse(bench.check_answer(dict(score=0.6, center=None), {"center": [0, 0, 0]})["ok"])
        self.assertFalse(bench.check_answer(q, {"min_score": 0.5})["ok"])
        self.assertIsNone(bench.check_answer(q, None))

    def test_slug_and_band(self):
        self.assertEqual(bench.slug("The cup, next to the can!"), "the_cup_next_to_the_can")
        self.assertEqual([bench.band(s) for s in (0.5, 0.49, 0.3, 0.29)], ["high", "low", "low", "none"])


class TestRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="qbench_"))
        self.runner = self.tmp / "runner"
        self.runner.mkdir()
        shutil.copy(FAKE, self.runner / "jetson_init.py")
        for f in ("run_test.py", "scene_prep.py", "testpack.pt", "heads.pth"):
            (self.runner / f).write_text("")
        (self.runner / "qwen").mkdir()
        self.scan = self.tmp / "live_183221" / "scan"
        for d in bench.SCAN_NEEDS:
            (self.scan / d).mkdir(parents=True)
        (self.scan / "results").mkdir()                  # the laptop's answer for one query
        (self.scan / "results" / "the_cup_next_to_the_can.json").write_text(
            json.dumps(dict(score=0.47, center_scene_m=[1.27, -0.44, 0.86])))
        self.cases = self.tmp / "cases.json"
        self.cases.write_text(json.dumps({"cases": [{
            "name": "thursday", "scan": str(self.scan),
            "queries": ["the cup next to the can", "the cup next to the ball", "nothing like this"],
            "expect": {"the cup next to the can": {"score": 0.461, "center": [1.268, -0.442, 0.862]},
                       "the cup next to the ball": {"score": 0.548}}}]}))
        self.out = self.tmp / "out"
        self.env = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.env)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def bench(self, *extra, mode="ok"):
        os.environ["FAKE_MODE"] = mode
        os.environ["FAKE_SLEEP"] = "0.05"
        argv = ["--runner", str(self.runner), "--python", sys.executable, "--out", str(self.out),
                "--interval", "0.02", "--settle", "1", "--force", *extra]
        rc, text = quiet(bench.main, argv)
        doc = json.loads((self.out / "results.json").read_text()) if (self.out / "results.json").exists() else None
        return rc, text, doc

    def test_cases_file(self):
        rc, text, doc = self.bench("--cases", str(self.cases))
        self.assertEqual(rc, 0, text)
        r, = doc["runs"]
        self.assertTrue(r["checks"]["finished"])
        self.assertEqual([q["score"] for q in r["queries"]], [0.461, 0.548, 0.112])
        self.assertEqual(r["init_s"], round(18.4 + 4.6, 1))
        self.assertTrue(r["checks"]["init_within_budget"])
        self.assertFalse(r["checks"]["first_query_high"])         # 0.461 is "low"
        self.assertEqual(r["checks"]["expected_answers"], [True, True])
        self.assertTrue(r["queries"][0]["laptop_match"])
        self.assertEqual(r["queries"][0]["top5_scores"][0], 0.461)  # from the runner's answer file
        self.assertIsNone(r["queries"][2]["center"])
        self.assertEqual((r["model_load_s"], r["ram_peak_gb"], r["frames_used"]), (93, 6.40, 15))
        self.assertIn("model load", [p["phase"] for p in r["phases"]])
        self.assertFalse((self.scan / "results_jetson").exists())  # the scan itself is left alone
        self.assertTrue((self.out / r["id"] / "scan" / "results_jetson" / "target.json").exists())
        for f in ("summary.md", f"{r['id']}.log", f"{r['id']}.samples.jsonl"):
            self.assertTrue((self.out / f).exists(), f)
        self.assertIn("=== BENCH SUMMARY", text)
        self.assertIn("first query >= 0.5 FAIL (0.461)", text)
        self.assertNotIn("weights  3/16", text)                   # one weights line, not sixteen

    def test_scans_and_sweep(self):
        rc, text, doc = self.bench(str(self.scan), "-q", "the cup next to the ball", "--frames", "15", "10")
        self.assertEqual(rc, 0, text)
        self.assertEqual([r["id"] for r in doc["runs"]], ["live_183221-f15-w8", "live_183221-f10-w8"])
        self.assertTrue(all(r["checks"]["first_query_high"] for r in doc["runs"]))
        self.assertIn("--frames 10", " ".join(doc["runs"][1]["command"]))

    def test_relative_out(self):
        """--out relative to where bench.py is started, while the runner starts in its own folder."""
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.out = Path("out")
            rc, text, doc = self.bench("--cases", str(self.cases))
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0, text)
        self.assertTrue(doc["runs"][0]["checks"]["finished"])
        self.assertTrue((self.tmp / "out" / "results.json").exists())

    def test_oom_retry(self):
        rc, _, doc = self.bench("--cases", str(self.cases), mode="oom_retry")
        r = doc["runs"][0]
        self.assertEqual(rc, 0)
        self.assertEqual((len(r["oom_retries"]), r["frames_used"]), (1, 10))
        self.assertFalse(r["checks"]["no_out_of_memory"])
        self.assertTrue(r["checks"]["finished"])

    def test_oom_fatal(self):
        rc, text, doc = self.bench("--cases", str(self.cases), mode="oom_fatal")
        r = doc["runs"][0]
        self.assertEqual(rc, 1)
        self.assertEqual((r["exit_code"], r["oom_fatal"]), (1, "the vision encoder"))
        self.assertIn("OUT OF MEMORY", r["error"])
        self.assertGreaterEqual(r["nvmap_warnings"], 1)
        self.assertFalse(r["checks"]["finished"])

    def test_crash(self):
        rc, _, doc = self.bench("--cases", str(self.cases), mode="crash")
        r = doc["runs"][0]
        self.assertEqual(rc, 1)
        self.assertIn("RuntimeError: CUDA error", r["error"])

    def test_timeout(self):
        t = time.monotonic()
        rc, _, doc = self.bench("--cases", str(self.cases), "--timeout", "2", mode="hang")
        r = doc["runs"][0]
        self.assertLess(time.monotonic() - t, 30)
        self.assertEqual(rc, 1)
        self.assertTrue(r["timed_out"])
        self.assertEqual(bench.heavy_processes(), [])            # the hung runner was stopped

    def test_dry_run_and_preflight(self):
        rc, text, doc = self.bench("--cases", str(self.cases), "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIsNone(doc)
        self.assertFalse(self.out.exists())
        self.assertIn("jetson_init.py", text)
        (self.runner / "heads.pth").unlink()
        rc, text, _ = self.bench("--cases", str(self.cases))
        self.assertEqual(rc, 2)
        self.assertIn("heads.pth", text)

    def test_bad_inputs(self):
        with self.assertRaises(SystemExit):
            quiet(bench.main, ["--runner", str(self.runner), str(self.scan)])         # a scan without -q
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"cases": [{"scan": str(self.scan), "queries": ["a"], "expect": {"b": {}}}]}))
        with self.assertRaises(SystemExit):
            quiet(bench.main, ["--runner", str(self.runner), "--cases", str(bad)])
        rc, text = quiet(bench.main, ["--runner", str(self.runner), "--force", str(self.tmp), "-q", "x", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("not a scan folder", text)

    def test_heavy_process_detected(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "jetson_init.py"])
        try:
            time.sleep(0.3)
            self.assertIn(p.pid, [pid for pid, _ in bench.heavy_processes()])
            problems, _, _ = bench.preflight(bench.argparse.Namespace(runner=self.runner))
            self.assertTrue(any("another heavy process" in x for x in problems))
        finally:
            p.kill()
            p.wait()


class TestPython310(unittest.TestCase):
    def test_syntax(self):
        """The Jetson (JetPack 6, Ubuntu 22.04) runs Python 3.10."""
        for f in list((ROOT / "core").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")):
            ast.parse(f.read_text(), str(f), feature_version=(3, 10))


if __name__ == "__main__":
    unittest.main()
