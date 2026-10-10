"""python3 -m unittest discover -s tests   (no Jetson needed: a stand-in runner, tests/fake_core/ground3d,
answers with the real runner's command line, progress lines and results.json)"""
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
sys.path.insert(0, str(ROOT / "core"))
from ground3d import bench  # noqa: E402

FAKE_CORE = Path(__file__).resolve().parent / "fake_core"


def quiet(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a)
    return rc, buf.getvalue()


class TestPieces(unittest.TestCase):
    def test_phases(self):
        self.assertEqual([bench.phase_of(x) for x in ("torch 2.11.0, Orin, RAM in use 1.00 of 7.43 GB", "loaded in 99 s, RAM",
                                                      "scene: 15 frames, ...", "=== GROUND3D SUMMARY (send me this) ===",
                                                      "  'q' (target 'cup'): score")],
                         ["model load", "scene prep", "queries", "exit", None])

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
        self.assertEqual([bench.band(s) for s in (0.5, 0.49, 0.3, 0.29)], ["high", "low", "low", "none"])


class TestRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gbench_"))
        self.up = self.tmp / "qwen3d"
        (self.up / "qwen3d").mkdir(parents=True)
        (self.up / "qwen3d" / "modeling_qwen2_5_vl_modified.py").write_text("")
        self.models = self.tmp / "models"
        (self.models / "qwen").mkdir(parents=True)
        (self.models / "qwen" / "config.json").write_text("{}")
        (self.models / "heads.pth").write_text("")
        (self.models / "processor").mkdir()
        self.scan = self.tmp / "scan_act1" / "scan"
        for d in bench.SCAN_NEEDS:
            (self.scan / d).mkdir(parents=True)
        self.cases = self.tmp / "cases.json"
        self.cases.write_text(json.dumps({"cases": [{
            "name": "act1", "scan": str(self.scan),
            "queries": ["the cup closest to the can", "the cup next to the ball", "nothing like this"],
            "expect": {"the cup closest to the can": {"score": 0.609, "center": [1.269, -0.442, 0.859]},
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
        argv = ["--core", str(FAKE_CORE), "--upstream", str(self.up), "--models", str(self.models), "--python", sys.executable,
                "--out", str(self.out), "--interval", "0.02", "--settle", "1", "--force", *extra]
        rc, text = quiet(bench.main, argv)
        doc = json.loads((self.out / "results.json").read_text()) if (self.out / "results.json").exists() else None
        return rc, text, doc

    def test_cases_file(self):
        rc, text, doc = self.bench("--cases", str(self.cases))
        self.assertEqual(rc, 0, text)
        r, = doc["runs"]
        self.assertTrue(r["checks"]["finished"])
        self.assertEqual([q["score"] for q in r["answers"]], [0.605, 0.555, 0.112])
        self.assertEqual(r["init_s"], round(17.5 + 5.6, 1))
        self.assertTrue(r["checks"]["init_within_budget"])
        self.assertTrue(r["checks"]["first_query_high"])
        self.assertEqual(r["checks"]["expected_answers"], [True, True])
        self.assertEqual((r["timing"]["load"], r["ram_gb"]["peak"], r["frames"]), (98.0, 6.5, 15))
        self.assertIn("model load", [p["phase"] for p in r["phases"]])
        for f in ("summary.md", f"{r['id']}.log", f"{r['id']}.samples.jsonl", f"{r['id']}/results.json"):
            self.assertTrue((self.out / f).exists(), f)
        self.assertIn("=== BENCH SUMMARY", text)
        self.assertNotIn("weights 3/16", text)                    # one weights line, not sixteen
        self.assertIn("weights 16/16", text)

    def test_relative_out_and_sweep(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            self.out = Path("out")
            rc, text, doc = self.bench(str(self.scan), "-q", "the cup next to the ball", "--frames", "15", "10")
        finally:
            os.chdir(cwd)
        self.assertEqual(rc, 0, text)
        self.assertEqual([r["id"] for r in doc["runs"]], ["scan_act1-f15", "scan_act1-f10"])
        self.assertIn("--frames 10", " ".join(doc["runs"][1]["command"]))
        self.assertTrue((self.tmp / "out" / "scan_act1-f10" / "results.json").exists())

    def test_update_cases(self):
        rc, text, _ = self.bench("--cases", str(self.cases), "--update-cases")
        self.assertEqual(rc, 0, text)
        exp = json.loads(self.cases.read_text())["cases"][0]["expect"]
        self.assertEqual(exp["the cup closest to the can"], {"score": 0.605, "center": [1.269, -0.437, 0.86]})
        self.assertEqual(exp["nothing like this"], {"score": 0.112})

    def test_oom_retry(self):
        rc, _, doc = self.bench("--cases", str(self.cases), mode="oom_retry")
        r = doc["runs"][0]
        self.assertEqual(rc, 0)
        self.assertEqual((len(r["retries"]), r["frames"]), (1, 10))
        self.assertFalse(r["checks"]["no_out_of_memory"])
        self.assertTrue(r["checks"]["finished"])

    def test_oom_fatal(self):
        rc, text, doc = self.bench("--cases", str(self.cases), mode="oom_fatal")
        r = doc["runs"][0]
        self.assertEqual((rc, r["exit_code"]), (1, 1))
        self.assertIn("OUT OF MEMORY", r["error"])
        self.assertFalse(r["checks"]["no_out_of_memory"])
        self.assertGreaterEqual(r["nvmap_warnings"], 1)
        self.assertIn("fatal", text)

    def test_crash(self):
        rc, _, doc = self.bench("--cases", str(self.cases), mode="crash")
        self.assertEqual(rc, 1)
        self.assertIn("RuntimeError: CUDA error", doc["runs"][0]["error"])

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
        self.assertIn("ground3d.ground", text)
        (self.models / "heads.pth").unlink()
        rc, text, _ = self.bench("--cases", str(self.cases))
        self.assertEqual(rc, 2)
        self.assertIn("heads.pth", text)

    def test_bad_inputs(self):
        base = ["--core", str(FAKE_CORE), "--upstream", str(self.up), "--models", str(self.models)]
        with self.assertRaises(SystemExit):
            quiet(bench.main, base + [str(self.scan)])                       # a scan without -q
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"cases": [{"scan": str(self.scan), "queries": ["a"], "expect": {"b": {}}}]}))
        with self.assertRaises(SystemExit):
            quiet(bench.main, base + ["--cases", str(bad)])
        rc, text = quiet(bench.main, base + ["--force", str(self.tmp), "-q", "x", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("not a scan folder", text)

    def test_heavy_process_detected(self):
        p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)", "-m", "ground3d.ground"])
        try:
            time.sleep(0.3)
            self.assertIn(p.pid, [pid for pid, _ in bench.heavy_processes()])
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
