"""python3 -m unittest discover -s tests   (from the repo root; no Jetson needed)"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "core" / "jetson"))
import jetson_stats as js  # noqa: E402

# tegrastats lines in the formats of JetPack 6 (Orin Nano) and JetPack 5 (GR3D with frequencies)
JP6 = ("10-10-2026 01:23:45 RAM 1882/7620MB (lfb 2x4MB) SWAP 0/3810MB (cached 0MB) "
       "CPU [1%@729,0%@729,3%@729,0%@729,off,off] GR3D_FREQ 45% cpu@46.468C soc2@45.312C soc0@46.031C "
       "gpu@46.625C tj@47.25C soc1@45.656C VDD_IN 4405mW/4405mW VDD_CPU_GPU_CV 480mW/480mW VDD_SOC 1281mW/1281mW")
JP5 = ("RAM 3245/7620MB (lfb 12x4MB) SWAP 12/3810MB (cached 0MB) CPU [12%@1510,5%@1510,8%@1510,4%@1510] "
       "EMC_FREQ 3%@2133 GR3D_FREQ 87%@[1020,1020] CPU@51.2C GPU@50.1C PMIC@100C AO@48C thermal@50.6C "
       "VDD_IN 9876mW/8123mW VDD_CPU_GPU_CV 3456mW/3000mW")
NO_RAILS = "RAM 900/7620MB SWAP 0/3810MB CPU [2%@729,1%@729] GR3D_FREQ 0% tj@-256C cpu@40C"


class TestParse(unittest.TestCase):
    def test_jetpack6(self):
        s = js.parse_tegrastats(JP6)
        self.assertEqual((s["ram_used_mb"], s["ram_total_mb"], s["swap_used_mb"]), (1882, 7620, 0))
        self.assertEqual(s["cpu_pct"], 1.0)                 # the two "off" cores don't count
        self.assertEqual(s["gpu_pct"], 45)
        self.assertEqual(s["temp_c"], 47.25)                # tj
        self.assertEqual(s["power_w"], 4.41)                # VDD_IN, the whole board
        self.assertEqual(s["rails_mw"]["VDD_SOC"], 1281)

    def test_jetpack5(self):
        s = js.parse_tegrastats(JP5)
        self.assertEqual(s["gpu_pct"], 87)
        self.assertEqual(s["power_w"], 9.88)                # the current value, not the average
        self.assertNotIn("pmic", s["temps_c"])              # PMIC reads a fixed 100 C
        self.assertEqual(s["temp_c"], 51.2)                 # no tj: the hottest sensor
        self.assertEqual(s["cpu_pct"], 7.2)

    def test_missing_fields(self):
        s = js.parse_tegrastats(NO_RAILS)
        self.assertNotIn("power_w", s)
        self.assertEqual(s["temp_c"], 40.0)                 # tj@-256C: sensor off, ignored
        self.assertEqual(js.parse_tegrastats("garbage"), {})


class TestSampler(unittest.TestCase):
    def test_phases(self):
        s = js.Sampler(interval=0.02, tegrastats=False)
        s.mark("a")
        time.sleep(0.15)
        s.mark("b")
        time.sleep(0.1)
        s.stop()
        segs = {g["phase"]: g for g in s.segments()}
        self.assertIn("a", segs)
        self.assertIn("b", segs)
        self.assertGreater(segs["a"]["samples"], 2)
        self.assertGreater(segs["a"]["ram_peak_gb"], 0)
        self.assertIsNone(segs["a"]["power_mean_w"])        # no tegrastats here
        self.assertIsNone(segs["a"]["energy_j"])
        self.assertFalse(s.has_tegrastats)
        self.assertIn("RAM", js.line(s.latest()))


if __name__ == "__main__":
    unittest.main()
