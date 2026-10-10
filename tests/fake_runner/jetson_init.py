"""Stand-in for the Qwen-3D runner (jetson_init.py, one-shot mode) in bench.py's tests: the same command
line, output lines and answer files, without the model. The output lines use the real runner's
format strings. FAKE_MODE picks what happens:
  ok         every query answered
  oom_retry  out of memory with 15 frames, then fine with 10 (the runner's own retry)
  oom_fatal  out of memory for good during the scene (exit 1)
  crash      an exception after the model load (exit 1, traceback)
  hang       stops answering after the model load (for --timeout)
  nvmap      ok, with NvMap memory-pressure warnings on stderr
FAKE_SLEEP: seconds per phase (default 0.05).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_MODE", "ok")
NAP = float(os.environ.get("FAKE_SLEEP", "0.05"))
TOTAL = 7.43
ANSWERS = {                                   # query -> (score, centre or None)
    "the cup next to the can": (0.461, [1.268, -0.442, 0.862]),
    "the cup next to the ball": (0.548, [0.951, -0.301, 0.858]),
    "nothing like this": (0.112, None),
}


def slug(q):
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:40]


def band(s):
    return "high" if s >= 0.5 else "low" if s >= 0.3 else "none"


def meminfo():
    return "used 4.90 GB, free 1.20 GB, file cache 0.80 GB (dirty 0 MB), swap used 0.00 GB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="?")
    ap.add_argument("queries", nargs="*")
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--max-depth", type=float, default=3.0)
    ap.add_argument("--quant", choices=["w8", "w4"], default="w8")
    args = ap.parse_args()
    scan = Path(args.scan).expanduser().resolve()
    out = scan / "results_jetson"
    out.mkdir(exist_ok=True)
    time.sleep(NAP)
    print(f"torch 2.11.0, Orin, RAM in use {0.85:.2f} of {TOTAL:.2f} GB")
    for i in range(1, 17):
        print(f"  weights {i:2d}/{16}: {0.3 * i:.2f} GB on GPU, RAM in use {0.85 + 0.25 * i:.2f} / {TOTAL:.2f} GB", flush=True)
    time.sleep(NAP)
    print(f"model loaded in {93.4:.0f} s, RAM in use {4.90:.2f} GB")
    print(f"RAM before the scene: {meminfo()}", flush=True)
    if MODE == "hang":
        time.sleep(3600)
    if MODE == "crash":
        raise RuntimeError("CUDA error: unspecified launch failure")
    if MODE == "nvmap":
        for _ in range(3):
            print("NvMapMemAllocInternalTagged: 1075072515 error 12", file=sys.stderr, flush=True)
    frames = args.frames
    if MODE in ("oom_retry", "oom_fatal"):
        time.sleep(NAP)
        if MODE == "oom_fatal":
            print("NvMapMemAllocInternalTagged: 1075072515 error 12", file=sys.stderr, flush=True)
            sys.exit(f"\nOUT OF MEMORY during the vision encoder: RAM peak {7.31:.2f} / {TOTAL:.2f} GB (GPU: 5.10 GB "
                     f"allocated, 5.60 GB reserved; RAM {meminfo()})"
                     "\nReboot, keep the desktop off and nothing else running; or try --frames 10, or --quant w4.")
        print(f"out of memory during the vision encoder with {frames} frames (RAM peak {7.10:.2f} GB); "
              f"retrying with {10} frames. RAM now: {meminfo()}", flush=True)
        frames = 10
    time.sleep(NAP)
    prep = {"load": 0.5, "3d": 0.3, "preprocess": 0.8, "tokens": 0.2, "dense": 1.0, "vit": 15.6}
    print(f"scene: {frames} frames -> {3123} point tokens, {4567} 2 cm voxels; "
          f"prep {18.4:.1f} s (" + ", ".join(f"{k} {v:.1f}" for k, v in prep.items())
          + f"), RAM in use {6.10:.2f} GB", flush=True)
    results = []
    for n, q in enumerate(args.queries):
        time.sleep(NAP)
        score, centre = ANSWERS.get(q, (0.6 + 0.01 * n, [0.5, round(0.1 * n, 3), 0.8]))
        s = dict(query=q, target=q.split()[1] if len(q.split()) > 1 else q, found=score >= 0.3, confidence=band(score),
                 score=round(score, 4), top5_scores=[round(score, 4), 0.2, 0.1, 0.05, 0.01], n_points=900 if centre else 0,
                 seconds=4.6 if n == 0 else 4.2, llm_seconds=3.1, peak_gpu_gb=5.2, llm_tokens=3200)
        if centre:
            s.update(center_scene_m=centre, size_xyz_m=[0.1, 0.1, 0.12])
        json.dump(s, open(out / f"{slug(q)}.json", "w"), indent=2)
        if not results:
            json.dump(dict(s, scan=str(scan)), open(out / "target.json", "w"), indent=2)
        cmp_ = None
        ref_f = scan / "results" / f"{slug(q)}.json"
        if ref_f.exists():
            ref = json.load(open(ref_f))
            line = f"laptop {ref['score']:.3f} ({band(ref['score'])})"
            if "center_scene_m" in s and "center_scene_m" in ref:
                d = sum((a - b) ** 2 for a, b in zip(s["center_scene_m"], ref["center_scene_m"])) ** 0.5
                ok = band(s["score"]) == band(ref["score"]) and d < 0.03
                cmp_ = (line + f", centre {100 * d:.1f} cm apart  {'MATCH' if ok else 'DIFFERENT'}", ok)
            else:
                cmp_ = (line + "  DIFFERENT", False)
        results.append((s, cmp_))
        print(f"  '{q}' (target '{s['target']}'): score {s['score']:.3f} ({s['confidence']}), {s['n_points']} points"
              + (f" at {s['center_scene_m']}" if "center_scene_m" in s else "")
              + f", {s['seconds']:.2f} s" + (f" | {cmp_[0]}" if cmp_ else ""), flush=True)
    print("\n=== SUMMARY (send me this) ===")
    print(f"device: Orin, torch 2.11.0, {args.quant} + vit8 + emb8, ViT unloaded after the scan")
    print(f"scan: {scan} ({frames} of the keyframes, 3123 point tokens, 4567 2 cm voxels)")
    print(f"time: model load {93.4:.0f} s | scene prep {18.4:.1f} s (" + ", ".join(f"{k} {v:.1f}" for k, v in prep.items()) + ")")
    print(f"RAM in use: idle {0.85:.2f} GB, peak {6.40:.2f} GB of {TOTAL:.2f} GB")
    for s, c in results:
        print(f"'{s['query']}': score {s['score']:.3f} ({s['confidence']}), {s['seconds']:.2f} s/query"
              + (f", centre {s['center_scene_m']}" if "center_scene_m" in s else "") + (f" | {c[0]}" if c else ""))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
