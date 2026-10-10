"""Stand-in for python -m ground3d.ground (one shot) in the bench tests: the same command line, progress
lines and results.json, without the model. FAKE_MODE picks what happens:
  ok         every query answered
  oom_retry  out of memory with 15 frames, then fine with 10
  oom_fatal  out of memory for good during the scene (exit 1)
  crash      an exception after the model load (exit 1)
  hang       stops after the model load (for --timeout)
  nvmap      ok, with NvMap memory-pressure warnings
FAKE_SLEEP: seconds per phase (default 0.05).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

MODE = os.environ.get("FAKE_MODE", "ok")
NAP = float(os.environ.get("FAKE_SLEEP", "0.05"))
ANSWERS = {"the cup closest to the can": (0.605, [1.269, -0.437, 0.86]),
           "the cup next to the ball": (0.555, [0.57, -0.774, 0.854]),
           "nothing like this": (0.112, None)}


def band(s):
    return "high" if s >= 0.5 else "low" if s >= 0.3 else "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("-q", "--query", action="append", default=[])
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--out")
    ap.add_argument("--upstream")
    ap.add_argument("--models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(scan=args.scan, queries=args.query, torch="2.11.0", device="Orin", answers=[], retries=[], error=None,
               ram_gb=dict(idle=0.99, loaded=5.26, peak=6.5, total=7.43))
    print("torch 2.11.0, Orin, RAM in use 0.99 of 7.43 GB", flush=True)
    for i in range(1, 17):
        print(f"  weights {i}/16 ({6 * i} s): {0.3 * i:.2f} GB on the GPU", flush=True)
    time.sleep(NAP)
    print("loaded in 99 s, RAM in use 5.26 GB", flush=True)
    res["timing"] = {"import": 1.0, "load": 98.0}

    def finish(code):
        (out / "results.json").write_text(json.dumps(res, indent=2))
        print("\n=== GROUND3D SUMMARY (send me this) ===", flush=True)
        sys.exit(code)
    if MODE == "hang":
        time.sleep(3600)
    if MODE == "crash":
        res["error"] = "RuntimeError: CUDA error: unspecified launch failure"
        print("FAILED: " + res["error"], flush=True)
        finish(1)
    if MODE in ("nvmap", "oom_fatal"):
        for _ in range(3):
            print("NvMapMemAllocInternalTagged: 1075072515 error 12", file=sys.stderr, flush=True)
    frames = args.frames
    if MODE == "oom_fatal":
        res["error"] = "OUT OF MEMORY: OutOfMemoryError: CUDA out of memory"
        print("FAILED: " + res["error"], flush=True)
        finish(1)
    if MODE == "oom_retry":
        print(f"out of memory with {frames} frames (CUDA out of memory); retrying with 10", flush=True)
        res["retries"].append(dict(frames=frames, retry_frames=10, error="CUDA out of memory"))
        frames = 10
    time.sleep(NAP)
    res["timing"]["scene"] = {"load": 0.2, "processor": 0.5, "tokens": 0.2, "target": 1.3, "vit": 15.3, "total": 17.5}
    res.update(frames=frames, voxels=2826, target_points=53351)
    print(f"scene: {frames} frames, 23x17 tokens each -> 2826 voxels, 53351 target points; vit 15.3 s, total 17.5 s; "
          f"RAM in use 6.09 GB", flush=True)
    for n, q in enumerate(args.query):
        time.sleep(NAP)
        score, c = ANSWERS.get(q, (0.6 + 0.01 * n, [0.5, round(0.1 * n, 3), 0.8]))
        r = dict(query=q, target=q.split()[1] if len(q.split()) > 1 else q, score=score, band=band(score),
                 top5=[score, 0.2, 0.1, 0.05, 0.01], n_points=188 if c else 0, center=c, size=[0.1, 0.1, 0.12] if c else None,
                 seconds=5.6 if n == 0 else 5.19, llm_seconds=3.9, tokens=2860, gpu_peak_gb=5.2)
        res["answers"].append(r)
        print(f"  '{q}' (target '{r['target']}'): score {score:.3f} ({band(score)}), {r['n_points']} points"
              + (f" at {c}" if c else "") + f", {r['seconds']:.2f} s", flush=True)
    finish(0)


if __name__ == "__main__":
    main()
