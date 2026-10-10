"""Which object did each Qwen-3D answer pick? Draws every answer onto the scan's keyframes.

    python -m ground3d.show <scan> <ground3d output folder> [--views 4] [--raw] [--out runs/ground3d_show]

The output folder is what python -m ground3d.ground --out wrote. Each answer is drawn from its refined cloud
(<slug>_refined.ply, from ground3d.refine) when there is one, else from the raw answer (--raw: always the raw
one), in the x-ray style of xray.render: one colour family for all objects (near pale cyan, far blue,
brighter where the surface faces the camera; hidden points faint). A number at each object matches the
legend (score and query). Writes view_<k>.jpg for the keyframes that see the first answer best.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray import frames as fr, render  # noqa: E402


def slug(q):
    """The answer's file name, as ground3d.ground writes it."""
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:48] or "query"


def read_ply(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index("end_header") + 1
    return np.array([[float(x) for x in ln.split()[:3]] for ln in lines[start:] if ln.strip()]).reshape(-1, 3)


def answer_points(res_dir, a, raw=False):
    """(points, which) for an answer: its refined cloud if there is one, else the raw answer."""
    refined = res_dir / f"{slug(a['query'])}_refined.ply"
    plain = res_dir / f"{slug(a['query'])}.ply"
    if refined.exists() and not raw:
        return read_ply(refined), "refined"
    if plain.exists():
        return read_ply(plain), "raw"
    return (np.array([a["center"]]) if a.get("center") else np.zeros((0, 3))), "centre"


def text(img, s, xy, scale=0.45, col=(255, 255, 255)):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw Qwen-3D answers onto the scan (see the top of this file)")
    ap.add_argument("scan")
    ap.add_argument("results", help="the folder python -m ground3d.ground --out wrote")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--raw", action="store_true", help="draw the raw answers even where a refined cloud exists")
    ap.add_argument("--out", default="runs/ground3d_show")
    args = ap.parse_args(argv)
    res_dir = Path(args.results).expanduser()
    res = json.loads((res_dir / "results.json").read_text())
    answers = [(a, *answer_points(res_dir, a, args.raw)) for a in res.get("answers", [])]
    if not answers:
        raise SystemExit(f"no answers in {res_dir / 'results.json'}")
    ids = fr.frame_ids(args.scan)
    first = next((pts for _, pts, _ in answers if len(pts)), None)
    scored = []
    for k in ids:
        rgb, depth, conf, K, T = fr.read_frame(args.scan, k)
        n = 0 if first is None else render.draw_cloud(np.zeros_like(rgb), first, K, T, depth)["drawn"]
        scored.append((n, k))
    pick = sorted(k for n, k in sorted(scored, reverse=True)[:args.views])
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for k in pick:
        rgb, depth, conf, K, T = fr.read_frame(args.scan, k)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tags = []
        for i, (a, pts, which) in enumerate(answers, 1):
            info = render.draw_cloud(img, pts, K, T, depth, centre=a.get("refined", {}).get("center") or a.get("center"))
            if info["label_xy"]:
                tags.append((i, info["label_xy"]))
        for i, (x, y) in tags:
            text(img, str(i), (max(0, x - 4), max(12, y - 6)), 0.5)
        y = 18
        for i, (a, pts, which) in enumerate(answers, 1):
            text(img, f"{i}  {a['score']:.2f} {a['query']}  ({len(pts)} {which})", (8, y))
            y += 18
        cv2.imwrite(str(out / f"view_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(pick)} views -> {out}/ (" + ", ".join(f"view_{k}.jpg" for k in pick) + ")")
    for i, (a, pts, which) in enumerate(answers, 1):
        print(f"  {i}  {a['score']:.3f} ({a['band']}) '{a['query']}': {len(pts)} {which} points"
              + (f" at {np.round(np.asarray(a['center']), 3).tolist()}" if a.get("center") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
