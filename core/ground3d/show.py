"""Which object did each Qwen-3D answer pick? Draws every answer's 3D points onto the scan's keyframes.

    python -m ground3d.show <scan> <ground3d output folder> [--views 4] [--out runs/ground3d_show]

The output folder is what python -m ground3d.ground --out wrote (results.json and one .ply per answer).
Points are drawn where they are in view; points hidden behind something nearer are drawn faint. One colour
per answer, with its query and score in the legend. Writes view_<k>.jpg for the keyframes that see the
first answer best.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray import frames as fr  # noqa: E402

COLORS = [(0, 220, 255), (255, 80, 200), (80, 255, 120), (255, 170, 0), (120, 140, 255), (255, 255, 80), (0, 140, 255),
          (200, 120, 255)]                                  # BGR


def slug(q):
    """The answer's file name, as ground3d.ground writes it."""
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:48] or "query"


def read_ply(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index("end_header") + 1
    return np.array([[float(x) for x in ln.split()[:3]] for ln in lines[start:] if ln.strip()]).reshape(-1, 3)


def visible(pts, depth, K, T, tol=0.04):
    """Pixel coordinates of the points in the image, and whether each is in front of what the depth sees."""
    u, v, z = fr.project(pts, K, T)
    h, w = depth.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (z > 0.05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    front = np.zeros(len(pts), bool)
    d = depth[vi[inside], ui[inside]]
    front[inside] = (d <= 0) | (z[inside] <= d + tol)
    return ui, vi, inside, front


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw Qwen-3D answers onto the scan (see the top of this file)")
    ap.add_argument("scan")
    ap.add_argument("results", help="the folder python -m ground3d.ground --out wrote")
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--out", default="runs/ground3d_show")
    args = ap.parse_args(argv)
    res_dir = Path(args.results).expanduser()
    res = json.loads((res_dir / "results.json").read_text())
    answers = []
    for i, a in enumerate(res.get("answers", [])):
        ply = res_dir / f"{slug(a['query'])}.ply"
        pts = read_ply(ply) if ply.exists() else (np.array([a["center"]]) if a.get("center") else np.zeros((0, 3)))
        answers.append((a, pts, COLORS[i % len(COLORS)]))
    if not answers:
        raise SystemExit(f"no answers in {res_dir / 'results.json'}")
    ids = fr.frame_ids(args.scan)
    first = next((pts for _, pts, _ in answers if len(pts)), None)
    scored = []
    for k in ids:
        rgb, depth, conf, K, T = fr.read_frame(args.scan, k)
        n = 0 if first is None else int(visible(first, depth, K, T)[3].sum())
        scored.append((n, k))
    pick = sorted(k for n, k in sorted(scored, reverse=True)[:args.views])
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for k in pick:
        rgb, depth, conf, K, T = fr.read_frame(args.scan, k)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faint = img.copy()
        for a, pts, col in answers:
            if not len(pts):
                continue
            ui, vi, inside, front = visible(pts, depth, K, T)
            for j in np.nonzero(inside & ~front)[0]:
                cv2.circle(faint, (int(ui[j]), int(vi[j])), 2, col, -1)
            for j in np.nonzero(front)[0]:
                cv2.circle(img, (int(ui[j]), int(vi[j])), 2, col, -1)
        img = cv2.addWeighted(img, 0.75, faint, 0.25, 0) if (faint != img).any() else img
        y = 18
        for a, pts, col in answers:
            text = f"{a['score']:.2f} {a['query']}"
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += 18
        cv2.imwrite(str(out / f"view_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"{len(pick)} views -> {out}/ (" + ", ".join(f"view_{k}.jpg" for k in pick) + ")")
    for a, pts, col in answers:
        print(f"  {a['score']:.3f} ({a['band']}) '{a['query']}': {len(pts)} points"
              + (f" at {np.round(np.asarray(a['center']), 3).tolist()}" if a.get("center") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
