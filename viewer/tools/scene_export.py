"""Export a private RGB-D scan into the viewer's static world-space scene."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
from xray.frames import backproject, frame_ids, read_frame


def export_scene(folder, objects=()):
    cells = {}
    calibration = None
    ids = frame_ids(folder)
    if not ids:
        raise ValueError("Scan has no color keyframes")
    for k in ids:
        rgb, depth, conf, K, T = read_frame(folder, k)
        if calibration is None:
            calibration = dict(width=rgb.shape[1], height=rgb.shape[0],
                               fx=float(K[0, 0]), fy=float(K[1, 1]),
                               cx=float(K[0, 2]), cy=float(K[1, 2]))
        mask = (conf == 2) & (depth > 0) & (depth < 3)
        points = backproject(depth, K, T, mask=mask, stride=3)
        colors = rgb[::3, ::3][mask[::3, ::3]]
        for point, color in zip(points, colors):
            if np.isfinite(point).all():
                cell = tuple(np.floor(point / .015).astype(np.int64))
                if cell not in cells:
                    cells[cell] = (point, color)
    if not cells:
        raise ValueError("Scan has no high-confidence points within 3 metres")
    points = np.array([v[0] for v in cells.values()])
    colors = np.array([v[1] for v in cells.values()])
    heights, edges = np.histogram(points[:, 2], bins=40, range=(.4, 1.2))
    if not heights.any():
        raise ValueError("No tabletop candidates between 0.4 and 1.2 metres")
    peak = int(heights.argmax())
    z = float((edges[peak] + edges[peak + 1]) / 2)
    surface = points[np.abs(points[:, 2] - z) <= .02, :2].astype(np.float32)
    outline = cv2.convexHull(surface).reshape(-1, 2).round(5).tolist() if len(surface) >= 3 else []
    # Deterministic even sampling preserves coverage after voxel thinning.
    indices = np.linspace(0, len(points) - 1, min(50000, len(points)), dtype=int)
    return dict(version=1, frame="world", up="z", units="m",
                table=dict(z=round(z, 5), outline=outline), camera=calibration,
                cloud=dict(points=points[indices].round(5).tolist(), colors=colors[indices].tolist()),
                objects=[dict(id=o["id"], label=o.get("label", "object"),
                              position=o["position"], points=[]) for o in objects])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--states", type=Path, help="Seed IDs from the first recorded state")
    args = ap.parse_args()
    objects = []
    if args.states:
        with args.states.open() as stream:
            first = next((line for line in stream if line.strip()), None)
        if first is None:
            ap.error("State recording is empty")
        objects = json.loads(first)["objects"]
    scene = export_scene(args.scan, objects)
    args.output.write_text(json.dumps(scene, separators=(",", ":"), allow_nan=False))
    print(f"Exported {len(scene['cloud']['points'])} points; table z={scene['table']['z']:.3f} m to {args.output}")


if __name__ == "__main__":
    main()
