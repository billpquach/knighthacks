"""A Qwen-3D answer -> a dense, clean cloud of that object from the scan's keyframes (its digital twin).

The answer is right about WHICH object, but its points come from the 2 cm target cloud the model predicts
on: sparse, and bleeding onto the table and the background. The model input is left alone; the object is
collected again from the scan, using the answer only as a seed:
  1. seed = the answer's points (2 cm voxels); table height = the most common height between 0.4 and
     1.2 m over the scan (1 cm bins)
  2. every keyframe that sees the seed's centre: its depth with confidence >= 1 and < 3 m, every 2nd pixel
     (the depth is upsampled from 256 x 192; every pixel only repeats points), without flying pixels
     (depth range in the 3x3 neighbourhood >= 2 cm + 1% of the depth), keeping the points within 2.5 cm
     of a seed voxel and above the table + 1 cm (no table, no shadows)
  3. all keyframes fused, one point per 5 mm cell, the largest connected part (cells within 1.5 cm);
     optionally without outliers (mean distance to 8 neighbours > mean + 2 sigma)
Output: the points, centre (median), size (5th-95th percentile extent). numpy only, on the CPU: it runs
after Qwen-3D's queries, without GPU memory.

    python -m ground3d.refine <scan> <ground3d output folder> [--outliers]
adds "refined" ({n_points, center, size, ...}) to each answer in results.json (the raw answer and its
score stay as they are) and writes <slug>_refined.ply next to the answer's .ply.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xray import frames as fr  # noqa: E402


def load_keyframes(scan):
    """[(depth, conf, K, T)] of every keyframe (no colour: refining doesn't need it)."""
    scan = Path(scan)
    out = []
    for k in fr.frame_ids(scan):
        depth = np.asarray(Image.open(scan / "depth" / f"{k}.png"), dtype=np.float32) / 1000.0
        cf = scan / "conf" / f"{k}.png"
        conf = np.asarray(Image.open(cf)) if cf.exists() else np.where(depth > 0, 2, 0).astype(np.uint8)
        out.append((depth, conf, np.loadtxt(scan / "intrinsic" / f"{k}.txt").reshape(3, 3),
                    np.loadtxt(scan / "pose" / f"{k}.txt").reshape(4, 4)))
    return out


def table_height(keyframes, lo=0.4, hi=1.2, cell=0.01, stride=8):
    """The most common height of confident scan points between lo and hi (None if hardly any are there)."""
    z = np.concatenate([fr.backproject(d, K, T, (c == 2) & (d > 0) & (d < 3.0), stride)[:, 2] for d, c, K, T in keyframes])
    zz = z[(z >= lo) & (z < hi)]
    if len(zz) < max(200, 0.01 * len(z)):
        return None
    counts = np.bincount(((zz - lo) / cell).astype(int))
    return float(lo + (np.argmax(counts) + 0.5) * cell)


def voxel_mean(p, size):
    cells = np.floor(p / size).astype(np.int64)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, p)
    return out / np.bincount(inv, minlength=n)[:, None]


def clean_pixels(depth, conf, max_depth=3.0, stride=2):
    """Pixels worth backprojecting: confident enough, near enough, not flying (a flying pixel sits on a depth
    edge: its 3x3 neighbourhood spans more than 2 cm + 1% of its depth)."""
    k = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(depth, k) - cv2.erode(depth, k)
    ok = (conf >= 1) & (depth > 0) & (depth < max_depth) & (spread < 0.02 + 0.01 * depth)
    sub = np.zeros_like(ok)
    sub[::stride, ::stride] = True
    return ok & sub


def near(points, seed, radius, chunk=4096):
    """Which points are within `radius` of any seed point (bounding box first, then exact distances)."""
    lo, hi = seed.min(0) - radius, seed.max(0) + radius
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    idx = np.nonzero(keep)[0]
    r2 = radius * radius
    for i in range(0, len(idx), chunk):
        part = idx[i:i + chunk]
        d2 = ((points[part, None, :] - seed[None, :, :]) ** 2).sum(-1).min(1)
        keep[part] = d2 <= r2
    return keep


def largest_component(points, cell, join):
    """The points of the largest group whose cells (side `cell`) touch within `join` metres."""
    if len(points) < 2:
        return np.ones(len(points), bool)
    r = int(np.ceil(join / cell))
    c = np.floor(points / cell).astype(np.int64)
    c -= c.min(0) - (r + 1)                                  # room for the neighbour offsets on every side
    span = c.max(0) + r + 2
    key = (c[:, 0] * span[1] + c[:, 1]) * span[2] + c[:, 2]
    order = np.argsort(key)
    skey = key[order]
    offs = [(dx, dy, dz) for dx in range(-r, r + 1) for dy in range(-r, r + 1) for dz in range(-r, r + 1)
            if (dx, dy, dz) != (0, 0, 0) and (dx * dx + dy * dy + dz * dz) * cell * cell <= join * join + 1e-12]
    src, dst = [], []
    for dx, dy, dz in offs:
        nk = key + (dx * span[1] + dy) * span[2] + dz
        pos = np.searchsorted(skey, nk)
        pos = np.clip(pos, 0, len(skey) - 1)
        hit = skey[pos] == nk
        src.append(np.nonzero(hit)[0])
        dst.append(order[pos[hit]])
    src, dst = np.concatenate(src), np.concatenate(dst)
    label = np.arange(len(points))
    while True:                                              # propagate the smallest label along the edges
        new = label.copy()
        np.minimum.at(new, src, label[dst])
        new = new[new]                                       # (jump: converges in far fewer rounds)
        if np.array_equal(new, label):
            break
        label = new
    counts = np.bincount(label)
    return label == np.argmax(counts)


def drop_outliers(points, k=8, sigma=2.0, chunk=2048):
    """Points whose mean distance to their k nearest neighbours is within mean + sigma * std."""
    if len(points) <= k + 1:
        return np.ones(len(points), bool)
    md = np.empty(len(points))
    for i in range(0, len(points), chunk):
        d = np.sqrt(((points[i:i + chunk, None, :] - points[None, :, :]) ** 2).sum(-1))
        md[i:i + chunk] = np.sort(d, axis=1)[:, 1:k + 1].mean(1)
    return md <= md.mean() + sigma * md.std()


def refine(seed_points, keyframes, table_z=None, seed_cell=0.02, radius=0.025, above_table=0.01, thin=0.005,
           join=0.015, outliers=False, max_depth=3.0):
    t0 = time.time()
    seed = voxel_mean(np.asarray(seed_points, float).reshape(-1, 3), seed_cell)
    centre = np.median(seed, axis=0)
    got, views = [], 0
    for depth, conf, K, T in keyframes:
        u, v, z = fr.project(centre[None], K, T)
        h, w = depth.shape
        if not (z[0] > 0.05 and 0 <= u[0] < w and 0 <= v[0] < h):
            continue
        views += 1
        p = fr.backproject(depth, K, T, clean_pixels(depth, conf, max_depth))
        p = p[near(p, seed, radius)]
        if table_z is not None:
            p = p[p[:, 2] > table_z + above_table]
        got.append(p)
    pts = np.concatenate(got) if got else np.zeros((0, 3))
    if len(pts):
        pts = voxel_mean(pts, thin)
        pts = pts[largest_component(pts, thin, join)]
        if outliers:
            pts = pts[drop_outliers(pts)]
    out = dict(points=pts, n_points=int(len(pts)), views=views, table_z=table_z, seed_points=int(len(seed)),
               seconds=round(time.time() - t0, 2))
    if len(pts):
        out.update(center=np.round(np.median(pts, axis=0), 4).tolist(),
                   size=np.round(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0), 4).tolist())
    else:
        out.update(center=None, size=None)
    return out


def summary(r):
    return {k: r[k] for k in ("n_points", "center", "size", "views", "table_z", "seed_points", "seconds")}


def write_ply(path, pts):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.4f")


def main(argv=None):
    from .show import read_ply, slug
    ap = argparse.ArgumentParser(description="Refine Qwen-3D answers into dense object clouds (see the top of this file)")
    ap.add_argument("scan")
    ap.add_argument("results", help="the folder python -m ground3d.ground --out wrote")
    ap.add_argument("--outliers", action="store_true", help="also drop statistical outliers")
    args = ap.parse_args(argv)
    res_dir = Path(args.results).expanduser()
    res = json.loads((res_dir / "results.json").read_text())
    t = time.time()
    keyframes = load_keyframes(args.scan)
    table = table_height(keyframes)
    t_load = time.time() - t
    print(f"{len(keyframes)} keyframes read in {t_load:.2f} s; table at "
          + ("(none found between 0.4 and 1.2 m)" if table is None else f"z = {table:.3f} m"))
    print(f"{'answer':40s} {'raw':>5s} {'refined':>8s} {'views':>5s}  {'moved sideways':>14s}  {'lowest above table':>18s}  {'s':>5s}")
    for a in res.get("answers", []):
        ply = res_dir / f"{slug(a['query'])}.ply"
        if not ply.exists():
            print(f"{a['query'][:40]:40s} (no points)")
            continue
        raw = read_ply(ply)
        r = refine(raw, keyframes, table, outliers=args.outliers)
        a["refined"] = summary(r)
        if r["n_points"]:
            write_ply(res_dir / f"{slug(a['query'])}_refined.ply", r["points"])
            side = 100 * float(np.linalg.norm(np.asarray(r["center"][:2]) - np.median(raw, axis=0)[:2]))
            low = "-" if table is None else f"{100 * (r['points'][:, 2].min() - table):.1f} cm"
            print(f"{a['query'][:40]:40s} {len(raw):5d} {r['n_points']:8d} {r['views']:5d}  {side:11.1f} cm  {low:>18s}  {r['seconds']:5.2f}")
        else:
            print(f"{a['query'][:40]:40s} {len(raw):5d}        0 {r['views']:5d}  (nothing kept)")
    (res_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"total {time.time() - t:.2f} s; {res_dir / 'results.json'} updated, *_refined.ply written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
