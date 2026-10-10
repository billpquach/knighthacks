"""A Qwen-3D answer (or a detected object) -> a dense, clean cloud of that object from the scan's keyframes:
its digital twin. The model input is left alone; the object is collected again from the scan.

  1. seed = the answer's points (2 cm voxels); table height = the most common height between 0.4 and 1.2 m
     over the scan (1 cm bins)
  2. every keyframe that sees the seed's centre:
     * with a detector mask (cup / ball / can): the detection of that kind whose mask holds the projected
       centre; its mask eroded 5 px, only its nearest layer (3 cm in front of the 15th-percentile depth
       inside the mask to 8 cm behind it), confidence >= 1, no flying pixels
     * without one: the depth near the seed (within 1.5 cm), confidence >= 1, no flying pixels
     every 2nd pixel (the depth is upsampled from 256 x 192: every pixel only repeats points), and only
     above the table + 1 cm (no table, no shadows)
  3. all keyframes fused, one point per 5 mm cell, the largest connected part (cells within 1.5 cm),
     then trimmed to a vertical cylinder around the median axis (6 cm for cups, 5 cm for cans, the
     ball's radius + 1 cm for balls); optionally without outliers (8-neighbour distance > mean + 2 sigma)
Output: points, centre (median), size (5th-95th percentile extent). numpy only, on the CPU.

    python -m ground3d.refine <scan> <ground3d output folder>        # refine Qwen-3D answers
    python -m ground3d.refine <scan> --objects --out <folder>        # every cup / ball / can the detector finds
Detections come from <scan>/detections_<labels>.npz; the detector runs once (--prompt) to make it, unless
--no-detector. Answers get "refined" ({n_points, center, size, ...}) in results.json and <slug>_refined.ply.
The objects mode writes the same layout, so python -m ground3d.show draws either.
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

DEFAULT_PROMPT = "solo cup,ball"


def load_keyframes(scan):
    """[(depth, conf, K, T)] of every keyframe, in frame-id order (no colour: refining doesn't need it)."""
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


def kind(text):
    """cup / ball / can / None from a label or a target phrase."""
    words = set(str(text).lower().replace(",", " ").split())
    if words & {"ball", "balls"}:
        return "ball"
    if words & {"cup", "cups", "mug", "glass"}:
        return "cup"
    if words & {"can", "cans", "soda"}:
        return "can"
    return None


def trim_radius(k, ball_diameter=0.064):
    return {"cup": 0.06, "can": 0.05, "ball": ball_diameter / 2 + 0.01}.get(k)


def voxel_mean(p, size):
    cells = np.floor(p / size).astype(np.int64)
    _, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    out = np.zeros((n, 3))
    np.add.at(out, inv, p)
    return out / np.bincount(inv, minlength=n)[:, None]


def clean_pixels(depth, conf, max_depth=3.0, stride=2):
    """Pixels worth backprojecting: confidence >= 1, near enough, not flying (a flying pixel sits on a depth
    edge: its 3x3 neighbourhood spans more than 2 cm + 1% of its depth), every `stride` pixels."""
    k = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(depth, k) - cv2.erode(depth, k)
    ok = (conf >= 1) & (depth > 0) & (depth < max_depth) & (spread < 0.02 + 0.01 * depth)
    sub = np.zeros_like(ok)
    sub[::stride, ::stride] = True
    return ok & sub


def nearest_layer(depth, mask, front=0.03, back=0.08):
    """The mask's pixels on its nearest surface: from `front` m in front of its 15th-percentile depth to
    `back` m behind it (drops the background seen past the object's edges)."""
    d = depth[mask]
    d = d[d > 0]
    if len(d) < 10:
        return np.zeros_like(mask)
    p15 = np.percentile(d, 15)
    return mask & (depth > p15 - front) & (depth < p15 + back)


def erode(mask, px):
    return cv2.erode(mask.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def near(points, seed, radius, chunk=4096):
    """Which points are within `radius` of any seed point (bounding box first, then exact distances)."""
    lo, hi = seed.min(0) - radius, seed.max(0) + radius
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    idx = np.nonzero(keep)[0]
    r2 = radius * radius
    for i in range(0, len(idx), chunk):
        part = idx[i:i + chunk]
        keep[part] = ((points[part, None, :] - seed[None, :, :]) ** 2).sum(-1).min(1) <= r2
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
        pos = np.clip(np.searchsorted(skey, nk), 0, len(skey) - 1)
        hit = skey[pos] == nk
        src.append(np.nonzero(hit)[0])
        dst.append(order[pos[hit]])
    src, dst = np.concatenate(src), np.concatenate(dst)
    label = np.arange(len(points))
    while True:                                              # propagate the smallest label along the edges
        new = label.copy()
        np.minimum.at(new, src, label[dst])
        new = new[new]                                       # (pointer jump: far fewer rounds)
        if np.array_equal(new, label):
            break
        label = new
    return label == np.argmax(np.bincount(label))


def in_cylinder(points, radius):
    """Points within `radius` (horizontally) of the cloud's median vertical axis."""
    axis = np.median(points[:, :2], axis=0)
    return np.hypot(*(points[:, :2] - axis).T) <= radius


def drop_outliers(points, k=8, sigma=2.0, chunk=2048):
    """Points whose mean distance to their k nearest neighbours is within mean + sigma * std."""
    if len(points) <= k + 1:
        return np.ones(len(points), bool)
    md = np.empty(len(points))
    for i in range(0, len(points), chunk):
        d = np.sqrt(((points[i:i + chunk, None, :] - points[None, :, :]) ** 2).sum(-1))
        md[i:i + chunk] = np.sort(d, axis=1)[:, 1:k + 1].mean(1)
    return md <= md.mean() + sigma * md.std()


def pick_detection(dets, centre_uv, want):
    """The highest-scoring detection of kind `want` whose mask holds the pixel centre_uv (None if none)."""
    u, v = centre_uv
    best = None
    for d in dets or []:
        if kind(d.label) != want:
            continue
        h, w = d.mask.shape
        if 0 <= v < h and 0 <= u < w and d.mask[v, u] and (best is None or d.score > best.score):
            best = d
    return best


def object_points(depth, conf, K, T, mask, max_depth=3.0, erode_px=5):
    """A detection's points in one keyframe: eroded mask, clean pixels, nearest layer."""
    m = erode(mask, erode_px) & clean_pixels(depth, conf, max_depth)
    return fr.backproject(depth, K, T, nearest_layer(depth, m))


def fit_circle(xy):
    """Least-squares circle through 2-D points (Kasa): (centre, radius), or None."""
    if len(xy) < 8:
        return None
    sol = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy ** 2).sum(1), rcond=None)[0]
    r2 = sol[2] + sol[:2] @ sol[:2]
    return (sol[:2], float(np.sqrt(r2))) if r2 > 0 else None


def angular_span(angles):
    """How much of the circle (radians) the angles cover: 2 pi minus the widest gap between them."""
    a = np.sort(np.mod(angles, 2 * np.pi))
    return float(2 * np.pi - np.max(np.diff(np.r_[a, a[0] + 2 * np.pi])))


def complete_round(pts, want, table_z=None, slice_h=0.005, sector_deg=15, spacing=0.005, seed=0):
    """Cups and balls are round about a vertical axis, so the sides a scan never saw can be filled in from the
    sides it did: per horizontal slice, the wall's radius around the axis, repeated in each empty direction.
    A scan from one side gives half a shell; this gives the whole object (and its true centre). Returns
    (points, how many were added); unchanged if the axis can't be found (too few points, or not round)."""
    if want not in ("cup", "ball") or len(pts) < 100:
        return pts, 0
    z0 = pts[:, 2].min()
    n_sl = max(1, int(np.ceil((pts[:, 2].max() - z0) / slice_h)))
    sl = np.minimum(((pts[:, 2] - z0) / slice_h).astype(int), n_sl - 1)
    centres = []
    for s in range(n_sl):                                   # the axis: the slices' circle centres
        xy = pts[sl == s, :2]
        f = fit_circle(xy) if len(xy) >= 15 else None
        if f is not None and 0.012 <= f[1] <= 0.07 and angular_span(np.arctan2(*(xy - f[0]).T[::-1])) >= np.radians(60):
            centres.append(f[0])
    if len(centres) < 3:
        return pts, 0
    axis = np.median(centres, axis=0)
    d = np.hypot(*(pts[:, :2] - axis).T)
    n_sec = int(round(360 / sector_deg))
    sector = (np.mod(np.arctan2(pts[:, 1] - axis[1], pts[:, 0] - axis[0]), 2 * np.pi) / np.radians(sector_deg)).astype(int) % n_sec
    rng, add = np.random.default_rng(seed), []
    for s in range(n_sl):
        m = sl == s
        if m.sum() < 5:
            continue
        r = float(np.percentile(d[m], 80))                  # the wall (not the cap inside it)
        z = z0 + (s + 0.5) * slice_h
        if r < 0.01 or (table_z is not None and z < table_z + 0.01):
            continue
        have = np.zeros(n_sec, bool)
        have[sector[m & (d > 0.6 * r)]] = True
        if have.all():
            continue
        a = np.linspace(0, 2 * np.pi, max(n_sec, int(2 * np.pi * r / spacing)), endpoint=False)
        a = a[~have[(a / np.radians(sector_deg)).astype(int) % n_sec]]
        a = a + rng.uniform(-0.5, 0.5, len(a)) * spacing / r          # a little jitter: no visible rings
        rr = r + rng.normal(0, 0.0015, len(a))
        add.append(np.c_[axis[0] + rr * np.cos(a), axis[1] + rr * np.sin(a), z + rng.uniform(-0.5, 0.5, len(a)) * slice_h])
    if not add:
        return pts, 0
    add = np.vstack(add)
    return np.vstack([pts, add]), len(add)


def refine(seed_points, keyframes, table_z=None, label=None, dets=None, ball_diameter=0.064, seed_cell=0.02,
           radius=0.015, above_table=0.01, thin=0.005, join=0.015, outliers=False, max_depth=3.0, complete=True):
    """dets: one list of xray.detect Detections per keyframe (or None). label: the object's label or target
    phrase (decides which detections fit and the size trim). complete: a cup's or ball's unseen sides filled in
    by symmetry (complete_round)."""
    t0 = time.time()
    want = kind(label) if label else None
    seed = voxel_mean(np.asarray(seed_points, float).reshape(-1, 3), seed_cell)
    centre = np.median(seed, axis=0)
    got, views, masked = [], 0, 0
    for i, (depth, conf, K, T) in enumerate(keyframes):
        u, v, z = fr.project(centre[None], K, T)
        h, w = depth.shape
        if not (z[0] > 0.05 and 0 <= u[0] < w and 0 <= v[0] < h):
            continue
        views += 1
        d = pick_detection(dets[i] if dets is not None else None, (int(round(u[0])), int(round(v[0]))), want) if want else None
        if d is not None:
            p = object_points(depth, conf, K, T, d.mask_at(depth.shape), max_depth)
            masked += 1
        else:
            p = fr.backproject(depth, K, T, clean_pixels(depth, conf, max_depth))
            p = p[near(p, seed, radius)]
        if table_z is not None:
            p = p[p[:, 2] > table_z + above_table]
        got.append(p)
    pts = np.concatenate(got) if got else np.zeros((0, 3))
    if len(pts):
        pts = voxel_mean(pts, thin)
        pts = pts[largest_component(pts, thin, join)]
        r_trim = trim_radius(want, ball_diameter)
        if r_trim is not None and len(pts):
            pts = pts[in_cylinder(pts, r_trim)]
        if outliers and len(pts):
            pts = pts[drop_outliers(pts)]
    filled = 0
    if complete and len(pts):
        pts, filled = complete_round(pts, want, table_z)
    out = dict(points=pts, n_points=int(len(pts)), filled=filled, views=views, masked_views=masked, kind=want,
               table_z=table_z, seed_points=int(len(seed)), seconds=round(time.time() - t0, 2))
    if len(pts):
        out.update(center=np.round(np.median(pts, axis=0), 4).tolist(),
                   size=np.round(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0), 4).tolist())
    else:
        out.update(center=None, size=None)
    return out


def summary(r):
    return {k: r[k] for k in ("n_points", "filled", "center", "size", "views", "masked_views", "kind", "table_z", "seed_points",
                              "seconds")}


def objects_from_detections(keyframes, dets, min_score=0.5, join=0.06, min_views=2, max_depth=3.0):
    """Every object the detector sees in at least min_views keyframes: [(label, seed points, views, best score)].
    Per keyframe each detection's points (eroded mask, nearest layer); detections of one label whose
    centres are within `join` of each other are the same object."""
    groups = []                                              # [label, [centres], [points], best score]
    for i, (depth, conf, K, T) in enumerate(keyframes):
        for d in dets[i] if dets is not None else []:
            if d.score < min_score or kind(d.label) is None:
                continue
            p = object_points(depth, conf, K, T, d.mask_at(depth.shape), max_depth)
            if len(p) < 30:
                continue
            c = np.median(p, axis=0)
            for g in groups:
                if g[0] == d.label and np.linalg.norm(np.mean(g[1], axis=0) - c) < join:
                    g[1].append(c)
                    g[2].append(p[::4])
                    g[3] = max(g[3], d.score)
                    break
            else:
                groups.append([d.label, [c], [p[::4]], d.score])
    out = [(g[0], voxel_mean(np.concatenate(g[2]), 0.02), len(g[1]), g[3]) for g in groups if len(g[1]) >= min_views]
    return sorted(out, key=lambda o: (o[0], -o[2]))


def write_ply(path, pts):
    with open(path, "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
        np.savetxt(f, pts, fmt="%.4f")


def scan_detections(scan, prompt, use_detector=True, log=print):
    """Per-keyframe detections (cached in the scan folder), or None without a detector."""
    from xray import detect
    labels = detect.parse_prompt(prompt)
    cached = detect.load_detections(detect.cache_path(scan, labels))
    if cached is None and use_detector:
        try:
            cached = detect.detect_scan(scan, labels, log=log)
        except (SystemExit, ImportError, RuntimeError) as e:
            log(f"(no detections: {e}; using the seed radius only)")
            return None
    if cached is None:
        return None
    return [cached.get(k, []) for k in fr.frame_ids(scan)]


def size_cm(size):
    return "x".join(f"{100 * s:.0f}" for s in size) + " cm" if size else "-"


def main(argv=None):
    from .show import read_ply, slug
    ap = argparse.ArgumentParser(description="Refine answers into dense object clouds (see the top of this file)")
    ap.add_argument("scan")
    ap.add_argument("results", nargs="?", help="the folder python -m ground3d.ground --out wrote")
    ap.add_argument("--objects", action="store_true", help="refine every object the detector finds instead")
    ap.add_argument("--out", help="with --objects: output folder (default runs/objects_<scan name>)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"detector labels (default \"{DEFAULT_PROMPT}\")")
    ap.add_argument("--no-detector", action="store_true", help="the seed radius only (no masks)")
    ap.add_argument("--ball-diameter", type=float, default=0.064)
    ap.add_argument("--outliers", action="store_true", help="also drop statistical outliers")
    args = ap.parse_args(argv)
    if not args.objects and not args.results:
        ap.error("give the ground3d output folder, or --objects")
    t = time.time()
    keyframes = load_keyframes(args.scan)
    table = table_height(keyframes)
    dets = None if args.no_detector else scan_detections(args.scan, args.prompt)
    print(f"{len(keyframes)} keyframes, table at " + ("(none between 0.4 and 1.2 m)" if table is None else f"z = {table:.3f} m")
          + f", detections: {'yes' if dets is not None else 'no'} ({time.time() - t:.1f} s)")
    if args.objects:
        if dets is None:
            raise SystemExit("--objects needs detections (a baked detector for --prompt, or the scan's detections file)")
        res_dir = Path(args.out or f"runs/objects_{Path(args.scan).resolve().parent.name}").expanduser()
        res_dir.mkdir(parents=True, exist_ok=True)
        found = objects_from_detections(keyframes, dets)
        answers, seeds = [], []
        count = {}
        for label, seed, nviews, score in found:
            count[label] = count.get(label, 0) + 1
            q = f"{label} {count[label]}"
            answers.append(dict(query=q, target=label, score=round(float(score), 3), band="high" if score >= 0.5 else "low",
                                n_points=int(len(seed)), center=np.round(np.median(seed, axis=0), 3).tolist(), detected_views=nviews))
            seeds.append(seed)
            write_ply(res_dir / f"{slug(q)}.ply", seed)
        res = dict(scan=str(Path(args.scan).expanduser().resolve()), source="detections", prompt=args.prompt, answers=answers)
    else:
        res_dir = Path(args.results).expanduser()
        res = json.loads((res_dir / "results.json").read_text())
        seeds = []
        for a in res.get("answers", []):
            ply = res_dir / f"{slug(a['query'])}.ply"
            seeds.append(read_ply(ply) if ply.exists() else np.zeros((0, 3)))
    print(f"{'object':40s} {'seed':>5s} {'refined':>8s} {'views':>9s}  {'size (5-95%)':>14s}  {'moved':>7s}  {'lowest':>8s}  {'s':>5s}")
    for a, seed in zip(res.get("answers", []), seeds):
        if not len(seed):
            print(f"{a['query'][:40]:40s} (no points)")
            continue
        r = refine(seed, keyframes, table, label=a.get("target") or a["query"], dets=dets,
                   ball_diameter=args.ball_diameter, outliers=args.outliers)
        a["refined"] = summary(r)
        if not r["n_points"]:
            print(f"{a['query'][:40]:40s} {len(seed):5d}        0 {r['views']:5d}      (nothing kept)")
            continue
        write_ply(res_dir / f"{slug(a['query'])}_refined.ply", r["points"])
        moved = 100 * float(np.linalg.norm(np.asarray(r["center"][:2]) - np.median(seed, axis=0)[:2]))
        low = "-" if table is None else f"{100 * (r['points'][:, 2].min() - table):.1f} cm"
        print(f"{a['query'][:40]:40s} {len(seed):5d} {r['n_points']:8d} {r['masked_views']:3d}/{r['views']:<3d}m  "
              f"{size_cm(r['size']):>14s}  {moved:4.1f} cm  {low:>8s}  {r['seconds']:5.2f}")
    (res_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(f"total {time.time() - t:.1f} s -> {res_dir}/ (results.json, *_refined.ply). Views: with a mask / seeing it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
