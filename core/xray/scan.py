"""Scan the scene with the phone: keyframes -> a scan folder (for Qwen-3D INIT and the tracker's maps).

    python -m xray.scan [--seconds 20] [--out runs/scan_HHMMSS]

Walk around the table slowly for --seconds. A frame becomes a keyframe once the camera has moved 10 cm or
turned 15 deg since the last one, unless it is much blurrier than the keyframes so far. At the end the world
frame is fixed for the session: z up (gravity, from the phone), the floor at z = 0, the scan centred in xy.
Then a check: each keyframe's depth, seen from another keyframe, should land on that frame's depth; a few
cm of disagreement or more means a pose, intrinsics or depth convention is wrong.

Outputs <out>/scan/ (color, depth, conf, intrinsic, pose, meta.json) and a SCAN SUMMARY to send back.
meta.json's world_from_arkit puts later frames of the same phone session in the same world.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import frames as fr
from .phone import Y_UP_TO_Z_UP


def sharpness(rgb):
    small = cv2.resize(rgb, (rgb.shape[1] // 2, rgb.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var())


def turned_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


class KeyframePicker:
    """Keeps a frame once the camera moved >= move m or turned >= turn deg since the last keyframe, and it
    isn't blurrier than blur x the median sharpness of the keyframes so far (after the first three)."""

    def __init__(self, move=0.10, turn=15.0, blur=0.6):
        self.move, self.turn, self.blur = move, turn, blur
        self.kept, self.sharp = [], []
        self.skipped_blurry = 0

    def offer(self, f):
        if self.kept:
            last = self.kept[-1].T
            if np.linalg.norm(f.T[:3, 3] - last[:3, 3]) < self.move and turned_deg(last[:3, :3], f.T[:3, :3]) < self.turn:
                return False
        s = sharpness(f.rgb)
        if len(self.sharp) >= 3 and s < self.blur * np.median(self.sharp):
            self.skipped_blurry += 1
            return False
        self.kept.append(f)
        self.sharp.append(s)
        return True


def capture(source, seconds, picker=None, on_frame=None):
    """Read the source for `seconds`; returns (keyframes, frames read)."""
    picker = picker or KeyframePicker()
    end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < end:
        f = source.read()
        n += 1
        picker.offer(f)
        if on_frame:
            on_frame(f, len(picker.kept), end - time.monotonic())
    return picker.kept, n, picker.skipped_blurry


def scene_points(keyframes, max_depth=4.0, stride=8, at_most=40):
    pts = []
    for f in keyframes[::max(1, len(keyframes) // at_most)]:
        m = (f.conf == 2) & (f.depth > 0) & (f.depth < max_depth)
        pts.append(fr.backproject(f.depth, f.K, f.T, m, stride))
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def fit_world(keyframes, max_depth=4.0, cell=0.01, min_share=0.02):
    """The shift that puts the floor at z = 0 and centres the scan in xy. The floor is the lowest 1 cm layer
    holding at least `min_share` of the confident points (LiDAR noise below it is sparse), refined to the
    median height of the points within 2 cm of it. If no floor is in view, that is the lowest large surface
    (the table)."""
    p = scene_points(keyframes, max_depth)
    if len(p) < 100:
        raise SystemExit(f"only {len(p)} confident depth points in the scan: is the LiDAR covered, or too far?")
    z = p[:, 2]
    lo = np.floor(z.min() / cell) * cell
    counts = np.bincount(((z - lo) / cell).astype(int))
    first = int(np.argmax(counts >= max(min_share * len(z), 30)))
    z0 = lo + (first + 0.5) * cell
    near = np.abs(z - z0) < 0.02
    floor_z = float(np.median(z[near]))
    x_mid = float((np.percentile(p[:, 0], 2) + np.percentile(p[:, 0], 98)) / 2)
    y_mid = float((np.percentile(p[:, 1], 2) + np.percentile(p[:, 1], 98)) / 2)
    shift = np.eye(4)
    shift[:3, 3] = [-x_mid, -y_mid, -floor_z]
    # gravity: the phone above the floor, its image-down axis pointing down. A wrong camera convention can still
    # give views that agree with each other (a mirrored world), but then the floor ends up above the camera.
    heights = np.array([f.T[2, 3] for f in keyframes]) - floor_z
    down = np.array([f.T[2, 1] for f in keyframes])
    info = dict(floor_z=round(floor_z, 4), floor_share=round(float(near.mean()), 3), points=int(len(p)),
                camera_height_m=round(float(np.median(heights)), 2), image_down_share=round(float((down < -0.3).mean()), 2),
                xy_centre=[round(x_mid, 3), round(y_mid, 3)],
                extent_m=np.round(np.percentile(p, 98, axis=0) - np.percentile(p, 2, axis=0), 2).tolist())
    return shift, info


def apply(shift, keyframes):
    for f in keyframes:
        f.T = shift @ f.T
    return keyframes


def agreement_cm(a, b, max_depth=4.0, stride=2):
    """Frame a's confident depth seen from frame b: median |depth difference| in cm over the pixels both see
    with high confidence (and the number of them)."""
    m = (a.conf == 2) & (a.depth > 0) & (a.depth < max_depth)
    p = fr.backproject(a.depth, a.K, a.T, m, stride)
    u, v, z = fr.project(p, b.K, b.T)
    h, w = b.depth.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (z > 0.1) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z = ui[ok], vi[ok], z[ok]
    seen = (b.conf[vi, ui] == 2) & (b.depth[vi, ui] > 0)
    if seen.sum() < 200:
        return None, int(seen.sum())
    return float(np.median(np.abs(b.depth[vi, ui][seen] - z[seen])) * 100), int(seen.sum())


def consistency(keyframes, checks=5, gap=4):
    """agreement_cm between keyframes from across the scan, each with a later one `gap` keyframes on (or
    nearer, if those two don't see the same surfaces): [(a, b, cm, pixels)]."""
    n = len(keyframes)
    out = []
    for a in sorted({int(round(x)) for x in np.linspace(0, max(0, n - 2), checks)}):
        for b in range(min(n - 1, a + gap), a, -1):
            cm, px = agreement_cm(keyframes[a], keyframes[b])
            if cm is not None or b == a + 1:
                out.append((a, b, cm, px))
                break
    return out


def write_scan(folder, keyframes, shift, **meta):
    folder = Path(folder)
    for k, f in enumerate(keyframes):
        fr.write_frame(folder, k, f)
    fr.write_meta(folder, keyframes, keyframes=True, world_from_arkit=shift @ Y_UP_TO_Z_UP, **meta)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scan the scene with the phone (see the top of this file)")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out", default=f"runs/scan_{time.strftime('%H%M%S')}")
    ap.add_argument("--min-keyframes", type=int, default=8)
    ap.add_argument("--max-depth", type=float, default=4.0, help="depth beyond this is ignored for the floor fit and checks")
    args = ap.parse_args(argv)
    from .phone import Phone
    phone = Phone()
    phone.read()                                            # (fills phone.info)
    print("stream:", phone.info, flush=True)
    input_ok = sys.stdin.isatty()
    if input_ok:
        input(f"press Enter, then walk slowly around the table for {args.seconds:.0f} s ")
    shown = [0.0]

    def progress(f, n, left):
        if time.monotonic() - shown[0] > 1.0:
            shown[0] = time.monotonic()
            print(f"  {max(0.0, left):4.1f} s left, {n} keyframes", flush=True)
    t = time.time()
    keyframes, n_read, blurry = capture(phone, args.seconds, KeyframePicker(), progress)
    secs = time.time() - t
    if len(keyframes) < args.min_keyframes:
        raise SystemExit(f"only {len(keyframes)} keyframes: move around more (10 cm or 15 deg between keyframes)")
    shift, world = fit_world(keyframes, args.max_depth)
    apply(shift, keyframes)
    out = Path(args.out).expanduser().resolve() / "scan"
    checks = consistency(keyframes)
    write_scan(out, keyframes, shift, source="record3d", stream=phone.info, world=world,
               consistency_cm=[[a, b, None if cm is None else round(cm, 2), px] for a, b, cm, px in checks])
    good = [cm for _, _, cm, _ in checks if cm is not None]
    upright = world["camera_height_m"] > 0.1 and world["image_down_share"] >= 0.8
    verdict = ("OK" if good and max(good) < 2.0 and upright else
               "CHECK (the floor is above the camera: a pose convention is wrong)" if not upright else
               "CHECK (poses, intrinsics or depth disagree)")
    print("\n=== SCAN SUMMARY (send me this) ===")
    print(f"stream: {phone.info}")
    print(f"{n_read} frames in {secs:.1f} s ({n_read / max(secs, 1e-6):.1f} fps) -> {len(keyframes)} keyframes "
          f"({blurry} blurry ones skipped)")
    print(f"world: floor was at z = {world['floor_z']:+.3f} m ({100 * world['floor_share']:.0f}% of the points near it), "
          f"scan centred on {world['xy_centre']}, extent {world['extent_m']} m; phone {world['camera_height_m']:.2f} m above "
          f"the floor, image down pointing down in {100 * world['image_down_share']:.0f}% of the keyframes")
    print("views agree: " + ", ".join(f"{a}->{b} {'-' if cm is None else f'{cm:.1f}'} cm ({px} px)" for a, b, cm, px in checks)
          + f": {verdict}")
    print(f"files: {out}", flush=True)
    sys.stderr.flush()
    os._exit(0)                     # record3d's teardown at a normal exit crashes (segfault); everything is written


if __name__ == "__main__":
    sys.exit(main())
