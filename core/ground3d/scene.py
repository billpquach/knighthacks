"""A scan folder -> the scene inputs Qwen-3D needs, in plain PyTorch (the same steps upstream's data loader
and forward take at inference, without torch_scatter or CUDA extensions).

Scan folder (what the recorder writes): color/<k>.jpg, depth/<k>.png (millimetres, uint16), conf/<k>.png
(LiDAR confidence 0-2), intrinsic/<k>.txt (3x3), pose/<k>.txt (4x4 camera-to-world, OpenCV camera: x right,
y down, z forward), in a world frame with z up, in metres.

Steps:
  1. n frames, evenly spread over the scan's keyframes
  2. each frame -> the Qwen image processor (pixel patches for the ViT) and, at the ViT's token grid
     (one token per 28 x 28 pixels after its 2 x 2 merge), the world point under each token
  3. tokens -> 5 cm voxels: each voxel's point is the mean of its tokens' points, and its pixel position
     the smallest (row, column) among them (for the model's position encoding)
  4. the target cloud the masks are predicted on: confident LiDAR points from every keyframe, 2 cm voxels,
     kept only near the tokens, with the 8 nearest token voxels of each (inverse-distance weights)
"""
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass
class Scene:
    scan: str
    frames: list                    # keyframe ids used
    grid: tuple                     # token grid per frame (rows, cols)
    pixel_values: list              # per frame, for the ViT
    grid_thw: list                  # per frame, for the ViT
    token_voxel: torch.Tensor       # [frames * rows * cols] -> voxel id
    voxel_xyz: torch.Tensor         # [V, 3] world
    voxel_pixel: torch.Tensor       # [V, 2] smallest (row, col) of the voxel's tokens
    target_xyz: torch.Tensor        # [M, 3] 2 cm voxels of the target cloud
    knn_idx: torch.Tensor           # [M, 8] nearest token voxels
    knn_w: torch.Tensor             # [M, 8] their weights (sum 1)
    timing: dict = field(default_factory=dict)

    @property
    def n_voxels(self):
        return int(self.voxel_xyz.shape[0])


def frame_ids(scan):
    return sorted(int(p.stem) for p in (Path(scan) / "color").glob("*.jpg"))


def load_frame(scan, k):
    scan = Path(scan)
    rgb = np.asarray(Image.open(scan / "color" / f"{k}.jpg").convert("RGB"))
    depth = np.asarray(Image.open(scan / "depth" / f"{k}.png"), dtype=np.float32) / 1000.0
    conf_f = scan / "conf" / f"{k}.png"
    conf = np.asarray(Image.open(conf_f)) if conf_f.exists() else np.where(depth > 0, 2, 0).astype(np.uint8)
    K = np.loadtxt(scan / "intrinsic" / f"{k}.txt").reshape(3, 3)
    T = np.loadtxt(scan / "pose" / f"{k}.txt").reshape(4, 4)
    return rgb, depth, conf, K, T


def token_grid(h, w, step=28, min_pixels=3136, max_pixels=390000):
    """Token rows and columns for an h x w image: the size is rounded to multiples of `step`, then scaled
    down (or up) to stay within the pixel budget, as the Qwen image processor does."""
    hh, ww = max(step, round(h / step) * step), max(step, round(w / step) * step)
    if hh * ww > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hh, ww = max(step, math.floor(h / beta / step) * step), max(step, math.floor(w / beta / step) * step)
    elif hh * ww < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hh, ww = math.ceil(h * beta / step) * step, math.ceil(w * beta / step) * step
    return hh // step, ww // step


def world_points(depth, K, T):
    """H x W x 3 world point of every pixel. A pixel without depth lands on the camera centre (upstream
    keeps invalid depth too, with its mask_valid off)."""
    h, w = depth.shape
    v, u = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    z = depth
    cam = torch.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)
    return cam @ T[:3, :3].T + T[:3, 3]


def voxel_ids(xyz, size):
    """Point -> voxel id: points rounding to the same cell of a `size` grid share an id (ids 0..n-1)."""
    cell = torch.round(xyz / size).long()
    cell -= cell.min(0).values
    span = cell.max(0).values + 1
    key = (cell[:, 0] * span[1] + cell[:, 1]) * span[2] + cell[:, 2]
    return torch.unique(key, return_inverse=True)[1]


def group_mean(x, ids, n=None):
    n = int(ids.max()) + 1 if n is None else n
    out = torch.zeros(n, x.shape[1], dtype=torch.float32, device=x.device).index_add_(0, ids, x.float())
    cnt = torch.zeros(n, device=x.device).index_add_(0, ids, torch.ones(len(ids), device=x.device))
    return out / cnt.clamp(min=1)[:, None]


def nearest(src, dst, k, chunk=4096):
    """For each dst point its k nearest src points: (indices [N, k], distances [N, k])."""
    idx, dist = [], []
    for i in range(0, len(dst), chunk):
        d, j = torch.cdist(dst[i:i + chunk], src).topk(k, dim=1, largest=False)
        idx.append(j)
        dist.append(d)
    return torch.cat(idx), torch.cat(dist)


@torch.no_grad()
def prepare(scan, image_processor, cfg, n_frames=15, max_depth=3.0, device="cuda", target_voxel=0.02,
            target_stride=3, near_tokens=0.08):
    t0 = time.time()
    timing = {}
    ids = frame_ids(scan)
    if not ids:
        raise SystemExit(f"{scan}: no color/*.jpg (not a scan folder?)")
    pick = sorted({ids[i] for i in np.linspace(0, len(ids) - 1, min(n_frames, len(ids))).round().astype(int)})
    frames = [load_frame(scan, k) for k in pick]
    timing["load"] = time.time() - t0

    # ViT input: the processor resizes each frame to its own multiple-of-28 size
    t = time.time()
    hwc = [torch.from_numpy(f[0].copy()) for f in frames]
    out = image_processor(images=hwc, do_rescale=True, return_tensors="pt")
    per = (out["image_grid_thw"].prod(1)).tolist()
    pixel_values = list(torch.split(out["pixel_values"], per))
    grid_thw = list(out["image_grid_thw"])
    timing["processor"] = time.time() - t

    # token points: each frame's world points at the token grid (nearest pixel)
    t = time.time()
    h, w = frames[0][1].shape
    step = cfg.MULTIVIEW_XYZ_SCALES[0]
    gh, gw = token_grid(h, w, step, cfg.INPUT.MIN_PIXEL, cfg.INPUT.MAX_PIXEL)
    merge = 2
    for g in grid_thw:
        if (int(g[1]) // merge, int(g[2]) // merge) != (gh, gw):
            raise SystemExit(f"the ViT grid {tuple(int(x) for x in g)} (after merge {int(g[1]) // merge}x{int(g[2]) // merge}) "
                             f"doesn't match the token grid {gh}x{gw} for {w}x{h} frames")
    pts = []
    for rgb, depth, conf, K, T in frames:
        xyz = world_points(torch.from_numpy(depth), torch.from_numpy(K).float(), torch.from_numpy(T).float())
        xyz = F.interpolate(xyz.permute(2, 0, 1)[None], size=(gh, gw), mode=cfg.MODEL.INTERPOLATION_METHOD)[0]
        pts.append(xyz.permute(1, 2, 0).reshape(-1, 3))
    tok_xyz = torch.cat(pts).to(device)
    token_voxel = voxel_ids(tok_xyz, cfg.INPUT.VOXEL_SIZE[0])
    n_vox = int(token_voxel.max()) + 1
    voxel_xyz = group_mean(tok_xyz, token_voxel, n_vox)
    rows = torch.arange(gh, device=device).repeat_interleave(gw).repeat(len(frames))
    cols = torch.arange(gw, device=device).repeat(gh).repeat(len(frames))
    pix = torch.stack([rows, cols], 1)
    voxel_pixel = torch.full((n_vox, 2), 1 << 30, dtype=torch.long, device=device)
    voxel_pixel.scatter_reduce_(0, token_voxel[:, None].expand(-1, 2), pix, "amin", include_self=True)
    timing["tokens"] = time.time() - t

    # target cloud: confident LiDAR from every keyframe -> 2 cm voxels near the tokens
    t = time.time()
    dense = []
    for k in ids:
        _, depth, conf, K, T = load_frame(scan, k)
        d = depth[::target_stride, ::target_stride]
        c = conf[::target_stride, ::target_stride]
        v, u = np.nonzero((c == 2) & (d > 0) & (d < max_depth))
        z = d[v, u]
        u, v = u * target_stride, v * target_stride
        cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
        dense.append(cam @ T[:3, :3].T + T[:3, 3])
    dense = torch.from_numpy(np.concatenate(dense).astype(np.float32)).to(device)
    dense_vox = group_mean(dense, voxel_ids(dense, target_voxel))
    _, d1 = nearest(voxel_xyz, dense_vox, 1)
    target_xyz = dense_vox[d1[:, 0] < near_tokens]
    knn_idx, knn_d = nearest(voxel_xyz, target_xyz, cfg.get("INTERP_NEIGHBORS", 8))
    knn_w = 1.0 / (knn_d + 1e-8)
    knn_w /= knn_w.sum(1, keepdim=True)
    timing["target"] = time.time() - t
    return Scene(str(scan), pick, (gh, gw), pixel_values, grid_thw, token_voxel, voxel_xyz, voxel_pixel,
                 target_xyz, knn_idx, knn_w, timing)
