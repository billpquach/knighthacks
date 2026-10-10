"""One RGB-D frame with its camera, and scan folders on disk.

A Frame's pose is camera-to-world with an OpenCV camera (x right, y down, z forward), in a world with z up,
in metres. A scan folder (what xray.scan writes, ground3d reads, and the replays use):
    color/<k>.jpg   depth/<k>.png (millimetres, uint16)   conf/<k>.png (LiDAR confidence 0-2)
    intrinsic/<k>.txt (3x3)   pose/<k>.txt (4x4)   meta.json (frame times, how the world frame was made)
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Frame:
    t: float                    # seconds since the stream started
    rgb: np.ndarray             # H x W x 3 uint8, RGB order
    depth: np.ndarray           # H x W float32 metres (0 = no depth)
    conf: np.ndarray            # H x W uint8 LiDAR confidence: 0 low, 1 medium, 2 high
    K: np.ndarray               # 3 x 3 intrinsics for this image size
    T: np.ndarray               # 4 x 4 camera-to-world (OpenCV camera, z-up world)
    index: int = 0

    @property
    def size(self):
        return self.rgb.shape[1], self.rgb.shape[0]

    @property
    def position(self):
        return self.T[:3, 3]

    @property
    def forward(self):
        return self.T[:3, 2]


def backproject(depth, K, T, mask=None, stride=1):
    """World points (N x 3) of the pixels in `mask` (default: every pixel with depth), every `stride` pixels."""
    if mask is None:
        mask = depth > 0
    v, u = np.nonzero(mask[::stride, ::stride])
    v, u = v * stride, u * stride
    z = depth[v, u]
    cam = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return cam @ T[:3, :3].T + T[:3, 3]


def project(points, K, T):
    """World points -> pixel u, v and camera depth z (z <= 0: behind the camera)."""
    cam = (points - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return cam[:, 0] * K[0, 0] / zs + K[0, 2], cam[:, 1] * K[1, 1] / zs + K[1, 2], z


def write_frame(folder, k, f):
    folder = Path(folder)
    for d in ("color", "depth", "conf", "intrinsic", "pose"):
        (folder / d).mkdir(parents=True, exist_ok=True)
    Image.fromarray(f.rgb).save(folder / "color" / f"{k}.jpg", quality=95)
    Image.fromarray(np.clip(np.round(f.depth * 1000), 0, 65535).astype(np.uint16)).save(folder / "depth" / f"{k}.png")
    Image.fromarray(f.conf.astype(np.uint8)).save(folder / "conf" / f"{k}.png")
    np.savetxt(folder / "intrinsic" / f"{k}.txt", f.K, fmt="%.6f")
    np.savetxt(folder / "pose" / f"{k}.txt", f.T, fmt="%.8f")


def write_meta(folder, frames, **extra):
    meta = dict(num_frames=len(frames), frame_times_s=[round(float(f.t), 4) for f in frames],
                image_size_wh=list(frames[0].size) if frames else [0, 0],
                pose="camera-to-world, OpenCV camera (x right, y down, z forward), world z up, metres", **extra)
    (Path(folder) / "meta.json").write_text(json.dumps(meta, indent=2, default=lambda o: np.asarray(o).tolist()))


def read_frame(folder, k):
    folder = Path(folder)
    rgb = np.asarray(Image.open(folder / "color" / f"{k}.jpg").convert("RGB"))
    depth = np.asarray(Image.open(folder / "depth" / f"{k}.png"), dtype=np.float32) / 1000.0
    cf = folder / "conf" / f"{k}.png"
    conf = np.asarray(Image.open(cf)) if cf.exists() else np.where(depth > 0, 2, 0).astype(np.uint8)
    K = np.loadtxt(folder / "intrinsic" / f"{k}.txt").reshape(3, 3)
    T = np.loadtxt(folder / "pose" / f"{k}.txt").reshape(4, 4)
    return rgb, depth, conf, K, T


def frame_ids(folder):
    return sorted(int(p.stem) for p in (Path(folder) / "color").glob("*.jpg"))


class FolderSource:
    """A scan or track folder read back as frames, in order (for replays and tests)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.ids = frame_ids(self.folder)
        meta = self.folder / "meta.json"
        self.meta = json.loads(meta.read_text()) if meta.exists() else {}
        times = self.meta.get("frame_times_s") or []
        self.times = times if len(times) == len(self.ids) else [i / 15.0 for i in range(len(self.ids))]

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for i, k in enumerate(self.ids):
            rgb, depth, conf, K, T = read_frame(self.folder, k)
            yield Frame(self.times[i], rgb, depth, conf, K, T, i)
