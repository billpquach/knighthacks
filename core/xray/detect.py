"""YOLOE open-vocabulary segmentation: text prompts -> labelled masks, boxes and scores, per RGB frame.

The prompts are baked into a small model file (this needs the MobileCLIP text encoder, about 570 MB:
do it on the laptop), then exported to TensorRT on the Jetson (the engine is specific to the board,
the TensorRT version and the image size). The label of a detection is its prompt.

    # laptop (pip install ultralytics "git+https://github.com/ultralytics/CLIP.git"), from core/:
    python -m xray.detect bake "solo cup,ball"            # -> models/yoloe-11s-seg_solo-cup_ball.pt
    # copy that file to the Jetson's models/, then on the Jetson (~10 min, FP16):
    python -m xray.detect export "solo cup,ball"          # -> models/yoloe-11s-seg_solo-cup_ball_640x480.engine
    # either machine: speed and what it finds on a folder of frames (a scan's color/ or any images)
    python -m xray.detect bench <folder> "solo cup,ball" [-n 30]
    python -m xray.detect show <image or folder> "solo cup,ball"   # draws the masks -> runs/detect_show/

From Python:
    det = Detector("solo cup,ball")      # the TensorRT engine for this prompt if there is one, else the baked .pt
    for d in det(rgb):                   # rgb: H x W x 3 uint8, RGB order
        d.label, d.score, d.box, d.mask  # mask: H x W bool, at the frame's size; d.mask_at((h, w)) for the depth size

Frames are portrait 480 x 640 (W x H) from the phone, so the default image size is 640 x 480 (H x W):
a frame goes in without letterboxing, and the engine is built for exactly that size.
"""
import argparse
import contextlib
import os
import re
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[2] / "models"     # repo root /models (not in git)
BASE = "yoloe-11s-seg"
IMGSZ = (640, 480)                                          # H, W: a portrait phone frame


def parse_prompt(prompt):
    """'solo cup, ball' -> ['solo cup', 'ball'] (comma separated; the order sets the class ids)."""
    labels = [p.strip() for p in prompt.split(",") if p.strip()]
    if not labels or len(set(labels)) != len(labels):
        raise SystemExit(f"bad prompt {prompt!r}: give comma-separated, different labels, e.g. \"solo cup,ball\"")
    return labels


def key(labels):
    return "_".join(re.sub(r"[^a-z0-9]+", "-", lab.lower()).strip("-") for lab in labels)


def pt_path(labels, base=BASE):
    return MODELS / f"{base}_{key(labels)}.pt"


def engine_path(labels, imgsz=IMGSZ, base=BASE):
    return MODELS / f"{base}_{key(labels)}_{imgsz[0]}x{imgsz[1]}.engine"


@dataclass
class Detection:
    label: str
    score: float
    box: tuple                      # x0, y0, x1, y1 in frame pixels
    mask: np.ndarray = field(repr=False)        # H x W bool, the frame's size
    _resized: dict = field(default_factory=dict, repr=False)

    def mask_at(self, shape):
        """The mask at another size (h, w), e.g. the LiDAR depth map's: nearest neighbour, cached."""
        shape = tuple(shape)
        if shape == self.mask.shape:
            return self.mask
        if shape not in self._resized:
            import cv2
            self._resized[shape] = cv2.resize(self.mask.view(np.uint8), (shape[1], shape[0]),
                                              interpolation=cv2.INTER_NEAREST).astype(bool)
        return self._resized[shape]

    @property
    def center(self):
        return (self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2


class Detector:
    def __init__(self, prompt, weights=None, conf=0.2, imgsz=IMGSZ, device=None):
        self.labels = parse_prompt(prompt) if isinstance(prompt, str) else list(prompt)
        self.conf, self.imgsz = conf, tuple(imgsz)
        if weights is None:
            eng, pt = engine_path(self.labels, self.imgsz), pt_path(self.labels)
            weights = eng if eng.exists() else pt
        self.weights = Path(weights)
        if not self.weights.exists():
            raise SystemExit(f"no model for {self.labels}: {self.weights} is missing. Bake it on the laptop "
                             f"(python -m xray.detect bake \"{','.join(self.labels)}\") and copy it to {MODELS}/")
        from ultralytics import YOLO, YOLOE
        self.engine = self.weights.suffix == ".engine"
        self.model = YOLO(str(self.weights), task="segment") if self.engine else YOLOE(str(self.weights))
        self.device = device
        self.ms = []                                        # time per call (everything: pre, model, post, masks)
        self.speed = []                                     # ultralytics' own split per call: preprocess, inference, postprocess

    @property
    def names(self):
        return self.model.names

    def __call__(self, rgb):
        """rgb: H x W x 3 uint8 in RGB order -> [Detection], best score first."""
        t = time.perf_counter()
        r = self.model.predict(np.ascontiguousarray(rgb[..., ::-1]), imgsz=self.imgsz, conf=self.conf,
                               retina_masks=True, verbose=False, device=self.device)[0]   # (ultralytics wants BGR)
        out = []
        if r.boxes is not None and len(r.boxes):
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            masks = (r.masks.data > 0.5).cpu().numpy() if r.masks is not None else None    # threshold on the GPU
            h, w = rgb.shape[:2]
            for i in np.argsort(-scores):
                m = masks[i] if masks is not None else np.zeros((h, w), bool)
                if m.shape != (h, w):                       # (only if the model didn't return frame-size masks)
                    import cv2
                    m = cv2.resize(m.view(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
                out.append(Detection(r.names[cls[i]], float(scores[i]), tuple(float(x) for x in boxes[i]), m))
        self.ms.append((time.perf_counter() - t) * 1000)
        self.speed.append(dict(r.speed))
        return out


# ---------------------------------------------------------------------------------------- per-scan cache
def cache_path(folder, labels):
    return Path(folder) / f"detections_{key(labels)}.npz"


def save_detections(path, per_frame, labels, shape):
    """per_frame: {frame id: [Detection]} -> one compressed file (masks bit-packed)."""
    rows = [(k, d) for k, ds in sorted(per_frame.items()) for d in ds]
    h, w = shape
    np.savez_compressed(
        path, labels=np.array(list(labels)), shape=np.array([h, w]), frames=np.array(sorted(per_frame), dtype=np.int64),
        frame=np.array([k for k, _ in rows], dtype=np.int64),
        label=np.array([list(labels).index(d.label) if d.label in labels else -1 for _, d in rows], dtype=np.int64),
        score=np.array([d.score for _, d in rows], dtype=np.float32),
        box=np.array([d.box for _, d in rows], dtype=np.float32).reshape(-1, 4),
        mask=np.array([np.packbits(d.mask.reshape(-1)) for _, d in rows], dtype=np.uint8).reshape(len(rows), -1))


class StoredDetections(Mapping):
    """save_detections' file read back as {frame id: [Detection]}. Each frame's detections are made when asked
    for: a long recording's masks, all unpacked, would take gigabytes."""

    def __init__(self, path):
        z = np.load(path)
        self.labels, self.shape = [str(x) for x in z["labels"]], tuple(int(x) for x in z["shape"])
        self._label, self._score, self._box, self._mask = z["label"], z["score"], z["box"], z["mask"]   # read once
        self._rows = {int(k): [] for k in z["frames"]}
        for i, k in enumerate(z["frame"].tolist()):
            self._rows.setdefault(int(k), []).append(i)

    def __getitem__(self, k):
        h, w = self.shape
        return [Detection(self.labels[self._label[i]] if self._label[i] >= 0 else "?", float(self._score[i]),
                          tuple(float(v) for v in self._box[i]),
                          np.unpackbits(self._mask[i])[:h * w].reshape(h, w).astype(bool)) for i in self._rows[k]]

    def __iter__(self):
        return iter(self._rows)

    def __len__(self):
        return len(self._rows)


def load_detections(path):
    """{frame id: [Detection]} from save_detections' file (None if there is none)."""
    path = Path(path)
    return StoredDetections(path) if path.exists() else None


def detect_scan(folder, labels, weights=None, conf=0.2, refresh=False, log=print):
    """Detections on every frame of a scan folder, cached as <folder>/detections_<labels>.npz (the detector
    runs once per scan; later calls read the file)."""
    labels = parse_prompt(labels) if isinstance(labels, str) else list(labels)
    path = cache_path(folder, labels)
    if not refresh:
        cached = load_detections(path)
        if cached is not None:
            return cached
    from .frames import frame_ids, read_frame
    det = Detector(labels, weights, conf)
    out, shape = {}, None
    for k in frame_ids(folder):
        rgb = read_frame(folder, k)[0]
        shape = rgb.shape[:2]
        out[k] = det(rgb)
    save_detections(path, out, labels, shape or (640, 480))
    log(f"detections for {labels} on {len(out)} frames ({np.median(det.ms):.0f} ms each) -> {path}")
    return out


# ---------------------------------------------------------------------------------------- commands
@contextlib.contextmanager
def in_models():
    """ultralytics downloads the base model and the text encoder into the current folder: make that models/."""
    MODELS.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(MODELS)
    try:
        yield
    finally:
        os.chdir(cwd)


def bake(labels, base=BASE):
    """Text prompts -> a YOLOE model with those classes built in (needs the CLIP text encoder)."""
    from ultralytics import YOLOE
    with in_models():
        model = YOLOE(f"{base}.pt")                         # downloaded the first time
        model.set_classes(labels, model.get_text_pe(labels))
    out = pt_path(labels, base)
    model.save(str(out))
    print(f"baked {labels} -> {out} ({out.stat().st_size / 1e6:.0f} MB). Copy it to the Jetson's {MODELS}/, then: "
          f"python -m xray.detect export \"{','.join(labels)}\"")
    return out


def export(labels, imgsz=IMGSZ, base=BASE):
    """The baked model -> a TensorRT FP16 engine for this board and image size."""
    from ultralytics import YOLOE
    pt = pt_path(labels, base)
    if not pt.exists():
        raise SystemExit(f"{pt} is missing: bake it on the laptop first (python -m xray.detect bake \"{','.join(labels)}\")")
    t = time.time()
    made = Path(YOLOE(str(pt)).export(format="engine", half=True, imgsz=list(imgsz), device=0))
    out = engine_path(labels, imgsz, base)
    made.replace(out)
    print(f"engine for {labels} at {imgsz[1]}x{imgsz[0]} (W x H) -> {out} in {time.time() - t:.0f} s")
    return out


def frames_in(path):
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    d = p / "color" if (p / "color").is_dir() else p
    files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return sorted(files, key=lambda f: (int(f.stem) if f.stem.isdigit() else 1 << 30, f.name))


def load_rgb(f):
    from PIL import Image
    return np.asarray(Image.open(f).convert("RGB"))


def bench(path, labels, n=30, weights=None, conf=0.2, imgsz=IMGSZ):
    files = frames_in(path)
    if not files:
        raise SystemExit(f"no images in {path}")
    det = Detector(labels, weights, conf, imgsz)
    pick = [files[int(i)] for i in np.linspace(0, len(files) - 1, min(n, len(files)))]
    for f in pick[:3]:
        det(load_rgb(f))                                    # warm-up (CUDA kernels, TensorRT contexts)
    det.ms.clear()
    det.speed.clear()
    seen = {lab: [] for lab in det.labels}
    counts = []
    for f in pick:
        dets = det(load_rgb(f))
        counts.append(len(dets))
        for d in dets:
            seen.setdefault(d.label, []).append(d.score)
    ms = np.array(det.ms)
    print("\n=== DETECT BENCH (send me this) ===")
    print(f"model: {det.weights.name} ({'TensorRT' if det.engine else 'PyTorch'}), image {imgsz[1]}x{imgsz[0]} (W x H), conf {conf}")
    print(f"frames: {len(pick)} of {len(files)} in {path}")
    print(f"time per frame: median {np.median(ms):.1f} ms, p95 {np.percentile(ms, 95):.1f} ms ({1000 / np.median(ms):.0f} fps)")
    parts = {k: np.median([s.get(k, 0) for s in det.speed]) for k in ("preprocess", "inference", "postprocess")}
    print("  of which (ultralytics): " + ", ".join(f"{k} {v:.1f} ms" for k, v in parts.items())
          + f", the rest (masks to numpy, Python) {np.median(ms) - sum(parts.values()):.1f} ms")
    print(f"detections per frame: median {np.median(counts):.0f}, max {max(counts)}")
    for lab, s in seen.items():
        print(f"  {lab!r}: {len(s)} detections" + (f", score {min(s):.2f}-{max(s):.2f} (median {np.median(s):.2f})" if s else ""))


COLORS = [(255, 200, 0), (0, 220, 255), (255, 0, 200), (0, 255, 120), (255, 120, 0), (120, 120, 255)]


def show(path, labels, weights=None, conf=0.2, imgsz=IMGSZ, limit=12, out_dir="runs/detect_show"):
    import cv2
    det = Detector(labels, weights, conf, imgsz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in frames_in(path)[:limit]:
        rgb = load_rgb(f)
        img = rgb[..., ::-1].copy()
        for d in det(rgb):
            col = COLORS[det.labels.index(d.label) % len(COLORS)] if d.label in det.labels else (200, 200, 200)
            img[d.mask] = (0.5 * img[d.mask] + 0.5 * np.array(col)).astype(np.uint8)
            x0, y0, x1, y1 = (int(v) for v in d.box)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, 1)
            cv2.putText(img, f"{d.label} {d.score:.2f}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                        cv2.LINE_AA)
        out = out_dir / f"{f.stem}_det.jpg"
        cv2.imwrite(str(out), img)
        print(f"{f.name}: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="YOLOE text-prompt segmentation (see the top of this file)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bake", "export", "bench", "show"):
        p = sub.add_parser(name)
        if name in ("bench", "show"):
            p.add_argument("path", help="an image, a folder of images, or a scan folder (uses color/)")
        p.add_argument("prompt", help='comma-separated labels, e.g. "solo cup,ball"')
        p.add_argument("--imgsz", type=int, nargs=2, default=list(IMGSZ), metavar=("H", "W"))
        if name in ("bench", "show"):
            p.add_argument("--weights", help="a .pt or .engine (default: the one for this prompt in models/)")
            p.add_argument("--conf", type=float, default=0.2)
        if name == "bench":
            p.add_argument("-n", type=int, default=30, help="frames to time")
        if name == "show":
            p.add_argument("--out", default="runs/detect_show", help="folder for the drawn images")
    args = ap.parse_args(argv)
    labels, imgsz = parse_prompt(args.prompt), tuple(args.imgsz)
    if args.cmd == "bake":
        bake(labels)
    elif args.cmd == "export":
        export(labels, imgsz)
    elif args.cmd == "bench":
        bench(args.path, labels, args.n, args.weights, args.conf, imgsz)
    else:
        show(args.path, labels, args.weights, args.conf, imgsz, out_dir=args.out)


if __name__ == "__main__":
    sys.exit(main())
