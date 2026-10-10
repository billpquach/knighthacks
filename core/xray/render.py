"""Object clouds in the x-ray style (show, the live overlay, the viewer): a sparse point cloud you can see the
real object through, with its far wall showing through the gaps. Density is the enemy; depth comes from
sparsity and shading.

    info = draw_cloud(img, points, K, T, depth=depth_image, centre=c, radius=r, target=True, order_seed=obj_id)

  * at most 1800 dots for the target, 900 for other objects, and no more than cover ~35% of the object's box
    on screen: the first n of a fixed random order per object (order_seed), so the dots don't flicker from
    frame to frame (the full cloud stays for geometry)
  * dots 2x2, the target's nearer half 3x3; blended (alpha 0.9 target, 0.65 others), drawn far first
  * colour by depth within the object (its 5th-95th percentile depth in this view), three stops (BGR):
    near (255,255,200) pale cyan -> mid (255,220,0) cyan -> far (255,130,20) azure
  * brightness: front-facing 0.62 + 0.5 x facing, back-facing a constant 0.55. The normal is the horizontal
    direction from the object's vertical axis (straight up within 0.5 x radius of it)
  * the object's own back wall is drawn (it shows through the gaps of the front wall); only points behind
    OTHER surfaces (the depth image nearer by more than the object's diameter + 2 cm) are hidden: faint
  * glow=True (hidden or remembered objects: OCCLUDED, LOST, under a box): the dot layer Gaussian-blurred
    (sigma 5 px) and added x1.6 over the image, then the dots on top
img is BGR (OpenCV), drawn in place.
"""
import cv2
import numpy as np

STOPS = np.array([[255.0, 255.0, 200.0], [255.0, 220.0, 0.0], [255.0, 130.0, 20.0]])     # BGR near, mid, far
_ORDERS = {}


def project(points, K, T):
    cam = (points - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return cam[:, 0] * K[0, 0] / zs + K[0, 2], cam[:, 1] * K[1, 1] / zs + K[1, 2], z


def subsample(points, n, order_seed=0):
    """The first n points of one fixed random order per cloud: the same dots every frame, and when n changes
    (the object grows or shrinks on screen) dots are only added or removed, never swapped."""
    if len(points) <= n:
        return points
    key = (order_seed, len(points))
    if key not in _ORDERS:
        if len(_ORDERS) > 256:
            _ORDERS.clear()
        _ORDERS[key] = np.random.default_rng(order_seed).permutation(len(points))
    return points[_ORDERS[key][:n]]


def ramp(rel):
    """0..1 -> BGR along near -> mid -> far."""
    rel = np.clip(rel, 0, 1)[:, None]
    lo = STOPS[0] + (STOPS[1] - STOPS[0]) * np.clip(rel * 2, 0, 1)
    hi = STOPS[1] + (STOPS[2] - STOPS[1]) * np.clip(rel * 2 - 1, 0, 1)
    return np.where(rel < 0.5, lo, hi)


def shade(points, z, cam_pos, centre, radius):
    """Per-point BGR: depth ramp within the object times the facing brightness."""
    z5, z95 = np.percentile(z, 5), np.percentile(z, 95)
    col = ramp((z - z5) / max(z95 - z5, 1e-6))
    radial = points[:, :2] - centre[:2]
    r = np.linalg.norm(radial, axis=1)
    normal = np.zeros_like(points)
    side = r > 0.5 * radius
    normal[side, :2] = radial[side] / r[side, None]
    normal[~side, 2] = 1.0                                   # near the axis: the top faces up
    to_cam = cam_pos[None] - points
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
    facing = (normal * to_cam).sum(1)
    bright = np.where(facing > 0, 0.62 + 0.5 * facing, 0.55)
    return np.clip(col * bright[:, None], 0, 255)


def _splat(layer, alpha, u, v, col, a, size):
    h, w = alpha.shape
    lo = -(size // 2)
    for dy in range(lo, lo + size):
        for dx in range(lo, lo + size):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            layer[vv[ok], uu[ok]] = col[ok]
            alpha[vv[ok], uu[ok]] = a[ok]


def screen_budget(points, K, T, shape, coverage=0.35, dot_px=6.0):
    """How many dots cover about `coverage` of the object's box on screen (a far, small object gets fewer)."""
    u, v, z = project(points[:: max(1, len(points) // 400)], K, T)
    ok = z > 0.05
    if not ok.any():
        return 0
    h, w = shape
    du = np.ptp(np.clip(u[ok], 0, w)) + 3
    dv = np.ptp(np.clip(v[ok], 0, h)) + 3
    return int(coverage * du * dv / dot_px)


def draw_cloud(img, points, K, T, depth=None, centre=None, radius=None, target=False, glow=False, order_seed=0,
               max_points=None, coverage=0.35, hidden_alpha=0.3):
    full = np.asarray(points, float).reshape(-1, 3)
    if not len(full):
        return dict(drawn=0, hidden=0, label_xy=None)
    c = np.median(full, axis=0) if centre is None else np.asarray(centre, float)
    if radius is None:
        radius = float(np.percentile(np.linalg.norm(full[:, :2] - c[:2], axis=1), 90)) or 0.03
    cap = max_points or (1800 if target else 900)
    pts = subsample(full, max(30, min(cap, screen_budget(full, K, T, img.shape[:2], coverage))), order_seed)
    u, v, z = project(pts, K, T)
    h, w = img.shape[:2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (z > 0.05) & (ui >= -2) & (ui < w + 2) & (vi >= -2) & (vi < h + 2)
    if not inside.any():
        return dict(drawn=0, hidden=0, label_xy=None)
    pts, ui, vi, z = pts[inside], ui[inside], vi[inside], z[inside]
    col = shade(pts, z, T[:3, 3], c, radius)
    hidden = np.zeros(len(pts), bool)
    if depth is not None:
        dh, dw = depth.shape
        d = depth[np.clip((vi * dh) // h, 0, dh - 1), np.clip((ui * dw) // w, 0, dw - 1)]
        hidden = (d > 0) & (z - d > 2 * radius + 0.02)       # behind another surface, not behind its own front
    order = np.argsort(-z)                                   # far first: the near wall ends on top
    pts, ui, vi, z, col, hidden = pts[order], ui[order], vi[order], z[order], col[order], hidden[order]
    a_dot = 0.9 if target else 0.65
    a = np.where(hidden, hidden_alpha, a_dot)
    near = (~hidden) & (z < np.median(z)) if target else np.zeros(len(z), bool)

    pad = 16 if glow else 3                                  # work in the object's crop of the image
    x0, x1 = max(0, ui.min() - pad), min(w, ui.max() + pad + 1)
    y0, y1 = max(0, vi.min() - pad), min(h, vi.max() + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return dict(drawn=0, hidden=0, label_xy=None)
    layer = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
    alpha = np.zeros((y1 - y0, x1 - x0), np.float32)
    uc, vc = ui - x0, vi - y0
    _splat(layer, alpha, uc[~near], vc[~near], col[~near], a[~near], 2)
    _splat(layer, alpha, uc[near], vc[near], col[near], a[near], 3)
    crop = img[y0:y1, x0:x1].astype(np.float32)
    if glow:
        halo = cv2.GaussianBlur(layer * alpha[..., None], (0, 0), 5)
        crop = np.minimum(crop + 1.6 * halo, 255)
    crop = crop * (1 - alpha[..., None]) + layer * alpha[..., None]
    img[y0:y1, x0:x1] = np.clip(crop, 0, 255).astype(np.uint8)
    top = int(np.argmin(vi))
    return dict(drawn=int((~hidden).sum()), hidden=int(hidden.sum()), label_xy=(int(ui[top]), int(vi[top])),
                shown=int(len(pts)), total=int(len(full)))
