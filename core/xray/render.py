"""Object clouds in the x-ray style, one colour family for every object (show, the live overlay, the viewer).

    info = draw_cloud(img, points, K, T, depth=depth_image, centre=object_centre)

  * colour by depth within the object, in this view: nearest pale cyan, farthest blue
  * brightness by how squarely the surface faces the camera; the surface normal is taken as the horizontal
    direction from the object's vertical centre axis (straight up near the axis): front-facing bright,
    back-facing dim
  * points nearer than the object's median depth are drawn 3x3, the others 2x2 (nearest drawn last)
  * points hidden behind something nearer in the depth image: a faint pass
img is BGR (OpenCV) and drawn in place. Returns where the object is in the image (for a label).
"""
import numpy as np

NEAR = np.array([255.0, 255.0, 200.0])          # BGR: pale cyan
FAR = np.array([255.0, 90.0, 30.0])             # BGR: blue


def project(points, K, T):
    cam = (points - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return cam[:, 0] * K[0, 0] / zs + K[0, 2], cam[:, 1] * K[1, 1] / zs + K[1, 2], z


def colours(points, z, cam_pos, centre=None, axis_share=0.3):
    """Per-point BGR colour: depth within the object -> pale cyan..blue, times facing brightness."""
    zmin, zmax = float(z.min()), float(z.max())
    rel = (z - zmin) / max(zmax - zmin, 1e-6)
    col = NEAR[None] * (1 - rel[:, None]) + FAR[None] * rel[:, None]
    c = np.median(points, axis=0) if centre is None else np.asarray(centre, float)
    radial = points[:, :2] - c[:2]
    r = np.linalg.norm(radial, axis=1)
    big = np.percentile(r, 90) if len(r) else 0.0
    normal = np.zeros_like(points)
    side = r > axis_share * max(big, 1e-6)
    normal[side, :2] = radial[side] / r[side, None]
    normal[~side, 2] = 1.0                                   # near the axis: the top faces up
    to_cam = cam_pos[None] - points
    to_cam /= np.maximum(np.linalg.norm(to_cam, axis=1, keepdims=True), 1e-9)
    facing = np.clip((normal * to_cam).sum(1), 0, 1)
    return col * (0.35 + 0.65 * facing)[:, None]


def _splat(img, u, v, col, size, alpha=1.0):
    h, w = img.shape[:2]
    lo = -(size // 2)
    for dy in range(lo, lo + size):
        for dx in range(lo, lo + size):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            if alpha >= 1.0:
                img[vv[ok], uu[ok]] = col[ok]
            else:
                img[vv[ok], uu[ok]] = (1 - alpha) * img[vv[ok], uu[ok]] + alpha * col[ok]


def draw_cloud(img, points, K, T, depth=None, centre=None, hidden_alpha=0.25, tol=0.04):
    points = np.asarray(points, float).reshape(-1, 3)
    if not len(points):
        return dict(drawn=0, hidden=0, label_xy=None)
    u, v, z = project(points, K, T)
    h, w = img.shape[:2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = (z > 0.05) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not inside.any():
        return dict(drawn=0, hidden=0, label_xy=None)
    p, ui, vi, z = points[inside], ui[inside], vi[inside], z[inside]
    col = colours(p, z, T[:3, 3], centre)
    front = np.ones(len(p), bool)
    if depth is not None:
        dh, dw = depth.shape
        du = np.clip((ui * dw) // w, 0, dw - 1)
        dv = np.clip((vi * dh) // h, 0, dh - 1)
        d = depth[dv, du]
        front = (d <= 0) | (z <= d + tol)
    order = np.argsort(-z)                                   # far first, so near points end on top
    p, ui, vi, z, col, front = p[order], ui[order], vi[order], z[order], col[order], front[order]
    hid = ~front
    if hid.any():
        _splat(img, ui[hid], vi[hid], col[hid], 2, hidden_alpha)
    near = front & (z < np.median(z))
    far = front & ~near
    _splat(img, ui[far], vi[far], col[far], 2)
    _splat(img, ui[near], vi[near], col[near], 3)
    top = int(np.argmin(vi))
    return dict(drawn=int(front.sum()), hidden=int(hid.sum()), label_xy=(int(ui[top]), int(vi[top])))
