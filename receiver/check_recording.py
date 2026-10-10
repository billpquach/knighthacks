"""Sanity-check a recording made by receiver.py.

python check_recording.py recordings/test1
"""
import glob
import sys

import msgpack
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "recordings/test1"
files = sorted(glob.glob(f"{folder}/*.msgpack"))
if not files:
    raise SystemExit("No .msgpack files found")
print(f"{len(files)} frames")

poses, ts, valid, dmax = [], [], [], []
for i, f in enumerate(files):
    m = msgpack.unpackb(open(f, "rb").read(), raw=False)
    if i == 0:
        h, w = m["depth_shape"]
        print("depth shape", (h, w), "rgb shape", m["rgb_shape"], "conf", m["conf"] is not None)
        print("K_color fx fy cx cy:", [round(v, 1) for v in m["K_color"]])
        print("K_depth fx fy cx cy:", [round(v, 1) for v in m["K_depth"]])
    d = np.frombuffer(m["depth_mm"], dtype="<u2")
    poses.append(np.array(m["pose"]).reshape(4, 4))
    ts.append(m["timestamp"])
    valid.append((d > 0).mean())
    dmax.append(d.max())

poses = np.array(poses)
t = poses[:, :3, 3]
print(f"position range (m): x {np.ptp(t[:, 0]):.2f}  y {np.ptp(t[:, 1]):.2f}  z {np.ptp(t[:, 2]):.2f}")
print("pose is identity every frame:", bool(np.allclose(poses, np.eye(4))))
R = poses[:, :3, :3]
print("rotation det range:", float(np.linalg.det(R).min()), float(np.linalg.det(R).max()), "(should be ~1)")
print(f"depth valid fraction: mean {np.mean(valid):.2f}, min {np.min(valid):.2f}")
print(f"depth max (mm): median {int(np.median(dmax))}")
dt = np.diff(ts)
print(f"frame gap: mean {dt.mean() * 1000:.0f} ms, max {dt.max() * 1000:.0f} ms")