# Robot Live Viewer

A local Three.js viewer for the Jetson's `/state` SSE stream (or `/state.json`, the newest
state once) and `/frame.jpg` x-ray camera feed. The default port is **8080**, defined in `replay_server.py`.
The interface supports split view, spatial view with camera inset, and camera
view with spatial inset. Select an object to inspect its bearing and distance;
selection is local to the viewer and never commands the rover.

## Run locally

Requires Node 22.12+ and Python 3.10+. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-replay.txt
cd viewer
npm ci
npm run build
cd ..
.venv/bin/python replay_server.py --port 8080
```

Open `http://localhost:8080`. The replay server serves the built frontend and
simulated telemetry plus matching synthetic camera images. SIMULATION remains
visible throughout: these are generated fixtures, not captured robot data.
If 8080 is occupied, pass another port and open that address.

For hot reload, leave the replay server running and execute `npm run dev` in
`viewer/`. Vite proxies the three stream endpoints to local port 8080.
Dependencies are bundled at build time; runtime requires no CDN or external
fonts. No cloud deployment is involved.

## Connect the Jetson

Open the radio/connection control and enter `http://<jetson>:8080`. The camera
endpoint defaults to `/frame.jpg`, as requested. The source URL, layout, and
camera path are saved in browser local storage. The original Jetson server
does not need `/viewer-info`; that optional endpoint labels local replay.

The browser must reach the source over the LAN, hotspot, or Tailscale. For a
viewer hosted on a different origin, the Jetson must allow CORS on `/state`,
`/state.json` and `/frame.jpg` (the replay server allows any origin on every
response). To read image timing headers, also send:

```text
Access-Control-Allow-Origin: <viewer-origin>
Access-Control-Expose-Headers: X-Seq, X-Age-Ms
```

Use HTTP for a local HTTP Jetson connection; HTTPS pages cannot fetch insecure
HTTP streams. Source settings report that mismatch. Same-origin hosting avoids
CORS: serve `viewer/dist/` at `/` from the real viewer server while retaining
its `/state` and `/frame.jpg` handlers. That real server is not present in this
checkout, so it has not been modified or deployed.

Images are pulled one at a time with request timeouts and retries. SSE reconnects
automatically. No valid state for three seconds shows an offline warning while
preserving the previous view. Camera failure is reported independently. Freeze
holds both displays while incoming state continues to be received; resume jumps
to the latest state. The page does not save frames or recordings.

## Telemetry and geometry

The state contract is supported: world coordinates in meters, z up,
`camera.position`, `camera.forward` (or row-major 3x4 `T_world_cam`), `objects[]`,
`target_id`, and `ball`, including nested `ball.balls[]`. IDs are looked up by
value, never used as array indices. Camera pose drives the actual 3D viewpoint.
No supplied camera intrinsics are assumed, so the visualization uses a 66-degree
vertical field of view rather than claiming pixel registration with the image.

The documented original stream has object centers and sizes but **does not
provide point clouds or scene occluders**. Without them, the viewer draws
explicitly labeled BOUNDING BOXES. It cannot infer an unseen wall from a target's
OCCLUDED state. The following optional, additive fields are proposed by this
viewer and still need to be emitted by the Jetson's perception server:

```json
{
  "objects": [{
    "id": 42,
    "label": "solo cup",
    "state": "OCCLUDED",
    "position": [0.5, 0.7, 0.75],
    "size": [0.1, 0.1, 0.11],
    "point_cloud": {
      "frame": "object",
      "points": [[0.01, 0.02, 0.03], [-0.01, 0.02, 0.03]]
    }
  }],
  "scene": {
    "floor_z": 0.70,
    "point_cloud": {"frame": "world", "points": [[0.1, 0.2, 0.70]]}
  },
  "occluders": [{
    "id": "box-1",
    "label": "box",
    "position": [0.5, 0.7, 0.8],
    "size": [0.2, 0.2, 0.2]
  }]
}
```

`point_cloud` may also appear on each ball and occluder. Points must be finite
XYZ triples. `frame: "world"` uses absolute scan coordinates; `frame: "object"`
adds the object's position and assumes world-aligned axes (no implicit rotation).
Clouds with unspecified frames are ignored. Client point budgets are 12k per
tracked object and 50k per environment cloud; the producer should downsample
before transmission. The grid is a reference plane, not measured LiDAR.

Remembered objects are drawn through supplied occluders. An arrow points toward
the selected target; off-screen objects get directional edge labels. Unknown
target IDs are harmless. Color, containment, and object state come from telemetry.

## Recorded replay

```bash
.venv/bin/python replay_server.py /path/to/sample_state_G8.jsonl --port 8080
```

JSONL is looped at 5 Hz with a REPLAY label. Camera images are unavailable for
state-only recordings: `/frame.jpg` returns 404 instead of displaying unrelated
synthetic frames. Multiple clients use the same elapsed-time frame index.
No real recording or Jetson was available during initial implementation.

## Verification

```bash
# Repository root
.venv/bin/python -m unittest discover -s tests -p test_replay_viewer.py -v
# viewer/
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

Browser tests require the replay server on 8080. Override `VIEWER_URL` or
`CHROMIUM_PATH` for an existing server/browser. They cover desktop/mobile layout,
nonblank moving WebGL pixels, camera loading, target selection, freeze/resume,
source validation, legacy geometry, off-screen arrows, and offline state.
