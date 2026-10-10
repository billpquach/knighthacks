# Robot Live Viewer

A local Three.js viewer for the Jetson's `/state` SSE stream (or `/state.json`, the newest
state once) and `/frame.jpg` x-ray camera feed. The default port is **8080**, defined in `replay_server.py`.
The default spatial view is a top-down memory map; the navigation control switches
to calibrated robot perspective. The interface supports split view, spatial view with camera inset, and camera
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
The server binds to loopback by default. Add `--host 0.0.0.0` only when deliberately
sharing with LAN clients; this exposes the recording and allows cross-origin reads.

For hot reload, leave the replay server running and execute `npm run dev` in
`viewer/`. Vite proxies the stream and scene endpoints to local port 8080.
Dependencies are bundled at build time; runtime requires no CDN or external
fonts. No cloud deployment is involved.

## Connect the Jetson

### Resource monitoring

The resource strip shows shared physical memory (CPU + GPU, not dedicated VRAM),
GPU and CPU utilization, temperature, board power, and swap, with 60-second
trends. Memory is shown in GiB. It uses the existing `core/jetson/jetson_stats.py`
sampler: `/proc/meminfo` for used/total memory and `tegrastats` for other fields.
It does not claim CUDA-process allocation, free dedicated VRAM, or GPU memory
bandwidth. Reference: [NVIDIA CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html#memory-management).

On the Jetson, from the repository root:

```bash
python3 -m core.jetson.metrics_server --host 0.0.0.0 --port 8083
```

In viewer connection settings, set **Jetson metrics URL** to
`http://<jetson-ip>:8083/metrics`. Leave it empty if the robot's existing server
already serves `/metrics` on the same origin. The standalone server needs only
Python's standard library; `tegrastats` must be on PATH with permissions to read
the sensors. Do not run the browser or whole viewer as root. Missing tegrastats
fields show `--` and PARTIAL; hardware access has not been verified yet.

The replay server also serves `/metrics`, detecting a Jetson device-tree model
before starting the sampler. A non-Jetson returns NOT CONNECTED with no laptop
resource values, even while playing a recording. Resource polling is independent
of camera/telemetry and display freeze, so live machine load remains current.
Polling is paused while the tab is hidden. Failed or stale readings clear values
and trends; old readings are never represented as current utilization.

`GET /metrics` returns version 1, `device`, `memory_kind: "shared"`, `available`,
`reason`, `sample_id`, `age_ms`, and `metrics` containing `ram_used_gb`,
`ram_total_gb`, `gpu_pct`, `cpu_pct`, `temp_c`, `power_w`, `swap_used_mb`.
The existing sampler's `_gb` keys contain GiB. Missing fields are null, not zero.
Consumers reject samples aged 5 seconds or unchanged IDs for 5 seconds. The
service uses one background sampler rather than a subprocess per request.

For the future live server, reuse one `Metrics` instance from
`core.jetson.metrics_server`, return `latest()` at `/metrics`, and call `stop()`
on shutdown. The standalone endpoint is read-only and permits CORS; it has no
authentication, so expose it only on a trusted LAN/hotspot, never the public web.

### Camera and state

Open the radio/connection control and enter `http://<jetson>:8080`. The camera
endpoint defaults to `/frame.jpg`, as requested. The source URL, layout, and
camera path are saved in browser local storage. The original Jetson server
does not need `/viewer-info`; that optional endpoint labels local replay.

The browser must reach the source over the LAN, hotspot, or Tailscale. For a
viewer hosted on a different origin, the Jetson must allow CORS on `/state`,
`/state.json`, `/scene` and `/frame.jpg` (the replay server allows any origin on every
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
value, never used as array indices. When `target_id` is null after reveal, the
arrow falls back to the yellow ball. An explicit local selection overrides that.

`GET /scene` supplies static geometry once at connection and again on a phase
transition to `track`. A missing endpoint leaves the state-only viewer functional.
Connection settings also accept a local scene JSON file, overriding server scene
loads until the next connection. The scene must use the same world as the states.

```json
{
  "version": 1, "frame": "world", "up": "z", "units": "m",
  "table": {"z": 0.67, "outline": [[0, 0], [1, 0], [1, 1]]},
  "cloud": {"points": [[0.1, 0.2, 0.67]], "colors": [[120, 160, 140]]},
  "objects": [{"id": 42, "label": "solo cup", "position": [0.5, 0.7, 0.73], "points": []}]
}
```

The environment is capped at 50k world-space points, with RGB bytes preserved
in the file but rendered dim grey-green. Object models are capped at 2k
translation-only offsets per object: IDs select the model, and the latest state
position translates it. Empty models render bounding boxes; the exporter does
not fabricate cup models. Tracked geometry is drawn on top of the static scan.
Static scan cups remain at their original locations when objects move.
Modeled occluders and live per-frame clouds are out of scope. The synthetic demo's
older optional geometry fields remain supported. The grid is only a reference.

Robot perspective uses `T_world_cam` with its y/z axes negated, without rotating
the world. The calibrated image is letterboxed to preserve its aspect ratio.
Optional `camera` on a scene (and `camera.intrinsics` on a state) accepts
`{width,height,fx,fy,cx,cy}`; state intrinsics take precedence. The exporter records
the first scan frame's lens. With no supplied lens, the G8 handoff calibration is
480x640, fx=fy=452, cx=240.7, cy=319.6. Other cameras must supply their own lens.
With `--track`, replay supplies each matched recorded frame's own pose and
intrinsics. This avoids using scan-time calibration or rounded stream poses for
camera registration. G8 camera/scene correspondence has been visually checked;
the static scan does not follow cups that move during tracking.

## Recorded replay

```bash
.venv/bin/python replay_server.py /path/to/sample_state_G8.jsonl --port 8080
```

JSONL is looped at 5 Hz with a REPLAY label. Camera images are unavailable for
state-only recordings: `/frame.jpg` returns 404 instead of displaying unrelated
synthetic frames. Multiple clients use the same elapsed-time frame index.
For full-rate G8 states, pass `--hz 15`. State-only playback uses fixed cadence.
With a track folder, recorded timestamps drive playback, looping at the recording
duration; `--hz` controls SSE delivery frequency, not playback speed. MP4 import
is not implemented.

Export and replay a private scan (never put recordings or generated scenes in Git):

```bash
.venv/bin/python viewer/tools/scene_export.py /private/G8/scan /private/G8/scene.json \
  --states docs/sample_state_G8.jsonl
.venv/bin/python replay_server.py docs/sample_state_G8.jsonl \
  --scene /private/G8/scene.json --port 8080
# Full-rate states with timestamp-matched camera frames:
.venv/bin/python replay_server.py /private/G8/states.jsonl --hz 15 \
  --scene /private/G8/scene.json --track /private/G8/track --port 8080
```

Track folders use the same RGB/intrinsic/pose/meta format as `xray.frames`.
Each state must match a frame within 50 ms; invalid timing or missing calibration
fails startup. `/frame.jpg?index=N` returns recorded RGB with a reconstructed
telemetry x-ray overlay for state index N; `/frame-raw.jpg?index=N` returns the
original JPEG. This is labeled REPLAY OVERLAY, not a recorded production overlay.
Select `/frame-raw.jpg` in connection settings to inspect unmodified RGB.
`X-State-Index` and `X-Frame-Time` expose synchronization. The browser waits for
the image to decode before committing its matching 3D state, including across
freeze/resume and loop boundaries. A missing image holds that paired display
and reports the camera error rather than pairing stale imagery with newer poses.

The exporter reuses `xray.frames`: every keyframe, every third pixel in each axis,
confidence 2, depth strictly between 0 and 3 m, one RGB sample per 1.5 cm voxel.
Table height is the peak 2 cm histogram bin between 0.4 and 1.2 m; outline is the
XY convex hull of points within 2 cm of that height. This simple estimator may
include other surfaces at the same height. Models are seeded from the first state
with empty point arrays for the tracker to fill later.

## Verification

```bash
# Repository root
.venv/bin/python -m unittest discover -s tests -p test_replay_viewer.py -v
.venv/bin/python -m unittest discover -s tests -p test_scene_export.py -v
# viewer/
npm test
npm run build
npx playwright install chromium
npm run test:browser
```

Browser tests require a synthetic replay server on 8080 (no state file). Override `VIEWER_URL` or
`CHROMIUM_PATH` for an existing server/browser. They cover desktop/mobile layout,
nonblank moving WebGL pixels, camera loading, target selection, freeze/resume,
source validation, legacy geometry, off-screen arrows, and offline state.
