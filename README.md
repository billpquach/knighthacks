Intelligent Spatial Awareness & Object Permanence
(NVIDIA winners: {
    Bill Quach
    Isabelle Mathew
    Sebastian Noel
    Alejandro Jaimes
})

## Robot viewer

The live visualization and camera interface is in [viewer/](viewer/README.md).
Build it with `npm ci && npm run build` from that folder, then run
`python replay_server.py` from the repository root and open
`http://localhost:8080`. Install `requirements-replay.txt` first if needed.
The local server supplies an explicitly labeled simulation until a Jetson URL
is connected. See the viewer README for the stream and point-cloud contract.

## Jetson pipeline

From `core/`, with the phone streaming over USB in Record3D:

```bash
python -m xray.live --scan-seconds 20 --ball --video     # scan the table, then the shell game
python -m xray.live --scan <scan folder> --results <ground3d output>   # act 1: Qwen-3D's answer is the target
```

It serves `/state`, `/state.json`, `/frame.jpg`, `/scene` and `/metrics` on port 8080 (the same endpoints as
`replay_server.py`), plus the viewer at `/` once it is built. It writes `states.jsonl` (and with `--video` the
x-ray view) to `runs/live_<time>/`. To work from a recording instead of the phone:

```bash
python -m xray.live --scan <rec>/scan --replay <rec>/track --ball          # the full live loop at recording speed
python -m xray.track <rec>/track --scan <rec>/scan --ball                  # tracker + game only -> states.jsonl
```

The stages: `xray.scan` (keyframes), `ground3d` (Qwen-3D INIT and `refine`: each object's digital twin),
`xray.detect` (YOLOE), `xray.track` (3D tracking with object permanence), `xray.shell` (balls under cups),
`xray.render` (the x-ray look) and `xray.live` (all of it, live).
