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
