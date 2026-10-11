# Intelligent Spatial Awareness & Object Permanence
Knight Hacks IX Hackathon Project
Team:
    Bill Quach,
    Isabelle Mathew,
    Sebastian Noel,
    Alejandro Jaimes

## Robot viewer

The live visualization and camera interface is in [viewer/](viewer/README.md).
The built viewer is committed in `viewer/dist/`, so `git pull` brings the current one; after changing
`viewer/src/`, rebuild with `npm ci && npm run build` from that folder and commit `viewer/dist/` too. Then run
`python replay_server.py` from the repository root and open
`http://localhost:8080`. Install `requirements-replay.txt` first if needed.
The local server supplies an explicitly labeled simulation until a Jetson URL
is connected. See the viewer README for the stream and point-cloud contract.

## Jetson pipeline

From `core/`, with the phone streaming over USB in Record3D at **15 fps** (faster than the Jetson tracks, frames queue up and the view falls further behind every second):

```bash
python -m xray.live --scan-seconds 20 --ball --video     # scan the table, then the shell game
python -m xray.live --scan-seconds 20 --ask --ball --video 2>&1 | tee -i live.log   # Qwen-3D questions, then "go"
python -m xray.live --scan <scan folder> --results <ground3d output>   # act 1: Qwen-3D's answer is the target
```

Scanning: circle the cups slowly (a quarter turn at least, half is better), about 40-60 cm from them and looking
down at about 45 degrees; the scan goes on until it has 20 views (`--min-keyframes`). Cups and balls the scan saw
from one side only are completed by symmetry (`ground3d.refine.complete_round`), so their twins are whole.

It serves `/state`, `/state.json`, `/frame.jpg`, `/scene` and `/metrics` on port 8080 (the same endpoints as
`replay_server.py`), plus the viewer at `/` once it is built. With `--ask`, Qwen-3D loads while the phone scans;
questions are typed in the console or sent with `curl -X POST <jetson>:8080/prompt -d 'the cup next to the ball'`,
and `go` hands over to YOLOE tracking (Qwen-3D exits first, so its memory is free). It writes `states.jsonl` (and with `--video` the
x-ray view) to `runs/live_<time>/`. To work from a recording instead of the phone:

```bash
python -m xray.live --scan <rec>/scan --replay <rec>/track --ball          # the full live loop at recording speed
python -m xray.track <rec>/track --scan <rec>/scan --ball                  # tracker + game only -> states.jsonl
```

### Rover

Two DC motors on an L298N, driven by an Arduino Uno over USB (`rover/rover.ino`) and steered by the phone's own
pose (ARKit, in the scan's world): no wheel encoders. Wiring and the serial commands are at the top of the sketch
(ENA D5, IN1 D7, IN2 D8, ENB D6, IN3 D9, IN4 D10; take the ENA/ENB jumpers off; common ground). Upload it with the
Arduino IDE, then on the Jetson (`pip install pyserial`; the Uno is `/dev/ttyACM0`, `sudo usermod -aG dialout $USER`):

```bash
python -m xray.rover ping                     # the Arduino answers
python -m xray.rover test                     # each wheel forward then back: fix directions in rover.ino if needed
python -m xray.rover drive 120 120 1          # straight ahead for 1 s (find the least power that moves it)
python -m xray.live --scan-seconds 25 --ball --rover --rover-scan    # scans by itself; "drive" fetches the target
```

While tracking, `drive` (typed, or `POST /prompt`) drives to the target (the hidden ball's cup, or Qwen-3D's
answer) and stops 12 cm short; `stop` stops. `--rover dry` logs the commands instead (any replay). `--rover-power
TURN DRIVE MIN` sets the wheel power. The Arduino stops the motors if the Jetson goes quiet for 0.4 s.

The stages: `xray.scan` (keyframes), `ground3d` (Qwen-3D INIT and `refine`: each object's digital twin),
`xray.detect` (YOLOE), `xray.track` (3D tracking with object permanence), `xray.shell` (balls under cups),
`xray.render` (the x-ray look) and `xray.live` (all of it, live).
