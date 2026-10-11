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

Two TT motors on an L298N, driven by an Arduino Uno over USB (`rover/rover.ino`) and steered by the phone's own
pose (ARKit, in the scan's world): no wheel encoders. Wiring (also at the top of the sketch): ENA/IN1/IN2 to D5/D7/D8
(left motor on OUT1/OUT2), ENB/IN3/IN4 to D6/D9/D10 (right motor on OUT3/OUT4), ENA/ENB jumpers off, 5V-EN jumper on,
battery (~9 V) on 12V and GND, and the L298N GND tied to the Uno GND. Clamp the motor wires firmly in the OUT
terminals; a wheel that spins the wrong way gets its two wires swapped. The serial line is `<left> <right>\n`
(-255..255) at 115200, the same as the existing rover firmware, so either sketch works. Upload it with the Arduino
IDE, then on the Jetson (`pip install pyserial`; the Uno is `/dev/ttyACM0`, `sudo usermod -aG dialout $USER`; on
Windows it is a COM port, found by itself or given as `--rover COM4`):

```bash
python -m xray.rover ping                     # the Arduino answers (optional: other firmware stays silent)
python -m xray.rover test                     # each wheel forward then back; drive -130 130 1 must turn left
python -m xray.rover drive 140 140 1          # straight ahead for 1 s (find the least power that moves it)
python -m xray.live --ball --rover --rover-scan    # scans by itself (about 25 s); "drive" fetches the target
```

While tracking, `drive` (typed, or `POST /prompt`) drives to the target (the hidden ball's cup, or Qwen-3D's
answer); `drive to cup 2`, `drive to the yellow ball` (its cup, if it is under one), `touch 1` pick one by name; `stop`
stops. It goes around the other cups and balls (via a point beside the target when the straight way is blocked), gently
touches a cup with its front edge, and stops 3 cm short of a ball in the open. `--rover-size FRONT WIDTH` (metres, default
0.11 0.15) is the phone-to-front-edge distance and the rover's width. `--rover dry` logs the commands instead (any replay). `--rover-power
TURN DRIVE MIN` sets the starting wheel power (default 115 120 100). From there the rover finds the power by itself,
from what the phone measures: a push to get the wheels going (a stalled motor only hums), then the least that keeps a
slow, steady pace (20 deg/s scanning, 25 deg/s turning to a target, 12 cm/s driving, 5 cm/s touching; it never
pushes a cup harder than what got it rolling). The L298N drops ~2 V, so the motors see ~7 V. The rover's scan stops
and shoots: 20 deg turns across 80 deg, a 15 cm roll, the same again, and one view at each stop once the phone has
stopped swaying (18 sharp views). The phone need not face exactly forward: `--rover-scan` measures the mount's angle on
its forward roll, or give it with `--rover-forward-offset DEG`. Left and right swapped (`drive -130 130 1` turns right): `SWAP_SIDES` in `xray/rover.py`
(`--swap 0|1` to try); while driving, a turn that goes the wrong way swaps them by itself. The Arduino stops the motors if
the Jetson goes quiet for 0.3 s; the Jetson resends ten times a second while driving.

The stages: `xray.scan` (keyframes), `ground3d` (Qwen-3D INIT and `refine`: each object's digital twin),
`xray.detect` (YOLOE), `xray.track` (3D tracking with object permanence), `xray.shell` (balls under cups),
`xray.render` (the x-ray look) and `xray.live` (all of it, live).
