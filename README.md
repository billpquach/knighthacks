# Permanence

**Persistent 3D object memory for robots, running entirely on an NVIDIA Jetson Orin Nano.**

Permanence gives a robot *object permanence*: it keeps a persistent, object-centric 3D world state that tracks
objects even when they are hidden, covered or carried inside containers, and it acts on that memory.
A 3D vision-language model (Qwen-3D) grounds a plain-language request to an object once. A real-time
open-vocabulary detector (YOLOE, TensorRT) then tracks every object through occlusion, and a rover drives to
targets that no camera can currently see.

> 1st Place Overall · 1st Place NVIDIA · 1st Place Hugging Face, Knight Hacks IX (UCF, October 2026, 200+ projects)
>
> [Devpost](https://devpost.com/software/permanence)

---

## Contents

- [Motivation](#motivation)
- [System overview](#system-overview)
- [Method](#method)
- [Performance](#performance)
- [Hardware](#hardware)
- [Installation](#installation)
- [Quick start without hardware](#quick-start-without-hardware)
- [Running on the robot](#running-on-the-robot)
- [Interfaces and API](#interfaces-and-api)
- [Configuration](#configuration)
- [Benchmarking and tests](#benchmarking-and-tests)
- [Repository layout](#repository-layout)
- [Troubleshooting](#troubleshooting)
- [Limitations and future work](#limitations-and-future-work)
- [Team](#team)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Motivation

Most robot perception acts on the latest camera frames. When an object is covered, carried off or stacked away,
it drops out of the robot's world. Occlusion is the normal case in physical environments: grippers cover the parts
they grasp, items go into bins and drawers, people walk behind shelves. Large models also cannot run on every
frame on edge hardware.

Permanence addresses both problems with a two-stage design. A large model **understands once**, a small model
**tracks continuously**, and an explicit **world state** between them holds each object's identity and position
through occlusion and containment.

---

## System overview

```
iPhone with LiDAR (Record3D over USB): RGB, depth, confidence, intrinsics, ARKit pose at 15 fps
        |
        v
NVIDIA Jetson Orin Nano 8 GB (unified CPU/GPU memory)
  1. Scene capture ...... rover stop-and-shoot scan -> keyframes -> 3D scene and object twins
  2. Grounding .......... Qwen-3D: "the cup on top of the box" -> one 3D target, then the model exits
  3. Detection .......... YOLOE (TensorRT FP16): labelled masks on every frame
  4. World state ........ 3D tracker with occlusion and containment reasoning
                          per-object state: VISIBLE / OCCLUDED / OUT_OF_VIEW / INSIDE / LOST
  5. Action ............. navigator -> serial -> Arduino Uno + L298N -> motors
  6. State server ....... SSE /state, MJPEG /stream.mjpg, /frame.jpg, /scene, /metrics, POST /prompt
        |
        +--> Three.js dashboard (browser; "hey rover" voice commands)
        +--> AR glasses HUD (web page)
```

The grounding model is slow but expressive. The detector is fast but only knows label words. The world state lets
the system resolve a request once and then hold that identity at camera rate, without re-running the large model.
This is the same slow/fast split used by dual-system robot foundation models, applied to perception memory.

---

## Method

### 1. Scene capture

The phone streams RGB, LiDAR depth, a confidence map, intrinsics and its ARKit camera pose over USB through
Record3D. A scan collects keyframes in a gravity-aligned world frame (z up, floor at z = 0).

- **Rover scan** (`--rover-scan`): stop and shoot. The rover turns in 20° steps across ±40°, rolls 15 cm and
  repeats. It keeps one view per stop, taken after the phone stops swaying on its mount, for about 18 sharp views
  in roughly 25 s.
- **Handheld scan:** a frame becomes a keyframe after the camera moves 10 cm or turns 15°, unless it is much
  blurrier than the keyframes so far.

### 2. Language grounding: Qwen-3D on an 8 GB Jetson

[Qwen-3D](https://github.com/ll220/qwen3d) is a 3D visual-grounding model built on Qwen2.5-VL-3B, with a trained
connector and a 3D mask decoder. Given a scan and a sentence, it returns the referred object's 3D points in a
single forward pass, without text generation.

- **Scene tokens.** 15 keyframes are spread over the scan. Each vision token (one per 28×28 pixels after the 2×2
  merge) is lifted to the 3D point under it using depth and pose, then averaged into 5 cm voxels (about 3,000
  tokens). The voxels enter the language model with 3D position encoding.
- **Masks and scores.** The connector and mask decoder predict masks on a dense target cloud: 2 cm voxels of
  confident LiDAR points, with features interpolated from the 8 nearest voxel tokens. The score is the mask
  probability at the tokens naming the object, parsed from the request ("cup" in "the cup next to the box").
  Each answer reports the top-5 candidates and a confidence band (high ≥ 0.5, low ≥ 0.3).
- **Edge port.** Scene preparation is reimplemented in plain PyTorch, without the original code's compiled CUDA
  extensions. `ground3d/upstream.py` imports the upstream model for inference only and copies none of its code.
- **8-bit weight quantization** (`ground3d/quant.py`):
  - Decoder and vision-encoder linear layers are stored as int8 with one bf16 scale per output row (absmax). Each
    layer expands to bf16 only for its own matmul. The token-embedding table is int8 as well, and only the rows
    looked up are expanded.
  - An optional 4-bit mode (groups of 64 weights, each with a scale and a minimum) saves about 1.2 GB more.
- **Streaming load.** The model is built with empty weights, then the safetensors are read one tensor at a time
  and quantized on the way in, 4,096 rows at a time, so no full-precision copy ever exists. Each file is dropped
  from the page cache (`posix_fadvise(DONTNEED)`), because on the Jetson the page cache shares RAM with the GPU.
- **Activation memory.** The MLP runs over 1,024 tokens at a time, the vision encoder is unloaded once the scan
  is encoded, and preparation falls back from 15 to 10 frames if memory runs out.
- **Process isolation.** The model runs in its own server process (`ground3d.ground --serve`), driven through
  job and result files with an atomic rename. It exits at the handoff to tracking, which frees all of its memory,
  CUDA context included. A watchdog ends it if the process that started it dies, and the live runner clears
  servers left over from crashed runs before loading.

### 3. Real-time detection: YOLOE on TensorRT

YOLOE-11s-seg provides open-vocabulary detection and instance segmentation. The text prompts (for example
`"solo cup,ball,box"`) are baked into the weights, so no text encoder runs on the Jetson. The baked model is
exported on the Jetson to a TensorRT FP16 engine with a fixed 640×480 input and no letterboxing.

### 4. World state: 3D tracking with object permanence

`xray/track.py` maintains every object's 3D position, velocity, size and state.

- **2D to 3D.** Each mask is eroded by 3 px. Only the nearest depth layer inside it is kept, which rejects
  background leaking through mask edges, and only confident, above-table points count. The center estimate is
  pushed back along the viewing ray by 0.75 × the object's radius, because the camera sees only the near surface.
- **Association.** Detections are matched to objects with the Hungarian algorithm, per label, on
  √(dxy² + (0.3·dz)²), since height is the noisiest axis. Each object's gate widens with its speed and its time
  unseen, and leaving a track unmatched has a fixed cost, so no detection is forced onto a track.
- **Filtering.** An alpha-beta filter tracks position and velocity, with steady gains for still objects and fast
  gains for moving ones.
- **Explaining missed detections.** The depth image at the predicted location decides the state: off the image is
  OUT_OF_VIEW, something nearer in front is OCCLUDED, and otherwise the object becomes LOST after 0.6 s. A state
  changes only after 3 consecutive frames agree, and confidence decays at a different rate in each hidden state.
- **Recovery.** Hidden objects are reacquired after repeated confident detections near their last position. An
  object moved while hidden is matched to the longest-unseen object of its label.

### 5. Containment (`xray/shell.py`)

- **Ball identity.** Balls are identified by color: the circular mean hue of their saturated pixels.
- **Going inside.** A ball becomes INSIDE a cup when it is unseen while fully in view, under a resting, nearly
  still cup. From then on it moves with that cup, whether or not the cup is visible.
- **Lifts and reveals.** Lifts are detected from cup height. A reveal compares the cup the system named with the
  cup that actually covered the ball, and reports RIGHT or WRONG.

### 6. Digital twins and the x-ray view

Dense object clouds are collected from the scan. Cups are completed as cylinders and boxes as cuboids, which fills
in sides the camera never saw. The x-ray view (`xray/render.py`) draws each twin at its remembered pose, through
whatever occludes it.

### 7. Action (`xray/rover.py`)

- **Positioning.** The rover drives toward targets using the phone's pose; it has no wheel encoders. It routes
  around other objects, including ones it cannot currently see, through approach points. It creeps in to touch
  cups and stops short of balls.
- **Pace control.** A pace controller measures turn rate and speed, then adjusts start-up and running power,
  because low-cost TT motors stall when started too gently. A self-check swaps left and right in software if turns
  go the wrong way.
- **Safety.** The Jetson resends the motor command every 100 ms and stops the motors if the control loop is silent
  for 0.5 s. The Arduino firmware also stops the motors 300 ms after the last command.

---

## Performance

Measured on a Jetson Orin Nano Developer Kit (8 GB), with the desktop disabled and `jetson_clocks` enabled.

| Quantity | Value |
|---|---|
| Qwen-3D weights, bf16 → int8 (this repository's quantization) | 7.1 GB → 4.5 GB |
| Qwen-3D model load (done before the demo with `--wait`) | ~100 s |
| Scan encoding (vision encoder, runs in the background) | ~20 s |
| Grounding query | ~4.5 s |
| YOLOE TensorRT FP16 engine | 22 MB, keeps up with the 15 fps camera |
| Both models resident | ~6.4 GB of ~7.4 GB visible to the OS |
| Rover scan | ~18 views in ~25 s |

Grounding is most reliable when a request uses a landmark ("the cup next to the box") and less reliable with
viewer-relative words (left, right, middle). The 4-bit mode changed answers in side-by-side tests, so demos use
8-bit.

---

## Hardware

| Component | Used | Notes |
|---|---|---|
| Compute | NVIDIA Jetson Orin Nano Developer Kit, 8 GB | JetPack 6; runs every model |
| Sensor | iPhone with LiDAR and the [Record3D](https://record3d.app/) app | USB into a Jetson USB-A port; USB streaming at 15 fps |
| Rover | 2 TT motors, L298N driver, Arduino Uno, ~9 V battery | Firmware in `rover/rover.ino` |
| Operator station | Any laptop with Chrome | Dashboard and voice commands |
| AR display (optional) | XREAL glasses with a Beam Pro | Web HUD (`viewer/public/hud.html`) |

**Rover wiring** (also at the top of `rover/rover.ino`):
- **Motor A (left):** ENA/IN1/IN2 to D5/D7/D8, motor on OUT1/OUT2.
- **Motor B (right):** ENB/IN3/IN4 to D6/D9/D10, motor on OUT3/OUT4.
- **Jumpers:** ENA/ENB jumpers off, 5V-EN jumper on.
- **Power:** battery on 12V and GND, and L298N GND tied to Uno GND.
- **Serial protocol:** `"<left> <right>\n"` (−255…255) at 115200 baud.

---

## Installation

### Jetson

1. **JetPack 6 and a virtual environment.** Flash JetPack 6 (CUDA and TensorRT included). Create the environment
   with access to JetPack's TensorRT bindings:

   ```bash
   python3 -m venv --system-site-packages ~/venvs/permanence
   source ~/venvs/permanence/bin/activate
   ```

2. **PyTorch.** Install NVIDIA's PyTorch build for your JetPack (see NVIDIA's *PyTorch for Jetson* instructions),
   then the Python dependencies:

   ```bash
   pip install "transformers==4.51.1" accelerate safetensors ultralytics record3d pyserial numpy opencv-python
   sudo usermod -aG dialout $USER     # serial access to the Arduino; log out and back in
   ```

3. **Clone this repository.**

   ```bash
   git clone https://github.com/billpquach/permanence.git
   cd permanence/core
   ```

4. **Qwen-3D code and weights.** The weights are not distributed with this repository. The grounding runner
   expects the upstream checkout and a model folder:

   ```text
   ~/qwen3d/                    git clone https://github.com/ll220/qwen3d ~/qwen3d
   ~/models/qwen3d/
     qwen/                      the fine-tuned Qwen2.5-VL-3B weights: *.safetensors (+ index) and config.json
     heads.pth                  the Qwen-3D checkpoint; only connector, text_connector and mask_decoder are read
     processor/                 the model's processor files (its tokenizer defines <|pointcloud_pad|>)
   ```

   Other locations can be given with `--upstream` and `--models` (or `--weights`, `--heads` and `--processor`),
   passed to the live runner through `--qwen-args`.

5. **YOLOE detector.** Baked prompt models for `"solo cup,ball"` and `"solo cup,ball,box"` are included in
   `models/`. Export the one you need to TensorRT on the Jetson (about 10 minutes; engines are specific to the
   board, the TensorRT version and the input size):

   ```bash
   python -m xray.detect export "solo cup,ball,box"
   ```

   To bake a new prompt set, use a laptop. It needs the MobileCLIP text encoder, about 570 MB:

   ```bash
   pip install ultralytics "git+https://github.com/ultralytics/CLIP.git"
   python -m xray.detect bake "solo cup,ball,box"      # then copy the .pt to the Jetson's models/
   ```

### Rover firmware

Upload `rover/rover.ino` with the Arduino IDE, connect the Uno to a Jetson USB-A port (it appears as
`/dev/ttyACM0`), then check it from `core/`:

```bash
python -m xray.rover ping            # optional: the firmware answers
python -m xray.rover test            # each wheel forward, then back
python -m xray.rover drive 140 140 1 # straight ahead for 1 s
```

### Dashboard (optional rebuild)

The built dashboard is committed in `viewer/dist/` and served by the Jetson. To change it, edit `viewer/src/`, then
rebuild with Node 22.12+ (`cd viewer && npm ci && npm run build`) and commit `viewer/dist/`.

---

## Quick start without hardware

`replay_server.py` serves the same endpoints as the Jetson with synthetic, clearly labelled states. Use it to
develop the dashboard or other clients.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-replay.txt
.venv/bin/python replay_server.py          # then open http://localhost:8080
```

See [viewer/README.md](viewer/README.md) for the state-stream and point-cloud contract.

---

## Running on the robot

### Preflight (once after boot)

```bash
sudo systemctl isolate multi-user.target   # desktop off: frees memory for Qwen-3D
sudo jetson_clocks
free -h                                    # about 1 GB used
```

Mount the phone on the rover facing forward at about 35 cm and start Record3D's USB streaming at 15 fps, with
Auto-Lock off.

### Full pipeline

From `core/`:

```bash
python -m xray.live --wait --ask --ball --rover --rover-scan --prompt "solo cup,ball,box" \
    --video --out runs/demo 2>&1 | tee -i demo.log
```

`--wait` loads the models first and shows READY (about 2 minutes). Then:

1. **"scan"**: the rover scans the scene by itself and the map fills in.
2. **A request**, for example *"the cup on top of the box"*: Qwen-3D answers with its confidence and runners-up.
   The newest answer becomes the target.
3. **"go"**: Qwen-3D exits and frees its memory, and YOLOE tracking starts.
4. Hide or shuffle objects: the world state follows them, and hidden objects are drawn through their covers.
5. **"drive"**: the rover drives to the target, routing around other objects.

Steps 2–5 can be one command: *"drive to the cup next to the box"* asks Qwen-3D, counts down for 3.5 s (say
"stop" to cancel), hands over to tracking and drives.

Press Ctrl+C once to end a run and wait for the output line; with `--video` the video is finalized first.

### Commands

Commands can be spoken in the dashboard after the wake word "hey rover", typed in the runner's console, or sent
with `curl -X POST http://<jetson-host>:8080/prompt -d '<command>'`.

| Command | Phase | Effect |
|---|---|---|
| `scan` | ready (`--wait`) | Starts the scan |
| *a description* | ask | Qwen-3D grounds it; the newest answer is the target |
| `go` | ask | Ends questions; Qwen-3D exits, tracking starts |
| `drive to <description>` | ask | Qwen-3D picks the target, then a 3.5 s countdown, then drive |
| `stop` / `cancel` | ask countdown | Cancels the pending drive |
| `drive` | track | Drives to the current target |
| `drive to cup 2`, `go to the green ball`, `drive to the box` | track | Drives to a named object (a ball under a cup means that cup) |
| `stop` | track | Stops the rover |

### Other modes

```bash
python -m xray.live --wait --ball --rover --rover-scan                # YOLOE only: no grounding step
python -m xray.live --scan-seconds 20 --ask --ball --video            # handheld scan, no rover
python -m xray.live --scan <rec>/scan --replay <rec>/track --ball     # a recording instead of the phone
python -m xray.track <rec>/track --scan <rec>/scan --ball             # tracker only -> states.jsonl
python -m ground3d.ground <scan> -q "the cup next to the box"         # grounding on a saved scan
```

Runs are written to `runs/<name>/`: the scan, `states.jsonl` (every frame's state), and `xray.mp4` with `--video`
or replayable frames with `--record`.

### Between runs

Memory can be reclaimed without a reboot:

```bash
pkill -9 -f xray.live; pkill -9 -f ground3d.ground
sync; echo 3 | sudo tee /proc/sys/vm/drop_caches
free -h
```

---

## Interfaces and API

The live runner (`xray.live`) and `replay_server.py` serve the same HTTP API on port 8080, with CORS on every
endpoint.

| Endpoint | Description |
|---|---|
| `GET /state` | Server-sent events: the newest state message, up to 10/s; on a slow link fewer, never stale. `?cloud=0` omits point clouds |
| `GET /state.json` | The newest state message, once |
| `GET /frame.jpg` | The newest x-ray image (JPEG); `?after=SEQ` waits for a newer one; `X-Seq` and `X-Age-Ms` headers |
| `GET /stream.mjpg` | The x-ray view as an MJPEG stream |
| `GET /scene` | The scan: table, colored cloud and each object's twin |
| `GET /metrics` | Jetson resource usage (memory, GPU, CPU, temperature, power via `tegrastats`) |
| `POST /prompt` | A command or question (plain text, or JSON `{"text": ...}`) |
| `GET /` | The dashboard (`viewer/dist`); `/hud.html` is the AR HUD |

**Dashboard.** Open it through an SSH tunnel so the browser allows the microphone for voice commands:

```bash
ssh -N -L 8080:localhost:8080 <user>@<jetson-host>      # then http://localhost:8080
```

**AR HUD.** Open `http://<jetson-host>:8080/hud.html` on the device that drives the glasses. Options:
- `mic=0`: no microphone (leaves voice to the dashboard).
- `video=0`: text only.
- `speak=0`: no spoken replies.
- `size=1.2`: scales the text.

---

## Configuration

Main `xray.live` options (`python -m xray.live -h` for all):

| Flag | Default | Description |
|---|---|---|
| `--ask` | off | Load Qwen-3D; ask questions after the scan; `go` to track |
| `--wait` | off | Load models first; start the scan on Enter or `scan` |
| `--ball` | off | Containment game: balls under cups |
| `--prompt` | `"solo cup,ball"` | Detector labels (needs a matching engine) |
| `--rover [PORT]` | off | Drive the rover (auto-detects the port; `dry` logs commands only) |
| `--rover-scan` | off | The rover performs the scan by itself |
| `--rover-size FRONT WIDTH` | `0.11 0.15` | Phone-to-front-edge distance and rover width (m) |
| `--rover-power TURN DRIVE MIN` | `115 120 100` | Starting motor power; the pace controller adapts from there |
| `--rover-forward-offset DEG` | measured | Phone mount angle relative to the rover's forward direction |
| `--qwen-stay` | off | Keep Qwen-3D loaded during tracking (use with `--qwen-args "--bits 4"`) |
| `--qwen-args` | | Extra arguments for the grounding server (`--models`, `--bits 4`, ...) |
| `--scan-seconds` / `--min-keyframes` | `20` / `20` | Handheld scan length and minimum views |
| `--scan` / `--replay` | | Reuse a scan folder / replay a recorded track |
| `--out` | `runs/live_<time>` | Output folder |
| `--video` / `--record` | off | Save the x-ray video / raw frames |
| `--port` | `8080` | HTTP port |

---

## Benchmarking and tests

**Grounding benchmark.** `ground3d.bench` measures load time, scene preparation, per-query time, peak memory and,
with `tegrastats`, GPU load, power and temperature per phase. It also checks answers against a cases file:

```bash
python -m ground3d.bench --cases ground3d/cases.example.json --frames 15 10
```

**Detector speed** on a folder of frames:

```bash
python -m xray.detect bench <folder> "solo cup,ball" -n 30
```

**Tests.** Python tests run from the repository root with `pytest tests/`. Dashboard tests run from `viewer/` with
`npm test`, and browser tests with `npm run test:browser`.

---

## Repository layout

```text
core/
  xray/            live runner, phone input, scan, detector, tracker, containment, rendering, rover control
  ground3d/        Qwen-3D grounding server, 8-bit/4-bit quantization, scene preparation, twin refinement, benchmark
  jetson/          tegrastats sampling and the /metrics server
viewer/            Three.js dashboard (src/, built dist/) and the AR HUD (public/hud.html)
rover/             Arduino firmware for the L298N rover
head-rover/        head-tracked steering (Pico W + GY-521) and rover firmware variants
models/            baked YOLOE prompt models (TensorRT engines are built on the Jetson)
replay_server.py   hardware-free stand-in for the Jetson's API
receiver/          standalone Record3D recorder and forwarder
glasses/           experimental Unity project for XREAL (superseded by the web HUD)
tests/             Python tests
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `out of memory` (NVML/NvMap) while loading Qwen-3D | Stop leftover processes and drop caches (see *Between runs*); keep the desktop off |
| `no frame from the phone` | Record3D open and streaming over USB (not Wi-Fi), phone awake and trusted; replug |
| Dashboard shows "Camera unavailable" | Check `http://<host>:8080/frame.jpg` directly; on a slow link, open the HUD with `video=0` |
| Wheels hum without moving | Fresh batteries, or raise `--rover-power` |
| Rover turns the wrong way | `python -m xray.rover drive -130 130 1` must turn left. If it turns right, try `--swap 1` on that command and set `SWAP_SIDES` in `xray/rover.py`; while driving, a self-check also swaps the sides automatically |
| Grounding picks the wrong object | Say `stop` during the countdown and use a landmark in the request |
| Voice is not recognized | Use the dashboard through `localhost` (SSH tunnel); voice needs internet for Chrome's speech service |

---

## Limitations and future work

**Limitations**
- Containment and permanence rules are hand-written, not learned.
- The system has been tested on tabletop scenes with a handful of objects.
- Grounding is weak on viewer-relative words.
- Camera pose comes from the phone's ARKit.
- The 8-bit weights reduce memory, not compute.
- Voice recognition uses an online service.

**Future work**
- A learned world model for predicting unseen motion, evaluated against the explicit state.
- Training data from logged occlusion episodes (`states.jsonl`).
- Real-to-sim evaluation in Isaac Sim.
- On-device SLAM (Isaac ROS cuVSLAM) instead of phone pose.
- TensorRT-LLM or fused int8 kernels for grounding.
- Keeping the grounding model resident on larger Jetson modules.
- Re-observing objects whose confidence has decayed.

---

## Team

| | Role |
|---|---|
| **Bill Quach** ([GitHub](https://github.com/billpquach)) | AI and perception pipeline: Jetson deployment, Qwen-3D port and quantization, YOLOE TensorRT, tracking and containment, digital twins, live runner, rover autonomy, voice |
| **Isabelle Mathew** ([LinkedIn](https://www.linkedin.com/in/isabelle-mathew-21159a3ab/)) | Rover hardware and firmware; head-tracked steering |
| **Sebastian Noel** ([GitHub](https://github.com/sebastian-noel)) | Three.js dashboard: live viewer, memory map, resource monitoring, object inspection |
| **Alejandro Jaimes** ([LinkedIn](https://www.linkedin.com/in/alejaimes/)) | AR glasses HUD |

---

## Acknowledgments

- **Models and tools:**
  - [Qwen-3D](https://github.com/ll220/qwen3d) and the [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) team.
  - [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOE.
  - [Record3D](https://record3d.app/) for RGB-D streaming from the iPhone.
- **Event:** built at Knight Hacks IX. Thanks to the organizers, and to NVIDIA and Hugging Face for their
  challenges.

## Citation

```bibtex
@misc{permanence2026,
  title        = {Permanence: Persistent 3D Object Memory for Robots on the Edge},
  author       = {Quach, Bill and Mathew, Isabelle and Noel, Sebastian and Jaimes, Alejandro},
  year         = {2026},
  howpublished = {Knight Hacks IX},
  url          = {https://github.com/billpquach/permanence}
}
```

## License

This repository does not yet include a license. Contact the authors before reusing the code.
