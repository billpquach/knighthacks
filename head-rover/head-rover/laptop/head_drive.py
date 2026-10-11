"""
Drive the rover with your head. No Jetson needed: the laptop is the hub.

  Pico W (glasses) --Wi-Fi UDP 5007--> laptop --USB serial--> Uno --> motor driver

Controls ("head joystick"):
  - Turn your head left/right past the deadband -> rover spins that way.
    Farther you turn, faster it spins. Face forward -> it stops turning.
  - Tilt your head DOWN (nod forward) -> drives forward. Tilt UP -> backs up.
  - Press the Pico's zero button while looking straight ahead to re-center yaw.
    Pitch neutral is captured from the first second of data after this starts.

Run:
  pip install pyserial
  python head_drive.py --port COM3              (Windows; check Device Manager)
  python head_drive.py --port /dev/cu.usbmodemXXXX   (Mac)
  Add --no-drive to only print what it WOULD send (wheels-off safety check).

In the Pico's main.py, set JETSON_IP to this laptop's IP on the hotspot.
Windows Mobile Hotspot: the laptop is almost always 192.168.137.1.
Safety: if the Pico stops sending for 0.5 s, the rover stops.
"""
import argparse, json, socket, time

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=False)
ap.add_argument("--no-drive", action="store_true")
args = ap.parse_args()
if not args.no_drive and not args.port:
    raise SystemExit("give --port, or use --no-drive to test without the rover")

# ---- tuning ----
YAW_DEADBAND = 10.0      # deg of head turn before the rover reacts
YAW_FULL = 45.0          # deg of head turn for max turn speed
PITCH_DEADBAND = 8.0     # deg of tilt before driving
PITCH_FULL = 30.0        # deg of tilt for max drive speed
MIN_PWM = 120            # TT motors stall below roughly this; raise if it hums
MAX_TURN = 200
MAX_DRIVE = 220
YAW_SIGN = 1             # flip to -1 if the rover turns opposite to your head
PITCH_SIGN = 1           # flip to -1 if nodding down makes it reverse
TIMEOUT = 0.5

def shaped(value, deadband, full, max_pwm):
    """Deadband + linear ramp from MIN_PWM to max_pwm. Returns signed PWM."""
    mag = abs(value)
    if mag < deadband:
        return 0
    frac = min((mag - deadband) / (full - deadband), 1.0)
    pwm = MIN_PWM + frac * (max_pwm - MIN_PWM)
    return pwm if value > 0 else -pwm

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5007))
sock.settimeout(0.05)

ser = None
if not args.no_drive:
    import serial
    ser = serial.Serial(args.port, 115200, timeout=0.1)
    time.sleep(2)  # Uno resets when the port opens

def send(left, right):
    left, right = int(max(-255, min(255, left))), int(max(-255, min(255, right)))
    if ser:
        ser.write(f"{left} {right}\n".encode())
    return left, right

print("waiting for the Pico on UDP 5007... (hold your head level and still)")
pitch_samples, pitch0 = [], None
last_rx, yaw, pitch = 0.0, 0.0, 0.0
last_print = 0.0
try:
    while True:
        try:
            data, _ = sock.recvfrom(256)
            msg = json.loads(data)
            yaw, pitch, last_rx = msg["yaw"], msg["pitch"], time.time()
            if pitch0 is None:
                pitch_samples.append(pitch)
                if len(pitch_samples) >= 50:      # ~1 s of data
                    pitch0 = sum(pitch_samples) / len(pitch_samples)
                    print(f"pitch neutral = {pitch0:.1f} deg. Driving enabled.")
        except socket.timeout:
            pass

        if pitch0 is None or time.time() - last_rx > TIMEOUT:
            l, r = send(0, 0)
        else:
            turn = YAW_SIGN * shaped(yaw, YAW_DEADBAND, YAW_FULL, MAX_TURN)
            drive = PITCH_SIGN * shaped(pitch - pitch0, PITCH_DEADBAND, PITCH_FULL, MAX_DRIVE)
            # yaw > 0 = head turned left (CCW) -> spin left: left wheel back, right wheel forward
            l, r = send(drive - turn, drive + turn)

        if time.time() - last_print > 0.2:
            last_print = time.time()
            p = (pitch - pitch0) if pitch0 is not None else 0.0
            print(f"yaw {yaw:6.1f}  pitch {p:6.1f}  ->  L {l:4d}  R {r:4d}")
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    send(0, 0)
    if ser:
        ser.close()
    print("stopped")
