"""
Rover faces the same direction as your head (left/right only). Laptop is the hub:
  Pico W (glasses) --Wi-Fi UDP 5007--> laptop --USB serial--> Uno (rover.ino) --> L298N

The rover has no heading sensor, so this ESTIMATES its heading from how long it has
been spinning (turn rate x time). You calibrate that rate once. It drifts a little,
so re-zero now and then: face the rover's direction and press the Pico's zero button
(or press Enter in this window).

Run:
  py head_follow.py --port COM4 --calibrate     (once; measures the rover's spin rate)
  py head_follow.py --port COM4                  (normal use)
"""
import argparse, json, os, socket, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
ap.add_argument("--calibrate", action="store_true")
args = ap.parse_args()

# ---- tuning ----
TURN_PWM = 170          # constant spin speed (constant speed = predictable rate). Raise if it stalls.
DEADBAND = 6.0          # deg: stop when rover is within this of your head direction
STOP_EARLY = 3.0        # deg: stop a bit early to allow for coasting
YAW_SIGN = 1            # flip to -1 if the rover turns the OPPOSITE way to your head
TIMEOUT = 0.5           # s without Pico data -> stop
RATE_FILE = "turn_rate.json"

import serial
ser = serial.Serial(args.port, 115200, timeout=0.1)
time.sleep(2)  # Uno resets when the port opens


def send(l, r):
    ser.write(f"{int(l)} {int(r)}\n".encode())


def spin(direction):
    """direction +1 = spin left (CCW), -1 = spin right, 0 = stop."""
    p = TURN_PWM * direction * YAW_SIGN
    send(-p, p)


# ---------------- calibration ----------------
if args.calibrate:
    secs = 3.0
    input(f"Rover on the floor, clear space around it. It will spin LEFT for {secs:.0f} s.\n"
          "Put a piece of tape on the floor under its nose so you can count turns. Press Enter...")
    end = time.time() + secs
    while time.time() < end:
        spin(+1)
        time.sleep(0.05)
    spin(0)
    deg = float(input("About how many DEGREES did it turn? (e.g. 1 full turn = 360, 1.5 turns = 540): "))
    rate = deg / secs
    json.dump({"deg_per_sec": rate, "pwm": TURN_PWM}, open(RATE_FILE, "w"))
    print(f"Saved: {rate:.0f} deg/s at PWM {TURN_PWM}. Now run without --calibrate.")
    send(0, 0)
    sys.exit()

if not os.path.exists(RATE_FILE):
    sys.exit("Run with --calibrate first.")
cal = json.load(open(RATE_FILE))
RATE = cal["deg_per_sec"]
if cal.get("pwm") != TURN_PWM:
    print("Warning: TURN_PWM changed since calibration; re-run --calibrate for accuracy.")

# ---------------- follow mode ----------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5007))
sock.settimeout(0.02)

state = {"rover_yaw": 0.0, "rezero": False}


def keyboard():
    while True:
        try:
            input()
        except EOFError:
            return
        state["rezero"] = True


threading.Thread(target=keyboard, daemon=True).start()
print(f"Following your head. Turn rate {RATE:.0f} deg/s. "
      "Press Enter (or the Pico button) to re-zero. Ctrl+C to stop.")

head_yaw, last_rx = 0.0, 0.0
direction = 0
last_t = time.time()
last_print = 0.0
prev_head = None
try:
    while True:
        try:
            data, _ = sock.recvfrom(256)
            msg = json.loads(data)
            head_yaw, last_rx = msg["yaw"], time.time()
            # Pico zero button: its yaw snaps back to 0 -> re-zero the rover estimate too
            if msg.get("zero") or (prev_head is not None and abs(prev_head) > 8 and abs(head_yaw) < 0.5):
                state["rezero"] = True
            prev_head = head_yaw
        except socket.timeout:
            pass

        now = time.time()
        dt = now - last_t
        last_t = now
        # integrate the rover's estimated heading for the time we were spinning
        state["rover_yaw"] += direction * RATE * dt

        if state["rezero"]:
            state["rover_yaw"] = 0.0
            state["rezero"] = False
            print("re-zeroed: rover and head both at 0")

        if now - last_rx > TIMEOUT:
            direction = 0
        else:
            err = head_yaw - state["rover_yaw"]
            if direction == 0:
                if abs(err) > DEADBAND:
                    direction = 1 if err > 0 else -1
            else:
                # keep turning until we're within STOP_EARLY of the target
                if (direction > 0 and err < STOP_EARLY) or (direction < 0 and err > -STOP_EARLY):
                    direction = 0
        spin(direction)

        if now - last_print > 0.2:
            last_print = now
            print(f"head {head_yaw:7.1f}  rover {state['rover_yaw']:7.1f}  "
                  f"{'LEFT ' if direction > 0 else 'RIGHT' if direction < 0 else 'hold '}")
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    send(0, 0)
    ser.close()
    print("stopped")
