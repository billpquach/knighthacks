"""
Bench/floor test for the rover from a laptop (stands in for the Jetson).
Streams commands at 10 Hz so the firmware's 300 ms safety stop doesn't kick in.

  pip install pyserial
  python drive_test.py --port COM3          (Windows)
  python drive_test.py --port /dev/ttyACM0  (Linux)
  python drive_test.py --port /dev/cu.usbmodem1101  (Mac)

Modes:
  default      runs a fixed sequence: forward, back, spin left, spin right
  --ramp       steps both wheels from 0 up to 255 to find the minimum PWM that moves
"""
import argparse, time
import serial

def hold(ser, left, right, seconds, label=""):
    if label:
        print(f"{label}: left={left} right={right} for {seconds}s")
    end = time.time() + seconds
    while time.time() < end:
        ser.write(f"{left} {right}\n".encode())
        time.sleep(0.1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--speed", type=int, default=180)
    ap.add_argument("--ramp", action="store_true")
    args = ap.parse_args()

    ser = serial.Serial(args.port, 115200, timeout=0.1)
    time.sleep(2)  # Uno resets when the port opens
    s = args.speed
    try:
        if args.ramp:
            for pwm in range(0, 256, 15):
                hold(ser, pwm, pwm, 1.0, f"ramp {pwm}")
        else:
            hold(ser, s, s, 1.5, "forward")
            hold(ser, 0, 0, 0.5, "stop")
            hold(ser, -s, -s, 1.5, "backward")
            hold(ser, 0, 0, 0.5, "stop")
            hold(ser, -s, s, 1.0, "spin left")
            hold(ser, 0, 0, 0.5, "stop")
            hold(ser, s, -s, 1.0, "spin right")
    finally:
        ser.write(b"0 0\n")
        ser.close()
        print("stopped")

if __name__ == "__main__":
    main()
