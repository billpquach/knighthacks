"""The rover: two DC motors on an L298N, driven by an Arduino Uno (rover/rover.ino) over USB serial, and steered by
the phone's own pose (ARKit, in the scan's world): no wheel encoders, no IMU. The phone is the rover's eyes and
its odometry.

    python -m xray.rover ping                    # is the Arduino there (and on which port)
    python -m xray.rover test                    # each wheel forward then back, 1 s each: check wiring, directions
    python -m xray.rover drive 140 140 1.5       # both wheels at 140 (of 255) for 1.5 s
    python -m xray.live --scan-seconds 20 --ball --rover --rover-scan      # the live runner drives it

With --rover the live runner listens for "drive" (or "fetch", "get it") on POST /prompt or the console while
tracking, and drives to the target: the tracker's target (the hidden ball's cup, or Qwen-3D's answer), at its
remembered position. "stop" stops. --rover-scan turns the rover through the scan by itself.

How it drives (Navigator): turn in place until the target is within ALIGN of straight ahead, then drive, steering
in proportion to the angle, slower when close, and stop GAP short of the target's edge. The phone sees the floor
from about 15 cm in front of it, so the last stretch is driven on the tracker's memory of where the target is.
The heading is the phone's forward direction projected on the floor, plus the mount's angle to the rover's forward
direction, measured during the scan's forward roll (--rover-scan) or given (--rover-forward-offset).

Serial protocol: "<left> <right>" lines, -255..255 (the earlier firmware's format, and rover/rover.ino's); "0 0"
stops. Replies are not needed (the earlier firmware sends none).

Safety: the Arduino stops the motors when it hears nothing for 0.3 s; here a command is resent every 0.1 s only
while the runner keeps refreshing it (so a stalled runner stops the rover too); Ctrl+C, an error or the end of the
run stop it; a drive gives up after TIMEOUT s.
"""
import argparse
import glob
import math
import sys
import threading
import time

import numpy as np


ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}      # Arduino, Arduino.org, CH340, FTDI, CP210x
SWAP_SIDES = True       # this rover's firmware and wiring take "<right> <left>": "-130 130" turned it right, not left


def find_port():
    """The Arduino's serial port (an Uno is /dev/ttyACM0 on Linux, a COM port on Windows), or None."""
    try:
        from serial.tools import list_ports
        ports = sorted(list_ports.comports(), key=lambda p: p.vid not in ARDUINO_VIDS)
        if ports and ports[0].vid in ARDUINO_VIDS:
            return ports[0].device
    except ImportError:
        pass
    ports = sorted(glob.glob("/dev/ttyACM*")) + sorted(glob.glob("/dev/ttyUSB*"))
    return ports[0] if ports else None


class Rover:
    """The motors. port: a serial device, "auto" (find it) or "dry" (no hardware: log the commands)."""

    def __init__(self, port="auto", swap=SWAP_SIDES, log=print):
        self.log, self.dry, self.ser, self.swap = log, port == "dry", None, swap
        self.turn_sign, self.turn_yaw, self.turn_checked = 0, None, True
        self.cmd, self.cmd_t, self.lock, self.io, self.closed = (0, 0), 0.0, threading.Lock(), threading.Lock(), False
        self.sent_log = 0.0
        if not self.dry:
            import serial                               # pip install pyserial
            port = find_port() if port == "auto" else port
            if port is None:
                raise SystemExit("no Arduino found: plug it in (and close the Arduino Serial Monitor), or use --rover dry")
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.2)
            except serial.SerialException as e:
                raise SystemExit(f"can't open {port}: {e} (permission? sudo usermod -aG dialout $USER, then log in again; "
                                 f"or sudo chmod 666 {port})")
            time.sleep(2.0)                             # opening the port resets the Uno
            self.ser.reset_input_buffer()
            answered = self.ping()
            self.log(f"rover: Arduino on {port}" + ("" if answered else " (no answer to ping: fine with the earlier "
                                                                          "firmware, which doesn't reply)"))
        else:
            self.log("rover: dry run (no motors; commands are logged)")
        threading.Thread(target=self._keepalive, daemon=True).start()

    def _send(self, line):
        if self.dry:
            if line == "0 0" or time.monotonic() - self.sent_log > 0.5:   # (not every resend)
                self.sent_log = time.monotonic()
                self.log(f"rover (dry): {line}")
            return
        try:
            with self.io:                               # (the runner and the keepalive both send)
                self.ser.write((line + "\n").encode())
                if self.ser.in_waiting:
                    self.ser.reset_input_buffer()       # (replies, if the firmware sends any, aren't needed)
        except Exception as e:                          # a lost cable: the Arduino's watchdog stops the motors
            self.log(f"rover: serial error: {e}")

    def ping(self):
        if self.dry:
            return True
        with self.io:
            self.ser.reset_input_buffer()
            self.ser.write(b"P\n")
            t = time.monotonic()
            while time.monotonic() - t < 0.5:
                if self.ser.readline().strip() == b"pong":
                    return True
            return False

    def line(self, left, right):
        return f"{right} {left}" if self.swap else f"{left} {right}"

    def drive(self, left, right, yaw=None):
        """Wheel speeds -255..255 (positive forward). Call it every frame while driving: if the calls stop for
        0.5 s, the rover stops. yaw: the heading now (heading(T)), to check that turns go the commanded way."""
        left, right = int(np.clip(left, -255, 255)), int(np.clip(right, -255, 255))
        if yaw is not None:
            self.check_turn(left, right, yaw)
        with self.lock:
            changed = (left, right) != self.cmd
            self.cmd, self.cmd_t = (left, right), time.monotonic()
        if changed:
            self._send(self.line(left, right))

    def check_turn(self, left, right, yaw):
        """On each turn in place, once the phone has turned 15 deg: if it turned the other way than commanded, the
        left and right motors are swapped (wiring or firmware), so swap them here for the rest of the run."""
        sign = (1 if right > left else -1) if left * right < 0 else 0      # +1: a left turn (counterclockwise)
        if sign == 0:
            return                                      # (pauses between turn pulses, driving ahead)
        if sign != self.turn_sign:
            self.turn_sign, self.turn_yaw, self.turn_checked = sign, yaw, False
        elif not self.turn_checked:
            moved = wrap(yaw - self.turn_yaw)
            if abs(moved) > math.radians(15):
                self.turn_checked = True
                if moved * sign < 0:
                    self.swap = not self.swap
                    self.log("rover: a turn went the wrong way: left and right motors swapped, fixed in software "
                             f"(SWAP_SIDES = {self.swap} in xray/rover.py makes it permanent)")

    def stop(self):
        with self.lock:
            self.cmd, self.cmd_t = (0, 0), time.monotonic()
        self._send("0 0")

    def _keepalive(self):                               # resends the command so the Arduino's watchdog stays quiet
        while not self.closed:
            time.sleep(0.1)
            with self.lock:
                cmd, age = self.cmd, time.monotonic() - self.cmd_t
            if cmd == (0, 0):
                continue
            if age > 0.5:                               # nobody is refreshing it: stop
                self.stop()
                self.log("rover: no fresh command for 0.5 s, stopped")
            else:
                self._send(self.line(*cmd))

    def close(self):
        if self.closed:
            return
        self.stop()
        self.closed = True
        if self.ser is not None:
            self.ser.close()


# ---------------------------------------------------------------------------------------- steering
def heading(T):
    """The rover's heading (radians, world frame, z up): the phone's forward direction projected on the floor."""
    f = np.asarray(T)[:3, 2]
    return math.atan2(f[1], f[0])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class TurnRate:
    """How fast the rover has been turning (rad/s) over the last `window` s, from successive headings: turning only
    while it is under a limit makes the motors pulse, for an average turn slower than they can run steadily."""

    def __init__(self, window=0.5):
        self.window, self.history = window, []

    def __call__(self, yaw, now):
        self.history = [(y, t) for y, t in self.history if now - t <= self.window] + [(yaw, now)]
        t0, turned = self.history[0][1], 0.0
        for (ya, _), (yb, _) in zip(self.history, self.history[1:]):
            turned += wrap(yb - ya)
        return turned / (now - t0) if now > t0 else 0.0


def with_minimum(v, minimum):
    """A wheel speed that actually moves the motor (DC motors don't turn below about minimum), or 0."""
    return 0 if v == 0 else int(math.copysign(max(abs(v), minimum), v))


class Navigator:
    """Wheel speeds to reach a target on the floor, from the phone's pose. step() each frame."""
    ALIGN = math.radians(12)            # turn in place until the target is this close to straight ahead
    REALIGN = math.radians(30)          # while driving, stop and turn again if it drifts this far
    GAP = 0.12                          # stop this far (m) from the target's edge
    TIMEOUT = 25.0                      # s
    MAX_RATE = math.radians(45)         # turning faster blurs the view and can lose the phone's tracking

    def __init__(self, turn=130, speed=140, minimum=100, steer=1.2, offset=0.0):
        self.turn, self.speed, self.minimum, self.steer = turn, speed, minimum, steer
        self.offset = offset                            # the rover's forward direction minus the phone's (radians)
        self.active, self.state, self.target_id, self.t0, self.info = False, "idle", None, 0.0, {}
        self.rate = TurnRate()

    def start(self, target_id):
        self.active, self.state, self.target_id, self.t0 = True, "turning", target_id, time.monotonic()

    def stop(self, state="stopped"):
        self.active, self.state = False, state

    def step(self, T, target, radius=0.04):
        """(left, right) for this frame; target: the target's world position (its remembered one when unseen)."""
        if not self.active:
            return 0, 0
        pos, now = np.asarray(T)[:2, 3], time.monotonic()
        rate = self.rate(heading(T), now)
        d = float(np.linalg.norm(np.asarray(target)[:2] - pos)) - radius
        err = wrap(math.atan2(target[1] - pos[1], target[0] - pos[0]) - heading(T) - self.offset)
        self.info = dict(distance_m=round(d, 3), bearing_deg=round(math.degrees(err), 1))
        if d <= self.GAP:
            self.stop("arrived")
            return 0, 0
        if time.monotonic() - self.t0 > self.TIMEOUT:
            self.stop("timeout")
            return 0, 0
        if self.state == "driving" and abs(err) > self.REALIGN or self.state != "driving" and abs(err) > self.ALIGN:
            self.state = "turning"                      # err > 0: the target is to the left: turn left
            if abs(rate) > self.MAX_RATE:               # too fast: coast this frame (the motors can't go slower)
                return 0, 0
            s = self.turn if err > 0 else -self.turn
            return with_minimum(-s, self.minimum), with_minimum(s, self.minimum)
        self.state = "driving"
        v = self.speed * min(1.0, max(0.6, d / 0.4))    # slower over the last 40 cm
        k = float(np.clip(self.steer * err, -0.6, 0.6)) * v
        return with_minimum(v - k, self.minimum), with_minimum(v + k, self.minimum)

    def status(self):
        return dict(state=self.state, target_id=self.target_id, **self.info)


class ScanRoutine:
    """The rover's own scan: look left and right, roll forward a little, look again (headings relative to where it
    started), pausing after each move so the scan gets sharp views. step() each frame until done."""
    STEPS = (("turn", 40), ("turn", -40), ("turn", 0), ("forward", 0.15), ("turn", 40), ("turn", -40), ("turn", 0))
    TOLERANCE = math.radians(6)
    PAUSE = 0.6                         # s still after each move
    STEP_TIMEOUT = 10.0                 # s for one move (then the next)
    MAX_RATE = math.radians(30)         # slow turns: sharp views and steady tracking

    def __init__(self, turn=115, speed=120, minimum=100):
        self.turn, self.speed, self.minimum = turn, speed, minimum
        self.i, self.yaw0, self.start_pos, self.t_step, self.paused_until = 0, None, None, None, 0.0
        self.rate = TurnRate()
        self.forward_offset = None                      # measured on the forward roll: rover forward - phone forward

    @property
    def done(self):
        return self.i >= len(self.STEPS)

    def next(self, now):
        self.i, self.start_pos, self.t_step, self.paused_until = self.i + 1, None, None, now + self.PAUSE

    def step(self, T):
        now = time.monotonic()
        rate = self.rate(heading(T), now)
        if self.done or now < self.paused_until:
            return 0, 0
        if self.yaw0 is None:
            self.yaw0 = heading(T)
        kind, amount = self.STEPS[self.i]
        if self.t_step is None:
            self.t_step, self.start_pos = now, np.asarray(T)[:2, 3].copy()
        if now - self.t_step > self.STEP_TIMEOUT:
            self.next(now)
            return 0, 0
        if kind == "turn":
            err = wrap(self.yaw0 + math.radians(amount) - heading(T))
            if abs(err) < self.TOLERANCE:
                self.next(now)
                return 0, 0
            if abs(rate) > self.MAX_RATE:
                return 0, 0
            s = self.turn if err > 0 else -self.turn
            return with_minimum(-s, self.minimum), with_minimum(s, self.minimum)
        moved = np.asarray(T)[:2, 3] - self.start_pos
        if np.linalg.norm(moved) >= amount:             # the way it rolled vs the way the phone faces: the mount
            offset = wrap(math.atan2(moved[1], moved[0]) - heading(T))
            if abs(offset) < math.radians(45):
                self.forward_offset = offset
            self.next(now)
            return 0, 0
        err = wrap(self.yaw0 - heading(T))              # straight ahead, holding the starting heading
        k = float(np.clip(1.2 * err, -0.5, 0.5)) * self.speed
        return with_minimum(self.speed - k, self.minimum), with_minimum(self.speed + k, self.minimum)


# ---------------------------------------------------------------------------------------- command line
def main(argv=None):
    ap = argparse.ArgumentParser(description="The rover's motors (see the top of this file)")
    ap.add_argument("cmd", choices=("ping", "test", "drive"))
    ap.add_argument("args", nargs="*", type=float, help="drive: LEFT RIGHT SECONDS")
    ap.add_argument("--port", default="auto")
    ap.add_argument("--swap", type=int, choices=(0, 1), default=int(SWAP_SIDES),
                    help=f"send \"<right> <left>\" (1) or \"<left> <right>\" (0); default {int(SWAP_SIDES)}")
    args = ap.parse_args(argv)
    rover = Rover(args.port, swap=bool(args.swap))
    try:
        if args.cmd == "ping":
            print("pong" if rover.ping() else "no answer")
        elif args.cmd == "test":
            for name, l, r in (("left wheel forward", 150, 0), ("left wheel back", -150, 0),
                               ("right wheel forward", 0, 150), ("right wheel back", 0, -150)):
                print(f"{name} (1 s)", flush=True)
                t = time.monotonic()
                while time.monotonic() - t < 1.0:
                    rover.drive(l, r)
                    time.sleep(0.05)
                rover.stop()
                time.sleep(0.7)
            print("A wheel turned the wrong way: swap that motor's wires at its OUT terminal (or set INVERT_LEFT / "
                  "INVERT_RIGHT in rover/rover.ino). Left and right swapped: swap the motors' OUT pairs.")
        else:
            if len(args.args) < 2:
                ap.error("drive LEFT RIGHT [SECONDS]")
            left, right, seconds = args.args[0], args.args[1], args.args[2] if len(args.args) > 2 else 1.0
            t = time.monotonic()
            while time.monotonic() - t < seconds:
                rover.drive(left, right)
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        rover.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
