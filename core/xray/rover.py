"""The rover: two DC motors on an L298N, driven by an Arduino Uno (rover/rover.ino) over USB serial, and steered by
the phone's own pose (ARKit, in the scan's world): no wheel encoders, no IMU. The phone is the rover's eyes and
its odometry.

    python -m xray.rover ping                    # is the Arduino there (and on which port)
    python -m xray.rover test                    # each wheel forward then back, 1 s each: check wiring, directions
    python -m xray.rover drive 140 140 1.5       # both wheels at 140 (of 255) for 1.5 s
    python -m xray.live --ball --rover --rover-scan      # the live runner drives it

With --rover the live runner listens for "drive" (or "fetch", "get it") on POST /prompt or the console while
tracking, and drives to the target: the tracker's target (the hidden ball's cup, or Qwen-3D's answer), at its
remembered position; "drive to cup 2" / "drive to the yellow ball" name one. "stop" stops. --rover-scan turns the
rover through the scan by itself.

How it drives (Navigator): turn in place until the target is within ALIGN of straight ahead, then drive, steering
in proportion to the angle, at one steady speed; touch a cup with the front edge, stop GAP short of a ball; go around
the other objects via an approach point beside the target when the straight way is blocked. The phone sees the floor
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
    """Wheel speeds to reach a target on the floor, from the phone's pose. step() each frame.

    Distances count from the rover's front edge (`front` m ahead of the phone). A cup is driven up to and gently
    touched: a slow straight creep from APPROACH away until the front edge is PUSH past the cup's edge (or the rover
    stops moving: something is in the way). A ball would roll away, so the rover stops GAP short of it.
    Other objects (obstacles) are kept CLEAR of the rover's body: when the straight way to the target would hit one,
    the rover first drives to an approach point beside the target (30 or 45 cm from it) from which the way in is
    clear and which it can reach in a straight line, then turns and goes in."""
    ALIGN = math.radians(12)            # turn in place until the target is this close to straight ahead
    REALIGN = math.radians(30)          # while driving, stop and turn again if it drifts this far
    REALIGN_NEAR = math.radians(18)     # ... within NEAR of the target (a wide angle there misses or hits it sideways)
    NEAR = 0.25
    GAP = 0.03                          # the front edge stops this far (m) from a target it doesn't touch
    APPROACH = 0.06                     # ... and the touch begins this far from a cup
    PUSH = 0.0                          # the touch: this far past the cup's edge (it coasts ~1 cm more)
    CREEP_S = 4.0                       # the touch gives up after this long
    CLEAR = 0.025                       # kept between the rover's body and an obstacle's edge
    AT_POINT = 0.03                     # an approach point counts as reached this close
    TIMEOUT = 30.0                      # s
    MAX_RATE = math.radians(90)         # a safety net (the turn power sets the speed): faster loses the tracking

    def __init__(self, turn=115, speed=120, minimum=100, steer=1.2, offset=0.0, front=0.11, width=0.15):
        self.turn, self.speed, self.minimum, self.steer = turn, speed, minimum, steer
        self.offset = offset                            # the rover's forward direction minus the phone's (radians)
        self.front, self.half_width = front, width / 2  # the front edge ahead of the phone; half the rover's width
        self.active, self.state, self.target_id, self.t0, self.info = False, "idle", None, 0.0, {}
        self.touch, self.planned, self.approach, self.creep = False, False, None, None
        self.rate = TurnRate()

    def start(self, target_id, touch=False):
        self.active, self.state, self.target_id, self.t0 = True, "turning", target_id, time.monotonic()
        self.touch, self.planned, self.approach, self.creep = touch, False, None, None

    def stop(self, state="stopped"):
        self.active, self.state = False, state

    def room(self, a, b, obstacles):
        """The least room (m) between the rover's body, its front edge going straight from a to b, and an obstacle
        [(position, radius)] (negative: it would hit it)."""
        v = b - a
        length = max(float(np.linalg.norm(v)), 1e-6)
        u = v / length
        n = np.array([-u[1], u[0]])
        least = 1.0
        for c, r in obstacles:
            q = np.asarray(c, float)[:2] - a
            t, side = float(q @ u), float(q @ n)
            least = min(least, math.hypot(max(-t, 0.0, t - length), max(abs(side) - self.half_width, 0.0)) - r)
        return least

    def plan(self, pos, target, radius, obstacles):
        """None if the straight way in is clear, else the approach point (see the class): one with room to spare
        on the way in and on the way there, the shortest such; failing that, the one with the most room."""
        def edge(frm):                                  # where the front edge meets the target, coming from frm
            v = target - frm
            return target - v / max(float(np.linalg.norm(v)), 1e-6) * radius
        straight = self.room(pos, edge(pos), obstacles) if obstacles else 1.0
        if straight >= self.CLEAR:
            return None
        best, best_cost, roomiest, most = None, None, None, straight
        for dist in (0.30, 0.45, 0.60):
            for k in range(36):
                phi = 2 * math.pi * k / 36
                point = target + dist * np.array([math.cos(phi), math.sin(phi)])
                way = point - pos
                there = point + way / max(float(np.linalg.norm(way)), 1e-6) * self.front
                way_in, way_there = self.room(point, edge(point), obstacles), self.room(pos, there, obstacles)
                spot = min(np.linalg.norm(point - np.asarray(c, float)[:2]) - r for c, r in obstacles) - self.half_width
                worst = min(way_in, way_there, spot)
                if worst > most:
                    roomiest, most = point, worst
                if way_in < self.CLEAR + 0.015 or way_there < self.CLEAR + 0.03 or spot < self.CLEAR:
                    continue                            # (steering curves the way there; the turn at the point)
                turn = abs(wrap(math.atan2(*(target - point)[::-1]) - math.atan2(*way[::-1])))
                cost = float(np.linalg.norm(way)) + dist + 0.1 * turn + 2.0 * max(0.0, 0.08 - way_in)
                if best_cost is None or cost < best_cost:
                    best, best_cost = point, cost
        return best if best is not None else roomiest

    def step(self, T, target, radius=0.04, obstacles=()):
        """(left, right) for this frame; target: the target's world position (its remembered one when unseen);
        obstacles: [(position, radius)] of the other objects."""
        if not self.active:
            return 0, 0
        pos, now = np.asarray(T)[:2, 3], time.monotonic()
        target = np.asarray(target, float)[:2]
        yaw = heading(T) + self.offset                  # the rover's forward direction
        rate = self.rate(heading(T), now)
        d = float(np.linalg.norm(target - pos)) - radius
        gap = d - self.front                            # from the rover's front edge to the target's edge
        err = wrap(math.atan2(target[1] - pos[1], target[0] - pos[0]) - yaw)
        self.info = dict(distance_m=round(d, 3), gap_m=round(gap, 3), bearing_deg=round(math.degrees(err), 1))
        if now - self.t0 > self.TIMEOUT:
            self.stop("timeout")
            return 0, 0
        if self.creep is not None:                      # the touch: slow and straight, blind at the end
            start_t, last_pos, last_t = self.creep
            if gap <= -self.PUSH or now - start_t > self.CREEP_S:
                self.stop("touched")
                return 0, 0
            if now - last_t > 0.6:                      # still pushing but not moving: it's touching something
                if np.linalg.norm(pos - last_pos) < 0.004:
                    self.stop("touched")
                    return 0, 0
                self.creep = (start_t, pos.copy(), now)
            k = float(np.clip(self.steer * err, -0.3, 0.3)) * self.minimum
            return with_minimum(self.minimum - k, self.minimum), with_minimum(self.minimum + k, self.minimum)
        if not self.planned:
            self.approach, self.planned = self.plan(pos, target, radius, list(obstacles)), True
        if self.approach is not None and np.linalg.norm(self.approach - pos) < self.AT_POINT:
            self.approach, self.state = None, "turning"     # at the approach point: face the target, then in
        if self.approach is None:
            if not self.touch and gap <= self.GAP:
                self.stop("arrived")
                return 0, 0
            if self.touch and gap <= self.APPROACH and abs(err) < self.ALIGN:
                self.state, self.creep = "touching", (now, pos.copy(), now + 0.4)    # (a moment to get going)
                return 0, 0
            moving = "driving"
            if gap <= self.APPROACH:                    # close: only straight ahead (never sideways into it)
                realign = self.ALIGN
            else:
                realign = self.REALIGN_NEAR if gap < self.NEAR else self.REALIGN
        else:
            aim = self.approach
            err = wrap(math.atan2(aim[1] - pos[1], aim[0] - pos[0]) - yaw)
            moving, realign = "detour", self.REALIGN
        if self.state in ("driving", "detour") and abs(err) > realign or \
                self.state not in ("driving", "detour") and abs(err) > self.ALIGN:
            self.state = "turning"                      # err > 0: the target is to the left: turn left
            if abs(rate) > self.MAX_RATE:               # too fast: coast this frame (the motors can't go slower)
                return 0, 0
            s = self.turn if err > 0 else -self.turn
            return with_minimum(-s, self.minimum), with_minimum(s, self.minimum)
        self.state = moving
        v = self.speed                                  # one steady, slow speed (no slowing down: no lurches)
        k = float(np.clip(self.steer * err, -0.6, 0.6)) * v
        return with_minimum(v - k, self.minimum), with_minimum(v + k, self.minimum)

    def status(self):
        return dict(state=self.state, target_id=self.target_id, touch=self.touch,
                    approach=None if self.approach is None else [round(float(x), 3) for x in self.approach],
                    **self.info)


class ScanRoutine:
    """The rover's own scan, stop and shoot: turn in 20 deg steps to 40 deg left, across to 40 deg right and back to
    where it started, roll forward 15 cm, and the same again (headings relative to the start). After each move it
    stands still for PAUSE, and the scan takes its views only then (`still`), once the phone has stopped wobbling on
    its mast: every view sharp, none smeared by a turn. step() each frame until done."""
    SWEEP = (("turn", 20), ("turn", 40), ("turn", 20), ("turn", 0),
             ("turn", -20), ("turn", -40), ("turn", -20), ("turn", 0))
    STEPS = SWEEP + (("forward", 0.15),) + SWEEP
    TOLERANCE = math.radians(5)
    PAUSE = 1.0                         # s standing still after each move
    SETTLE = 0.4                        # s into a pause before views count (the mast stops swaying)
    STEP_TIMEOUT = 10.0                 # s for one move (then the next)
    MAX_RATE = math.radians(60)         # a safety net (the turn power sets the speed)

    def __init__(self, turn=115, speed=120, minimum=100):
        self.turn, self.speed, self.minimum = turn, speed, minimum
        self.i, self.yaw0, self.start_pos, self.t_step = 0, None, None, None
        self.paused_from, self.paused_until, self.last_rate = 0.0, 0.0, 0.0
        self.rate = TurnRate()
        self.forward_offset = None                      # measured on the forward roll: rover forward - phone forward

    @property
    def done(self):
        return self.i >= len(self.STEPS) and time.monotonic() >= self.paused_until

    @property
    def still(self):
        """Standing still and settled: the moment for a view."""
        now = time.monotonic()
        return self.paused_from + self.SETTLE <= now < self.paused_until and abs(self.last_rate) < math.radians(4)

    def pause(self, now):
        self.paused_from, self.paused_until = now, now + self.PAUSE

    def next(self, now):
        self.i, self.start_pos, self.t_step = self.i + 1, None, None
        self.pause(now)

    def step(self, T):
        now = time.monotonic()
        rate = self.last_rate = self.rate(heading(T), now)
        if self.yaw0 is None:                           # a still moment first: the view straight ahead
            self.yaw0 = heading(T)
            self.pause(now)
        if now < self.paused_until or self.i >= len(self.STEPS):
            return 0, 0
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
