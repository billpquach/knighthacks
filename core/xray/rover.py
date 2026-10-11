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
        self.turn_sign, self.turn_yaw, self.turn_checked, self.wrong, self.last_yaw = 0, None, True, 0, None
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
        """On each turn in place: if the phone turns the other way than commanded, the left and right motors are
        swapped (wiring or firmware), so swap them here for the rest of the run. It counts from the moment the rover
        isn't still drifting the other way from its last move (coasting past a target isn't a wrong turn), and
        swaps only after 20 deg the wrong way twice in a row."""
        now = time.monotonic()
        rate = 0.0 if self.last_yaw is None else wrap(yaw - self.last_yaw[0]) / max(1e-3, now - self.last_yaw[1])
        self.last_yaw = (yaw, now)
        sign = (1 if right > left else -1) if left * right < 0 else 0      # +1: a left turn (counterclockwise)
        if sign == 0:
            return                                      # (pauses between moves, driving ahead)
        if sign != self.turn_sign:
            self.turn_sign, self.turn_yaw, self.turn_checked = sign, None, False
        if self.turn_checked:
            return
        if self.turn_yaw is None:                       # start counting once it isn't drifting the other way
            if rate * sign >= 0 or abs(rate) < math.radians(8):
                self.turn_yaw = yaw
            return
        moved = wrap(yaw - self.turn_yaw)
        if abs(moved) < math.radians(20):
            return
        if moved * sign > 0:                            # the commanded way: fine for this turn
            self.wrong, self.turn_checked = 0, True
            return
        self.wrong += 1
        self.turn_yaw = yaw                             # (and keep watching this turn)
        if self.wrong >= 2:
            self.swap, self.wrong, self.turn_checked = not self.swap, 0, True
            self.log("rover: turns went the wrong way: left and right motors swapped, fixed in software "
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


class Speedometer:
    """How fast the phone has been moving over the floor (m/s) over the last `window` s."""

    def __init__(self, window=0.3):
        self.window, self.history = window, []

    def __call__(self, pos, now):
        self.history = [(p, t) for p, t in self.history if now - t <= self.window] + [(np.asarray(pos)[:2].copy(), now)]
        (p0, t0), (p1, t1) = self.history[0], self.history[-1]
        return float(np.linalg.norm(p1 - p0)) / (t1 - t0) if t1 > t0 else 0.0


class Pace:
    """Wheel power for a steady, slow pace (a turn rate in rad/s, or m/s ahead), from what the phone measures. A DC
    motor needs more power to start than to keep going, and below that it only hums. So it remembers two powers:
    `start`, what last got it going (a little less is tried each time; while it only hums, more every moment), and
    `run`, the least that keeps it at the target pace once going (adjusted while going; raised if it stalls). Call
    idle() while standing still, so the next move starts from rest."""

    def __init__(self, target, power, low, high=210, gain=80.0):
        self.target, self.low, self.high, self.gain = target, low, high, gain
        self.start = self.run = float(power)
        self.going, self.t = False, None

    def __call__(self, measured, now):
        dt = 0.0 if self.t is None else min(0.2, now - self.t)
        self.t = now
        if abs(measured) < 0.4 * self.target:           # not going: the starting power, more each moment it hums
            if self.going:                              # (it was going and stalled: running power too low)
                self.going, self.run = False, min(self.high, self.run + 5)
            self.start = min(self.high, self.start + self.gain * dt)
            return self.start
        if not self.going:                              # it started: next time try a little less to start it
            self.going, self.start = True, max(self.low, self.start - 3)
        err = float(np.clip((self.target - abs(measured)) / self.target, -1.0, 1.0))
        self.run = float(np.clip(self.run + 0.5 * self.gain * err * dt, self.low, self.high))
        return self.run

    def idle(self):
        self.going, self.t = False, None


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
    MAX_RATE = math.radians(90)         # a safety net: faster blurs the view and can lose the phone's tracking
    TURN_RATE = math.radians(25)        # the pace of a turn in place
    SPEED = 0.12                        # m/s driving
    CREEP = 0.05                        # m/s touching

    def __init__(self, turn=115, speed=120, minimum=100, steer=1.2, offset=0.0, front=0.11, width=0.15):
        self.turn, self.speed, self.minimum, self.steer = turn, speed, minimum, steer   # (starting powers)
        self.turn_pace, self.drive_pace = Pace(self.TURN_RATE, turn, minimum), Pace(self.SPEED, speed, minimum)
        self.creep_pace, self.spin, self.speedo, self.mode = None, TurnRate(0.25), Speedometer(), None
        self.offset = offset                            # the rover's forward direction minus the phone's (radians)
        self.front, self.half_width = front, width / 2  # the front edge ahead of the phone; half the rover's width
        self.active, self.state, self.target_id, self.t0, self.info = False, "idle", None, 0.0, {}
        self.touch, self.planned, self.approach, self.creep = False, False, None, None
        self.rate = TurnRate()

    def start(self, target_id, touch=False):
        self.active, self.state, self.target_id, self.t0 = True, "turning", target_id, time.monotonic()
        self.touch, self.planned, self.approach, self.creep = touch, False, None, None
        self.moving("still")

    def moving(self, mode):
        """Switching between still, turning and driving: the pace not in use starts from rest next time."""
        if mode != self.mode:
            self.turn_pace.idle()
            self.drive_pace.idle()
            self.mode = mode

    def stop(self, state="stopped"):
        self.active, self.state = False, state
        self.moving("still")

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
        spin, speed = self.spin(heading(T), now), self.speedo(pos, now)
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
            if now - last_t > 0.6:                      # pushing as hard as it may but not moving: touching it
                if np.linalg.norm(pos - last_pos) < 0.004 and self.creep_pace.start >= self.creep_pace.high - 1:
                    self.stop("touched")
                    return 0, 0
                self.creep = (start_t, pos.copy(), now)
            v = self.creep_pace(speed, now)             # (never harder than what got it rolling before, + 10)
            k = float(np.clip(self.steer * err, -0.3, 0.3)) * v
            return with_minimum(v - k, self.minimum), with_minimum(v + k, self.minimum)
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
                cap = max(self.minimum, self.drive_pace.start) + 15     # (never pushes harder than that)
                self.creep_pace = Pace(self.CREEP, min(cap, self.drive_pace.run), self.minimum, high=cap)
                self.creep_pace.start, self.creep_pace.going = cap - 15, self.drive_pace.going
                self.moving("still")
                return self.step(T, target, radius, obstacles)      # (straight on: no stop in between)
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
            self.moving("turning")
            if abs(rate) > self.MAX_RATE:               # (a safety net)
                return 0, 0
            p = int(round(self.turn_pace(spin, now)))   # a steady, slow turn (more to start it, see Pace)
            return (-p, p) if err > 0 else (p, -p)
        self.state = moving
        self.moving("driving")
        v = self.drive_pace(speed, now)                 # one steady, slow speed (no slowing down: no lurches)
        k = float(np.clip(self.steer * err, -0.6, 0.6)) * v
        return with_minimum(v - k, self.minimum), with_minimum(v + k, self.minimum)

    def status(self):
        return dict(state=self.state, target_id=self.target_id, touch=self.touch,
                    approach=None if self.approach is None else [round(float(x), 3) for x in self.approach],
                    **self.info)


class ScanRoutine:
    """The rover's own scan, stop and shoot: turn in 20 deg steps to 40 deg left, across to 40 deg right and back to
    where it started, roll forward 15 cm, and the same again (headings relative to the start), at a slow, steady
    pace (Pace: a push to get going, then the least that keeps it turning). After each move it stands still for
    PAUSE, and the scan takes its views only then (`still`), once the phone has stopped wobbling on its mast: every
    view sharp, none smeared by a turn. step() each frame until done."""
    SWEEP = (("turn", 20), ("turn", 40), ("turn", 20), ("turn", 0),
             ("turn", -20), ("turn", -40), ("turn", -20), ("turn", 0))
    STEPS = SWEEP + (("forward", 0.15),) + SWEEP
    TOLERANCE = math.radians(5)
    PAUSE = 1.0                         # s standing still after each move
    PAUSE_MAX = 2.5                     # ... longer, while the scan waits for its view and the phone still sways
    SETTLE = 0.4                        # s into a pause before views count (the mast stops swaying)
    STEP_TIMEOUT = 10.0                 # s for one move (then the next)
    MAX_RATE = math.radians(60)         # a safety net
    TURN_RATE = math.radians(20)        # the pace of the turns: slow, so the phone barely sways
    SPEED = 0.08                        # m/s rolling forward

    def __init__(self, turn=115, speed=120, minimum=100):
        self.turn, self.speed, self.minimum = turn, speed, minimum   # (starting powers)
        self.i, self.yaw0, self.start_pos, self.t_step = 0, None, None, None
        self.paused_from, self.paused_until, self.last_rate = 0.0, 0.0, 0.0
        self.wants_views, self.viewed = False, None     # (set by the scan: it waits at each stop for its view)
        self.rate, self.spin, self.speedo = TurnRate(), TurnRate(0.25), Speedometer()
        self.turn_pace, self.roll_pace = Pace(self.TURN_RATE, turn, minimum), Pace(self.SPEED, speed, minimum)
        self.forward_offset = None                      # measured on the forward roll: rover forward - phone forward

    @property
    def pausing(self):
        now = time.monotonic()
        return now < self.paused_until or (self.wants_views and self.viewed != self.paused_from
                                           and now < self.paused_from + self.PAUSE_MAX)

    @property
    def done(self):
        return self.i >= len(self.STEPS) and not self.pausing

    @property
    def still(self):
        """Standing still and settled: the moment for a view."""
        return self.pausing and time.monotonic() >= self.paused_from + self.SETTLE and abs(self.last_rate) < math.radians(4)

    def pause(self, now):
        self.paused_from, self.paused_until = now, now + self.PAUSE
        self.turn_pace.idle()                           # (the next move starts from rest)
        self.roll_pace.idle()

    def next(self, now):
        self.i, self.start_pos, self.t_step = self.i + 1, None, None
        self.pause(now)

    def step(self, T):
        now = time.monotonic()
        rate = self.last_rate = self.rate(heading(T), now)
        spin, speed = self.spin(heading(T), now), self.speedo(np.asarray(T)[:2, 3], now)
        if self.yaw0 is None:                           # a still moment first: the view straight ahead
            self.yaw0 = heading(T)
            self.pause(now)
        if self.pausing or self.i >= len(self.STEPS):
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
            p = int(round(self.turn_pace(spin, now)))   # a steady, slow turn (more to start it, see Pace)
            return (-p, p) if err > 0 else (p, -p)
        moved = np.asarray(T)[:2, 3] - self.start_pos
        if np.linalg.norm(moved) >= amount:             # the way it rolled vs the way the phone faces: the mount
            offset = wrap(math.atan2(moved[1], moved[0]) - heading(T))
            if abs(offset) < math.radians(45):
                self.forward_offset = offset
            self.next(now)
            return 0, 0
        err = wrap(self.yaw0 - heading(T))              # straight ahead, holding the starting heading
        v = self.roll_pace(speed, now)
        k = float(np.clip(1.2 * err, -0.5, 0.5)) * v
        return with_minimum(v - k, self.minimum), with_minimum(v + k, self.minimum)


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
