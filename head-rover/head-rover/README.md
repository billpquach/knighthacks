# Head-tracked rover

A small 2WD rover you steer with your head. A gyro on a pair of glasses (Raspberry Pi Pico W + GY-521) streams head angles over Wi-Fi to a laptop, and the laptop drives the rover's Arduino over USB.

This folder is standalone: it doesn't need the Jetson code in the rest of the repo. Later, the Jetson can replace the laptop as the receiver, since it speaks the same protocols.

```
Pico W + GY-521 (glasses) --Wi-Fi UDP :5007--> laptop --USB serial--> Arduino Uno --> L298N --> motors
```

## Layout

| Path | Runs on | What it does |
|---|---|---|
| `firmware/rover_l298n/` | Arduino Uno | Motor firmware. Accepts `"<left> <right>\n"` (−255..255) at 115200 baud and stops if commands stop for 300 ms. |
| `firmware/l298n_selftest/` | Arduino Uno | Spins each motor at full speed with no laptop involved, for hardware checks. |
| `pico/main.py` | Pico W | Reads the GY-521 and sends `{"yaw","pitch","zero"}` over UDP at 50 Hz. Wi-Fi power saving is off, it reconnects automatically, and the LED shows a heartbeat. |
| `pico/config_example.py` | Pico W | Template for Wi-Fi and receiver settings. Copy it to `config.py` (gitignored). |
| `pico/gyro_test.py` | Pico W | Sensor check over USB. |
| `laptop/head_drive.py` | Laptop | **Head joystick:** turn your head to spin, tilt it to drive forward or back. |
| `laptop/head_follow.py` | Laptop | **Follow mode:** the rover turns to face the same direction as your head (yaw only). Run `--calibrate` once. |
| `laptop/drive_test.py` | Laptop | Motor test sequence over serial (`--ramp` finds the minimum PWM). |
| `laptop/head_receiver.py` | Laptop | Prints incoming head data to check the network. |

## Hardware

**Rover:** 2WD chassis with TT motors, an Arduino Uno and an L298N, powered by a ~9 V AA pack.

| L298N | Connects to |
|---|---|
| 12V | battery + |
| GND | battery − **and** Uno GND |
| 5V | unconnected (5V-EN jumper on) |
| ENA, IN1, IN2 | Uno D5, D7, D8 (left motor, OUT1/OUT2) |
| ENB, IN3, IN4 | Uno D6, D9, D10 (right motor, OUT3/OUT4) |

Remove the ENA and ENB jumpers.

**Glasses:** a Pico W and a GY-521, mounted flat with the chip facing up.

| GY-521 | Pico W |
|---|---|
| VCC | 3V3 (pin 36) |
| GND | GND (pin 38) |
| SDA | GP4 (pin 6) |
| SCL | GP5 (pin 7) |

A zero button connects GP15 (pin 20) to GND (pin 18).

## Setup

```
py -m pip install -r laptop/requirements.txt
```

1. **Arduino:** upload `firmware/rover_l298n/rover_l298n.ino`.
2. **Pico** (MicroPython for the Pico W must already be flashed):
   ```
   cd pico
   copy config_example.py config.py      # then edit it
   py -m mpremote cp config.py :config.py
   py -m mpremote cp main.py :main.py
   py -m mpremote reset
   ```
3. **Network:** use a phone hotspot on **2.4 GHz** (iPhone: Maximize Compatibility). The Pico W can't join 5 GHz or WPA2-Enterprise networks. If no data arrives, allow UDP 5007 through the firewall:
   `netsh advfirewall firewall add rule name="Pico UDP 5007" dir=in action=allow protocol=UDP localport=5007`

## Run

```
cd laptop
py drive_test.py --port COM4               # wheels off the ground first
py head_drive.py --port COM4               # head joystick
py head_follow.py --port COM4 --calibrate  # once, on the demo floor
py head_follow.py --port COM4              # rover faces where you look
```

Face the same direction as the rover, then press the zero button (or Enter in `head_follow.py`) to align them.

## Troubleshooting

- **Wheels hum but don't move:** raise `MIN_PWM` or `TURN_PWM`. The L298N drops about 2 V.
- **A wheel spins backward:** swap that motor's wires at its OUT terminal.
- **The rover turns the wrong way:** set `YAW_SIGN = -1`.
- **Pico LED:** a steady flicker means it's sending, a slow blink means it's reconnecting, and a frozen LED means it's crashed (power-cycle it).
