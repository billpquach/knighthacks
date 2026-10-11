# Head tracker for the glasses: Raspberry Pi Pico W + GY-521 (MPU-6050), MicroPython.
# Upload with:  py -m mpremote cp config.py :config.py + main.py :main.py
#
# Wiring (GY-521 -> Pico W):
#   VCC -> 3V3 (pin 36)    GND -> GND (pin 38)
#   SDA -> GP4 (pin 6)     SCL -> GP5 (pin 7)
#   Optional re-center button: GP15 (pin 20) -> button -> GND
#
# Sends UDP JSON ~50x/s to the Jetson: {"yaw": deg, "pitch": deg}
#   yaw   = head rotation left/right since start or last re-center (+ = left/CCW)
#   pitch = head tilt up/down from the accelerometer (for an optional servo later)
# Keep your head STILL for ~2 s after power-up while it calibrates (LED solid).

import network, socket, time, math, json
from machine import Pin, I2C

# Wi-Fi name/password and the receiver's IP live in config.py (not committed).
# Copy config_example.py to config.py, fill it in, and upload both files.
from config import WIFI_SSID, WIFI_PASS, RECEIVER_IP
JETSON_IP = RECEIVER_IP        # laptop now; the Jetson later
JETSON_PORT = 5007
RATE_HZ = 50

led = Pin("LED", Pin.OUT)
button = Pin(15, Pin.IN, Pin.PULL_UP)
i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)   # 100 kHz: more tolerant of jumper wires
ADDR = 0x68

def wr(reg, val):
    i2c.writeto_mem(ADDR, reg, bytes([val]))

def rd16(reg):
    b = i2c.readfrom_mem(ADDR, reg, 2)
    v = (b[0] << 8) | b[1]
    return v - 65536 if v > 32767 else v

# Wake up, gyro +/-250 dps (131 LSB per deg/s), accel +/-2 g, low-pass filter ~44 Hz
wr(0x6B, 0x00)
wr(0x1B, 0x00)
wr(0x1C, 0x00)
wr(0x1A, 0x03)
time.sleep_ms(100)

if ADDR not in i2c.scan():
    while True:  # fast blink = sensor not found, check wiring
        led.toggle(); time.sleep_ms(100)

# Gyro bias calibration
led.on()
n, bias = 400, 0.0
for _ in range(n):
    bias += rd16(0x47)
    time.sleep_ms(5)
bias /= n
led.off()

# Wi-Fi
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
try:
    wlan.config(pm=0xa11140)   # turn OFF Wi-Fi power saving (it stalls UDP after a while)
except Exception:
    pass

def connect_wifi():
    if wlan.isconnected():
        return
    wlan.connect(WIFI_SSID, WIFI_PASS)
    while not wlan.isconnected():
        led.toggle(); time.sleep_ms(400)
    print("connected", wlan.ifconfig()[0])

connect_wifi()
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
dest = socket.getaddrinfo(JETSON_IP, JETSON_PORT)[0][-1]

yaw = 0.0
period = 1.0 / RATE_HZ
last = time.ticks_us()
sent = 0
fails = 0
while True:
    # Heartbeat: LED flickers while packets are going out (solid = stuck, slow blink = reconnecting)
    if sent % 25 == 0:
        led.toggle()
    if not wlan.isconnected() or fails > 50:
        print("wifi lost, reconnecting")
        try:
            wlan.disconnect()
        except Exception:
            pass
        connect_wifi()
        fails = 0
    now = time.ticks_us()
    dt = time.ticks_diff(now, last) / 1e6
    last = now

    try:
        gz = (rd16(0x47) - bias) / 131.0
        ax, ay, az = rd16(0x3B), rd16(0x3D), rd16(0x3F)
    except OSError:
        # I2C glitch (loose wire / noise): wake the sensor again and skip this sample
        try:
            wr(0x6B, 0x00)
        except OSError:
            pass
        time.sleep_ms(5)
        continue

    if abs(gz) > 0.5:          # deadband kills slow drift while still
        yaw += gz * dt
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

    zero = button.value() == 0
    if zero:                   # re-center: face the rover's direction and press
        yaw = 0.0

    try:
        sock.sendto(json.dumps({"yaw": round(yaw, 2), "pitch": round(pitch, 1), "zero": zero}), dest)
        sent += 1
        fails = 0
    except OSError:
        fails += 1

    spent = time.ticks_diff(time.ticks_us(), now) / 1e6
    if spent < period:
        time.sleep(period - spent)
