# Step 1 sensor check for the Pico W + GY-521 (MicroPython).
# Run it from Thonny (green Run button) with the Pico plugged in over USB.
# No Wi-Fi needed. Prints the I2C scan, then live yaw/pitch in the Shell.

from machine import Pin, I2C
import time, math

i2c = I2C(0, sda=Pin(4), scl=Pin(5), freq=100000)
found = i2c.scan()
print("I2C devices found:", [hex(a) for a in found])
if 0x68 not in found:
    print("MPU-6050 NOT found at 0x68.")
    print("Check: VCC->3V3 (pin 36), GND->GND (pin 38), SDA->GP4 (pin 6), SCL->GP5 (pin 7).")
    print("If it shows 0x69 instead, the AD0 pin is pulled high; tie AD0 to GND.")
    raise SystemExit

ADDR = 0x68
i2c.writeto_mem(ADDR, 0x6B, b"\x00")   # wake up
i2c.writeto_mem(ADDR, 0x1B, b"\x00")   # gyro +/-250 dps
i2c.writeto_mem(ADDR, 0x1A, b"\x03")   # low-pass filter
time.sleep_ms(100)

def rd16(reg):
    b = i2c.readfrom_mem(ADDR, reg, 2)
    v = (b[0] << 8) | b[1]
    return v - 65536 if v > 32767 else v

print("Hold still: calibrating for 2 s...")
bias = sum(rd16(0x47) for _ in range(400) if not time.sleep_ms(5)) / 400
print("gyro Z bias (raw):", round(bias, 1), " (anything under ~500 is normal)")

yaw, last = 0.0, time.ticks_us()
print("Now turn the board left and right. Ctrl+C to stop.")
while True:
    now = time.ticks_us()
    dt = time.ticks_diff(now, last) / 1e6
    last = now
    gz = (rd16(0x47) - bias) / 131.0
    if abs(gz) > 0.5:
        yaw += gz * dt
    ax, ay, az = rd16(0x3B), rd16(0x3D), rd16(0x3F)
    pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    print("yaw {:7.1f}   pitch {:6.1f}".format(yaw, pitch))
    time.sleep_ms(50)
