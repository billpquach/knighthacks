"""
Step 3 check: run on the laptop or Jetson to confirm the Pico's Wi-Fi stream arrives.
  python head_receiver.py
Set JETSON_IP in the Pico's main.py to THIS computer's IP. Turn your head; numbers change.
If nothing prints: same Wi-Fi network? firewall blocking UDP 5007? (Windows: allow Python)
"""
import json, socket, time

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(("0.0.0.0", 5007))
print("listening on UDP 5007...")
count, t0 = 0, time.time()
while True:
    data, addr = s.recvfrom(256)
    msg = json.loads(data)
    count += 1
    rate = count / max(time.time() - t0, 1e-3)
    print(f"from {addr[0]}  yaw {msg['yaw']:7.1f}  pitch {msg['pitch']:6.1f}  ({rate:4.0f} msg/s)")
