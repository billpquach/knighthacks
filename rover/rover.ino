// The rover's two DC motors, driven over USB serial by the Jetson (core/xray/rover.py): L298N + Arduino Uno.
//
// Wiring (change the pin numbers below if yours differ):
//   L298N ENA -> D5 (PWM)   IN1 -> D7   IN2 -> D8     left motor on OUT1 / OUT2
//   L298N ENB -> D6 (PWM)   IN3 -> D9   IN4 -> D10    right motor on OUT3 / OUT4
//   L298N GND -> Arduino GND (a common ground is required); battery + -> L298N +12V, battery - -> L298N GND
//   Take the jumpers off ENA and ENB: with them on, the motors only run at full speed. The 5V-EN jumper stays on,
//   the L298N 5V terminal unconnected. Clamp the motor wires firmly in the OUT terminals (strip 5-7 mm).
//   The Uno itself is powered by the Jetson's USB. A ~9 V pack leaves the TT motors ~7 V (the L298N drops ~2 V).
//   A wheel turning the wrong way: swap that motor's two wires (or set INVERT_ below).
//
// Commands, one per line at 115200 baud:
//   <left> <right>     wheel speeds -255..255, positive = forward (the format of the earlier firmware too)
//   M <left> <right>   the same                                       -> "ok"
//   S                  stop now (no ramp)                             -> "ok"
//   P                  ping                                           -> "pong"
// Safety: with no command for 300 ms both motors stop (the Jetson resends every 100 ms while driving), so the
// rover stops if the Jetson crashes or the cable comes out. Speeds ramp by at most RAMP per 10 ms, so starting
// and stopping never jolt the phone mast.

const int ENA = 5, IN1 = 7, IN2 = 8;                   // left motor
const int ENB = 6, IN3 = 9, IN4 = 10;                  // right motor
const bool INVERT_LEFT = false, INVERT_RIGHT = false;  // set true if that wheel turns backwards on "140 140"
const int RAMP = 8;                                    // PWM steps per 10 ms (0 -> 160 in 0.2 s)
const unsigned long WATCHDOG_MS = 300;

int targetL = 0, targetR = 0, curL = 0, curR = 0;
unsigned long lastCmd = 0, lastStep = 0;
char line[40];
byte len = 0;

void setMotor(int en, int a, int b, int speed, bool invert) {
  if (invert) speed = -speed;
  digitalWrite(a, speed > 0 ? HIGH : LOW);
  digitalWrite(b, speed < 0 ? HIGH : LOW);
  analogWrite(en, abs(speed));
}

int approach(int cur, int target) {
  if (cur < target) return min(cur + RAMP, target);
  if (cur > target) return max(cur - RAMP, target);
  return cur;
}

void handle(char *s) {
  if (s[0] == 'M' || s[0] == '-' || (s[0] >= '0' && s[0] <= '9')) {
    int l, r;
    if (sscanf(s[0] == 'M' ? s + 1 : s, "%d %d", &l, &r) == 2) {
      targetL = constrain(l, -255, 255);
      targetR = constrain(r, -255, 255);
      lastCmd = millis();
      Serial.println("ok");
    } else {
      Serial.println("err");
    }
  } else if (s[0] == 'S') {
    targetL = targetR = curL = curR = 0;
    setMotor(ENA, IN1, IN2, 0, INVERT_LEFT);
    setMotor(ENB, IN3, IN4, 0, INVERT_RIGHT);
    lastCmd = millis();
    Serial.println("ok");
  } else if (s[0] == 'P') {
    Serial.println("pong");
  } else if (s[0]) {
    Serial.println("err");
  }
}

void setup() {
  int pins[] = {ENA, IN1, IN2, ENB, IN3, IN4};
  for (int p : pins) {
    pinMode(p, OUTPUT);
    digitalWrite(p, LOW);
  }
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      line[len] = 0;
      if (len) handle(line);
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
  unsigned long now = millis();
  if (now - lastStep >= 10) {
    lastStep = now;
    if (now - lastCmd > WATCHDOG_MS) targetL = targetR = 0;   // no news from the Jetson: stop
    curL = approach(curL, targetL);
    curR = approach(curR, targetR);
    setMotor(ENA, IN1, IN2, curL, INVERT_LEFT);
    setMotor(ENB, IN3, IN4, curR, INVERT_RIGHT);
  }
}
