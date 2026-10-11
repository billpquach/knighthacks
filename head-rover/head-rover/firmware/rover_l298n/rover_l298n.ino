// Rover firmware: Arduino Uno + L298N + 2WD chassis
// Listens on serial for lines like "120 -80\n" (left right, -255..255)
// and stops automatically if no command arrives for 300 ms.

// L298N pins (ENA/ENB must be PWM pins; remove the ENA/ENB jumpers on the board)
const int ENA = 5;   // left motor speed
const int IN1 = 7;
const int IN2 = 8;
const int ENB = 6;   // right motor speed
const int IN3 = 9;
const int IN4 = 10;

const unsigned long TIMEOUT_MS = 300;
unsigned long lastCmd = 0;
String buf = "";

void setMotor(int en, int a, int b, int speed) {
  speed = constrain(speed, -255, 255);
  digitalWrite(a, speed > 0);
  digitalWrite(b, speed < 0);
  analogWrite(en, abs(speed));
}

void drive(int left, int right) {
  setMotor(ENA, IN1, IN2, left);
  setMotor(ENB, IN3, IN4, right);
}

void setup() {
  int pins[] = {ENA, IN1, IN2, ENB, IN3, IN4};
  for (int p : pins) pinMode(p, OUTPUT);
  drive(0, 0);
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      int sp = buf.indexOf(' ');
      if (sp > 0) {
        drive(buf.substring(0, sp).toInt(), buf.substring(sp + 1).toInt());
        lastCmd = millis();
      }
      buf = "";
    } else if (c != '\r') {
      buf += c;
    }
  }
  if (millis() - lastCmd > TIMEOUT_MS) drive(0, 0);  // safety stop
}
