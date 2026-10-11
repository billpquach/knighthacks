// L298N self-test: no laptop script, no serial commands needed.
// Same pins as rover.ino. Upload, then watch the wheels (keep them off the table).
// The Uno's onboard LED (pin 13) lights while a motor is commanded on,
// so you can tell "code is running" apart from "motor isn't responding".

const int ENA = 5, IN1 = 7, IN2 = 8;   // left motor  (OUT1/OUT2)
const int ENB = 6, IN3 = 9, IN4 = 10;  // right motor (OUT3/OUT4)
const int LED = 13;

void motor(int en, int a, int b, int speed) {
  digitalWrite(a, speed > 0);
  digitalWrite(b, speed < 0);
  analogWrite(en, abs(speed));
}

void step(const char* label, int left, int right) {
  Serial.println(label);
  digitalWrite(LED, HIGH);
  motor(ENA, IN1, IN2, left);
  motor(ENB, IN3, IN4, right);
  delay(1500);
  motor(ENA, IN1, IN2, 0);
  motor(ENB, IN3, IN4, 0);
  digitalWrite(LED, LOW);
  delay(700);
}

void setup() {
  Serial.begin(115200);
  int pins[] = {ENA, IN1, IN2, ENB, IN3, IN4, LED};
  for (int p : pins) pinMode(p, OUTPUT);
  Serial.println("L298N self-test starting");
}

void loop() {
  step("LEFT forward  (full speed)", 255, 0);
  step("LEFT backward (full speed)", -255, 0);
  step("RIGHT forward  (full speed)", 0, 255);
  step("RIGHT backward (full speed)", 0, -255);
  step("BOTH forward", 255, 255);
  Serial.println("--- repeating ---");
  delay(1500);
}
