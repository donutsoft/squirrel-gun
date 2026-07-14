#include <Arduino.h>
#include <Servo.h>

const size_t MAX_LINE = 96;
const size_t MAX_TIMED_PINS = 16;
const int MIN_PULSE_US = 500;
const int MAX_PULSE_US = 2500;
const float PAN_MAX_DEGREES = 270.0;
const float TILT_MAX_DEGREES = 180.0;
const uint8_t FIRST_CONTROLLABLE_PIN = 2;
const uint8_t LAST_CONTROLLABLE_PIN = 19;

struct TimedPin {
  uint8_t pin;
  unsigned long offAt;
};

struct ServoSlot {
  uint8_t pin;
  bool attached;
  Servo servo;
};

TimedPin timedPins[MAX_TIMED_PINS];
ServoSlot panServo;
ServoSlot tiltServo;
char lineBuffer[MAX_LINE];
size_t lineLength = 0;

void sendOk() {
  Serial.println(F("OK"));
}

void sendError() {
  Serial.println(F("ERROR"));
}

long parseLong(const char *text, bool &ok) {
  if (text == nullptr || *text == '\0') {
    ok = false;
    return 0;
  }
  char *end = nullptr;
  long value = strtol(text, &end, 10);
  ok = end != text && *end == '\0';
  return value;
}

float parseFloatValue(const char *text, bool &ok) {
  if (text == nullptr || *text == '\0') {
    ok = false;
    return 0.0;
  }
  char *end = nullptr;
  float value = strtod(text, &end);
  ok = end != text && *end == '\0';
  return value;
}

bool parsePin(const char *text, uint8_t &pin) {
  bool ok = false;
  long value = parseLong(text, ok);
  if (!ok || value < FIRST_CONTROLLABLE_PIN || value > LAST_CONTROLLABLE_PIN) {
    return false;
  }
  pin = static_cast<uint8_t>(value);
  return true;
}

int servoPulseUs(float degrees, float maxDegrees) {
  if (degrees < 0.0) {
    degrees = 0.0;
  }
  if (degrees > maxDegrees) {
    degrees = maxDegrees;
  }
  float fraction = maxDegrees <= 0.0 ? 0.0 : degrees / maxDegrees;
  return static_cast<int>(MIN_PULSE_US + (fraction * (MAX_PULSE_US - MIN_PULSE_US)) + 0.5);
}

void writeServo(ServoSlot &slot, uint8_t pin, float degrees, float maxDegrees) {
  if (!slot.attached || slot.pin != pin) {
    if (slot.attached) {
      slot.servo.detach();
    }
    slot.pin = pin;
    slot.servo.attach(pin, MIN_PULSE_US, MAX_PULSE_US);
    slot.attached = true;
  }
  slot.servo.writeMicroseconds(servoPulseUs(degrees, maxDegrees));
}

void setOutput(uint8_t pin, uint8_t value) {
  pinMode(pin, OUTPUT);
  digitalWrite(pin, value);
}

void forceAllOutputsOff() {
  for (uint8_t pin = FIRST_CONTROLLABLE_PIN; pin <= LAST_CONTROLLABLE_PIN; pin++) {
    setOutput(pin, LOW);
  }
}

bool timedPinInUse(const TimedPin &timedPin) {
  return timedPin.pin >= FIRST_CONTROLLABLE_PIN && timedPin.pin <= LAST_CONTROLLABLE_PIN;
}

void scheduleTimedOff(uint8_t pin, unsigned long durationMs) {
  setOutput(pin, HIGH);
  unsigned long offAt = millis() + durationMs;
  for (size_t i = 0; i < MAX_TIMED_PINS; i++) {
    if (timedPinInUse(timedPins[i]) && timedPins[i].pin == pin) {
      timedPins[i].offAt = offAt;
      return;
    }
  }
  for (size_t i = 0; i < MAX_TIMED_PINS; i++) {
    if (!timedPinInUse(timedPins[i])) {
      timedPins[i].pin = pin;
      timedPins[i].offAt = offAt;
      return;
    }
  }
  setOutput(pin, LOW);
}

void serviceTimedPins() {
  unsigned long now = millis();
  for (size_t i = 0; i < MAX_TIMED_PINS; i++) {
    if (timedPinInUse(timedPins[i]) && static_cast<long>(now - timedPins[i].offAt) >= 0) {
      setOutput(timedPins[i].pin, LOW);
      timedPins[i].pin = 0;
      timedPins[i].offAt = 0;
    }
  }
}

void handleCommand(char *line) {
  char *command = strtok(line, " ");
  char *arg1 = strtok(nullptr, " ");
  char *arg2 = strtok(nullptr, " ");
  char *arg3 = strtok(nullptr, " ");
  char *arg4 = strtok(nullptr, " ");
  char *extra = strtok(nullptr, " ");

  if (command == nullptr || extra != nullptr) {
    sendError();
    return;
  }

  uint8_t pin = 0;
  if (strcmp(command, "ON") == 0 || strcmp(command, "OFF") == 0) {
    if (arg2 != nullptr || !parsePin(arg1, pin)) {
      sendError();
      return;
    }
    setOutput(pin, strcmp(command, "ON") == 0 ? HIGH : LOW);
    sendOk();
    return;
  }

  if (strcmp(command, "TIMED-ON") == 0) {
    bool ok = false;
    long durationMs = parseLong(arg2, ok);
    if (arg3 != nullptr || !parsePin(arg1, pin) || !ok || durationMs < 0) {
      sendError();
      return;
    }
    scheduleTimedOff(pin, static_cast<unsigned long>(durationMs));
    sendOk();
    return;
  }

  if (strcmp(command, "PANTILT") == 0) {
    uint8_t panPin = 0;
    uint8_t tiltPin = 0;
    bool panOk = false;
    bool tiltOk = false;
    float panDegrees = parseFloatValue(arg3, panOk);
    float tiltDegrees = parseFloatValue(arg4, tiltOk);
    if (!parsePin(arg1, panPin) || !parsePin(arg2, tiltPin) || !panOk || !tiltOk) {
      sendError();
      return;
    }
    writeServo(panServo, panPin, panDegrees, PAN_MAX_DEGREES);
    writeServo(tiltServo, tiltPin, tiltDegrees, TILT_MAX_DEGREES);
    sendOk();
    return;
  }

  sendError();
}

void processSerial() {
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') {
      continue;
    }
    if (c == '\n') {
      lineBuffer[lineLength] = '\0';
      if (lineLength > 0) {
        handleCommand(lineBuffer);
      }
      lineLength = 0;
      continue;
    }
    if (lineLength + 1 < MAX_LINE) {
      lineBuffer[lineLength++] = c;
    } else {
      lineLength = 0;
      sendError();
    }
  }
}

void setup() {
  forceAllOutputsOff();
  Serial.begin(115200);
}

void loop() {
  processSerial();
  serviceTimedPins();
}
