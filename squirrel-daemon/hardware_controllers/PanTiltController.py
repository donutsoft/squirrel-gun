import pigpio

class PanTiltController:
    MIN_US = 500
    MAX_US = 2500
    PAN_MAX_DEG = 270
    TILT_MAX_DEG = 180
    # Simple per-axis offsets (UI domain). Positive means add to UI angle.
    # Example: UI 135 + 28 = 163; UI 90 + 6 = 96
    PAN_OFFSET_DEG = 28.0
    TILT_OFFSET_DEG = 6.0

    def __init__(self):
        self.pigpio = pigpio.pi()
        self.pan_pin =  18
        self.tilt_pin = 19
        # Fixed offsets; kept as attributes for possible runtime tuning
        self._pan_offset = float(self.PAN_OFFSET_DEG)
        self._tilt_offset = float(self.TILT_OFFSET_DEG)

    def _setAngle(self, pin: int, angle: float, max_degrees: float) -> None:
        # Clamp to [0, max_degrees]
        a = max(0.0, min(float(max_degrees), float(angle)))
        # Map to pulse width range
        pulse = self.MIN_US + (a / float(max_degrees)) * (self.MAX_US - self.MIN_US)
        self.pigpio.set_servo_pulsewidth(pin, pulse)

    def setPan(self, angle: float) -> None:
        # Add offset in UI domain, then invert for physical direction
        servo_angle = float(self.PAN_MAX_DEG) - (float(angle) + self._pan_offset)
        self._setAngle(self.pan_pin, servo_angle, self.PAN_MAX_DEG)

    def setTilt(self, angle: float) -> None:
        # Add offset in UI domain, then invert for physical direction
        servo_angle = float(self.TILT_MAX_DEG) - (float(angle) + self._tilt_offset)
        self._setAngle(self.tilt_pin, servo_angle, self.TILT_MAX_DEG)

    def setPanTilt(self, pan: float, tilt: float) -> None:
        self.setPan(pan)
        self.setTilt(tilt)
