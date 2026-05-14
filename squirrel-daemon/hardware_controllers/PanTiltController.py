import os

from hardware_controllers.KernelPwmServo import KernelPwmServo


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
        self.pan_pin =  18
        self.tilt_pin = 19
        pwm_chip = os.environ.get("SQUIRREL_PWM_CHIP", "pwmchip0")
        # On Raspberry Pi 5, GPIO 18/19 are hardware PWM channels 2/3.
        self.pan_servo = self._make_servo(
            int(os.environ.get("SQUIRREL_PAN_PWM_CHANNEL", "2")),
            pwm_chip,
            self.PAN_MAX_DEG,
        )
        self.tilt_servo = self._make_servo(
            int(os.environ.get("SQUIRREL_TILT_PWM_CHANNEL", "3")),
            pwm_chip,
            self.TILT_MAX_DEG,
        )
        # Fixed offsets; kept as attributes for possible runtime tuning
        self._pan_offset = float(self.PAN_OFFSET_DEG)
        self._tilt_offset = float(self.TILT_OFFSET_DEG)

    def _make_servo(self, channel: int, pwm_chip: str, max_degrees: float) -> KernelPwmServo:
        return KernelPwmServo(
            channel,
            chip=pwm_chip,
            min_angle=0.0,
            max_angle=float(max_degrees),
            min_pulse_us=self.MIN_US,
            max_pulse_us=self.MAX_US,
        )

    def _setAngle(self, servo: KernelPwmServo, angle: float, max_degrees: float) -> None:
        # Clamp to [0, max_degrees]
        a = max(0.0, min(float(max_degrees), float(angle)))
        servo.set_angle(a)

    def setPan(self, angle: float) -> None:
        # Add offset in UI domain, then invert for physical direction
        servo_angle = float(self.PAN_MAX_DEG) - (float(angle) + self._pan_offset)
        if angle >= 70 and angle <= 200:
            self._setAngle(self.pan_servo, servo_angle, self.PAN_MAX_DEG)

    def setTilt(self, angle: float) -> None:
        # Add offset in UI domain, then invert for physical direction
        servo_angle = float(self.TILT_MAX_DEG) - (float(angle) + self._tilt_offset)
        if angle >=20 and angle <= 170:
            self._setAngle(self.tilt_servo, servo_angle, self.TILT_MAX_DEG)

    def setPanTilt(self, pan: float, tilt: float) -> None:
        self.setPan(pan)
        self.setTilt(tilt)
