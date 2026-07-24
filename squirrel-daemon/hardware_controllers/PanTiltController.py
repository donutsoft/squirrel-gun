import os

from hardware_controllers.SerialHardwareController import get_serial_hardware_controller


class PanTiltController:
    MIN_US = 500
    MAX_US = 2500
    PAN_MAX_DEG = 270
    TILT_MAX_DEG = 180
    PAN_MIN_SAFE_DEG = 70
    PAN_MAX_SAFE_DEG = 200
    TILT_MIN_SAFE_DEG = 20
    TILT_MAX_SAFE_DEG = 170
    # Simple per-axis offsets (UI domain). Positive means add to UI angle.
    # Example: UI 135 + 28 = 163; UI 90 + 6 = 96
    PAN_OFFSET_DEG = 28.0
    TILT_OFFSET_DEG = 6.0

    def __init__(self):
        self.pan_pin = int(os.environ.get("SQUIRREL_PAN_PIN", "5"))
        self.tilt_pin = int(os.environ.get("SQUIRREL_TILT_PIN", "4"))
        self._serial = get_serial_hardware_controller()
        self._pan_offset = float(self.PAN_OFFSET_DEG)
        self._tilt_offset = float(self.TILT_OFFSET_DEG)
        self._last_pan = 135.0
        self._last_tilt = 90.0

    def _servo_angle(self, angle: float, offset: float, max_degrees: float) -> float:
        return float(max_degrees) - (float(angle) + offset)

    def setPan(self, angle: float) -> None:
        self.setPanTilt(angle, self._last_tilt)

    def setTilt(self, angle: float) -> None:
        self.setPanTilt(self._last_pan, angle)

    def setPanTilt(self, pan: float, tilt: float) -> None:
        pan_ok = self.PAN_MIN_SAFE_DEG <= pan <= self.PAN_MAX_SAFE_DEG
        tilt_ok = self.TILT_MIN_SAFE_DEG <= tilt <= self.TILT_MAX_SAFE_DEG
        if pan_ok and tilt_ok:
            self._last_pan = float(pan)
            self._last_tilt = float(tilt)
            pan_angle = max(
                0.0,
                min(
                    float(self.PAN_MAX_DEG),
                    self._servo_angle(pan, self._pan_offset, self.PAN_MAX_DEG),
                ),
            )
            tilt_angle = max(
                0.0,
                min(
                    float(self.TILT_MAX_DEG),
                    self._servo_angle(tilt, self._tilt_offset, self.TILT_MAX_DEG),
                ),
            )
            self._serial.command(
                "PANTILT",
                self.pan_pin,
                self.tilt_pin,
                round(pan_angle, 2),
                round(tilt_angle, 2),
            )
