import os
from typing import Optional

from hardware_controllers.SerialHardwareController import get_serial_hardware_controller


class LaserController:
    def __init__(self, pin: Optional[int] = None):
        if pin is None:
            pin = int(os.environ.get("SQUIRREL_LASER_PIN", "3"))
        self._pin = int(pin)
        self._serial = get_serial_hardware_controller()

    def turn_on(self):
        self._serial.trace_event(f"laser-on-start pin={self._pin}")
        print("Turning on the laser.")
        response = self._serial.command("ON", self._pin)
        self._serial.trace_event(f"laser-on-complete pin={self._pin} response={response!r}")

    def turn_off(self):
        self._serial.trace_event(f"laser-off-start pin={self._pin}")
        print("Turning off the laser.")
        response = self._serial.command("OFF", self._pin)
        self._serial.trace_event(f"laser-off-complete pin={self._pin} response={response!r}")

    def cleanup(self):
        self.turn_off()
