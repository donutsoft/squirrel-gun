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
        print("Turning on the laser.")
        self._serial.command("ON", self._pin)

    def turn_off(self):
        print("Turning off the laser.")
        self._serial.command("OFF", self._pin)

    def cleanup(self):
        self.turn_off()
