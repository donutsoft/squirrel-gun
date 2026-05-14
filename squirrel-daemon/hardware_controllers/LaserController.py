from gpiozero import OutputDevice

from hardware_controllers.GpioFactory import configure_pin_factory


configure_pin_factory()


class LaserController:
    _devices = {}

    def __init__(self, pin: int = 23):
        self._pin = int(pin)
        if self._pin not in LaserController._devices:
            LaserController._devices[self._pin] = OutputDevice(
                self._pin,
                active_high=True,
                initial_value=False,
            )
        self._device = LaserController._devices[self._pin]

    def turn_on(self):
        print("Turning on the laser.")
        self._device.on()

    def turn_off(self):
        print("Turning off the laser.")
        self._device.off()

    def cleanup(self):
        self._device.off()
        self._device.close()
        LaserController._devices.pop(self._pin, None)
