from gpiozero import Device
from gpiozero.pins.lgpio import LGPIOFactory


def configure_pin_factory() -> None:
    if Device.pin_factory is not None:
        return

    Device.pin_factory = LGPIOFactory()
