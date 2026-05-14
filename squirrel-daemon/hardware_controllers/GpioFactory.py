from gpiozero import Device


def configure_pin_factory() -> None:
    if Device.pin_factory is not None:
        return

    try:
        from gpiozero.pins.lgpio import LGPIOFactory

        Device.pin_factory = LGPIOFactory()
    except Exception as exc:
        from gpiozero.pins.mock import MockFactory

        print(f"[GPIO] Falling back to mock gpiozero pin factory: {exc}")
        Device.pin_factory = MockFactory()
