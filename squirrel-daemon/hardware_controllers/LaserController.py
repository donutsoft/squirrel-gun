try:
    import RPi.GPIO as GPIO  # type: ignore
except Exception:
    # Provide a no-op GPIO fallback for non-Raspberry Pi environments
    class _DummyGPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0

        @staticmethod
        def setmode(mode):
            print(f"[LaserController] GPIO.setmode({mode}) [noop]")

        @staticmethod
        def setup(pin, mode):
            print(f"[LaserController] GPIO.setup(pin={pin}, mode={mode}) [noop]")

        @staticmethod
        def output(pin, value):
            print(f"[LaserController] GPIO.output(pin={pin}, value={value}) [noop]")

        @staticmethod
        def cleanup():
            print("[LaserController] GPIO.cleanup() [noop]")

    GPIO = _DummyGPIO()  # type: ignore

class LaserController:
    def __init__(self, pin: int = 23):
        self._pin = int(pin)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self._pin, GPIO.OUT)

    def turn_on(self):
        print("Turning on the laser.")
        GPIO.output(self._pin, GPIO.HIGH)

    def turn_off(self):
        print("Turning off the laser.")
        GPIO.output(self._pin, GPIO.LOW)

    def cleanup(self):
        GPIO.cleanup()
