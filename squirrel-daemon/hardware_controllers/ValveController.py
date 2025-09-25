import sys
import time
import RPi.GPIO as GPIO


def run_valve(duration_seconds: float, pin: int = 24) -> None:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    try:
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(duration_seconds)
    finally:
        # Ensure the valve is switched off at the very end
        GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()


def main(argv):
    if len(argv) < 2:
        print("Usage: python ValveController.py <duration_seconds>")
        sys.exit(1)
    try:
        duration = float(argv[1])
        if duration < 0:
            raise ValueError("Duration must be non-negative")
    except ValueError as e:
        print(f"Invalid duration: {e}")
        sys.exit(2)

    run_valve(duration)


if __name__ == "__main__":
    main(sys.argv)
