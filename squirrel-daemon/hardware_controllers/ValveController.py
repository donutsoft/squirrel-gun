import os
import sys
import time
import fcntl

from gpiozero import OutputDevice

try:
    from hardware_controllers.GpioFactory import configure_pin_factory
except ImportError:
    from GpioFactory import configure_pin_factory


LOCK_PATH = "/tmp/valve_controller.lock"


def acquire_lock(blocking: bool = False):
    f = open(LOCK_PATH, "w")
    try:
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        fcntl.flock(f.fileno(), flags)
        f.write(f"{os.getpid()}\n")
        f.flush()
        return f
    except BlockingIOError:
        f.close()
        return None


def release_lock(lock_file):
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        try:
            lock_file.close()
        except Exception:
            pass


def run_valve(duration_seconds: float, pin: int = 24) -> None:
    lock_file = acquire_lock(blocking=False)
    if lock_file is None:
        print("ValveController already running; exiting.")
        return

    configure_pin_factory()
    valve = OutputDevice(pin, active_high=True, initial_value=False)

    try:
        try:
            valve.on()
            time.sleep(duration_seconds)
        finally:
            # Ensure the valve is switched off at the very end
            valve.off()
            valve.close()
    finally:
        release_lock(lock_file)


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
