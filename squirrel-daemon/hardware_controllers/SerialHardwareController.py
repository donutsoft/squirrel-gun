import os
import select
import termios
import threading
import time
from typing import Optional


class SerialHardwareController:
    """Line-oriented serial transport for the Arduino hardware controller."""

    _BAUD_RATES = {
        9600: termios.B9600,
        19200: termios.B19200,
        38400: termios.B38400,
        57600: termios.B57600,
        115200: termios.B115200,
    }

    def __init__(
        self,
        port: Optional[str] = None,
        baud: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ):
        self.port = port or os.environ.get("SQUIRREL_SERIAL_PORT", "/dev/ttyUSB0")
        self.baud = int(baud or os.environ.get("SQUIRREL_SERIAL_BAUD", "115200"))
        self.timeout_sec = float(
            timeout_sec or os.environ.get("SQUIRREL_SERIAL_TIMEOUT_SEC", "2.0")
        )
        self.startup_delay_sec = float(
            os.environ.get("SQUIRREL_SERIAL_STARTUP_DELAY_SEC", "2.0")
        )
        self._fd: Optional[int] = None
        self._rx = bytearray()
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        finally:
            self._fd = None
            self._rx.clear()

    def _open_unlocked(self) -> int:
        if self._fd is not None:
            return self._fd

        speed = self._BAUD_RATES.get(self.baud)
        if speed is None:
            raise ValueError(f"Unsupported serial baud rate: {self.baud}")

        fd = os.open(self.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(fd)
            attrs[0] = 0
            attrs[1] = 0
            attrs[2] = attrs[2] & ~termios.CSIZE
            attrs[2] = attrs[2] | termios.CS8 | termios.CREAD | termios.CLOCAL
            attrs[2] = attrs[2] & ~termios.PARENB
            attrs[2] = attrs[2] & ~termios.CSTOPB
            if hasattr(termios, "CRTSCTS"):
                attrs[2] = attrs[2] & ~termios.CRTSCTS
            attrs[3] = 0
            attrs[4] = speed
            attrs[5] = speed
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except Exception:
            os.close(fd)
            raise

        self._fd = fd
        if self.startup_delay_sec > 0:
            time.sleep(self.startup_delay_sec)
            termios.tcflush(fd, termios.TCIOFLUSH)
        return fd

    def command(self, name: str, *args: object) -> str:
        parts = [name, *(str(arg) for arg in args)]
        line = " ".join(parts)
        return self.send_line(line)

    def send_line(self, line: str) -> str:
        if "\n" in line or "\r" in line:
            raise ValueError("Serial command must be a single line")

        with self._lock:
            fd = self._open_unlocked()
            payload = f"{line}\n".encode("ascii")
            deadline = time.monotonic() + self.timeout_sec
            os.write(fd, payload)

            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                readable, _, _ = select.select([fd], [], [], min(0.1, remaining))
                if not readable:
                    continue
                chunk = os.read(fd, 256)
                if not chunk:
                    continue
                self._rx.extend(chunk)
                while b"\n" in self._rx:
                    raw, _, rest = self._rx.partition(b"\n")
                    self._rx = bytearray(rest)
                    response = raw.decode("ascii", errors="replace").strip()
                    if not response:
                        continue
                    if response == "OK":
                        return response
                    raise RuntimeError(f"Arduino rejected {line!r}: {response}")

            self._close_unlocked()
            raise TimeoutError(f"Timed out waiting for Arduino OK after {line!r}")


_default_controller: Optional[SerialHardwareController] = None
_default_lock = threading.Lock()


def get_serial_hardware_controller() -> SerialHardwareController:
    global _default_controller
    with _default_lock:
        if _default_controller is None:
            _default_controller = SerialHardwareController()
        return _default_controller
