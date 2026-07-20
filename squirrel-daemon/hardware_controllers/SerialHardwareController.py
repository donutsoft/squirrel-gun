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
        self._trace_path = os.environ.get("SQUIRREL_SERIAL_TRACE_PATH", "").strip()
        self._trace_fd: Optional[int] = None
        self._trace_lock = threading.Lock()
        if self._trace_path:
            self._trace_fd = os.open(
                self._trace_path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o640,
            )
            self.trace_event(
                f"controller-created port={self.port!r} baud={self.baud} "
                f"timeout_sec={self.timeout_sec} startup_delay_sec={self.startup_delay_sec}"
            )

    def trace_event(self, message: str) -> None:
        """Append a diagnostic event without affecting hardware behavior."""
        trace_fd = self._trace_fd
        if trace_fd is None:
            return
        now = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        millis = int((now % 1.0) * 1000.0)
        thread = threading.current_thread()
        line = (
            f"{timestamp}.{millis:03d} "
            f"thread={thread.name!r} ident={thread.ident} {message}\n"
        ).encode("utf-8", errors="replace")
        try:
            with self._trace_lock:
                os.write(trace_fd, line)
        except OSError:
            # Tracing must never change command behavior.
            pass

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _close_unlocked(self) -> None:
        if self._fd is None:
            return
        fd = self._fd
        self.trace_event(f"serial-close fd={fd}")
        try:
            os.close(fd)
        finally:
            self._fd = None
            self._rx.clear()

    def _open_unlocked(self) -> int:
        if self._fd is not None:
            return self._fd

        speed = self._BAUD_RATES.get(self.baud)
        if speed is None:
            raise ValueError(f"Unsupported serial baud rate: {self.baud}")

        self.trace_event(f"serial-open-start port={self.port!r}")
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
        self.trace_event(f"serial-opened fd={fd}")
        if self.startup_delay_sec > 0:
            self.trace_event(
                f"serial-startup-delay fd={fd} seconds={self.startup_delay_sec}"
            )
            time.sleep(self.startup_delay_sec)
            termios.tcflush(fd, termios.TCIOFLUSH)
        self.trace_event(f"serial-ready fd={fd}")
        return fd

    def command(self, name: str, *args: object) -> str:
        parts = [name, *(str(arg) for arg in args)]
        line = " ".join(parts)
        return self.send_line(line)

    def send_line(self, line: str) -> str:
        if "\n" in line or "\r" in line:
            raise ValueError("Serial command must be a single line")

        lock_started = time.monotonic()
        self.trace_event(f"command-wait line={line!r}")
        try:
            with self._lock:
                lock_wait_ms = (time.monotonic() - lock_started) * 1000.0
                self.trace_event(
                    f"command-lock-acquired line={line!r} wait_ms={lock_wait_ms:.3f}"
                )
                fd = self._open_unlocked()
                payload = f"{line}\n".encode("ascii")
                deadline = time.monotonic() + self.timeout_sec
                written = os.write(fd, payload)
                self.trace_event(
                    f"command-tx fd={fd} line={line!r} bytes={written}/{len(payload)}"
                )

                while time.monotonic() < deadline:
                    remaining = max(0.0, deadline - time.monotonic())
                    readable, _, _ = select.select([fd], [], [], min(0.1, remaining))
                    if not readable:
                        continue
                    chunk = os.read(fd, 256)
                    if not chunk:
                        continue
                    self.trace_event(f"command-rx fd={fd} bytes={chunk!r}")
                    self._rx.extend(chunk)
                    while b"\n" in self._rx:
                        raw, _, rest = self._rx.partition(b"\n")
                        self._rx = bytearray(rest)
                        response = raw.decode("ascii", errors="replace").strip()
                        if not response:
                            continue
                        if response == "OK":
                            elapsed_ms = (time.monotonic() - lock_started) * 1000.0
                            self.trace_event(
                                f"command-ok fd={fd} line={line!r} elapsed_ms={elapsed_ms:.3f}"
                            )
                            return response
                        self.trace_event(
                            f"command-rejected fd={fd} line={line!r} response={response!r}"
                        )
                        raise RuntimeError(f"Arduino rejected {line!r}: {response}")

                self.trace_event(f"command-timeout fd={fd} line={line!r}")
                self._close_unlocked()
                raise TimeoutError(f"Timed out waiting for Arduino OK after {line!r}")
        except BaseException as exc:
            self.trace_event(
                f"command-exception line={line!r} type={type(exc).__name__} value={exc!r}"
            )
            raise


_default_controller: Optional[SerialHardwareController] = None
_default_lock = threading.Lock()


def get_serial_hardware_controller() -> SerialHardwareController:
    global _default_controller
    with _default_lock:
        if _default_controller is None:
            _default_controller = SerialHardwareController()
        return _default_controller
