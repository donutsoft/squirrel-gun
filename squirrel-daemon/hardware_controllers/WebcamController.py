from pathlib import Path
from typing import Optional, Iterator
import time
import threading
import cv2  # type: ignore

class WebcamController:
    def __init__(self, device: str = "/dev/video0", width: Optional[int] = None, height: Optional[int] = None):
        self.device = device
        self.width = 1280
        self.height = 720
        self._fps = 15
        self._quality = 80
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: bytes = b""
        self._seq = 0
        self._cond = threading.Condition()

    def _device_index(self) -> int:
        index = 0
        if isinstance(self.device, str) and self.device.startswith("/dev/video"):
            try:
                index = int(self.device.replace("/dev/video", ""))
            except ValueError:
                index = 0
        return index

    def _open_capture(self) -> "cv2.VideoCapture":
        cap = cv2.VideoCapture(self._device_index())
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        return cap

    def _capture_with_opencv(self, outfile: Path) -> bool:
        cap = self._open_capture()

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False

        # Write as JPEG
        ok = cv2.imwrite(str(outfile), frame)
        return bool(ok)

    def capture(self, outfile: Path) -> Path:
        outfile = outfile.resolve()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        # Fast path: if we already have a recent frame, use it without locking
        data: Optional[bytes] = self._latest if self._latest else None
        if data:
            with open(outfile, 'wb') as f:
                f.write(data)
            return outfile
        self._capture_with_opencv(outfile)
        return outfile

    def start_stream(self, fps: int = 15, quality: int = 80) -> None:
        self._fps = max(1, int(fps))
        self._quality = max(1, min(100, int(quality)))
        if self._running and self._thread and self._thread.is_alive():
            return
        self._running = True

        def _producer() -> None:
            delay = 1.0 / self._fps
            cap = self._open_capture()
            failures = 0
            try:
                while self._running:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        failures += 1
                        time.sleep(0.05)
                        if failures >= 40:
                            cap.release()
                            cap = self._open_capture()
                            failures = 0
                        continue
                    failures = 0
                    ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
                    if not ok:
                        time.sleep(delay)
                        continue
                    jpg = encoded.tobytes()
                    with self._cond:
                        self._latest = jpg
                        self._seq += 1
                        self._cond.notify_all()
                    time.sleep(delay)
            finally:
                cap.release()

        self._thread = threading.Thread(target=_producer, name="WebcamProducer", daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def mjpeg(self, fps: int = 15, quality: int = 80, boundary: str = "frame") -> Iterator[bytes]:
        self.start_stream(fps=fps, quality=quality)
        delay = 1.0 / self._fps
        last_seq = -1
        while True:
            with self._cond:
                if self._seq == last_seq:
                    self._cond.wait(timeout=delay)
                data = self._latest
                last_seq = self._seq
            if not data:
                time.sleep(delay)
                continue
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
