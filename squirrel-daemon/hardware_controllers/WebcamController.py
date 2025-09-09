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
        """Open the camera with settings that favor low latency.

        - Request V4L2 backend when available.
        - Set small internal buffer to avoid buildup.
        - Prefer MJPG fourcc to reduce encode latency on UVC cams.
        """
        try:
            cap = cv2.VideoCapture(self._device_index(), cv2.CAP_V4L2)
        except Exception:
            cap = cv2.VideoCapture(self._device_index())

        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        # Hint the backend to keep a tiny buffer (best-effort; some backends ignore it)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Prefer MJPG if the camera supports it; this often drops latency on UVC devices
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
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
            # Emit at configured FPS, but always publish the freshest frame.
            # Reads run in a tight loop; only latest frame at each tick is encoded.
            target_dt = 1.0 / self._fps
            next_emit = time.monotonic()
            cap = self._open_capture()
            failures = 0
            last_frame = None
            try:
                while self._running:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        last_frame = frame
                        failures = 0
                    else:
                        failures += 1
                        time.sleep(0.005)
                        if failures >= 40:
                            cap.release()
                            cap = self._open_capture()
                            failures = 0
                        # Even on failure, check if it's time to emit (we'll skip if no frame)

                    now = time.monotonic()
                    if now >= next_emit:
                        if last_frame is not None:
                            ok2, encoded = cv2.imencode(
                                '.jpg', last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
                            )
                            if ok2:
                                jpg = encoded.tobytes()
                                with self._cond:
                                    self._latest = jpg
                                    self._seq += 1
                                    self._cond.notify_all()
                        # Schedule next emit; if we're behind, skip ahead without sleeping
                        next_emit += target_dt
                        # Avoid runaway if system time drifted or we were paused
                        if now - next_emit > 2 * target_dt:
                            next_emit = now + target_dt

                    # Light backoff to avoid pegging CPU while still keeping latency low
                    time.sleep(0.001)
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
