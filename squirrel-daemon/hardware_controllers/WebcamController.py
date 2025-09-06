from pathlib import Path
from typing import Optional, Iterator
import time
import cv2  # type: ignore

class WebcamController:
    def __init__(self, device: str = "/dev/video0", width: Optional[int] = None, height: Optional[int] = None):
        self.device = device
        self.width = 1280
        self.height = 720

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
        self._capture_with_opencv(outfile)
        return outfile

    def mjpeg(self, fps: int = 15, quality: int = 80, boundary: str = "frame") -> Iterator[bytes]:
        """Yield an MJPEG multipart stream as byte chunks.

        Each yield is a complete part including headers, suitable for
        multipart/x-mixed-replace responses.
        """
        delay = 1.0 / max(1, fps)
        cap = self._open_capture()
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.05)
                    continue
                ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if not ok:
                    continue
                jpg = encoded.tobytes()
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(delay)
        finally:
            cap.release()
