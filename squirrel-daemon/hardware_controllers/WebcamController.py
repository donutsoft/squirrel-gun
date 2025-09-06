from pathlib import Path
from typing import Optional
import cv2  # type: ignore

class WebcamController:
    def __init__(self, device: str = "/dev/video0", width: Optional[int] = None, height: Optional[int] = None):
        self.device = device
        self.width = 1280
        self.height = 720

    def _capture_with_opencv(self, outfile: Path) -> bool:
        index = 0
        # If device is like /dev/videoN, extract N as index for OpenCV
        if self.device.startswith("/dev/video"):
            try:
                index = int(self.device.replace("/dev/video", ""))
            except ValueError:
                index = 0

        cap = cv2.VideoCapture(index)
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

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
