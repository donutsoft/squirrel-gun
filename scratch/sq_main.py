from __future__ import annotations

import os
from typing import List, Tuple, Optional

import cv2

DEBUG = True

class SquirrelDetector:
    """
    Minimal detector/plotter for squirrels.

    - Public API: `plot(img_bgr)` where `img_bgr` is a NumPy array (OpenCV image in BGR).
    - Tries Ultralytics YOLO if installed (uses YOLO_WEIGHTS env or 'yolov8n.pt').
    - Optionally falls back to an OpenCV cascade if `cascade_path` points to a valid XML.
    """

    def __init__(self, model_path: Optional[str] = None, cascade_path: Optional[str] = None, conf: float = 0.25):
        self.model_path = model_path or os.environ.get("YOLO_WEIGHTS", "yolov8n.pt")
        self.conf = conf
        self._yolo_model = None

    def _try_load_yolo(self):
        if self._yolo_model is not None:
            return self._yolo_model
        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo_model = YOLO(self.model_path)
        except Exception:
            self._yolo_model = None
        return self._yolo_model

    def plot(self, img_bgr) -> List[Tuple[int, int, int, int]]:
        """
        Accepts an image (NumPy array, BGR as from OpenCV), runs detection, and plots results.

        Returns (detections, used_detector) where detections are (x, y, w, h) in pixels.
        """
        if img_bgr is None:
            raise ValueError("img_bgr must be a valid OpenCV image (NumPy array)")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        detections: List[Tuple[int, int, int, int]] = []
        used_detector = "None"

        # Try YOLO
        model = self._try_load_yolo()
        if model is not None:
            results = model.predict(source=img_bgr, conf=self.conf, verbose=False)
            if results:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    for b in r.boxes:
                        xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append((x1, y1, x2 - x1, y2 - y1))
                    used_detector = "YOLOv8"
            

        if DEBUG:
            from matplotlib import pyplot as plt

            img_out = img_rgb.copy()
            for (x, y, w, h) in detections:
               cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 3)
               
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_rgb)
            axes[0].set_title("Original")
            axes[0].axis("off")

            axes[1].imshow(img_out)
            axes[1].set_title(f"Detected ({used_detector})")
            axes[1].axis("off")

            print(f"Detections: {len(detections)} using {used_detector}")

            plt.tight_layout()
            plt.show()

        return detections

# Example usage (commented):
import cv2
img = cv2.imread("sq1.jpeg")
print(SquirrelDetector().plot(img))

