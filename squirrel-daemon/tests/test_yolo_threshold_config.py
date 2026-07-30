from __future__ import annotations

import math
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

try:
    import turbojpeg  # type: ignore  # noqa: F401
except ImportError:
    sys.modules["turbojpeg"] = types.SimpleNamespace(
        TurboJPEG=object,
        TJPF_BGR=0,
        TJSAMP_420=0,
    )

from event_detection.yolo import (  # noqa: E402
    DEFAULT_YOLO_SCORE_THRESH,
    YOLOEventDetector,
    validate_yolo_score_threshold,
)
from hardware_controllers import WebcamController as webcam_module  # noqa: E402
from hardware_controllers.WebcamController import WebcamController  # noqa: E402


class FakeDetector:
    def __init__(self) -> None:
        self.threshold = None

    def configure(self, **kwargs) -> None:
        if "score_thresh" in kwargs:
            self.threshold = kwargs["score_thresh"]

    def config(self):
        return {"enabled": True}


class YOLOThresholdTests(unittest.TestCase):
    def test_default_matches_calibrated_yolo26_threshold(self) -> None:
        self.assertEqual(0.05, DEFAULT_YOLO_SCORE_THRESH)

    def test_threshold_validation_rejects_out_of_range_values(self) -> None:
        for value in (0, -0.1, 1.01, math.nan, math.inf, "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    validate_yolo_score_threshold(value)

    def test_controller_updates_loaded_detector_and_reports_threshold(self) -> None:
        controller = WebcamController.__new__(WebcamController)
        controller._yolo_score_thresh = DEFAULT_YOLO_SCORE_THRESH
        controller._squirrel_detector = FakeDetector()
        controller._detector = FakeDetector()

        actual = controller.set_yolo_score_threshold(0.72)

        self.assertEqual(0.72, actual)
        self.assertEqual(0.72, controller._squirrel_detector.threshold)
        self.assertEqual(0.72, controller.motion_config()["yolo_score_thresh"])

    def test_lazy_detector_receives_persisted_threshold(self) -> None:
        controller = WebcamController.__new__(WebcamController)
        controller._yolo_score_thresh = 0.72
        controller._squirrel_detector = None
        controller._squirrel_detector_lock = threading.Lock()

        with patch.object(webcam_module, "YOLOEventDetector", FakeDetector):
            detector = controller.get_squirrel_detector()

        self.assertEqual(0.72, detector.threshold)


if __name__ == "__main__":
    unittest.main()
