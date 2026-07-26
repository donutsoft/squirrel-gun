from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path

import cv2
import numpy as np


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from event_detection.yolo import YOLOEventDetector  # noqa: E402
from hardware_controllers.WebcamController import WebcamController  # noqa: E402


class FakeWriter:
    def __init__(self) -> None:
        self.frames = []

    def write(self, frame) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        return None


class RecordingTrainingSourceTests(unittest.TestCase):
    def test_persisted_recording_has_lossless_detector_ready_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recordings_dir = root / "recordings"
            source_root = recordings_dir / ".training-source"
            recordings_dir.mkdir()
            source_root.mkdir()

            first = np.arange(720 * 1280 * 3, dtype=np.uint8).reshape((720, 1280, 3))
            second = np.flip(first, axis=1).copy()
            writer = FakeWriter()

            controller = WebcamController.__new__(WebcamController)
            controller._recordings_dir = recordings_dir
            controller._training_sources_dir = source_root
            controller._frame_buffer = deque([(100.0, first), (100.5, second)])
            controller._frame_buffer_lock = threading.Lock()
            controller._fps = 3
            controller._pre_motion_sec = 5.0
            controller._post_motion_sec = 5.0
            controller._open_video_writer = lambda *_args: writer

            recording = controller._persist_buffer_to_file(
                event_ts=100.0,
                last_event_ts=100.0,
            )

            self.assertIsNotNone(recording)
            assert recording is not None
            source_dir = source_root / recording.stem
            source_paths = sorted(source_dir.glob("frame*.png"))
            self.assertEqual(2, len(source_paths))
            self.assertEqual(2, len(writer.frames))
            expected = YOLOEventDetector.prepare_input_frame(first)
            actual = cv2.imread(str(source_paths[0]), cv2.IMREAD_COLOR)
            self.assertTrue(np.array_equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
