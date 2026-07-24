from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from evaluate_thresholds import (  # noqa: E402
    RecordingResult,
    analyze_images,
    analyze_video,
    generated_negative_paths,
    threshold_summary,
)


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class FakeCV2:
    def __init__(self, frames, images=None):
        self.frames = frames
        self.images = images or {}
        self.capture = None

    def VideoCapture(self, _path):
        self.capture = FakeCapture(self.frames)
        return self.capture

    def imread(self, path):
        return self.images.get(path)


class FakeDetector:
    def __init__(self, scores_by_frame):
        self.scores_by_frame = scores_by_frame
        self.thresholds = []

    def predict_candidates(self, frame, *, score_thresh):
        self.thresholds.append(score_thresh)
        frame_index = int(frame[0, 0, 0])
        detections = [
            {"score": score}
            for score in self.scores_by_frame.get(frame_index, [])
            if score >= score_thresh
        ]
        return frame, detections


def result(label, score):
    return RecordingResult(
        recording=f"{label}_{score}.mp4",
        label=label,
        source="video",
        frames_processed=1,
        detection_count=1,
        best_score=score,
        best_frame=0,
        best_image=None,
        stored_best_score=None,
    )


class ThresholdEvaluatorTests(unittest.TestCase):
    def test_analyze_video_uses_low_threshold_and_keeps_peak_score(self):
        frames = [
            np.full((2, 2, 3), index, dtype=np.uint8)
            for index in range(4)
        ]
        detector = FakeDetector({
            0: [0.2],
            1: [0.8, 0.3],
            2: [],
            3: [0.7],
        })
        fake_cv2 = FakeCV2(frames)

        actual = analyze_video(
            Path("recording.mp4"),
            "true_positive",
            detector,
            cv2_module=fake_cv2,
            mining_threshold=0.001,
            every_n_frames=1,
            stored_score=None,
        )

        self.assertEqual(4, actual.frames_processed)
        self.assertEqual(4, actual.detection_count)
        self.assertEqual(0.8, actual.best_score)
        self.assertEqual(1, actual.best_frame)
        self.assertEqual([0.001] * 4, detector.thresholds)
        self.assertTrue(fake_cv2.capture.released)

    def test_threshold_summary_reports_event_level_recall_and_false_positives(self):
        results = [
            result("true_positive", 0.5),
            result("true_positive", 0.8),
            result("false_positive", 0.6),
            result("false_positive", 0.75),
        ]

        summary = threshold_summary(results, [0.7, 0.8])

        self.assertEqual(1, summary[0]["true_positives_detected"])
        self.assertEqual(1, summary[0]["false_positives_triggered"])
        self.assertEqual(0.5, summary[0]["recall"])
        self.assertEqual(0.5, summary[0]["false_positive_rate"])
        self.assertEqual(0.5, summary[0]["balanced_accuracy"])
        self.assertEqual(1, summary[1]["true_positives_detected"])
        self.assertEqual(0, summary[1]["false_positives_triggered"])

    def test_missing_false_positive_video_can_use_generated_negative_frames(self):
        record = {
            "generated": [
                {"kind": "negatives", "name": "one.jpg", "score": 0.8},
                {"kind": "positives", "name": "ignored.jpg"},
                {"kind": "negatives", "name": "two.jpg", "score": 0.7},
            ]
        }

        paths = generated_negative_paths(record, Path("/data"))

        self.assertEqual([
            Path("/data/negatives/one.jpg"),
            Path("/data/negatives/two.jpg"),
        ], paths)

    def test_analyze_images_scores_generated_frame_fallback(self):
        first = np.zeros((2, 2, 3), dtype=np.uint8)
        second = np.full((2, 2, 3), 1, dtype=np.uint8)
        detector = FakeDetector({0: [0.4], 1: [0.75]})
        paths = [Path("/data/one.jpg"), Path("/data/two.jpg")]
        fake_cv2 = FakeCV2([], {
            str(paths[0]): first,
            str(paths[1]): second,
        })

        actual = analyze_images(
            "missing.mp4",
            "false_positive",
            paths,
            detector,
            cv2_module=fake_cv2,
            mining_threshold=0.001,
            stored_score=0.8,
        )

        self.assertTrue(actual.scorable)
        self.assertEqual("generated_frames", actual.source)
        self.assertEqual(0.75, actual.best_score)
        self.assertEqual("two.jpg", actual.best_image)


if __name__ == "__main__":
    unittest.main()
