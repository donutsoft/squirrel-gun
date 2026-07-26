from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from recording_labeler import RecordingLabelService  # noqa: E402


class FakeCapture:
    def __init__(self, frames):
        self.frames = [frame.copy() for frame in frames]
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, _prop):
        return len(self.frames)

    def release(self):
        return None


class FakeCV2:
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    IMWRITE_JPEG_QUALITY = cv2.IMWRITE_JPEG_QUALITY

    def __init__(self, frames):
        self.frames = frames

    def VideoCapture(self, _path):
        return FakeCapture(self.frames)

    @staticmethod
    def imwrite(path, image, options):
        return cv2.imwrite(path, image, options)

    @staticmethod
    def imread(path):
        return cv2.imread(path)


class FakeDetector:
    def __init__(self, detections_by_frame, live_threshold=0.4):
        self.detections_by_frame = detections_by_frame
        self.live_threshold = live_threshold

    def config(self):
        return {"score_thresh": self.live_threshold}

    def predict_candidates(self, frame, *, score_thresh):
        index = int(frame[0, 0, 0])
        detections = [
            dict(item)
            for item in self.detections_by_frame.get(index, [])
            if float(item["score"]) >= score_thresh
        ]
        letterboxed = np.full((320, 320, 3), index, dtype=np.uint8)
        return letterboxed, detections


def detection(score, x1=10, y1=20, x2=30, y2=40):
    return {"score": score, "class": 0, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


class RecordingLabelServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "rec_20260717_120000_123.mp4"
        self.video.touch()
        self.frames = [np.full((8, 12, 3), index, dtype=np.uint8) for index in range(4)]

    def tearDown(self):
        self.temp.cleanup()

    def make_service(self, detector):
        service = RecordingLabelService(
            self.root / "squirrel-training-data",
            lambda: detector,
            cv2_module=FakeCV2(self.frames),
        )
        self.addCleanup(service._executor.shutdown, wait=True)
        return service

    def wait_for_job(self, service, name):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            status = service.status(name)
            if status["state"] in ("complete", "error"):
                return status
            time.sleep(0.01)
        self.fail("labeling job did not finish")

    def test_false_positive_saves_only_frames_at_or_above_live_threshold(self):
        detector = FakeDetector({
            0: [detection(0.10)],
            1: [detection(0.80)],
            2: [detection(0.70)],
            3: [detection(0.60)],
        })
        service = self.make_service(detector)

        service.start(self.video, "false_positive")
        status = self.wait_for_job(service, self.video.name)

        self.assertEqual("complete", status["state"])
        self.assertEqual(1, status["saved_frames"])
        names = {path.name for path in service.negatives_dir.glob("*.jpg")}
        self.assertEqual({
            "rec_20260717_120000_123_frame00000001.jpg",
        }, names)
        self.assertEqual([], service._read_bbox_rows())

    def test_uses_lossless_detector_frames_instead_of_mp4_frames(self):
        source_dir = self.video.parent / ".training-source" / self.video.stem
        source_dir.mkdir(parents=True)
        source_frame = np.full((320, 320, 3), 9, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(source_dir / "frame00000000.png"), source_frame))
        detector = FakeDetector({
            0: [detection(0.90)],
            9: [detection(0.80)],
        })
        service = self.make_service(detector)

        service.start(self.video, "false_positive")
        status = self.wait_for_job(service, self.video.name)

        self.assertEqual("complete", status["state"])
        self.assertEqual(1, status["processed_frames"])
        frames = service.list_frames()["negatives"]
        self.assertEqual(1, len(frames))
        self.assertEqual(0.80, frames[0]["score"])

    def test_true_positive_saves_below_live_threshold_and_writes_one_best_box(self):
        detector = FakeDetector({
            0: [detection(0.04)],
            1: [detection(0.10)],
            2: [detection(0.50)],
            3: [detection(0.20), detection(0.30, 50, 60, 70, 80)],
        })
        service = self.make_service(detector)

        service.start(self.video, "true_positive")
        status = self.wait_for_job(service, self.video.name)

        self.assertEqual("complete", status["state"])
        self.assertEqual(2, status["saved_frames"])
        names = {path.name for path in service.positives_dir.glob("*.jpg")}
        self.assertEqual({
            "rec_20260717_120000_123_frame00000001.jpg",
            "rec_20260717_120000_123_frame00000003.jpg",
        }, names)
        rows = service._read_bbox_rows()
        self.assertEqual(2, len(rows))
        self.assertEqual({"rat"}, {row["label"] for row in rows})
        self.assertEqual({"10", "50"}, {row["xmin"] for row in rows})
        frames = {item["name"]: item for item in service.list_frames()["positives"]}
        frame = frames["rec_20260717_120000_123_frame00000003.jpg"]
        self.assertEqual(0.30, frame["score"])
        self.assertEqual(1, len(frame["boxes"]))

    def test_recreates_dataset_if_deleted_while_service_is_running(self):
        detector = FakeDetector({1: [detection(0.80)]})
        service = self.make_service(detector)
        shutil.rmtree(service.data_root)

        service.start(self.video, "false_positive")
        status = self.wait_for_job(service, self.video.name)

        self.assertEqual("complete", status["state"])
        self.assertTrue(service.staging_dir.is_dir())
        self.assertTrue(service.positives_dir.is_dir())
        self.assertTrue(service.negatives_dir.is_dir())
        self.assertTrue(service.bbox_path.is_file())
        self.assertTrue(service.manifest_path.is_file())

    def test_replacing_box_removes_all_old_boxes_and_clear_removes_replacement(self):
        detector = FakeDetector({
            1: [
                detection(0.10),
                detection(0.20, 50, 60, 70, 80),
            ],
        })
        service = self.make_service(detector)
        service.start(self.video, "true_positive")
        self.assertEqual("complete", self.wait_for_job(service, self.video.name)["state"])
        name = "rec_20260717_120000_123_frame00000001.jpg"
        rows = service._read_bbox_rows()
        rows.append({
            "image": name,
            "label": "rat",
            "xmin": 1,
            "ymin": 2,
            "xmax": 3,
            "ymax": 4,
        })
        service._write_bbox_rows(rows)

        result = service.replace_bounding_box(
            name,
            {"xmin": 5, "ymin": 6, "xmax": 100, "ymax": 110},
        )

        self.assertEqual(2, result["removed_boxes"])
        saved = [row for row in service._read_bbox_rows() if row["image"] == name]
        self.assertEqual(1, len(saved))
        self.assertEqual("5", saved[0]["xmin"])

        result = service.replace_bounding_box(name, None)
        self.assertEqual(1, result["removed_boxes"])
        self.assertIsNone(result["box"])
        self.assertEqual([], service._read_bbox_rows())

    def test_rejects_box_outside_image(self):
        detector = FakeDetector({1: [detection(0.10)]})
        service = self.make_service(detector)
        service.start(self.video, "true_positive")
        self.assertEqual("complete", self.wait_for_job(service, self.video.name)["state"])
        name = "rec_20260717_120000_123_frame00000001.jpg"

        with self.assertRaisesRegex(ValueError, "within the 320x320 image"):
            service.replace_bounding_box(
                name,
                {"xmin": 5, "ymin": 6, "xmax": 321, "ymax": 110},
            )

    def test_deleting_positive_also_removes_its_bbox_rows(self):
        detector = FakeDetector({1: [detection(0.10)]})
        service = self.make_service(detector)
        service.start(self.video, "true_positive")
        self.assertEqual("complete", self.wait_for_job(service, self.video.name)["state"])
        name = "rec_20260717_120000_123_frame00000001.jpg"

        result = service.delete_frame("positives", name)

        self.assertEqual(1, result["removed_boxes"])
        self.assertFalse((service.positives_dir / name).exists())
        self.assertEqual([], service._read_bbox_rows())
        self.assertEqual(0, service.status(self.video.name)["saved_frames"])

    def test_relabeling_replaces_previous_generated_assets(self):
        detector = FakeDetector({
            0: [detection(0.10)],
            1: [detection(0.20)],
            2: [detection(0.50)],
        })
        service = self.make_service(detector)
        service.start(self.video, "true_positive")
        self.assertEqual("complete", self.wait_for_job(service, self.video.name)["state"])
        self.assertEqual(2, len(list(service.positives_dir.glob("*.jpg"))))

        service.start(self.video, "false_positive")
        status = self.wait_for_job(service, self.video.name)

        self.assertEqual("false_positive", status["label"])
        self.assertEqual([], list(service.positives_dir.glob("*.jpg")))
        self.assertEqual([], service._read_bbox_rows())
        self.assertEqual(1, len(list(service.negatives_dir.glob("*.jpg"))))


if __name__ == "__main__":
    unittest.main()
