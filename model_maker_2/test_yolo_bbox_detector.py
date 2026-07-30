from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from model_maker_2 import yolo_bbox_detector
from model_maker_2.yolo_bbox_detector import (
    _ab_deltas,
    YOLOBBoxDetector,
    summarize_image_scores,
    train_overrides_for_profile,
    write_bbox_size_limits,
)


class FixedSceneProfileTests(unittest.TestCase):
    def test_fixed_scene_disables_every_spatial_or_compositing_transform(self) -> None:
        profile = train_overrides_for_profile("fixed_scene")

        for key in (
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "bgr",
            "hsv_h",
            "hsv_s",
        ):
            self.assertEqual(0.0, profile[key], key)
        self.assertEqual(0, profile["close_mosaic"])
        self.assertEqual(0.4, profile["hsv_v"])

    def test_default_profile_passes_no_augmentation_overrides(self) -> None:
        self.assertEqual({}, train_overrides_for_profile("default"))

    def test_default_training_profile_uses_ultralytics_augmentations(self) -> None:
        parameter = inspect.signature(YOLOBBoxDetector.train).parameters[
            "augmentation_profile"
        ]

        self.assertEqual("default", parameter.default)

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown augmentation profile"):
            train_overrides_for_profile("moving_camera")

    def test_train_passes_fixed_scene_profile_and_shared_seed_to_ultralytics(self) -> None:
        class FakeModel:
            def __init__(self, model: str) -> None:
                self.model_name = model
                self.train_args = None

            def train(self, **kwargs):
                self.train_args = kwargs

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            positives = root / "positives"
            positives.mkdir()
            bbox_file = root / "bboxes.txt"
            bbox_file.write_text("image,label,xmin,ymin,xmax,ymax\n")
            yolo_root = root / "yolo"
            yolo_root.mkdir()
            (yolo_root / "dataset.yaml").write_text("path: .\n")
            detector = YOLOBBoxDetector(
                positives,
                bbox_file,
                yolo_root,
                seed=42,
            )
            detector._prepared = True

            with patch.object(yolo_bbox_detector, "YOLO", FakeModel):
                trained = detector.train(
                    device="cpu",
                    workers=0,
                    amp=False,
                    verbose=False,
                    augmentation_profile="fixed_scene",
                )

        self.assertEqual(42, trained.train_args["seed"])
        self.assertEqual("yolo26n.pt", trained.model_name)
        self.assertEqual(0.0, trained.train_args["mosaic"])
        self.assertEqual(0.0, trained.train_args["translate"])
        self.assertEqual(0.0, trained.train_args["scale"])
        self.assertEqual(0.4, trained.train_args["hsv_v"])

    def test_prepare_excludes_bbox_rows_for_external_holdout_images(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            positives = root / "positives"
            positives.mkdir()
            bbox_file = root / "bboxes.txt"
            bbox_file.write_text(
                "image,label,xmin,ymin,xmax,ymax\n"
                "held-out.jpg,rat,10,10,20,20\n"
            )
            yolo_root = root / "yolo"
            detector = YOLOBBoxDetector(positives, bbox_file, yolo_root)

            detector._prepare_yolo(val_split=0.2)

            self.assertTrue(detector._prepared)
            self.assertFalse((yolo_root / "images" / "train" / "held-out.jpg").exists())
            self.assertFalse((yolo_root / "images" / "val" / "held-out.jpg").exists())

    def test_bbox_limits_skip_external_holdout_before_opening_image(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            positives = root / "positives"
            positives.mkdir()
            bbox_file = root / "bboxes.txt"
            bbox_file.write_text(
                "image,label,xmin,ymin,xmax,ymax\n"
                "held-out.jpg,rat,10,10,20,20\n"
            )

            with patch.object(
                yolo_bbox_detector.Image,
                "open",
                side_effect=AssertionError("missing paths must not reach Pillow"),
            ):
                with self.assertRaisesRegex(ValueError, "no valid bbox rows"):
                    write_bbox_size_limits(
                        bbox_file,
                        positives,
                        root / "limits.json",
                    )


class ABMetricsTests(unittest.TestCase):
    def test_image_level_metrics_separate_recall_from_false_positive_rate(self) -> None:
        rows = summarize_image_scores(
            [
                (True, 0.90),
                (True, 0.55),
                (False, 0.70),
                (False, 0.10),
            ],
            [0.5, 0.8],
        )

        self.assertEqual(2, len(rows))
        self.assertEqual(1.0, rows[0]["positive_recall"])
        self.assertEqual(0.5, rows[0]["false_positive_rate"])
        self.assertAlmostEqual(2.0 / 3.0, rows[0]["image_precision"])
        self.assertEqual(0.5, rows[1]["positive_recall"])
        self.assertEqual(0.0, rows[1]["false_positive_rate"])
        self.assertEqual(1.0, rows[1]["image_precision"])

    def test_ab_delta_is_fixed_scene_minus_control(self) -> None:
        control = {
            "thresholds": [
                {
                    "threshold": 0.6,
                    "positive_recall": 0.8,
                    "false_positive_rate": 0.4,
                    "image_precision": 0.5,
                }
            ]
        }
        fixed_scene = {
            "thresholds": [
                {
                    "threshold": 0.6,
                    "positive_recall": 0.9,
                    "false_positive_rate": 0.1,
                    "image_precision": 0.75,
                }
            ]
        }

        row = _ab_deltas(control, fixed_scene)[0]
        self.assertAlmostEqual(0.1, row["positive_recall_delta"])
        self.assertAlmostEqual(-0.3, row["false_positive_rate_delta"])
        self.assertAlmostEqual(0.25, row["image_precision_delta"])

    def test_invalid_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than 0"):
            summarize_image_scores([(True, 0.5)], [0.0])


if __name__ == "__main__":
    unittest.main()
