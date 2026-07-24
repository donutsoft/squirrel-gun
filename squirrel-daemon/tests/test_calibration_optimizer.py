from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from aim_model import LinearAimer
from calibration_optimizer import (
    has_converged,
    has_monotonic_axes,
    has_screen_coverage,
    is_inconsistent_observation,
    is_usable_checkpoint,
    prioritize_targets,
    screen_targets,
    split_error_outliers,
    summarize_round,
    target_error_px,
)


class CalibrationOptimizerTests(unittest.TestCase):
    def test_screen_targets_cover_inset_grid(self) -> None:
        targets = screen_targets(3, 4, 0.1)

        self.assertEqual(12, len(targets))
        self.assertEqual((0.1, 0.1), targets[0])
        self.assertAlmostEqual(0.9, max(u for u, _ in targets))
        self.assertAlmostEqual(0.9, max(v for _, v in targets))
        self.assertEqual(len(targets), len(set(targets)))

    def test_pixel_error_uses_image_dimensions(self) -> None:
        error = target_error_px(
            (0.5, 0.5),
            actual_x=70,
            actual_y=35,
            image_width=100,
            image_height=50,
        )

        self.assertAlmostEqual(math.hypot(20, 10), error)

    def test_convergence_requires_accuracy_and_detection_coverage(self) -> None:
        accurate = summarize_round([4.0, 6.0, 8.0, 10.0], requested=4)
        missed = summarize_round([4.0, 6.0, 8.0], requested=4)

        self.assertTrue(
            has_converged(
                accurate,
                target_rmse_px=8.0,
                target_max_error_px=10.0,
                min_success_rate=0.9,
            )
        )
        self.assertFalse(
            has_converged(
                missed,
                target_rmse_px=8.0,
                target_max_error_px=10.0,
                min_success_rate=0.9,
            )
        )

    def test_worst_targets_are_revisited_first(self) -> None:
        targets = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]

        ordered = prioritize_targets(
            targets,
            {(0.1, 0.1): 3.0, (0.5, 0.5): 25.0, (0.9, 0.9): 8.0},
            1,
        )

        self.assertEqual((0.5, 0.5), ordered[0])

    def test_rejects_bottom_edge_axis_reversal(self) -> None:
        folded = LinearAimer(
            pan=[
                83.1381, 49.4599, 46.9023, 75.8246, -271.243,
                163.08, -47.4318, 253.4422, -235.9851, 33.3398,
            ],
            tilt=[
                40.0612, 22.4009, 19.2469, -97.5136, 153.3596,
                -62.5799, 75.9178, -96.6305, -11.4302, 39.3624,
            ],
        )

        self.assertFalse(has_monotonic_axes(folded))
        self.assertTrue(has_monotonic_axes(LinearAimer.default()))

    def test_rejects_stationary_false_dot_at_different_angles(self) -> None:
        previous = [{
            "pan": 105.0,
            "tilt": 55.0,
            "x_px": 250.0,
            "y_px": 640.0,
            "img_w": 1000.0,
            "img_h": 1000.0,
        }]
        false_detection = {
            "pan": 165.0,
            "tilt": 75.0,
            "x_px": 252.0,
            "y_px": 643.0,
            "img_w": 1000.0,
            "img_h": 1000.0,
        }
        repeat_measurement = {
            **previous[0],
            "pan": 106.0,
            "tilt": 56.0,
        }

        self.assertTrue(is_inconsistent_observation(false_detection, previous))
        self.assertFalse(is_inconsistent_observation(repeat_measurement, previous))

    def test_isolated_large_error_does_not_poison_round_metrics(self) -> None:
        rows = [{"error_px": value} for value in (12, 14, 18, 20, 21, 24, 28, 31, 396)]

        accepted, rejected, limit = split_error_outliers(rows)
        metrics = summarize_round(
            [float(row["error_px"]) for row in accepted],
            requested=len(rows),
        )

        self.assertEqual([396], [row["error_px"] for row in rejected])
        self.assertIsNotNone(limit)
        self.assertLess(metrics.rmse_px, 30)

    def test_safe_partial_checkpoint_can_replace_empty_default(self) -> None:
        usable = summarize_round([18, 20, 24, 28, 32, 35], requested=10)
        inaccurate = summarize_round([18, 20, 24, 28, 190, 220, 250], requested=10)

        self.assertTrue(is_usable_checkpoint(usable))
        self.assertFalse(is_usable_checkpoint(inaccurate))

    def test_safe_checkpoint_must_cover_entire_screen(self) -> None:
        full_screen = [
            {"target_u": 0.08, "target_v": 0.08},
            {"target_u": 0.92, "target_v": 0.08},
            {"target_u": 0.08, "target_v": 0.92},
            {"target_u": 0.92, "target_v": 0.92},
        ]
        top_only = [
            {"target_u": 0.08, "target_v": 0.08},
            {"target_u": 0.92, "target_v": 0.08},
            {"target_u": 0.08, "target_v": 0.30},
            {"target_u": 0.92, "target_v": 0.30},
        ]

        self.assertTrue(has_screen_coverage(full_screen))
        self.assertFalse(has_screen_coverage(top_only))


class LinearAimerTests(unittest.TestCase):
    def test_default_seed_matches_known_field_of_view(self) -> None:
        model = LinearAimer.default()

        self.assertEqual((108.0, 57.0), model.predict(0.0, 0.0))
        self.assertEqual((173.0, 58.0), model.predict(1.0, 0.0))
        self.assertEqual((110.0, 89.0), model.predict(0.0, 1.0))
        self.assertEqual((171.0, 89.0), model.predict(1.0, 1.0))

    def test_dense_training_uses_cubic_model(self) -> None:
        rows = []
        for row in range(5):
            v = row / 4
            for col in range(5):
                u = col / 4
                rows.append({
                    "img_w": 1000,
                    "img_h": 800,
                    "x_px": u * 1000,
                    "y_px": v * 800,
                    "pan": 108 + 61 * u + 2 * v - 3 * u * v + 2 * u**3,
                    "tilt": 57 + v * 30 + u + 1.5 * v**3,
                })

        model = LinearAimer.default()
        model.fit_from_clicks(rows)

        self.assertEqual(10, len(model.pan))
        pan, tilt = model.predict(0.37, 0.63)
        expected_pan = 108 + 61 * 0.37 + 2 * 0.63 - 3 * 0.37 * 0.63 + 2 * 0.37**3
        expected_tilt = 57 + 0.63 * 30 + 0.37 + 1.5 * 0.63**3
        self.assertAlmostEqual(expected_pan, pan, delta=0.5)
        self.assertAlmostEqual(expected_tilt, tilt, delta=0.5)

    def test_closed_loop_retraining_reduces_screen_error(self) -> None:
        targets = screen_targets(5, 5, 0.1)
        model = LinearAimer.default()
        training_rows = []
        round_rmse = []

        for _round in range(2):
            errors = []
            for u, v in targets:
                pan, tilt = model.predict(u, v)
                # Simulated installation: angle-to-pixel behavior differs from
                # the bootstrap model but is stable and learnable.
                actual_u = (pan - 100.0) / 80.0
                actual_v = (tilt - 50.0) / 40.0
                errors.append(math.hypot(actual_u - u, actual_v - v) * 1000.0)
                training_rows.append({
                    "img_w": 1000,
                    "img_h": 1000,
                    "x_px": actual_u * 1000,
                    "y_px": actual_v * 1000,
                    "pan": pan,
                    "tilt": tilt,
                })

            round_rmse.append(summarize_round(errors, len(targets)).rmse_px)
            model = LinearAimer.default()
            model.fit_from_clicks(training_rows)

        self.assertIsNotNone(round_rmse[0])
        self.assertIsNotNone(round_rmse[1])
        self.assertLess(round_rmse[1], round_rmse[0] * 0.1)


if __name__ == "__main__":
    unittest.main()
