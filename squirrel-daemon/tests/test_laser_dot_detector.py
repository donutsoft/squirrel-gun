from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from laser_dot_detector import LaserDotOptions, detect_laser_dot


class VerifiedLaserDotDetectorTests(unittest.TestCase):
    def test_two_cycle_verification_rejects_transient_motion(self) -> None:
        background = np.full((120, 160, 3), 20, dtype=np.uint8)
        off_one = background.copy()
        off_two = background.copy()
        on_one = background.copy()
        on_two = background.copy()

        # The actual laser is present at the same location in both on frames.
        cv2.circle(on_one, (125, 30), 4, (255, 255, 255), -1)
        cv2.circle(on_two, (125, 30), 4, (255, 255, 255), -1)
        # A larger transient change appears only during the first cycle.
        cv2.circle(on_one, (35, 90), 8, (255, 255, 255), -1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                "off_one": root / "off_one.png",
                "on_one": root / "on_one.png",
                "off_two": root / "off_two.png",
                "on_two": root / "on_two.png",
            }
            for name, image in (
                ("off_one", off_one),
                ("on_one", on_one),
                ("off_two", off_two),
                ("on_two", on_two),
            ):
                self.assertTrue(cv2.imwrite(str(paths[name]), image))

            result = detect_laser_dot(
                paths["on_one"],
                paths["off_one"],
                options=LaserDotOptions(
                    min_area=4,
                    min_width=2,
                    min_height=2,
                    min_delta_peak=5,
                    min_delta_mean=1,
                    min_on_peak=40,
                    min_on_mean=20,
                ),
                expected_uv=(125 / 160, 30 / 120),
                max_expected_distance_fraction=0.4,
                verification_on_image_path=paths["on_two"],
                verification_off_image_path=paths["off_two"],
            )

        self.assertTrue(result["verified"])
        self.assertEqual("bright-diff-verified", result["method"])
        self.assertIsNotNone(result["dot"])
        self.assertAlmostEqual(125, result["dot"]["cx"], delta=2)
        self.assertAlmostEqual(30, result["dot"]["cy"], delta=2)


if __name__ == "__main__":
    unittest.main()
