from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


DAEMON_ROOT = Path(__file__).resolve().parents[1]
if str(DAEMON_ROOT) not in sys.path:
    sys.path.insert(0, str(DAEMON_ROOT))

from db import ClickStore
from recoil_calibration import calculate_recoil_calibration, compensated_angles


class SimpleAimer:
    def predict(self, u: float, v: float):
        return 100.0 + (10.0 * u), 50.0 + (20.0 * v)


class RecoilCalibrationTests(unittest.TestCase):
    def test_laser_rising_produces_downward_prefire_compensation(self) -> None:
        calibration = calculate_recoil_calibration(
            baseline_dot={"cx": 500.0, "cy": 400.0},
            firing_dot={"cx": 500.0, "cy": 380.0},
            image_width=1000.0,
            image_height=800.0,
            calibration_pan=140.0,
            calibration_tilt=80.0,
            aimer=SimpleAimer(),
        )

        self.assertEqual(-20.0, calibration["shift_y_px"])
        self.assertEqual(20.0, calibration["compensation_y_px"])
        self.assertEqual(0.025, calibration["compensation_v"])
        self.assertGreater(calibration["tilt_offset_deg"], 0.0)

    def test_compensated_angles_apply_both_stored_offsets(self) -> None:
        pan, tilt = compensated_angles(
            pan=135.0,
            tilt=90.0,
            calibration={"pan_offset_deg": -0.5, "tilt_offset_deg": 1.25},
            pan_min=70.0,
            pan_max=200.0,
            tilt_min=20.0,
            tilt_max=170.0,
        )

        self.assertEqual(134.5, pan)
        self.assertEqual(91.25, tilt)

    def test_compensation_outside_safe_angle_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "safe range"):
            compensated_angles(
                pan=199.0,
                tilt=90.0,
                calibration={"pan_offset_deg": 2.0, "tilt_offset_deg": 0.0},
                pan_min=70.0,
                pan_max=200.0,
                tilt_min=20.0,
                tilt_max=170.0,
            )

    def test_recoil_calibration_is_persisted_in_database(self) -> None:
        calibration = calculate_recoil_calibration(
            baseline_dot={"cx": 300.0, "cy": 250.0},
            firing_dot={"cx": 302.0, "cy": 238.0},
            image_width=640.0,
            image_height=480.0,
            calibration_pan=135.0,
            calibration_tilt=90.0,
            aimer=SimpleAimer(),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ClickStore(Path(directory) / "clicks.db")
            calibration_id = store.record_recoil_calibration(
                calibration,
                water_duration_sec=2.0,
                metadata={"run_id": "recoil_test"},
            )
            stored = store.latest_recoil_calibration()

        self.assertIsNotNone(stored)
        self.assertEqual(calibration_id, stored["id"])
        self.assertEqual(-12.0, stored["shift_y_px"])
        self.assertEqual("recoil_test", stored["metadata"]["run_id"])


if __name__ == "__main__":
    unittest.main()
