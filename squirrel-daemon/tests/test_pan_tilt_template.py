from pathlib import Path
import unittest


class PanTiltTemplateInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.template = (
            Path(__file__).resolve().parents[1]
            / "templates"
            / "PanTiltControl.html"
        ).read_text(encoding="utf-8")

    def test_left_click_aims_and_right_click_records(self) -> None:
        self.assertIn("img.addEventListener('click'", self.template)
        self.assertIn("await aimAtImagePoint(point)", self.template)
        self.assertIn("img.addEventListener('contextmenu'", self.template)
        self.assertIn("e.preventDefault()", self.template)
        self.assertIn("await recordImagePoint(point)", self.template)
        self.assertIn("fetch('/api/aim'", self.template)
        self.assertIn("fetch('/api/click'", self.template)

    def test_legacy_record_mode_toggle_is_removed(self) -> None:
        self.assertNotIn('id="record-clicks"', self.template)
        self.assertNotIn("recordToggle", self.template)

    def test_recoil_calibration_and_compensated_fire_controls_are_wired(self) -> None:
        self.assertIn('id="calibrate-recoil"', self.template)
        self.assertIn("fetch('/api/calibration/recoil'", self.template)
        self.assertIn('id="recoil-calibration-status"', self.template)
        self.assertIn("data.recoil_calibration_id", self.template)


if __name__ == "__main__":
    unittest.main()
