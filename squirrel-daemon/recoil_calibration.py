from __future__ import annotations

import math
from typing import Any, Dict, Protocol, Tuple


class AimModel(Protocol):
    def predict(self, u: float, v: float) -> Tuple[float, float]:
        ...


def _finite(name: str, value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _dot_center(name: str, dot: Dict[str, Any]) -> Tuple[float, float]:
    if not isinstance(dot, dict):
        raise ValueError(f"{name} laser dot is missing")
    return _finite(f"{name}.cx", dot.get("cx")), _finite(f"{name}.cy", dot.get("cy"))


def calculate_recoil_calibration(
    *,
    baseline_dot: Dict[str, Any],
    firing_dot: Dict[str, Any],
    image_width: float,
    image_height: float,
    calibration_pan: float,
    calibration_tilt: float,
    aimer: AimModel,
) -> Dict[str, float]:
    """Convert observed laser motion into a pre-fire aim correction.

    ``shift_*`` is the observed laser movement caused by water pressure.
    ``compensation_*`` is the opposite movement requested before opening the
    valve. The angle offsets are local deltas from the current aim model, which
    lets firing compensate without changing the normal laser calibration.
    """
    width = _finite("image_width", image_width)
    height = _finite("image_height", image_height)
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    baseline_x, baseline_y = _dot_center("baseline", baseline_dot)
    firing_x, firing_y = _dot_center("firing", firing_dot)
    for name, value, limit in (
        ("baseline.cx", baseline_x, width),
        ("baseline.cy", baseline_y, height),
        ("firing.cx", firing_x, width),
        ("firing.cy", firing_y, height),
    ):
        if value < 0 or value > limit:
            raise ValueError(f"{name} is outside the image")

    shift_x = firing_x - baseline_x
    shift_y = firing_y - baseline_y
    compensation_x = -shift_x
    compensation_y = -shift_y
    baseline_u = baseline_x / width
    baseline_v = baseline_y / height
    compensated_u = baseline_u + (compensation_x / width)
    compensated_v = baseline_v + (compensation_y / height)
    if not 0.0 <= compensated_u <= 1.0 or not 0.0 <= compensated_v <= 1.0:
        raise ValueError(
            "measured recoil cannot be compensated at this image position; "
            "aim farther from the image edge and retry"
        )

    base_pan, base_tilt = aimer.predict(baseline_u, baseline_v)
    compensated_pan, compensated_tilt = aimer.predict(compensated_u, compensated_v)
    pan_offset = _finite("pan_offset_deg", compensated_pan - base_pan)
    tilt_offset = _finite("tilt_offset_deg", compensated_tilt - base_tilt)

    return {
        "baseline_x_px": baseline_x,
        "baseline_y_px": baseline_y,
        "firing_x_px": firing_x,
        "firing_y_px": firing_y,
        "shift_x_px": shift_x,
        "shift_y_px": shift_y,
        "shift_magnitude_px": math.hypot(shift_x, shift_y),
        "compensation_x_px": compensation_x,
        "compensation_y_px": compensation_y,
        "compensation_u": compensation_x / width,
        "compensation_v": compensation_y / height,
        "pan_offset_deg": pan_offset,
        "tilt_offset_deg": tilt_offset,
        "img_w": width,
        "img_h": height,
        "calibration_pan": _finite("calibration_pan", calibration_pan),
        "calibration_tilt": _finite("calibration_tilt", calibration_tilt),
    }


def compensated_angles(
    *,
    pan: float,
    tilt: float,
    calibration: Dict[str, Any],
    pan_min: float,
    pan_max: float,
    tilt_min: float,
    tilt_max: float,
) -> Tuple[float, float]:
    """Apply a stored recoil correction, rejecting unsafe partial corrections."""
    compensated_pan = _finite("pan", pan) + _finite(
        "pan_offset_deg", calibration.get("pan_offset_deg")
    )
    compensated_tilt = _finite("tilt", tilt) + _finite(
        "tilt_offset_deg", calibration.get("tilt_offset_deg")
    )
    safe_pan_min = _finite("pan_min", pan_min)
    safe_pan_max = _finite("pan_max", pan_max)
    safe_tilt_min = _finite("tilt_min", tilt_min)
    safe_tilt_max = _finite("tilt_max", tilt_max)
    if not safe_pan_min <= compensated_pan <= safe_pan_max:
        raise ValueError("recoil-compensated pan angle is outside the safe range")
    if not safe_tilt_min <= compensated_tilt <= safe_tilt_max:
        raise ValueError("recoil-compensated tilt angle is outside the safe range")
    return compensated_pan, compensated_tilt
