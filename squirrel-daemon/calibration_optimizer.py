from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Dict, Iterable, List, Protocol, Tuple


Target = Tuple[float, float]


class AimModel(Protocol):
    def predict(self, u: float, v: float) -> Tuple[float, float]:
        ...


@dataclass(frozen=True)
class RoundMetrics:
    requested: int
    measured: int
    missed: int
    success_rate: float
    mean_error_px: float | None
    rmse_px: float | None
    max_error_px: float | None

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "measured": self.measured,
            "missed": self.missed,
            "success_rate": self.success_rate,
            "mean_error_px": self.mean_error_px,
            "rmse_px": self.rmse_px,
            "max_error_px": self.max_error_px,
        }


def screen_targets(rows: int, cols: int, margin: float) -> List[Target]:
    """Return a deterministic, serpentine grid spanning the usable image area."""
    rows = max(2, int(rows))
    cols = max(2, int(cols))
    margin = max(0.0, min(0.45, float(margin)))
    span = 1.0 - (2.0 * margin)
    targets: List[Target] = []
    for row in range(rows):
        v = margin + span * (row / (rows - 1))
        col_range: Iterable[int] = range(cols - 1, -1, -1) if row % 2 else range(cols)
        for col in col_range:
            u = margin + span * (col / (cols - 1))
            targets.append((u, v))
    return targets


def target_error_px(
    target: Target,
    *,
    actual_x: float,
    actual_y: float,
    image_width: float,
    image_height: float,
) -> float:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    target_x = float(target[0]) * float(image_width)
    target_y = float(target[1]) * float(image_height)
    return math.hypot(float(actual_x) - target_x, float(actual_y) - target_y)


def summarize_round(errors_px: Iterable[float], requested: int) -> RoundMetrics:
    requested = max(0, int(requested))
    errors = [float(value) for value in errors_px if math.isfinite(float(value)) and float(value) >= 0.0]
    measured = len(errors)
    missed = max(0, requested - measured)
    success_rate = (measured / requested) if requested else 0.0
    if not errors:
        return RoundMetrics(
            requested=requested,
            measured=0,
            missed=missed,
            success_rate=success_rate,
            mean_error_px=None,
            rmse_px=None,
            max_error_px=None,
        )
    return RoundMetrics(
        requested=requested,
        measured=measured,
        missed=missed,
        success_rate=success_rate,
        mean_error_px=sum(errors) / measured,
        rmse_px=math.sqrt(sum(value * value for value in errors) / measured),
        max_error_px=max(errors),
    )


def has_converged(
    metrics: RoundMetrics,
    *,
    target_rmse_px: float,
    target_max_error_px: float,
    min_success_rate: float,
) -> bool:
    return bool(
        metrics.rmse_px is not None
        and metrics.max_error_px is not None
        and metrics.success_rate >= float(min_success_rate)
        and metrics.rmse_px <= float(target_rmse_px)
        and metrics.max_error_px <= float(target_max_error_px)
    )


def prioritize_targets(
    targets: Iterable[Target],
    previous_errors: Dict[Target, float],
    round_index: int,
) -> List[Target]:
    """Try the previous round's worst locations first without losing coverage."""
    ordered = list(targets)
    if previous_errors:
        ordered.sort(key=lambda target: previous_errors.get(target, -1.0), reverse=True)
    elif round_index % 2:
        ordered.reverse()
    return ordered


def has_monotonic_axes(model: AimModel, samples_per_axis: int = 21) -> bool:
    """Reject fitted surfaces that fold over and reverse an aiming axis."""
    samples_per_axis = max(3, int(samples_per_axis))
    coordinates = [index / (samples_per_axis - 1) for index in range(samples_per_axis)]
    for v in coordinates:
        previous_pan = model.predict(coordinates[0], v)[0]
        for u in coordinates[1:]:
            pan = model.predict(u, v)[0]
            if not math.isfinite(pan) or pan <= previous_pan:
                return False
            previous_pan = pan
    for u in coordinates:
        previous_tilt = model.predict(u, coordinates[0])[1]
        for v in coordinates[1:]:
            tilt = model.predict(u, v)[1]
            if not math.isfinite(tilt) or tilt <= previous_tilt:
                return False
            previous_tilt = tilt
    return True


def is_inconsistent_observation(
    candidate: dict,
    previous_rows: Iterable[dict],
    *,
    pixel_radius_fraction: float = 0.025,
    min_angle_delta_deg: float = 4.0,
) -> bool:
    """Detect a stationary false dot reported for substantially different angles."""
    width = float(candidate["img_w"])
    height = float(candidate["img_h"])
    if width <= 0 or height <= 0:
        raise ValueError("candidate image dimensions must be positive")
    u = float(candidate["x_px"]) / width
    v = float(candidate["y_px"]) / height
    pan = float(candidate["pan"])
    tilt = float(candidate["tilt"])
    radius = max(0.0, float(pixel_radius_fraction))
    min_angle_delta = max(0.0, float(min_angle_delta_deg))

    for row in previous_rows:
        row_width = float(row["img_w"])
        row_height = float(row["img_h"])
        if row_width <= 0 or row_height <= 0:
            continue
        row_u = float(row["x_px"]) / row_width
        row_v = float(row["y_px"]) / row_height
        if math.hypot(u - row_u, v - row_v) > radius:
            continue
        angle_delta = math.hypot(
            pan - float(row["pan"]),
            tilt - float(row["tilt"]),
        )
        if angle_delta >= min_angle_delta:
            return True
    return False


def split_error_outliers(
    rows: Iterable[dict],
    *,
    minimum_samples: int = 8,
    minimum_allowance_px: float = 50.0,
    mad_multiplier: float = 6.0,
) -> Tuple[List[dict], List[dict], float | None]:
    """Separate isolated recognition errors from a coherent calibration round."""
    candidates = list(rows)
    if len(candidates) < max(3, int(minimum_samples)):
        return candidates, [], None
    errors = [float(row["error_px"]) for row in candidates]
    median = statistics.median(errors)
    deviations = [abs(error - median) for error in errors]
    mad = statistics.median(deviations)
    robust_sigma = 1.4826 * mad
    limit = median + max(
        float(minimum_allowance_px),
        float(mad_multiplier) * robust_sigma,
    )
    accepted = [row for row in candidates if float(row["error_px"]) <= limit]
    rejected = [row for row in candidates if float(row["error_px"]) > limit]
    return accepted, rejected, limit


def is_usable_checkpoint(
    metrics: RoundMetrics,
    *,
    min_success_rate: float = 0.6,
    max_rmse_px: float = 75.0,
    max_error_px: float = 120.0,
) -> bool:
    """Safety floor for replacing an empty/default model after calibration."""
    return bool(
        metrics.rmse_px is not None
        and metrics.max_error_px is not None
        and metrics.success_rate >= float(min_success_rate)
        and metrics.rmse_px <= float(max_rmse_px)
        and metrics.max_error_px <= float(max_error_px)
    )


def has_screen_coverage(
    rows: Iterable[dict],
    *,
    minimum_span: float = 0.75,
) -> bool:
    """Require validation evidence at both axes' extremes and in all quadrants."""
    points = [
        (float(row["target_u"]), float(row["target_v"]))
        for row in rows
    ]
    if not points:
        return False
    u_values = [point[0] for point in points]
    v_values = [point[1] for point in points]
    if (
        max(u_values) - min(u_values) < float(minimum_span)
        or max(v_values) - min(v_values) < float(minimum_span)
    ):
        return False
    quadrants = {
        (u >= 0.5, v >= 0.5)
        for u, v in points
    }
    return len(quadrants) == 4
