from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2  # type: ignore
import numpy as np


@dataclass
class LaserDotOptions:
    min_area: float = 20.0
    max_area: float = 300.0
    max_aspect: float = 2.5
    min_width: float = 4.0
    min_height: float = 4.0
    threshold: Optional[float] = None
    percentile: float = 99.8
    min_score: float = 18.0
    blur_size: int = 31
    blue_tolerance: float = 25.0
    open_size: int = 1
    close_size: int = 9
    min_delta_peak: float = 10.0
    min_delta_mean: float = 1.5
    min_on_peak: float = 180.0
    min_on_mean: float = 80.0
    peak_window: int = 21
    peak_candidates: int = 8


def _odd_size(size: int) -> int:
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def _load_image(path: Path) -> Any:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _scene_mode(image: Any) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)
    b, g, r = [channel.astype(np.float32) for channel in cv2.split(image)]
    colorfulness = np.sqrt(((r - g) ** 2).mean() + ((((r + g) / 2) - b) ** 2).mean())

    if saturation.mean() < 25 and np.percentile(saturation, 90) < 55 and colorfulness < 10:
        return "night"
    return "day"


def _threshold_mask(score: Any, options: LaserDotOptions) -> Any:
    threshold = options.threshold
    if threshold is None:
        threshold = max(options.min_score, float(np.percentile(score, options.percentile)))

    return (score >= threshold).astype(np.uint8) * 255


def _red_delta_support(on_image: Any, off_image: Any, options: LaserDotOptions) -> Any:
    b, g, r = [channel.astype(np.float32) for channel in cv2.split(on_image)]
    off_b, off_g, off_r = [channel.astype(np.float32) for channel in cv2.split(off_image)]

    red_delta = r - off_r
    green_delta = g - off_g
    blue_delta = b - off_b

    return np.minimum(red_delta - green_delta, red_delta - blue_delta + options.blue_tolerance)


def _red_diff_score(on_image: Any, off_image: Any, options: LaserDotOptions) -> tuple[Any, Any]:
    delta_chroma = _red_delta_support(on_image, off_image, options)
    return 3 * np.maximum(delta_chroma, 0), delta_chroma


def _bright_diff_score(on_image: Any, off_image: Any, options: LaserDotOptions) -> tuple[Any, Any]:
    gray = cv2.cvtColor(on_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    off_gray = cv2.cvtColor(off_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    brightness_delta = gray - off_gray
    blur_shape = (_odd_size(options.blur_size), _odd_size(options.blur_size))
    local_delta = brightness_delta - cv2.GaussianBlur(brightness_delta, blur_shape, 0)
    score = np.maximum(brightness_delta, 0) + (2 * np.maximum(local_delta, 0))
    return score, brightness_delta


def _candidates_from_score(
    score: Any,
    mask: Any,
    delta_support: Any,
    on_gray: Any,
    options: LaserDotOptions,
) -> tuple[list[Dict[str, float]], Any]:
    if options.open_size > 1:
        kernel = np.ones((int(options.open_size), int(options.open_size)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if options.close_size > 1:
        kernel = np.ones((int(options.close_size), int(options.close_size)), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[Dict[str, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if not options.min_area <= area <= options.max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < options.min_width or h < options.min_height:
            continue
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > options.max_aspect:
            continue

        crop = score[y : y + h, x : x + w]
        delta_crop = delta_support[y : y + h, x : x + w]
        on_crop = on_gray[y : y + h, x : x + w]
        peak = float(crop.max())
        mean = float(crop.mean())
        delta_peak = float(delta_crop.max())
        delta_mean = float(delta_crop.mean())
        if delta_peak < options.min_delta_peak or delta_mean < options.min_delta_mean:
            continue

        contour_mask = np.zeros((h, w), np.uint8)
        shifted = contour.copy()
        shifted[:, :, 0] -= x
        shifted[:, :, 1] -= y
        cv2.drawContours(contour_mask, [shifted], -1, 1, thickness=-1)
        weights = np.maximum(delta_crop, 0) * contour_mask
        weight_sum = float(weights.sum())
        contour_on = on_crop[contour_mask > 0]
        on_peak = float(contour_on.max()) if contour_on.size else float(on_crop.max())
        on_mean = float(contour_on.mean()) if contour_on.size else float(on_crop.mean())
        if on_peak < options.min_on_peak or on_mean < options.min_on_mean:
            continue

        if weight_sum > 0:
            ys, xs = np.indices(weights.shape)
            cx = float(x + ((xs * weights).sum() / weight_sum))
            cy = float(y + ((ys * weights).sum() / weight_sum))
        else:
            cx = float(x + (w / 2))
            cy = float(y + (h / 2))

        candidates.append(
            {
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h),
                "cx": cx,
                "cy": cy,
                "area": float(area),
                "score": peak,
                "mean_score": mean,
                "delta_peak": delta_peak,
                "delta_mean": delta_mean,
                "on_peak": on_peak,
                "on_mean": on_mean,
                "sum_score": float(crop.sum()),
                "rank_score": float((delta_peak * 4.0) + (on_peak * 2.0) + mean - (np.sqrt(area) * 0.2)),
                "source": "contour",
            }
        )

    return candidates, mask


def _dedupe_candidates(candidates: list[Dict[str, float]]) -> list[Dict[str, float]]:
    candidates.sort(key=lambda candidate: candidate["rank_score"], reverse=True)
    deduped: list[Dict[str, float]] = []
    for candidate in candidates:
        cx = float(candidate["cx"])
        cy = float(candidate["cy"])
        duplicate = False
        for kept in deduped:
            dx = cx - float(kept["cx"])
            dy = cy - float(kept["cy"])
            if (dx * dx + dy * dy) <= 100.0:
                duplicate = True
                break
        if not duplicate:
            deduped.append(candidate)
    return deduped


def _peak_candidates(
    score: Any,
    delta_support: Any,
    on_gray: Any,
    options: LaserDotOptions,
) -> list[Dict[str, float]]:
    positive = np.maximum(delta_support, 0).astype(np.float32)
    peak_mask = (positive >= float(options.min_delta_peak)).astype(np.uint8) * 255
    if not np.any(peak_mask):
        return []

    # Work from strongest local delta peaks. Suppress a small area around each
    # chosen peak so a saturated laser spot contributes one compact candidate.
    work = positive.copy()
    candidates: list[Dict[str, float]] = []
    h_img, w_img = positive.shape[:2]
    half = max(2, int(options.peak_window) // 2)
    max_candidates = max(1, int(options.peak_candidates))

    for _ in range(max_candidates):
        _, peak_value, _, max_loc = cv2.minMaxLoc(work)
        if float(peak_value) < float(options.min_delta_peak):
            break

        px, py = max_loc
        x1 = max(0, int(px) - half)
        y1 = max(0, int(py) - half)
        x2 = min(w_img, int(px) + half + 1)
        y2 = min(h_img, int(py) + half + 1)

        local_delta = positive[y1:y2, x1:x2]
        local_score = score[y1:y2, x1:x2]
        local_on = on_gray[y1:y2, x1:x2]
        local_mask = local_delta >= max(float(options.min_delta_peak), float(peak_value) * 0.35)
        if not np.any(local_mask):
            work[y1:y2, x1:x2] = 0
            continue

        ys, xs = np.where(local_mask)
        bx1 = int(xs.min())
        bx2 = int(xs.max()) + 1
        by1 = int(ys.min())
        by2 = int(ys.max()) + 1
        candidate_delta = local_delta[by1:by2, bx1:bx2]
        candidate_score = local_score[by1:by2, bx1:bx2]
        candidate_on = local_on[by1:by2, bx1:bx2]
        candidate_mask = local_mask[by1:by2, bx1:bx2]
        area = float(candidate_mask.sum())
        if area < float(options.min_area):
            work[y1:y2, x1:x2] = 0
            continue

        delta_values = candidate_delta[candidate_mask]
        on_values = candidate_on[candidate_mask]
        delta_peak = float(delta_values.max())
        delta_mean = float(delta_values.mean())
        on_peak = float(on_values.max())
        on_mean = float(on_values.mean())
        if (
            delta_peak < options.min_delta_peak
            or delta_mean < options.min_delta_mean
            or on_peak < options.min_on_peak
            or on_mean < options.min_on_mean
        ):
            work[y1:y2, x1:x2] = 0
            continue

        weights = candidate_delta * candidate_mask
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            wys, wxs = np.indices(weights.shape)
            cx = float(x1 + bx1 + ((wxs * weights).sum() / weight_sum))
            cy = float(y1 + by1 + ((wys * weights).sum() / weight_sum))
        else:
            cx = float(x1 + bx1 + ((bx2 - bx1) / 2))
            cy = float(y1 + by1 + ((by2 - by1) / 2))

        box_x = float(x1 + bx1)
        box_y = float(y1 + by1)
        box_w = float(bx2 - bx1)
        box_h = float(by2 - by1)
        if box_w < float(options.min_width) or box_h < float(options.min_height):
            work[y1:y2, x1:x2] = 0
            continue
        aspect = max(box_w, box_h) / max(1.0, min(box_w, box_h))
        if aspect > float(options.max_aspect):
            work[y1:y2, x1:x2] = 0
            continue

        mean_score = float(candidate_score[candidate_mask].mean())
        candidates.append({
            "x": box_x,
            "y": box_y,
            "w": box_w,
            "h": box_h,
            "cx": cx,
            "cy": cy,
            "area": area,
            "score": float(candidate_score[candidate_mask].max()),
            "mean_score": mean_score,
            "delta_peak": delta_peak,
            "delta_mean": delta_mean,
            "on_peak": on_peak,
            "on_mean": on_mean,
            "sum_score": float(candidate_score[candidate_mask].sum()),
            "rank_score": float((delta_peak * 6.0) + (on_peak * 3.0) + mean_score - (np.sqrt(area) * 0.2)),
            "source": "peak",
        })

        work[max(0, py - half):min(h_img, py + half + 1), max(0, px - half):min(w_img, px + half + 1)] = 0

    return candidates


def detect_laser_dot(
    on_image_path: Path,
    off_image_path: Path,
    options: Optional[LaserDotOptions] = None,
    debug_paths: Optional[Dict[str, Path]] = None,
    expected_xy: Optional[tuple[float, float]] = None,
    max_expected_distance: Optional[float] = None,
) -> Dict[str, Any]:
    options = options or LaserDotOptions()
    on_image = _load_image(Path(on_image_path))
    off_image = _load_image(Path(off_image_path))
    if on_image.shape != off_image.shape:
        raise ValueError("On and off images must have the same dimensions")

    scene = _scene_mode(on_image)
    method = "bright-diff" if scene == "night" else "red-diff"
    score, delta_support = (
        _bright_diff_score(on_image, off_image, options)
        if scene == "night"
        else _red_diff_score(on_image, off_image, options)
    )
    on_gray = cv2.cvtColor(on_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = _threshold_mask(score, options)
    candidates, _mask = _candidates_from_score(score, mask, delta_support, on_gray, options)
    candidates = _dedupe_candidates(candidates + _peak_candidates(score, delta_support, on_gray, options))
    h, w = on_image.shape[:2]
    if expected_xy is not None:
        expected_x, expected_y = expected_xy
        for candidate in candidates:
            dx = float(candidate["cx"]) - float(expected_x)
            dy = float(candidate["cy"]) - float(expected_y)
            distance = float(np.sqrt((dx * dx) + (dy * dy)))
            candidate["expected_distance_px"] = distance
            candidate["expected_rank_score"] = float(candidate["rank_score"] - (distance * 4.0))
        eligible = candidates
        if max_expected_distance is not None:
            eligible = [
                candidate
                for candidate in candidates
                if float(candidate.get("expected_distance_px", 0.0)) <= float(max_expected_distance)
            ]
        eligible.sort(key=lambda candidate: candidate.get("expected_rank_score", candidate["rank_score"]), reverse=True)
        dot = eligible[0] if eligible else None
    else:
        dot = candidates[0] if candidates else None

    if debug_paths:
        annotated = on_image.copy()
        for candidate in candidates:
            x = int(round(candidate["x"]))
            y = int(round(candidate["y"]))
            cw = int(round(candidate["w"]))
            ch = int(round(candidate["h"]))
            cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (255, 255, 0), 1)

        if dot is not None:
            x = int(round(dot["x"]))
            y = int(round(dot["y"]))
            dw = int(round(dot["w"]))
            dh = int(round(dot["h"]))
            cx = int(round(dot["cx"]))
            cy = int(round(dot["cy"]))
            cv2.rectangle(annotated, (x, y), (x + dw, y + dh), (0, 255, 255), 3)
            cv2.drawMarker(
                annotated,
                (cx, cy),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                markerSize=12,
                thickness=1,
            )
        else:
            cv2.putText(
                annotated,
                "NO DOT",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        annotated_path = debug_paths.get("annotated")
        if annotated_path:
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(annotated_path), annotated)

        mask_path = debug_paths.get("mask")
        if mask_path:
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), _mask)

        score_path = debug_paths.get("score")
        if score_path:
            score_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(score_path), np.clip(score * 8, 0, 255).astype(np.uint8))

    return {
        "scene": scene,
        "method": method,
        "dot": dot,
        "candidates": candidates,
        "image_width": int(w),
        "image_height": int(h),
    }
