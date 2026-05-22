#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SCRIPT_DIR.parent
_DEFAULT_POSITIVES_DIR = _REPO_DIR / "dataset" / "positives"
_DEFAULT_BBOX_FILE = _REPO_DIR / "dataset" / "bboxes.txt"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_REQUIRED_FIELDS = ["image", "label", "xmin", "ymin", "xmax", "ymax"]


Row = Dict[str, str]
Box = Tuple[int, int, int, int]
DisplayMode = str


def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    unique: List[Path] = []
    seen = set()
    for path in paths:
        try:
            key = path.resolve()
        except Exception:
            key = path.absolute()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def find_latest_yolo_weights(explicit: Optional[Union[str, Path]] = None) -> Path:
    roots = _unique_paths([Path.cwd(), _SCRIPT_DIR, _REPO_DIR])

    if explicit is not None:
        explicit_path = Path(explicit).expanduser()
        candidates = [explicit_path] if explicit_path.is_absolute() else [root / explicit_path for root in roots]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        print(f"[WARN] Weights not found at {explicit_path}; searching runs/detect instead.", file=sys.stderr)

    matches: List[Path] = []
    for filename in ("best.pt", "last.pt"):
        for root in roots:
            matches.extend(root.glob(f"runs/detect/*/weights/{filename}"))

    matches = _unique_paths([path for path in matches if path.is_file()])
    if not matches:
        searched = ", ".join(str(root / "runs/detect/*/weights/{best,last}.pt") for root in roots)
        raise FileNotFoundError(f"no YOLO weights found. Searched: {searched}")

    def sort_key(path: Path) -> Tuple[int, float]:
        preferred = 1 if path.name == "best.pt" else 0
        return (preferred, path.stat().st_mtime)

    return max(matches, key=sort_key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Review existing positive-image bounding boxes against YOLO detections. "
            "Drawn boxes are red; detected boxes are green. Press Y to replace the "
            "CSV box with the detected box."
        )
    )
    parser.add_argument(
        "--positives-dir",
        type=Path,
        default=_DEFAULT_POSITIVES_DIR,
        help="Directory containing positive images. Defaults to ../dataset/positives.",
    )
    parser.add_argument(
        "--bbox-file",
        type=Path,
        default=_DEFAULT_BBOX_FILE,
        help="CSV bbox file. Defaults to ../dataset/bboxes.txt.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Optional .pt weights. Defaults to latest runs/detect weights.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO input size.")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections per image.")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Restrict to class ids, e.g. --classes 0.")
    parser.add_argument("--device", type=str, default=None, help="Device id/name, e.g. cpu, mps, or 0.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after reviewing this many positive images.",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        help="Start at this image filename or relative path within positives-dir.",
    )
    parser.add_argument(
        "--backup-suffix",
        default=".bak",
        help="Suffix for one-time bbox CSV backup when the first accepted change is written.",
    )
    return parser.parse_args()


def _load_rows(bbox_file: Path) -> Tuple[List[str], List[Row]]:
    with bbox_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        missing = [field for field in _REQUIRED_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(f"{bbox_file} is missing required columns: {', '.join(missing)}")
        return fieldnames, list(reader)


def _write_rows(bbox_file: Path, fieldnames: Sequence[str], rows: Sequence[Row]) -> None:
    tmp_path = bbox_file.with_suffix(bbox_file.suffix + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, bbox_file)


def _remove_rows(rows: Sequence[Row], row_indexes: Sequence[int]) -> List[Row]:
    remove = set(row_indexes)
    return [row for i, row in enumerate(rows) if i not in remove]


def _index_rows(rows: Sequence[Row]) -> Dict[str, List[int]]:
    by_image: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        image = (row.get("image") or "").strip()
        if image:
            by_image.setdefault(image, []).append(i)
    return by_image


def _positive_images(positives_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in positives_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTS
    )


def _box_from_row(row: Row) -> Box:
    return (
        int(float(row["xmin"])),
        int(float(row["ymin"])),
        int(float(row["xmax"])),
        int(float(row["ymax"])),
    )


def _box_area(box: Box) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _clamp_box(box: Sequence[Union[int, float]], width: int, height: int) -> Box:
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    return xa, ya, xb, yb


def _best_detection(result: Any, width: int, height: int) -> Optional[Tuple[Box, float]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return None
    try:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
    except Exception:
        return None
    if not xyxy:
        return None

    candidates: List[Tuple[Box, float]] = []
    for coords, conf in zip(xyxy, confs):
        box = _clamp_box(coords, width, height)
        if _box_area(box) > 0:
            candidates.append((box, float(conf)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def _draw_box(image: Any, box: Box, color: Tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text_y = max(18, y1 - 6)
    cv2.putText(image, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _fit_for_display(image: Any, max_width: int = 1400, max_height: int = 900) -> Any:
    height, width = image.shape[:2]
    scale = min(float(max_width) / float(width), float(max_height) / float(height), 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _overlay_status(
    image: Any,
    rel_image: str,
    index: int,
    total: int,
    detected: Optional[Tuple[Box, float]],
    original_count: int,
    mode: DisplayMode,
) -> None:
    lines = [
        f"{index}/{total} {rel_image}",
        f"View {mode.upper()}  W view  Y accept  D delete  N/Enter skip  Esc quit",
    ]
    if detected is None:
        lines.append("No detection above threshold")
    else:
        lines.append(f"Detection confidence {detected[1]:.3f}")
    if original_count > 1:
        lines.append(f"{original_count} CSV boxes; Y updates the first row")

    y = 22
    for i, line in enumerate(lines):
        scale = 0.45 if i in (0, 1) else 0.55
        outline = 3
        thickness = 1 if i in (0, 1) else 2
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += 20 if i in (0, 1) else 24


def _render_display(
    image: Any,
    original_boxes: Sequence[Box],
    detected: Optional[Tuple[Box, float]],
    rel_image: str,
    index: int,
    total: int,
    mode: DisplayMode,
) -> Any:
    display = image.copy()

    if mode in ("red", "both"):
        for original_box in original_boxes:
            _draw_box(display, original_box, (0, 0, 255), "drawn")

    if mode in ("green", "both") and detected is not None:
        _draw_box(display, detected[0], (0, 255, 0), f"detected {detected[1]:.2f}")

    _overlay_status(display, rel_image, index, total, detected, len(original_boxes), mode)
    return _fit_for_display(display)


def _update_row(row: Row, box: Box) -> None:
    x1, y1, x2, y2 = box
    row["xmin"] = str(x1)
    row["ymin"] = str(y1)
    row["xmax"] = str(x2)
    row["ymax"] = str(y2)


def _maybe_skip_to_start(images: List[Path], positives_dir: Path, start_at: Optional[str]) -> List[Path]:
    if not start_at:
        return images
    normalized = start_at.strip()
    for i, path in enumerate(images):
        rel = path.relative_to(positives_dir).as_posix()
        if rel == normalized or path.name == normalized:
            return images[i:]
    raise ValueError(f"--start-at image not found under {positives_dir}: {start_at}")


def main() -> int:
    args = parse_args()

    if cv2 is None:
        raise RuntimeError("opencv-python is required for interactive image display.")
    if YOLO is None:
        raise RuntimeError("ultralytics is required to run YOLO inference.")

    positives_dir = args.positives_dir.expanduser()
    bbox_file = args.bbox_file.expanduser()
    if not positives_dir.exists():
        raise FileNotFoundError(f"positives directory not found: {positives_dir}")
    if not bbox_file.exists():
        raise FileNotFoundError(f"bbox file not found: {bbox_file}")

    fieldnames, rows = _load_rows(bbox_file)
    rows_by_image = _index_rows(rows)
    images = _maybe_skip_to_start(_positive_images(positives_dir), positives_dir, args.start_at)
    if args.limit is not None:
        images = images[: max(0, int(args.limit))]
    if not images:
        print(f"No positive images found under {positives_dir}", file=sys.stderr)
        return 2

    weights = find_latest_yolo_weights(args.weights)
    model = YOLO(str(weights))
    print(f"[WEIGHTS] {weights}")
    print(f"[BBOX] {bbox_file}")
    print(f"[POSITIVES] {positives_dir} images={len(images)}")

    backup_written = False
    accepted = 0
    deleted = 0
    skipped_no_row = 0
    skipped_no_detection = 0
    mode: DisplayMode = "both"
    modes: List[DisplayMode] = ["red", "green", "both"]
    window_name = "tighten_bounding_boxes"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        should_quit = False
        for offset, image_path in enumerate(images, start=1):
            if should_quit:
                break
            rel_image = image_path.relative_to(positives_dir).as_posix()
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[SKIP] unreadable image: {image_path}", file=sys.stderr)
                continue

            height, width = image.shape[:2]
            predict_kwargs: Dict[str, Any] = {
                "source": str(image_path),
                "conf": args.conf,
                "iou": args.iou,
                "imgsz": args.imgsz,
                "max_det": args.max_det,
                "classes": args.classes,
                "agnostic_nms": False,
                "verbose": False,
            }
            if args.device is not None:
                predict_kwargs["device"] = args.device
            results = model.predict(**predict_kwargs)
            detected = _best_detection(results[0], width, height) if results else None

            row_indexes = rows_by_image.get(rel_image, [])
            original_boxes: List[Box] = []
            for row_index in row_indexes:
                try:
                    original_boxes.append(_box_from_row(rows[row_index]))
                except Exception as exc:
                    print(f"[WARN] invalid bbox row for {rel_image}: {exc}", file=sys.stderr)

            while True:
                display = _render_display(
                    image,
                    original_boxes,
                    detected,
                    rel_image,
                    offset,
                    len(images),
                    mode,
                )
                cv2.imshow(window_name, display)

                key = cv2.waitKey(0) & 0xFF
                if key == 27:
                    print("[QUIT]")
                    should_quit = True
                    break
                if key in (ord("w"), ord("W")):
                    mode = modes[(modes.index(mode) + 1) % len(modes)]
                    continue
                break

            if should_quit:
                break
            if key in (ord("d"), ord("D")):
                if not backup_written:
                    backup_path = Path(str(bbox_file) + args.backup_suffix)
                    shutil.copy2(bbox_file, backup_path)
                    backup_written = True
                    print(f"[BACKUP] {backup_path}")

                rows = _remove_rows(rows, row_indexes)
                _write_rows(bbox_file, fieldnames, rows)
                rows_by_image = _index_rows(rows)
                image_path.unlink()
                deleted += 1
                print(f"[DELETE] {rel_image}: removed {len(row_indexes)} bbox row(s) and deleted image")
                continue
            if key not in (ord("y"), ord("Y")):
                continue
            if detected is None:
                skipped_no_detection += 1
                print(f"[SKIP] {rel_image}: no detection to accept")
                continue
            if not row_indexes:
                skipped_no_row += 1
                print(f"[SKIP] {rel_image}: no row in bbox CSV to replace")
                continue

            if not backup_written:
                backup_path = Path(str(bbox_file) + args.backup_suffix)
                shutil.copy2(bbox_file, backup_path)
                backup_written = True
                print(f"[BACKUP] {backup_path}")

            _update_row(rows[row_indexes[0]], detected[0])
            _write_rows(bbox_file, fieldnames, rows)
            accepted += 1
            print(f"[UPDATE] {rel_image}: {detected[0]} conf={detected[1]:.3f}")
    finally:
        cv2.destroyAllWindows()

    print(
        "[DONE] "
        f"accepted={accepted} deleted={deleted} "
        f"skipped_no_detection={skipped_no_detection} skipped_no_csv_row={skipped_no_row}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
