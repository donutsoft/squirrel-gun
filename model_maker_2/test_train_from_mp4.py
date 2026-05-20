from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None


VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_OUTPUT_DIR = Path("analysis_output") / "video_tests"
SCRIPT_DIR = Path(__file__).resolve().parent
LETTERBOX_PAD_COLOR = 114


@dataclass
class VideoTestResult:
    path: Path
    processed_frames: int
    total_frames: Optional[int]
    detection_count: int
    mined_training_images: int
    best_confidence: float
    best_frame: Optional[int]
    best_image: Optional[Path]
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.detection_count > 0


@dataclass
class BridgeCandidate:
    video_path: Path
    frame_index: int
    timestamp_sec: float
    image: Any
    detections: list[dict[str, int | float | str]]


@dataclass
class BBoxRow:
    image: str
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def iter_video_paths(paths: Sequence[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_dir():
            iterator = expanded.rglob("*") if recursive else expanded.iterdir()
            for child in sorted(iterator):
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                    yield child
        elif expanded.is_file() and expanded.suffix.lower() in VIDEO_EXTENSIONS:
            yield expanded
        else:
            yield expanded


def safe_stem(path: Path) -> str:
    keep = []
    for char in path.stem:
        keep.append(char if char.isalnum() or char in "._-" else "_")
    return "".join(keep).strip("_") or "video"


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    unique: list[Path] = []
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


def find_latest_yolo_weights(explicit: Optional[Path] = None) -> Path:
    roots = unique_paths([Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent])
    if explicit is not None:
        explicit_path = explicit.expanduser()
        candidates = [explicit_path] if explicit_path.is_absolute() else [root / explicit_path for root in roots]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        print(f"[WARN] Weights not found at {explicit_path}; searching runs/detect instead.", file=sys.stderr)

    matches: list[Path] = []
    for filename in ("best.pt", "last.pt"):
        for root in roots:
            matches.extend(root.glob(f"runs/detect/*/weights/{filename}"))
    matches = unique_paths(path for path in matches if path.is_file())
    if not matches:
        searched = ", ".join(str(root / "runs/detect/*/weights/{best,last}.pt") for root in roots)
        raise FileNotFoundError(f"no YOLO weights found. Searched: {searched}")

    def sort_key(path: Path) -> Tuple[int, float]:
        preferred = 1 if path.name == "best.pt" else 0
        return (preferred, path.stat().st_mtime)

    return max(matches, key=sort_key)


def letterbox_frame(frame: Any, target: int) -> Any:
    if cv2 is None or np is None:
        raise RuntimeError("opencv-python and numpy are required for letterboxing.")

    h, w = frame.shape[:2]
    scale = min(float(target) / float(w), float(target) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    if len(frame.shape) == 2 or frame.shape[2] == 1:
        canvas = np.full((target, target), LETTERBOX_PAD_COLOR, dtype=resized.dtype)
    else:
        canvas = np.full((target, target, frame.shape[2]), LETTERBOX_PAD_COLOR, dtype=resized.dtype)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def letterbox_meta(width: int, height: int, target: int) -> Tuple[float, int, int]:
    scale = min(float(target) / float(width), float(target) / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    left = (target - new_w) // 2
    top = (target - new_h) // 2
    return scale, left, top


def class_name(names: Any, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
    except Exception:
        return str(class_id)


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(0, min(width - 1, int(round(x2))))
    iy2 = max(0, min(height - 1, int(round(y2))))
    return ix1, iy1, ix2, iy2


def detections_from_result(
    result: Any,
    names: Any,
    frame_width: int,
    frame_height: int,
    scale: float,
    left: int,
    top: int,
) -> list[dict[str, int | float | str]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    try:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()
    except Exception:
        return []

    detections: list[dict[str, int | float | str]] = []
    for coords, score, cls in zip(xyxy, confs, clss):
        x1_lb, y1_lb, x2_lb, y2_lb = [float(v) for v in coords]
        x1 = (x1_lb - left) / scale
        y1 = (y1_lb - top) / scale
        x2 = (x2_lb - left) / scale
        y2 = (y2_lb - top) / scale
        ix1, iy1, ix2, iy2 = clamp_box(x1, y1, x2, y2, frame_width, frame_height)
        class_id = int(cls)
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name(names, class_id),
                "confidence": float(score),
                "xmin": ix1,
                "ymin": iy1,
                "xmax": ix2,
                "ymax": iy2,
            }
        )
    return detections


def letterbox_detections_from_result(
    result: Any,
    names: Any,
    img_size: int,
) -> list[dict[str, int | float | str]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    try:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()
    except Exception:
        return []

    detections: list[dict[str, int | float | str]] = []
    for coords, score, cls in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = [float(v) for v in coords]
        ix1, iy1, ix2, iy2 = clamp_box(x1, y1, x2, y2, img_size, img_size)
        class_id = int(cls)
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name(names, class_id),
                "confidence": float(score),
                "xmin": ix1,
                "ymin": iy1,
                "xmax": ix2,
                "ymax": iy2,
            }
        )
    return detections


def annotate_frame(frame: Any, detections: Sequence[dict[str, int | float | str]]) -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to annotate frames.")

    annotated = frame.copy()
    for det in detections:
        x1 = int(det["xmin"])
        y1 = int(det["ymin"])
        x2 = int(det["xmax"])
        y2 = int(det["ymax"])
        label = f"{det['class_name']}:{float(det['confidence']):.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def annotate_bbox_rows(frame: Any, rows: Sequence[BBoxRow]) -> Any:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to annotate frames.")

    annotated = frame.copy()
    for row in rows:
        cv2.rectangle(annotated, (row.xmin, row.ymin), (row.xmax, row.ymax), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            row.label,
            (row.xmin, max(18, row.ymin - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def read_bbox_rows(bbox_csv: Path) -> dict[str, list[BBoxRow]]:
    grouped: dict[str, list[BBoxRow]] = {}
    with bbox_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "label", "xmin", "ymin", "xmax", "ymax"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{bbox_csv} missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            image = str(row["image"]).strip()
            if not image:
                continue
            grouped.setdefault(image, []).append(
                BBoxRow(
                    image=image,
                    label=str(row["label"]).strip() or "object",
                    xmin=int(float(row["xmin"])),
                    ymin=int(float(row["ymin"])),
                    xmax=int(float(row["xmax"])),
                    ymax=int(float(row["ymax"])),
                )
            )
    return grouped


def display_training_candidates(
    *,
    training_dir: Path,
    bbox_csv: Optional[Path],
    start: int,
    scale: float,
    wait_ms: int,
    save_overlays: Optional[Path],
) -> int:
    if cv2 is None:
        print("ERROR opencv-python is required to display training candidates.", file=sys.stderr)
        return 2

    training_dir = training_dir.expanduser()
    bbox_path = bbox_csv.expanduser() if bbox_csv is not None else training_dir / "bboxes.csv"
    if not bbox_path.exists():
        print(f"ERROR bbox file not found: {bbox_path}", file=sys.stderr)
        return 2

    try:
        grouped = read_bbox_rows(bbox_path)
    except Exception as exc:
        print(f"ERROR could not read {bbox_path}: {exc}", file=sys.stderr)
        return 2

    images = sorted(grouped)
    if not images:
        print(f"No training candidates found in {bbox_path}.")
        return 0

    overlay_dir: Optional[Path] = None
    if save_overlays is not None:
        overlay_dir = save_overlays.expanduser()
        overlay_dir.mkdir(parents=True, exist_ok=True)

    index = max(0, min(int(start), len(images) - 1))
    window_name = "training candidates"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        image_rel = images[index]
        image_path = training_dir / image_rel
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read image: {image_path}", file=sys.stderr)
            annotated = None
        else:
            annotated = annotate_bbox_rows(image, grouped[image_rel])
            title = f"{index + 1}/{len(images)} {image_rel}"
            cv2.putText(
                annotated,
                title,
                (6, annotated.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            shown = annotated
            if scale > 0 and scale != 1.0:
                shown = cv2.resize(annotated, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow(window_name, shown)
            if overlay_dir is not None:
                out_path = overlay_dir / image_path.name
                cv2.imwrite(str(out_path), annotated)

        print(f"[{index + 1}/{len(images)}] {image_rel} boxes={len(grouped[image_rel])}")
        key = cv2.waitKey(max(0, int(wait_ms))) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("p"), ord("b"), 81):
            index = max(0, index - 1)
        elif key in (ord("n"), ord(" "), 83, 13, 10, 255):
            index += 1
            if index >= len(images):
                break
        elif annotated is None:
            index += 1
            if index >= len(images):
                break

    cv2.destroyAllWindows()
    return 0


def write_bridge_candidates(
    candidates: Sequence[BridgeCandidate],
    *,
    training_dir: Path,
    bbox_writer: csv.DictWriter,
    source_writer: csv.DictWriter,
    label: str,
    jpeg_quality: int,
) -> int:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to write training images.")

    images_dir = training_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for candidate in candidates:
        if not candidate.detections:
            continue

        best_score = max(float(det["confidence"]) for det in candidate.detections)
        image_name = (
            f"{safe_stem(candidate.video_path)}_frame{candidate.frame_index:08d}"
            f"_score{best_score:.3f}.jpg"
        )
        image_path = images_dir / image_name
        if not cv2.imwrite(str(image_path), candidate.image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]):
            continue

        image_rel = image_path.relative_to(training_dir).as_posix()
        for det in candidate.detections:
            bbox_writer.writerow(
                {
                    "image": image_rel,
                    "label": label,
                    "xmin": int(det["xmin"]),
                    "ymin": int(det["ymin"]),
                    "xmax": int(det["xmax"]),
                    "ymax": int(det["ymax"]),
                }
            )
            source_writer.writerow(
                {
                    "image": image_rel,
                    "source_video": candidate.video_path.as_posix(),
                    "frame": candidate.frame_index,
                    "timestamp_sec": f"{candidate.timestamp_sec:.3f}",
                    "class_id": int(det["class_id"]),
                    "class_name": str(det["class_name"]),
                    "confidence": f"{float(det['confidence']):.6f}",
                    "xmin": int(det["xmin"]),
                    "ymin": int(det["ymin"]),
                    "xmax": int(det["xmax"]),
                    "ymax": int(det["ymax"]),
                }
            )
        written += 1
    return written


def analyze_video(
    video_path: Path,
    model: Any,
    names: Any,
    *,
    output_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
    classes: Optional[Sequence[int]],
    device: Optional[str],
    frame_stride: int,
    limit_frames: Optional[int],
    save_best_image: bool,
    bridge_conf: float,
    max_gap_frames: int,
    training_dir: Path,
    training_label: str,
    training_jpeg_quality: int,
    detections_writer: csv.DictWriter,
    bbox_writer: csv.DictWriter,
    source_writer: csv.DictWriter,
    verbose: bool,
) -> VideoTestResult:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to read video files.")

    if not video_path.exists():
        return VideoTestResult(video_path, 0, None, 0, 0, 0.0, None, None, f"video not found: {video_path}")
    if not video_path.is_file():
        return VideoTestResult(video_path, 0, None, 0, 0, 0.0, None, None, f"not a file: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return VideoTestResult(video_path, 0, None, 0, 0, 0.0, None, None, f"not an MP4 file: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoTestResult(video_path, 0, None, 0, 0, 0.0, None, None, f"could not open video: {video_path}")

    frame_stride = max(1, int(frame_stride))
    max_gap_frames = max(1, int(max_gap_frames))
    predict_conf = min(float(conf), float(bridge_conf))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_frames = total_raw if total_raw > 0 else None

    processed = 0
    detection_count = 0
    mined_training_images = 0
    frame_index = -1
    best_confidence = 0.0
    best_frame: Optional[int] = None
    best_annotated: Any = None
    seen_left_detection = False
    gap_frames = 0
    gap_candidates: list[BridgeCandidate] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_index += 1
            if frame_index % frame_stride != 0:
                continue

            height, width = frame.shape[:2]
            scale, left, top = letterbox_meta(width, height, imgsz)
            model_frame = letterbox_frame(frame, target=imgsz)
            results = model.predict(
                source=model_frame,
                conf=predict_conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                classes=list(classes) if classes is not None else None,
                agnostic_nms=False,
                device=device,
                verbose=False,
            )
            result = results[0] if results else None
            original_detections = [] if result is None else detections_from_result(result, names, width, height, scale, left, top)
            letterbox_detections = [] if result is None else letterbox_detections_from_result(result, names, imgsz)
            detections = [det for det in original_detections if float(det["confidence"]) >= float(conf)]
            bridge_detections = [
                det for det in letterbox_detections
                if float(bridge_conf) <= float(det["confidence"]) < float(conf)
            ]

            processed += 1
            timestamp_sec = (frame_index / fps) if fps > 0.0 else 0.0

            if detections:
                if seen_left_detection and gap_candidates and gap_frames <= max_gap_frames:
                    mined_training_images += write_bridge_candidates(
                        gap_candidates,
                        training_dir=training_dir,
                        bbox_writer=bbox_writer,
                        source_writer=source_writer,
                        label=training_label,
                        jpeg_quality=training_jpeg_quality,
                    )
                seen_left_detection = True
                gap_frames = 0
                gap_candidates = []
            elif seen_left_detection:
                gap_frames += 1
                if gap_frames > max_gap_frames:
                    seen_left_detection = False
                    gap_frames = 0
                    gap_candidates = []
                elif bridge_detections:
                    gap_candidates.append(
                        BridgeCandidate(
                            video_path=video_path,
                            frame_index=frame_index,
                            timestamp_sec=timestamp_sec,
                            image=model_frame.copy(),
                            detections=bridge_detections,
                        )
                    )

            for det in detections:
                confidence = float(det["confidence"])
                detection_count += 1
                detections_writer.writerow(
                    {
                        "video": video_path.as_posix(),
                        "frame": frame_index,
                        "timestamp_sec": f"{timestamp_sec:.3f}",
                        "class_id": int(det["class_id"]),
                        "class_name": str(det["class_name"]),
                        "confidence": f"{confidence:.6f}",
                        "xmin": int(det["xmin"]),
                        "ymin": int(det["ymin"]),
                        "xmax": int(det["xmax"]),
                        "ymax": int(det["ymax"]),
                    }
                )
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_frame = frame_index
                    best_annotated = annotate_frame(frame, detections) if save_best_image else None

            if verbose and (detections or processed % 100 == 0):
                print(
                    f"[FRAME] {video_path} frame={frame_index} "
                    f"detections={len(detections)} bridge={len(bridge_detections)}"
                )

            if limit_frames is not None and processed >= max(0, int(limit_frames)):
                break
    except Exception as exc:
        return VideoTestResult(
            video_path,
            processed,
            total_frames,
            detection_count,
            mined_training_images,
            best_confidence,
            best_frame,
            None,
            str(exc),
        )
    finally:
        cap.release()

    best_image: Optional[Path] = None
    if best_annotated is not None and best_frame is not None:
        image_dir = output_dir / "best_frames"
        image_dir.mkdir(parents=True, exist_ok=True)
        best_image = image_dir / f"{safe_stem(video_path)}_frame{best_frame:08d}_score{best_confidence:.3f}.jpg"
        cv2.imwrite(str(best_image), best_annotated)

    return VideoTestResult(
        path=video_path,
        processed_frames=processed,
        total_frames=total_frames,
        detection_count=detection_count,
        mined_training_images=mined_training_images,
        best_confidence=best_confidence,
        best_frame=best_frame,
        best_image=best_image,
    )


def write_summary(output_dir: Path, results: Sequence[VideoTestResult]) -> Path:
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "passed",
                "processed_frames",
                "total_frames",
                "detection_count",
                "mined_training_images",
                "best_confidence",
                "best_frame",
                "best_image",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "video": result.path.as_posix(),
                    "passed": "yes" if result.passed else "no",
                    "processed_frames": result.processed_frames,
                    "total_frames": "" if result.total_frames is None else result.total_frames,
                    "detection_count": result.detection_count,
                    "mined_training_images": result.mined_training_images,
                    "best_confidence": f"{result.best_confidence:.6f}",
                    "best_frame": "" if result.best_frame is None else result.best_frame,
                    "best_image": "" if result.best_image is None else result.best_image.as_posix(),
                    "error": result.error or "",
                }
            )
    return summary_path


def print_result(result: VideoTestResult) -> None:
    total = "unknown" if result.total_frames is None else str(result.total_frames)
    if result.error:
        print(f"ERROR {result.path}: {result.error}")
    elif result.passed:
        image = "" if result.best_image is None else f" image={result.best_image}"
        print(
            f"PASS {result.path} detections={result.detection_count} "
            f"best={result.best_confidence:.3f} frame={result.best_frame} "
            f"processed={result.processed_frames}/{total} mined={result.mined_training_images}{image}"
        )
    else:
        print(
            f"FAIL {result.path} no detections "
            f"processed={result.processed_frames}/{total} mined={result.mined_training_images}"
        )


def parse_scan_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the latest trained YOLO model over MP4 files that should contain squirrels."
    )
    parser.add_argument("videos", nargs="+", type=Path, help="MP4 file(s) or directories of MP4 files")
    parser.add_argument("--weights", type=Path, default=None, help="Optional .pt weights. Defaults to latest runs/detect weights.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for summary and detection CSVs")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directories")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument(
        "--bridge-conf",
        type=float,
        default=0.05,
        help="Lower confidence threshold used to mine missing detections between confident detections",
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=30,
        help="Maximum no-detection gap length that can produce bridge training images",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=None,
        help="Output directory for mined letterboxed training images and bboxes.csv. Defaults to OUTPUT_DIR/training_candidates",
    )
    parser.add_argument("--training-label", default="squirrel", help="Label written to mined bboxes.csv")
    parser.add_argument("--training-jpeg-quality", type=int, default=95, help="JPEG quality for mined training images")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="YOLO input size")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections per frame")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Restrict to class ids, e.g. --classes 0")
    parser.add_argument("--device", type=str, default=None, help="Device id/name, e.g. cpu, mps, or 0")
    parser.add_argument("--frame-stride", type=int, default=1, help="Analyze every Nth frame")
    parser.add_argument("--limit-frames", type=int, default=None, help="Stop each video after processing this many sampled frames")
    parser.add_argument("--no-save-best-images", action="store_true", help="Do not save annotated best frames")
    parser.add_argument("--verbose", action="store_true", help="Print periodic frame progress")
    args = parser.parse_args(argv)
    args.command = "scan"
    return args


def parse_display_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="test_videos.py display",
        description="Display mined training candidate images with bboxes overlaid.",
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "training_candidates",
        help="Directory containing images/ and bboxes.csv",
    )
    parser.add_argument(
        "--bboxes",
        type=Path,
        default=None,
        help="Optional bbox CSV. Defaults to TRAINING_DIR/bboxes.csv",
    )
    parser.add_argument("--start", type=int, default=0, help="Zero-based image index to start at")
    parser.add_argument("--scale", type=float, default=2.0, help="Display scale factor")
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=0,
        help="Milliseconds to wait per image. 0 waits for a keypress.",
    )
    parser.add_argument(
        "--save-overlays",
        type=Path,
        default=None,
        help="Optional directory to also write annotated overlay images",
    )
    args = parser.parse_args(argv)
    args.command = "display"
    return args


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    items = list(sys.argv[1:] if argv is None else argv)
    if items and items[0] == "display":
        return parse_display_args(items[1:])
    return parse_scan_args(items)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if getattr(args, "command", "scan") == "display":
        return display_training_candidates(
            training_dir=args.training_dir,
            bbox_csv=args.bboxes,
            start=int(args.start),
            scale=float(args.scale),
            wait_ms=int(args.wait_ms),
            save_overlays=args.save_overlays,
        )

    if YOLO is None:
        print("ERROR ultralytics is not installed. Install it to run inference.", file=sys.stderr)
        return 2
    if cv2 is None:
        print("ERROR opencv-python is required to read video files.", file=sys.stderr)
        return 2
    if np is None:
        print("ERROR numpy is required to prepare video frames.", file=sys.stderr)
        return 2

    videos = list(iter_video_paths(args.videos, recursive=bool(args.recursive)))
    if not videos:
        print("ERROR no MP4 files found.", file=sys.stderr)
        return 2
    if float(args.bridge_conf) > float(args.conf):
        print("ERROR --bridge-conf must be less than or equal to --conf.", file=sys.stderr)
        return 2

    weights = find_latest_yolo_weights(args.weights)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_csv = output_dir / "detections.csv"
    training_dir = args.training_dir.expanduser() if args.training_dir is not None else output_dir / "training_candidates"
    training_dir.mkdir(parents=True, exist_ok=True)
    bbox_csv = training_dir / "bboxes.csv"
    source_csv = training_dir / "sources.csv"

    print(f"[WEIGHTS] {weights}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[TRAINING] {training_dir}")

    model = YOLO(str(weights))
    names = getattr(model, "names", {})
    results: list[VideoTestResult] = []
    with (
        detections_csv.open("w", newline="") as f,
        bbox_csv.open("w", newline="") as bbox_f,
        source_csv.open("w", newline="") as source_f,
    ):
        detections_writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frame",
                "timestamp_sec",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        detections_writer.writeheader()
        bbox_writer = csv.DictWriter(
            bbox_f,
            fieldnames=["image", "label", "xmin", "ymin", "xmax", "ymax"],
        )
        bbox_writer.writeheader()
        source_writer = csv.DictWriter(
            source_f,
            fieldnames=[
                "image",
                "source_video",
                "frame",
                "timestamp_sec",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        source_writer.writeheader()

        for video in videos:
            result = analyze_video(
                video,
                model,
                names,
                output_dir=output_dir,
                conf=float(args.conf),
                iou=float(args.iou),
                imgsz=int(args.imgsz),
                max_det=int(args.max_det),
                classes=args.classes,
                device=args.device,
                frame_stride=int(args.frame_stride),
                limit_frames=args.limit_frames,
                save_best_image=not bool(args.no_save_best_images),
                bridge_conf=float(args.bridge_conf),
                max_gap_frames=int(args.max_gap_frames),
                training_dir=training_dir,
                training_label=str(args.training_label),
                training_jpeg_quality=min(max(int(args.training_jpeg_quality), 1), 100),
                detections_writer=detections_writer,
                bbox_writer=bbox_writer,
                source_writer=source_writer,
                verbose=bool(args.verbose),
            )
            results.append(result)
            print_result(result)

    summary_csv = write_summary(output_dir, results)
    passed = sum(1 for result in results if result.passed)
    mined = sum(result.mined_training_images for result in results)
    print(f"Summary: {passed}/{len(results)} video(s) detected squirrels.")
    print(f"Mined training images: {mined}")
    print(f"CSV: {summary_csv}")
    print(f"Detections: {detections_csv}")
    print(f"Training bboxes: {bbox_csv}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
