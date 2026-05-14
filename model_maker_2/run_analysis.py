from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

try:
    from yolo_bbox_detector import find_latest_yolo_weights, letterbox_frame
except ModuleNotFoundError:
    from .yolo_bbox_detector import find_latest_yolo_weights, letterbox_frame


_DEFAULT_IMGSZ = 320


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000.0))
    return f"{millis // 1000:06d}_{millis % 1000:03d}"


def _class_name(names: Any, class_id: int) -> str:
    try:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        return str(names[class_id])
    except Exception:
        return str(class_id)


def _letterbox_meta(width: int, height: int, target: int) -> Tuple[float, int, int]:
    scale = min(float(target) / float(width), float(target) / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    left = (target - new_w) // 2
    top = (target - new_h) // 2
    return scale, left, top


def _clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    ix1 = max(0, min(width - 1, int(round(x1))))
    iy1 = max(0, min(height - 1, int(round(y1))))
    ix2 = max(0, min(width - 1, int(round(x2))))
    iy2 = max(0, min(height - 1, int(round(y2))))
    return ix1, iy1, ix2, iy2


def _detections_from_result(
    result: Any,
    names: Any,
    frame_width: int,
    frame_height: int,
    scale: float,
    left: int,
    top: int,
) -> List[Dict[str, Union[int, float, str]]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    try:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        clss = boxes.cls.detach().cpu().tolist()
    except Exception:
        return []

    detections: List[Dict[str, Union[int, float, str]]] = []
    for coords, score, cls in zip(xyxy, confs, clss):
        x1_lb, y1_lb, x2_lb, y2_lb = [float(v) for v in coords]
        x1 = (x1_lb - left) / scale
        y1 = (y1_lb - top) / scale
        x2 = (x2_lb - left) / scale
        y2 = (y2_lb - top) / scale
        ix1, iy1, ix2, iy2 = _clamp_box(x1, y1, x2, y2, frame_width, frame_height)
        class_id = int(cls)
        detections.append(
            {
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "confidence": float(score),
                "xmin": ix1,
                "ymin": iy1,
                "xmax": ix2,
                "ymax": iy2,
            }
        )
    return detections


def _annotate_frame(frame: Any, detections: Sequence[Dict[str, Union[int, float, str]]]) -> Any:
    annotated = frame.copy()
    for det in detections:
        x1 = int(det["xmin"])
        y1 = int(det["ymin"])
        x2 = int(det["xmax"])
        y2 = int(det["ymax"])
        class_name = str(det["class_name"])
        confidence = float(det["confidence"])
        label = f"{class_name}:{confidence:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = max(18, y1 - 6)
        cv2.putText(
            annotated,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def analyze_video(
    video: Union[str, Path],
    output_dir: Union[str, Path],
    weights: Optional[Union[str, Path]] = None,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = _DEFAULT_IMGSZ,
    max_det: int = 20,
    classes: Optional[Sequence[int]] = None,
    device: Optional[Union[int, str]] = None,
    jpeg_quality: int = 95,
    frame_stride: int = 1,
    limit: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[int, int, Path]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required to read and write video frames.")
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed. Install it to run inference.")

    video_path = Path(video).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"video must be a file: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise ValueError(f"expected an .mp4 file, got: {video_path}")

    frame_stride = max(1, int(frame_stride))
    jpeg_quality = min(max(int(jpeg_quality), 1), 100)
    resolved_weights = find_latest_yolo_weights(weights)

    out = Path(output_dir).expanduser()
    frames_dir = out / "frames"
    annotated_dir = out / "annotated"
    frames_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out / "detections.csv"

    model = YOLO(str(resolved_weights))
    names = getattr(model, "names", {})
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[WEIGHTS] {resolved_weights}")
    print(f"[VIDEO] {video_path} frames={total_frames} fps={fps:.3f}")

    processed = 0
    detected_frames = 0
    frame_index = -1

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "frame",
                "timestamp_sec",
                "frame_path",
                "annotated_frame",
                "class_id",
                "class_name",
                "confidence",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ],
        )
        writer.writeheader()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % frame_stride != 0:
                continue

            height, width = frame.shape[:2]
            seconds = (frame_index / fps) if fps > 0.0 else 0.0
            time_part = _format_seconds(seconds)
            frame_name = f"{video_path.stem}_frame{frame_index:08d}_t{time_part}.jpg"
            frame_path = frames_dir / frame_name
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            scale, left, top = _letterbox_meta(width, height, imgsz)
            model_frame = letterbox_frame(frame, target=imgsz)
            results = model.predict(
                source=model_frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                classes=list(classes) if classes is not None else None,
                agnostic_nms=False,
                device=device,
                verbose=False,
            )
            result = results[0] if results else None
            detections = (
                []
                if result is None
                else _detections_from_result(result, names, width, height, scale, left, top)
            )

            annotated_rel = ""
            if detections:
                annotated_path = annotated_dir / frame_name
                annotated = _annotate_frame(frame, detections)
                cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                annotated_rel = annotated_path.relative_to(out).as_posix()
                detected_frames += 1

            for det in detections:
                writer.writerow(
                    {
                        "video": video_path.as_posix(),
                        "frame": frame_index,
                        "timestamp_sec": f"{seconds:.3f}",
                        "frame_path": frame_path.relative_to(out).as_posix(),
                        "annotated_frame": annotated_rel,
                        "class_id": int(det["class_id"]),
                        "class_name": str(det["class_name"]),
                        "confidence": f"{float(det['confidence']):.6f}",
                        "xmin": int(det["xmin"]),
                        "ymin": int(det["ymin"]),
                        "xmax": int(det["xmax"]),
                        "ymax": int(det["ymax"]),
                    }
                )

            processed += 1
            if verbose and (detections or processed % 100 == 0):
                print(f"[FRAME] {frame_index} detections={len(detections)}")
            if limit is not None and processed >= limit:
                break

    cap.release()
    print(f"[DONE] processed_frames={processed} detected_frames={detected_frames} metadata={csv_path}")
    return processed, detected_frames, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split an MP4 into frames and run the latest generated YOLO model against each frame."
    )
    parser.add_argument("video", type=Path, help="MP4 file to analyze")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to analysis_output/<video-stem>",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Optional .pt weights. Defaults to latest runs/detect weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=_DEFAULT_IMGSZ, help="YOLO input size")
    parser.add_argument("--max_det", type=int, default=20, help="Maximum detections per frame")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Restrict to class ids, e.g. --classes 0")
    parser.add_argument("--device", type=str, default=None, help="Device id/name, e.g. cpu, mps, or 0")
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for saved frames")
    parser.add_argument("--frame_stride", type=int, default=1, help="Analyze every Nth frame")
    parser.add_argument("--limit", type=int, default=None, help="Stop after processing this many saved frames")
    parser.add_argument("--verbose", action="store_true", help="Print periodic frame progress")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("analysis_output") / args.video.stem

    analyze_video(
        video=args.video,
        output_dir=output_dir,
        weights=args.weights,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        classes=args.classes,
        device=args.device,
        jpeg_quality=args.jpeg_quality,
        frame_stride=args.frame_stride,
        limit=args.limit,
        verbose=args.verbose,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
