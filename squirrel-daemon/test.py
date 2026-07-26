#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple


VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv"}


@dataclass
class Detection:
    frame_index: int
    timestamp_sec: float
    score: float
    rect: Optional[Tuple[int, int, int, int]]
    class_id: Optional[int]
    image_path: Optional[Path] = None
    annotated_image_path: Optional[Path] = None


@dataclass
class VideoResult:
    path: Path
    processed_frames: int
    total_frames: Optional[int]
    detections: list[Detection]
    error: Optional[str] = None

    @property
    def detected(self) -> bool:
        return bool(self.detections)


def iter_video_paths(paths: Sequence[Path], recursive: bool) -> Iterable[Path]:
    for path in paths:
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            for child in sorted(iterator):
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                    yield child
        elif path.is_file():
            yield path
        else:
            yield path


def event_to_detection(
    event: object,
    frame_index: int,
    timestamp_sec: float,
    *,
    image_path: Optional[Path] = None,
    annotated_image_path: Optional[Path] = None,
) -> Detection:
    extra = getattr(event, "extra", None) or {}
    score = float(extra.get("score", 0.0))
    class_id_raw = extra.get("class")
    class_id = int(class_id_raw) if class_id_raw is not None else None
    return Detection(
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
        score=score,
        rect=getattr(event, "rect", None),
        class_id=class_id,
        image_path=image_path,
        annotated_image_path=annotated_image_path,
    )


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_") or "video"


def annotate_detection_frame(frame: Any, detection: Detection) -> Any:
    import cv2  # type: ignore

    annotated = frame.copy()
    if detection.rect is not None:
        x, y, width, height = detection.rect
        x2 = x + width
        y2 = y + height
        cv2.rectangle(annotated, (x, y), (x2, y2), (0, 255, 0), 2)
        label = f"{detection.class_id if detection.class_id is not None else '?'}:{detection.score:.2f}"
        cv2.putText(
            annotated,
            label,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


def read_video_frame(path: Path, frame_index: int) -> Any:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not re-open video: {path}")

    try:
        if cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame

        cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"could not re-open video: {path}")

        for _ in range(frame_index + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"could not read frame {frame_index} from {path}")
        return frame
    finally:
        cap.release()


def save_detection_images(
    output_dir: Path,
    video_path: Path,
    detection: Detection,
) -> Tuple[Path, Path]:
    import cv2  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{safe_stem(video_path)}_frame{detection.frame_index:08d}_score{detection.score:.3f}"
    image_path = output_dir / f"{prefix}.jpg"
    annotated_image_path = output_dir / f"{prefix}_bbox.jpg"
    frame = read_video_frame(video_path, detection.frame_index)
    annotated_frame = annotate_detection_frame(frame, detection)

    if not cv2.imwrite(str(image_path), frame):
        raise RuntimeError(f"could not write image: {image_path}")
    if not cv2.imwrite(str(annotated_image_path), annotated_frame):
        raise RuntimeError(f"could not write annotated image: {annotated_image_path}")

    return image_path, annotated_image_path


def analyze_video(
    path: Path,
    detector: Any,
    *,
    every_n_frames: int,
    max_hits: int,
    output_dir: Path,
    verbose: bool,
) -> VideoResult:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return VideoResult(path=path, processed_frames=0, total_frames=None, detections=[], error="could not open video")

    total_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = total_raw if total_raw > 0 else None
    detections: list[Detection] = []
    best_detection_index: Optional[int] = None
    processed = 0
    frame_index = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_index += 1
            if frame_index % every_n_frames != 0:
                continue

            processed += 1
            timestamp_sec = (frame_index / fps) if fps > 0 else 0.0
            try:
                result = detector.process(frame, now_ts=time.time())
            except RuntimeError as exc:
                return VideoResult(
                    path=path,
                    processed_frames=processed,
                    total_frames=total_frames,
                    detections=detections,
                    error=str(exc),
                )
            if result.events:
                for event in result.events:
                    detection = event_to_detection(event, frame_index, timestamp_sec)
                    detections.append(detection)
                    if best_detection_index is None or detection.score > detections[best_detection_index].score:
                        best_detection_index = len(detections) - 1

                if verbose:
                    frame_detections = (
                        event_to_detection(event, frame_index, timestamp_sec)
                        for event in result.events
                    )
                    best = max(frame_detections, key=lambda item: item.score)
                    print(
                        f"[hit] {path} frame={frame_index} "
                        f"t={timestamp_sec:.2f}s score={best.score:.3f}"
                    )

                if max_hits > 0 and len(detections) >= max_hits:
                    break
    finally:
        cap.release()

    if best_detection_index is not None:
        try:
            image_path, annotated_image_path = save_detection_images(
                output_dir,
                path,
                detections[best_detection_index],
            )
        except RuntimeError as exc:
            return VideoResult(
                path=path,
                processed_frames=processed,
                total_frames=total_frames,
                detections=detections,
                error=str(exc),
            )
        detections[best_detection_index].image_path = image_path
        detections[best_detection_index].annotated_image_path = annotated_image_path

    return VideoResult(
        path=path,
        processed_frames=processed,
        total_frames=total_frames,
        detections=detections,
    )


def print_result(result: VideoResult) -> None:
    if result.error:
        print(f"ERROR {result.path}: {result.error}")
        return

    total = "unknown" if result.total_frames is None else str(result.total_frames)
    if not result.detected:
        print(f"NO SQUIRREL {result.path} processed={result.processed_frames}/{total}")
        return

    best = max(result.detections, key=lambda item: item.score)
    rect = "" if best.rect is None else f" rect={best.rect}"
    images = ""
    if best.image_path and best.annotated_image_path:
        images = f" image={best.image_path} bbox={best.annotated_image_path}"
    print(
        f"SQUIRREL {result.path} hits={len(result.detections)} "
        f"best_score={best.score:.3f} frame={best.frame_index} "
        f"t={best.timestamp_sec:.2f}s processed={result.processed_frames}/{total}{rect}{images}"
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the daemon's YOLO Edge TPU squirrel detector over MP4 recordings."
    )
    parser.add_argument("videos", nargs="+", type=Path, help="Video file(s) or directories to scan")
    parser.add_argument(
        "--model",
        default="best_full_integer_quant_edgetpu.tflite",
        help="Model filename under squirrel-daemon, or an absolute/relative path",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum detection confidence")
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Analyze every Nth frame.",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=0,
        help="Stop each video after this many detections. Defaults to 0, which scans the whole video and saves the best hit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("detections"),
        help="Directory for extracted detection images.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan directories")
    parser.add_argument("--verbose", action="store_true", help="Print detections as they happen")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    every_n_frames = max(1, int(args.every))
    max_hits = max(0, int(args.max_hits))

    videos = list(iter_video_paths(args.videos, recursive=bool(args.recursive)))
    if not videos:
        print("No video files found.", file=sys.stderr)
        return 2

    from event_detection.yolo import YOLOEventDetector

    detector = YOLOEventDetector(model_filename=str(args.model))
    detector.configure(score_thresh=float(args.conf))

    had_error = False
    detected_count = 0
    for video in videos:
        result = analyze_video(
            video,
            detector,
            every_n_frames=every_n_frames,
            max_hits=max_hits,
            output_dir=args.output_dir,
            verbose=bool(args.verbose),
        )
        print_result(result)
        had_error = had_error or result.error is not None
        detected_count += 1 if result.detected else 0

    print(f"Summary: {detected_count}/{len(videos)} video(s) had squirrel detections.")
    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
