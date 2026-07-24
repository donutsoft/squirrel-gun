#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_THRESHOLDS = (0.4, 0.5, 0.6, 0.7, 0.75, 0.8)
VALID_LABELS = ("true_positive", "false_positive")
TPU_TUNING_NOTE = (
    "Edge-TPU confidence values are not interchangeable with desktop .pt-model "
    "confidence values. Validate every threshold adjustment on the deployed TPU."
)


@dataclass
class RecordingResult:
    recording: str
    label: str
    source: str
    frames_processed: int
    detection_count: int
    best_score: Optional[float]
    best_frame: Optional[int]
    best_image: Optional[str]
    stored_best_score: Optional[float]
    error: Optional[str] = None

    @property
    def scorable(self) -> bool:
        return self.error is None and self.best_score is not None


def validate_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"{name} must be greater than 0 and at most 1")
    return value


def load_manifest(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text())
    recordings = data.get("recordings")
    if not isinstance(recordings, dict):
        raise ValueError(f"manifest has no recordings object: {path}")

    selected: Dict[str, Dict[str, Any]] = {}
    for name, record in recordings.items():
        if not isinstance(record, dict):
            continue
        label = str(record.get("label", ""))
        if label in VALID_LABELS:
            selected[str(name)] = record
    return selected


def stored_best_score(record: Dict[str, Any]) -> Optional[float]:
    scores: List[float] = []
    for item in record.get("generated", []):
        if not isinstance(item, dict) or item.get("score") is None:
            continue
        try:
            scores.append(float(item["score"]))
        except (TypeError, ValueError):
            continue
    return max(scores) if scores else None


def generated_negative_paths(record: Dict[str, Any], data_root: Path) -> List[Path]:
    paths: List[Path] = []
    for item in record.get("generated", []):
        if not isinstance(item, dict) or item.get("kind") != "negatives":
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            paths.append(data_root / "negatives" / name)
    return paths


def best_detection(detections: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    candidates = list(detections)
    if not candidates:
        return None
    return max(candidates, key=lambda item: float(item.get("score", 0.0)))


def analyze_video(
    path: Path,
    label: str,
    detector: Any,
    *,
    cv2_module: Any,
    mining_threshold: float,
    every_n_frames: int,
    stored_score: Optional[float],
) -> RecordingResult:
    cap = cv2_module.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return RecordingResult(
            recording=path.name,
            label=label,
            source="video",
            frames_processed=0,
            detection_count=0,
            best_score=None,
            best_frame=None,
            best_image=None,
            stored_best_score=stored_score,
            error="could not open video",
        )

    processed = 0
    detection_count = 0
    frame_index = -1
    top_score: Optional[float] = None
    top_frame: Optional[int] = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if frame_index % every_n_frames != 0:
                continue

            processed += 1
            _letterboxed, detections = detector.predict_candidates(
                frame,
                score_thresh=mining_threshold,
            )
            detection_count += len(detections)
            candidate = best_detection(detections)
            if candidate is None:
                continue
            score = float(candidate["score"])
            if top_score is None or score > top_score:
                top_score = score
                top_frame = frame_index
    finally:
        cap.release()

    return RecordingResult(
        recording=path.name,
        label=label,
        source="video",
        frames_processed=processed,
        detection_count=detection_count,
        best_score=top_score if top_score is not None else 0.0,
        best_frame=top_frame,
        best_image=None,
        stored_best_score=stored_score,
    )


def analyze_images(
    recording: str,
    label: str,
    paths: Sequence[Path],
    detector: Any,
    *,
    cv2_module: Any,
    mining_threshold: float,
    stored_score: Optional[float],
) -> RecordingResult:
    processed = 0
    detection_count = 0
    missing: List[str] = []
    top_score: Optional[float] = None
    top_image: Optional[str] = None

    for path in paths:
        image = cv2_module.imread(str(path))
        if image is None:
            missing.append(path.name)
            continue
        processed += 1
        _letterboxed, detections = detector.predict_candidates(
            image,
            score_thresh=mining_threshold,
        )
        detection_count += len(detections)
        candidate = best_detection(detections)
        if candidate is None:
            continue
        score = float(candidate["score"])
        if top_score is None or score > top_score:
            top_score = score
            top_image = path.name

    error = None
    if processed == 0:
        error = "no generated negative frames could be read"
    elif missing:
        error = f"missing {len(missing)} generated negative frame(s)"

    return RecordingResult(
        recording=recording,
        label=label,
        source="generated_frames",
        frames_processed=processed,
        detection_count=detection_count,
        best_score=(top_score if top_score is not None else 0.0) if error is None else None,
        best_frame=None,
        best_image=top_image,
        stored_best_score=stored_score,
        error=error,
    )


def missing_result(
    recording: str,
    label: str,
    stored_score: Optional[float],
) -> RecordingResult:
    return RecordingResult(
        recording=recording,
        label=label,
        source="missing",
        frames_processed=0,
        detection_count=0,
        best_score=None,
        best_frame=None,
        best_image=None,
        stored_best_score=stored_score,
        error="recording is missing and no usable frame fallback is available",
    )


def threshold_summary(
    results: Sequence[RecordingResult],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    true_positives = [
        result for result in results
        if result.label == "true_positive" and result.scorable
    ]
    false_positives = [
        result for result in results
        if result.label == "false_positive" and result.scorable
    ]

    summary: List[Dict[str, Any]] = []
    for threshold in thresholds:
        tp_detected = sum(
            1 for result in true_positives
            if float(result.best_score or 0.0) >= threshold
        )
        fp_triggered = sum(
            1 for result in false_positives
            if float(result.best_score or 0.0) >= threshold
        )
        tp_total = len(true_positives)
        fp_total = len(false_positives)
        recall = (tp_detected / tp_total) if tp_total else None
        false_positive_rate = (fp_triggered / fp_total) if fp_total else None
        specificity = (1.0 - false_positive_rate) if false_positive_rate is not None else None
        balanced_accuracy = (
            (recall + specificity) / 2.0
            if recall is not None and specificity is not None
            else None
        )
        summary.append({
            "threshold": threshold,
            "true_positives_detected": tp_detected,
            "true_positives_total": tp_total,
            "recall": recall,
            "false_positives_triggered": fp_triggered,
            "false_positives_total": fp_total,
            "false_positive_rate": false_positive_rate,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
        })
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_recordings_csv(path: Path, results: Sequence[RecordingResult]) -> None:
    fields = (
        "recording",
        "label",
        "source",
        "frames_processed",
        "detection_count",
        "best_score",
        "best_frame",
        "best_image",
        "stored_best_score",
        "score_delta",
        "error",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            score_delta = None
            if result.best_score is not None and result.stored_best_score is not None:
                score_delta = result.best_score - result.stored_best_score
            writer.writerow({
                **asdict(result),
                "score_delta": score_delta,
            })


def format_ratio(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100.0:5.1f}%"


def print_summary(summary: Sequence[Dict[str, Any]]) -> None:
    print()
    print("Threshold  TP recall       FP triggered    FP rate  Balanced")
    for row in summary:
        print(
            f"{float(row['threshold']):8.3f}  "
            f"{int(row['true_positives_detected']):2d}/{int(row['true_positives_total']):<2d} "
            f"{format_ratio(row['recall']):>7s}    "
            f"{int(row['false_positives_triggered']):2d}/{int(row['false_positives_total']):<2d} "
            f"{format_ratio(row['false_positive_rate']):>7s}  "
            f"{format_ratio(row['balanced_accuracy']):>7s}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    daemon_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Replay labeled squirrel recordings through the deployed Edge TPU model "
            "and calculate event-level confidence-threshold metrics. "
            + TPU_TUNING_NOTE
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=daemon_root.parent / "squirrel-training-data" / "recording_labels.json",
        help="Recording-label manifest.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=daemon_root.parent / "squirrel-training-data",
        help="Directory containing positives/ and negatives/.",
    )
    parser.add_argument(
        "--recordings-dir",
        type=Path,
        default=daemon_root / "static" / "recordings",
        help="Directory containing labeled recordings.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=daemon_root / "best_full_integer_quant_edgetpu.tflite",
        help="Edge-TPU TFLite model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("threshold_evaluation"),
        help="Directory for recordings.csv and report.json.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Live confidence thresholds to evaluate.",
    )
    parser.add_argument(
        "--mining-threshold",
        type=float,
        default=0.001,
        help="Low inference threshold used for the single replay pass.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Evaluate every Nth video frame.",
    )
    parser.add_argument(
        "--no-frame-fallback",
        action="store_true",
        help="Do not evaluate saved negative frames when a false-positive video is missing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each recording result.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        mining_threshold = validate_probability(args.mining_threshold, "mining threshold")
        thresholds = sorted({
            validate_probability(value, "threshold")
            for value in args.thresholds
        })
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    manifest_path = args.manifest.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    recordings_dir = args.recordings_dir.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    every_n_frames = max(1, int(args.every))

    try:
        records = load_manifest(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not load manifest: {exc}", file=sys.stderr)
        return 2
    if not records:
        print("ERROR: manifest contains no true_positive or false_positive labels", file=sys.stderr)
        return 2
    if not model_path.is_file():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 2

    import cv2  # type: ignore
    from event_detection.yolo import YOLOEventDetector

    print(f"Model: {model_path}")
    print(f"SHA-256: {sha256_file(model_path)}")
    print(f"Manifest: {manifest_path}")
    print(f"Labeled recordings: {len(records)}")
    print(f"Threshold note: {TPU_TUNING_NOTE}")
    print(
        "Initializing the Edge TPU. Stop the running squirrel daemon first "
        "if it owns the TPU device."
    )

    try:
        detector = YOLOEventDetector(model_filename=str(model_path))
    except Exception as exc:
        print(f"ERROR: could not initialize detector: {exc}", file=sys.stderr)
        return 1

    results: List[RecordingResult] = []
    started_at = time.time()
    for index, recording in enumerate(sorted(records), start=1):
        record = records[recording]
        label = str(record["label"])
        stored_score = stored_best_score(record)
        video_path = recordings_dir / recording
        try:
            if video_path.is_file():
                result = analyze_video(
                    video_path,
                    label,
                    detector,
                    cv2_module=cv2,
                    mining_threshold=mining_threshold,
                    every_n_frames=every_n_frames,
                    stored_score=stored_score,
                )
            elif label == "false_positive" and not args.no_frame_fallback:
                fallback_paths = generated_negative_paths(record, data_root)
                if fallback_paths:
                    result = analyze_images(
                        recording,
                        label,
                        fallback_paths,
                        detector,
                        cv2_module=cv2,
                        mining_threshold=mining_threshold,
                        stored_score=stored_score,
                    )
                else:
                    result = missing_result(recording, label, stored_score)
            else:
                result = missing_result(recording, label, stored_score)
        except Exception as exc:
            result = RecordingResult(
                recording=recording,
                label=label,
                source="error",
                frames_processed=0,
                detection_count=0,
                best_score=None,
                best_frame=None,
                best_image=None,
                stored_best_score=stored_score,
                error=str(exc),
            )
        results.append(result)

        if args.verbose or result.error:
            score_text = "n/a" if result.best_score is None else f"{result.best_score:.6f}"
            message = (
                f"[{index:02d}/{len(records):02d}] {label:14s} "
                f"{recording} source={result.source} best={score_text}"
            )
            if result.error:
                message += f" error={result.error}"
            print(message)

        if result.source == "error":
            print("Stopping after an inference error; partial results will be written.", file=sys.stderr)
            break

    source_counts: Dict[str, int] = {}
    for result in results:
        source_counts[result.source] = source_counts.get(result.source, 0) + 1
    print(
        "Sources: "
        + ", ".join(f"{source}={count}" for source, count in sorted(source_counts.items()))
    )

    summary = threshold_summary(results, thresholds)
    print_summary(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "recordings.csv"
    json_path = output_dir / "report.json"
    write_recordings_csv(csv_path, results)

    report = {
        "generated_at": time.time(),
        "elapsed_seconds": time.time() - started_at,
        "model": str(model_path),
        "model_sha256": sha256_file(model_path),
        "manifest": str(manifest_path),
        "recordings_dir": str(recordings_dir),
        "data_root": str(data_root),
        "threshold_tuning_note": TPU_TUNING_NOTE,
        "mining_threshold": mining_threshold,
        "every_n_frames": every_n_frames,
        "thresholds": summary,
        "source_counts": source_counts,
        "results": [asdict(result) for result in results],
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print()
    print(f"Per-recording CSV: {csv_path}")
    print(f"JSON report:       {json_path}")

    errors = [result for result in results if result.error]
    if errors:
        print(f"Completed with {len(errors)} missing/error result(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
