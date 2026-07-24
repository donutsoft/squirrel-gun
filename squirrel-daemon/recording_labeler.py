from __future__ import annotations

import csv
import heapq
import json
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import cv2  # type: ignore


BBOX_FIELDS = ("image", "label", "xmin", "ymin", "xmax", "ymax")
FRAME_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")
LABELS = ("true_positive", "false_positive")


class LabelingInProgress(RuntimeError):
    pass


class RecordingLabelService:
    """Mine persistent training frames from user-labeled recordings.

    The data root is intentionally supplied by the app. On the Pi it is a
    sibling of ``squirrel-daemon`` (next to the settings database), so deploys
    that replace the application directory do not remove reviewed frames.
    """

    def __init__(
        self,
        data_root: Path,
        detector_provider: Callable[[], Any],
        *,
        cv2_module: Any = cv2,
        positive_mining_threshold: float = 0.05,
        false_positive_mining_threshold: float = 0.001,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.positives_dir = self.data_root / "positives"
        self.negatives_dir = self.data_root / "negatives"
        self.bbox_path = self.data_root / "bboxes.txt"
        self.manifest_path = self.data_root / "recording_labels.json"
        self.staging_dir = self.data_root / ".staging"
        self._detector_provider = detector_provider
        self._cv2 = cv2_module
        self._positive_mining_threshold = float(positive_mining_threshold)
        self._false_positive_mining_threshold = float(false_positive_mining_threshold)
        self._lock = threading.RLock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._latest_job_by_recording: Dict[str, str] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recording-label")

        self.positives_dir.mkdir(parents=True, exist_ok=True)
        self.negatives_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        if not self.bbox_path.exists():
            self._write_bbox_rows([])

    def start(self, video_path: Path, label: str) -> Dict[str, Any]:
        video_path = Path(video_path).resolve()
        label = str(label).strip()
        if label not in LABELS:
            raise ValueError(f"label must be one of: {', '.join(LABELS)}")
        if not video_path.is_file():
            raise FileNotFoundError(f"recording not found: {video_path}")

        with self._lock:
            for job in self._jobs.values():
                if job["recording"] == video_path.name and job["state"] in ("queued", "running"):
                    raise LabelingInProgress(f"{video_path.name} is already being labeled")
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "recording": video_path.name,
                "label": label,
                "state": "queued",
                "processed_frames": 0,
                "total_frames": None,
                "saved_frames": 0,
                "error": None,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self._jobs[job_id] = job
            self._latest_job_by_recording[video_path.name] = job_id
            self._executor.submit(self._run_job, job_id, video_path, label)
            return dict(job)

    def status(self, recording_name: str) -> Dict[str, Any]:
        recording_name = str(recording_name)
        with self._lock:
            job_id = self._latest_job_by_recording.get(recording_name)
            if job_id is not None and self._jobs[job_id]["state"] != "complete":
                return dict(self._jobs[job_id])
            manifest = self._read_manifest()
            completed = manifest.get("recordings", {}).get(recording_name)
            if isinstance(completed, dict):
                return {
                    "recording": recording_name,
                    "label": completed.get("label"),
                    "state": "complete",
                    "processed_frames": completed.get("processed_frames", 0),
                    "total_frames": completed.get("processed_frames", 0),
                    "saved_frames": completed.get("saved_frames", 0),
                    "error": None,
                    "updated_at": completed.get("updated_at"),
                }
        return {
            "recording": recording_name,
            "label": None,
            "state": "unlabeled",
            "processed_frames": 0,
            "total_frames": None,
            "saved_frames": 0,
            "error": None,
        }

    def list_frames(self) -> Dict[str, Any]:
        with self._lock:
            bbox_rows = self._read_bbox_rows()
            boxes_by_image: Dict[str, List[Dict[str, Any]]] = {}
            for row in bbox_rows:
                boxes_by_image.setdefault(row["image"], []).append(dict(row))
            generated_metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}
            manifest = self._read_manifest()
            for recording_name, record in manifest.get("recordings", {}).items():
                if not isinstance(record, dict):
                    continue
                for item in record.get("generated", []):
                    if not isinstance(item, dict):
                        continue
                    key = (str(item.get("kind", "")), str(item.get("name", "")))
                    generated_metadata[key] = {
                        "source_recording": recording_name,
                        "score": item.get("score"),
                    }

            result: Dict[str, List[Dict[str, Any]]] = {"positives": [], "negatives": []}
            for kind, directory in (("positives", self.positives_dir), ("negatives", self.negatives_dir)):
                for path in directory.glob("*.jpg"):
                    stat = path.stat()
                    boxes = boxes_by_image.get(path.name, []) if kind == "positives" else []
                    metadata = generated_metadata.get((kind, path.name), {})
                    result[kind].append({
                        "kind": kind,
                        "name": path.name,
                        "size": stat.st_size,
                        "mtime": stat.st_mtime,
                        "display_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                        "boxes": boxes,
                        "source_recording": metadata.get("source_recording"),
                        "score": metadata.get("score"),
                    })
                result[kind].sort(key=lambda item: item["mtime"], reverse=True)
            return {
                **result,
                "data_root": str(self.data_root),
                "bbox_path": str(self.bbox_path),
            }

    def frame_path(self, kind: str, name: str) -> Path:
        directory = self._frame_directory(kind)
        self._validate_frame_name(name)
        path = (directory / name).resolve()
        if directory.resolve() not in path.parents:
            raise ValueError("invalid frame path")
        return path

    def delete_frame(self, kind: str, name: str) -> Dict[str, Any]:
        with self._lock:
            path = self.frame_path(kind, name)
            if not path.is_file():
                raise FileNotFoundError(f"frame not found: {name}")
            path.unlink()

            removed_boxes = 0
            if kind == "positives":
                rows = self._read_bbox_rows()
                kept = [row for row in rows if row["image"] != name]
                removed_boxes = len(rows) - len(kept)
                self._write_bbox_rows(kept)

            manifest = self._read_manifest()
            changed = False
            for record in manifest.get("recordings", {}).values():
                generated = record.get("generated", []) if isinstance(record, dict) else []
                filtered = [item for item in generated if not (item.get("kind") == kind and item.get("name") == name)]
                if len(filtered) != len(generated):
                    record["generated"] = filtered
                    record["saved_frames"] = len(filtered)
                    changed = True
            if changed:
                self._write_manifest(manifest)
            return {"deleted": name, "kind": kind, "removed_boxes": removed_boxes}

    def _run_job(self, job_id: str, video_path: Path, label: str) -> None:
        stage_dir = self.staging_dir / job_id
        try:
            with self._lock:
                self._update_job(job_id, state="running")
            stage_dir.mkdir(parents=True, exist_ok=False)
            summary = self._analyze(video_path, label, stage_dir, job_id)
            with self._lock:
                self._commit(video_path.name, label, stage_dir, summary)
                self._update_job(
                    job_id,
                    state="complete",
                    saved_frames=len(summary["images"]),
                    live_threshold=summary["live_threshold"],
                )
        except Exception as exc:
            with self._lock:
                self._update_job(job_id, state="error", error=str(exc))
        finally:
            if stage_dir.exists():
                shutil.rmtree(stage_dir)

    def _analyze(self, video_path: Path, label: str, stage_dir: Path, job_id: str) -> Dict[str, Any]:
        detector = self._detector_provider()
        config = detector.config()
        live_threshold = float(config.get("score_thresh", 0.6))
        if not 0.0 < live_threshold <= 1.0:
            raise ValueError(f"invalid configured squirrel threshold: {live_threshold}")

        cap = self._cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"could not open recording: {video_path.name}")

        total = int(cap.get(self._cv2.CAP_PROP_FRAME_COUNT) or 0)
        with self._lock:
            self._update_job(job_id, total_frames=total if total > 0 else None)

        images: List[Dict[str, Any]] = []
        bbox_rows: List[Dict[str, Any]] = []
        top_negatives: List[Tuple[float, int, Any]] = []
        frame_index = -1
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frame_index += 1
                mining_threshold = (
                    self._false_positive_mining_threshold
                    if label == "false_positive"
                    else min(self._positive_mining_threshold, live_threshold / 2.0)
                )
                letterboxed, detections = detector.predict_candidates(
                    frame,
                    score_thresh=mining_threshold,
                )

                if label == "false_positive":
                    if detections:
                        score = max(float(item["score"]) for item in detections)
                        entry = (score, frame_index, letterboxed.copy())
                        if len(top_negatives) < 2:
                            heapq.heappush(top_negatives, entry)
                        elif (score, frame_index) > (top_negatives[0][0], top_negatives[0][1]):
                            heapq.heapreplace(top_negatives, entry)
                else:
                    low_detections = [
                        item for item in detections
                        if mining_threshold <= float(item["score"]) < live_threshold
                    ]
                    if low_detections:
                        name = f"{video_path.stem}_frame{frame_index:08d}.jpg"
                        self._write_image(stage_dir / name, letterboxed)
                        images.append({"kind": "positives", "name": name})
                        for detection in low_detections:
                            bbox_rows.append(self._bbox_row(name, detection))

                if frame_index % 5 == 0:
                    with self._lock:
                        self._update_job(job_id, processed_frames=frame_index + 1)
        finally:
            cap.release()

        if label == "false_positive":
            for score, index, image in sorted(top_negatives, key=lambda item: (item[0], item[1]), reverse=True):
                name = f"{video_path.stem}_frame{index:08d}.jpg"
                self._write_image(stage_dir / name, image)
                images.append({"kind": "negatives", "name": name, "score": score})

        with self._lock:
            self._update_job(job_id, processed_frames=frame_index + 1 if frame_index >= 0 else 0)
        return {
            "images": images,
            "bbox_rows": bbox_rows,
            "processed_frames": frame_index + 1 if frame_index >= 0 else 0,
            "live_threshold": live_threshold,
        }

    def _commit(self, recording_name: str, label: str, stage_dir: Path, summary: Dict[str, Any]) -> None:
        manifest = self._read_manifest()
        recordings = manifest.setdefault("recordings", {})
        previous = recordings.get(recording_name, {})
        previous_generated = previous.get("generated", []) if isinstance(previous, dict) else []
        old_positive_names = {
            item["name"] for item in previous_generated
            if item.get("kind") == "positives" and isinstance(item.get("name"), str)
        }

        bbox_rows = [row for row in self._read_bbox_rows() if row["image"] not in old_positive_names]
        bbox_rows.extend(summary["bbox_rows"])

        for item in previous_generated:
            try:
                old_path = self.frame_path(str(item.get("kind", "")), str(item.get("name", "")))
            except ValueError:
                continue
            if old_path.is_file():
                old_path.unlink()

        for item in summary["images"]:
            destination = self.frame_path(item["kind"], item["name"])
            (stage_dir / item["name"]).replace(destination)

        self._write_bbox_rows(bbox_rows)
        recordings[recording_name] = {
            "label": label,
            "processed_frames": int(summary["processed_frames"]),
            "saved_frames": len(summary["images"]),
            "live_threshold": float(summary["live_threshold"]),
            "generated": summary["images"],
            "updated_at": time.time(),
        }
        self._write_manifest(manifest)

    @staticmethod
    def _bbox_row(name: str, detection: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "image": name,
            "label": "rat",
            "xmin": int(detection["x1"]),
            "ymin": int(detection["y1"]),
            "xmax": int(detection["x2"]),
            "ymax": int(detection["y2"]),
        }

    def _write_image(self, path: Path, image: Any) -> None:
        ok = self._cv2.imwrite(str(path), image, [self._cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError(f"could not write training frame: {path}")

    def _read_bbox_rows(self) -> List[Dict[str, Any]]:
        if not self.bbox_path.exists():
            return []
        with self.bbox_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(BBOX_FIELDS):
                raise RuntimeError(f"invalid bbox header in {self.bbox_path}")
            return [dict(row) for row in reader]

    def _write_bbox_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.data_root.mkdir(parents=True, exist_ok=True)
        temporary = self.bbox_path.with_suffix(".txt.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(BBOX_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.bbox_path)

    def _read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": 1, "recordings": {}}
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid recording label manifest: {self.manifest_path}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("recordings"), dict):
            raise RuntimeError(f"invalid recording label manifest: {self.manifest_path}")
        return value

    def _write_manifest(self, manifest: Dict[str, Any]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _frame_directory(self, kind: str) -> Path:
        if kind == "positives":
            return self.positives_dir
        if kind == "negatives":
            return self.negatives_dir
        raise ValueError("kind must be positives or negatives")

    @staticmethod
    def _validate_frame_name(name: str) -> None:
        if not FRAME_NAME_RE.fullmatch(str(name)) or Path(name).name != name:
            raise ValueError("invalid frame filename")

    def _update_job(self, job_id: str, **changes: Any) -> None:
        job = self._jobs[job_id]
        job.update(changes)
        job["updated_at"] = time.time()
