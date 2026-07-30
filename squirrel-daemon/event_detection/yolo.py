from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import time

import numpy as np  # type: ignore
import cv2  # type: ignore

from .base import EventDetector, DetectionEvent, DetectionResult
from ultralytics import YOLO  # type: ignore


DEFAULT_YOLO_SCORE_THRESH = 0.05


def validate_yolo_score_threshold(value: Any) -> float:
    threshold = float(value)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("YOLO score threshold must be greater than 0 and at most 1")
    return threshold


class YOLOEventDetector(EventDetector):
    """Ultralytics YOLO wrapper for TFLite model that emits bbox events.

    Usage is intentionally simple:
      model = YOLO("best_full_integer_quant_edgetpu.tflite", task='detect')
      model.predict(image)
    """

    def __init__(self, model_filename: str = "best_full_integer_quant_edgetpu.tflite") -> None:
        self._enabled = True
        # On-device evaluation of the default-augmentation YOLO26n Edge-TPU
        # model found that 0.05 rejected every available negative holdout while
        # retaining 61/75 positive holdouts. Edge-TPU scores are model-specific,
        # so recalibrate after changing models.
        self._score_thresh = DEFAULT_YOLO_SCORE_THRESH
        self._frame_skip = 0
        self._allowed_classes: Optional[Sequence[int]] = None
        self._label_map: Optional[Dict[int, str]] = None
        self._suppress_until_ts = 0.0
        self._counter = 0
        self._events_published = 0
        self._last_confidence = 0.0
        self._process_time_total_sec = 0.0
        self._process_time_count = 0
        self._last_process_time_sec = 0.0
        self._timing_stage_names = ('preprocess', 'inference', 'postprocess', 'save_image', 'annotate')
        self._last_timing_stages_sec = {name: 0.0 for name in self._timing_stage_names}
        self._timing_stage_totals_sec = {name: 0.0 for name in self._timing_stage_names}
        self._warmed_up = False
        self._warmup_time_sec = 0.0
        self._inference_lock = threading.Lock()
        self._daemon_root = Path(__file__).resolve().parents[1]
        self._detection_output_dir = self._daemon_root / "detections"
        self._save_detection_images = True
        self._bbox_limits_path: Optional[Path] = self._daemon_root / "yolo_bbox_limits.json"
        self._max_bbox_width_frac: Optional[float] = None
        self._max_bbox_height_frac: Optional[float] = None
        self._max_bbox_area_frac: Optional[float] = None
        self._load_bbox_size_limits(self._bbox_limits_path)
        # Resolve model path relative to squirrel-daemon root and load via Ultralytics
        self._model_path = (self._daemon_root / model_filename)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self._model = YOLO(str(self._model_path), task='detect')
        self._warmup_model()

    def _is_tpu_delegate_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return (
            "libedgetpu" in message
            or "edgetpu" in message
            or "delegate" in message
        )

    def _predict_tpu(self, image: Any, score_thresh: Optional[float] = None) -> Any:
        threshold = self._score_thresh if score_thresh is None else float(score_thresh)
        try:
            # The live camera loop and recording-label worker share one Edge TPU
            # interpreter. Ultralytics/TFLite inference is not safe to enter from
            # both threads at once.
            with self._inference_lock:
                return self._model.predict(image, verbose=False, conf=threshold)  # type: ignore
        except (RuntimeError, ValueError) as exc:
            if not self._is_tpu_delegate_error(exc):
                raise
            raise RuntimeError(
                "Edge TPU inference failed while loading the TFLite delegate. "
                "This detector is TPU-only, so no CPU fallback was used. "
                f"Model: {self._model_path}. "
                f"Underlying error: {exc}. "
                "Check that the Coral TPU is attached, the Edge TPU runtime is installed, "
                "the current user can access the TPU device, and no other process is already using it. "
                "Run `uv run check_edgetpu.py` on the Pi for device, permission, and delegate diagnostics."
            ) from exc

    def _warmup_model(self) -> None:
        """Force delegate/model initialization before the first real event frame."""
        start = time.perf_counter()
        warmup_frame = np.full((320, 320, 3), 114, dtype=np.uint8)
        self._predict_tpu(warmup_frame)
        self._warmup_time_sec = max(0.0, time.perf_counter() - start)
        self._warmed_up = True

    # --- EventDetector API ---
    def enabled(self) -> bool:
        return bool(self._enabled)

    def configure(self, **kwargs: Any) -> None:
        if 'enabled' in kwargs:
            self._enabled = bool(kwargs['enabled'])
        if 'score_thresh' in kwargs and kwargs['score_thresh'] is not None:
            self._score_thresh = validate_yolo_score_threshold(kwargs['score_thresh'])
        if 'frame_skip' in kwargs and kwargs['frame_skip'] is not None:
            self._frame_skip = max(0, int(kwargs['frame_skip']))
        if 'classes' in kwargs and kwargs['classes'] is not None:
            self._allowed_classes = [int(c) for c in kwargs['classes']]
        if 'save_detection_images' in kwargs and kwargs['save_detection_images'] is not None:
            self._save_detection_images = bool(kwargs['save_detection_images'])
        if 'detection_output_dir' in kwargs and kwargs['detection_output_dir'] is not None:
            output_dir = Path(str(kwargs['detection_output_dir']))
            if not output_dir.is_absolute():
                output_dir = self._daemon_root / output_dir
            self._detection_output_dir = output_dir
        if 'bbox_limits_path' in kwargs:
            if kwargs['bbox_limits_path'] is None:
                self._bbox_limits_path = None
                self._max_bbox_width_frac = None
                self._max_bbox_height_frac = None
                self._max_bbox_area_frac = None
            else:
                p = Path(str(kwargs['bbox_limits_path']))
                if not p.is_absolute():
                    p = self._daemon_root / p
                self._bbox_limits_path = p
                self._load_bbox_size_limits(p)
        for key, attr in (
            ('max_bbox_width_frac', '_max_bbox_width_frac'),
            ('max_bbox_height_frac', '_max_bbox_height_frac'),
            ('max_bbox_area_frac', '_max_bbox_area_frac'),
        ):
            if key in kwargs:
                value = kwargs[key]
                setattr(self, attr, None if value is None else float(value))
        if 'labels' in kwargs and kwargs['labels'] is not None:
            # Accept dict {id: name} or path to labels file
            if isinstance(kwargs['labels'], dict):
                self._label_map = {int(k): str(v) for k, v in kwargs['labels'].items()}
            else:
                p = Path(str(kwargs['labels']))
                lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
                self._label_map = {i: name for i, name in enumerate(lines)}

    def set_zone(self, zone: Optional[Sequence[float]]) -> None:
        # Detection zones are not applicable for YOLO; ignore.
        return

    def get_zone(self) -> Optional[Tuple[float, float, float, float]]:
        # Not applicable for YOLO
        return None

    def suppress(self, duration_sec: float) -> None:
        d = float(duration_sec)
        self._suppress_until_ts = time.time() + max(0.0, d)

    def info(self, frame_size: Tuple[int, int]) -> Dict[str, Any]:
        w, h = frame_size
        return {
            'enabled': bool(self.enabled()),
            'rect': None,
            'center': None,
            'u': None,
            'v': None,
            'width': int(w) if w else None,
            'height': int(h) if h else None,
            'detections': None,
            'last_confidence': float(self._last_confidence),
            'events_published': int(self._events_published),
            'using_tpu': True,
            'process_time_ms': float(self._last_process_time_sec * 1000.0),
            'avg_process_time_ms': float(self._avg_process_time_sec() * 1000.0),
            'processed_frames': int(self._process_time_count),
            'warmed_up': bool(self._warmed_up),
            'warmup_time_ms': float(self._warmup_time_sec * 1000.0),
            **self._timing_stage_metrics(),
        }

    def config(self) -> Dict[str, Any]:
        return {
            'enabled': bool(self._enabled),
            'score_thresh': float(self._score_thresh),
            'frame_skip': int(self._frame_skip),
            'zone': None,
            'allowed_classes': list(self._allowed_classes) if self._allowed_classes is not None else None,
            'model_path': str(self._model_path),
            'save_detection_images': bool(self._save_detection_images),
            'detection_output_dir': str(self._detection_output_dir),
            'bbox_limits_path': str(self._bbox_limits_path) if self._bbox_limits_path is not None else None,
            'max_bbox_width_frac': self._max_bbox_width_frac,
            'max_bbox_height_frac': self._max_bbox_height_frac,
            'max_bbox_area_frac': self._max_bbox_area_frac,
            'warmed_up': bool(self._warmed_up),
            'warmup_time_ms': float(self._warmup_time_sec * 1000.0),
        }

    def reset_metrics(self) -> None:
        self._events_published = 0
        self._last_confidence = 0.0
        self._process_time_total_sec = 0.0
        self._process_time_count = 0
        self._last_process_time_sec = 0.0
        self._last_timing_stages_sec = {name: 0.0 for name in self._timing_stage_names}
        self._timing_stage_totals_sec = {name: 0.0 for name in self._timing_stage_names}

    def predict_candidates(
        self,
        frame: Any,
        *,
        score_thresh: float,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Return a 320px letterboxed frame and detections in that space.

        Recording labeling deliberately uses a lower threshold than live event
        detection so hard positives just below the configured live threshold
        are available for training.
        """
        threshold = validate_yolo_score_threshold(score_thresh)
        letterboxed, _scale, _left, _top = self._letterbox(frame)
        results = self._predict_tpu(letterboxed, score_thresh=threshold)
        detections = self._parse_ultralytics(
            results,
            letterboxed.shape[1],
            letterboxed.shape[0],
            score_thresh=threshold,
        )
        detections = [
            detection
            for detection in detections
            if self._bbox_size_allowed(
                detection,
                letterboxed.shape[1],
                letterboxed.shape[0],
            )
        ]
        return letterboxed, detections

    @classmethod
    def prepare_input_frame(cls, frame: Any) -> Any:
        """Return the exact 320px image format supplied to the detector."""
        letterboxed, _scale, _left, _top = cls._letterbox(frame)
        return letterboxed

    @staticmethod
    def _letterbox(frame: Any) -> Tuple[Any, float, int, int]:
        target = 320
        pad_color = 114
        height, width = frame.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("frame must have positive width and height")
        scale = min(target / float(width), target / float(height))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
        letterboxed = np.full((target, target, 3), pad_color, dtype=resized.dtype)
        top = (target - new_height) // 2
        left = (target - new_width) // 2
        letterboxed[top:top + new_height, left:left + new_width] = resized
        return letterboxed, scale, left, top

    # --- Core processing ---
    def process(self, frame: Any, now_ts: Optional[float] = None) -> DetectionResult:
        if now_ts is None:
            now_ts = time.time()
        if not self.enabled():
            raise RuntimeError("YOLOEventDetector is disabled")

        # Optionally skip frames
        c = self._counter
        self._counter = c + 1
        if (c % max(1, int(self._frame_skip) + 1)) != 0:
            return DetectionResult(frame=frame, events=[], metrics={})

        process_start = time.perf_counter()
        work = frame.copy()
        h, w = work.shape[:2]
        # Preprocess: letterbox to 320x320 with gray 0x72 (114) like the
        # persistent training dataset.
        lb, scale, left, top = self._letterbox(work)
        target = int(lb.shape[0])
        preprocess_end = time.perf_counter()
        # Run Ultralytics YOLO on the letterboxed image
        results = self._predict_tpu(lb)
        inference_end = time.perf_counter()
        # Parse detections in letterbox space (TARGET x TARGET)
        detections_lb = self._parse_ultralytics(results, target, target)
        raw_detection_count = len(detections_lb)
        detections_lb = [d for d in detections_lb if self._bbox_size_allowed(d, target, target)]
        filtered_detection_count = raw_detection_count - len(detections_lb)
        save_start = time.perf_counter()
        saved_image_path = self._save_detection_image(lb, detections_lb, now_ts)
        save_end = time.perf_counter()
        # Map detections back to original frame coordinates
        detections = []
        for d in detections_lb:
            x1_lb, y1_lb, x2_lb, y2_lb = d['x1'], d['y1'], d['x2'], d['y2']
            x1 = int(round((x1_lb - left) / scale))
            y1 = int(round((y1_lb - top) / scale))
            x2 = int(round((x2_lb - left) / scale))
            y2 = int(round((y2_lb - top) / scale))
            # Clamp to image bounds
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cx': cx, 'cy': cy, 'score': d['score'], 'class': d.get('class', -1)})
        postprocess_end = time.perf_counter()

        # No zone filtering for YOLO detector

        events: List[DetectionEvent] = []
        # Draw and build events for all detections (could limit to top-1 if preferred)
        for d in detections:
            x1, y1, x2, y2 = d['x1'], d['y1'], d['x2'], d['y2']
            conf = float(d['score'])
            cls = int(d.get('class', -1))
            self._last_confidence = conf
            cv2.rectangle(work, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{self._label_map.get(cls, str(cls)) if self._label_map else cls}:{conf:.2f}"
            cv2.putText(work, str(label), (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if now_ts < self._suppress_until_ts:
                continue
            cx = float(d['cx']); cy = float(d['cy'])
            events.append(DetectionEvent(
                ts=now_ts,
                rect=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                center=(cx, cy),
                extra={
                    'score': conf,
                    'class': cls,
                    **({'image_path': str(saved_image_path)} if saved_image_path is not None else {}),
                }
            ))

        if events:
            self._events_published += len(events)
        annotate_end = time.perf_counter()
        self._record_timing_stages({
            'preprocess': preprocess_end - process_start,
            'inference': inference_end - preprocess_end,
            'postprocess': (save_start - inference_end) + (postprocess_end - save_end),
            'save_image': save_end - save_start,
            'annotate': annotate_end - postprocess_end,
        })
        self._record_process_time(process_start)
        return DetectionResult(frame=work, events=events, metrics={
            'count': len(detections),
            'raw_count': raw_detection_count,
            'filtered_count': filtered_detection_count,
            'last_confidence': self._last_confidence,
            'process_time_ms': float(self._last_process_time_sec * 1000.0),
            'avg_process_time_ms': float(self._avg_process_time_sec() * 1000.0),
            'processed_frames': int(self._process_time_count),
            **self._timing_stage_metrics(),
        })

    def _record_process_time(self, start: float) -> None:
        elapsed = max(0.0, time.perf_counter() - float(start))
        self._last_process_time_sec = elapsed
        self._process_time_total_sec += elapsed
        self._process_time_count += 1

    def _avg_process_time_sec(self) -> float:
        if self._process_time_count <= 0:
            return 0.0
        return float(self._process_time_total_sec) / float(self._process_time_count)

    def _record_timing_stages(self, stages: Dict[str, float]) -> None:
        for name in self._timing_stage_names:
            elapsed = max(0.0, float(stages.get(name, 0.0)))
            self._last_timing_stages_sec[name] = elapsed
            self._timing_stage_totals_sec[name] += elapsed

    def _timing_stage_metrics(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        count = max(0, int(self._process_time_count))
        for name in self._timing_stage_names:
            last_sec = float(self._last_timing_stages_sec.get(name, 0.0))
            total_sec = float(self._timing_stage_totals_sec.get(name, 0.0))
            avg_sec = (total_sec / float(count)) if count > 0 else 0.0
            metrics[f'{name}_time_ms'] = last_sec * 1000.0
            metrics[f'avg_{name}_time_ms'] = avg_sec * 1000.0
        return metrics

    def _load_bbox_size_limits(self, path: Optional[Path]) -> None:
        self._max_bbox_width_frac = None
        self._max_bbox_height_frac = None
        self._max_bbox_area_frac = None
        if path is None or not path.exists():
            return

        try:
            data = json.loads(path.read_text())
            self._max_bbox_width_frac = self._optional_positive_float(data.get('max_width_frac'))
            self._max_bbox_height_frac = self._optional_positive_float(data.get('max_height_frac'))
            self._max_bbox_area_frac = self._optional_positive_float(data.get('max_area_frac'))
        except Exception:
            return

    @staticmethod
    def _optional_positive_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        parsed = float(value)
        if parsed <= 0.0:
            return None
        return min(1.0, parsed)

    def _bbox_size_allowed(self, detection: Dict[str, Any], img_w: int, img_h: int) -> bool:
        width = max(0.0, float(detection['x2']) - float(detection['x1']))
        height = max(0.0, float(detection['y2']) - float(detection['y1']))
        if width <= 0.0 or height <= 0.0:
            return False

        if self._max_bbox_width_frac is not None and width / float(img_w) > float(self._max_bbox_width_frac):
            return False
        if self._max_bbox_height_frac is not None and height / float(img_h) > float(self._max_bbox_height_frac):
            return False
        if self._max_bbox_area_frac is not None and (width * height) / float(img_w * img_h) > float(self._max_bbox_area_frac):
            return False
        return True

    def _save_detection_image(self, image: Any, detections: Sequence[Dict[str, Any]], now_ts: float) -> Optional[Path]:
        if not self._save_detection_images or not detections:
            return None

        best = max(detections, key=lambda item: float(item.get('score', 0.0)))
        score = float(best.get('score', 0.0))
        cls = int(best.get('class', -1))
        millis = int((now_ts - int(now_ts)) * 1000.0)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
        path = self._detection_output_dir / f"yolo_{stamp}_{millis:03d}_score{score:.3f}_cls{cls}.jpg"

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if cv2.imwrite(str(path), image):
                return path
        except Exception:
            pass
        return None

    # --- helpers ---
    def _parse_ultralytics(
        self,
        results: Any,
        img_w: int,
        img_h: int,
        score_thresh: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        if not results:
            return detections
        r0 = results[0]
        boxes = r0.boxes
        xyxy = boxes.xyxy
        conf = boxes.conf
        cls = boxes.cls
        xyxy = np.array(xyxy)
        conf = np.array(conf).flatten()
        classes = np.array(cls).flatten() if cls is not None else np.full(conf.shape, -1)
        threshold = self._score_thresh if score_thresh is None else float(score_thresh)
        for i in range(len(conf)):
            score = float(conf[i])
            if score < threshold:
                continue
            icls = int(classes[i])
            if self._allowed_classes is not None and icls not in self._allowed_classes:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[i].tolist()]
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w - 1, x2))
            y2 = max(0, min(img_h - 1, y2))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'cx': cx, 'cy': cy, 'score': score, 'class': icls})
        return detections
