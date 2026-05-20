from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
import time

import numpy as np  # type: ignore
import cv2  # type: ignore

from .base import EventDetector, DetectionEvent, DetectionResult
from ultralytics import YOLO  # type: ignore


class YOLOEventDetector(EventDetector):
    """Ultralytics YOLO wrapper for TFLite model that emits bbox events.

    Usage is intentionally simple:
      model = YOLO("best_full_integer_quant_edgetpu.tflite", task='detect')
      model.predict(image)
    """

    def __init__(self, model_filename: str = "best_full_integer_quant_edgetpu.tflite") -> None:
        self._enabled = True
        self._score_thresh = 0.4
        self._frame_skip = 0
        self._allowed_classes: Optional[Sequence[int]] = None
        self._label_map: Optional[Dict[int, str]] = None
        self._suppress_until_ts = 0.0
        self._counter = 0
        self._events_published = 0
        self._last_confidence = 0.0
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

    def _is_tpu_delegate_error(self, exc: BaseException) -> bool:
        message = str(exc).lower()
        return (
            "libedgetpu" in message
            or "edgetpu" in message
            or "delegate" in message
        )

    def _predict_tpu(self, image: Any) -> Any:
        try:
            return self._model.predict(image, verbose=False, conf=float(self._score_thresh))  # type: ignore
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

    # --- EventDetector API ---
    def enabled(self) -> bool:
        return bool(self._enabled)

    def configure(self, **kwargs: Any) -> None:
        if 'enabled' in kwargs:
            self._enabled = bool(kwargs['enabled'])
        if 'score_thresh' in kwargs and kwargs['score_thresh'] is not None:
            self._score_thresh = float(kwargs['score_thresh'])
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
        }

    def reset_metrics(self) -> None:
        self._events_published = 0
        self._last_confidence = 0.0

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

        work = frame.copy()
        h, w = work.shape[:2]
        # Preprocess: letterbox to 320x320 with gray 0x72 (114) like extract_frames.py
        TARGET = 320
        PAD_COLOR = 114
        scale = min(TARGET / float(w), TARGET / float(h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(work, (new_w, new_h), interpolation=interp)
        lb = np.full((TARGET, TARGET, 3), PAD_COLOR, dtype=resized.dtype)
        top = (TARGET - new_h) // 2
        left = (TARGET - new_w) // 2
        lb[top:top+new_h, left:left+new_w] = resized
        # Run Ultralytics YOLO on the letterboxed image
        results = self._predict_tpu(lb)
        # Parse detections in letterbox space (TARGET x TARGET)
        detections_lb = self._parse_ultralytics(results, TARGET, TARGET)
        raw_detection_count = len(detections_lb)
        detections_lb = [d for d in detections_lb if self._bbox_size_allowed(d, TARGET, TARGET)]
        filtered_detection_count = raw_detection_count - len(detections_lb)
        saved_image_path = self._save_detection_image(lb, detections_lb, now_ts)
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
        return DetectionResult(frame=work, events=events, metrics={
            'count': len(detections),
            'raw_count': raw_detection_count,
            'filtered_count': filtered_detection_count,
            'last_confidence': self._last_confidence,
        })

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
    def _parse_ultralytics(self, results: Any, img_w: int, img_h: int) -> List[Dict[str, Any]]:
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
        for i in range(len(conf)):
            score = float(conf[i])
            if score < float(self._score_thresh):
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
