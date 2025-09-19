from __future__ import annotations

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
        # Resolve model path relative to squirrel-daemon root and load via Ultralytics
        self._model_path = (Path(__file__).resolve().parents[1] / model_filename)
        if not self._model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self._model_path}")
        self._model = YOLO(str(self._model_path), task='detect')

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
            'using_tpu': None,
        }

    def config(self) -> Dict[str, Any]:
        return {
            'enabled': bool(self._enabled),
            'score_thresh': float(self._score_thresh),
            'frame_skip': int(self._frame_skip),
            'zone': None,
            'allowed_classes': list(self._allowed_classes) if self._allowed_classes is not None else None,
            'model_path': str(self._model_path),
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
        results = self._model.predict(lb, verbose=False, conf=float(self._score_thresh))  # type: ignore
        # Parse detections in letterbox space (TARGET x TARGET)
        detections_lb = self._parse_ultralytics(results, TARGET, TARGET)
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
                extra={'score': conf, 'class': cls}
            ))

        if events:
            self._events_published += len(events)
        return DetectionResult(frame=work, events=events, metrics={'count': len(detections), 'last_confidence': self._last_confidence})

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
