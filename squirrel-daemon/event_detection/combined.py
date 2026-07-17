from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import cv2  # type: ignore

from .base import EventDetector, DetectionEvent, DetectionResult
from .motion import MotionDetector
from .yolo import YOLOEventDetector


class CombinedMotionYOLOEventDetector(EventDetector):
    """Motion-gated YOLO detector.

    Motion remains the primary event source so every motion event can trigger
    recording. YOLO is only run after motion is detected, and squirrel
    confirmation is attached to the motion event metadata.
    """

    def __init__(self, yolo_detector: Optional[YOLOEventDetector] = None) -> None:
        self._motion = MotionDetector()
        self._yolo = yolo_detector or YOLOEventDetector()
        self._events_published = 0
        self._last_squirrel_confidence = 0.0
        self._last_yolo_count = 0

    def yolo_detector(self) -> YOLOEventDetector:
        return self._yolo

    def enabled(self) -> bool:
        return self._motion.enabled()

    def configure(self, **kwargs: Any) -> None:
        self._motion.configure(**kwargs)
        yolo_kwargs: Dict[str, Any] = {}
        for key in (
            'score_thresh',
            'classes',
            'save_detection_images',
            'detection_output_dir',
            'bbox_limits_path',
            'max_bbox_width_frac',
            'max_bbox_height_frac',
            'max_bbox_area_frac',
            'labels',
        ):
            if key in kwargs:
                yolo_kwargs[key] = kwargs[key]
        if yolo_kwargs:
            self._yolo.configure(**yolo_kwargs)

    def set_zone(self, zone: Optional[Sequence[float]]) -> None:
        self._motion.set_zone(zone)

    def get_zone(self) -> Optional[Tuple[float, float, float, float]]:
        return self._motion.get_zone()

    def suppress(self, duration_sec: float) -> None:
        self._motion.suppress(duration_sec)
        self._yolo.suppress(duration_sec)

    def info(self, frame_size: Tuple[int, int]) -> Dict[str, Any]:
        info = self._motion.info(frame_size)
        yolo_info = self._yolo.info(frame_size)
        yolo_timing = {
            f'yolo_{key}': value
            for key, value in yolo_info.items()
            if key.endswith('_time_ms') and key not in ('process_time_ms', 'avg_process_time_ms')
        }
        info.update({
            'detector': 'combined',
            'squirrel_detected': bool(self._last_yolo_count > 0),
            'yolo_count': int(self._last_yolo_count),
            'last_squirrel_confidence': float(self._last_squirrel_confidence),
            'events_published': int(self._events_published),
            'motion_process_time_ms': info.get('process_time_ms'),
            'motion_avg_process_time_ms': info.get('avg_process_time_ms'),
            'motion_processed_frames': info.get('processed_frames'),
            'yolo_process_time_ms': yolo_info.get('process_time_ms'),
            'yolo_avg_process_time_ms': yolo_info.get('avg_process_time_ms'),
            'yolo_processed_frames': yolo_info.get('processed_frames'),
            'yolo_warmed_up': yolo_info.get('warmed_up'),
            'yolo_warmup_time_ms': yolo_info.get('warmup_time_ms'),
            **yolo_timing,
        })
        return info

    def config(self) -> Dict[str, Any]:
        cfg = self._motion.config()
        yolo_cfg = self._yolo.config()
        yolo_info = self._yolo.info((0, 0))
        cfg.update({
            'detector': 'combined',
            'yolo_score_thresh': yolo_cfg.get('score_thresh'),
            'yolo_frame_skip': yolo_cfg.get('frame_skip'),
            'yolo_allowed_classes': yolo_cfg.get('allowed_classes'),
            'yolo_model_path': yolo_cfg.get('model_path'),
            'yolo_save_detection_images': yolo_cfg.get('save_detection_images'),
            'yolo_detection_output_dir': yolo_cfg.get('detection_output_dir'),
            'yolo_bbox_limits_path': yolo_cfg.get('bbox_limits_path'),
            'squirrel_detected': bool(self._last_yolo_count > 0),
            'yolo_count': int(self._last_yolo_count),
            'last_squirrel_confidence': float(self._last_squirrel_confidence),
            'yolo_avg_process_time_ms': yolo_info.get('avg_process_time_ms'),
            'yolo_processed_frames': yolo_info.get('processed_frames'),
        })
        return cfg

    def reset_metrics(self) -> None:
        self._motion.reset_metrics()
        self._yolo.reset_metrics()
        self._events_published = 0
        self._last_squirrel_confidence = 0.0
        self._last_yolo_count = 0

    def process(self, frame: Any, now_ts: Optional[float] = None) -> DetectionResult:
        if now_ts is None:
            now_ts = time.time()
        try:
            original_frame = frame.copy()
        except Exception:
            original_frame = frame
        motion_result = self._motion.process(frame, now_ts=now_ts)
        if not motion_result.events:
            self._last_yolo_count = 0
            return DetectionResult(frame=motion_result.frame, events=[], metrics={
                **motion_result.metrics,
                'yolo_count': 0,
                'squirrel_detected': False,
            })

        yolo_events: List[DetectionEvent] = []
        yolo_frame = motion_result.frame
        yolo_metrics: Dict[str, Any] = {}
        try:
            # Run YOLO on the original frame so motion overlays do not affect inference.
            yolo_result = self._yolo.process(original_frame, now_ts=now_ts)
            yolo_events = list(yolo_result.events)
            yolo_frame = yolo_result.frame
            yolo_metrics = dict(yolo_result.metrics or {})
        except Exception as exc:
            yolo_metrics = {'error': str(exc)}
            yolo_events = []

        squirrel_detected = bool(yolo_events)
        self._last_yolo_count = len(yolo_events)
        if yolo_events:
            self._last_squirrel_confidence = max(
                float((event.extra or {}).get('score', 0.0))
                for event in yolo_events
            )
        else:
            self._last_squirrel_confidence = 0.0

        out_frame = self._compose_frame(motion_result.frame, yolo_frame, squirrel_detected)
        best_yolo = self._best_yolo_event(yolo_events)
        events = [
            self._annotate_motion_event(event, squirrel_detected, best_yolo, yolo_metrics)
            for event in motion_result.events
        ]
        if events:
            self._events_published += len(events)

        return DetectionResult(frame=out_frame, events=events, metrics={
            **motion_result.metrics,
            'yolo_count': int(self._last_yolo_count),
            'last_squirrel_confidence': float(self._last_squirrel_confidence),
            'squirrel_detected': bool(squirrel_detected),
            'yolo': yolo_metrics,
            'motion_process_time_ms': motion_result.metrics.get('process_time_ms'),
            'motion_avg_process_time_ms': motion_result.metrics.get('avg_process_time_ms'),
            'motion_processed_frames': motion_result.metrics.get('processed_frames'),
            'yolo_process_time_ms': yolo_metrics.get('process_time_ms'),
            'yolo_avg_process_time_ms': yolo_metrics.get('avg_process_time_ms'),
            'yolo_processed_frames': yolo_metrics.get('processed_frames'),
            **{
                f'yolo_{key}': value
                for key, value in yolo_metrics.items()
                if key.endswith('_time_ms') and key not in ('process_time_ms', 'avg_process_time_ms')
            },
        })

    def _compose_frame(self, motion_frame: Any, yolo_frame: Any, squirrel_detected: bool) -> Any:
        if squirrel_detected:
            try:
                return yolo_frame.copy()
            except Exception:
                return yolo_frame
        try:
            frame = motion_frame.copy()
            cv2.putText(
                frame,
                "motion: no squirrel",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            return frame
        except Exception:
            return motion_frame

    @staticmethod
    def _best_yolo_event(events: Sequence[DetectionEvent]) -> Optional[DetectionEvent]:
        if not events:
            return None
        return max(events, key=lambda event: float((event.extra or {}).get('score', 0.0)))

    @staticmethod
    def _annotate_motion_event(
        event: DetectionEvent,
        squirrel_detected: bool,
        best_yolo: Optional[DetectionEvent],
        yolo_metrics: Dict[str, Any],
    ) -> DetectionEvent:
        extra = dict(event.extra or {})
        extra.update({
            'detector': 'combined',
            'motion_detected': True,
            'squirrel_detected': bool(squirrel_detected),
            'yolo_count': int(yolo_metrics.get('count', 0) or 0),
            'yolo_raw_count': int(yolo_metrics.get('raw_count', 0) or 0),
            'yolo_filtered_count': int(yolo_metrics.get('filtered_count', 0) or 0),
            'yolo_last_confidence': float(yolo_metrics.get('last_confidence', 0.0) or 0.0),
        })
        if 'error' in yolo_metrics:
            extra['yolo_error'] = str(yolo_metrics['error'])
        if best_yolo is not None:
            yolo_extra = dict(best_yolo.extra or {})
            extra.update({
                'squirrel_rect': best_yolo.rect,
                'squirrel_center': best_yolo.center,
                'score': float(yolo_extra.get('score', 0.0)),
                'class': yolo_extra.get('class'),
            })
            if 'image_path' in yolo_extra:
                extra['image_path'] = yolo_extra['image_path']

        return DetectionEvent(
            ts=event.ts,
            rect=event.rect,
            center=event.center,
            u=event.u,
            v=event.v,
            width=event.width,
            height=event.height,
            extra=extra,
        )
