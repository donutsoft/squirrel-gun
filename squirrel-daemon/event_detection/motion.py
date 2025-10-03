from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

import cv2  # type: ignore

from .base import EventDetector, DetectionEvent, DetectionResult


class MotionDetector(EventDetector):
    """Background-subtraction motion detector extracted from WebcamController.

    This implementation draws overlays directly onto the provided frame.
    """

    def __init__(self) -> None:
        # Config
        self._enabled = True
        self._min_area = 1500
        self._alpha = 0.05
        self._persist_ms = 250
        self._bg_mode = 'knn'  # 'avg' | 'mog2' | 'knn'
        self._prefer_tracking = True
        self._frame_skip = 0
        self._scale = 0.5
        self._zone: Optional[Tuple[float, float, float, float]] = None

        # State
        self._bg_gray = None  # type: ignore
        self._bg_subtractor = None  # type: ignore
        self._morph_kernel = None  # type: ignore
        self._candidate_rect: Optional[Tuple[int, int, int, int]] = None
        self._candidate_start_ts = 0.0
        self._suppress_until_ts = 0.0
        self._last_motion_rect: Optional[Tuple[int, int, int, int]] = None
        self._counter = 0

        # Metrics
        self._last_fg_pixels = 0
        self._last_largest_area = 0
        self._peak_largest_area = 0
        self._events_published = 0

    # --- EventDetector API ---
    def enabled(self) -> bool:
        return bool(self._enabled)

    def configure(self, **kwargs: Any) -> None:
        if 'enabled' in kwargs:
            self._enabled = bool(kwargs['enabled'])
        if 'min_area' in kwargs and kwargs['min_area'] is not None:
            try:
                self._min_area = max(0, int(kwargs['min_area']))
            except Exception:
                pass
        if 'alpha' in kwargs and kwargs['alpha'] is not None:
            try:
                a = float(kwargs['alpha'])
                if 0.0 < a <= 0.5:
                    self._alpha = a
            except Exception:
                pass
        if 'persist_ms' in kwargs and kwargs['persist_ms'] is not None:
            try:
                self._persist_ms = max(0, int(kwargs['persist_ms']))
            except Exception:
                pass
        if 'bg_mode' in kwargs and kwargs['bg_mode'] is not None:
            m = str(kwargs['bg_mode']).lower().strip()
            if m in ('avg', 'mog2', 'knn'):
                if m != self._bg_mode:
                    self._bg_mode = m
                    self._bg_subtractor = None
        if 'prefer_tracking' in kwargs and kwargs['prefer_tracking'] is not None:
            self._prefer_tracking = bool(kwargs['prefer_tracking'])
        if 'frame_skip' in kwargs and kwargs['frame_skip'] is not None:
            try:
                self._frame_skip = max(0, int(kwargs['frame_skip']))
            except Exception:
                pass
        if 'scale' in kwargs and kwargs['scale'] is not None:
            try:
                s = float(kwargs['scale'])
                if 0.1 <= s <= 1.0:
                    self._scale = s
            except Exception:
                pass

        if self._enabled:
            self._bg_gray = None
            self._bg_subtractor = None
            self._last_motion_rect = None
            self._candidate_rect = None
            self._candidate_start_ts = 0.0

    def set_zone(self, zone: Optional[Sequence[float]]) -> None:
        if zone is None:
            self._zone = None
            return
        try:
            x, y, w, h = [float(v) for v in zone]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            w = max(0.0, min(1.0 - x, w))
            h = max(0.0, min(1.0 - y, h))
            self._zone = (x, y, w, h)
        except Exception:
            pass

    def get_zone(self) -> Optional[Tuple[float, float, float, float]]:
        if isinstance(self._zone, tuple) and len(self._zone) == 4:
            return (float(self._zone[0]), float(self._zone[1]), float(self._zone[2]), float(self._zone[3]))
        return None

    def suppress(self, duration_sec: float) -> None:
        try:
            d = float(duration_sec)
        except Exception:
            d = 0.5
        self._suppress_until_ts = time.time() + max(0.0, d)

    def info(self, frame_size: Tuple[int, int]) -> Dict[str, Any]:
        w, h = frame_size
        rect = tuple(int(v) for v in self._last_motion_rect) if self._last_motion_rect else None
        info = {
            'enabled': bool(self._enabled),
            'rect': rect,
            'center': None,
            'u': None,
            'v': None,
            'width': int(w) if w else None,
            'height': int(h) if h else None,
            'fg_pixels': int(self._last_fg_pixels),
            'largest_area': int(self._last_largest_area),
            'peak_largest_area': int(self._peak_largest_area),
        }
        if rect is not None and w and h:
            x, y, rw, rh = rect
            cx = x + rw / 2.0
            cy = y + rh / 2.0
            info['center'] = (cx, cy)
            try:
                info['u'] = float(cx) / float(w)
                info['v'] = float(cy) / float(h)
            except Exception:
                info['u'] = None
                info['v'] = None
        return info

    def config(self) -> Dict[str, Any]:
        return {
            'enabled': bool(self._enabled),
            'min_area': int(self._min_area),
            'alpha': float(self._alpha),
            'persist_ms': int(self._persist_ms),
            'bg_mode': str(self._bg_mode),
            'prefer_tracking': bool(self._prefer_tracking),
            'frame_skip': int(self._frame_skip),
            'scale': float(self._scale),
            'zone': tuple(self._zone) if isinstance(self._zone, tuple) else None,
        }

    def reset_metrics(self) -> None:
        self._peak_largest_area = 0
        self._events_published = 0

    # --- Core processing ---
    def process(self, frame: Any, now_ts: Optional[float] = None) -> DetectionResult:
        if now_ts is None:
            now_ts = time.time()
        if not self._enabled:
            return DetectionResult(frame=frame, events=[], metrics=self._metrics())

        # Only run motion detection every Nth frame
        counter = getattr(self, '_counter', 0)
        do_motion = (counter % max(1, int(self._frame_skip) + 1)) == 0
        try:
            self._counter = counter + 1
        except Exception:
            pass

        events: List[DetectionEvent] = []
        candidates: List[Tuple[int, int, int, int, int]] = []

        if do_motion:
            # Downscale for faster processing
            try:
                s = float(self._scale)
            except Exception:
                s = 0.5
            s = 0.5 if not (0.1 <= s <= 1.0) else s
            small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            thresh = None
            if self._bg_mode == 'mog2':
                if self._bg_subtractor is None:
                    self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
                fg = self._bg_subtractor.apply(gray, learningRate=float(self._alpha))
                _, thresh = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)
            elif self._bg_mode == 'knn':
                if self._bg_subtractor is None:
                    self._bg_subtractor = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=400.0, detectShadows=True)
                fg = self._bg_subtractor.apply(gray, learningRate=float(self._alpha))
                _, thresh = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)
            else:
                if self._bg_gray is None or getattr(self._bg_gray, 'shape', None) != gray.shape:
                    self._bg_gray = gray.copy().astype('float')
                cv2.accumulateWeighted(gray, self._bg_gray, float(self._alpha))
                bg_uint8 = cv2.convertScaleAbs(self._bg_gray)
                delta = cv2.absdiff(bg_uint8, gray)
                _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

            # Morphological cleanup
            kernel = self._morph_kernel
            if kernel is None:
                try:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    self._morph_kernel = kernel
                except Exception:
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            try:
                fg_small = int(cv2.countNonZero(thresh))
            except Exception:
                fg_small = 0
            min_area_small = int(float(self._min_area) * (s * s))
            for c in cnts:
                if c is None:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                area = w * h
                if area >= int(min_area_small):
                    X = int(round(x / s))
                    Y = int(round(y / s))
                    W = int(round(w / s))
                    H = int(round(h / s))
                    candidates.append((X, Y, W, H, int(round(area / (s * s)))))

            # Optional zone filtering
            zone = self._zone
            if zone and isinstance(zone, tuple) and len(zone) == 4:
                try:
                    fh, fw = frame.shape[:2]
                    zx = int(max(0, min(fw, round(float(zone[0]) * fw))))
                    zy = int(max(0, min(fh, round(float(zone[1]) * fh))))
                    zw = int(max(0, min(fw - zx, round(float(zone[2]) * fw))))
                    zh = int(max(0, min(fh - zy, round(float(zone[3]) * fh))))
                    def inside(px: int, py: int) -> bool:
                        return (zx <= px <= (zx + zw)) and (zy <= py <= (zy + zh))
                    filtered: List[Tuple[int, int, int, int, int]] = []
                    for X, Y, W, H, A in candidates:
                        cx = X + W // 2
                        cy = Y + H // 2
                        if inside(cx, cy):
                            filtered.append((X, Y, W, H, A))
                    candidates = filtered
                    # Draw zone overlay for visualization
                    try:
                        cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 0, 0), 2)
                    except Exception:
                        pass
                except Exception:
                    pass

            # Update metrics
            try:
                self._last_fg_pixels = int(round(fg_small / (s * s)))
            except Exception:
                self._last_fg_pixels = fg_small
            try:
                cur_largest = int(max([r[4] for r in candidates])) if candidates else 0
                if cur_largest > 0:
                    self._last_largest_area = cur_largest
                    if cur_largest > self._peak_largest_area:
                        self._peak_largest_area = cur_largest
            except Exception:
                pass

        # Choose best candidate
        best: Optional[Tuple[int, int, int, int]] = None
        def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            ax2, ay2 = ax + aw, ay + ah
            bx2, by2 = bx + bw, by + bh
            inter_w = max(0, min(ax2, bx2) - max(ax, bx))
            inter_h = max(0, min(ay2, by2) - max(ay, by))
            inter = inter_w * inter_h
            union = aw * ah + bw * bh - inter
            return (inter / union) if union > 0 else 0.0

        prev = self._last_motion_rect or self._candidate_rect
        if candidates:
            if self._prefer_tracking and prev is not None:
                best_by_iou = max(candidates, key=lambda r: _iou((r[0], r[1], r[2], r[3]), prev))
                if _iou((best_by_iou[0], best_by_iou[1], best_by_iou[2], best_by_iou[3]), prev) >= 0.05:
                    best = (best_by_iou[0], best_by_iou[1], best_by_iou[2], best_by_iou[3])
                else:
                    best_largest = max(candidates, key=lambda r: r[4])
                    best = (best_largest[0], best_largest[1], best_largest[2], best_largest[3])
            else:
                best_largest = max(candidates, key=lambda r: r[4])
                best = (best_largest[0], best_largest[1], best_largest[2], best_largest[3])

        # Temporal persistence and suppression
        if best is None:
            self._candidate_rect = None
            self._candidate_start_ts = 0.0
            self._last_motion_rect = None
        else:
            if int(self._persist_ms) <= 0 and now_ts >= self._suppress_until_ts:
                self._last_motion_rect = best
                x, y, w, h = best
                try:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                except Exception:
                    pass
                evt = self._make_event(best, frame, now_ts)
                events.append(evt)
                self._events_published += 1
            else:
                if self._candidate_rect is None or _iou(best, self._candidate_rect) < 0.3:
                    self._candidate_rect = best
                    self._candidate_start_ts = now_ts
                persisted = (now_ts - self._candidate_start_ts) * 1000.0
                if persisted >= float(self._persist_ms) and now_ts >= self._suppress_until_ts:
                    self._last_motion_rect = best
                    x, y, w, h = best
                    try:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    except Exception:
                        pass
                    evt = self._make_event(best, frame, now_ts)
                    events.append(evt)
                    self._events_published += 1
                else:
                    # Show a yellow box while accumulating persistence
                    x, y, w, h = best
                    try:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                    except Exception:
                        pass

        return DetectionResult(frame=frame, events=events, metrics=self._metrics())

    # --- helpers ---
    def _metrics(self) -> Dict[str, Any]:
        return {
            'fg_pixels': int(self._last_fg_pixels),
            'largest_area': int(self._last_largest_area),
            'peak_largest_area': int(self._peak_largest_area),
            'events_published': int(self._events_published),
        }

    def _make_event(self, rect: Tuple[int, int, int, int], frame: Any, ts: float) -> DetectionEvent:
        x, y, w, h = rect
        cx = x + w / 2.0
        cy = y + h / 2.0
        # Note: u/v are calculated by caller based on camera size; leave None here
        return DetectionEvent(
            ts=ts,
            rect=(int(x), int(y), int(w), int(h)),
            center=(float(cx), float(cy)),
            u=None, v=None,
            width=None, height=None,
            extra={'fg_pixels': int(self._last_fg_pixels), 'largest_area': int(self._last_largest_area)},
        )

