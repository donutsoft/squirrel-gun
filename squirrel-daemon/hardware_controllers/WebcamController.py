from pathlib import Path
from typing import Optional, Iterator, Callable, Any
import time
import threading
import cv2  # type: ignore

class WebcamController:
    def __init__(self, device: str = "/dev/video0", width: Optional[int] = None, height: Optional[int] = None):
        self.device = device
        self.width = 1280
        self.height = 720
        self._fps = 15
        self._quality = 80
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: bytes = b""
        self._seq = 0
        self._cond = threading.Condition()
        # Motion detection settings/state
        self._motion_enabled = True
        self._motion_min_area = 800  # pixels in full-res image
        self._motion_alpha = 0.05    # background learning rate
        self._bg_gray = None         # type: ignore
        self._last_motion_rect = None  # type: ignore
        # Temporal persistence and suppression
        self._persist_ms = 0  # 0 disables persistence; report immediately
        self._candidate_rect = None  # type: ignore
        self._candidate_start_ts = 0.0
        self._suppress_until_ts = 0.0
        self._publish: Optional[Callable[[str, Any], None]] = None
        # Background subtraction / selection strategy
        self._bg_mode = 'mog2'  # one of: 'avg', 'mog2', 'knn'
        self._bg_subtractor = None  # type: ignore
        # Prefer to follow previous target over raw largest blob
        self._prefer_tracking = True
        # Performance controls
        self._motion_frame_skip = 2   # compute motion every N+1 frames (2 => ~5 Hz @ 15 fps)
        self._motion_scale = 0.5      # process motion at half resolution

    def _device_index(self) -> int:
        index = 0
        if isinstance(self.device, str) and self.device.startswith("/dev/video"):
            try:
                index = int(self.device.replace("/dev/video", ""))
            except ValueError:
                index = 0
        return index

    def _open_capture(self) -> "cv2.VideoCapture":
        """Open the camera with settings that favor low latency.

        - Request V4L2 backend when available.
        - Set small internal buffer to avoid buildup.
        - Prefer MJPG fourcc to reduce encode latency on UVC cams.
        """
        try:
            cap = cv2.VideoCapture(self._device_index(), cv2.CAP_V4L2)
        except Exception:
            cap = cv2.VideoCapture(self._device_index())

        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        # Hint the backend to keep a tiny buffer (best-effort; some backends ignore it)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        # Prefer MJPG if the camera supports it; this often drops latency on UVC devices
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass
        return cap

    def _capture_with_opencv(self, outfile: Path) -> bool:
        cap = self._open_capture()

        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False

        # Write as JPEG
        ok = cv2.imwrite(str(outfile), frame)
        return bool(ok)

    def capture(self, outfile: Path) -> Path:
        outfile = outfile.resolve()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        # Fast path: if we already have a recent frame, use it without locking
        data: Optional[bytes] = self._latest if self._latest else None
        if data:
            with open(outfile, 'wb') as f:
                f.write(data)
            return outfile
        self._capture_with_opencv(outfile)
        return outfile

    def start_stream(self, fps: int = 15, quality: int = 80) -> None:
        self._fps = max(1, int(fps))
        self._quality = max(1, min(100, int(quality)))
        if self._running and self._thread and self._thread.is_alive():
            return
        self._running = True

        def _producer() -> None:
            # Emit at configured FPS, but always publish the freshest frame.
            # Reads run in a tight loop; only latest frame at each tick is encoded.
            target_dt = 1.0 / self._fps
            next_emit = time.monotonic()
            cap = self._open_capture()
            failures = 0
            last_frame = None
            try:
                while self._running:
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        # Optional: motion detection and overlay
                        if self._motion_enabled:
                            try:
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                                # Only run motion detection every Nth frame and on downscaled image
                                try:
                                    counter = getattr(self, '_motion_counter')
                                except Exception:
                                    counter = 0
                                do_motion = (counter % max(1, int(self._motion_frame_skip) + 1)) == 0
                                try:
                                    setattr(self, '_motion_counter', counter + 1)
                                except Exception:
                                    pass
                                candidates = []
                                if do_motion:
                                    try:
                                        s = float(self._motion_scale)
                                    except Exception:
                                        s = 0.5
                                    s = 0.5 if not (0.1 <= s <= 1.0) else s
                                    small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                                    # Compute foreground mask based on configured background model
                                    thresh = None
                                    if self._bg_mode == 'mog2':
                                        if self._bg_subtractor is None:
                                            # Detect shadows; treat 255 as foreground later
                                            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
                                        fg = self._bg_subtractor.apply(small, learningRate=float(self._motion_alpha))
                                        # Remove shadows (value 127) by thresholding to 255 only
                                        _, thresh = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)
                                    elif self._bg_mode == 'knn':
                                        if self._bg_subtractor is None:
                                            self._bg_subtractor = cv2.createBackgroundSubtractorKNN(history=400, dist2Threshold=400.0, detectShadows=True)
                                        fg = self._bg_subtractor.apply(small, learningRate=float(self._motion_alpha))
                                        _, thresh = cv2.threshold(fg, 254, 255, cv2.THRESH_BINARY)
                                    else:
                                        # Running average background (simple, low-cost)
                                        if self._bg_gray is None or getattr(self._bg_gray, 'shape', None) != small.shape:
                                            self._bg_gray = small.copy().astype('float')
                                        cv2.accumulateWeighted(small, self._bg_gray, float(self._motion_alpha))
                                        bg_uint8 = cv2.convertScaleAbs(self._bg_gray)
                                        delta = cv2.absdiff(bg_uint8, small)
                                        _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)

                                    # Morphological cleanup: remove speckles and fill small gaps
                                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=1)

                                    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    min_area_small = int(float(self._motion_min_area) * (s * s))
                                    for c in cnts:
                                        if c is None:
                                            continue
                                        x, y, w, h = cv2.boundingRect(c)
                                        area = w * h
                                        if area >= max(1, int(min_area_small)):
                                            # Scale back to full-res coordinates
                                            X = int(round(x / s))
                                            Y = int(round(y / s))
                                            W = int(round(w / s))
                                            H = int(round(h / s))
                                            candidates.append((X, Y, W, H, int(round(area / (s * s)))))

                                best = None
                                # Choose candidate: prefer overlap with previous to "follow" target
                                def _iou(a, b) -> float:
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
                                        # Pick with maximum IoU to previous; fallback to largest area if overlap tiny
                                        best_by_iou = max(candidates, key=lambda r: _iou((r[0], r[1], r[2], r[3]), prev))
                                        if _iou((best_by_iou[0], best_by_iou[1], best_by_iou[2], best_by_iou[3]), prev) >= 0.05:
                                            best = (best_by_iou[0], best_by_iou[1], best_by_iou[2], best_by_iou[3])
                                        else:
                                            best_largest = max(candidates, key=lambda r: r[4])
                                            best = (best_largest[0], best_largest[1], best_largest[2], best_largest[3])
                                    else:
                                        best_largest = max(candidates, key=lambda r: r[4])
                                        best = (best_largest[0], best_largest[1], best_largest[2], best_largest[3])
                                # Temporal persistence: optionally require stable presence before reporting
                                now_ts = time.time()

                                if best is None:
                                    self._candidate_rect = None
                                    self._candidate_start_ts = 0.0
                                    self._last_motion_rect = None
                                else:
                                    # If persistence disabled, emit immediately (subject to suppression)
                                    if int(self._persist_ms) <= 0 and now_ts >= self._suppress_until_ts:
                                        self._last_motion_rect = best
                                        x, y, w, h = best
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                        if self._publish is not None and self.width and self.height:
                                            cx = x + w / 2.0
                                            cy = y + h / 2.0
                                            try:
                                                u = float(cx) / float(self.width)
                                                v = float(cy) / float(self.height)
                                            except Exception:
                                                u, v = None, None
                                            evt = {
                                                'ts': now_ts,
                                                'rect': (int(x), int(y), int(w), int(h)),
                                                'center': (float(cx), float(cy)),
                                                'u': u, 'v': v,
                                                'width': int(self.width), 'height': int(self.height),
                                            }
                                            try:
                                                self._publish('motion', evt)
                                            except Exception:
                                                pass
                                    else:
                                        if self._candidate_rect is None or _iou(best, self._candidate_rect) < 0.3:
                                            self._candidate_rect = best
                                            self._candidate_start_ts = now_ts
                                        # Only accept as motion if persisted long enough and not suppressed
                                        persisted = (now_ts - self._candidate_start_ts) * 1000.0
                                        if persisted >= float(self._persist_ms) and now_ts >= self._suppress_until_ts:
                                            self._last_motion_rect = best
                                            x, y, w, h = best
                                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                            # Publish motion event (non-blocking) if a bus is set
                                            if self._publish is not None and self.width and self.height:
                                                cx = x + w / 2.0
                                                cy = y + h / 2.0
                                                try:
                                                    u = float(cx) / float(self.width)
                                                    v = float(cy) / float(self.height)
                                                except Exception:
                                                    u, v = None, None
                                                evt = {
                                                    'ts': now_ts,
                                                    'rect': (int(x), int(y), int(w), int(h)),
                                                    'center': (float(cx), float(cy)),
                                                    'u': u, 'v': v,
                                                    'width': int(self.width), 'height': int(self.height),
                                                }
                                                try:
                                                    self._publish('motion', evt)
                                                except Exception:
                                                    pass
                                        else:
                                            # Show a yellow box while accumulating persistence (optional visual feedback)
                                            x, y, w, h = best
                                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
                            except Exception:
                                # If OpenCV processing fails for any reason, continue without overlay
                                pass
                        last_frame = frame
                        failures = 0
                    else:
                        failures += 1
                        time.sleep(0.005)
                        if failures >= 40:
                            cap.release()
                            cap = self._open_capture()
                            failures = 0
                        # Even on failure, check if it's time to emit (we'll skip if no frame)

                    now = time.monotonic()
                    if now >= next_emit:
                        if last_frame is not None:
                            ok2, encoded = cv2.imencode(
                                '.jpg', last_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
                            )
                            if ok2:
                                jpg = encoded.tobytes()
                                with self._cond:
                                    self._latest = jpg
                                    self._seq += 1
                                    self._cond.notify_all()
                        # Schedule next emit; if we're behind, skip ahead without sleeping
                        next_emit += target_dt
                        # Avoid runaway if system time drifted or we were paused
                        if now - next_emit > 2 * target_dt:
                            next_emit = now + target_dt

                    # Light backoff to avoid pegging CPU while still keeping latency low
                    time.sleep(0.001)
            finally:
                cap.release()

        self._thread = threading.Thread(target=_producer, name="WebcamProducer", daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def mjpeg(self, fps: int = 15, quality: int = 80, boundary: str = "frame") -> Iterator[bytes]:
        self.start_stream(fps=fps, quality=quality)
        delay = 1.0 / self._fps
        last_seq = -1
        while True:
            with self._cond:
                if self._seq == last_seq:
                    self._cond.wait(timeout=delay)
                data = self._latest
                last_seq = self._seq
            if not data:
                time.sleep(delay)
                continue
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")

    

    def motion_info(self) -> dict:
        """Return the latest motion rectangle and normalized center.

        Returns a dict with keys: enabled, rect, center, u, v, width, height.
        If no motion is detected or not enabled, rect/center/u/v are None.
        """
        rect = None
        with self._cond:
            if self._last_motion_rect is not None:
                rect = tuple(int(v) for v in self._last_motion_rect)
        info = {
            'enabled': bool(self._motion_enabled),
            'rect': rect,
            'center': None,
            'u': None,
            'v': None,
            'width': int(self.width) if self.width else None,
            'height': int(self.height) if self.height else None,
        }
        if rect is not None and self.width and self.height:
            x, y, w, h = rect
            cx = x + w / 2.0
            cy = y + h / 2.0
            info['center'] = (cx, cy)
            try:
                info['u'] = float(cx) / float(self.width)
                info['v'] = float(cy) / float(self.height)
            except Exception:
                info['u'] = None
                info['v'] = None
        return info

    def motion_config(self) -> dict:
        """Return current motion detector configuration."""
        return {
            'enabled': bool(self._motion_enabled),
            'min_area': int(self._motion_min_area),
            'alpha': float(self._motion_alpha),
            'persist_ms': int(self._persist_ms),
            'bg_mode': str(self._bg_mode),
            'prefer_tracking': bool(self._prefer_tracking),
            'frame_skip': int(getattr(self, '_motion_frame_skip', 0)),
            'scale': float(getattr(self, '_motion_scale', 1.0)),
        }

    def set_motion_publisher(self, publish: Optional[Callable[[str, Any], None]]) -> None:
        """Set a callback to publish motion events with signature (topic, data)."""
        self._publish = publish

    def suppress_motion(self, duration_sec: float = 0.5) -> None:
        try:
            d = float(duration_sec)
        except Exception:
            d = 0.5
        self._suppress_until_ts = time.time() + max(0.0, d)

    # Public controls for motion detection
    def set_motion_detection(self, enabled: bool, min_area: Optional[int] = None, alpha: Optional[float] = None, persist_ms: Optional[int] = None, bg_mode: Optional[str] = None, prefer_tracking: Optional[bool] = None, frame_skip: Optional[int] = None, scale: Optional[float] = None) -> None:
        self._motion_enabled = bool(enabled)
        if min_area is not None:
            try:
                self._motion_min_area = max(1, int(min_area))
            except Exception:
                pass
        if alpha is not None:
            try:
                a = float(alpha)
                # keep sane bounds for learning rate
                if 0.0 < a <= 0.5:
                    self._motion_alpha = a
            except Exception:
                pass
        if persist_ms is not None:
            try:
                self._persist_ms = max(0, int(persist_ms))
            except Exception:
                pass
        if bg_mode is not None:
            m = str(bg_mode).lower().strip()
            if m in ('avg', 'mog2', 'knn'):
                self._bg_mode = m
            # Reset background subtractor instance when mode changes
            self._bg_subtractor = None
        if prefer_tracking is not None:
            try:
                self._prefer_tracking = bool(prefer_tracking)
            except Exception:
                pass
        if frame_skip is not None:
            try:
                self._motion_frame_skip = max(0, int(frame_skip))
            except Exception:
                pass
        if scale is not None:
            try:
                s = float(scale)
                if 0.1 <= s <= 1.0:
                    self._motion_scale = s
            except Exception:
                pass
        # Reset background model when toggling on to avoid stale state
        if self._motion_enabled:
            self._bg_gray = None
            self._bg_subtractor = None
            self._last_motion_rect = None
            self._candidate_rect = None
            self._candidate_start_ts = 0.0
