from pathlib import Path
from typing import Optional, Iterator, Callable, Any, Union, Tuple, List, Dict
from collections import deque
import time
import threading
import cv2  # type: ignore
from turbojpeg import TurboJPEG, TJPF_BGR, TJSAMP_420  # type: ignore
from event_detection.motion import MotionDetector
from event_detection.yolo import YOLOEventDetector

class WebcamController:
    def __init__(self, device: str = "/dev/video0", width: Optional[int] = None, height: Optional[int] = None):
        self.device = device
        self.width = 1280
        self.height = 720
        self._fps = 15
        # Slightly lower default JPEG quality to reduce encode latency
        self._quality = 70
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest: bytes = b""
        self._seq = 0
        self._cond = threading.Condition()
        # TurboJPEG encoder instance (required)
        self._tj = TurboJPEG()
        # Number of active MJPEG consumers
        self._subscribers = 0
        # Active event detector
        self._detector_type = 'motion'
        self._detector = MotionDetector()
        # Store motion zone centrally; applied only to motion detector
        self._zone: Optional[Tuple[float, float, float, float]] = None
        # Low-latency streaming mode flag (default ON)
        self._low_latency = True
        # Preview downscale factor for low-latency mode (encode fewer pixels)
        self._preview_scale = 0.5  # 0.1..1.0
        self._publish: Optional[Callable[[str, Any], None]] = None

        # Diagnostics for motion → recording path
        self._motion_events_published = 0
        self._motion_triggers_received = 0

        # Recording state and config
        from pathlib import Path as _P
        self._recordings_dir = _P(__file__).resolve().parents[1] / 'static' / 'recordings'
        try:
            self._recordings_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Screenshots directory (per-motion snapshot)
        self._snapshots_dir = self._recordings_dir / 'shots'
        try:
            self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # No internal auto-record on motion; use event bus subscription instead
        self._record_on_motion_enabled = True
        self._record_duration_sec = 30.0
        self._snapshot_on_motion_enabled = True
        self._recording_active = False
        self._recording_end_ts = 0.0
        self._video_writer = None  # type: ignore
        self._recording_path: Optional[_P] = None  # type: ignore
        self._recording_lock = threading.Lock()
        self._bus = None  # type: ignore
        # Preferred recording FPS (writer). Decoupled from streaming FPS.
        self._record_fps = 15

        # Pre-roll buffer for recording N seconds before trigger
        self._preroll_sec: float = 5.0
        self._preroll_enabled: bool = True
        # Store tuples of (ts, frame_bgr). Using uncompressed frames for speed.
        # Note: 1280x720x3 @ 15fps for 5s ≈ ~200MB peak usage.
        self._preroll: deque[Tuple[float, Any]] = deque()
        self._preroll_lock = threading.Lock()
        self._preroll_flush_done = False

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
        # Hint desired FPS to reduce internal buffering when possible
        try:
            cap.set(cv2.CAP_PROP_FPS, float(self._fps))
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

        # Encode using TurboJPEG and write to file
        try:
            jpg = self._tj.encode(frame, quality=int(self._quality), pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420)
            with open(outfile, 'wb') as f:
                f.write(jpg)
            return True
        except Exception:
            return False

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

    def _open_video_writer(self, path: Path, fps: int, frame_size: tuple[int, int]):
        """Open a VideoWriter with sane cross-platform defaults.

        Default priority is 'mp4v' (widely available) then 'avc1'/'H264'.
        Override order via env var 'SQ_VIDEO_FOURCCS' (comma-separated tags).
        Returns an opened writer or None on failure.
        """
        fpsf = float(max(1, fps))
        w, h = int(frame_size[0]), int(frame_size[1])
        # Ensure even dimensions for YUV420 encoders (avoid green artifacts)
        if (w % 2) != 0 or (h % 2) != 0:
            w -= (w % 2)
            h -= (h % 2)
            frame_size = (w, h)
        # Codec preference (env override supported)
        import os
        env_order = os.getenv('SQ_VIDEO_FOURCCS', '').strip()
        if env_order:
            order = [t.strip() for t in env_order.split(',') if t.strip()]
        else:
            # Prefer mp4v first to avoid hardware H.264 encoder errors on systems
            # without h264 encoders (e.g., missing v4l2m2m device in containers)
            order = ['mp4v', 'avc1', 'H264']
        for fourcc_tag in order:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_tag)
                writer = cv2.VideoWriter(str(path), fourcc, fpsf, frame_size)
                if writer is not None and getattr(writer, 'isOpened', lambda: True)():
                    return writer
                try:
                    writer.release()
                except Exception:
                    pass
            except Exception:
                pass
        return None

    def save_snapshot(self, out_path: Optional[Path] = None) -> Optional[Path]:
        """Save the latest available JPEG frame to a file.

        If out_path is None, write to `static/recordings/shots/snap_YYYYmmDD_HHMMSSfff.jpg`.
        Returns the path on success, or None if no frame available.
        """
        # Get latest in-memory JPEG if available
        data: Optional[bytes] = None
        with self._cond:
            if self._latest:
                data = bytes(self._latest)

        ts = time.time()
        if out_path is None:
            fname = time.strftime('snap_%Y%m%d_%H%M%S', time.localtime(ts)) + f"{int((ts % 1) * 1000):03d}.jpg"
            out_path = (self._snapshots_dir / fname)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if data:
            try:
                with open(out_path, 'wb') as f:
                    f.write(data)
                return out_path
            except Exception:
                return None
        # Fallback: capture a fresh frame directly
        try:
            ok = self._capture_with_opencv(out_path)
            return out_path if ok else None
        except Exception:
            return None

    def start_recording(self, duration_sec: float = 30.0, extend: bool = True) -> Optional[Path]:
        """Start recording for duration_sec seconds.

        If a recording is already active and extend is True, extend the end time.
        Returns the output file path for a new recording, else None.
        """
        out_path: Optional[Path] = None
        with self._recording_lock:
            now = time.time()
            if self._recording_active:
                if extend:
                    self._recording_end_ts = max(self._recording_end_ts, now + max(0.1, float(duration_sec)))
                return None
            ts = time.strftime('%Y%m%d_%H%M%S')
            out_path = self._recordings_dir / f'rec_{ts}.mp4'
            self._recording_path = out_path
            self._recording_active = True
            self._recording_end_ts = now + max(0.1, float(duration_sec))
            # Ensure preroll will be flushed at start of this new recording
            self._preroll_flush_done = False
            # Close any previous writer defensively
            try:
                if self._video_writer is not None:
                    self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None
            # OpenCV writer will be initialized lazily on first frame
        return out_path

    def stop_recording(self) -> None:
        with self._recording_lock:
            self._recording_active = False
            self._recording_end_ts = 0.0
            try:
                if self._video_writer is not None:
                    self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None
            self._recording_path = None
            # Reset flush flag so next recording can include pre-roll
            self._preroll_flush_done = False

    def is_recording(self) -> bool:
        with self._recording_lock:
            return bool(self._recording_active and time.time() < self._recording_end_ts)

    # External integration: listen to event bus for motion events
    def set_event_bus(self, bus: Any, record_on_motion: bool = True, duration_sec: float = 30.0) -> None:
        """Subscribe to motion events on the provided bus to trigger recording.

        The bus must support subscribe(topic, handler) and will receive 'motion' events.
        """
        self._bus = bus
        self._record_on_motion_enabled = bool(record_on_motion)
        try:
            self._record_duration_sec = float(duration_sec)
        except Exception:
            self._record_duration_sec = 30.0

        def _on_motion(evt: Any) -> None:
            if not self._record_on_motion_enabled:
                return
            try:
                self._motion_triggers_received += 1
                # Start or extend recording; take a snapshot only on new start
                started_path = self.start_recording(duration_sec=float(getattr(self, '_record_duration_sec', 30.0)), extend=True)
                if started_path is not None and bool(getattr(self, '_snapshot_on_motion_enabled', True)):
                    try:
                        # Name snapshot using the same timestamp as the recording
                        rec_name = getattr(started_path, 'name', '')
                        if rec_name.startswith('rec_') and rec_name.endswith('.mp4') and len(rec_name) >= 4 + 15 + 4:
                            ts = rec_name[len('rec_'):-len('.mp4')]
                            shot_path = self._snapshots_dir / f'snap_{ts}.jpg'
                        else:
                            ts = time.strftime('%Y%m%d_%H%M%S')
                            shot_path = self._snapshots_dir / f'snap_{ts}.jpg'
                        # Prefer the exact triggering frame if present on the event
                        data = None
                        try:
                            data = evt.get('jpeg') if isinstance(evt, dict) else None
                        except Exception:
                            data = None
                        if data:
                            try:
                                shot_path.parent.mkdir(parents=True, exist_ok=True)
                                with open(shot_path, 'wb') as f:
                                    f.write(data)
                            except Exception:
                                # Fallback to grabbing latest frame if writing bytes failed
                                self.save_snapshot(out_path=shot_path)
                        else:
                            # Fallback to latest available
                            self.save_snapshot(out_path=shot_path)
                    except Exception:
                        pass
            except Exception:
                pass

        # Subscribe for motion events to trigger recording
        try:
            if hasattr(bus, 'subscribe'):
                bus.subscribe('motion', _on_motion)
        except Exception:
            pass

    # Recording config helpers
    def get_recording_config(self) -> dict:
        return {
            'record_on_motion': bool(self._record_on_motion_enabled),
            'duration_sec': float(self._record_duration_sec),
            'snapshot_on_motion': bool(self._snapshot_on_motion_enabled),
            'preroll_sec': float(getattr(self, '_preroll_sec', 0.0)),
        }

    def set_recording_config(self, record_on_motion: Optional[bool] = None, duration_sec: Optional[float] = None, snapshot_on_motion: Optional[bool] = None, preroll_sec: Optional[float] = None) -> None:
        if record_on_motion is not None:
            self._record_on_motion_enabled = bool(record_on_motion)
        if duration_sec is not None:
            try:
                self._record_duration_sec = max(1.0, float(duration_sec))
            except Exception:
                pass
        if snapshot_on_motion is not None:
            self._snapshot_on_motion_enabled = bool(snapshot_on_motion)
        if preroll_sec is not None:
            try:
                p = float(preroll_sec)
                if p <= 0:
                    self._preroll_sec = 0.0
                    self._preroll_enabled = False
                else:
                    self._preroll_sec = min(30.0, p)  # cap to 30s for safety
                    self._preroll_enabled = True
            except Exception:
                pass


    def start_stream(self, fps: int = 15, quality: int = 80) -> None:
        # In low-latency mode with motion off, bias for lower encode cost
        if bool(getattr(self, '_low_latency', False)) and not bool(self._detector.enabled()):
            fps = min(int(fps), 12)
            quality = min(int(quality), 70)
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
                        raw_frame = frame
                        # Append to pre-roll buffer as raw BGR frame (uncompressed)
                        try:
                            if bool(getattr(self, '_preroll_enabled', False)) and float(getattr(self, '_preroll_sec', 0.0)) > 0.0:
                                ts_now = time.time()
                                with self._preroll_lock:
                                    # Store a copy to decouple from OpenCV's reused buffers
                                    self._preroll.append((ts_now, raw_frame.copy()))
                                    # Drop frames older than preroll window
                                    cutoff = ts_now - float(getattr(self, '_preroll_sec', 0.0))
                                    while self._preroll and self._preroll[0][0] < cutoff:
                                        self._preroll.popleft()
                                    # Hard cap to avoid unbounded memory in case of FPS spikes
                                    max_frames = int(max(1, self._fps) * max(1.0, float(getattr(self, '_preroll_sec', 0.0)) * 2.0))
                                    while len(self._preroll) > max_frames:
                                        self._preroll.popleft()
                        except Exception:
                            pass
                        # Run motion detector (draw overlays within detector)
                        if self._detector.enabled():
                            try:
                                res = self._detector.process(raw_frame, now_ts=time.time())
                                frame = res.frame
                                if self._publish is not None and res.events:
                                    for e in res.events:
                                        # Compute normalized coords
                                        if self.width and self.height and e.center is not None:
                                            try:
                                                cx, cy = e.center
                                                u = float(cx) / float(self.width)
                                                v = float(cy) / float(self.height)
                                            except Exception:
                                                u, v = None, None
                                        else:
                                            u = v = None
                                        evt = {
                                            'ts': e.ts,
                                            'rect': e.rect,
                                            'center': e.center,
                                            'u': u, 'v': v,
                                            'width': int(self.width), 'height': int(self.height),
                                        }
                                        # Include metrics if present
                                        if e.extra:
                                            evt.update(e.extra)
                                        # Attach JPEG of the overlay frame for fidelity
                                        try:
                                            evt['jpeg'] = self._tj.encode(frame, quality=int(self._quality), pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420)
                                        except Exception:
                                            pass
                                        try:
                                            self._motion_events_published += 1
                                            self._publish('motion', evt)
                                        except Exception:
                                            pass
                            except Exception:
                                frame = raw_frame
                        else:
                            frame = raw_frame
                        last_frame = frame
                        # Handle recording lifecycle and write frames
                        try:
                            with self._recording_lock:
                                nowt = time.time()
                                if self._recording_active and nowt >= self._recording_end_ts:
                                    # stop
                                    try:
                                        if self._video_writer is not None:
                                            self._video_writer.release()
                                    except Exception:
                                        pass
                                    self._video_writer = None
                                    self._recording_active = False
                                    self._recording_path = None
                                    self._preroll_flush_done = False
                                if self._recording_active:
                                    if self._video_writer is None:
                                        h, w = raw_frame.shape[:2]
                                        # Use camera-reported FPS when available; fall back to configured record FPS
                                        try:
                                            cam_fps = cap.get(cv2.CAP_PROP_FPS)
                                            fps_w = max(1, int(round(cam_fps))) if cam_fps and cam_fps > 0 else max(1, int(getattr(self, '_record_fps', 15)))
                                        except Exception:
                                            fps_w = max(1, int(getattr(self, '_record_fps', 15)))
                                        path = self._recording_path or (self._recordings_dir / f'rec_{time.strftime("%Y%m%d_%H%M%S")}.mp4')
                                        self._recording_path = path
                                        self._video_writer = self._open_video_writer(path, fps_w, (w, h))
                                        # On first open, flush pre-roll frames at start of clip
                                        if self._video_writer is not None and not bool(getattr(self, '_preroll_flush_done', False)):
                                            try:
                                                # Copy current buffer snapshot to avoid holding lock while decoding
                                                with self._preroll_lock:
                                                    preroll_items = list(self._preroll)
                                                if preroll_items:
                                                    # Determine writer size
                                                    try:
                                                        ws = int(getattr(self._video_writer, 'get', lambda *_: 0)(cv2.CAP_PROP_FRAME_WIDTH))
                                                        hs = int(getattr(self._video_writer, 'get', lambda *_: 0)(cv2.CAP_PROP_FRAME_HEIGHT))
                                                    except Exception:
                                                        ws, hs = w, h
                                                    for _, fr0 in preroll_items:
                                                        try:
                                                            fr = fr0
                                                            if ws and hs and (fr0.shape[1] != ws or fr0.shape[0] != hs):
                                                                fr = cv2.resize(fr0, (ws, hs), interpolation=cv2.INTER_LINEAR)
                                                            self._video_writer.write(fr)
                                                        except Exception:
                                                            continue
                                            finally:
                                                self._preroll_flush_done = True
                                    if self._video_writer is not None:
                                        try:
                                            # Resize to writer size if necessary to satisfy encoder requirements
                                            try:
                                                ws = int(getattr(self._video_writer, 'get', lambda *_: 0)(cv2.CAP_PROP_FRAME_WIDTH))
                                                hs = int(getattr(self._video_writer, 'get', lambda *_: 0)(cv2.CAP_PROP_FRAME_HEIGHT))
                                            except Exception:
                                                ws, hs = raw_frame.shape[1], raw_frame.shape[0]
                                            fr = raw_frame
                                            if ws and hs and (raw_frame.shape[1] != ws or raw_frame.shape[0] != hs):
                                                fr = cv2.resize(raw_frame, (ws, hs), interpolation=cv2.INTER_LINEAR)
                                            self._video_writer.write(fr)
                                        except Exception:
                                            pass
                        except Exception:
                            pass
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
                        if last_frame is not None and int(getattr(self, '_subscribers', 0)) > 0:
                            # Prepare frame for encoding (downscale in low-latency + motion-off mode)
                            frame_to_encode = last_frame
                            try:
                                if bool(getattr(self, '_low_latency', False)) and not bool(self._detector.enabled()):
                                    s = float(getattr(self, '_preview_scale', 1.0))
                                    if 0.1 <= s < 1.0:
                                        frame_to_encode = cv2.resize(last_frame, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                            except Exception:
                                frame_to_encode = last_frame

                            # Encode with TurboJPEG (required)
                            try:
                                jpg = self._tj.encode(frame_to_encode, quality=int(self._quality), pixel_format=TJPF_BGR, jpeg_subsample=TJSAMP_420)
                            except Exception:
                                jpg = None

                            if jpg is not None:
                                with self._cond:
                                    self._latest = jpg
                                    self._seq += 1
                                    self._cond.notify_all()
                        # Schedule next emit based on current time to prevent burst catch-up
                        next_emit = now + target_dt

                    # Light backoff: scale sleep by how far we are from next emit
                    slack = max(0.0, next_emit - now)
                    if slack > (0.5 * target_dt):
                        time.sleep(min(0.003, slack * 0.25))
                    else:
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
        # Track subscriber lifecycle
        try:
            try:
                self._subscribers += 1
            except Exception:
                pass
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
        finally:
            try:
                self._subscribers = max(0, int(getattr(self, '_subscribers', 0)) - 1)
            except Exception:
                pass

    

    def motion_info(self) -> dict:
        """Return active detector info and latest state/metrics."""
        return self._detector.info((int(self.width) if self.width else 0,
                                    int(self.height) if self.height else 0))

    def reset_motion_peak(self) -> None:
        self._detector.reset_metrics()

    def motion_config(self) -> dict:
        """Return current detector configuration."""
        return self._detector.config()

    # Diagnostics helpers
    def motion_counters(self) -> dict:
        return {
            'events_published': int(self._motion_events_published),
            'triggers_received': int(self._motion_triggers_received),
        }

    def reset_motion_counters(self) -> None:
        self._motion_events_published = 0
        self._motion_triggers_received = 0

    def set_motion_publisher(self, publish: Optional[Callable[[str, Any], None]]) -> None:
        """Set a callback to publish motion events with signature (topic, data)."""
        self._publish = publish

    def suppress_motion(self, duration_sec: float = 0.5) -> None:
        self._detector.suppress(float(duration_sec))

    # Low-latency streaming control
    def set_low_latency_mode(self, enabled: bool) -> None:
        self._low_latency = bool(enabled)
    def get_low_latency_mode(self) -> bool:
        return bool(getattr(self, '_low_latency', False))

    # Preview downscale control (for encoding only; recording stays full-res)
    def set_preview_scale(self, scale: float) -> None:
        try:
            s = float(scale)
            if 0.1 <= s <= 1.0:
                self._preview_scale = s
        except Exception:
            pass
    def get_preview_scale(self) -> float:
        try:
            return float(getattr(self, '_preview_scale', 1.0))
        except Exception:
            return 1.0

    # Public controls for motion detection
    def set_motion_detection(self, enabled: bool, min_area: Optional[int] = None, alpha: Optional[float] = None, persist_ms: Optional[int] = None, bg_mode: Optional[str] = None, prefer_tracking: Optional[bool] = None, frame_skip: Optional[int] = None, scale: Optional[float] = None) -> None:
        # Configure detector
        self._detector.configure(
            enabled=bool(enabled),
            min_area=min_area,
            alpha=alpha,
            persist_ms=persist_ms,
            bg_mode=bg_mode,
            prefer_tracking=prefer_tracking,
            frame_skip=frame_skip,
            scale=scale,
        )
        # Adjust streaming hints based on enabled flag
        try:
            if not bool(enabled):
                self._low_latency = True
                self._preview_scale = 0.5
            else:
                self._low_latency = bool(getattr(self, '_low_latency', False) and False)
                self._preview_scale = 1.0
        except Exception:
            pass

    # Motion zone controls (normalized rect x,y,w,h)
    def set_motion_zone(self, zone: Optional[Union[Tuple[float, float, float, float], List[float], Dict]]) -> None:
        if zone is None:
            self._zone = None
            # Apply to motion detector only
            if self._detector_type == 'motion':
                self._detector.set_zone(None)
            return
        # Normalize and persist
        if isinstance(zone, dict):
            x = float(zone.get('x', 0.0))
            y = float(zone.get('y', 0.0))
            w = float(zone.get('w', 0.0))
            h = float(zone.get('h', 0.0))
        else:
            x, y, w, h = [float(v) for v in zone]  # type: ignore
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        self._zone = (x, y, w, h)
        if self._detector_type == 'motion':
            self._detector.set_zone(self._zone)

    def motion_zone(self) -> Optional[tuple[float, float, float, float]]:
        z = self._zone
        if z is None:
            return None
        return (float(z[0]), float(z[1]), float(z[2]), float(z[3]))

    # Detector switching
    def set_detector_type(self, kind: str) -> str:
        kind = str(kind).lower().strip()
        if kind not in ('motion', 'yolo'):
            raise ValueError('invalid detector type')
        if kind == getattr(self, '_detector_type', 'motion'):
            return self._detector_type
        if kind == 'motion':
            det = MotionDetector()
            # Apply stored zone when enabling motion
            if self._zone is not None:
                det.set_zone(self._zone)
        else:
            det = YOLOEventDetector()
        self._detector = det
        self._detector_type = kind
        return self._detector_type

    def get_detector_type(self) -> str:
        return getattr(self, '_detector_type', 'motion')
