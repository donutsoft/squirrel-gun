from flask import Flask, render_template, request, jsonify, send_file, Response
from hardware_controllers.PanTiltController import PanTiltController
from hardware_controllers.WebcamController import WebcamController
from hardware_controllers.WaterController import WaterController
from pathlib import Path
from db import ClickStore
from aim_model import LinearAimer
from event_bus import EventBus
import time
import re

# Serve static files from root (e.g., "/logo.svg").
app = Flask(__name__, static_url_path='')
pantilt = PanTiltController()
webcam = WebcamController()
store = ClickStore()
water = WaterController()
bus = EventBus()

# Wire WebcamController to publish motion events to the bus
try:
    webcam.set_motion_publisher(bus.publish)
except Exception:
    pass

# Subscribe webcam controller to the bus to start/extend recording on motion
try:
    webcam.set_event_bus(bus, record_on_motion=True, duration_sec=30.0)
except Exception:
    pass

# Track current angles in-process (servos don't report position)
# Store current angles as a single immutable tuple so reads/writes are atomic
global current
current = (135.0, 90.0)
follow_motion_enabled = True
_last_follow_ts = 0.0

# Attempt to center hardware on startup; ignore failures if hardware not present
try:
    pantilt.setPanTilt(current[0], current[1])
except Exception as e:
    print(f"Warning: Failed to center pan/tilt on startup: {e}")

def _build_aimer(min_rows: int = 10) -> tuple[LinearAimer, int, bool]:
    """Create a fresh LinearAimer from click data in the DB.

    Returns (model, n_rows, trained) where trained indicates whether
    the model was fitted from data (True) or is a default (False).
    """
    rows = store.list(limit=5000)
    if len(rows) >= max(1, int(min_rows)):
        m = LinearAimer.default()
        m.fit_from_clicks(rows)
        return m, len(rows), True
    return LinearAimer.default(), len(rows), False


def _clamp(v: float, lo: float, hi: float) -> float:
    return hi if v > hi else lo if v < lo else v

## DB initialized by ClickStore on import


@app.route('/')
def index():
    return render_template('PanTiltControl.html')


@app.get('/recordings')
def recordings_page():
    base = Path(__file__).parent / 'static' / 'recordings'
    files = []
    try:
        base.mkdir(parents=True, exist_ok=True)
        for p in base.glob('*.mp4'):
            try:
                stat = p.stat()
                files.append({
                    'name': p.name,
                    'url': f"/recordings/{p.name}",
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'display_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                })
            except Exception:
                pass
        files.sort(key=lambda x: x['mtime'], reverse=True)
    except Exception:
        pass
    return render_template('Recordings.html', files=files)


@app.get('/api/recordings')
def recordings_api():
    base = Path(__file__).parent / 'static' / 'recordings'
    files = []
    try:
        base.mkdir(parents=True, exist_ok=True)
        for p in base.glob('*.mp4'):
            try:
                stat = p.stat()
                files.append({
                    'name': p.name,
                    'url': f"/recordings/{p.name}",
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'display_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                })
            except Exception:
                pass
        files.sort(key=lambda x: x['mtime'], reverse=True)
    except Exception:
        pass
    return jsonify({'files': files})


_REC_NAME_RE = re.compile(r'^rec_\d{8}_\d{6}\.mp4$')


@app.post('/api/recordings/delete')
def recordings_delete():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()
    if not _REC_NAME_RE.match(name):
        return jsonify({"error": "invalid filename"}), 400
    base = Path(__file__).parent / 'static' / 'recordings'
    path = base / name
    try:
        # Ensure path stays within the recordings dir
        if not path.resolve().is_file() or base.resolve() not in path.resolve().parents:
            return jsonify({"error": "file not found"}), 404
        path.unlink()
        return jsonify({"status": "ok", "deleted": name})
    except FileNotFoundError:
        return jsonify({"error": "file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/api/recordings/clear')
def recordings_clear():
    base = Path(__file__).parent / 'static' / 'recordings'
    deleted = 0
    try:
        for p in base.glob('*.mp4'):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": str(e), "deleted": deleted}), 500
    return jsonify({"status": "ok", "deleted": deleted})


@app.post('/api/pan-tilt')
def set_pan_tilt():
    data = request.get_json(silent=True) or {}
    if 'pan' not in data or 'tilt' not in data:
        return jsonify({"error": "pan and tilt required"}), 400
    try:
        pan = float(data['pan'])
        tilt = float(data['tilt'])
    except (ValueError, TypeError):
        return jsonify({"error": "pan/tilt must be numbers"}), 400

    # Clamp to hardware limits before applying and storing
    pan = _clamp(pan, 0.0, float(getattr(pantilt, 'PAN_MAX_DEG', 180)))
    tilt = _clamp(tilt, 0.0, float(getattr(pantilt, 'TILT_MAX_DEG', 180)))
    pantilt.setPanTilt(pan, tilt)
    # Atomic replace of the tuple; avoids the need for locks
    global current
    current = (pan, tilt)
    return jsonify({"status": "ok", "pan": pan, "tilt": tilt})


@app.get('/webcam/capture')
def webcam_capture():
    # Save latest capture in static folder so it can also be served directly if desired
    out_path = Path(__file__).parent / 'static' / 'webcam.jpg'
    try:
        saved = webcam.capture(out_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # Return the image directly so visiting the endpoint displays it
    return send_file(saved, mimetype='image/jpeg')


@app.get('/webcam/stream')
def webcam_stream():
    """MJPEG stream with low-latency defaults and tunable params.

    Query params:
      - fps: int, frames per second (default 5)
      - quality: int 1-100, JPEG quality (default 75)
      - width, height: ints; if provided, update controller target resolution
    """
    try:
        fps = int(request.args.get('fps', '5'))
    except ValueError:
        fps = 5
    try:
        quality = int(request.args.get('quality', '75'))
    except ValueError:
        quality = 75
    # Optional resolution hints
    try:
        w = int(request.args['width']) if 'width' in request.args else None
    except ValueError:
        w = None
    try:
        h = int(request.args['height']) if 'height' in request.args else None
    except ValueError:
        h = None
    if w and w > 0:
        webcam.width = w
    if h and h > 0:
        webcam.height = h

    # Optional motion detection controls
    motion_q = request.args.get('motion')
    min_area_q = request.args.get('motion_min_area')
    alpha_q = request.args.get('motion_alpha')
    bg_mode_q = request.args.get('bg_mode')
    prefer_tracking_q = request.args.get('prefer_tracking')
    frame_skip_q = request.args.get('frame_skip')
    scale_q = request.args.get('scale')
    if (motion_q is not None or min_area_q is not None or alpha_q is not None or
        bg_mode_q is not None or prefer_tracking_q is not None or frame_skip_q is not None or scale_q is not None):
        try:
            enabled = (str(motion_q).strip() not in ('0', 'false', 'False', 'None', '')) if motion_q is not None else getattr(webcam, '_motion_enabled', False)
        except Exception:
            enabled = True
        min_area = None
        alpha = None
        try:
            if min_area_q is not None:
                min_area = int(min_area_q)
        except Exception:
            pass
        try:
            if alpha_q is not None:
                alpha = float(alpha_q)
        except Exception:
            pass
        bg_mode = None
        if bg_mode_q is not None:
            bg_mode = str(bg_mode_q)
        prefer_tracking = None
        if prefer_tracking_q is not None:
            prefer_tracking = not (str(prefer_tracking_q).strip() in ('0', 'false', 'False'))
        frame_skip = None
        try:
            if frame_skip_q is not None:
                frame_skip = int(frame_skip_q)
        except Exception:
            frame_skip = None
        scale = None
        try:
            if scale_q is not None:
                scale = float(scale_q)
        except Exception:
            scale = None
        try:
            webcam.set_motion_detection(enabled=enabled, min_area=min_area, alpha=alpha,
                                        bg_mode=bg_mode, prefer_tracking=prefer_tracking,
                                        frame_skip=frame_skip, scale=scale)
        except Exception:
            pass

    # Delegate streaming logic to the WebcamController
    resp = Response(webcam.mjpeg(fps=fps, quality=quality, boundary='frame'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    # Disable buffering/caching in browsers and proxies to avoid growing delay
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    # If behind nginx, this disables proxy buffering for this response
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@app.post('/api/motion/config')
def motion_config():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', True))
    min_area = data.get('min_area')
    alpha = data.get('alpha')
    persist_ms = data.get('persist_ms')
    bg_mode = data.get('bg_mode')
    prefer_tracking = data.get('prefer_tracking')
    frame_skip = data.get('frame_skip')
    scale = data.get('scale')
    try:
        webcam.set_motion_detection(enabled=enabled, min_area=min_area, alpha=alpha, persist_ms=persist_ms,
                                    bg_mode=bg_mode, prefer_tracking=prefer_tracking,
                                    frame_skip=frame_skip, scale=scale)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({
        "status": "ok",
        "enabled": enabled,
        "min_area": min_area,
        "alpha": alpha,
        "persist_ms": persist_ms,
        "bg_mode": bg_mode,
        "prefer_tracking": prefer_tracking,
        "frame_skip": frame_skip,
        "scale": scale,
    })


@app.get('/api/motion/center')
def motion_center():
    info = webcam.motion_info()
    present = info.get('rect') is not None and info.get('enabled')
    return jsonify({
        'present': bool(present),
        'enabled': bool(info.get('enabled')),
        'rect': info.get('rect'),
        'center': info.get('center'),
        'u': info.get('u'),
        'v': info.get('v'),
        'width': info.get('width'),
        'height': info.get('height'),
    })


@app.get('/api/motion/config')
def motion_config_get():
    try:
        cfg = webcam.motion_config()
        return jsonify(cfg)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/api/motion/follow')
def motion_follow_set():
    global follow_motion_enabled
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', False))
    follow_motion_enabled = enabled
    return jsonify({"status": "ok", "enabled": enabled})


@app.get('/api/motion/follow')
def motion_follow_get():
    return jsonify({"enabled": bool(globals().get('follow_motion_enabled', False))})


@app.post('/api/click')
def record_click():
    data = request.get_json(silent=True) or {}
    try:
        x = float(data.get('x'))
        y = float(data.get('y'))
        w = float(data.get('width'))
        h = float(data.get('height'))
    except (TypeError, ValueError):
        return jsonify({"error": "x, y, width, height required"}), 400
    if any(v is None for v in [x, y, w, h]) or w <= 0 or h <= 0:
        return jsonify({"error": "invalid coordinates"}), 400
    # Read both values from the tuple atomically
    pan, tilt = current
    store.record(pan, tilt, x, y, w, h)
    return jsonify({"status": "ok", "pan": pan, "tilt": tilt, "x": x, "y": y, "width": w, "height": h})


@app.post('/api/aim')
def aim_to_click():
    """Aim the laser at an image coordinate.

    Accepts JSON with only x and y. If values are in [0,1], treat as
    normalized coordinates (u,v). If values are > 1, treat as pixel
    coordinates and normalize using the webcam's configured resolution.
    """
    global current
    data = request.get_json(silent=True) or {}
    try:
        x = float(data.get('x'))
        y = float(data.get('y'))
    except (TypeError, ValueError):
        return jsonify({"error": "x and y required"}), 400
    if x is None or y is None:
        return jsonify({"error": "x and y required"}), 400

    # Determine normalization
    cam_w = float(getattr(webcam, 'width', 0) or 0)
    cam_h = float(getattr(webcam, 'height', 0) or 0)
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        u, v = x, y
        x_px = x * cam_w if cam_w > 0 else None
        y_px = y * cam_h if cam_h > 0 else None
        mode = 'normalized'
    else:
        # Pixels provided; normalize by camera resolution if available
        if cam_w <= 0 or cam_h <= 0:
            return jsonify({"error": "camera resolution unknown; provide normalized x,y in [0,1]"}), 400
        u = x / cam_w
        v = y / cam_h
        x_px = x
        y_px = y
        mode = 'pixels'

    # Predict absolute pan/tilt from (u,v) using a fresh model from DB
    aimer, n_rows, trained = _build_aimer()
    pred_pan, pred_tilt = aimer.predict(u, v)
    new_pan = _clamp(pred_pan, 0.0, float(getattr(pantilt, 'PAN_MAX_DEG', 180)))
    new_tilt = _clamp(pred_tilt, 0.0, float(getattr(pantilt, 'TILT_MAX_DEG', 180)))

    pantilt.setPanTilt(new_pan, new_tilt)
    current = (new_pan, new_tilt)

    return jsonify({
        "status": "ok",
        "x": x, "y": y, "mode": mode,
        "u": u, "v": v,
        "x_px": x_px, "y_px": y_px,
        "pan": new_pan, "tilt": new_tilt,
        "model": aimer.to_dict(),
        "n_rows": n_rows, "trained": trained,
    })


# Subscribe to motion events and aim when enabled (throttled)
def _on_motion(evt: dict) -> None:
    global follow_motion_enabled, _last_follow_ts, current
    try:
        if not follow_motion_enabled:
            return
        u = evt.get('u')
        v = evt.get('v')
        if u is None or v is None:
            return
        now = time.time()
        # Throttle movements to ~5 Hz
        if (now - _last_follow_ts) < 0.18:
            return
        _last_follow_ts = now

        aimer, _n, _trained = _build_aimer()
        pred_pan, pred_tilt = aimer.predict(float(u), float(v))
        new_pan = _clamp(pred_pan, 0.0, float(getattr(pantilt, 'PAN_MAX_DEG', 180)))
        new_tilt = _clamp(pred_tilt, 0.0, float(getattr(pantilt, 'TILT_MAX_DEG', 180)))
        pantilt.setPanTilt(new_pan, new_tilt)
        # Briefly suppress motion to avoid feedback from the laser dot and servo motion
        try:
            webcam.suppress_motion(0.5)
        except Exception:
            pass
        current = (new_pan, new_tilt)
    except Exception:
        # Avoid crashing the bus on handler errors
        pass


bus.subscribe('motion', _on_motion)


@app.get('/api/clicks')
def list_clicks():
    try:
        limit = int(request.args.get('limit', '100'))
        offset = int(request.args.get('offset', '0'))
    except ValueError:
        return jsonify({"error": "invalid limit/offset"}), 400
    rows = store.list(limit=limit, offset=offset)
    return jsonify({"rows": rows})


@app.post('/api/clicks/clear')
def clear_clicks():
    deleted = store.clear()
    return jsonify({"status": "ok", "deleted": deleted})


@app.get('/api/model')
def get_model():
    model, n_rows, trained = _build_aimer()
    return jsonify({"model": model.to_dict(), "n_rows": n_rows, "trained": trained})


@app.post('/api/model/train')
def train_model():
    # Stateless: build and return a fresh model from DB without persisting
    model, n_rows, trained = _build_aimer()
    if not trained:
        return jsonify({"error": "not enough clicks to train (need >= 10)", "n_rows": n_rows}), 400
    return jsonify({"status": "ok", "model": model.to_dict(), "n_rows": n_rows, "trained": trained})


@app.post('/api/water/fire')
def water_fire():
    """Fire water for a short duration (default 2s)."""
    data = request.get_json(silent=True) or {}
    try:
        duration = float(data.get('duration', 2))
    except (TypeError, ValueError):
        duration = 2.0
    # Clamp to a reasonable range to avoid accidents
    duration = _clamp(duration, 0.1, 10.0)
    try:
        water.startWatering(duration)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"status": "ok", "duration": duration})
