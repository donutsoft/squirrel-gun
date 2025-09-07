from flask import Flask, render_template, request, jsonify, send_file, Response
from hardware_controllers.PanTiltController import PanTiltController
from hardware_controllers.WebcamController import WebcamController
from pathlib import Path
from db import ClickStore
from aim_model import LinearAimer

# Serve static files from root (e.g., "/logo.svg").
app = Flask(__name__, static_url_path='')
pantilt = PanTiltController()
webcam = WebcamController()
store = ClickStore()

# Track current angles in-process (servos don't report position)
# Store current angles as a single immutable tuple so reads/writes are atomic
global current
current = (135.0, 90.0)

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
    # Delegate streaming logic to the WebcamController
    return Response(webcam.mjpeg(fps=5, quality=80, boundary='frame'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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
