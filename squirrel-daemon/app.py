from flask import Flask, render_template, request, jsonify, send_file, Response
from hardware_controllers.PanTiltController import PanTiltController
from hardware_controllers.LaserController import LaserController
from hardware_controllers.WebcamController import WebcamController
from hardware_controllers.WaterController import WaterController
from pathlib import Path
from db import ClickStore
from aim_model import LinearAimer
from laser_dot_detector import LaserDotOptions, detect_laser_dot
from event_bus import EventBus
from .aiming import Calculate_Hose_Angles
import time
import re
import math
import threading
import json
import html
import random
from typing import Optional

# Serve static files from root (e.g., "/logo.svg").
app = Flask(__name__, static_url_path='')
pantilt = PanTiltController()
webcam = WebcamController()
store = ClickStore()
water = WaterController()
laser = LaserController()
bus = EventBus()

# Track laser enabled state (default ON; overridden by persisted if present)
laser_enabled = True

# Wire WebcamController to publish motion events to the bus
webcam.set_motion_publisher(bus.publish)

# Subscribe webcam controller to the bus to start/extend recording on motion
# Initial values will be overridden by persisted settings load below if present
webcam.set_event_bus(bus, record_on_motion=True, duration_sec=30.0)

# Track current angles in-process (servos don't report position)
# Store current angles as a single immutable tuple so reads/writes are atomic
current = (135.0, 90.0)
# Default: don't aim the laser on motion (can be overridden by persisted setting)
follow_motion_enabled = False
_last_follow_ts = 0.0

# Water-on-motion control (disabled by default) with cooldown
water_on_motion_enabled = False
_last_water_fire_ts = 0.0
_WATER_COOLDOWN_SEC = 60.0
_calibration_lock = threading.Lock()

# Load persisted settings and apply to webcam and follow-motion
persisted = store.get_settings([
    'motion.enabled', 'motion.min_area', 'motion.alpha', 'motion.persist_ms', 'motion.bg_mode',
    'motion.prefer_tracking', 'motion.frame_skip', 'motion.scale',
    'record.record_on_motion', 'record.duration_sec', 'record.snapshot_on_motion',
    'follow_motion.enabled',
    'motion.zone',
    'water_on_motion.enabled',
    'laser.enabled',
    'detector.type',
])
# Apply detector type
if 'detector.type' in persisted:
    webcam.set_detector_type(persisted['detector.type'])
# Apply motion settings (use current config as defaults)
cur_motion = webcam.motion_config()
webcam.set_motion_detection(
    enabled=bool(persisted.get('motion.enabled', cur_motion.get('enabled'))),
    min_area=persisted.get('motion.min_area', cur_motion.get('min_area')),
    alpha=persisted.get('motion.alpha', cur_motion.get('alpha')),
    persist_ms=persisted.get('motion.persist_ms', cur_motion.get('persist_ms')),
    bg_mode=persisted.get('motion.bg_mode', cur_motion.get('bg_mode')),
    prefer_tracking=persisted.get('motion.prefer_tracking', cur_motion.get('prefer_tracking')),
    frame_skip=persisted.get('motion.frame_skip', cur_motion.get('frame_skip')),
    scale=persisted.get('motion.scale', cur_motion.get('scale')),
)
# Apply recording settings
cur_rec = webcam.get_recording_config()
webcam.set_recording_config(
    record_on_motion=persisted.get('record.record_on_motion', cur_rec.get('record_on_motion')),
    duration_sec=persisted.get('record.duration_sec', cur_rec.get('duration_sec')),
    snapshot_on_motion=persisted.get('record.snapshot_on_motion', cur_rec.get('snapshot_on_motion'))
)
# Apply follow motion flag
if 'follow_motion.enabled' in persisted:
    follow_motion_enabled = bool(persisted['follow_motion.enabled'])
# Apply motion zone if present (normalized x,y,w,h)
if 'motion.zone' in persisted:
    webcam.set_motion_zone(persisted.get('motion.zone'))
# Apply water-on-motion flag
if 'water_on_motion.enabled' in persisted:
    water_on_motion_enabled = bool(persisted['water_on_motion.enabled'])
# Apply laser enabled flag (default True)
if 'laser.enabled' in persisted:
    laser_enabled = bool(persisted['laser.enabled'])

# Center hardware on startup.
pantilt.setPanTilt(current[0], current[1])

# Set laser state on startup.
if laser_enabled:
    laser.turn_on()
else:
    laser.turn_off()

def _build_aimer(min_rows: int = 10) -> tuple[LinearAimer, int, bool]:
    """Create a fresh LinearAimer from click data in the DB.

    Returns (model, n_rows, trained) where trained indicates whether
    the model was fitted from data (True) or is a default (False).
    """
    rows = store.list(limit=5000)
    if len(rows) >= max(1, int(min_rows)):
        m = LinearAimer.default()
        # If a motion zone is set, use its center as focus for higher local accuracy
        focus = None
        try:
            z = store.get_setting('motion.zone', None)
            if isinstance(z, dict):
                x = float(z.get('x', 0.5))
                y = float(z.get('y', 0.5))
                w = float(z.get('w', 0.0))
                h = float(z.get('h', 0.0))
                # Use center of zone; values are expected normalized 0..1
                focus = (x + max(0.0, w) * 0.5, y + max(0.0, h) * 0.5)
                # Clamp focus to bounds
                fx = 0.0 if focus[0] < 0.0 else 1.0 if focus[0] > 1.0 else focus[0]
                fy = 0.0 if focus[1] < 0.0 else 1.0 if focus[1] > 1.0 else focus[1]
                focus = (fx, fy)
        except Exception:
            focus = None
        # Use a modest sigma so points near the focus dominate (higher accuracy in small area)
        m.fit_from_clicks(rows, focus=focus, sigma=0.2)
        return m, len(rows), True
    return LinearAimer.default(), len(rows), False


def _clamp(v: float, lo: float, hi: float) -> float:
    return hi if v > hi else lo if v < lo else v


def _parse_degree_corner(data: dict, name: str) -> tuple[float, float]:
    raw = data.get(name)
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must include pan and tilt")
    pan = float(raw.get("pan"))
    tilt = float(raw.get("tilt"))
    if not math.isfinite(pan) or not math.isfinite(tilt):
        raise ValueError(f"{name} pan/tilt must be finite numbers")
    pan = _clamp(pan, 0.0, float(getattr(pantilt, 'PAN_MAX_DEG', 270)))
    tilt = _clamp(tilt, 0.0, float(getattr(pantilt, 'TILT_MAX_DEG', 180)))
    return pan, tilt


def _interpolate_calibration_grid(
    *,
    top_left: tuple[float, float],
    top_right: tuple[float, float],
    bottom_left: tuple[float, float],
    bottom_right: tuple[float, float],
    rows: int,
    cols: int,
) -> list[dict]:
    points: list[dict] = []
    for row in range(rows):
        v = row / max(1, rows - 1)
        left_pan = top_left[0] + ((bottom_left[0] - top_left[0]) * v)
        left_tilt = top_left[1] + ((bottom_left[1] - top_left[1]) * v)
        right_pan = top_right[0] + ((bottom_right[0] - top_right[0]) * v)
        right_tilt = top_right[1] + ((bottom_right[1] - top_right[1]) * v)
        col_range = range(cols - 1, -1, -1) if row % 2 else range(cols)
        for col in col_range:
            u = col / max(1, cols - 1)
            points.append({
                "row": row,
                "col": col,
                "pan": left_pan + ((right_pan - left_pan) * u),
                "tilt": left_tilt + ((right_tilt - left_tilt) * u),
            })
    return points


def _calibration_debug_url(path: Path) -> str:
    static_root = (Path(__file__).parent / "static").resolve()
    rel = path.resolve().relative_to(static_root)
    return "/" + "/".join(rel.parts)


def _write_calibration_debug_report(run_dir: Path, *, run_id: str, saved: list[dict], misses: list[dict]) -> Path:
    rows = sorted(
        list(saved) + list(misses),
        key=lambda item: int(item.get("index", 0)),
    )
    parts = [
        "<!doctype html>",
        "<html><head><meta charset=\"utf-8\"><title>Calibration Debug</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;background:#0f172a;color:#e5e7eb;margin:24px}",
        "h1{font-size:22px} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}",
        ".card{background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px}",
        ".meta{color:#9ca3af;font-size:13px;margin:4px 0 10px}.miss{color:#fecaca}.saved{color:#bbf7d0}",
        "img{display:block;width:100%;height:auto;background:#000;border:1px solid #374151;border-radius:6px;margin:8px 0}",
        "a{color:#93c5fd} code{color:#fbbf24}",
        "button{background:#ef4444;color:#fff;border:none;padding:4px 8px;border-radius:4px;cursor:pointer;font-size:12px;margin-left:8px}",
        "button:disabled{background:#4b5563;cursor:not-allowed}",
        "</style></head><body>",
        f"<h1>Calibration Debug {html.escape(run_id)}</h1>",
        f"<p class=\"meta\">Saved {len(saved)} point(s), missed {len(misses)} point(s).</p>",
        "<div class=\"grid\">",
    ]
    for row in rows:
        db_id = row.get("id")
        status = "saved" if db_id is not None else "miss"
        reason = row.get("reason", "saved")
        pan = float(row.get("pan", 0.0))
        tilt = float(row.get("tilt", 0.0))
        idx = int(row.get("index", 0))
        r = row.get("row", "")
        c = row.get("col", "")
        debug = row.get("debug", {}) if isinstance(row.get("debug"), dict) else {}
        dot = row.get("dot")
        dot_json = html.escape(json.dumps(dot, sort_keys=True)) if dot is not None else "null"
        
        delete_btn = ""
        if db_id is not None:
            delete_btn = f"<button onclick=\"deletePoint({db_id}, this)\">Delete from DB</button>"

        parts.extend([
            "<section class=\"card\">",
            f"<h2>#{idx} row {html.escape(str(r))}, col {html.escape(str(c))} {delete_btn}</h2>",
            f"<div class=\"meta {status}\">{html.escape(str(reason))}</div>",
            f"<div class=\"meta\">pan {pan:.2f}, tilt {tilt:.2f}; dot <code>{dot_json}</code></div>",
        ])
        for label, key in (("Laser off", "off_url"), ("Laser on", "on_url"), ("Annotated", "annotated_url"), ("Mask", "mask_url"), ("Score", "score_url")):
            url = debug.get(key)
            if not url:
                continue
            parts.append(f"<div class=\"meta\"><a href=\"{html.escape(url)}\">{html.escape(label)}</a></div>")
            parts.append(f"<img src=\"{html.escape(url)}\" alt=\"{html.escape(label)}\">")
        parts.append("</section>")
    
    parts.extend([
        "</div>",
        "<script>",
        "async function deletePoint(id, btn) {",
        "  if (!confirm('Delete this point from the click database?')) return;",
        "  btn.disabled = true;",
        "  try {",
        "    const res = await fetch('/api/clicks/delete', {",
        "      method: 'POST',",
        "      headers: { 'Content-Type': 'application/json' },",
        "      body: JSON.stringify({ ids: [id] })",
        "    });",
        "    if (res.ok) {",
        "      btn.textContent = 'Deleted';",
        "      btn.closest('.card').style.opacity = '0.5';",
        "    } else {",
        "      const data = await res.json();",
        "      alert('Error: ' + (data.error || res.statusText));",
        "      btn.disabled = false;",
        "    }",
        "  } catch (e) {",
        "    alert('Error: ' + e);",
        "    btn.disabled = false;",
        "  }",
        "}",
        "</script>",
        "</body></html>"
    ])
    report_path = run_dir / "index.html"
    report_path.write_text("\n".join(parts), encoding="utf-8")
    return report_path


def _parse_laser_dot_options(data: dict) -> LaserDotOptions:
    options_data = data.get("dot_options") if isinstance(data.get("dot_options"), dict) else {}
    options = LaserDotOptions(
        min_area=float(options_data.get("min_area", 20.0)),
        max_area=float(options_data.get("max_area", 300.0)),
        max_aspect=float(options_data.get("max_aspect", 2.5)),
        min_width=float(options_data.get("min_width", 4.0)),
        min_height=float(options_data.get("min_height", 4.0)),
        percentile=float(options_data.get("percentile", 99.8)),
        min_score=float(options_data.get("min_score", 18.0)),
        blur_size=int(options_data.get("blur_size", 31)),
        blue_tolerance=float(options_data.get("blue_tolerance", 25.0)),
        open_size=int(options_data.get("open_size", 1)),
        close_size=int(options_data.get("close_size", 9)),
        min_delta_peak=float(options_data.get("min_delta_peak", 10.0)),
        min_delta_mean=float(options_data.get("min_delta_mean", 1.5)),
        min_on_peak=float(options_data.get("min_on_peak", 180.0)),
        min_on_mean=float(options_data.get("min_on_mean", 80.0)),
        peak_window=int(options_data.get("peak_window", 21)),
        peak_candidates=int(options_data.get("peak_candidates", 8)),
    )
    if "threshold" in options_data and options_data.get("threshold") not in (None, ""):
        options.threshold = float(options_data.get("threshold"))
    return options


def _calibration_dot_options_response(options: LaserDotOptions) -> dict:
    return {
        "min_area": options.min_area,
        "max_aspect": options.max_aspect,
        "min_width": options.min_width,
        "min_height": options.min_height,
        "min_delta_peak": options.min_delta_peak,
        "min_delta_mean": options.min_delta_mean,
        "min_score": options.min_score,
        "percentile": options.percentile,
        "close_size": options.close_size,
        "min_on_peak": options.min_on_peak,
        "min_on_mean": options.min_on_mean,
        "peak_window": options.peak_window,
        "peak_candidates": options.peak_candidates,
    }


def _new_calibration_run(prefix: str) -> tuple[str, Path]:
    run_ts = time.time()
    run_id = time.strftime(f"{prefix}_%Y%m%d_%H%M%S", time.localtime(run_ts)) + f"_{int((run_ts % 1) * 1000):03d}"
    run_dir = Path(__file__).parent / "static" / "calibration_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _capture_calibration_sample(
    *,
    run_dir: Path,
    prefix: str,
    pan: float,
    tilt: float,
    options: LaserDotOptions,
    move_settle_sec: float,
    laser_settle_sec: float,
    capture_timeout_sec: float,
    discard_frames: int,
) -> tuple[dict, dict]:
    global current

    off_path = run_dir / f"{prefix}_off.jpg"
    on_path = run_dir / f"{prefix}_on.jpg"
    annotated_path = run_dir / f"{prefix}_annotated.jpg"
    mask_path = run_dir / f"{prefix}_mask.png"
    score_path = run_dir / f"{prefix}_score.png"
    debug = {
        "off_url": _calibration_debug_url(off_path),
        "on_url": _calibration_debug_url(on_path),
        "annotated_url": _calibration_debug_url(annotated_path),
        "mask_url": _calibration_debug_url(mask_path),
        "score_url": _calibration_debug_url(score_path),
    }

    laser.turn_off()
    pantilt.setPanTilt(pan, tilt)
    current = (pan, tilt)
    if move_settle_sec > 0:
        time.sleep(move_settle_sec)
    try:
        webcam.suppress_motion(move_settle_sec + laser_settle_sec + 1.0)
    except Exception:
        pass

    if laser_settle_sec > 0:
        time.sleep(laser_settle_sec)
    webcam.capture_after_discard(
        off_path,
        timeout_sec=capture_timeout_sec,
        raw=True,
        discard_frames=discard_frames,
    )

    laser.turn_on()
    if laser_settle_sec > 0:
        time.sleep(laser_settle_sec)
    webcam.capture_after_discard(
        on_path,
        timeout_sec=capture_timeout_sec,
        raw=True,
        discard_frames=discard_frames,
    )

    if not off_path.exists() or not on_path.exists():
        return {
            "dot": None,
            "reason": "capture failed: image file not written",
            "image_width": 0,
            "image_height": 0,
        }, debug

    detected = detect_laser_dot(
        on_path,
        off_path,
        options=options,
        debug_paths={
            "annotated": annotated_path,
            "mask": mask_path,
            "score": score_path,
        },
    )
    return detected, debug


def _dot_edge_reason(dot: dict, img_w: float, img_h: float, edge_margin_px: float, edge_margin_frac: float) -> tuple[Optional[str], float]:
    dot_x = float(dot["x"])
    dot_y = float(dot["y"])
    dot_w = float(dot["w"])
    dot_h = float(dot["h"])
    effective_edge_margin = max(edge_margin_px, min(img_w, img_h) * edge_margin_frac)
    if (
        dot_x <= effective_edge_margin
        or dot_y <= effective_edge_margin
        or (dot_x + dot_w) >= (img_w - effective_edge_margin)
        or (dot_y + dot_h) >= (img_h - effective_edge_margin)
    ):
        return "dot clipped at image edge", effective_edge_margin
    return None, effective_edge_margin

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


@app.get('/motion-zone')
def motion_zone_page():
    return render_template('MotionZone.html')


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


@app.get('/snapshots')
def snapshots_page():
    base = Path(__file__).parent / 'static' / 'recordings' / 'shots'
    files = []
    try:
        base.mkdir(parents=True, exist_ok=True)
        for p in base.glob('*.jpg'):
            try:
                stat = p.stat()
                files.append({
                    'name': p.name,
                    'url': f"/recordings/shots/{p.name}",
                    'size': stat.st_size,
                    'mtime': stat.st_mtime,
                    'display_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                })
            except Exception:
                pass
        files.sort(key=lambda x: x['mtime'], reverse=True)
    except Exception:
        pass
    return render_template('Snapshots.html', files=files)


@app.get('/api/recording/config')
def recording_config_get():
    try:
        cfg = webcam.get_recording_config()
        return jsonify(cfg)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/api/recording/config')
def recording_config_set():
    data = request.get_json(silent=True) or {}
    record_on_motion = data.get('record_on_motion')
    duration_sec = data.get('duration_sec')
    snapshot_on_motion = data.get('snapshot_on_motion')
    try:
        webcam.set_recording_config(record_on_motion=record_on_motion, duration_sec=duration_sec, snapshot_on_motion=snapshot_on_motion)
        try:
            if record_on_motion is not None: store.set_setting('record.record_on_motion', bool(record_on_motion))
            if duration_sec is not None: store.set_setting('record.duration_sec', float(duration_sec))
            if snapshot_on_motion is not None: store.set_setting('record.snapshot_on_motion', bool(snapshot_on_motion))
        except Exception:
            pass
        return jsonify({"status": "ok", **webcam.get_recording_config()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get('/api/recording/status')
def recording_status():
    try:
        active = webcam.is_recording()
        ends_at = None
        seconds_left = None
        path = None
        try:
            ends_at = getattr(webcam, '_recording_end_ts', 0.0)
            seconds_left = max(0.0, float(ends_at - time.time())) if active else 0.0
        except Exception:
            pass
        try:
            p = getattr(webcam, '_recording_path', None)
            path = str(p) if p else None
        except Exception:
            path = None
        return jsonify({
            'active': bool(active),
            'ends_at': ends_at,
            'seconds_left': seconds_left,
            'path': path,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/api/recording/start')
def recording_start():
    data = request.get_json(silent=True) or {}
    try:
        duration = float(data.get('duration', 10))
    except Exception:
        duration = 10.0
    try:
        p = webcam.start_recording(duration_sec=max(1.0, duration), extend=False)
        return jsonify({
            'status': 'ok',
            'started': bool(p is not None),
            'path': str(p) if p else getattr(webcam, '_recording_path', None),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/api/recording/stop')
def recording_stop():
    try:
        webcam.stop_recording()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_REC_NAME_RE = re.compile(r'^(rec_(\d{8}_\d{6}))\.mp4$')


@app.post('/api/recordings/delete')
def recordings_delete():
    data = request.get_json(silent=True) or {}
    name = str(data.get('name', '')).strip()
    m = _REC_NAME_RE.match(name)
    if not m:
        return jsonify({"error": "invalid filename"}), 400
    base = Path(__file__).parent / 'static' / 'recordings'
    path = base / name
    try:
        # Ensure path stays within the recordings dir
        if not path.resolve().is_file() or base.resolve() not in path.resolve().parents:
            return jsonify({"error": "file not found"}), 404
        path.unlink()
        # Also delete associated snapshots (same timestamp)
        ts = m.group(2)
        shots_dir = base / 'shots'
        deleted_shots = []
        # Delete the exact match and any suffix variants (older naming)
        for sp in [shots_dir / f'snap_{ts}.jpg'] + list(shots_dir.glob(f'snap_{ts}*.jpg')):
            try:
                if sp.is_file():
                    sp.unlink()
                    deleted_shots.append(sp.name)
            except Exception:
                pass
        return jsonify({"status": "ok", "deleted": name, "deleted_shots": deleted_shots})
    except FileNotFoundError:
        return jsonify({"error": "file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post('/api/recordings/clear')
def recordings_clear():
    base = Path(__file__).parent / 'static' / 'recordings'
    deleted = 0
    deleted_shots = 0
    try:
        for p in base.glob('*.mp4'):
            try:
                p.unlink()
                deleted += 1
            except Exception:
                pass
        # Also clear all snapshots
        shots_dir = base / 'shots'
        for sp in shots_dir.glob('*.jpg'):
            try:
                sp.unlink()
                deleted_shots += 1
            except Exception:
                pass
    except Exception as e:
        return jsonify({"error": str(e), "deleted": deleted, "deleted_shots": deleted_shots}), 500
    return jsonify({"status": "ok", "deleted": deleted, "deleted_shots": deleted_shots})


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
        if not saved:
            return jsonify({"error": "failed to capture image from webcam"}), 500
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
    preview_scale_q = request.args.get('preview_scale')
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
    # Optional preview downscale hint (only affects JPEG stream, not recording)
    if preview_scale_q is not None:
        try:
            webcam.set_preview_scale(float(preview_scale_q))
        except Exception:
            pass

    # Optional low-latency streaming hint
    low_lat_q = request.args.get('low_latency')
    if low_lat_q is not None:
        try:
            ll = not (str(low_lat_q).strip() in ('0', 'false', 'False'))
            try:
                webcam.set_low_latency_mode(ll)
            except Exception:
                pass
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
    # Persist provided fields
    try:
        if 'enabled' in data: store.set_setting('motion.enabled', bool(enabled))
        if min_area is not None: store.set_setting('motion.min_area', int(min_area))
        if alpha is not None: store.set_setting('motion.alpha', float(alpha))
        if persist_ms is not None: store.set_setting('motion.persist_ms', int(persist_ms))
        if bg_mode is not None: store.set_setting('motion.bg_mode', str(bg_mode))
        if prefer_tracking is not None: store.set_setting('motion.prefer_tracking', bool(prefer_tracking))
        if frame_skip is not None: store.set_setting('motion.frame_skip', int(frame_skip))
        if scale is not None: store.set_setting('motion.scale', float(scale))
    except Exception:
        pass
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


@app.get('/api/motion/zone')
def motion_zone_get():
    try:
        z = webcam.motion_zone()
        return jsonify({ 'zone': z })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500


@app.post('/api/motion/zone')
def motion_zone_set():
    data = request.get_json(silent=True) or {}
    # Accept normalized x,y,w,h in [0,1], or pixel coords if width/height provided
    x = data.get('x'); y = data.get('y'); w = data.get('w'); h = data.get('h')
    # If values look > 1, assume pixels and normalize using camera resolution
    try:
        fx = float(x) if x is not None else None
        fy = float(y) if y is not None else None
        fw = float(w) if w is not None else None
        fh = float(h) if h is not None else None
    except Exception:
        return jsonify({ 'error': 'invalid x/y/w/h' }), 400
    # Normalize if necessary
    if any(v is None for v in (fx, fy, fw, fh)):
        return jsonify({ 'error': 'x,y,w,h required' }), 400
    if (fx > 1.0 or fy > 1.0 or fw > 1.0 or fh > 1.0):
        # Use webcam reported resolution for normalization
        cam_w = float(getattr(webcam, 'width', 0) or 0)
        cam_h = float(getattr(webcam, 'height', 0) or 0)
        if cam_w <= 0 or cam_h <= 0:
            return jsonify({ 'error': 'camera resolution unknown; provide normalized values 0..1' }), 400
        fx /= cam_w; fw /= cam_w
        fy /= cam_h; fh /= cam_h
    # Clamp and apply
    fx = max(0.0, min(1.0, float(fx)))
    fy = max(0.0, min(1.0, float(fy)))
    fw = max(0.0, min(1.0 - fx, float(fw)))
    fh = max(0.0, min(1.0 - fy, float(fh)))
    zone = { 'x': fx, 'y': fy, 'w': fw, 'h': fh }
    try:
        webcam.set_motion_zone(zone)
        store.set_setting('motion.zone', zone)
        return jsonify({ 'status': 'ok', 'zone': zone })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500


@app.post('/api/motion/zone/clear')
def motion_zone_clear():
    try:
        webcam.set_motion_zone(None)
        store.set_setting('motion.zone', None)
        return jsonify({ 'status': 'ok' })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500


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
        'fg_pixels': info.get('fg_pixels'),
        'largest_area': info.get('largest_area'),
        'peak_largest_area': info.get('peak_largest_area'),
    })


@app.get('/api/motion/counters')
def motion_counters():
    try:
        return jsonify(webcam.motion_counters())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.post('/api/motion/counters/reset')
def motion_counters_reset():
    try:
        webcam.reset_motion_counters()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.post('/api/motion/peak/reset')
def motion_peak_reset():
    try:
        webcam.reset_motion_peak()
        return jsonify({'status': 'ok'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.get('/api/motion/config')
def motion_config_get():
    try:
        cfg = webcam.motion_config()
        return jsonify(cfg)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Detector type endpoints
@app.get('/api/detector/type')
def detector_type_get():
    try:
        return jsonify({ 'type': webcam.get_detector_type() })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500


@app.post('/api/detector/type')
def detector_type_set():
    data = request.get_json(silent=True) or {}
    kind = str(data.get('type', 'motion')).lower().strip()
    if kind not in ('motion', 'yolo'):
        return jsonify({ 'error': 'invalid type' }), 400
    try:
        t = webcam.set_detector_type(kind)
        try:
            store.set_setting('detector.type', t)
        except Exception:
            pass
        return jsonify({ 'status': 'ok', 'type': t })
    except Exception as e:
        return jsonify({ 'error': str(e) }), 500


@app.post('/api/motion/follow')
def motion_follow_set():
    global follow_motion_enabled
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', False))
    follow_motion_enabled = enabled
    try:
        store.set_setting('follow_motion.enabled', bool(enabled))
    except Exception:
        pass
    return jsonify({"status": "ok", "enabled": enabled})


@app.get('/api/motion/follow')
def motion_follow_get():
    return jsonify({"enabled": bool(globals().get('follow_motion_enabled', False))})


@app.post('/api/motion/water')
def motion_water_set():
    """Enable/disable firing water on motion events (with cooldown)."""
    global water_on_motion_enabled
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get('enabled', False))
    water_on_motion_enabled = enabled
    try:
        store.set_setting('water_on_motion.enabled', bool(enabled))
    except Exception:
        pass
    return jsonify({"status": "ok", "enabled": enabled})


@app.get('/api/motion/water')
def motion_water_get():
    now = time.time()
    last = float(globals().get('_last_water_fire_ts', 0.0) or 0.0)
    cd = float(globals().get('_WATER_COOLDOWN_SEC', 60.0) or 60.0)
    remaining = max(0.0, (last + cd) - now)
    return jsonify({
        "enabled": bool(globals().get('water_on_motion_enabled', False)),
        "cooldown_remaining_sec": remaining,
        "cooldown_sec": cd,
    })


@app.post('/api/calibration/automatic')
def automatic_calibration():
    global current, follow_motion_enabled

    if not _calibration_lock.acquire(blocking=False):
        return jsonify({"error": "automatic calibration is already running"}), 409

    data = request.get_json(silent=True) or {}
    try:
        top_left = _parse_degree_corner(data, "top_left")
        top_right = _parse_degree_corner(data, "top_right")
        bottom_left = _parse_degree_corner(data, "bottom_left")
        bottom_right = _parse_degree_corner(data, "bottom_right")
        rows = max(2, min(25, int(data.get("rows", 5))))
        cols = max(2, min(25, int(data.get("cols", 5))))
        move_settle_sec = max(0.0, min(5.0, float(data.get("move_settle_sec", 0.35))))
        laser_settle_sec = max(0.0, min(5.0, float(data.get("laser_settle_sec", 0.5))))
        capture_timeout_sec = max(0.1, min(5.0, float(data.get("capture_timeout_sec", 2.0))))
        discard_frames = max(0, min(15, int(data.get("discard_frames", 4))))
        edge_margin_px = max(0.0, min(200.0, float(data.get("edge_margin_px", 20.0))))
        edge_margin_frac = max(0.0, min(0.25, float(data.get("edge_margin_frac", 0.03))))
        phase2_count = max(0, min(50, int(data.get("phase2_count", 10))))

        options = _parse_laser_dot_options(data)
    except (TypeError, ValueError) as e:
        _calibration_lock.release()
        return jsonify({"error": str(e)}), 400

    points = _interpolate_calibration_grid(
        top_left=top_left,
        top_right=top_right,
        bottom_left=bottom_left,
        bottom_right=bottom_right,
        rows=rows,
        cols=cols,
    )
    original_current = current
    original_follow = bool(globals().get("follow_motion_enabled", False))
    original_laser_enabled = bool(globals().get("laser_enabled", True))
    saved: list[dict] = []
    misses: list[dict] = []
    run_ts = time.time()
    run_id = time.strftime("cal_%Y%m%d_%H%M%S", time.localtime(run_ts)) + f"_{int((run_ts % 1) * 1000):03d}"
    run_dir = Path(__file__).parent / "static" / "calibration_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trained_p2 = False
    try:
        follow_motion_enabled = False
        try:
            webcam.suppress_motion(move_settle_sec + laser_settle_sec + 1.0)
        except Exception:
            pass

        # Phase 1: Degree Bounds Sweep
        for idx, point in enumerate(points):
            pan = float(point["pan"])
            tilt = float(point["tilt"])
            prefix = f"p1_{idx:03d}_r{int(point['row']):02d}_c{int(point['col']):02d}"

            detected, debug = _capture_calibration_sample(
                run_dir=run_dir,
                prefix=prefix,
                pan=pan,
                tilt=tilt,
                options=options,
                move_settle_sec=move_settle_sec,
                laser_settle_sec=laser_settle_sec,
                capture_timeout_sec=capture_timeout_sec,
                discard_frames=discard_frames,
            )

            dot = detected.get("dot")
            if dot is None:
                misses.append({
                    "index": idx,
                    "phase": 1,
                    "row": point["row"],
                    "col": point["col"],
                    "pan": pan,
                    "tilt": tilt,
                    "reason": "dot not found",
                    "debug": debug,
                    "scene": detected.get("scene"),
                    "method": detected.get("method"),
                })
                continue

            img_w = float(detected["image_width"])
            img_h = float(detected["image_height"])
            reason, effective_edge_margin = _dot_edge_reason(dot, img_w, img_h, edge_margin_px, edge_margin_frac)
            if reason:
                misses.append({
                    "index": idx,
                    "phase": 1,
                    "row": point["row"],
                    "col": point["col"],
                    "pan": pan,
                    "tilt": tilt,
                    "reason": reason,
                    "dot": dot,
                    "debug": debug,
                    "edge_margin_px": effective_edge_margin,
                    "scene": detected.get("scene"),
                    "method": detected.get("method"),
                })
                continue

            x = float(dot["cx"])
            y = float(dot["cy"])
            click_id = store.record(pan, tilt, x, y, img_w, img_h)
            saved.append({
                "id": click_id,
                "index": idx,
                "phase": 1,
                "row": point["row"],
                "col": point["col"],
                "pan": pan,
                "tilt": tilt,
                "x": x,
                "y": y,
                "width": img_w,
                "height": img_h,
                "dot": dot,
                "debug": debug,
                "scene": detected.get("scene"),
                "method": detected.get("method"),
            })

        # Phase 2: Reverse-Mapping Random Point Validation
        if phase2_count > 0:
            # Re-train model from Phase 1 data
            aimer, _, trained_p2 = _build_aimer()
            if trained_p2:
                for idx_p2 in range(phase2_count):
                    # Random normalized coords, slightly inset to avoid edge issues
                    u = random.uniform(0.1, 0.9)
                    v = random.uniform(0.1, 0.9)
                    pan, tilt = aimer.predict(u, v)
                    pan = _clamp(pan, 0.0, float(getattr(pantilt, 'PAN_MAX_DEG', 270)))
                    tilt = _clamp(tilt, 0.0, float(getattr(pantilt, 'TILT_MAX_DEG', 180)))

                    prefix = f"p2_{idx_p2:03d}"
                    detected, debug = _capture_calibration_sample(
                        run_dir=run_dir,
                        prefix=prefix,
                        pan=pan,
                        tilt=tilt,
                        options=options,
                        move_settle_sec=move_settle_sec,
                        laser_settle_sec=laser_settle_sec,
                        capture_timeout_sec=capture_timeout_sec,
                        discard_frames=discard_frames,
                    )

                    dot = detected.get("dot")
                    if dot is None:
                        misses.append({
                            "index": len(points) + idx_p2,
                            "phase": 2,
                            "row": "p2",
                            "col": idx_p2,
                            "pan": pan,
                            "tilt": tilt,
                            "reason": f"dot not found for image target ({u:.2f}, {v:.2f})",
                            "debug": debug,
                            "scene": detected.get("scene"),
                            "method": detected.get("method"),
                        })
                        continue

                    img_w = float(detected["image_width"])
                    img_h = float(detected["image_height"])
                    reason, effective_edge_margin = _dot_edge_reason(dot, img_w, img_h, edge_margin_px, edge_margin_frac)
                    if reason:
                        misses.append({
                            "index": len(points) + idx_p2,
                            "phase": 2,
                            "row": "p2",
                            "col": idx_p2,
                            "pan": pan,
                            "tilt": tilt,
                            "reason": reason,
                            "dot": dot,
                            "debug": debug,
                            "edge_margin_px": effective_edge_margin,
                            "scene": detected.get("scene"),
                            "method": detected.get("method"),
                        })
                        continue

                    x = float(dot["cx"])
                    y = float(dot["cy"])
                    click_id = store.record(pan, tilt, x, y, img_w, img_h)
                    saved.append({
                        "id": click_id,
                        "index": len(points) + idx_p2,
                        "phase": 2,
                        "row": "p2",
                        "col": idx_p2,
                        "pan": pan,
                        "tilt": tilt,
                        "x": x,
                        "y": y,
                        "width": img_w,
                        "height": img_h,
                        "dot": dot,
                        "debug": debug,
                        "scene": detected.get("scene"),
                        "method": detected.get("method"),
                    })

    finally:
        try:
            if original_laser_enabled:
                laser.turn_on()
            else:
                laser.turn_off()
        except Exception:
            pass
        try:
            pantilt.setPanTilt(original_current[0], original_current[1])
            current = original_current
        except Exception:
            pass
        follow_motion_enabled = original_follow
        _calibration_lock.release()

    report_path = _write_calibration_debug_report(run_dir, run_id=run_id, saved=saved, misses=misses)
    report_url = _calibration_debug_url(report_path)

    return jsonify({
        "status": "ok",
        "requested": len(points) + (phase2_count if trained_p2 else 0),
        "saved_count": len(saved),
        "miss_count": len(misses),
        "debug_run_id": run_id,
        "debug_report_url": report_url,
        "debug_dir_url": _calibration_debug_url(run_dir),
        "rows": rows,
        "cols": cols,
        "phase2_count": phase2_count if trained_p2 else 0,
        "edge_margin_px": edge_margin_px,
        "edge_margin_frac": edge_margin_frac,
        "laser_settle_sec": laser_settle_sec,
        "discard_frames": discard_frames,
        "dot_options": _calibration_dot_options_response(options),
        "saved": saved,
        "misses": misses,
        "pan": current[0],
        "tilt": current[1],
    })


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

    """
    print(f"Received aim request for {x},{y}", flush=True)
    results  = Calculate_Hose_Angles.get_yaw_pitch(x,y)
    print(f"Calculated angles")

    pred_pan = results[0]
    pred_tilt = results[1]

    new_pan = _clamp(pred_pan, 0.0, float(getattr(pantilt, 'PAN_MAX_DEG', 180)))
    new_tilt = _clamp(pred_tilt, 0.0, float(getattr(pantilt, 'TILT_MAX_DEG', 180)))

    pantilt.setPanTilt(new_pan, new_tilt)
    current = (new_pan, new_tilt)
    print(f"Result {new_pan},{new_tilt}", flush=True)
    """
    
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


# Subscribe to motion events and optionally fire water (with cooldown)
def _on_motion_water(evt: dict) -> None:
    global water_on_motion_enabled, _last_water_fire_ts
    try:
        if not water_on_motion_enabled:
            return
        now = time.time()
        # Enforce cooldown between water firings
        if (now - _last_water_fire_ts) < float(_WATER_COOLDOWN_SEC):
            return
        _last_water_fire_ts = now

        # Fire asynchronously to avoid blocking the event bus thread
        def _do_fire():
            try:
                water.startWatering(2.0)
            except Exception:
                pass

        import threading as _th
        _th.Thread(target=_do_fire, name='WaterOnMotion', daemon=True).start()
    except Exception:
        # Avoid crashing the bus on handler errors
        pass


bus.subscribe('motion', _on_motion_water)

@app.get('/api/clicks')
def list_clicks():
    try:
        limit = int(request.args.get('limit', '100'))
        offset = int(request.args.get('offset', '0'))
    except ValueError:
        return jsonify({"error": "invalid limit/offset"}), 400
    rows = store.list(limit=limit, offset=offset)
    return jsonify({"rows": rows})


@app.post('/api/clicks/delete')
def delete_clicks():
    data = request.get_json(silent=True) or {}
    ids = data.get('ids')
    if not isinstance(ids, list):
        return jsonify({"error": "ids must be a list"}), 400
    try:
        deleted = store.delete_ids([int(i) for i in ids])
        return jsonify({"status": "ok", "deleted": deleted})
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400


@app.post('/api/clicks/prune-outliers')
def prune_clicks():
    data = request.get_json(silent=True) or {}
    try:
        threshold = float(data.get('max_error_deg', 15.0))
    except (TypeError, ValueError):
        return jsonify({"error": "max_error_deg must be a number"}), 400
    if not math.isfinite(threshold) or threshold <= 0.0:
        return jsonify({"error": "max_error_deg must be > 0"}), 400
    try:
        limit = int(data.get('limit', 5000))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer"}), 400
    limit = max(1, min(50000, limit))
    dry_run = bool(data.get('dry_run', False))

    rows = store.list(limit=limit)
    if not rows:
        return jsonify({
            "status": "ok",
            "deleted": 0,
            "evaluated": 0,
            "threshold": threshold,
            "dry_run": dry_run,
            "candidates": 0,
        })

    # Mirror training focus used by _build_aimer so residuals stay comparable
    focus = None
    try:
        z = store.get_setting('motion.zone', None)
        if isinstance(z, dict):
            x = float(z.get('x', 0.5))
            y = float(z.get('y', 0.5))
            w = float(z.get('w', 0.0))
            h = float(z.get('h', 0.0))
            fx = x + max(0.0, w) * 0.5
            fy = y + max(0.0, h) * 0.5
            fx = 0.0 if fx < 0.0 else 1.0 if fx > 1.0 else fx
            fy = 0.0 if fy < 0.0 else 1.0 if fy > 1.0 else fy
            focus = (fx, fy)
    except Exception:
        focus = None

    model = LinearAimer.default()
    model.fit_from_clicks(rows, focus=focus, sigma=0.2)

    evaluated = 0
    outlier_ids: list[int] = []
    sample_residuals: list[dict[str, float | int]] = []
    for r in rows:
        try:
            w = float(r['img_w'])
            h = float(r['img_h'])
            if w <= 0 or h <= 0:
                continue
            u = float(r['x_px']) / w
            v = float(r['y_px']) / h
            pred_pan, pred_tilt = model.predict(u, v)
            err = math.hypot(float(r['pan']) - pred_pan, float(r['tilt']) - pred_tilt)
        except Exception:
            continue
        evaluated += 1
        if err > threshold:
            outlier_ids.append(int(r['id']))
            if len(sample_residuals) < 5:
                sample_residuals.append({
                    'id': int(r['id']),
                    'error_deg': float(err),
                    'pan': float(r['pan']),
                    'tilt': float(r['tilt']),
                })

    deleted = 0
    if outlier_ids and not dry_run:
        deleted = store.delete_ids(outlier_ids)

    return jsonify({
        "status": "ok",
        "deleted": deleted,
        "candidates": len(outlier_ids),
        "evaluated": evaluated,
        "threshold": threshold,
        "dry_run": dry_run,
        "sample": sample_residuals,
    })


@app.get('/api/laser')
def get_laser():
    return jsonify({"enabled": bool(laser_enabled)})


@app.post('/api/laser')
def set_laser():
    global laser_enabled
    data = request.get_json(silent=True) or {}
    enabled = data.get('enabled')
    if not isinstance(enabled, bool):
        return jsonify({"error": "enabled must be boolean"}), 400
    try:
        if enabled:
            laser.turn_on()
        else:
            laser.turn_off()
    except Exception as e:
        return jsonify({"error": f"failed to set laser: {e}"}), 500
    laser_enabled = enabled
    try:
        store.set_setting('laser.enabled', bool(laser_enabled))
    except Exception:
        pass
    return jsonify({"status": "ok", "enabled": bool(laser_enabled)})


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
