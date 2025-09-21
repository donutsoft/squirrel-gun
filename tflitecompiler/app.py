import io
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import threading
import queue
import json
import logging

from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, Response, stream_with_context
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
# Allow large uploads (MB). Default 1024MB. Override with MAX_UPLOAD_MB env var.
try:
    app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_UPLOAD_MB', '1024')) * 1024 * 1024
except Exception:
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class QueueWriter:
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, s: str):
        if not s:
            return 0
        self.q.put(s)
        return len(s)

    def flush(self):
        pass


# In-memory job registry for streaming logs and download info
JOBS = {}


class QueueLogHandler(logging.Handler):
    """Logging handler that writes formatted records into a Queue for SSE streaming."""
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()
        # Ensure each log record ends with a newline for readability
        if not msg.endswith("\n"):
            msg += "\n"
        self.q.put(msg)


def run_export(job_id: str, upload_path: Path):
    """Export to EdgeTPU using Ultralytics YOLO export only.
    Streams logs to JOBS[job_id]['queue'] and records output.
    Uses the original uploaded filename and removes it after export.
    """

    q: queue.Queue = JOBS[job_id]["queue"]
    writer = QueueWriter(q)

    exported_path = None
    # Attach logging handler to stream Ultralytics logs as they occur
    root_logger = logging.getLogger()
    ul_logger = logging.getLogger("ultralytics")
    handler = QueueLogHandler(q)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    ul_logger.addHandler(handler)
    # Ensure INFO level so typical logs are emitted
    root_prev_level = root_logger.level
    ul_prev_level = ul_logger.level
    root_logger.setLevel(logging.INFO)
    ul_logger.setLevel(logging.INFO)

    with redirect_stdout(writer), redirect_stderr(writer):
        try:
            print("Starting job...\n")
            print(f"Loading model from: {upload_path}")
            # Import inside the redirected/logging context to capture import-time logs
            from ultralytics import YOLO
            model = YOLO(str(upload_path))
            print("Starting export to EdgeTPU via Ultralytics...")
            result = model.export(format='edgetpu', imgsz=320, nms=False, batch=1, dynamic=False)
            print(f"Export result: {result}")
        except Exception as e:
            print("Export failed:", repr(e))
        finally:
            # Detach handlers and restore levels to avoid duplicate logs in subsequent jobs
            try:
                root_logger.removeHandler(handler)
            except Exception:
                pass
            try:
                ul_logger.removeHandler(handler)
            except Exception:
                pass
            root_logger.setLevel(root_prev_level)
            ul_logger.setLevel(ul_prev_level)

    # Try to locate the produced *edgetpu.tflite file
    candidates = list(BASE_DIR.rglob("*edgetpu.tflite"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        src = candidates[0]
        # Unique per-job filename to avoid conflicts
        target = OUTPUT_DIR / f"{job_id}_edge_tpu.tflite"
        target.write_bytes(src.read_bytes())
        exported_path = target

    # Remove the original uploaded file after export completes (success or fail)
    try:
        if upload_path.exists():
            upload_path.unlink()
    except Exception:
        pass

    JOBS[job_id]["download"] = exported_path.name if exported_path else None
    JOBS[job_id]["done"] = True


@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part provided')
        return render_template('upload.html')

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return render_template('upload.html')

    # Save uploaded file
    saved_path = UPLOAD_DIR / file.filename
    file.save(saved_path)

    # Create job and start background export
    job_id = str(uuid4())
    JOBS[job_id] = {"queue": queue.Queue(), "done": False, "download": None}

    t = threading.Thread(target=run_export, args=(job_id, saved_path), daemon=True)
    t.start()

    return render_template('upload.html', job_id=job_id)


@app.route('/stream/<job_id>')
def stream(job_id):
    if job_id not in JOBS:
        return Response("", status=404)

    def event_stream():
        q: queue.Queue = JOBS[job_id]["queue"]
        # Send an initial message so the client sees activity promptly
        init = {"type": "status", "message": "connected"}
        yield f"data: {json.dumps(init)}\n\n"
        # Periodic heartbeats to keep proxies/browsers flushing
        import time
        last_ping = time.time()
        while True:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                if JOBS[job_id]["done"]:
                    download = JOBS[job_id]["download"]
                    download_url = f"/download/{download}" if download else None
                    data = {"type": "done", "download": download_url}
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                # Heartbeat every 2s
                if time.time() - last_ping > 2:
                    yield ": ping\n\n"
                    last_ping = time.time()
                continue
            else:
                data = {"type": "log", "message": chunk}
                yield f"data: {json.dumps(data)}\n\n"

    resp = Response(stream_with_context(event_stream()), mimetype='text/event-stream')
    resp.headers["Cache-Control"] = "no-cache, no-transform"
    resp.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if present
    resp.headers["Connection"] = "keep-alive"
    return resp


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    # Run the Flask dev server (sufficient for simple usage inside container)
    app.run(host='0.0.0.0', port=8000, debug=False)
