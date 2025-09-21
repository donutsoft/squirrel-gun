import io
import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import threading
import queue
import json

from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash, Response
from uuid import uuid4

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

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


def run_export(job_id: str, upload_path: Path):
    """Export to EdgeTPU using Ultralytics YOLO export only.
    Streams logs to JOBS[job_id]['queue'] and records output.
    """
    from ultralytics import YOLO

    q: queue.Queue = JOBS[job_id]["queue"]
    writer = QueueWriter(q)

    # As requested, use fixed name "uploadedfile.tp" regardless of input name
    fixed_input = UPLOAD_DIR / "uploadedfile.tp"
    fixed_input.write_bytes(upload_path.read_bytes())

    exported_path = None
    with redirect_stdout(writer), redirect_stderr(writer):
        try:
            print(f"Loading model from: {fixed_input}")
            model = YOLO(str(fixed_input))
            print("Starting export to EdgeTPU via Ultralytics...")
            result = model.export(format='edgetpu', imgsz=320, nms=False, batch=1, dynamic=False)
            print(f"Export result: {result}")
        except Exception as e:
            print("Export failed:", repr(e))

    # Try to locate the produced *edgetpu.tflite file
    candidates = list(BASE_DIR.rglob("*edgetpu.tflite"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        src = candidates[0]
        # Unique per-job filename to avoid conflicts
        target = OUTPUT_DIR / f"{job_id}_edge_tpu.tflite"
        target.write_bytes(src.read_bytes())
        exported_path = target

    JOBS[job_id]["download"] = exported_path.name if exported_path else None
    JOBS[job_id]["done"] = True


@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part provided')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

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
        while True:
            try:
                chunk = q.get(timeout=0.5)
            except queue.Empty:
                if JOBS[job_id]["done"]:
                    data = {
                        "type": "done",
                        "download": (
                            url_for('download_file', filename=JOBS[job_id]["download"]) if JOBS[job_id]["download"] else None
                        ),
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                continue
            else:
                data = {"type": "log", "message": chunk}
                yield f"data: {json.dumps(data)}\n\n"

    resp = Response(event_stream(), mimetype='text/event-stream')
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == '__main__':
    # Run the Flask dev server (sufficient for simple usage inside container)
    app.run(host='0.0.0.0', port=8000, debug=False)
