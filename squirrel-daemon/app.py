from flask import Flask, render_template, request, jsonify, send_file, Response
from hardware_controllers.PanTiltController import PanTiltController
from hardware_controllers.WebcamController import WebcamController
from pathlib import Path

# Serve static files from root (e.g., "/logo.svg").
app = Flask(__name__, static_url_path='')
pantilt = PanTiltController()
webcam = WebcamController()


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

    pantilt.setPanTilt(pan, tilt)
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
    return Response(webcam.mjpeg(fps=60, quality=80, boundary='frame'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
