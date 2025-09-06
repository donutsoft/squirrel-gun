from flask import Flask, render_template, request, jsonify
from hardware_controllers.PanTiltController import PanTiltController

# Serve static files from root (e.g., "/logo.svg").
app = Flask(__name__, static_url_path='')
pantilt = PanTiltController()


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
