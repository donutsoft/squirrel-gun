#!/usr/bin/env bash
set -euo pipefail

# ---- Configurable parameters (override via env) ----
DEVICE=${DEVICE:-/dev/video0}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
FPS=${FPS:-10}
PORT=${PORT:-8090}

echo "[run.sh] Using camera ${DEVICE} at ${WIDTH}x${HEIGHT} @ ${FPS}fps; MJPEG passthrough on :${PORT}"

# Start pigpio daemon for servo control (if available)
if command -v pigpiod >/dev/null 2>&1; then
  if ! pgrep -x pigpiod >/dev/null 2>&1; then
    echo "[run.sh] Starting pigpiod"
    sudo pigpiod || true
  fi
else
  echo "[run.sh] Warning: pigpiod not found; pan/tilt may not work"
fi

# Configure the V4L2 device to MJPG at the requested resolution and FPS
if command -v v4l2-ctl >/dev/null 2>&1; then
  echo "[run.sh] Configuring ${DEVICE} to MJPG ${WIDTH}x${HEIGHT} @ ${FPS}fps"
  v4l2-ctl -d "${DEVICE}" --set-fmt-video=width=${WIDTH},height=${HEIGHT},pixelformat=MJPG || true
  v4l2-ctl -d "${DEVICE}" --set-parm=${FPS} || true
else
  echo "[run.sh] Warning: v4l2-ctl not found; skipping camera preconfiguration"
fi

# Start an ffmpeg mpjpeg HTTP server that passes through camera MJPEG stream (no re-encode)
if command -v ffmpeg >/dev/null 2>&1; then
  echo "[run.sh] Starting ffmpeg MJPEG server on port ${PORT} (passthrough)"
  # Kill any previous ffmpeg on the same port
  pkill -f "ffmpeg .*mpjpeg .*:${PORT}/stream.mjpg" >/dev/null 2>&1 || true
  (
    exec ffmpeg -hide_banner -loglevel warning -nostdin -fflags nobuffer \
      -f v4l2 -input_format mjpeg -framerate ${FPS} -video_size ${WIDTH}x${HEIGHT} -i "${DEVICE}" \
      -c:v copy -f mpjpeg -listen 1 "http://0.0.0.0:${PORT}/stream.mjpg"
  ) >/tmp/ffmpeg_mjpeg.log 2>&1 &
  echo "[run.sh] ffmpeg log: /tmp/ffmpeg_mjpeg.log"
else
  echo "[run.sh] Warning: ffmpeg not found; passthrough stream will be unavailable"
fi

# Start Flask app (threaded) so it doesn't block while streaming
echo "[run.sh] Starting Flask app"
uv run python -m flask run --host="0.0.0.0" --with-threads &
