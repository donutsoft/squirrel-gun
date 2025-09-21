#!/usr/bin/env bash
set -euo pipefail

# Ensure uv is on PATH (installed under /root/.local/bin by the Dockerfile)
export PATH="/root/.local/bin:${PATH}"

echo "Starting EdgeTPU export service..."

if ! python -c "import ultralytics" >/dev/null 2>&1; then
  echo "Ultralytics not found. Installing with uv..."
  uv pip install --system ultralytics
fi

exec python /app/app.py

