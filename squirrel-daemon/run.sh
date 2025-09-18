sudo pigpiod
# Enable threaded dev server so long-lived stream doesn't block other requests
uv run python -m flask run --host="0.0.0.0" --with-threads &
