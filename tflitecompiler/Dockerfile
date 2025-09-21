FROM python:3.11-slim

# Install system deps and Coral EdgeTPU compiler
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        build-essential \
        libgl1 \
        libglib2.0-0; \
    mkdir -p /usr/share/keyrings; \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends edgetpu-compiler; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv and Python deps via uv (no requirements.txt)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# App code
COPY app.py ./
COPY templates ./templates

# Expose web port
EXPOSE 8000

# Run with Flask's debugger enabled and no auto-reloader to avoid double runs
CMD ["uv", "run", "--no-project", "--with", "Flask==2.3.3", "--with", "opencv-python-headless", "--with", "ultralytics", "flask", "--app", "app", "run", "--debug", "--no-reload", "--host=0.0.0.0", "--port=8000"]
