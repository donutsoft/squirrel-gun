FROM python:3.11-slim

# Install system deps and Coral EdgeTPU compiler
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        build-essential \
        libglib2.0-0 \
        libgl1; \
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
RUN uv pip install --system Flask==2.3.3 opencv-python-headless ultralytics

# App code
COPY app.py ./
COPY templates ./templates

# Expose web port
EXPOSE 8000

# Run the web service
CMD ["python", "app.py"]
