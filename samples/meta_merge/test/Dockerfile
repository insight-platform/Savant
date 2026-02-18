FROM ghcr.io/insight-platform/savant-rs-py314:savant-latest

# libxcb + libgl required by opencv (ultralytics dep) in headless container
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with CUDA for GPU inference (cu130 for Python 3.14), then ultralytics
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu130 \
    && pip install --no-cache-dir Pillow ultralytics>=8.0.0

# Pre-download YOLO model to /opt/models (avoids read-only /app at runtime)
RUN mkdir -p /opt/models && cd /opt/models && python -c "from ultralytics import YOLO; YOLO('yolo11m.pt')"
