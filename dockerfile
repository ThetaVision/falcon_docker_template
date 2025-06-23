# -----------------------------------------------------------------------------
# Jetson Orin Nano / JetPack 6.2  (L4T r36.4.x)
# GUI-capable container with PyTorch + TensorRT
# -----------------------------------------------------------------------------
ARG L4T_TAG=r36.4.0          
# keep in sync with host JetPack minor
FROM dustynv/l4t-pytorch:${L4T_TAG}

# PyTorch 2.3, CUDA 12.x

LABEL maintainer="cviviers@thetavision.nl"
LABEL description="Jetson Orin Nano / JetPack 6.2 GUI-capable container with PyTorch + TensorRT"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
# Ubuntu packages for GUI + GStreamer camera IO
# --------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-opencv \
        libcanberra-gtk3-module \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        gstreamer1.0-plugins-base \
        v4l-utils \
        libgstreamer-plugins-base1.0-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------
# PyPI: TensorRT front-ends & extra DL utilities
# --------------------------------------------------------------------------
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
        nvidia-pyindex torch-tensorrt pycuda==2024.1 \
        opencv-python==4.10.0.82 \
        matplotlib  tqdm  pillow

# --------------------------------------------------------------------------
# Your application
# --------------------------------------------------------------------------
WORKDIR /workspace/app
# COPY requirements.txt .       
# RUN pip install --no-cache-dir -r requirements.txt || true
# COPY src/ .

CMD ["python3", "app.py"]
