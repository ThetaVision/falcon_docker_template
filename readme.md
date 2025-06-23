# Jetson Orin Nano ‑ GUI + PyTorch + TensorRT Docker

> **JetPack 6.2 / L4T r36.4.3** · tested on Jetson Orin Nano 8 GB (Ubuntu 22.04)

This repository shows how to run a hardware‑accelerated PyTorch application that opens a native window on the Jetson‑attached display **from inside a Docker container**. It also exposes every common peripheral (CSI/USB cameras, GPIO/I²C/SPI, audio, NVENC, etc.) so you can treat the container like the bare‑metal host.

The sample app (`src/app.py`) captures a live camera feed, runs a pretrained **MobileNetV3‑Small** on the GPU via cuDNN, and overlays the top‑1 ImageNet label plus FPS using OpenCV.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Directory structure](#directory-structure)
3. [Quick start](#quick-start)
4. [How it works](#how-it-works)
5. [Customising the image](#customising-the-image)
6. [Troubleshooting & FAQ](#troubleshooting--faq)
7. [Updating for new JetPack releases](#updating-for-new-jetpack-releases)
8. [Credits & useful links](#credits--useful-links)

---

## Prerequisites

| Item                       | Version                | Notes                                                     |                        |
| -------------------------- | ---------------------- | --------------------------------------------------------- | ---------------------- |
| JetPack (host)             | **6.2** (L4T r36.4.\*) | \`dpkg -l | grep nvidia-l4t-core\`                        | grep nvidia-l4t-core\` |
| Docker Engine              | ≥ 24 CE                | Installed automatically by JetPack 6 but update if needed |                        |
| `nvidia‑container‑runtime` | ships with JetPack     | Confirms GPU pass‑through                                 |                        |
| Internet (first run)       | optional               | Downloads the torchvision weights                         |                        |

**Hardware tested** : Orin Nano 8 GB DevKit ‑ but the exact same setup works on Orin NX / AGX Orin so long as JetPack 6.2.

---

## Directory structure

```
.
├── Dockerfile           # base image + GUI & PyTorch libs
├── run.sh               # convenience wrapper for docker run
├── requirements.txt     # optional, your extra PyPI deps
└── src/
    └── app.py           # demo application (camera + MobileNetV3)
```

> **Tip:** commit your models to the `models/` folder and mount it with `-v $(pwd)/models:/workspace/models`.

---

## Quick start

```bash
# 1 — clone / download this repo on the Jetson
cd jetson-gui-docker

# 2 — build the image (≈ 5 min on Nano, mostly PyPI wheels)
docker build -t myorg/jetson-gui:latest .

# 3 — allow root containers to access the X server (run once per boot)
xhost +local:root

# 4 — run it!
./run.sh                 # opens an OpenCV window on the HDMI/DP screen
```

You should see live camera video with a green class‑label overlay and an FPS counter. Press **Esc** or `Ctrl‑C` in the terminal to quit.

---

## How it works

| Layer                | What we do                                                                               | Why                                                                           |
| -------------------- | ---------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Base image**       | `dustynv/l4t-pytorch:r36.4.3`                                                            | Matches JetPack 6.2 ABI; bundles CUDA 12, cuDNN 9, TensorRT 8.6, PyTorch 2.3. |
| **GUI support**      | Install `python3-opencv` + GTK helpers and mount `/tmp/.X11‑unix` + `$DISPLAY`.          | `cv2.imshow()` opens a native X11 window; GL/EGL surfaces still accelerated.  |
| **GPU / TensorRT**   | `--gpus all` + env `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.                   | Enables CUDA cores + NVENC/NVDEC + Video4Linux in‑container.                  |
| **Peripherals**      | `--privileged` (easiest) **or** granular `--device /dev/video0 --device /dev/i2c‑1 ...`. | Cameras, GPIO, I²C, etc. appear exactly as on host.                           |
| **IPC optimisation** | `--ipc host`                                                                             | Lets PyTorch share CUDA tensors across forked workers.                        |

---

## Customising the image

### Adding Python dependencies

Edit `requirements.txt` **or** append new `pip install` lines inside `Dockerfile`:

```Dockerfile
RUN pip install --no-cache-dir pyserial==3.5 pandas==2.2.2
```

### Using your own application code

Replace `src/app.py` with your project, or mount a host folder read‑write:

```bash
docker run ... -v $(pwd)/my_app:/workspace/app myorg/jetson-gui
```

### Slimming security surface

Instead of `--privileged`, cherry‑pick exactly what you need:

```bash
--device /dev/video0 \        # CSI or USB camera
--device /dev/i2c-1 \         # IMU, touch panel, etc.
--device /dev/gpiochip0 \
--cap-add SYS_RAWIO           # for GPIO bit‑banging
```

### Wayland (optional)

If your host runs Weston or GNOME Wayland, set:

```bash
-e WAYLAND_DISPLAY=$WAYLAND_DISPLAY \
-v $XDG_RUNTIME_DIR/$WAYLAND_DISPLAY:/tmp/$WAYLAND_DISPLAY \
```

OpenCV with GTK works the same; Qt/SDL apps need the `qtwayland5` package.

---

## Troubleshooting & FAQ

| Problem                                                      | Fix                                                                                                                    |                                                                       |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| ``** → **``                                                  | (1) Tag mismatch – make sure the image tag (e.g. `r36.4.3`) matches \`dpkg -l                                          | grep nvidia-l4t-core`.  (2) Forgot `--gpus all`or`--runtime nvidia\`. |
| OpenCV window blank white                                    | Install `libgtk-3-0` &  `libcanberra-gtk3-module` (already in Dockerfile).                                             |                                                                       |
| *Cannot open /dev/video0*                                    | Camera busy on host (`argus_camera`), wrong index, or missing `--device /dev/video0`.                                  |                                                                       |
| *“no kernel image is available for execution on the device”* | You built against a different CUDA major/minor. Re‑pull an image matching host JetPack or rebuild PyTorch from source. |                                                                       |
| Model download behind proxy                                  | Pre‑download weights and COPY them into the image, or set `TORCH_HOME=/workspace/models`.                              |                                                                       |

---

## Updating for new JetPack releases

1. Check your host’s new L4T version, e.g. `36.4.3`.
2. In `Dockerfile`, set `ARG L4T_TAG=r36.4.3` **and** update the `FROM dustynv/l4t-pytorch:${L4T_TAG}` line.
3. Rebuild the image. If TensorRT SONAME changed, rebuild any hand‑crafted `.engine` files or ONNX→TRT conversions.
4. Retest with the demo.

When NVIDIA publishes official NGC images (“nvcr.io/nvidia/l4t-pytorch”), switch the `FROM` line and drop the community tag.

---

## Credits & useful links

- Dusty Nvidia’s Jetson container registry – [https://hub.docker.com/r/dustynv](https://hub.docker.com/r/dustynv)
- Official **Jetson Linux R36 documentation** – [https://docs.nvidia.com/jetson/](https://docs.nvidia.com/jetson/)
- Jetson DIY community forum – [https://forums.developer.nvidia.com/c/agx-orin/](https://forums.developer.nvidia.com/c/agx-orin/)
- PyTorch → TensorRT integration – [https://pytorch.org/TensorRT](https://pytorch.org/TensorRT)

---

### License

This sample is released under the MIT License. See `LICENSE` for details.

