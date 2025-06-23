#!/usr/bin/env python3
"""
Jetson Orin Nano – MobileNetV3 live-camera demo
==============================================
• Offline-ready (weights baked into the Docker image)
• Uses TorchVision’s preset transforms (no manual mean/std)
• Shows top-1 ImageNet class and moving-average FPS

Press Esc to quit.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  CUDA availability
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not visible – did you start the container with '--gpus all'?")
    device = torch.device("cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # 2.  Model and preprocessing
    # ------------------------------------------------------------------
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).to(device).eval()
    labels: list[str] = weights.meta["categories"]
    preprocess = weights.transforms()      # Resize → CenterCrop → ToTensor → Normalize

    # ------------------------------------------------------------------
    # 3.  Camera setup
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Unable to open /dev/video0 – check camera index/permissions")

    # Warm-up a few frames for exposure stabilisation
    for _ in range(4):
        cap.read()

    fps_window: list[float] = []

    try:
        while True:
            tic = time.time()
            ok, frame_bgr = cap.read()
            if not ok:
                break

            # ---------- Preprocess ----------
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)          # ← convert to PIL
            tensor = preprocess(img_pil).unsqueeze(0).to(device)

            # ---------- Inference ----------
            with torch.no_grad():
                logits = model(tensor)
                class_id = int(logits.argmax(1))
                label = labels[class_id]

            # ---------- FPS ----------
            fps_window.append(1.0 / max(time.time() - tic, 1e-3))
            if len(fps_window) > 20:
                fps_window.pop(0)
            fps = sum(fps_window) / len(fps_window)

            # ---------- Overlay ----------
            cv2.putText(
                frame_bgr,
                f"{label}  ({fps:.1f} FPS)",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Jetson-MobileNetV3", frame_bgr)

            if cv2.waitKey(1) & 0xFF == 27:  # Esc key
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
