#!/usr/bin/env python3
"""
Jetson Orin Nano – MobileNetV3 live-camera demo
==============================================
* Runs fully offline (weights are baked into the Docker image).
* Uses TorchVision’s built-in `weights.transforms()`, so there’s
  no hard-coded mean/std and no missing-file issues.
* Displays top-1 ImageNet label and a moving-average FPS counter.

Press **Esc** to quit the window.
"""
from __future__ import annotations

import time

import cv2
import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def main() -> None:
    # ------------------------------------------------------------------
    # 1.  CUDA check
    # ------------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not visible – did you start the container with '--gpus all'?"
        )
    device = torch.device("cuda")
    print("CUDA device:", torch.cuda.get_device_name(0))

    # ------------------------------------------------------------------
    # 2.  Model + preprocessing pipeline
    # ------------------------------------------------------------------
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights).to(device).eval()
    labels: list[str] = weights.meta["categories"]
    preprocess = weights.transforms()  # Resize → CenterCrop → ToTensor → Normalize

    # ------------------------------------------------------------------
    # 3.  Camera
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError("Unable to open /dev/video0 – check camera index/permissions")

    # Warm-up for exposure/white-balance
    for _ in range(4):
        cap.read()

    fps_window: list[float] = []

    try:
        while True:
            tic = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            # BGR → RGB for TorchVision and apply transforms
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inp = preprocess(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(inp)
                class_id: int = int(pred.argmax(1))
                label = labels[class_id]

            # FPS (moving average of last 20 frames)
            fps_window.append(1.0 / max(time.time() - tic, 1e-3))
            if len(fps_window) > 20:
                fps_window.pop(0)
            fps = sum(fps_window) / len(fps_window)

            # Overlay and display
            cv2.putText(
                frame,
                f"{label}  ({fps:.1f} FPS)",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Jetson-MobileNetV3", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
