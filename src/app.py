#!/usr/bin/env python3
"""
Self‑contained MobileNetV3 camera demo for Jetson
------------------------------------------------
• No runtime internet needed – weights are pre‑fetched during the Docker build.
• ImageNet labels come from TorchVision’s built‑in metadata (no external file).
"""
import time
import cv2
import torch
import torchvision.transforms as T
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# -----------------------------------------------------------------------------
# 1.  CUDA / device setup
# -----------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not visible – did you run the container with --gpus all ?")

device = torch.device("cuda")
print("CUDA device:", torch.cuda.get_device_name(0))

# -----------------------------------------------------------------------------
# 2.  Load model + labels (weights already cached inside the image)
# -----------------------------------------------------------------------------
weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights).to(device).eval()
labels = weights.meta["categories"]  # 1000‑class ImageNet names

# Normalisation values are also provided by the weights metadata
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=weights.meta["mean"], std=weights.meta["std"]),
])

# -----------------------------------------------------------------------------
# 3.  Camera
# -----------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0 – check camera & permissions")

# Warm‑up
for _ in range(4):
    cap.read()

fps_hist = []
try:
    while True:
        tic = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        img = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            top1 = logits.argmax(1).item()
            label = labels[top1]

        fps = 1.0 / max(time.time() - tic, 1e-3)
        fps_hist.append(fps)
        if len(fps_hist) > 20:
            fps_hist.pop(0)

        cv2.putText(frame,
                    f"{label}  ({sum(fps_hist)/len(fps_hist):.1f} FPS)",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Jetson‑MobileNetV3", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
