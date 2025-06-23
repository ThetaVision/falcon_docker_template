#!/usr/bin/env python3
"""
Live camera + MobileNetV3-Small demo for Jetson Orin Nano
--------------------------------------------------------

* Shows a camera preview in an OpenCV window
* Runs MobileNetV3-Small on each frame (224×224 center crop)
* Overlays the top-1 ImageNet class label + FPS
"""
import cv2
import time
import torch
import torchvision.transforms as T
import torchvision.models as models

# ---------------------------------------------------------------------
# 1.  Initial checks
# ---------------------------------------------------------------------
assert torch.cuda.is_available(), "CUDA not visible – did you run with --gpus all ?"
device = torch.device("cuda")

print("CUDA device:", torch.cuda.get_device_name(0))

# ---------------------------------------------------------------------
# 2.  Pre-load model & labels
# ---------------------------------------------------------------------
model = models.mobilenet_v3_small(weights="IMAGENET1K_V1").to(device)
model.eval()                                   # inference mode

# ImageNet labels
with open("/usr/local/share/imagenet_classes.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]

# Pre-processing pipeline
transform = T.Compose([
    T.ToPILImage(),
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------
# 3.  Camera setup
# ---------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Could not open /dev/video0 – check camera & permissions")

# Warm up
for _ in range(5):
    cap.read()

fps_hist = []

# ---------------------------------------------------------------------
# 4.  Main loop
# ---------------------------------------------------------------------
try:
    while True:
        tic = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Stream ended / camera disconnected")
            break

        # Keep original for display; prepare a copy for inference
        img = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img)
            top1_id = logits.argmax(dim=1).item()
            top1_label = labels[top1_id]

        # -----------------------------------------------------------------
        # 5.  GUI overlay & show
        # -----------------------------------------------------------------
        fps = 1.0 / max(time.time() - tic, 1e-3)
        fps_hist.append(fps)
        if len(fps_hist) > 20:       # moving window
            fps_hist.pop(0)
        cv2.putText(frame, f"{top1_label}  ({sum(fps_hist)/len(fps_hist):.1f} FPS)",
                    (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.imshow("Jetson-PyTorch-Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:   # ESC to quit
            break

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
