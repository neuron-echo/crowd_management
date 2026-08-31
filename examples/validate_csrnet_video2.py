"""Run CSRNet on every frame of video samples/2.mp4 using the EXACT same
multi-scale inference pipeline as the original csrnetb.py (paper code).

Scales: [0.7, 1.0, 1.3] with scale correction count / (s*s)
Final count: mean across scales — identical to paper.
"""

import sys
import torch
sys.path.insert(0, '.')

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T

from crowd_management.models.csrnet.weights import build_csrnet_model, resolve_checkpoint_path

# ── Load model (same as paper: load_weights=True to skip VGG re-download) ──
ckpt  = resolve_checkpoint_path(None)
model = build_csrnet_model(str(ckpt), device='cpu', strict_checkpoint=True)

# ── Identical transform to csrnetb.py ──────────────────────────────────────
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

SCALES = [0.7, 1.0, 1.3]   # exactly as in csrnetb.py

def multiscale_count(pil_img: Image.Image) -> float:
    """Replicate csrnetb.py multi-scale inference with scale correction."""
    orig_w, orig_h = pil_img.size
    counts = []
    for s in SCALES:
        new_w = int(orig_w * s)
        new_h = int(orig_h * s)
        # enforce multiple of 8 (CSRNet downsampling factor)
        new_w = new_w - new_w % 8
        new_h = new_h - new_h % 8
        scaled = pil_img.resize((new_w, new_h))
        tensor = transform(scaled).unsqueeze(0)
        with torch.no_grad():
            density = model(tensor)
        raw_count = density.sum().item()
        # scale correction — same as paper
        corrected = raw_count / (s * s)
        counts.append(corrected)
    return float(np.mean(counts))

# ── Video ────────────────────────────────────────────────────────────────────
cap   = cv2.VideoCapture('video samples/2.mp4')
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps   = cap.get(cv2.CAP_PROP_FPS)

print(f"Video  : video samples/2.mp4")
print(f"Frames : {total}  |  FPS: {fps:.1f}  |  Duration: {total/fps:.1f}s")
print(f"Method : Multi-scale {SCALES} + scale correction (same as csrnetb.py)")
print()
print(f"{'Frame':>6}  {'Time(s)':>8}  {'Crowd Count':>12}")
print("-" * 35)

frame_id = 0
all_counts = []

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    count = multiscale_count(pil_img)
    all_counts.append(count)
    t_sec = frame_id / fps

    print(f"{frame_id:>6}  {t_sec:>8.2f}  {count:>12.1f}", flush=True)
    frame_id += 1

cap.release()

print("-" * 35)
print(f"Frames processed : {frame_id}")
print(f"Min count        : {min(all_counts):.1f}")
print(f"Max count        : {max(all_counts):.1f}")
print(f"Avg count        : {sum(all_counts)/len(all_counts):.1f}")
