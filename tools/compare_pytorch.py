"""Ground-truth reference: one RIFE_m interpolation at timestep=0.5, save raw RGB bytes + float .npy.

Mirrors inference_img.py single-step path. Output saved as both:
  - mid_pytorch.png   (uint8 BGR, cv2.imwrite - matches original)
  - mid_pytorch.rgb    (raw uint8 RGB, row-major, for byte-exact PSNR vs Rust)
"""
import sys
import os
import numpy as np
import cv2
import torch

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
from model.RIFE import Model

WEIGHTS = r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_m\RIFE_m_train_log"
IMG0 = r"C:\Users\MONZik\Desktop\InterModule\demo\I0_0.png"
IMG1 = r"C:\Users\MONZik\Desktop\InterModule\demo\I0_1.png"
OUT_PNG = r"C:\Users\MONZik\Desktop\InterModule\demo\mid_pytorch.png"
OUT_RGB = r"C:\Users\MONZik\Desktop\InterModule\demo\mid_pytorch.rgb"

torch.set_grad_enabled(False)
device = torch.device("cpu")

# pkl was saved from CUDA; force CPU load.
_orig_load = torch.load
torch.load = lambda *a, **k: _orig_load(*a, **{**k, "map_location": "cpu"})

model = Model(arbitrary=True)          # arbitrary=True -> IFNet_m (RIFEm / lite)
model.load_model(WEIGHTS, -1)
model.eval()
model.device()

img0 = cv2.imread(IMG0, cv2.IMREAD_UNCHANGED)   # BGR uint8
img1 = cv2.imread(IMG1, cv2.IMREAD_UNCHANGED)
img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

n, c, h, w = img0.shape
ph = ((h - 1) // 32 + 1) * 32
pw = ((w - 1) // 32 + 1) * 32
padding = (0, pw - w, 0, ph - h)
img0 = torch.nn.functional.pad(img0, padding)
img1 = torch.nn.functional.pad(img1, padding)

mid = model.inference(img0, img1, timestep=0.5)   # [1,3,ph,pw] BGR float [0,1]
mid = mid[0, :, :h, :w]                            # crop to original

# PNG (BGR uint8, matches original cv2.imwrite)
cv2.imwrite(OUT_PNG, (mid * 255).byte().cpu().numpy().transpose(1, 2, 0))

# Raw RGB uint8 (swap BGR->RGB for Rust comparison)
mid_rgb = mid[[2, 1, 0], :, :]                     # BGR -> RGB
raw = (mid_rgb * 255).byte().cpu().numpy().transpose(1, 2, 0).tobytes()
with open(OUT_RGB, "wb") as f:
    f.write(raw)

print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_RGB} ({len(raw)} bytes, {w}x{h}x3 RGB)")
print(f"mid range: min={mid.min():.4f} max={mid.max():.4f} mean={mid.mean():.4f}")
