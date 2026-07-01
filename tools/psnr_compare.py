"""PSNR comparison: Rust mid_rust.png vs PyTorch mid_pytorch.rgb (ground truth)."""
import numpy as np
from PIL import Image

rust_path = r"C:\Users\MONZik\Desktop\InterModule\demo\mid_rust.png"
py_rgb_path = r"C:\Users\MONZik\Desktop\InterModule\demo\mid_pytorch.rgb"

# Rust output (RGB PNG)
rust = np.array(Image.open(rust_path).convert("RGB"), dtype=np.float32)  # HxWx3 RGB

# PyTorch ground truth (raw RGB bytes)
py = np.fromfile(py_rgb_path, dtype=np.uint8).reshape(rust.shape[0], rust.shape[1], 3)
py = py.astype(np.float32)

assert rust.shape == py.shape, f"shape mismatch: {rust.shape} vs {py.shape}"

mse = np.mean((rust - py) ** 2)
psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
max_abs = np.max(np.abs(rust - py))
mean_abs = np.mean(np.abs(rust - py))

print(f"shape: {rust.shape}")
print(f"MSE:      {mse:.4f}")
print(f"PSNR:     {psnr:.2f} dB")
print(f"max |diff|: {max_abs}")
print(f"mean |diff|: {mean_abs:.4f}")
print(f"identical pixels: {np.sum(np.abs(rust - py) == 0)}/{rust.size} ({100*np.sum(np.abs(rust-py)==0)/rust.size:.1f}%)")

if psnr >= 35:
    print("\nPASS: PSNR >= 35 dB — port is numerically correct.")
elif psnr >= 30:
    print("\nCLOSE: PSNR >= 30 dB — minor numerical differences, likely OK.")
else:
    print("\nFAIL: PSNR < 30 dB — port has a bug.")
