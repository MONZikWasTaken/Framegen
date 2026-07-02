"""PSNR quality bench for RIFE ONNX variants on real video triplets.

For frames (f[i-1], f[i+1]) predict f[i], compare to the real f[i].
Reports mean PSNR per model + trivial floors (copy-prev, average-of-neighbors).

Usage: python quality_bench.py <video> <model.onnx> [more.onnx ...] [--step N]
"""
import subprocess
import sys

import numpy as np
import onnxruntime as ort

args = [a for a in sys.argv[1:] if not a.startswith("--")]
step = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--step=")), 6))
video, models = args[0], args[1:]
W, H = 1280, 720
PW, PH = 1280, 736


def read_frames(path):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        capture_output=True, check=True)
    raw = np.frombuffer(p.stdout, np.uint8)
    n = len(raw) // (W * H * 3)
    return raw[: n * W * H * 3].reshape(n, H, W, 3)


def to_input(bgr):  # HWC uint8 BGR -> [1,3,PH,PW] float32, zero-pad bottom/right
    x = np.zeros((1, 3, PH, PW), np.float32)
    x[0, :, :H, :W] = bgr.transpose(2, 0, 1).astype(np.float32) / 255.0
    return x


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


frames = read_frames(video)
idx = list(range(1, len(frames) - 1, step))
print(f"{video}: {len(frames)} frames, {len(idx)} triplets (step={step})")

floors = {"copy-prev": [], "avg-neighbors": []}
for i in idx:
    floors["copy-prev"].append(psnr(frames[i - 1], frames[i]))
    floors["avg-neighbors"].append(
        psnr(((frames[i - 1].astype(np.uint16) + frames[i + 1]) // 2).astype(np.uint8), frames[i]))
for k, v in floors.items():
    print(f"{k:>28}: PSNR {np.mean(v):6.2f} dB")

for m in models:
    sess = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
    scores = []
    for i in idx:
        out = sess.run(None, {"img0": to_input(frames[i - 1]), "img1": to_input(frames[i + 1])})[0]
        pred = (np.clip(out[0, :, :H, :W], 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        scores.append(psnr(pred, frames[i]))
    name = m.split("\\")[-1].split("/")[-1]
    print(f"{name:>28}: PSNR {np.mean(scores):6.2f} dB  (min {np.min(scores):5.2f})")
