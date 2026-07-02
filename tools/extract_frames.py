"""Extract movie frames to JPEGs + build a filtered triplet index for distillation training.

Keeps native resolution (crops are taken at train time). A triplet (i-1, i, i+1) is kept when
PSNR(f[i-1], f[i+1]) is between CUT_DB (scene cut / chaos) and STATIC_DB (nothing moves) —
both computed on a 1/4-scale grayscale for speed.

Usage (training venv):
    python extract_frames.py <movie> [more movies ...] [--out=E:\\data\\framecast]
Writes frames to  <out>/frames/<stem>/%06d.jpg
and the index to  <out>/frames/<stem>/triplets.txt (one frame number per line).
"""
import os
import subprocess
import sys

import cv2
import numpy as np

CUT_DB, STATIC_DB = 16.0, 45.0

out_root = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--out=")), r"E:\data\framecast")
movies = [a for a in sys.argv[1:] if not a.startswith("--")]


def probe_wh(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")
    return int(w), int(h)


def psnr_small(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    mse = np.mean(d * d)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


for movie in movies:
    stem = os.path.splitext(os.path.basename(movie))[0]
    W, H = probe_wh(movie)
    frame_dir = os.path.join(out_root, "frames", stem)
    os.makedirs(frame_dir, exist_ok=True)

    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", movie, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE)
    small_prev = []  # ring of last two 1/4-scale grays
    keep = []
    i = 0
    while True:
        buf = proc.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        cv2.imwrite(os.path.join(frame_dir, f"{i:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        small = cv2.cvtColor(cv2.resize(frame, (W // 4, H // 4)), cv2.COLOR_BGR2GRAY)
        if len(small_prev) == 2:
            p = psnr_small(small_prev[0], small)  # f[i-2] vs f[i] -> triplet centered at i-1
            if CUT_DB < p < STATIC_DB:
                keep.append(i - 1)
        small_prev = (small_prev + [small])[-2:]
        i += 1
    proc.wait()

    with open(os.path.join(frame_dir, "triplets.txt"), "w") as f:
        f.write("\n".join(str(k) for k in keep))
    print(f"{stem}: {i} frames ({W}x{H}), {len(keep)} usable triplets "
          f"({100 * len(keep) / max(1, i - 2):.0f}%)")
