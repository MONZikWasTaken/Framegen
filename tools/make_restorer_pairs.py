"""Generate SR-restorer training pairs: OUR interpolated mid (half-res, with the
real warp/halo artifacts) -> the true intermediate frame (full-res GT).

The shipped TinySR was trained on clean box-downscales, so it sharpens artifacts
as faithfully as detail. The restorer sees what it will actually eat in the
player: tfact2 outputs. Two motion regimes are sampled to match the wild:
  - stride-2, t=0.5  (factor x2, the dominant case)          ~70%
  - stride-4, t in {0.25, 0.5, 0.75} (real GT, bigger motion,
    off-center t like the display-Hz mode)                    ~30%

Output layout (E:\\data\\framegen\\restorer_pairs by default):
  train/<clip>_<i>_<code>.png   our mid, half-res, PNG (no recompression)
  train/pairs.txt               "<mid_rel>\t<gt_abs>" per line
  eval/...                      same scheme from the held-out BBB/jellyfish clips

Usage (training venv):
    python make_restorer_pairs.py [--limit 12000]
"""
import argparse
import os
import subprocess
import sys

os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";")
    if "nvidia gpu computing toolkit" not in p.lower())

import cv2
import numpy as np
import torch

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_student import load_ifnet
from train_tfact2 import TFact2

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLIM = r"E:\data\framegen\ckpt_1blk_slim\student_last.pkl"


def load_net(ckpt, device):
    ck = torch.load(ckpt, map_location="cpu")
    slim = load_ifnet(SLIM, device).eval()
    net = TFact2(slim.block0, ck["c"]).to(device)
    net.load_state_dict(ck["sd"])
    return net.eval()


@torch.no_grad()
def run_mid(net, f0, f1, t, device):
    """f0/f1: HxWx3 BGR uint8 (already half-res, even dims) -> mid uint8 same size."""
    h, w = f0.shape[:2]
    ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    x = torch.zeros(2, 3, ph, pw, device=device)
    x[0, :, :h, :w] = torch.from_numpy(f0.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    x[1, :, :h, :w] = torch.from_numpy(f1.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
    pred, _ = net(x[:1], x[1:], torch.tensor([t], device=device))
    return (pred[0, :, :h, :w].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)


def half(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\data\framegen\frames_v060\frames")
    ap.add_argument("--out", default=r"E:\data\framegen\restorer_pairs")
    ap.add_argument("--ckpt", default=r"E:\data\framegen\ckpt_big_v060\tfact2_best.pt")
    ap.add_argument("--limit", type=int, default=12000)
    args = ap.parse_args()

    device = torch.device("cuda")
    net = load_net(args.ckpt, device)
    rng = np.random.default_rng(7)

    # ---- train pairs from the frames dataset ----
    items = []  # (dir, center_idx, max_idx)
    for stem in sorted(os.listdir(args.data)):
        idx_file = os.path.join(args.data, stem, "triplets.txt")
        if not os.path.isfile(idx_file):
            continue
        d = os.path.join(args.data, stem)
        idxs = [int(l) for l in open(idx_file) if l.strip()]
        mx = max(idxs) + 1
        items += [(d, i, mx) for i in idxs]
    if len(items) > args.limit:
        keep = rng.choice(len(items), args.limit, replace=False)
        items = [items[k] for k in sorted(keep)]

    tdir = os.path.join(args.out, "train")
    os.makedirs(tdir, exist_ok=True)
    lines, done = [], 0
    for d, i, mx in items:
        clip = os.path.basename(d)
        rd = lambda k: cv2.imread(os.path.join(d, f"{k:06d}.jpg"))
        stride4 = rng.random() < 0.3 and i - 2 >= 0 and i + 2 <= mx
        if stride4:
            k = int(rng.integers(1, 4))  # GT at i-2+k, t = k/4
            a, b, g, t = rd(i - 2), rd(i + 2), rd(i - 2 + k), k / 4.0
            code = f"s4t{k}"
        else:
            a, b, g, t = rd(i - 1), rd(i + 1), rd(i), 0.5
            code = "s2"
        if a is None or b is None or g is None:
            continue
        h, w = g.shape[:2]
        h, w = h - h % 2, w - w % 2  # even dims so half-res aligns exactly 2:1
        mid = run_mid(net, half(a[:h, :w]), half(b[:h, :w]), t, device)
        name = f"{clip}_{i:06d}_{code}.png"
        cv2.imwrite(os.path.join(tdir, name), mid)
        lines.append(f"{name}\t{os.path.join(d, f'{i - 2 + k:06d}.jpg') if stride4 else os.path.join(d, f'{i:06d}.jpg')}")
        done += 1
        if done % 500 == 0:
            print(f"train {done}/{len(items)}", flush=True)
    with open(os.path.join(tdir, "pairs.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"train done: {done} pairs")

    # ---- eval pairs from the held-out clips (GT saved as PNG too) ----
    edir = os.path.join(args.out, "eval")
    os.makedirs(edir, exist_ok=True)
    W, H = 1280, 720
    elines = []
    for clipf in ("bbb_720_10s.mp4", "jellyfish_720_10s.mp4"):
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(REPO, "assets", clipf),
                            "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
        raw = np.frombuffer(p.stdout, np.uint8)
        n = len(raw) // (W * H * 3)
        frames = raw[: n * W * H * 3].reshape(n, H, W, 3)
        stem = clipf.split("_")[0]
        for i in range(1, n - 1, 25):
            mid = run_mid(net, half(frames[i - 1]), half(frames[i + 1]), 0.5, device)
            mname, gname = f"{stem}_{i:04d}_mid.png", f"{stem}_{i:04d}_gt.png"
            cv2.imwrite(os.path.join(edir, mname), mid)
            cv2.imwrite(os.path.join(edir, gname), frames[i])
            elines.append(f"{mname}\t{gname}")
    with open(os.path.join(edir, "pairs.txt"), "w") as f:
        f.write("\n".join(elines) + "\n")
    print(f"eval done: {len(elines)} pairs")


if __name__ == "__main__":
    main()
