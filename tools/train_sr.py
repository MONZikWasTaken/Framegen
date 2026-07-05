"""Train a tiny 2x super-resolution net for the player's present path (anime upscale).

Residual-vs-bilinear design: out = bilinear2x(x) + detail(x), so the net only learns
the missing high frequencies - tiny (3 convs, c channels) and stable. Trained on the
same movie frames as the interpolation students: GT = random full-res crop, input =
2x box-downscale. BGR /255 domain, matching the rest of the pipeline.

Usage (training venv):
    python train_sr.py --data E:\\data\\framegen\\frames --out E:\\data\\framegen\\ckpt_sr
Checkpoint: sr_last.pt / sr_best.pt (plain state_dict).
"""
import argparse
import math
import os
import random
import subprocess
import time

os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";") if "nvidia gpu computing toolkit" not in p.lower())

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TinySR(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.c1 = nn.Conv2d(3, c, 3, 1, 1)
        self.a1 = nn.PReLU(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.a2 = nn.PReLU(c)
        self.c3 = nn.Conv2d(c, c, 3, 1, 1)
        self.a3 = nn.PReLU(c)
        self.c4 = nn.Conv2d(c, 12, 3, 1, 1)  # 12 = 3ch x (2x2) pixel shuffle
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.a1(self.c1(x))
        d = self.a2(self.c2(d))
        d = self.a3(self.c3(d))
        return base + self.shuffle(self.c4(d))


class SRData(Dataset):
    def __init__(self, root, crop=192):
        self.crop = crop
        self.items = []
        for stem in os.listdir(root):
            d = os.path.join(root, stem)
            if not os.path.isfile(os.path.join(d, "triplets.txt")):
                continue
            with open(os.path.join(d, "triplets.txt")) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append((d, int(line)))
        if not self.items:
            raise RuntimeError("no frames")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        d, i = self.items[k]
        img = cv2.imread(os.path.join(d, f"{i:06d}.jpg"))
        h, w = img.shape[:2]
        c = self.crop
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        gt = img[y:y + c, x:x + c]
        if random.random() < 0.5:
            gt = gt[:, ::-1]
        lo = cv2.resize(gt, (c // 2, c // 2), interpolation=cv2.INTER_AREA)
        to = lambda a: torch.from_numpy(
            np.ascontiguousarray(a.transpose(2, 0, 1))).float() / 255.0
        return to(lo), to(gt)


def psnr_t(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


@torch.no_grad()
def evaluate(net, frames, device):
    """PSNR of net-2x vs GT on held-out BBB frames (downscaled inputs), vs bilinear."""
    sn, sb = [], []
    for f in frames:
        gt = torch.from_numpy(f.transpose(2, 0, 1)).float().div(255)[None].to(device)
        lo = F.interpolate(gt, scale_factor=0.5, mode="area")
        up = net(lo).clamp(0, 1)
        base = F.interpolate(lo, scale_factor=2, mode="bilinear", align_corners=False).clamp(0, 1)
        sn.append(psnr_t(up, gt))
        sb.append(psnr_t(base, gt))
    return float(np.mean(sn)), float(np.mean(sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\data\framegen\frames")
    ap.add_argument("--out", default=r"E:\data\framegen\ckpt_sr")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # held-out eval frames from BBB (not in the training set)
    W, H = 1280, 720
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(REPO, "assets", "bbb_720_10s.mp4"),
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    raw = np.frombuffer(p.stdout, np.uint8)
    n = len(raw) // (W * H * 3)
    ev = raw[: n * W * H * 3].reshape(n, H, W, 3)[::40][:8]

    net = TinySR(args.channels).to(device)
    data = SRData(args.data)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    s0, sb = evaluate(net, ev, device)
    log(f"init: net={s0:.2f} dB · bilinear={sb:.2f} dB | {len(data)} crops, {args.steps} steps")
    best = s0

    step, t0, run = 0, time.time(), 0.0
    while step < args.steps:
        for lo, gt in loader:
            if step >= args.steps:
                break
            lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * step / args.steps))
            for g in opt.param_groups:
                g["lr"] = lr
            lo = lo.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            out = net(lo)
            loss = F.l1_loss(out, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
            if step % 200 == 0:
                log(f"step {step}/{args.steps} loss={run / 200:.5f} {step * args.batch / (time.time() - t0):.0f} img/s")
                run = 0.0
            if step % args.eval_every == 0:
                sn, _ = evaluate(net, ev, device)
                torch.save(net.state_dict(), os.path.join(args.out, "sr_last.pt"))
                mark = ""
                if sn > best:
                    best = sn
                    torch.save(net.state_dict(), os.path.join(args.out, "sr_best.pt"))
                    mark = "  ** best"
                log(f"eval @ {step}: net={sn:.2f} dB (bilinear {sb:.2f}){mark}")

    log(f"done. best={best:.2f} dB vs bilinear {sb:.2f}")


if __name__ == "__main__":
    main()
