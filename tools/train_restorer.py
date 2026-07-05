"""Train the SR-restorer: same TinySR topology as the shipped 2x SR, but on REAL
pairs - our tfact2 mid at half-res in, the true frame at full-res out. The net
learns to upscale AND repair interpolation artifacts (halo, ghosting) instead of
faithfully sharpening them.

Training mix (matches how the extension uses SR):
  ~70% restorer pairs  (mid.png -> GT jpg)      - the FG-on path (mids only)
  ~30% clean SR pairs  (box-downscaled GT -> GT) - the FG-off pure-upscaler path

Eval: held-out BBB/jellyfish mids (restorer_pairs/eval) - PSNR of net(mid) vs GT,
against bilinear2x and the SHIPPED sr_best baseline, printed every eval.

Arch options: --channels 16 --mid-convs 2 is a drop-in weight swap for the
extension (same 3.68ms@480p). --channels 24 --mid-convs 1 is the alt candidate
(needs a small sr.js tweak; bench before shipping).

Usage (training venv):
    python train_restorer.py --out E:\\data\\framegen\\ckpt_restorer
"""
import argparse
import math
import os
import random
import time

os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";") if "nvidia gpu computing toolkit" not in p.lower())

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TinySR(nn.Module):
    """out = bilinear2x(x) + shuffle(detail(x)); mid_convs=2 == the shipped arch."""

    def __init__(self, c=16, mid_convs=2):
        super().__init__()
        self.c1 = nn.Conv2d(3, c, 3, 1, 1)
        self.a1 = nn.PReLU(c)
        self.mids = nn.ModuleList(nn.Conv2d(c, c, 3, 1, 1) for _ in range(mid_convs))
        self.acts = nn.ModuleList(nn.PReLU(c) for _ in range(mid_convs))
        self.c4 = nn.Conv2d(c, 12, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        d = self.a1(self.c1(x))
        for conv, act in zip(self.mids, self.acts):
            d = act(conv(d))
        return base + self.shuffle(self.c4(d))


def as_shipped_state(net):
    """Rename mids.* back to the shipped c2/c3 keys so export_sr_weights.py and
    sr.js see the exact layout they already know (only valid for mid_convs=2)."""
    sd, out = net.state_dict(), {}
    ren = {"mids.0.": "c2.", "acts.0.": "a2.", "mids.1.": "c3.", "acts.1.": "a3."}
    for k, v in sd.items():
        for a, b in ren.items():
            if k.startswith(a):
                k = b + k[len(a):]
                break
        out[k] = v
    return out


class PairData(Dataset):
    def __init__(self, root, crop=192, clean_frac=0.3):
        self.crop, self.clean_frac = crop, clean_frac
        self.items = []
        tdir = os.path.join(root, "train")
        for ln in open(os.path.join(tdir, "pairs.txt")):
            ln = ln.strip()
            if ln:
                m, g = ln.split("\t")
                self.items.append((os.path.join(tdir, m), g))
        if not self.items:
            raise RuntimeError("no pairs")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        mp, gp = self.items[k]
        gt = cv2.imread(gp)
        h, w = gt.shape[:2]
        h, w = h - h % 2, w - w % 2
        gt = gt[:h, :w]
        c = self.crop
        y = random.randint(0, h - c) & ~1
        x = random.randint(0, w - c) & ~1
        gtc = gt[y:y + c, x:x + c]
        if random.random() < self.clean_frac:  # clean SR sample: box-down of GT
            lo = cv2.resize(gtc, (c // 2, c // 2), interpolation=cv2.INTER_AREA)
        else:  # restorer sample: our mid
            mid = cv2.imread(mp)
            lo = mid[y // 2:(y + c) // 2, x // 2:(x + c) // 2]
        if random.random() < 0.5:
            lo, gtc = lo[:, ::-1], gtc[:, ::-1]
        to = lambda a: torch.from_numpy(
            np.ascontiguousarray(a.transpose(2, 0, 1))).float() / 255.0
        return to(lo), to(gtc)


def psnr_t(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def load_eval(root, device):
    edir = os.path.join(root, "eval")
    out = []
    for ln in open(os.path.join(edir, "pairs.txt")):
        ln = ln.strip()
        if not ln:
            continue
        m, g = ln.split("\t")
        to = lambda p: torch.from_numpy(cv2.imread(os.path.join(edir, p)).transpose(2, 0, 1)).float().div(255)[None].to(device)
        out.append((to(m), to(g)))
    return out


@torch.no_grad()
def evaluate(net, pairs):
    sn, sb = [], []
    for lo, gt in pairs:
        sn.append(psnr_t(net(lo).clamp(0, 1), gt))
        sb.append(psnr_t(F.interpolate(lo, scale_factor=2, mode="bilinear",
                                       align_corners=False).clamp(0, 1), gt))
    return float(np.mean(sn)), float(np.mean(sb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=r"E:\data\framegen\restorer_pairs")
    ap.add_argument("--out", default=r"E:\data\framegen\ckpt_restorer")
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--mid-convs", type=int, default=2)
    ap.add_argument("--clean-frac", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--baseline", default=r"E:\data\framegen\ckpt_sr\sr_best.pt")
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

    ev = load_eval(args.pairs, device)

    base_line = ""
    if os.path.isfile(args.baseline):  # shipped SR on the SAME eval mids
        old = TinySR(16, 2).to(device)
        sd = torch.load(args.baseline, map_location="cpu")
        old.load_state_dict({{"c2.weight": "mids.0.weight", "c2.bias": "mids.0.bias",
                              "a2.weight": "acts.0.weight", "c3.weight": "mids.1.weight",
                              "c3.bias": "mids.1.bias", "a3.weight": "acts.1.weight"}.get(k, k): v
                             for k, v in sd.items()})
        old.eval()
        s_old, _ = evaluate(old, ev)
        base_line = f" · shipped-SR={s_old:.2f} dB"
        del old

    net = TinySR(args.channels, args.mid_convs).to(device)
    data = PairData(args.pairs, clean_frac=args.clean_frac)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    s0, sb = evaluate(net, ev)
    log(f"init: net={s0:.2f} dB · bilinear={sb:.2f} dB{base_line} | "
        f"{len(data)} pairs, {args.steps} steps, c={args.channels} mids={args.mid_convs}")
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
            loss = F.l1_loss(net(lo), gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
            if step % 200 == 0:
                log(f"step {step}/{args.steps} loss={run / 200:.5f} "
                    f"{step * args.batch / (time.time() - t0):.0f} img/s")
                run = 0.0
            if step % args.eval_every == 0:
                sn, _ = evaluate(net, ev)
                save = as_shipped_state(net) if args.mid_convs == 2 else net.state_dict()
                torch.save(save, os.path.join(args.out, "restorer_last.pt"))
                mark = ""
                if sn > best:
                    best = sn
                    torch.save(save, os.path.join(args.out, "restorer_best.pt"))
                    mark = "  ** best"
                log(f"eval @ {step}: net={sn:.2f} dB (bilinear {sb:.2f}{base_line}){mark}")

    log(f"done. best={best:.2f} dB · bilinear={sb:.2f}{base_line}")


if __name__ == "__main__":
    main()
