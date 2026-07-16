"""Framegen V7 trainer: distill from PRECOMPUTED EMA-VFI targets (tea_v7).

The teacher was run over the dataset once (make_teacher_v7.py, 4 targets per
triplet: s2 t=0.5 + s4 t=0.25/0.5/0.75, JPEG q97) - no teacher forward in the
training loop, so a step costs student-only compute. No flow-distill term:
EMA-VFI outputs frames, not our flow format (documented trade-off; the lap
losses carry the signal).

Student = the shipped tfact2 family (trunk 1/16 + FiLM t-head + 1/4-res
refine) at an arbitrary trunk width c:
  c=120 - the current model (Framegen v0.6.x)
  c<120 - "V7 Small" candidates (96, 80): same quality bar, fewer ms
  c>120 - "V7 Large" candidates: beat full RIFE within the ~12 ms budget
Warm start: channel-sliced from the v0.6.0 checkpoint (--init); film layers
re-created at the new width (zero-init last layer keeps the start sane), the
width-independent refine head is copied as is. For c>120 the trunk grows: new
channels are randomly initialized, sliced ones keep v0.6.0 weights.

Usage (server):
    python train_v7.py --channels 96 --steps 8000 --out /workspace/ckpt_v7s_probe
"""
import argparse
import math
import os
import random
import sys
import time

os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";")
    if "nvidia gpu computing toolkit" not in p.lower())

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.environ.get("RIFE_REF", r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.IFNet_m import IFBlock
from model.laplacian import LapLoss

from train_student import load_eval_triplets
from train_tfact import TFactSlim
from train_tfact2 import RefineNet, TFact2, eval_t2

cv2.setNumThreads(0)


def slice_or_grow(dst, src):
    """Copy the overlapping channel block of src into dst (both conv/linear/prelu
    tensors); extra dst channels keep their fresh init."""
    idx = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
    dst[idx] = src[idx]


def build_student(init_ckpt, c, device):
    """TFact2 at trunk width c, warm-started from a (c=120) tfact2 checkpoint."""
    ck = torch.load(init_ckpt, map_location="cpu")
    src_sd = ck["sd"]
    block = IFBlock(7, c=c)
    net = TFact2(block, c).to(device)
    c_src = ck["c"]
    with torch.no_grad():
        dst_sd = net.state_dict()
        for k, dst in dst_sd.items():
            if k not in src_sd:
                continue
            src = src_sd[k]
            if k.startswith("core.film.2."):
                # film output is [scale(c) | bias(c)] concatenated - slice halves
                m = min(c, c_src)
                dst[:m] = src[:m]
                dst[c:c + m] = src[c_src:c_src + m]
            elif src.shape == dst.shape:
                dst.copy_(src)
            else:
                slice_or_grow(dst, src)
        net.load_state_dict(dst_sd)
    return net


class V7Data(Dataset):
    """Samples with a precomputed teacher target. 50% s2 (t=0.5), 50% s4
    (t in {.25,.5,.75} with REAL GT). Spatial aug is applied to all four
    images; temporal swap mirrors t and swaps inputs (gt/teacher unchanged)."""

    def __init__(self, data_root, tea_root, crop=256):
        self.crop = crop
        self.tea = tea_root
        self.items = []
        for stem in sorted(os.listdir(data_root)):
            idx_file = os.path.join(data_root, stem, "triplets.txt")
            td = os.path.join(self.tea, stem)
            if not os.path.isfile(idx_file) or not os.path.isdir(td):
                continue
            d = os.path.join(data_root, stem)
            have = set(os.listdir(td))
            for line in open(idx_file):
                line = line.strip()
                if not line:
                    continue
                i = int(line)
                s2 = f"{i:06d}_s2.jpg" in have
                s4 = all(f"{i:06d}_s4t{k}.jpg" in have for k in (1, 2, 3))
                if s2:
                    self.items.append((d, i, s4))
        if not self.items:
            raise RuntimeError("no (triplet, teacher) pairs found")

    def __len__(self):
        return len(self.items)

    def _rd(self, d, k):
        img = cv2.imread(os.path.join(d, f"{k:06d}.jpg"))
        if img is None:
            raise RuntimeError(f"missing {d}\\{k:06d}.jpg")
        return img

    def __getitem__(self, j):
        d, i, has_s4 = self.items[j]
        clip = os.path.basename(d)
        if has_s4 and random.random() < 0.5:
            k = random.randint(1, 3)
            f0, f1 = self._rd(d, i - 2), self._rd(d, i + 2)
            gt = self._rd(d, i - 2 + k)
            tea = cv2.imread(os.path.join(self.tea, clip, f"{i:06d}_s4t{k}.jpg"))
            t = k / 4.0
        else:
            f0, f1 = self._rd(d, i - 1), self._rd(d, i + 1)
            gt = self._rd(d, i)
            tea = cv2.imread(os.path.join(self.tea, clip, f"{i:06d}_s2.jpg"))
            t = 0.5
        f = [f0, gt, f1, tea]
        h, w = f[0].shape[:2]
        c = self.crop
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        f = [im[y:y + c, x:x + c] for im in f]
        if random.random() < 0.5:
            f = [im[:, ::-1] for im in f]
        if random.random() < 0.5:
            f = [im[::-1, :] for im in f]
        if random.random() < 0.5:
            f = [f[2], f[1], f[0], f[3]]
            t = 1.0 - t
        # uint8 out of the worker: the f32 conversion runs on the GPU after H2D
        # (same cast + same /255 = bit-identical values), so collate/pin/upload
        # move 4x fewer bytes per batch
        frames = torch.from_numpy(
            np.ascontiguousarray(np.stack(f).transpose(0, 3, 1, 2)))
        return frames, torch.tensor(t, dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\data\framegen\frames_v060\frames")
    ap.add_argument("--tea-root", default=r"A:\framegen_dataset\tea_v7")
    ap.add_argument("--out", default=r"E:\data\framegen\ckpt_v7")
    ap.add_argument("--init", default=r"E:\data\framegen\ckpt_big_v060\tfact2_best.pt")
    ap.add_argument("--channels", type=int, default=96)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--lambda-tea", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-oversample", type=int, default=6)
    ap.add_argument("--resume", default=None,
                    help="v7 checkpoint with optimizer state - continues the SAME run "
                         "(model + Adam moments + step), for host-eviction recovery")
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

    net = build_student(args.init, args.channels, device)

    data = V7Data(args.data, args.tea_root, args.crop)
    n_real = sum(1 for d, _, _ in data.items if "real_" in os.path.basename(d))
    if n_real and args.real_oversample > 1:
        extra = [it for it in data.items if "real_" in os.path.basename(it[0])]
        data.items.extend(extra * (args.real_oversample - 1))
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                        # one torch thread per worker: the default intra-op pool times
                        # N workers thrashes many-core boxes into single-digit GPU util
                        worker_init_fn=lambda _: torch.set_num_threads(1))
    eval_sets = load_eval_triplets()
    lap = LapLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step = 0
    if args.resume:
        rk = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(rk["sd"])
        if "opt" in rk:
            opt.load_state_dict(rk["opt"])
        start_step = rk.get("step", 0)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        u = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * u))

    base = eval_t2(net, eval_sets, device)
    log(f"init c={args.channels} (sliced from v0.6.0): {base} | "
        f"{len(data)} samples ({n_real} real x{args.real_oversample}), {args.steps} steps")
    best = sum(base.values())

    step, t0 = start_step, time.time()
    # loss accumulates as a GPU f64 scalar: .item() every step blocked the CPU
    # behind the whole step graph (~3-4ms of kernel enqueue the GPU then idles
    # through). f64 accumulation of f32 losses matches the old Python-float sum
    # bit for bit; only the logging boundary syncs now.
    run_t = torch.zeros((), device=device, dtype=torch.float64)
    net.train()
    while step < args.steps:
        for batch, bt in loader:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = batch.to(device, non_blocking=True).float().div_(255.0)
            t = bt.to(device, non_blocking=True).view(-1, 1, 1, 1)
            img0, gt, img1, tea = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
            pred, _ = net(img0, img1, t)
            loss = lap(pred, gt) + args.lambda_tea * lap(pred, tea)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run_t += loss.detach()
            step += 1
            if step % 200 == 0:
                log(f"step {step}/{args.steps} loss={run_t.item() / 200:.4f} "
                    f"lr={lr_at(step):.2e} "
                    f"{(step - start_step) * args.batch / (time.time() - t0):.1f} img/s")
                run_t.zero_()
            if step % args.eval_every == 0:
                net.eval()
                scores = eval_t2(net, eval_sets, device)
                net.train()
                torch.save({"sd": net.state_dict(), "c": args.channels,
                            "opt": opt.state_dict(), "step": step},
                           os.path.join(args.out, "v7_last.pt"))
                mark = ""
                if sum(scores.values()) > best:
                    best = sum(scores.values())
                    torch.save({"sd": net.state_dict(), "c": args.channels},
                               os.path.join(args.out, "v7_best.pt"))
                    mark = "  ** best"
                log(f"eval @ {step}: {scores}{mark}")

    log(f"done. best sum PSNR = {best:.2f}")


if __name__ == "__main__":
    main()
