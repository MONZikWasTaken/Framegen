"""t-factored slim student: a timestep-FREE trunk that runs ONCE per frame pair,
plus a tiny FiLM(t)-conditioned head that runs per interpolated frame.

Why: the shipped 1-block slim student re-runs the ENTIRE network for every t
(x6 factor = 5 full passes). Here the trunk (conv0 + 6 of the 8 convblocks,
~72% of compute) is t-independent; only the head (FiLM + 2 convblocks +
residual + lastconv, ~28%) depends on t. At x6 that is 0.72 + 5*0.28 = 2.1
network-equivalents instead of 5 - a ~2.4x throughput win at high factors.

Init trick: the timestep input channel is folded into the first conv's bias at
t=0.5 (bias += 0.5 * W_t summed over the kernel), so the untrained factored
model is EXACTLY the slim student at t=0.5; FiLM starts at zero (identity) and
distillation teaches it the t-conditioning.

Usage (training venv):
    python train_tfact.py --slim-ckpt E:\\data\\framegen\\ckpt_1blk_slim\\student_last.pkl
Checkpoints: tfact_last.pt / tfact_best.pt in --out (raw state_dict + meta).
"""
import argparse
import copy
import math
import os
import sys
import time

os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";")
    if "nvidia gpu computing toolkit" not in p.lower())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.laplacian import LapLoss
from model.warplayer import warp

from train_student import (TripletData, load_eval_triplets, load_ifnet,
                           teacher_forward)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAMBDA_TEA, LAMBDA_FLOW = 0.2, 0.01


class TFactSlim(nn.Module):
    """Factored 1-block IFBlock (scale=4). trunk(pair) -> features; head(t) -> flow."""

    def __init__(self, src_block, c):
        super().__init__()
        self.c = c
        # conv0 without the timestep channel; fold t=0.5 * W_t into the bias
        self.conv0 = copy.deepcopy(src_block.conv0)
        w7 = src_block.conv0[0][0].weight.data          # [c/2, 7, 3, 3]
        b7 = src_block.conv0[0][0].bias.data
        first = nn.Conv2d(6, c // 2, 3, 2, 1)
        with torch.no_grad():
            first.weight.copy_(w7[:, :6])
            first.bias.copy_(b7 + 0.5 * w7[:, 6].sum(dim=(1, 2)))
        self.conv0[0][0] = first
        self.trunk = nn.Sequential(*[copy.deepcopy(src_block.convblock[i]) for i in range(6)])
        self.head = nn.Sequential(*[copy.deepcopy(src_block.convblock[i]) for i in (6, 7)])
        self.lastconv = copy.deepcopy(src_block.lastconv)
        self.film = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 2 * c))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)

    def trunk_forward(self, img0, img1, scale=4):
        x = torch.cat((img0, img1), 1)
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        feat0 = self.conv0(x)
        return feat0, self.trunk(feat0)

    def head_forward(self, feat0, h, t, scale=4):
        g = self.film(t.view(-1, 1))                    # [B, 2c]
        s, b = g[:, :self.c, None, None], g[:, self.c:, None, None]
        hh = self.head(h * (1 + s) + b)
        tmp = self.lastconv(hh + feat0)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale * 2, tmp[:, 4:5]

    def forward(self, img0, img1, t, scale=4):
        feat0, h = self.trunk_forward(img0, img1, scale)
        flow, mask = self.head_forward(feat0, h, t, scale)
        w0 = warp(img0, flow[:, :2])
        w1 = warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return w0 * m + w1 * (1 - m), flow


@torch.no_grad()
def eval_tfact(net, eval_sets, device):
    """t=0.5 PSNR on the standard eval triplets (t!=0.5 is spot-checked after
    training with tools/quality_bench-style stride-4 pairs)."""
    W, H, PW, PH = 1280, 720, 1280, 736
    res = {}
    for name, trips in eval_sets:
        mid = []
        for f0, gt, f1 in trips:
            def prep(im):
                # uint8 up, convert on GPU: same cast + /255 (bit-identical),
                # 4x less H2D and no CPU float pass per eval frame
                x = torch.zeros(1, 3, PH, PW, device=device)
                x[0, :, :H, :W] = torch.from_numpy(
                    np.ascontiguousarray(im.transpose(2, 0, 1))).to(device).float().div_(255.0)
                return x
            tt = torch.tensor([0.5], device=device)
            pred, _ = net(prep(f0), prep(f1), tt)
            pred = (pred[0, :, :H, :W].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
            mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
            mid.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        res[name] = float(np.mean(mid))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\data\framegen\frames")
    ap.add_argument("--out", default=r"E:\data\framegen\ckpt_tfact")
    ap.add_argument("--teacher", default=os.path.join(
        os.environ["TEMP"], "opencode", "rife_m", "RIFE_m_train_log", "flownet.pkl"))
    ap.add_argument("--slim-ckpt", default=r"E:\data\framegen\ckpt_1blk_slim\student_last.pkl")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=8e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
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

    teacher = load_ifnet(args.teacher, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    slim = load_ifnet(args.slim_ckpt, device)
    c = slim.block0.conv0[1][0].weight.shape[0]
    net = TFactSlim(slim.block0, c).to(device)
    del slim

    data = TripletData(args.data, args.crop, arbitrary_t=True)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    eval_sets = load_eval_triplets()
    lap = LapLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        u = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * u))

    base = eval_tfact(net, eval_sets, device)
    log(f"init (== slim at t=0.5 by construction): {base} | c={c}, "
        f"{len(data)} triplets, {args.steps} steps")
    best = sum(base.values())

    step, t0, run = 0, time.time(), 0.0
    net.train()
    while step < args.steps:
        for batch, bt in loader:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = batch.to(device, non_blocking=True)
            t = bt.to(device, non_blocking=True).view(-1, 1, 1, 1)
            img0, gt, img1 = batch[:, 0], batch[:, 1], batch[:, 2]
            tea_out, tea_flow = teacher_forward(teacher, img0, img1, t)
            pred, flow = net(img0, img1, t)
            loss = (lap(pred, gt)
                    + LAMBDA_TEA * lap(pred, tea_out)
                    + LAMBDA_FLOW * F.l1_loss(flow, tea_flow))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
            if step % 100 == 0:
                log(f"step {step}/{args.steps} loss={run / 100:.4f} "
                    f"lr={lr_at(step):.2e} {step * args.batch / (time.time() - t0):.1f} img/s")
                run = 0.0
            if step % args.eval_every == 0:
                net.eval()
                scores = eval_tfact(net, eval_sets, device)
                net.train()
                torch.save({"sd": net.state_dict(), "c": c}, os.path.join(args.out, "tfact_last.pt"))
                mark = ""
                if sum(scores.values()) > best:
                    best = sum(scores.values())
                    torch.save({"sd": net.state_dict(), "c": c}, os.path.join(args.out, "tfact_best.pt"))
                    mark = "  ** best"
                log(f"eval @ {step}: {scores}{mark}")

    log(f"done. best sum PSNR = {best:.2f}")


if __name__ == "__main__":
    main()
