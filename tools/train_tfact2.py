"""tfact2 = t-factored student + a tiny REFINE head (occlusion repair).

The coarse student warps-and-blends; everything it cannot know (disocclusions,
halo at motion edges) stays broken. Full RIFE fixes this with contextnet+unet —
far too heavy. Here: a 4-conv c=24 net at HALF resolution eats [warped0,
warped1, mask, normalized flow] (11ch) and emits a bounded residual that is
upsampled and added to the blend. Last conv is zero-init and the residual goes
through sigmoid*2-1, so step 0 is EXACTLY the tfact checkpoint.

Teacher = full RIFE incl. ITS refinement — precisely the signal the head must
learn. Real-footage dirs (frames/real_*) are oversampled 6x: the base set is
three Blender movies and real video was underrepresented.

Usage (training venv):
    python train_tfact2.py
Checkpoints: tfact2_last.pt / tfact2_best.pt in --out.
"""
import argparse
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

from train_student import TripletData, load_eval_triplets, load_ifnet, teacher_forward
from train_tfact import TFactSlim

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAMBDA_TEA, LAMBDA_FLOW = 0.25, 0.01
FLOW_NORM = 20.0  # half-res flow / this -> sane activation range


class RefineNet(nn.Module):
    def __init__(self, c=16):
        super().__init__()
        self.c0 = nn.Conv2d(11, c, 3, 1, 1)
        self.a0 = nn.PReLU(c)
        self.c1 = nn.Conv2d(c, c, 3, 1, 1)
        self.a1 = nn.PReLU(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1)
        self.a2 = nn.PReLU(c)
        self.c3 = nn.Conv2d(c, 3, 3, 1, 1)
        nn.init.zeros_(self.c3.weight)
        nn.init.zeros_(self.c3.bias)  # sigmoid(0)*2-1 = 0 -> starts as identity

    def forward(self, x):
        x = self.a0(self.c0(x))
        x = self.a1(self.c1(x))
        x = self.a2(self.c2(x))
        return torch.sigmoid(self.c3(x)) * 2.0 - 1.0


class TFact2(nn.Module):
    def __init__(self, src_block, c):
        super().__init__()
        self.core = TFactSlim(src_block, c)
        self.refine = RefineNet(16)

    def forward(self, img0, img1, t, scale=4):
        feat0, h = self.core.trunk_forward(img0, img1, scale)
        flow, mask = self.core.head_forward(feat0, h, t, scale)
        w0 = warp(img0, flow[:, :2])
        w1 = warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        merged = w0 * m + w1 * (1 - m)
        # QUARTER res: half-res refine measured +7ms@720p in WGSL — 16x fewer pixels
        # brings it to ~1ms; occlusion halos are mid-frequency, quarter carries them
        q = lambda z: F.interpolate(z, scale_factor=0.25, mode="bilinear", align_corners=False)
        rin = torch.cat((q(w0), q(w1), q(m), q(flow) * (0.25 / FLOW_NORM)), 1)
        res = F.interpolate(self.refine(rin), scale_factor=4, mode="bilinear", align_corners=False)
        return torch.clamp(merged + res, 0, 1), flow


@torch.no_grad()
def eval_t2(net, eval_sets, device):
    W, H, PW, PH = 1280, 720, 1280, 736
    res = {}
    for name, trips in eval_sets:
        vals = []
        for f0, gt, f1 in trips:
            def prep(im):
                x = torch.zeros(1, 3, PH, PW, device=device)
                x[0, :, :H, :W] = torch.from_numpy(
                    im.transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)
                return x
            pred, _ = net(prep(f0), prep(f1), torch.tensor([0.5], device=device))
            pred = (pred[0, :, :H, :W].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
            mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
            vals.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        res[name] = float(np.mean(vals))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\data\framecast\frames")
    ap.add_argument("--out", default=r"E:\data\framecast\ckpt_tfact2")
    ap.add_argument("--teacher", default=os.path.join(
        os.environ["TEMP"], "opencode", "rife_m", "RIFE_m_train_log", "flownet.pkl"))
    ap.add_argument("--slim-ckpt", default=r"E:\data\framecast\ckpt_1blk_slim\student_last.pkl")
    ap.add_argument("--tfact-ckpt", default=r"E:\data\framecast\ckpt_tfact\tfact_best.pt")
    ap.add_argument("--steps", type=int, default=25000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--real-oversample", type=int, default=6)
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
    ck = torch.load(args.tfact_ckpt, map_location="cpu")
    net = TFact2(slim.block0, ck["c"]).to(device)
    net.core.load_state_dict(ck["sd"])
    del slim

    data = TripletData(args.data, args.crop, arbitrary_t=True)
    n_real = sum(1 for d, _ in data.items if "real_" in os.path.basename(d))
    if n_real and args.real_oversample > 1:
        extra = [(d, i) for d, i in data.items if "real_" in os.path.basename(d)]
        data.items.extend(extra * (args.real_oversample - 1))
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

    base = eval_t2(net, eval_sets, device)
    log(f"init (== tfact by construction): {base} | {len(data)} triplets "
        f"({n_real} real, oversampled x{args.real_oversample}), {args.steps} steps")
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
                scores = eval_t2(net, eval_sets, device)
                net.train()
                torch.save({"sd": net.state_dict(), "c": ck["c"]},
                           os.path.join(args.out, "tfact2_last.pt"))
                mark = ""
                if sum(scores.values()) > best:
                    best = sum(scores.values())
                    torch.save({"sd": net.state_dict(), "c": ck["c"]},
                               os.path.join(args.out, "tfact2_best.pt"))
                    mark = "  ** best"
                log(f"eval @ {step}: {scores}{mark}")

    log(f"done. best sum PSNR = {best:.2f}")


if __name__ == "__main__":
    main()
