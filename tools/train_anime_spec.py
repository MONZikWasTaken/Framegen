"""Anime specialist: finetune the v0.6.0 tfact2 checkpoint on ATD-12K.

The film-specialist lesson: a specialist only makes sense on a DIFFERENT data
distribution. ATD-12K is exactly that - 12k curated animation triplets (large
flat regions, hard cuts, motion on twos), nothing like our Blender/real mix.

ATD has GT only at t=0.5, and a t=0.5-only finetune destroys the timestep
conditioning (measured back in the tfact days). So training alternates:
  even steps: ATD triplet, t=0.5 - GT lap loss + teacher distill (as always)
  odd steps:  ATD pair (frame1, frame3), random t in {0.25..0.75} - teacher-only

Eval every N steps on: ATD test_2k_540p subset (the number that must go UP)
and the standard BBB/jellyfish clips (the canary - some drop is fine for a
specialist, a collapse is not).

Usage (server, training venv):
    python train_anime_spec.py --atd /data/atd12k/datasets --out /data/ckpt_anime
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
from model.laplacian import LapLoss

from train_student import load_eval_triplets, load_ifnet, teacher_forward
from train_tfact2 import TFact2, eval_t2


class ATDData(Dataset):
    """One item = one ATD triplet folder. gt_mode=True yields (f1,f2,f3), t=0.5;
    gt_mode=False yields (f1, dummy, f3) with a random off-center t (teacher-only)."""

    def __init__(self, root, crop=256, gt_mode=True):
        self.crop, self.gt_mode = crop, gt_mode
        self.dirs = []
        for stem in sorted(os.listdir(root)):
            d = os.path.join(root, stem)
            if os.path.isfile(os.path.join(d, "frame1.jpg")) or \
               os.path.isfile(os.path.join(d, "frame1.png")):
                self.dirs.append(d)
        if not self.dirs:
            raise RuntimeError(f"no ATD triplets under {root}")

    def __len__(self):
        return len(self.dirs)

    def _read(self, d, k):
        for ext in ("jpg", "png"):
            p = os.path.join(d, f"frame{k}.{ext}")
            if os.path.isfile(p):
                img = cv2.imread(p)
                if img is not None:
                    return img
        raise RuntimeError(f"missing frame{k} in {d}")

    def __getitem__(self, i):
        d = self.dirs[i]
        f = [self._read(d, 1), self._read(d, 2), self._read(d, 3)]
        t = 0.5
        if not self.gt_mode:
            t = random.choice((0.25, 0.375, 0.625, 0.75))  # keep t-conditioning alive
        h, w = f[0].shape[:2]
        c = self.crop
        if h < c or w < c:  # a few ATD frames are smaller than the crop
            s = c / min(h, w)
            f = [cv2.resize(im, (max(c, int(w * s + 0.5)), max(c, int(h * s + 0.5)))) for im in f]
            h, w = f[0].shape[:2]
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        f = [im[y:y + c, x:x + c] for im in f]
        if random.random() < 0.5:
            f = [im[:, ::-1] for im in f]
        if random.random() < 0.5:
            f = [im[::-1, :] for im in f]
        if random.random() < 0.5:
            f = [f[2], f[1], f[0]]
            t = 1.0 - t
        frames = torch.from_numpy(
            np.ascontiguousarray(np.stack(f).transpose(0, 3, 1, 2))).float() / 255.0
        return frames, torch.tensor(t, dtype=torch.float32)


@torch.no_grad()
def eval_atd(net, trips, device):
    """PSNR at t=0.5 on ATD test triplets (uint8 HWC BGR, any size)."""
    vals = []
    for f0, gt, f1 in trips:
        h, w = f0.shape[:2]
        ph, pw = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        x = torch.zeros(2, 3, ph, pw, device=device)
        x[0, :, :h, :w] = torch.from_numpy(f0.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
        x[1, :, :h, :w] = torch.from_numpy(f1.transpose(2, 0, 1).astype(np.float32) / 255).to(device)
        pred, _ = net(x[:1], x[1:], torch.tensor([0.5], device=device))
        out = (pred[0, :, :h, :w].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
        mse = np.mean((out.astype(np.float64) - gt.astype(np.float64)) ** 2)
        vals.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
    return float(np.mean(vals))


def load_atd_eval(test_root, n=100):
    dirs = sorted(os.listdir(test_root))
    step = max(1, len(dirs) // n)
    trips = []
    for stem in dirs[::step][:n]:
        d = os.path.join(test_root, stem)
        fr = []
        for k in (1, 2, 3):
            img = None
            for ext in ("png", "jpg"):
                p = os.path.join(d, f"frame{k}.{ext}")
                if os.path.isfile(p):
                    img = cv2.imread(p)
                    break
            fr.append(img)
        if all(f is not None for f in fr):
            trips.append(tuple(fr))
    return trips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atd", default=r"E:\data\framecast\atd12k\datasets")
    ap.add_argument("--out", default=r"E:\data\framecast\ckpt_anime")
    ap.add_argument("--teacher", default=os.path.join(
        os.environ.get("TEMP", "/tmp"), "opencode", "rife_m", "RIFE_m_train_log", "flownet.pkl"))
    ap.add_argument("--slim-ckpt", default=r"E:\data\framecast\ckpt_1blk_slim\student_last.pkl")
    ap.add_argument("--tfact-ckpt", default=r"E:\data\framecast\ckpt_big_v060\tfact2_best.pt")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lambda-tea", type=float, default=0.25)
    ap.add_argument("--lambda-flow", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-n", type=int, default=100)
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

    teacher = load_ifnet(args.teacher, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    slim = load_ifnet(args.slim_ckpt, device)
    ck = torch.load(args.tfact_ckpt, map_location="cpu")
    net = TFact2(slim.block0, ck["c"]).to(device)
    net.load_state_dict(ck["sd"])
    del slim

    mk = lambda gt_mode: DataLoader(
        ATDData(os.path.join(args.atd, "train_10k"), args.crop, gt_mode),
        batch_size=args.batch, shuffle=True, num_workers=args.workers // 2,
        pin_memory=True, drop_last=True, persistent_workers=args.workers > 1)
    gt_loader, t_loader = mk(True), mk(False)

    atd_eval = load_atd_eval(os.path.join(args.atd, "test_2k_540p"), args.eval_n)
    eval_sets = load_eval_triplets()  # BBB/jelly canary
    lap = LapLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        u = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-6 + 0.5 * (args.lr - 1e-6) * (1 + math.cos(math.pi * u))

    net.eval()
    a0 = eval_atd(net, atd_eval, device)
    base = eval_t2(net, eval_sets, device)
    log(f"init (v0.6.0): ATD={a0:.2f} dB | canary {base} | "
        f"{len(gt_loader.dataset)} triplets, {args.steps} steps")
    best = a0

    step, t0, run = 0, time.time(), 0.0
    net.train()
    it_gt, it_t = iter(gt_loader), iter(t_loader)
    while step < args.steps:
        gt_step = step % 2 == 0
        try:
            batch, bt = next(it_gt if gt_step else it_t)
        except StopIteration:
            if gt_step:
                it_gt = iter(gt_loader)
                batch, bt = next(it_gt)
            else:
                it_t = iter(t_loader)
                batch, bt = next(it_t)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = batch.to(device, non_blocking=True)
        t = bt.to(device, non_blocking=True).view(-1, 1, 1, 1)
        img0, gt, img1 = batch[:, 0], batch[:, 1], batch[:, 2]
        tea_out, tea_flow = teacher_forward(teacher, img0, img1, t)
        pred, flow = net(img0, img1, t)
        if gt_step:
            loss = (lap(pred, gt)
                    + args.lambda_tea * lap(pred, tea_out)
                    + args.lambda_flow * F.l1_loss(flow, tea_flow))
        else:  # off-center t: no GT exists - the teacher IS the target
            loss = lap(pred, tea_out) + args.lambda_flow * F.l1_loss(flow, tea_flow)
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
            a = eval_atd(net, atd_eval, device)
            canary = eval_t2(net, eval_sets, device)
            net.train()
            torch.save({"sd": net.state_dict(), "c": ck["c"]},
                       os.path.join(args.out, "anime_last.pt"))
            mark = ""
            if a > best:
                best = a
                torch.save({"sd": net.state_dict(), "c": ck["c"]},
                           os.path.join(args.out, "anime_best.pt"))
                mark = "  ** best"
            log(f"eval @ {step}: ATD={a:.2f} dB (init {a0:.2f}) | canary {canary}{mark}")

    log(f"done. best ATD={best:.2f} dB (init {a0:.2f})")


if __name__ == "__main__":
    main()
