"""Distill a 2-block RIFE student from the full 3-block+refine teacher.

Student = block0 + block1 of IFNet_m (init from teacher), scales [4,2], no refinement —
same graph as the `2blk_noref` ablation (docs/phase5_ablation.md), which measured
4x on browser webgpu / 1.9x on TensorRT at -2.6..-0.8 dB. Training tries to win that
PSNR back at identical inference cost.

Loss: LapLoss(student, gt) + LAMBDA_TEA * LapLoss(student, teacher_out)
      + LAMBDA_FLOW * L1(student_flow, teacher_flow)

Data: JPEG frame dirs + triplets.txt from tools/extract_frames.py (random 256 crops,
flips, temporal swap). Eval: fixed triplets from the two held-out clips (BBB, Jellyfish),
full 720p, uint8 PSNR — directly comparable to tools/quality_bench.py numbers.

Usage (training venv):
    python train_student.py --data E:\\data\\framecast\\frames --out E:\\data\\framecast\\ckpt
Checkpoints: student_last.pkl / student_best.pkl in --out, saved as a FULL IFNet_m
state_dict with "module." prefixes -> drop-in for restore-dir/flownet.pkl consumers
(export via tools/export_ablation.py with FRAMECAST_WEIGHTS pointing at --out).
"""
import argparse
import math
import os
import random
import subprocess
import sys
import time

# The system CUDA dir carries foreign cuDNN sublibraries (engines_tensor_ir/ext) that
# torch's bundled cuDNN discovers via PATH and rejects with SUBLIBRARY_VERSION_MISMATCH.
# Strip it from this process before torch loads cuDNN (see cuda13-env-gotchas).
os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";")
    if "nvidia gpu computing toolkit" not in p.lower())

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
from model.IFNet_m import IFNet_m
from model.laplacian import LapLoss
from model.warplayer import warp

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_CLIPS = [os.path.join(REPO, "assets", "bbb_720_10s.mp4"),
              os.path.join(REPO, "assets", "jellyfish_720_10s.mp4")]
LAMBDA_TEA, LAMBDA_FLOW = 0.2, 0.01


def load_ifnet(weights_pkl, device):
    sd = torch.load(weights_pkl, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items() if "module." in k}
    net = IFNet_m()
    net.load_state_dict(sd)
    return net.to(device)


def two_block_forward(net, img0, img1, scales=(4, 2)):
    """Student inference path (matches export_ablation configs). Returns (merged, flow).
    len(scales) picks how many IFBlocks run (1..3)."""
    timestep = (img0[:, :1] * 0 + 1) * 0.5
    stu = [net.block0, net.block1, net.block2][: len(scales)]
    flow = None
    mask = None
    warped_img0, warped_img1 = img0, img1
    for i in range(len(scales)):
        if flow is not None:
            flow_d, mask_d = stu[i](
                torch.cat((img0, img1, timestep, warped_img0, warped_img1, mask), 1),
                flow, scale=scales[i])
            flow = flow + flow_d
            mask = mask + mask_d
        else:
            flow, mask = stu[i](torch.cat((img0, img1, timestep), 1), None, scale=scales[i])
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
    m = torch.sigmoid(mask)
    return warped_img0 * m + warped_img1 * (1 - m), flow


@torch.no_grad()
def teacher_forward(net, img0, img1):
    """Full frozen teacher: 3 blocks + refinement. Returns (merged_refined, flow)."""
    x = torch.cat((img0, img1), 1)
    flow_list, mask, merged, *_ = net(x, scale=[4, 2, 1], timestep=0.5)
    return merged[2], flow_list[2]


class TripletData(Dataset):
    def __init__(self, data_root, crop):
        self.crop = crop
        self.items = []
        for stem in os.listdir(data_root):
            idx_file = os.path.join(data_root, stem, "triplets.txt")
            if not os.path.isfile(idx_file):
                continue
            with open(idx_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.items.append((os.path.join(data_root, stem), int(line)))
        if not self.items:
            raise RuntimeError(f"no triplets found under {data_root}")

    def __len__(self):
        return len(self.items)

    def _read(self, d, i):
        img = cv2.imread(os.path.join(d, f"{i:06d}.jpg"))  # BGR uint8
        if img is None:
            raise RuntimeError(f"missing frame {d}\\{i:06d}.jpg")
        return img

    def __getitem__(self, k):
        d, i = self.items[k]
        f = [self._read(d, i - 1), self._read(d, i), self._read(d, i + 1)]
        h, w = f[0].shape[:2]
        c = self.crop
        y, x = random.randint(0, h - c), random.randint(0, w - c)
        f = [im[y:y + c, x:x + c] for im in f]
        if random.random() < 0.5:
            f = [im[:, ::-1] for im in f]
        if random.random() < 0.5:
            f = [im[::-1, :] for im in f]
        if random.random() < 0.5:
            f = [f[2], f[1], f[0]]
        return torch.from_numpy(
            np.ascontiguousarray(np.stack(f).transpose(0, 3, 1, 2))).float() / 255.0  # [3,C,c,c]


def load_eval_triplets(step=25):
    """Decode held-out clips -> list of (name, [ (f0,f1,f2) uint8 HWC BGR ])."""
    out = []
    for clip in EVAL_CLIPS:
        W, H = 1280, 720
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
        raw = np.frombuffer(p.stdout, np.uint8)
        n = len(raw) // (W * H * 3)
        frames = raw[: n * W * H * 3].reshape(n, H, W, 3)
        trips = [(frames[i - 1], frames[i], frames[i + 1]) for i in range(1, n - 1, step)]
        out.append((os.path.basename(clip), trips))
    return out


@torch.no_grad()
def eval_psnr(net, eval_sets, device, scales=(4, 2)):
    """Full-720p uint8 PSNR, same pre/post as quality_bench.py."""
    W, H, PW, PH = 1280, 720, 1280, 736
    res = {}
    for name, trips in eval_sets:
        scores = []
        for f0, gt, f1 in trips:
            def prep(im):
                x = torch.zeros(1, 3, PH, PW, device=device)
                x[0, :, :H, :W] = torch.from_numpy(
                    im.transpose(2, 0, 1).astype(np.float32) / 255.0).to(device)
                return x
            pred, _ = two_block_forward(net, prep(f0), prep(f1), scales)
            pred = (pred[0, :, :H, :W].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
            mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
            scores.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
        res[name] = float(np.mean(scores))
    return res


def save_full_state(student, teacher_sd, path, prefixes):
    """Full IFNet_m dict: trained blocks over the teacher's remaining weights."""
    sd = dict(teacher_sd)
    for k, v in student.state_dict().items():
        if k.startswith(prefixes):
            sd[k] = v.detach().cpu()
    torch.save({"module." + k: v for k, v in sd.items()}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=r"E:\data\framecast\frames")
    ap.add_argument("--out", default=r"E:\data\framecast\ckpt")
    ap.add_argument("--weights", default=os.path.join(
        os.environ["TEMP"], "opencode", "rife_m", "RIFE_m_train_log", "flownet.pkl"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", default="")
    ap.add_argument("--scales", default="4,2",
                    help="IFBlock scales, e.g. '4,2' (2 blocks), '8,4', '4' (1 block)")
    args = ap.parse_args()
    scales = tuple(int(s) for s in args.scales.split(","))
    prefixes = tuple(f"block{i}." for i in range(len(scales)))

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True  # fp32 train, TF32 tensor cores
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "train.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    teacher = load_ifnet(args.weights, device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher_sd = {k: v.cpu() for k, v in teacher.state_dict().items()}

    student = load_ifnet(args.resume if args.resume else args.weights, device)
    # only the blocks the student actually runs are trained; freeze the rest
    for name, p in student.named_parameters():
        p.requires_grad_(name.startswith(prefixes))
    train_params = [p for p in student.parameters() if p.requires_grad]

    data = TripletData(args.data, args.crop)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    eval_sets = load_eval_triplets()
    lap = LapLoss()
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + math.cos(math.pi * t))

    base = eval_psnr(student, eval_sets, device, scales)
    log(f"init (untrained cut, scales={scales}): {base}  | {len(data)} triplets, "
        f"batch {args.batch}, {args.steps} steps")
    best = sum(base.values())

    step = 0
    t0 = time.time()
    run_loss = 0.0
    student.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            batch = batch.to(device, non_blocking=True)
            img0, gt, img1 = batch[:, 0], batch[:, 1], batch[:, 2]
            tea_out, tea_flow = teacher_forward(teacher, img0, img1)
            pred, flow = two_block_forward(student, img0, img1, scales)
            loss = (lap(pred, gt)
                    + LAMBDA_TEA * lap(pred, tea_out)
                    + LAMBDA_FLOW * F.l1_loss(flow, tea_flow))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            run_loss += loss.item()
            step += 1

            if step % 100 == 0:
                ips = step * args.batch / (time.time() - t0)
                log(f"step {step}/{args.steps} loss={run_loss / 100:.4f} "
                    f"lr={lr_at(step):.2e} {ips:.1f} img/s")
                run_loss = 0.0
            if step % args.eval_every == 0:
                student.eval()
                scores = eval_psnr(student, eval_sets, device, scales)
                student.train()
                save_full_state(student, teacher_sd,
                                os.path.join(args.out, "student_last.pkl"), prefixes)
                mark = ""
                if sum(scores.values()) > best:
                    best = sum(scores.values())
                    save_full_state(student, teacher_sd,
                                    os.path.join(args.out, "student_best.pkl"), prefixes)
                    mark = "  ** best"
                log(f"eval @ {step}: {scores}{mark}")

    log(f"done. best sum PSNR = {best:.2f}")


if __name__ == "__main__":
    main()
