"""Train the v2 SR student: upscale + compression cleanup in one net.

Field report (2026-07-13, 360p anime): the shipped model sharpens compression
artifacts as faithfully as detail - it was trained on CLEAN box-downscales.
This trainer degrades the low-res inputs with a compression pipeline (JPEG
quality ladder + optional slight blur, Real-ESRGAN-style but lighter) so the
student learns restore+upscale jointly, and adds anime (ATD-12K) to the mix.

Training mix per sample:
  ~45% film frames   (frames/<clip>/NNNNNN.jpg, triplets.txt dirs)
  ~35% anime frames  (atd12k/datasets/train_10k/<dir>/frame{1,2,3}.png)
  ~20% restorer mids (restorer_pairs/train/pairs.txt - our tfact2 outputs)
Film/anime lo inputs get the degradation; mids stay as produced (they are
clean GPU outputs in the player).

Eval tracks (every --eval-every steps, PSNR vs GT):
  clean   BBB downscale -> up (the old metric, keeps the clean case honest)
  jpeg    BBB downscale + JPEG q40 (deblock metric)
  anime   ATD test frames downscale + JPEG q40 (THE field-report metric)
  mid     restorer_pairs/eval (regression vs the shipped restorer)
Best checkpoint by 0.35*anime + 0.25*jpeg + 0.2*mid + 0.2*clean.

Usage (training venv):
    E:\\venvs\\fg-train\\Scripts\\python.exe train_sr2.py --out E:\\data\\framecast\\ckpt_sr2
Checkpoints are saved with the SHIPPED key layout (c1..c4, a1..a3) - feed
sr2_best.pt straight to export_sr_weights.py.
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
    sd, out = net.state_dict(), {}
    ren = {"mids.0.": "c2.", "acts.0.": "a2.", "mids.1.": "c3.", "acts.1.": "a3."}
    for k, v in sd.items():
        for a, b in ren.items():
            if k.startswith(a):
                k = b + k[len(a):]
                break
        out[k] = v
    return out


def degrade(lo, rng):
    """Compression-ish degradation of a low-res BGR u8 image. ~15% pass clean."""
    r = rng.random()
    if r < 0.15:
        return lo
    if rng.random() < 0.25:  # slight softness before compression (scaler blur)
        sigma = rng.uniform(0.3, 0.8)
        lo = cv2.GaussianBlur(lo, (0, 0), sigma)
    q = rng.randint(30, 80)
    ok, enc = cv2.imencode(".jpg", lo, [cv2.IMWRITE_JPEG_QUALITY, q])
    lo = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.25:  # re-encode: streams get recompressed all the time
        q2 = rng.randint(50, 85)
        ok, enc = cv2.imencode(".jpg", lo, [cv2.IMWRITE_JPEG_QUALITY, q2])
        lo = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return lo


class MixData(Dataset):
    def __init__(self, film_root, anime_root, pairs_root, crop=192,
                 p_film=0.45, p_anime=0.35):
        self.crop = crop
        self.p_film, self.p_anime = p_film, p_anime
        self.film = []
        for stem in os.listdir(film_root):
            d = os.path.join(film_root, stem)
            if not os.path.isfile(os.path.join(d, "triplets.txt")):
                continue
            for line in open(os.path.join(d, "triplets.txt")):
                line = line.strip()
                if line:
                    self.film.append(os.path.join(d, f"{int(line):06d}.jpg"))
        self.anime = []
        troot = os.path.join(anime_root, "datasets", "train_10k")
        for stem in os.listdir(troot):
            d = os.path.join(troot, stem)
            for fn in ("frame1.jpg", "frame2.jpg", "frame3.jpg",
                       "frame1.png", "frame2.png", "frame3.png"):  # train split ships jpg, test png
                p = os.path.join(d, fn)
                if os.path.isfile(p):
                    self.anime.append(p)
        self.mids = []
        tdir = os.path.join(pairs_root, "train")
        for ln in open(os.path.join(tdir, "pairs.txt")):
            ln = ln.strip()
            if ln:
                m, g = ln.split("\t")
                self.mids.append((os.path.join(tdir, m), g))
        if not (self.film and self.anime and self.mids):
            raise RuntimeError(f"missing data: film={len(self.film)} anime={len(self.anime)} mids={len(self.mids)}")

    def __len__(self):
        return len(self.film) + len(self.anime) + len(self.mids)

    def __getitem__(self, k):
        rng = random.Random(k ^ random.getrandbits(30))
        r = rng.random()
        c = self.crop
        if r < self.p_film + self.p_anime:  # degraded-source SR sample
            path = rng.choice(self.film if r < self.p_film else self.anime)
            img = cv2.imread(path)
            h, w = img.shape[:2]
            if h < c or w < c:  # small anime frames: upscale GT to fit the crop
                s = c / min(h, w)
                img = cv2.resize(img, (int(w * s) + 1, int(h * s) + 1), interpolation=cv2.INTER_CUBIC)
                h, w = img.shape[:2]
            y, x = rng.randint(0, h - c), rng.randint(0, w - c)
            gt = img[y:y + c, x:x + c]
            lo = cv2.resize(gt, (c // 2, c // 2), interpolation=cv2.INTER_AREA)
            lo = degrade(lo, rng)
        else:  # restorer sample: our interpolated mid at half res
            mp, gp = rng.choice(self.mids)
            gt_img = cv2.imread(gp)
            h, w = gt_img.shape[:2]
            h, w = h - h % 2, w - w % 2
            y = rng.randint(0, h - c) & ~1
            x = rng.randint(0, w - c) & ~1
            gt = gt_img[y:y + c, x:x + c]
            mid = cv2.imread(mp)
            lo = mid[y // 2:(y + c) // 2, x // 2:(x + c) // 2]
        if rng.random() < 0.5:
            lo, gt = lo[:, ::-1], gt[:, ::-1]
        to = lambda a: torch.from_numpy(
            np.ascontiguousarray(a.transpose(2, 0, 1))).float() / 255.0
        return to(lo), to(gt)


def psnr_t(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def jpeg_np(img_u8_bgr, q):
    ok, enc = cv2.imencode(".jpg", img_u8_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def build_evals(pairs_root, anime_root, device):
    """Four tracks: clean/jpeg (BBB), anime-jpeg (ATD test), mid (restorer eval)."""
    W, H = 1280, 720
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", os.path.join(REPO, "assets", "bbb_720_10s.mp4"),
                        "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
    raw = np.frombuffer(p.stdout, np.uint8)
    n = len(raw) // (W * H * 3)
    bbb = raw[: n * W * H * 3].reshape(n, H, W, 3)[::40][:8]
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().div(255)[None].to(device)

    def sr_pairs(frames, q=None):
        out = []
        for f in frames:
            h, w = f.shape[:2]
            h, w = h - h % 2, w - w % 2
            f = f[:h, :w]
            lo = cv2.resize(f, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
            if q:
                lo = jpeg_np(lo, q)
            out.append((to_t(lo), to_t(f)))
        return out

    troot = os.path.join(anime_root, "datasets", "test_2k_540p")
    dirs = sorted(os.listdir(troot))[::len(os.listdir(troot)) // 8][:8]
    anime = [cv2.imread(os.path.join(troot, d, "frame2.png")) for d in dirs]

    mids = []
    edir = os.path.join(pairs_root, "eval")
    for ln in open(os.path.join(edir, "pairs.txt")):
        ln = ln.strip()
        if ln:
            m, g = ln.split("\t")
            mids.append((to_t(cv2.imread(os.path.join(edir, m))), to_t(cv2.imread(os.path.join(edir, g)))))

    return {"clean": sr_pairs(bbb), "jpeg": sr_pairs(bbb, 40),
            "anime": sr_pairs(anime, 40), "mid": mids}


@torch.no_grad()
def evaluate(net, evals):
    res = {}
    for name, pairs in evals.items():
        sn, sb = [], []
        for lo, gt in pairs:
            sn.append(psnr_t(net(lo).clamp(0, 1), gt))
            sb.append(psnr_t(F.interpolate(lo, scale_factor=2, mode="bilinear",
                                           align_corners=False).clamp(0, 1), gt))
        res[name] = (float(np.mean(sn)), float(np.mean(sb)))
    return res


SCORE_W = {"anime": 0.35, "jpeg": 0.25, "mid": 0.2, "clean": 0.2}
score = lambda r: sum(SCORE_W[k] * r[k][0] for k in SCORE_W)
fmt = lambda r: " ".join(f"{k}={v[0]:.2f}({v[1]:.2f})" for k, v in r.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", default=r"E:\data\framecast\frames")
    ap.add_argument("--anime", default=r"E:\data\framecast\atd12k")
    ap.add_argument("--pairs", default=r"E:\data\framecast\restorer_pairs")
    ap.add_argument("--out", default=r"E:\data\framecast\ckpt_sr2")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--mid-convs", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--baseline", default=r"E:\data\framecast\ckpt_restorer\restorer_best.pt")
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

    evals = build_evals(args.pairs, args.anime, device)

    base_line = ""
    if os.path.isfile(args.baseline):  # the SHIPPED weights on the same tracks
        old = TinySR(16, 2).to(device)
        sd = torch.load(args.baseline, map_location="cpu")
        old.load_state_dict({{"c2.weight": "mids.0.weight", "c2.bias": "mids.0.bias",
                              "a2.weight": "acts.0.weight", "c3.weight": "mids.1.weight",
                              "c3.bias": "mids.1.bias", "a3.weight": "acts.1.weight"}.get(k, k): v
                             for k, v in sd.items()})
        old.eval()
        base_line = f"\n  shipped: {fmt(evaluate(old, evals))}"
        del old

    net = TinySR(args.channels, args.mid_convs).to(device)
    data = MixData(args.film, args.anime, args.pairs)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)

    r0 = evaluate(net, evals)
    log(f"init: {fmt(r0)}{base_line}\n  data: film={len(data.film)} anime={len(data.anime)} mids={len(data.mids)}"
        f" | {args.steps} steps c={args.channels} mids={args.mid_convs}")
    best = score(r0)

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
                r = evaluate(net, evals)
                save = as_shipped_state(net) if args.mid_convs == 2 else net.state_dict()
                torch.save(save, os.path.join(args.out, "sr2_last.pt"))
                mark = ""
                if score(r) > best:
                    best = score(r)
                    torch.save(save, os.path.join(args.out, "sr2_best.pt"))
                    mark = "  ** best"
                log(f"eval @ {step}: {fmt(r)} score={score(r):.3f}{mark}")

    log(f"done. best score={best:.3f}{base_line}")


if __name__ == "__main__":
    main()
