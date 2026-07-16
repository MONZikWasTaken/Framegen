"""Arbitrary-t quality check for the t-factored student vs the slim baseline.

Stride-4 pairs (i, i+4) from the held-out eval clips give REAL ground truth at
t=0.25/0.5/0.75 (frames i+1..i+3). This is the metric that dies when timestep
conditioning is broken (a t-blind model scores below the blend floor ~33.8).

Usage: python eval_tfact_arb.py [tfact_ckpt]
"""
import os
import subprocess
import sys

os.environ["PATH"] = ";".join(
    p for p in os.environ["PATH"].split(";")
    if "nvidia gpu computing toolkit" not in p.lower())

import numpy as np
import torch

sys.path.insert(0, r"C:\Users\MONZik\AppData\Local\Temp\opencode\rife_ref")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_tfact import TFactSlim
from train_tfact2 import TFact2
from train_student import load_ifnet, two_block_forward

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = [os.path.join(REPO, "assets", "bbb_720_10s.mp4"),
         os.path.join(REPO, "assets", "jellyfish_720_10s.mp4")]
SLIM = r"E:\data\framegen\ckpt_1blk_slim\student_last.pkl"


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else r"E:\data\framegen\ckpt_tfact\tfact_best.pt"
    device = torch.device("cuda")
    ck = torch.load(ckpt, map_location="cpu")
    slim = load_ifnet(SLIM, device).eval()
    kind = TFact2 if any(k.startswith("core.") for k in ck["sd"]) else TFactSlim
    net = kind(slim.block0, ck["c"]).to(device)
    net.load_state_dict(ck["sd"])
    net.eval()

    W, H, PW, PH = 1280, 720, 1280, 736

    def prep(im):
        # uint8 up, convert on GPU: bit-identical values, 4x less H2D per frame
        x = torch.zeros(1, 3, PH, PW, device=device)
        x[0, :, :H, :W] = torch.from_numpy(
            np.ascontiguousarray(im.transpose(2, 0, 1))).to(device).float().div_(255.0)
        return x

    for clip in CLIPS:
        p = subprocess.run(["ffmpeg", "-v", "error", "-i", clip, "-f", "rawvideo",
                            "-pix_fmt", "bgr24", "-"], capture_output=True, check=True)
        raw = np.frombuffer(p.stdout, np.uint8)
        n = len(raw) // (W * H * 3)
        frames = raw[: n * W * H * 3].reshape(n, H, W, 3)
        rows = {"tfact": {0.25: [], 0.5: [], 0.75: []},
                "slim": {0.25: [], 0.5: [], 0.75: []}}
        with torch.no_grad():
            for i in range(0, n - 4, 25):
                f0, f4 = prep(frames[i]), prep(frames[i + 4])
                for k in (1, 2, 3):
                    t = k / 4.0
                    gt = frames[i + k]
                    pred, _ = net(f0, f4, torch.tensor([t], device=device))
                    out = (pred[0, :, :H, :W].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                    rows["tfact"][t].append(psnr(out, gt))
                    pred2, _ = two_block_forward(slim, f0, f4, (4,), t)
                    out2 = (pred2[0, :, :H, :W].clamp(0, 1) * 255).byte().cpu().numpy().transpose(1, 2, 0)
                    rows["slim"][t].append(psnr(out2, gt))
        name = os.path.basename(clip)
        for model, d in rows.items():
            s = " ".join(f"t={t}: {np.mean(v):.2f}" for t, v in sorted(d.items()))
            print(f"{name} {model:6s} {s}")


if __name__ == "__main__":
    main()
