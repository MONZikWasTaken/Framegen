"""Torch reference for the tfact WGSL graph: deterministic LCG input frames
(same generator as the browser parity test), forward at t=0.33/0.66,
expected RGBA bytes -> assets/tfact_ref.bin ([2, H, W, 4], alpha=255).

Usage: python make_tfact_ref.py [ckpt] [W H]
"""
import os
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
from train_student import load_ifnet

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt = sys.argv[1] if len(sys.argv) > 1 else r"E:\data\framecast\ckpt_tfact\tfact_last.pt"
W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
H = int(sys.argv[3]) if len(sys.argv) > 3 else 256


def lcg_frame(shift):
    # numpy RNG, saved to a file the browser fetches — a JS LCG re-implementation
    # silently diverges (s*1103515245 exceeds float64's exact-integer range)
    rng = np.random.default_rng(20260703)
    px = rng.integers(0, 256, W * H * 4, dtype=np.uint8)
    if shift:
        px = ((px.astype(np.int32) + shift) & 255).astype(np.uint8)
    return px.reshape(H, W, 4)


def to_bgr_tensor(rgba, device):
    x = rgba[..., :3].astype(np.float32) / 255.0        # HWC RGB
    x = x[..., ::-1]                                     # BGR (runtime domain)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu")
    slim = load_ifnet(r"E:\data\framecast\ckpt_1blk_slim\student_last.pkl", device)
    kind = TFact2 if any(k.startswith("core.") for k in ck["sd"]) else TFactSlim
    net = kind(slim.block0, ck["c"]).to(device)
    net.load_state_dict(ck["sd"])
    net.eval()

    f0, f1 = lcg_frame(0), lcg_frame(30)
    np.concatenate([f0.ravel(), f1.ravel()]).tofile(os.path.join(REPO, "assets", "tfact_in.bin"))
    img0 = to_bgr_tensor(f0, device)
    img1 = to_bgr_tensor(f1, device)
    out = np.empty((2, H, W, 4), np.uint8)
    with torch.no_grad():
        for k, t in enumerate((0.33, 0.66)):
            tt = torch.tensor([t], device=device)
            pred, _ = net(img0, img1, tt)                # [1,3(BGR),H,W]
            bgr = pred[0].clamp(0, 1).cpu().numpy()
            rgb = np.round(bgr[::-1].transpose(1, 2, 0) * 255).astype(np.uint8)
            out[k, ..., :3] = rgb
            out[k, ..., 3] = 255
    path = os.path.join(REPO, "assets", "tfact_ref.bin")
    out.tofile(path)
    print("wrote", path, out.shape)


if __name__ == "__main__":
    main()
