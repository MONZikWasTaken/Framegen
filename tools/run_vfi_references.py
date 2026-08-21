#!/usr/bin/env python3
"""Run the official Practical-RIFE and IFRNet checkpoints on the fixed VFI shootout."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "mps":
        torch.mps.synchronize()


def load_image(path: Path, dev: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    pixels = torch.from_numpy(__import__("numpy").asarray(image).copy())
    return pixels.permute(2, 0, 1).float().div_(255).unsqueeze(0).to(dev)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    pixels = tensor.detach().clamp(0, 1).mul(255).byte().cpu()[0].permute(1, 2, 0).numpy()
    Image.fromarray(pixels, "RGB").save(path)


def pad_rife(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = image.shape[-2:]
    ph = (64 - h % 64) % 64
    pw = (64 - w % 64) % 64
    return F.pad(image, (0, pw, 0, ph)), (h, w)


def pad_ifrnet(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    h, w = image.shape[-2:]
    ph = (64 - h % 64) % 64
    pw = (64 - w % 64) % 64
    padding = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
    return F.pad(image, padding, mode="replicate"), padding


def load_rife(root: Path, checkpoint: Path, dev: torch.device):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(checkpoint.parent.parent))
    from train_log.RIFE_HDv3 import Model

    model = Model()
    model.load_model(str(checkpoint.parent))
    model.flownet.to(dev)
    model.eval()
    return model


def load_ifrnet(root: Path, checkpoint: Path, dev: torch.device):
    sys.path.insert(0, str(root))
    from models.IFRNet import Model

    model = Model().to(dev)
    model.load_state_dict(torch.load(checkpoint, map_location=dev, weights_only=True))
    model.eval()
    return model


def median_ms(run, dev: torch.device) -> tuple[torch.Tensor, float]:
    for _ in range(2):
        run()
    synchronize(dev)
    samples = []
    output = None
    for _ in range(5):
        started = time.perf_counter()
        output = run()
        synchronize(dev)
        samples.append((time.perf_counter() - started) * 1000)
    return output, sorted(samples)[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=Path, default=ROOT / "benchmarks/vfi-shootout/scenes.json")
    parser.add_argument("--out", type=Path, default=ROOT / ".bench/results/reference")
    parser.add_argument("--rife-root", type=Path, required=True)
    parser.add_argument("--rife-checkpoint", type=Path, required=True)
    parser.add_argument("--ifrnet-root", type=Path, required=True)
    parser.add_argument("--ifrnet-checkpoint", type=Path, required=True)
    args = parser.parse_args()

    dev = device()
    rife_dev = torch.device("cpu")
    print(f"reference devices: Practical-RIFE={rife_dev}, IFRNet={dev}")
    args.out.mkdir(parents=True, exist_ok=True)
    rife = load_rife(args.rife_root, args.rife_checkpoint, rife_dev)
    ifrnet = load_ifrnet(args.ifrnet_root, args.ifrnet_checkpoint, dev)
    records = []

    for scene in json.loads(args.scenes.read_text()):
        i0 = load_image(ROOT / scene["i0"], rife_dev)
        i1 = load_image(ROOT / scene["i1"], rife_dev)

        rife0, (h, w) = pad_rife(i0)
        rife1, _ = pad_rife(i1)
        with torch.inference_mode():
            rife_output, rife_ms = median_ms(lambda: rife.inference(rife0, rife1, timestep=0.5), rife_dev)
        rife_output = rife_output[..., :h, :w]

        ifr0, padding = pad_ifrnet(i0.to(dev))
        ifr1, _ = pad_ifrnet(i1.to(dev))
        embt = torch.tensor(0.5, device=dev).view(1, 1, 1, 1)
        with torch.inference_mode():
            ifr_output, ifr_ms = median_ms(lambda: ifrnet.inference(ifr0, ifr1, embt), dev)
        left, right, top, bottom = padding
        ifr_output = ifr_output[..., top:ifr_output.shape[-2] - bottom, left:ifr_output.shape[-1] - right]

        scene_out = args.out / scene["id"]
        scene_out.mkdir(exist_ok=True)
        save_image(rife_output, scene_out / "practical-rife-v425.png")
        save_image(ifr_output, scene_out / "ifrnet.png")
        records.append({"scene": scene["id"], "practical-rife-v425-ms": rife_ms, "ifrnet-ms": ifr_ms})
        print(f"{scene['id']}: RIFE {rife_ms:.1f}ms | IFRNet {ifr_ms:.1f}ms")

    (args.out / "timings.json").write_text(json.dumps({"devices": {"practical-rife": str(rife_dev), "ifrnet": str(dev)}, "records": records}, indent=2) + "\n")


if __name__ == "__main__":
    main()
