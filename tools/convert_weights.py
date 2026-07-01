#!/usr/bin/env python3
"""Convert a RIFE_m (RIFEm / lite) PyTorch checkpoint to safetensors for the Rust/candle port.

Reference model: hzwer/ECCV2022-RIFE, model.IFNet_m (arbitrary=True path in model.RIFE.Model).
Paper RIFE_m weights:  https://drive.google.com/file/d/147XVsDXBfJPlyct2jfo9kpbL944mNeZr/view?usp=sharing
                        (unzip -> train_log/flownet.pkl)

Key names are preserved verbatim from the PyTorch state_dict (after stripping any
DataParallel "module." prefix) so the Rust VarBuilder can load them by the same paths:
    block0.conv0.0.weight   block0.conv0.0.bias
    block0.conv0.1.weight                  (PReLU weight, shape [c/2] then [c])
    block0.convblock.0.weight ...
    block0.lastconv.weight  block0.lastconv.bias
    block1.*  block2.*  block_tea.*
    contextnet.conv1.0.*  contextnet.conv1.1.*  ... conv4
    unet.down0..down3  unet.up0..up3  unet.conv.*

Usage:
    python convert_weights.py <flownet.pkl> <rife_lite.safetensors> [--manifest out.json] [--fp16]
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="path to flownet.pkl (RIFE_m state_dict)")
    ap.add_argument("output", help="path to write rife_lite.safetensors")
    ap.add_argument("--manifest", help="optional json key->shape manifest")
    ap.add_argument("--fp16", action="store_true", help="store as fp16 (default fp32)")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        print("torch is required:  pip install torch safetensors", file=sys.stderr)
        return 2

    from safetensors.torch import save_file

    src = Path(args.input)
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return 1

    state = torch.load(str(src), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    converted = {}
    for k, v in state.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        t = v.detach().cpu().contiguous()
        if args.fp16:
            t = t.half()
        else:
            t = t.float()
        converted[name] = t

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(out), metadata={"format": "pt", "variant": "rife_m_lite"})
    print(f"wrote {len(converted)} tensors -> {out} ({out.stat().st_size // 1024} KiB)")

    if args.manifest:
        manifest = {name: list(t.shape) for name, t in converted.items()}
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"wrote manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
