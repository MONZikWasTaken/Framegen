"""Export an official IFRNet checkpoint for Framegen's WebGPU runtime.

The output follows Framegen's existing flat-blob convention: every tensor is
stored as contiguous little-endian float32 and the JSON manifest maps its
original PyTorch state-dict name to shape and element offset.  Keeping the
names intact makes the graph/checkpoint contract auditable and avoids a second
model-specific conversion format.

Usage:
  python tools/export_ifrnet_weights.py CHECKPOINT assets/rt_ifrnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="official IFRNet .pth checkpoint")
    parser.add_argument("output", type=Path, help="output stem, without .bin/.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "encoder.pyramid1.0.0.weight" not in state:
        raise ValueError("not an IFRNet inference checkpoint")

    items = sorted(state.items())
    manifest: dict[str, object] = {
        "format": "framegen-ifrnet-v1",
        "source": "IFRNet official inference checkpoint",
        "tensors": {},
    }
    tensors: dict[str, dict[str, object]] = manifest["tensors"]  # type: ignore[assignment]
    offset = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".bin").open("wb") as blob:
        for name, value in items:
            array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
            tensors[name] = {"shape": list(array.shape), "offset": offset}
            blob.write(array.tobytes(order="C"))
            offset += array.size

    manifest["bytes"] = offset * 4
    manifest["tensorCount"] = len(tensors)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    print(f"{len(tensors)} tensors, {offset * 4 / 1024 / 1024:.1f} MiB -> {args.output}.bin/.json")


if __name__ == "__main__":
    main()
