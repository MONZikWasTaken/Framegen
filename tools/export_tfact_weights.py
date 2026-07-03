"""Export the t-factored student (tools/train_tfact.py checkpoint) to the flat
f32 blob + manifest the WGSL runtime loads. Names map back onto the classic
block0.* layout (trunk[i] -> convblock.i, head[j] -> convblock.6+j) so the
runtime reuses every conv bind group; the film MLP ships under film.*.

Usage: python export_tfact_weights.py [E:\\data\\framecast\\ckpt_tfact\\tfact_best.pt] [rt_tfact]
"""
import json
import os
import sys

import numpy as np
import torch

src = sys.argv[1] if len(sys.argv) > 1 else r"E:\data\framecast\ckpt_tfact\tfact_best.pt"
stem = sys.argv[2] if len(sys.argv) > 2 else "rt_tfact"
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

ck = torch.load(src, map_location="cpu")
sd = ck["sd"]

# tfact name -> runtime manifest name
name_map = {}
for tail in ("0.0.weight", "0.0.bias", "0.1.weight", "1.0.weight", "1.0.bias", "1.1.weight"):
    name_map[f"conv0.{tail}"] = f"block0.conv0.{tail}"
for i in range(6):
    for tail in ("0.weight", "0.bias", "1.weight"):
        name_map[f"trunk.{i}.{tail}"] = f"block0.convblock.{i}.{tail}"
for j in range(2):
    for tail in ("0.weight", "0.bias", "1.weight"):
        name_map[f"head.{j}.{tail}"] = f"block0.convblock.{6 + j}.{tail}"
name_map["lastconv.weight"] = "block0.lastconv.weight"
name_map["lastconv.bias"] = "block0.lastconv.bias"
for k in ("film.0.weight", "film.0.bias", "film.2.weight", "film.2.bias"):
    name_map[k] = k

blob, manifest, offset = [], {}, 0
for src_name, dst_name in name_map.items():
    a = sd[src_name].numpy().astype(np.float32)
    manifest[dst_name] = {"shape": list(a.shape), "offset": offset}
    blob.append(a.ravel())
    offset += a.size

flat = np.concatenate(blob)
with open(os.path.join(out_dir, stem + ".bin"), "wb") as f:
    f.write(flat.tobytes())
with open(os.path.join(out_dir, stem + ".json"), "w") as f:
    json.dump(manifest, f)
print(f"{len(manifest)} tensors, {flat.size * 4 / 1e6:.1f} MB -> assets/{stem}.bin/.json")
