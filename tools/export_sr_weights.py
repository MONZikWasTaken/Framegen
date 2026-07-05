"""Export the TinySR checkpoint to a flat f32 blob for web/rt/sr.js.

Usage: python export_sr_weights.py [E:\\data\\framegen\\ckpt_sr\\sr_best.pt] [rt_sr]
Output: assets/<stem>.bin + assets/<stem>.json (name -> {shape, offset in floats}).
"""
import json
import os
import sys

import numpy as np
import torch

src = sys.argv[1] if len(sys.argv) > 1 else r"E:\data\framegen\ckpt_sr\sr_best.pt"
stem = sys.argv[2] if len(sys.argv) > 2 else "rt_sr"
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

sd = torch.load(src, map_location="cpu")
blob, manifest, offset = [], {}, 0
for k in sorted(sd.keys()):
    a = sd[k].numpy().astype(np.float32)
    manifest[k] = {"shape": list(a.shape), "offset": offset}
    blob.append(a.ravel())
    offset += a.size

flat = np.concatenate(blob)
with open(os.path.join(out_dir, stem + ".bin"), "wb") as f:
    f.write(flat.tobytes())
with open(os.path.join(out_dir, stem + ".json"), "w") as f:
    json.dump(manifest, f)
print(f"{len(manifest)} tensors, {flat.size * 4 // 1024} KB -> assets/{stem}.bin/.json")
