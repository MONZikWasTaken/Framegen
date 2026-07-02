"""Export the 1-block student's block0 weights to a flat f32 blob for the custom
WebGPU runtime (web/rt/). Output: assets/rt_1blk.bin + assets/rt_1blk.json
(manifest: name -> {shape, offset} in float32 elements).

Usage: python export_rt_weights.py [models/rife_lite_student_1blk.safetensors]
"""
import json
import os
import struct
import sys

import numpy as np
from safetensors.numpy import load_file

src = sys.argv[1] if len(sys.argv) > 1 else r"models\rife_lite_student_1blk.safetensors"
stem = sys.argv[2] if len(sys.argv) > 2 else "rt_1blk"
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")

state = load_file(src)
keys = sorted(k for k in state.keys() if k.startswith("block0."))

blob = []
manifest = {}
offset = 0
for k in keys:
    a = state[k].astype(np.float32)
    manifest[k] = {"shape": list(a.shape), "offset": offset}
    blob.append(a.ravel())
    offset += a.size

flat = np.concatenate(blob)
with open(os.path.join(out_dir, stem + ".bin"), "wb") as f:
    f.write(flat.tobytes())
with open(os.path.join(out_dir, stem + ".json"), "w") as f:
    json.dump(manifest, f)
print(f"{len(keys)} tensors, {flat.size * 4 // 1024 // 1024} MB -> assets/" + stem + ".bin/.json")
