#!/usr/bin/env python3
"""Reconstruct train_log/flownet.pkl from models/rife_lite.safetensors.

Inverse of convert_weights.py: that script stripped the DataParallel "module."
prefix and cast to fp32; RIFE.Model.load_model filters for keys CONTAINING
"module.", so we put the prefix back. Output is a plain torch state_dict pickle,
byte-compatible with what load_model/export_onnx.py expect.

Usage:
    python restore_pkl.py [<rife_lite.safetensors>] [<out_train_log_dir>]

Defaults: models/rife_lite.safetensors -> %TEMP%/opencode/rife_m/RIFE_m_train_log/
"""
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

repo = Path(__file__).resolve().parent.parent
src = Path(sys.argv[1]) if len(sys.argv) > 1 else repo / "models" / "rife_lite.safetensors"
out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else \
    Path(os.environ["TEMP"]) / "opencode" / "rife_m" / "RIFE_m_train_log"

state = load_file(str(src))
restored = {"module." + k: v for k, v in state.items()}

out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "flownet.pkl"
torch.save(restored, str(out))
print(f"wrote {len(restored)} tensors -> {out} ({out.stat().st_size // 1024 // 1024} MB)")
