"""Convert an existing fp32 ONNX to fp16 (compute in half, IO stays fp32).

No torch/weights needed — pure ONNX graph rewrite via onnxconverter-common.
IO kept as float32 (keep_io_types) so the browser/JS can feed plain Float32Array.

Usage:
  python tools/export_fp16.py assets/rife_lite_inlined.onnx assets/rife_lite_fp16.onnx
"""
import sys, os
import onnx
from onnxconverter_common import float16

src = sys.argv[1] if len(sys.argv) > 1 else "assets/rife_lite_inlined.onnx"
dst = sys.argv[2] if len(sys.argv) > 2 else "assets/rife_lite_fp16.onnx"

m = onnx.load(src, load_external_data=False)
mf = float16.convert_float_to_float16(m, keep_io_types=True, disable_shape_infer=True)
onnx.save_model(mf, dst, save_as_external_data=False)
print(f"wrote {dst} ({os.path.getsize(dst)//1024//1024} MB, src {os.path.getsize(src)//1024//1024} MB)")
