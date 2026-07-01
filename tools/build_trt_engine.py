"""Convert rife_lite.onnx -> TensorRT engine with fp16."""
import tensorrt as trt
import os

import sys, os
ONNX = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\MONZik\Desktop\InterModule\assets\rife_lite_inlined.onnx"
ENGINE = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\MONZik\Desktop\InterModule\assets\rife_lite_trt_fp16.engine"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  parse error {i}: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")
print("parsed ONNX OK")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4 GB workspace

if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 enabled")
else:
    print("FP16 not available")

print("building engine (this takes a few minutes)...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("engine build failed")

with open(ENGINE, "wb") as f:
    f.write(serialized)
print(f"wrote {ENGINE} ({os.path.getsize(ENGINE) / 1024 / 1024:.1f} MB)")
