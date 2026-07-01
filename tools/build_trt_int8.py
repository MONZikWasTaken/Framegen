"""Build a TensorRT engine with INT8 quantization using entropy calibration.

INT8 needs a calibration dataset: representative input pairs so TensorRT can pick per-tensor
dynamic ranges. We decode an anime clip at the engine's NATIVE resolution, build consecutive
(img0, img1) pairs, and preprocess each exactly like scripts/trt_server.py (RGB->BGR, /255,
CHW, zero-pad bottom/right to the engine's /32 size). FP16 is left enabled as a fallback for
layers INT8 can't handle (mixed precision).

Usage:
  python build_trt_int8.py <onnx> <engine_out> <calib_video> <native_w> <native_h> [max_pairs]

Example (720p):
  python build_trt_int8.py assets/rife_lite_inlined.onnx assets/rife_lite_int8.engine \
      demo/test_10s.mp4 1280 720 200
"""
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys, os, subprocess

ONNX = sys.argv[1]
ENGINE = sys.argv[2]
CALIB_VIDEO = sys.argv[3]
NW = int(sys.argv[4])
NH = int(sys.argv[5])
MAX_PAIRS = int(sys.argv[6]) if len(sys.argv) > 6 else 200
CACHE = ENGINE + ".calib"

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open(ONNX, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  parse error {i}: {parser.get_error(i)}")
        raise RuntimeError("ONNX parse failed")

# Engine (padded /32) input dims from the network.
in0 = network.get_input(0)
_, C, EH, EW = tuple(in0.shape)
print(f"engine input: {C}x{EH}x{EW}, native: {C}x{NH}x{NW}")
if NH > EH or NW > EW:
    raise RuntimeError(f"native {NH}x{NW} exceeds engine {EH}x{EW}")

# ---- calibration frames: decode the clip at native res, make consecutive pairs ----
print(f"decoding {CALIB_VIDEO} -> {NW}x{NH} raw...")
raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", CALIB_VIDEO, "-vf", f"scale={NW}:{NH}",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    capture_output=True, check=True,
).stdout
fb = NH * NW * 3
nframes = len(raw) // fb
frames = [np.frombuffer(raw, np.uint8, count=fb, offset=i * fb).reshape(NH, NW, 3)
          for i in range(nframes)]
pairs = [(frames[i], frames[i + 1]) for i in range(nframes - 1)][:MAX_PAIRS]
print(f"{nframes} frames -> {len(pairs)} calibration pairs")
if not pairs:
    raise RuntimeError("no calibration pairs")


def preprocess(hwc):
    # native RGB24 HWC uint8 -> engine-sized CHW float32 (BGR, /255, zero-padded bottom/right)
    chw = hwc[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    out = np.zeros((1, C, EH, EW), dtype=np.float32)
    out[0, :, :NH, :NW] = chw
    return np.ascontiguousarray(out)


class RifeCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, pairs):
        super().__init__()
        self.pairs = pairs
        self.idx = 0
        nbytes = C * EH * EW * 4
        self.d0 = cuda.mem_alloc(nbytes)
        self.d1 = cuda.mem_alloc(nbytes)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.pairs):
            return None
        a, b = self.pairs[self.idx]
        self.idx += 1
        if self.idx % 50 == 0:
            print(f"  calibrating pair {self.idx}/{len(self.pairs)}")
        cuda.memcpy_htod(self.d0, preprocess(a))
        cuda.memcpy_htod(self.d1, preprocess(b))
        return [int(self.d0) if "img0" in n else int(self.d1) for n in names]

    def read_calibration_cache(self):
        if os.path.exists(CACHE):
            print(f"using calibration cache {CACHE}")
            return open(CACHE, "rb").read()
        return None

    def write_calibration_cache(self, cache):
        with open(CACHE, "wb") as f:
            f.write(cache)


config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 enabled (mixed-precision fallback)")
if not builder.platform_has_fast_int8:
    raise RuntimeError("platform has no fast INT8")
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = RifeCalibrator(pairs)
print("INT8 enabled with entropy calibration")

print("building INT8 engine (calibration passes take a few minutes)...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    raise RuntimeError("engine build failed")
with open(ENGINE, "wb") as f:
    f.write(serialized)
print(f"wrote {ENGINE} ({os.path.getsize(ENGINE) / 1024 / 1024:.1f} MB)")
