"""Compare two TensorRT engines' outputs on a real frame pair (PSNR).

Runs each engine with the same server-style preprocessing (native -> BGR/255/CHW ->
zero-pad to engine size -> infer -> crop top-left -> native) and reports PSNR of the
second engine's output against the first (reference). Use to measure INT8 degradation
vs the FP16 engine.

Usage: python trt_compare.py <ref_engine> <test_engine> <video> <native_w> <native_h> [frame_idx]
"""
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import sys, subprocess

REF, TEST, VIDEO = sys.argv[1], sys.argv[2], sys.argv[3]
NW, NH = int(sys.argv[4]), int(sys.argv[5])
FRAME = int(sys.argv[6]) if len(sys.argv) > 6 else 10

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)


def load(path):
    eng = runtime.deserialize_cuda_engine(open(path, "rb").read())
    ctx = eng.create_execution_context()
    names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
    sh = {n: tuple(eng.get_tensor_shape(n)) for n in names}
    return eng, ctx, names, sh


def infer(path, a, b):
    eng, ctx, names, sh = load(path)
    _, C, EH, EW = sh[names[0]]
    stream = cuda.Stream()
    h_in0 = cuda.pagelocked_empty((1, C, EH, EW), np.float32)
    h_in1 = cuda.pagelocked_empty((1, C, EH, EW), np.float32)
    h_out = cuda.pagelocked_empty(sh[names[-1]], np.float32)
    d0 = cuda.mem_alloc(h_in0.nbytes); d1 = cuda.mem_alloc(h_in1.nbytes); do = cuda.mem_alloc(h_out.nbytes)
    for n in names:
        if "img0" in n: ctx.set_tensor_address(n, int(d0))
        elif "img1" in n: ctx.set_tensor_address(n, int(d1))
        else: ctx.set_tensor_address(n, int(do))

    def prep(dst, hwc):
        chw = hwc[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        dst[:] = 0.0
        dst[0, :, :NH, :NW] = chw

    prep(h_in0, a); prep(h_in1, b)
    cuda.memcpy_htod_async(d0, h_in0, stream)
    cuda.memcpy_htod_async(d1, h_in1, stream)
    ctx.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(h_out, do, stream)
    stream.synchronize()
    out = h_out[0, :, :NH, :NW].transpose(1, 2, 0)[:, :, ::-1]
    return (out * 255.0).clip(0, 255).astype(np.uint8)


raw = subprocess.run(
    ["ffmpeg", "-v", "error", "-i", VIDEO, "-vf", f"scale={NW}:{NH}",
     "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
fb = NH * NW * 3
a = np.frombuffer(raw, np.uint8, count=fb, offset=FRAME * fb).reshape(NH, NW, 3)
b = np.frombuffer(raw, np.uint8, count=fb, offset=(FRAME + 1) * fb).reshape(NH, NW, 3)

ref = infer(REF, a, b).astype(np.float32)
test = infer(TEST, a, b).astype(np.float32)
mse = np.mean((ref - test) ** 2)
psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
diff = np.abs(ref - test)
print(f"PSNR(test vs ref) = {psnr:.2f} dB")
print(f"max abs diff = {diff.max():.0f}, mean abs diff = {diff.mean():.3f}, pixels diff>2 = {(diff > 2).mean() * 100:.2f}%")
