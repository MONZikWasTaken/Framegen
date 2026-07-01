"""Benchmark TensorRT engine."""
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import time, sys, os

ENGINE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\MONZik\Desktop\InterModule\assets\rife_lite_trt_fp16.engine"

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
with open(ENGINE, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
stream = cuda.Stream()

n_io = engine.num_io_tensors
io_names = [engine.get_tensor_name(i) for i in range(n_io)]
print(f"IO: {io_names}")

# Get shapes from engine
in0_shape = engine.get_tensor_shape(io_names[0])
in1_shape = engine.get_tensor_shape(io_names[1])
out_shape = engine.get_tensor_shape(io_names[-1])
print(f"shapes: in0={in0_shape} in1={in1_shape} out={out_shape}")

H, W = in0_shape[2], in0_shape[3]
print(f"resolution: {W}x{H}")

host_in0 = cuda.pagelocked_empty(tuple(in0_shape), dtype=np.float32)
host_in1 = cuda.pagelocked_empty(tuple(in1_shape), dtype=np.float32)
host_out = cuda.pagelocked_empty(tuple(out_shape), dtype=np.float32)

dev_in0 = cuda.mem_alloc(host_in0.nbytes)
dev_in1 = cuda.mem_alloc(host_in1.nbytes)
dev_out = cuda.mem_alloc(host_out.nbytes)

for name in io_names:
    if "img0" in name: context.set_tensor_address(name, int(dev_in0))
    elif "img1" in name: context.set_tensor_address(name, int(dev_in1))
    elif "mid" in name: context.set_tensor_address(name, int(dev_out))

np.copyto(host_in0, np.random.randn(*host_in0.shape).astype(np.float32) * 0.5 + 0.5)
np.copyto(host_in1, np.random.randn(*host_in1.shape).astype(np.float32) * 0.5 + 0.5)

for _ in range(3):
    cuda.memcpy_htod_async(dev_in0, host_in0, stream)
    cuda.memcpy_htod_async(dev_in1, host_in1, stream)
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

N = 20
times = []
for _ in range(N):
    cuda.memcpy_htod_async(dev_in0, host_in0, stream)
    cuda.memcpy_htod_async(dev_in1, host_in1, stream)
    t0 = time.perf_counter()
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)

times.sort()
p50 = times[N // 2]
p10 = times[N // 10]
p90 = times[N * 9 // 10]
mean = sum(times) / N
print(f"\n=== TensorRT FP16 {W}x{H}, {N} iters ===")
print(f"p50={p50:.1f}ms  p10={p10:.1f}ms  p90={p90:.1f}ms  mean={mean:.1f}ms")
print(f"fps: {1000/p50:.1f}")
print(f"real-time 48fps (2x@24): {'PASS' if p50 <= 41.7 else 'FAIL'} (need 41.7ms)")
print(f"real-time 60fps: {'PASS' if p50 <= 16.6 else 'FAIL'} (need 16.6ms)")
