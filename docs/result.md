# InterModule - Final Report (Native Phase)

## Project: Real-time RIFE-Lite frame interpolation in Rust + GPU

**Date:** July 1, 2026
**GPU:** NVIDIA GeForce RTX 4060 Ti (8 GB VRAM, compute capability 8.9, Ada Lovelace)
**CUDA:** 13.1 (nvcc 13.1.80)
**Rust:** 1.90.0, candle-core/candle-nn 0.11.0
**TensorRT:** 10.13.3.9
**OS:** Windows 11, MSVC 14.44

---

> Path note: after this report was written the native crate moved - src/, csrc/ and
> tests/ now live under crates/framegen-native/. File names below are otherwise accurate.

## 1. Chronology

### Stage 1: Port RIFE-Lite (RIFEm) to Rust/candle

**Source:** `hzwer/ECCV2022-RIFE`, branch `main`, `model/IFNet_m.py` - the lite RIFE
variant (the arbitrary=True path in `model.RIFE.Model`).

**Weights:** the RIFE_m paper checkpoint `flownet.pkl` (42 MB), Google Drive ID
`147XVsDXBfJPlyct2jfo9kpbL944mNeZr`, converted to `rife_lite.safetensors` (190 tensors,
41 MB, fp32) by `tools/convert_weights.py`. PyTorch state_dict keys kept 1-to-1 after
stripping the DataParallel `module.` prefix.

**IFNet_m architecture (reimplemented in `src/model.rs`, 325 lines):**
- `IFBlock`: conv0 (two stride-2 downsamples, /4), convblock (8 residual convs),
  lastconv (ConvTranspose2d upsample /2). Output: 4-ch flow + 1-ch mask.
- `IFNet_m`: 3 IFBlocks (block0 c=240, block1 c=150, block2 c=90) + block_tea (unused at
  inference, needs gt) + Contextnet (c=16, 4 Conv2 levels) + Unet (c=16, 4 down + 4 up).
- Backward warp = port of `model/warplayer.py`:
  `grid_sample(align_corners=True, padding_mode='border')`.
- Forward: 3 scale iterations (4, 2, 1) -> flow refinement -> contextnet+unet residual.

**candle 0.11 API notes:**
- `grid_sample` does NOT exist - the backward warp had to be hand-written.
- `candle_nn::prelu` (per-channel, key `.weight`), `Conv2d`, `ConvTranspose2d`,
  `upsample_bilinear2d(h, w, align_corners)` and
  `upsample_bilinear2d_with_scale(scale_h, scale_w, align_corners)` all exist and match
  PyTorch, with identical `.weight`/`.bias` keys.
- `gather` exists but needs contiguous tensors (`.contiguous()` mandatory after broadcast).
- `slice_set` exists (replacement for `F.pad`); requires contiguous src and dst.
- `VarBuilder::from_mmaped_safetensors` is `unsafe`, mmap-loads weights.
- `Device::synchronize()` is essential for honest GPU timing - otherwise commands are only
  queued and calls return instantly.

**Correctness:** PSNR 88.31 dB vs PyTorch `inference_img.py` on the demo frames (256x448,
timestep=0.5); 99.99% of pixels identical, the remaining 33 (0.01%) differ by 1 (uint8
rounding noise). Parity requires: BGR input (the model was trained on cv2 BGR), /32
padding, and truncation (`.byte()` in PyTorch = `as u8` in Rust, not `.round()`).

### Stage 2: mp4 I/O harness

`src/io/video.rs` - an ffmpeg subprocess pipeline:
- decode: `ffmpeg -i input.mp4 -f rawvideo -pix_fmt rgb24 -` -> stdout
- encode: `ffmpeg -f rawvideo -pix_fmt rgb24 -s WxH -r fps -i - -c:v libx264 ... output.mp4`

No temp files - streaming through pipes, only 2 frames plus the intermediate in memory.
Supports `--times N` (2x, 4x, ...) and `--scale 0.5` (lower-res processing).

### Stage 3: CUDA GPU backend (candle)

Build: `cargo build --release --features cuda` (vcvars64.bat on PATH for nvcc);
`.cargo/config.toml` redirects `target-dir` to E: (drive C: was full).

Problem 1: `Tensor::full(0f32, ...)` + `Tensor::cat` for padding failed on CUDA with
`CUDA_ERROR_INVALID_VALUE` for sizes not divisible by 32. Fixed with `Tensor::zeros` +
`slice_set`.

Problem 2 (critical): timings without `Device::synchronize()` were phantom.
`rife.interpolate()` queues the CUDA work in ~19 ms; actual execution takes ~1175 ms at
1080p. All initial "16.6 ms = 60 fps" numbers were wrong. With `synchronize()`:
1069 ms at 1080p, ~371 ms at 720p.

### Stage 4: candle optimizations (by decreasing impact)

#### 4.1. Fused CUDA warp kernel (`src/warp.rs`)

The gather-based warp (4 gather ops with broadcast indices plus dtype conversions, times
14 warp calls per pass: 6 in IFNet + 8 in Contextnet) was the top bottleneck - gather on
GPU is random memory access. Replaced with a `CustomOp2` running one fused
`backward_warp_bilinear` CUDA kernel (the `WARP_CUDA` const in `warp.rs`), compiled at
runtime via NVRTC (`cudarc::nvrtc::safe::compile_ptx_with_opts`) and loaded via
`CudaDevice::get_or_load_custom_func`. One thread per output element, bilinear with border
padding, align_corners=True (flow normalized by (W-1)/2, (H-1)/2).

**Result:** 1069 ms -> 691 ms at 1080p (1.55x). PSNR 88.73 dB - slightly higher, the
gather dtype-conversion noise is gone.

candle CustomOp2-on-CUDA API notes:
- `candle_core::cuda::{cudarc, WrapErr}` is the correct import path.
- `CudaStorage::as_cuda_slice::<f32>()` gets the slice;
  `CudaDevice::alloc_uninit(&shape, dtype)` (unsafe) allocates the output.
- `dev.get_or_load_custom_func(name, module_name, &ptx_src)` caches kernels;
  `Ptx::to_src()` returns the PTX source string.
- `dev.cuda_stream().launch_builder(&func)` - builder pattern for kernel arguments;
  temporaries (int casts) must be bound to a `let` or the borrow checker rejects them;
  `builder.arg(&cuda_view)` passes the device pointer (`slice` is a method, not a field).

#### 4.2. fp16 half precision

`RifeLite::load(path, dtype, device)` with `DType::F16`; weights load directly as fp16 via
`VarBuilder::from_mmaped_safetensors(&[path], DType::F16, device)`. `interpolate()` casts
inputs to the model dtype; the warp kernel is f32-only, so warp casts to f32 internally and
the result is cast back to f32. One fix: `Tensor::full(timestep as f32, ...)` in forward
was hardcoded f32 - add `.to_dtype(self.dtype)`.

**Result:** 691 ms -> 566 ms at 1080p (1.22x, not the hoped-for 2x). The 4060 Ti is
memory-bandwidth bound on these conv sizes, not compute bound - reading weights from VRAM
is the limit, not MACs, so fp16 tensor cores do not pay. PSNR dropped to 56.99 dB (still
above the 35 dB gate, 0 pixels with diff>1).

#### 4.3. GPU-side download

`tensor_to_rgb24` originally pulled f32 to the CPU (`to_vec3()`) and did a per-pixel
BGR->RGB and *255 loop: ~202 ms at 1080p. Moved clamp, *255 affine, BGR->RGB (narrow+cat)
and the U8 cast onto the GPU; only the u8 buffer is downloaded. 202 ms -> 5.7 ms (35x).

#### 4.4. Lower-res inference (`--scale 0.5`)

`RifeLite::interpolate_scaled(img0, img1, timestep, scale)`: resize input to
`proc_h = H*scale, proc_w = W*scale` via `upsample_bilinear2d`, pad to /32, run with
adjusted scale_list `[4/scale, 2/scale, 1/scale]` (as `inference_img.py --scale`), crop
and upscale back. **Result:** 566 ms -> 265 ms at 1080p (2.13x). Quality: PSNR 28.59 dB at
256x448 - scale=0.5 is too aggressive at small resolutions; at 1080p->540p the loss is
less visible but still artifacted.

#### 4.5. Summary

| Step | 1080p | 720p | PSNR |
|---|---|---|---|
| Baseline (gather warp, fp32, async) | 1069 ms | 371 ms | 88.31 dB |
| + fused CUDA kernel | 691 ms | 340 ms | 88.73 dB |
| + fp16 | 566 ms | 320 ms | 56.99 dB |
| + GPU download | ~570 ms | ~320 ms | - |
| + scale=0.5 | 265 ms | 213 ms | ~28 dB |
| **Total speedup** | **4x** | **1.7x** | - |

Real video pipeline speed (wall time / frames): 1080p scale=1.0: 664 ms/frame (1.5 fps);
1080p scale=0.5: 329 ms/frame (3.0 fps); 720p scale=0.5: ~225 ms/frame (4.4 fps).

### Stage 5: cuDNN - FAILURE

candle 0.11 without the `cudnn` feature uses generic CUDA conv kernels (no cuDNN
Winograd/GEMM), 5-10x slower. Attempt: `pip install nvidia-cudnn-cu12` (cuDNN 9.23.2.1),
DLLs+headers copied into the CUDA toolkit dir, import library created via
`lib /DEF:cudnn_full.def /OUT:cudnn.lib`; also needed `nvidia-cublas-cu12`
(cublasLt64_12.dll, cublas64_12.dll). It built, but cuDNN 9 (cu12) is **slower** than no
cuDNN on CUDA 13: 1080p scale=0.5: 406 ms vs 262 ms; 720p scale=0.5: 364 ms vs 213 ms.
Cause: the cu12 wheel is built for CUDA 12 and hits suboptimal or incompatible paths on
CUDA 13. Fix would be native cuDNN for CUDA 13 from NVIDIA (login required) or a future
`nvidia-cudnn-cu13` wheel. Disabled (`--features cuda` without `cudnn`).

### Stage 6: TensorRT - SUCCESS

Instead of candle inference: export to ONNX -> TensorRT engine.

#### 6.1. ONNX export

`tools/export_onnx.py`: `torch.onnx.export(dynamo=True, opset_version=20)`.
- `dynamo=True` is mandatory - `torch.jit.trace` fails on the dynamic code in
  `warplayer.py` (backwarp_tenGrid cache + grid_sample).
- The dynamo exporter writes external data (`*.onnx.data`) that TensorRT cannot read;
  inline afterwards: `onnx.save_model(m, path, save_as_external_data=False)`.
- The `onnxscript` pip package is required for dynamo export.

Result: 316 ONNX nodes, 197 initializers, 40.8 MB of weights. Two static-shape engines
(dynamic_axes=None for maximum TRT optimization): 720p (736x1280 padded) and
1080p (1088x1920 padded).

#### 6.2. Engine build

`tools/build_trt_engine.py`: `trt.Builder` + `OnnxParser` + `BuilderFlag.FP16`, 4 GB
workspace (`MemoryPoolType.WORKSPACE`). Engine sizes: 24.1 MB (720p), 23.5 MB (1080p),
fp16 weights. INT8 fails without a calibration dataset (`engine build failed` - needs an
`IInt8Calibrator`; not done). `config.builder_optimization_level = 5` timed out after
15 minutes; stayed at the default level 3.

#### 6.3. Benchmark

`tools/bench_trt.py` (pycuda, synchronized):

| Resolution | p50 | p10 | FPS | Real-time 48fps (2x@24) | Real-time 60fps |
|---|---|---|---|---|---|
| 720p (736x1280) | **20.8 ms** | 18.8 ms | **48.1** | **PASS** (< 41.7 ms) | FAIL (needs 16.6 ms) |
| 1080p (1088x1920) | **51.7 ms** | 48.6 ms | **19.4** | FAIL (needs 41.7 ms) | FAIL |

720p passes real-time 2x interpolation of 24 fps anime; p10 = 18.8 ms is near 60 fps under
favorable conditions. 1080p is just over budget; INT8 or optimization level 5 might close it.

#### 6.4. Rust integration

First approach: a Python subprocess server (`scripts/trt_server.py`). Rust launches
`python trt_server.py engine.engine <native_h> <native_w>` and streams raw RGB24 frame
pairs over stdin, reading interpolated frames from stdout (ffmpeg decode -> stdin -> TRT
-> stdout -> ffmpeg encode). Two bugs, both fixed:

1. **Deadlock on large frames.** Rust wrote both frames before reading stdout; at
   2.8-6.3 MB per frame (far above the pipe buffer) writer and reader blocked each other.
   Fix: a dedicated writer thread feeds stdin via `mpsc::sync_channel` while the main
   thread reads stdout (`src/io/video_trt.rs`).
2. **Frame-size mismatch (the real hang on real video).** The engine expects /32-padded
   input (720p -> **736x1280**) while Rust sent native 720x1280 and only NN-resized the
   output; the server blocked waiting for missing bytes. Fix: `trt_server.py` now works in
   native resolution - it zero-pads to engine size itself (bottom/right, as `F.pad` in
   `compare_pytorch.py`), infers, crops the output (top-left) and returns native frames;
   Rust passes native H/W and the erroneous output resize was removed.

**Status: works.** End-to-end on `demo/test_720p.mp4` (1280x720, 72 frames): 72 in ->
143 out (2x, 48 fps), exit 0, no hangs. Pass-through frames match the source (MAE ~0.6 =
h264 noise); interpolated frames are a correct midpoint. Wall time ~11.6 fps at 720p
(Python subprocess + ffmpeg overhead included; pure TRT inference stays 20.8 ms = 48 fps).
Limitation: `--times` only 2 (fixed timestep=0.5 in the engine).

---

## 2. Project layout

```
InterModule/
|-- Cargo.toml              # candle 0.11, features: cuda, cudnn (broken), bin
|-- .cargo/config.toml      # target-dir = E:\cargo-target\InterModule (C: full)
|-- AGENTS.md               # build/lint/test commands
|-- README.md               # project vision (anime 24->60fps in browser)
|-- docs/
|   |-- result.md               # this report
|   |-- rife_lite_reference.md  # PyTorch reference (ground truth for the port)
|-- tools/
|   |-- convert_weights.py     # flownet.pkl -> rife_lite.safetensors
|   |-- compare_pytorch.py     # ground truth generator (PyTorch inference_img)
|   |-- psnr_compare.py        # PSNR Rust vs PyTorch
|   |-- export_onnx.py         # PyTorch -> ONNX (dynamo, opset 20)
|   |-- build_trt_engine.py    # ONNX -> TensorRT engine (fp16)
|   |-- build_trt_best.py      # ONNX -> TensorRT (opt level 5, tactics) - unfinished
|   |-- build_trt_int8.py      # INT8 - does not work without a calibrator
|   |-- bench_trt.py           # TensorRT benchmark (pycuda)
|   |-- trt_server.py          # Python inference server for Rust
|-- src/
|   |-- lib.rs               # RifeLite API: load(), interpolate(), interpolate_scaled()
|   |-- model.rs             # IFNet_m reimplementation (325 lines)
|   |-- warp.rs              # CustomOp2 backward warp (fused CUDA kernel + CPU fallback)
|   |-- imgutil.rs           # image/tensor conversion (BGR, GPU-side)
|   |-- io/
|   |   |-- video.rs         # candle-based video pipeline (ffmpeg pipes)
|   |   |-- video_trt.rs     # TensorRT video pipeline (Python subprocess)
|   |-- bin/
|       |-- interpolate.rs   # CLI: img/video/trt subcommands
|       |-- smoke.rs         # load weights + one forward
|       |-- profile.rs       # per-component timing (with synchronize)
|-- weights/
|   |-- rife_lite.safetensors            # 190 tensors, 41 MB, fp32
|   |-- rife_lite_manifest.json          # key->shape manifest
|   |-- rife_lite_trt_fp16.engine        # TensorRT 720p engine, 24.1 MB
|   |-- rife_lite_1080p_trt_fp16.engine  # TensorRT 1080p engine, 23.5 MB
|   |-- rife_lite_1080x1920.onnx         # ONNX model, 40.2 MB
|-- demo/
    |-- I0_0.png, I0_1.png          # RIFE demo frames
    |-- mid_pytorch.png/rgb         # PyTorch ground truth
    |-- mid_rust.png                # Rust fp32 output (PSNR 88.31)
    |-- mid_rust_cuda.png           # Rust CUDA fp32 output (PSNR 88.73)
    |-- mid_rust_fp16.png           # Rust CUDA fp16 output (PSNR 56.99)
    |-- test_10s.mp4                # 10-sec 1080p 24fps anime clip
    |-- test_10s_2x*.mp4            # various 2x outputs
```

---

## 3. Full results

### Correctness (vs PyTorch inference_img.py, 256x448, timestep=0.5)

| Configuration | PSNR | diff>1 pixels | Status |
|---|---|---|---|
| Rust CPU, gather warp, fp32 | 88.31 dB | 33/344064 (0.01%) | PASS |
| Rust CUDA, fused kernel, fp32 | 88.73 dB | 0/344064 (0.0%) | PASS |
| Rust CUDA, fused kernel, fp16 | 56.99 dB | 0/344064 (0.0%) | PASS |
| Rust CUDA, fused kernel, fp16, scale=0.5 | 28.59 dB | 193018/344064 (56%) | low res |

### Speed (synchronized, real GPU time)

| Backend | Resolution | scale | Time/frame | FPS | Real-time 48fps? |
|---|---|---|---|---|---|
| candle CPU | 720p | 1.0 | 4889 ms | 0.2 | no |
| candle CPU | 1080p | 1.0 | ~11000 ms | 0.09 | no |
| candle CUDA fp32 | 1080p | 1.0 | 1069 ms | 0.9 | no |
| candle CUDA fp32 + fused warp | 1080p | 1.0 | 691 ms | 1.4 | no |
| candle CUDA fp16 + fused warp | 1080p | 1.0 | 566 ms | 1.8 | no |
| candle CUDA fp16 + fused warp + GPU download | 1080p | 1.0 | ~570 ms | 1.8 | no |
| candle CUDA fp16 + all + scale=0.5 | 1080p | 0.5 | 265 ms | 3.8 | no |
| candle CUDA fp16 + all + scale=0.5 | 720p | 0.5 | 213 ms | 4.7 | no |
| candle CUDA + cuDNN 9 (cu12) | 1080p | 0.5 | 406 ms | 2.5 | no (WORSE!) |
| **TensorRT fp16** | **720p** | **1.0** | **20.8 ms** | **48.1** | **PASS** |
| TensorRT fp16 | 1080p | 1.0 | 51.7 ms | 19.4 | no (close) |

### Video pipeline (wall time, including ffmpeg decode/encode)

| Backend | Resolution | scale | Wall time | Frames | Effective FPS |
|---|---|---|---|---|---|
| candle CPU | 720p | 1.0 | 353 s | 71 inter | 0.2 |
| candle CUDA fp32 | 720p | 1.0 | 26 s | 71 inter | 2.7 |
| candle CUDA fp16 | 1080p | 0.5 | 79 s | 239 inter | 3.0 |
| TensorRT (pipeline) | 720p | 1.0 | - | - | hung (I/O bugs, fixed - see 6.4) |

---

## 4. Lessons

- Always call `dev.synchronize()` before `Instant::elapsed()` when timing GPU code;
  `src/bin/profile.rs` does this correctly (see stage 3 for the phantom-timing incident).
- For grid_sample/warp-style ops, write a fused CUDA kernel - gather is random memory
  access; candle's `CustomOp2` trait + NVRTC runtime compilation is a workable path.
- Tensor cores only pay off on compute-bound shapes (large batches, large channel counts);
  batch=1 inference here is memory-bandwidth bound, hence fp16 = 1.22x not 2x.
- NVIDIA pip wheels are tied to the CUDA major version: cu12 != cu13. Use the native cuDNN
  ZIP matching the CUDA version.
- TensorRT (the engine behind `torch.compile` + `torch_tensorrt`) gave 10x over candle
  (1069 ms -> 20.8 ms at 720p) via fusion, auto-tuning, fp16 tensor cores and layout
  optimization. candle is good for prototyping and portability, not production inference
  speed - on NVIDIA, TensorRT is the only path to real-time.
- Disk space was a hidden blocker: C: was 99.9% full (0.5 GB free) and cargo died with
  "no space on device"; fixed via `target-dir = "E:\cargo-target\InterModule"`. Note
  `tokenizers` is a mandatory (non-feature-gated) candle-core dependency on non-wasm,
  ~500 MB at build time.

---

## 5. Open problems and speedup ideas

Fixed along the way: the TRT pipeline I/O deadlock (frames larger than the pipe buffer,
plus the /32 padding mismatch) - see 6.4.

Not reached:
- **1080p real-time:** 51.7 ms vs the 41.7 ms budget for 48 fps.
- **60 fps (16.6 ms) at any resolution:** 720p p10 = 18.8 ms - close, not stable.
- **Browser path (wasm + WebGPU):** not started. candle 0.11 has no wgpu feature
  (cuda/metal only); the browser needs `ort` (ONNX Runtime Web) with the WebGPU EP or
  custom wgpu compute shaders, plus WebCodecs decode and Canvas paint. A separate project,
  no candle reuse.
- **Anime mode:** not started. SVP's "Animation" mode detects only global motion
  (pan/zoom) and interpolates that, leaving static objects untouched; raw RIFE smears
  anime (hard contours, sudden occlusion).

Speedup ideas, prioritized:
1. **INT8 with calibration (potential 2-3x).** ~500 representative anime frames; subclass
   `trt.IInt8EntropyCalibrator2` (implement `get_batch` returning [img0, img1] as np
   arrays, `get_batch_size` = 1, `read_calibration_cache`, `write_calibration_cache`);
   rebuild with `BuilderFlag.INT8`. Expected: 720p 20.8 -> ~7-10 ms (60 fps PASS),
   1080p 51.7 -> ~20-25 ms (48 fps PASS).
2. **CUDA Graphs (1.3-1.5x).** `context.record_cursor()` + `context.replay()`; removes
   kernel launch overhead (~200 kernels x ~5-10 us = 1-2 ms). Not in the Python benchmark.
3. **Optimization level 5 (1.1-1.3x).** Tries all tactic combinations; did not finish in
   15 minutes, needs a 30-60 minute timeout; may give 10-30% on some convolutions.
4. **Stream overlap (1.2x).** Decode the next frame on a second CUDA stream; needs double
   buffering.
5. **Remove block_tea from the ONNX (5-10%).** IFBlock(16+4+1, c=90), unused at inference
   (needs gt) - one IFBlock out of four.
6. **Fuse preprocessing into the graph (2-5 ms).** RGB->BGR + /255 + pad currently run
   outside the engine; in-graph they fuse with the first conv.
7. **Native Rust TensorRT bindings.** Removes the ~5-10 ms/frame-pair Python stdin/stdout
   overhead; all inference in one Rust process. (Since completed - see Conclusions.)
8. **Dynamic shapes.** One engine for any resolution instead of fixed 736x1280 /
   1088x1920, at a 10-20% speed penalty.

---

## 6. Reproduction commands

### Build and test (CPU, no GPU)

```pwsh
cargo check                          # typecheck
cargo clippy --all-targets -- -D warnings  # lint
cargo test                           # warp_identity test
cargo run --release --bin rife-smoke -- --weights weights\rife_lite.safetensors
cargo run --release --bin rife-interpolate -- --weights weights\rife_lite.safetensors img --img0 demo\I0_0.png --img1 demo\I0_1.png --out demo\mid_rust.png --timestep 0.5
```

### Build with CUDA

```pwsh
# vcvars64.bat must be on PATH for nvcc
cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && cargo build --release --features cuda"
```

### Profiling

```pwsh
cmd /c "`"...\vcvars64.bat`" >nul 2>&1 && cargo run --release --features cuda --bin rife-profile -- --weights weights\rife_lite.safetensors --h 1080 --w 1920 --scale 1.0"
```

### TensorRT

```pwsh
# 1. Export ONNX
python scripts\export_onnx.py 720 1280
# 2. Inline weights
python -c "import onnx; m=onnx.load('weights/rife_lite_720x1280.onnx', load_external_data=True); onnx.save_model(m, 'weights/rife_lite_720x1280.onnx', save_as_external_data=False)"
# 3. Build TRT engine
python scripts\build_trt_engine.py weights\rife_lite_720x1280.onnx weights\rife_lite_trt_fp16.engine
# 4. Benchmark
python scripts\bench_trt.py weights\rife_lite_trt_fp16.engine
```

### Weight conversion (one-time)

```pwsh
python scripts\convert_weights.py flownet.pkl weights\rife_lite.safetensors --manifest weights\rife_lite_manifest.json
```

### PSNR check

```pwsh
python scripts\compare_pytorch.py    # ground truth
cargo run --release --features cuda --bin rife-interpolate -- --weights weights\rife_lite.safetensors img --img0 demo\I0_0.png --img1 demo\I0_1.png --out demo\mid_rust.png --timestep 0.5
python scripts\psnr_compare.py
```

---

## 7. Dependencies and environment

- **Rust:** `candle-core = "0.11"`, `candle-nn = "0.11"` (Conv2d, PReLU, etc.),
  `anyhow = "1"`, `half = "2"` (f16), `clap = "4"`, `image = "0.25"` (PNG I/O).
- **Python:** `torch = "2.10.0+cpu"` (model loading, ONNX export), `onnx` (inlining),
  `onnxscript` (dynamo exporter dependency), `tensorrt = "10.13.3.9"`, `pycuda`,
  `opencv-python` (BGR reference I/O). `nvidia-cudnn-cu12 = "9.23.2.1"` - DO NOT USE
  (conflicts with CUDA 13); `nvidia-cublas-cu12` - only needed if cuDNN is enabled.
- **System:** CUDA Toolkit 13.1 (nvcc 13.1.80); Visual Studio 2022 (MSVC 14.44) for nvcc
  and the linker; ffmpeg 8.0.1 (decode/encode) + ffprobe.

---

## 8. Conclusions

### What works

1. **The RIFE-Lite port to Rust/candle** - verified at PSNR 88.31 dB, pixel-accurate.
2. **The custom fused CUDA warp kernel** - 1.55x, correctness preserved.
3. **TensorRT fp16 at 720p** - real-time 48 fps PASS (20.8 ms against a 41.7 ms budget).
4. **The full mp4 -> 2x -> mp4 pipeline** via ffmpeg pipes (candle version).
5. **Weight conversion** pkl -> safetensors, keys matching 1-to-1.
6. **Native in-process TensorRT from Rust (no Python)** - direct FFI to `nvinfer_10.dll`
   through a C++ shim (`csrc/trt_shim.cpp`, `build.rs` feature `trt`, `src/trt.rs`
   `RifeTrt`). TRT headers taken from GitHub `release/10.13`; the import lib was generated
   from the DLL (no NVIDIA login needed). Speed equals engine speed: 720p fp16 21.6 ms
   (48 fps PASS), INT8 17.1 ms / p10 16 ms, 1080p 52.9 ms. Replaced `trt_server.py`
   (deleted). End-to-end `rife-trt` pipeline: 720p 72 -> 143 frames in 2.42 s (59 fps wall
   vs 26 fps on the old Python path); correctness confirmed (pass-through MAE 0.6,
   interpolation is a symmetric midpoint, BGR order correct). The `ort` path was rejected:
   on CUDA 13 the ORT CUDA EP returns zeros (the cuDNN cu12 conflict) and the TRT EP fails
   to load.

### Real-time summary (native, in-process)

| Resolution | infer | 48fps (2x@24) | wall pipeline |
|---|---|---|---|
| 720p fp16 | 21.6 ms | PASS | 59 fps (2.42 s / 72 frames) |
| 720p INT8 | 17.1 ms (p10 16.0) | PASS | - |
| 1080p fp16 | 52.9 ms | FAIL (close) | - |

Still open: 1080p real-time (51.7 ms vs 41.7 ms), stable 60 fps (720p p10 = 18.8 ms),
cuDNN on CUDA 13, the browser path, anime mode - see section 5.

### Recommended next steps

**INT8 quantization with calibration** is the fastest route to 60 fps at 720p and
real-time at 1080p (expected: 720p -> ~8 ms = 120 fps, 1080p -> ~20 ms = 48 fps PASS).
The second priority, native Rust TensorRT bindings, has since landed (item 6 above). Third:
the browser path via the `ort` crate + WebGPU - candle is no longer needed for production
inference, but the Rust port of the model remains the reference implementation for
validation.
