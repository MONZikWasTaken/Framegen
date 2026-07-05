# AGENTS.md - Framegen

Real-time RIFE-Lite (RIFEm) frame interpolation. Two backends behind one `FrameInterpolator` trait:
- **candle** (`RifeCandle`) - pixel-accurate Rust/candle port, used as the correctness reference/oracle.
- **TensorRT** (`RifeTrt`, feature `trt`) - native in-process FFI, the real-time path (no Python).

The neutral `rife-core` crate owns `Frame`, the `FrameInterpolator` trait, and the single-source
pre/post (`to_input`/`from_output`: BGR, /255, pad-to-32, crop). Both backends route through it.

See `docs/result.md` for the full engineering journal.

## Build / typecheck / lint

```pwsh
cargo check                                  # typecheck (default = CPU/candle)
cargo build --release                        # candle CLIs
cargo clippy --all-targets -- -D warnings    # lint
cargo test                                   # unit tests
cargo build --release -p framegen --features "bin trt" --bin rife-trt   # native TensorRT (needs MSVC + CUDA)
```

## Layout

```
extension/          Chrome MV3 extension (content.js = full pipeline)
web/                WGSL runtime (rt/), player demo, parity harness (rt_test.html)
crates/rife-core/   Frame, FrameInterpolator trait, prepost (to_input/from_output) - no candle/cuda deps
crates/rife-wgpu/   the WGSL kernels hosted on native wgpu (Vulkan/DX12/Metal)
crates/framegen-native/  candle oracle + native TensorRT backend (package name: framegen)
  src/lib.rs        RifeCandle (interpolate_scaled tensor API) + impl FrameInterpolator
  src/model.rs      IFNet_m reimplementation on candle
  src/warp.rs       backward warp (fused CUDA CustomOp2 + CPU fallback)
  src/trt.rs        RifeTrt - native TensorRT FFI + impl FrameInterpolator (feature `trt`)
  src/imgutil.rs    candle<->prepost glue: image/tensor conversion (feature `bin`)
  src/io/           shared ffmpeg reader; candle and TRT video pipelines
  src/bin/*         CLIs: rife-interpolate, rife-smoke, rife-profile, rife-trt, rife-trt-bench
  tests/parity.rs   gated candle-vs-trt agreement test (--features trt -- --ignored)
  csrc/trt_shim.cpp extern "C" shim over nvinfer 10 (built by build.rs when `trt` is on)
third_party/tensorrt/  public TRT headers + generated nvinfer_10.lib (SDK bootstrap, gitignored)
models/             rife_lite.safetensors + manifest (source-of-truth weights)
assets/  (gitignored)  engines, onnx, caches - large / regenerable
tools/              training (distill/SR), export, benchmarks, packaging
demo/               small test fixtures (I0_*.png, test_720p.mp4, parity .rgb refs)
docs/               rife_lite_reference.md (ground truth), result.md (journal)
```

## Weight conversion (one-time, Python + torch)

```pwsh
python tools\convert_weights.py <flownet.pkl> models\rife_lite.safetensors --manifest models\rife_lite_manifest.json
```
Source checkpoint (RIFE_m / lite, `IFNet_m`):
https://drive.google.com/file/d/147XVsDXBfJPlyct2jfo9kpbL944mNeZr/view - unzip, convert `train_log/flownet.pkl`.

## Bootstrap third_party (feature `trt`, one-time, not committed)

`third_party/tensorrt/` is gitignored (NVIDIA-derived). To regenerate on a fresh checkout:
- Headers: fetch `include/*.h` from github.com/NVIDIA/TensorRT branch `release/10.13`
  into `third_party/tensorrt/include/`.
- Import lib: `dumpbin /exports nvinfer_10.dll` → `.def` → `lib /def:… /out:third_party/tensorrt/lib/nvinfer_10.lib /machine:x64`
  (nvinfer_10.dll ships in the pip `tensorrt`/`tensorrt_libs` package).

## Native TensorRT (real-time, no Python)

In-process TRT via C++ shim + FFI, feature `trt`. Runtime needs the TRT + CUDA DLLs on PATH:

```pwsh
$env:PATH="C:\Users\MONZik\AppData\Roaming\Python\Python312\site-packages\tensorrt_libs;" +
          "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin;$env:PATH"
rife-trt --engine assets\rife_lite_trt_fp16.engine --input demo\test_720p.mp4 --output out.mp4
rife-trt-bench assets\rife_lite_trt_fp16.engine          # in-process latency
```
Build engines: `python tools\build_trt_engine.py assets\rife_lite_inlined.onnx assets\rife_lite_trt_fp16.engine`
(INT8 was measured and is a dead end - only ~1.1x on a bandwidth-bound model.)

## Conventions

- candle 0.11; match PyTorch semantics exactly (align_corners, padding, PReLU, BGR) - validated
  against `inference_img.py`, not eyeballed.
- No comments in source unless marking a genuine port deviation/TODO.
- `assets/` is gitignored (regenerable). Weights of record live in `models/`.
