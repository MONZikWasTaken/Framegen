# AGENTS.md — Framecast

Real-time RIFE-Lite (RIFEm) frame interpolation. Two backends behind one `FrameInterpolator` trait:
- **candle** (`RifeCandle`) — pixel-accurate Rust/candle port, used as the correctness reference/oracle.
- **TensorRT** (`RifeTrt`, feature `trt`) — native in-process FFI, the real-time path (no Python).

The neutral `rife-core` crate owns `Frame`, the `FrameInterpolator` trait, and the single-source
pre/post (`to_input`/`from_output`: BGR, /255, pad-to-32, crop). Both backends route through it.

See `ROADMAP.md` for direction and `docs/result.md` for the full engineering journal.

## Build / typecheck / lint

```pwsh
cargo check                                  # typecheck (default = CPU/candle)
cargo build --release                        # candle CLIs
cargo clippy --all-targets -- -D warnings    # lint
cargo test                                   # unit tests
cargo build --release --features trt --bin rife-trt   # native TensorRT (needs MSVC + CUDA)
```

## Layout

```
crates/rife-core/   Frame, FrameInterpolator trait, prepost (to_input/from_output) — no candle/cuda deps
src/lib.rs          RifeCandle (interpolate_scaled tensor API) + impl FrameInterpolator
src/model.rs        IFNet_m reimplementation on candle
src/warp.rs         backward warp (fused CUDA CustomOp2 + CPU fallback)
src/trt.rs          RifeTrt — native TensorRT FFI + impl FrameInterpolator (feature `trt`)
src/imgutil.rs      candle<->prepost glue: image/tensor conversion (feature `bin`)
src/io/ffmpeg.rs    shared ffmpeg reader + decoder/encoder spawn
src/io/video.rs     candle ffmpeg pipeline
src/io/video_trt.rs native TensorRT ffmpeg pipeline (feature `trt`)
src/bin/*           CLIs: rife-interpolate, rife-smoke, rife-profile, rife-trt, rife-trt-bench
tests/parity.rs     gated candle-vs-trt agreement test (--features trt -- --ignored)
csrc/trt_shim.cpp   extern "C" shim over nvinfer 10
build.rs            compiles the shim when feature `trt` is on
third_party/tensorrt/  public TRT headers + generated nvinfer_10.lib (SDK bootstrap)
models/             rife_lite.safetensors + manifest (source-of-truth weights)
assets/  (gitignored)  engines, onnx, caches — large / regenerable
tools/              python build-time tooling (export, convert, build-engine, bench)
demo/               small test fixtures (I0_*.png, test_720p.mp4, test_10s.mp4)
docs/               rife_lite_reference.md (ground truth), result.md (journal)
web/                browser spike (ort-web + WebGPU)
```

## Weight conversion (one-time, Python + torch)

```pwsh
python tools\convert_weights.py <flownet.pkl> models\rife_lite.safetensors --manifest models\rife_lite_manifest.json
```
Source checkpoint (RIFE_m / lite, `IFNet_m`):
https://drive.google.com/file/d/147XVsDXBfJPlyct2jfo9kpbL944mNeZr/view — unzip, convert `train_log/flownet.pkl`.

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
INT8: `python tools\build_trt_int8.py assets\rife_lite_inlined.onnx assets\rife_lite_int8.engine demo\test_10s.mp4 1280 720 200`

## Conventions

- candle 0.11; match PyTorch semantics exactly (align_corners, padding, PReLU, BGR) — validated
  against `inference_img.py`, not eyeballed.
- No comments in source unless marking a genuine port deviation/TODO.
- `assets/` is gitignored (regenerable). Weights of record live in `models/`.
