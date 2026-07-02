# Framecast web

Browser demo + probes for the ort-web (WebGPU) path.

## Run

```
python -m http.server 8123 --bind 127.0.0.1
```

- `web/demo.html` — drop two frames (or "Use demo pair") → interpolated middle frame + slider.
- `web/probe.html` — ceiling probe: p50/p10 fps, mirrored ort-web console (node placement +
  per-kernel WebGPU profiling), fp32/fp16 × webgpu/wasm.
- `web/slowmo.html` — 2× slow-mo: load a short clip → precompute interpolated frames → play
  smooth. Not real-time (precompute-then-play). Fixed 2× (model timestep=0.5).

## Pre/post

`web/rife_prepost.js` mirrors `crates/rife-core/src/prepost.rs` (BGR, ÷255, pad-to-32, crop)
over RGBA8 canvas pixels. Unit tests: `node web/prepost.test.js`.

## Backends (EP dropdown)

- **webgpu** — works everywhere with WebGPU, but ~1957 ms/frame at 720p on a 4060 Ti (the ort-web
  WebGPU EP does not fuse the graph).
- **webnn** — routes to **DirectML** on Windows, which fuses: **~558 ms (3.5× faster)**, output
  identical. Requires the pages to load `ort.all.min.js` (they do) and the Chrome flag
  `chrome://flags/#web-machine-learning-neural-network` → **Enabled** → Relaunch (experimental,
  off by default in stable Chrome). First session build is slow (~3.4 s, one-time).
- **wasm** — CPU baseline (~9 s), for correctness checks only.

Measured levers (4060 Ti): fp16, graph-capture, gpu-buffer output = 0% on WebGPU; WebNN = 3.5×.
See `../docs/phase2_ortweb_ceiling.md`.

## Resolutions (res dropdown)

Fixed-shape models, one per resolution; cost scales ~linearly with pixels (bandwidth-bound).
Measured webgpu p50 on the 4060 Ti: **720p = 1957 ms · 480p = 1064 ms · 360p = 655 ms**
(WebNN ≈ ⅓ of each). Frames larger than the engine are downscaled to fit before inference.

## Quality tiers (quality dropdown)

Tiers (see `../docs/phase5_ablation.md`):
**full** (whole net) · **fast** (refinement cut: ~1.5× on webgpu, −1.3..+0.2 dB) ·
**fastest** (distilled 2-block student: ~4× on webgpu, −0.5..−1.4 dB vs full).
Measured on the 4060 Ti: 480p/fastest webgpu = **~250 ms (4 fps)**; on WebNN (DirectML)
= **96 ms (10.4 fps)**, 360p/fastest WebNN = **59 ms (16.9 fps)** — measured in real
Chrome via `web/webnn_autotest.html` + `tools/collect_server.py`.

Regenerate the models (assets/ is gitignored):

```
python tools/restore_pkl.py                    # rebuild flownet.pkl from models/rife_lite.safetensors
python tools/export_onnx.py 480 854            # full model  -> assets/rife_lite_480x854.onnx
python tools/export_onnx.py 360 640            #             -> assets/rife_lite_360x640.onnx
python tools/export_ablation.py 720 1280       # all 5 ablation variants at 720p
python tools/export_ablation.py 480 854 noref 2blk_noref
python tools/export_ablation.py 360 640 noref 2blk_noref
```

## Notes

- Models are fixed-shape (`720`: `[1,3,736,1280]`, `480`: `[1,3,480,864]`, `360`: `[1,3,384,640]`);
  smaller frames are zero-padded to the engine size and cropped back (native ≤ engine), same as
  the native TensorRT path. The map lives in `rife_session.js` (`MODELS`).
- Verified: on the demo pair the browser output matches the PyTorch middle frame
  (`demo/mid_pytorch.png`) to `mean|Δ|=0.10`, `max|Δ|=22` — pixel-accurate.
- Real WebGPU **speed** must be measured in a browser with a real GPU adapter (Playwright's
  chromium has `navigator.gpu === false`; it can only verify correctness on the wasm EP).
