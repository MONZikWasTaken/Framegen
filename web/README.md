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

## Notes

- Model `assets/rife_lite_inlined.onnx` is fixed-shape `[1,3,736,1280]`; smaller frames are
  zero-padded to that and cropped back (native ≤ engine), same as the native TensorRT path.
- Verified: on the demo pair the browser output matches the PyTorch middle frame
  (`demo/mid_pytorch.png`) to `mean|Δ|=0.10`, `max|Δ|=22` — pixel-accurate.
- Real WebGPU **speed** must be measured in a browser with a real GPU adapter (Playwright's
  chromium has `navigator.gpu === false`; it can only verify correctness on the wasm EP).
