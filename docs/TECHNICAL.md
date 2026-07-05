# Framegen - technical notes

Everything in the hot path is ours:

- **The model** - a distilled, slimmed RIFE-family student (IFNet_m, arbitrary
  timestep) trained by us; 2.9 MB of weights (v7 small, the default).
- **The inference runtime** - hand-written WGSL compute kernels on raw WebGPU
  (`web/rt/rt.js`), matching the PyTorch/ONNX reference to within one uint8
  LSB (mean |err| < 0.01, max 1). No onnxruntime-web,
  no TensorFlow.js, no WASM BLAS. The same kernels run natively via `wgpu`
  (`crates/rife-wgpu`).
- **The pipeline** - fully GPU-resident: video frame → texture → dedup/cut
  detection → interpolation → (optional 2× super-resolution) → canvas. The CPU
  only ever sees an 8-byte dedup statistic per frame.

## Numbers (RTX 4060 Ti, 720p, one interpolated frame)

| backend | latency | note |
|---|---|---|
| ort-web WebGPU EP, full model (day 0) | 1957 ms | where we started |
| ort-web WebGPU EP, distilled student | 218 ms | model shrink alone |
| **our WGSL runtime, slim student** | **9.8 ms** | plain WebGPU, no flags |
| **our WGSL runtime, potato student** | **6 ms** | 1 MB weights |
| our kernels on native wgpu (Rust) | 9.5 ms | Vulkan/DX12/Metal, any GPU |
| TensorRT engine (NVIDIA path) | 5.8 ms | 178 fps end-to-end with audio |

Browser total speed-up: **×200-×330** depending on the student (1957 → 6-10 ms).
All latencies are p50 on an otherwise idle 4060 Ti; expect ±10% run to run.
The remaining gap to TensorRT is structural - WGSL has no tensor-core access
yet (waiting on `subgroup-matrix` shipping in Chrome). The kernels are NOT
memory-bound: measured at ~5% of achievable bandwidth (issue/shared-memory
bound), which is why INT8 and layer fusion were evaluated and rejected.

## Model ladder (distilled from RIFE-lite, PSNR on held-out clips)

| student | BBB dB | Jellyfish dB | weights | 720p mid |
|---|---|---|---|---|
| teacher (full) | 41.50 | 37.67 | 43 MB f32 | 21.6 ms (TRT) |
| 2-block | 40.14 | 37.14 | - (not exported) | - |
| 1-block | 39.68 | 35.25 | 17 MB f32 | 25 ms (WGSL) |
| slim c=120 | 39.31 | 34.83 | 4.5 MB | 9.8 ms |
| t-factored slim | 39.48 | 34.99 | 4.5 MB | 4.9 ms trunk + 2.1 ms/mid |
| tfact2 = t-factored + refine (v6) | 39.94 | 35.44 | 4.6 MB | 5.51 ms full 2x cycle |
| **v7 small (c=96, default)** | **39.83** | **35.60** | **2.9 MB** | **3.75 ms full 2x cycle** |
| potato c=60 | 39.05 | 34.44 | 1.1 MB | 6 ms |

The t-factored student splits the network into a timestep-free trunk (run once
per frame pair) and a tiny FiLM(t) head (run per interpolated frame): at 6x a
mid costs ~2 ms instead of a full 8 ms pass (~2.4x), and it scores HIGHER than
plain slim at every timestep (t=0.25: 37.56 vs 36.86 dB on stride-4 pairs).
tfact2 adds a quarter-res refine head (occlusion repair) on top and shipped as
the extension default from v0.5.0 to v0.7.4 (still selectable as "v6").

v7 small (v0.7.0) opens the v7 generation: distilled from a stronger teacher
(EMA-VFI, Apache-2.0) whose outputs were precomputed over the whole dataset.
It matches-or-beats tfact2 (summed PSNR 75.43 vs 75.38) at two-thirds the
trunk width: the full 2x cycle drops from 3.05 to 2.57 ms at 480p and from
5.51 to 3.75 ms at 720p on the reference GPU. The extension default since
v0.7.5.

All students keep **arbitrary timestep** (t = k/n for 2×-6× factors) - trained
with stride-4 ground-truth samples, not just t=0.5. Plus **TinySR**: a 26 KB
residual 2× upscaler (+1.1 dB over bilinear) applied to interpolated frames.

## The extension pipeline

Chrome MV3, works on YouTube and most video sites (`all_frames` covers
cross-origin iframe players):

- 2×-6× factor, or **auto** - an AIMD controller driven by a leaky-bucket drop
  detector picks the highest factor the GPU and compositor actually sustain
- anime mode: GPU dedup detects frames drawn "on twos" and doubles the budget;
  scene-cut detection avoids interpolating across cuts
- quality presets 360p → 1080p inserts, hot-swapped without restart
- own glass player UI (the canvas covers native controls): play/seek/volume,
  fullscreen, click-to-pause, compare slider (original | interpolated),
  optional HDR via inverse tone mapping, debug telemetry HUD
- just-in-time GPU scheduling: each mid is submitted one compute-time before
  its display slot; presentation delay is ~2 frame-times, not a whole batch
- the dedup readback is pipelined (a GPU→CPU roundtrip never blocks the frame
  loop); shaders compile in parallel off the main thread
- the hot path allocates ~1 KB per interpolated frame (measured; GC pauses
  used to be a visible frame-drop source at 240 Hz)

## Repo layout

```
extension/               Chrome extension (content.js = full pipeline)
web/rt/rt.js             WGSL inference runtime (the heart of the project)
web/rt/sr.js             TinySR 2x upscaler kernels
web/player.html          standalone real-time player demo (+ worker)
web/rt_test.html         parity harness vs the ONNX reference (+built-in bench)
crates/rife-wgpu         same kernels on native wgpu (Rust)
crates/framegen-native   native TensorRT path + candle correctness oracle
packages/rt              the runtime as an npm library (npm "framegen", LGPL)
tools/                   training (distill/SR), export, benchmarks, packaging
docs/                    measurements and phase notes
```

## Training your own weights

`tools/train_student.py` distills from the RIFE-lite teacher (`--slim C` for
thin channels, `--arbitrary-t` to keep timestep conditioning - without it any
finetune collapses t≠0.5). `tools/train_v7.py` distills from precomputed
EMA-VFI targets (`tools/make_teacher_v7.py` generates them).
`tools/train_sr.py` trains TinySR.
`tools/export_rt_weights.py` / `export_sr_weights.py` produce the `.bin/.json`
blobs the runtime loads. Frames are extracted with `tools/extract_frames.py`
from any movies you have locally.

## Known limitations

- **Chrome only** (WebGPU + shader-f16; the UI also uses base-select).
- **DRM sites (Netflix, Crunchyroll/EME) cannot work** - the browser hands us
  black frames by design. YouTube and plain `<video>`/MSE sites are fine.
- SDR sources only get *simulated* HDR (inverse tone mapping, RTX-Video-HDR
  style) - the browser never exposes true HDR video data.
- Interpolation is honest about impossible cases: 5 fps sources have too little
  information between frames; artifacts on fast motion are expected there.
