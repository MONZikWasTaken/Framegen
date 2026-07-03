# Framecast

Real-time neural frame interpolation for video **in the browser** — a Chrome
extension that takes any `<video>` on any page (24 fps anime, 30 fps footage)
and plays it at 2×–6× the frame rate, entirely on your GPU, with no server and
no external ML runtime.

Everything in the hot path is ours:

- **The model** — a distilled, slimmed RIFE-family student (IFNet_m, arbitrary
  timestep) trained by us; 4 MB of weights.
- **The inference runtime** — hand-written WGSL compute kernels on raw WebGPU
  (`web/rt/rt.js`), bit-exact against the ONNX reference. No onnxruntime-web,
  no TensorFlow.js, no WASM BLAS. The same kernels run natively via `wgpu`
  (`crates/rife-wgpu`).
- **The pipeline** — fully GPU-resident: video frame → texture → dedup/cut
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

Browser total speed-up: **×390** (1957 → 5 ms). The remaining gap to TensorRT
is structural — WGSL has no tensor-core access yet (waiting on
`subgroup-matrix` shipping in Chrome).

## Model ladder (distilled from RIFE-lite, PSNR on held-out clips)

| student | BBB dB | Jellyfish dB | weights | 720p mid |
|---|---|---|---|---|
| teacher (full) | 41.50 | 37.67 | 16 MB | 21.6 ms (TRT) |
| 2-block | 40.14 | 37.14 | 8 MB | — |
| 1-block | 39.68 | 35.25 | 16 MB f32 | 25 ms (WGSL) |
| slim c=120 | 39.31 | 34.83 | 4 MB | 9.8 ms |
| **t-factored slim (default)** | **39.48** | **34.99** | **4.5 MB** | **4.9 ms trunk + 2.1 ms/mid** |
| potato c=60 | 39.05 | 34.44 | 1 MB | 6 ms |

The t-factored student splits the network into a timestep-free trunk (run once
per frame pair) and a tiny FiLM(t) head (run per interpolated frame): at 6x a
mid costs ~2 ms instead of a full 8 ms pass (~2.4x), and it scores HIGHER than
plain slim at every timestep (t=0.25: 37.08 vs 36.86 dB on stride-4 pairs).

All students keep **arbitrary timestep** (t = k/n for 2×–6× factors) — trained
with stride-4 ground-truth samples, not just t=0.5. Plus **TinySR**: a 26 KB
residual 2× upscaler (+1.1 dB over bilinear) applied to interpolated frames.

## The extension (`extension/`)

Chrome MV3, works on YouTube and most video sites (`all_frames` covers
cross-origin iframe players):

- 2×–6× factor, or **auto** — an AIMD controller driven by a leaky-bucket drop
  detector picks the highest factor the GPU and compositor actually sustain
- anime mode: GPU dedup detects frames drawn "on twos" and doubles the budget;
  scene-cut detection avoids interpolating across cuts
- quality presets 360p → 1080p inserts, hot-swapped without restart
- own glass player UI (the canvas covers native controls): play/seek/volume,
  fullscreen, click-to-pause, compare slider (original | interpolated),
  optional HDR via inverse tone mapping, debug telemetry HUD
- just-in-time GPU scheduling: each mid is submitted one compute-time before
  its display slot; presentation delay is ~2 frame-times, not a whole batch

**Install:** grab `framecast-extension.zip` from Releases (or run
`tools/build_extension.ps1` with weights in `assets/`), then
`chrome://extensions` → Developer mode → Load unpacked.

## Repo layout

```
web/rt/rt.js         WGSL inference runtime (the heart of the project)
web/rt/sr.js         TinySR 2x upscaler kernels
web/player.html      standalone real-time player demo (+ worker)
web/rt_test.html     bit-exactness harness vs ONNX reference
extension/           Chrome extension (content.js = full pipeline)
crates/rife-wgpu     same kernels on native wgpu (Rust)
src/, csrc/          native TensorRT path (engines, video pipeline)
tools/               training (distill/SR), export, benchmarks, packaging
docs/                measurements and phase notes
```

## Training your own weights

`tools/train_student.py` distills from the RIFE-lite teacher (`--slim C` for
thin channels, `--arbitrary-t` to keep timestep conditioning — without it any
finetune collapses t≠0.5). `tools/train_sr.py` trains TinySR.
`tools/export_rt_weights.py` / `export_sr_weights.py` produce the `.bin/.json`
blobs the runtime loads. Frames are extracted with `tools/extract_frames.py`
from any movies you have locally.

## Known limitations

- **Chrome only** (WebGPU + shader-f16; the UI also uses base-select).
- **DRM sites (Netflix, Crunchyroll/EME) cannot work** — the browser hands us
  black frames by design. YouTube and plain `<video>`/MSE sites are fine.
- SDR sources only get *simulated* HDR (inverse tone mapping, RTX-Video-HDR
  style) — the browser never exposes true HDR video data.
- Interpolation is honest about impossible cases: 5 fps sources have too little
  information between frames; artifacts on fast motion are expected there.

## License

Code is MIT (see `LICENSE`). **Model weights** are distributed for
non-commercial research/personal use — they were trained on open movies
(Sintel, Tears of Steel, Elephants Dream, Big Buck Bunny) but distilled from a
RIFE-family teacher and evaluated on research datasets; treat them accordingly.
