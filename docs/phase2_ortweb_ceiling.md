# Phase 2 - ort-web WebGPU ceiling probe

> Historical note: the ort-web-era files referenced below (probe.html, rife_session.js,
> restore_pkl.py, export_fp16.py) were removed once the custom WGSL runtime replaced
> that stack; they live in git history.


Goal (ROADMAP Phase 2): find the exact CPU-fallback nodes on the ort-web WebGPU EP,
measure fp16, measure 480p - decide whether the ort-web path is viable or whether we
must build our own WGSL warp (Phase 3).

## What is HW-independent (done here, no GPU needed)

### Operator inventory of `rife_lite_inlined.onnx` (opset 20, 316 nodes, 15 types)

| count | op | WebGPU EP (ort-web) |
|------:|----|---------------------|
| 58 | PRelu | supported |
| 55 | Conv | supported |
| 43 | Slice | supported |
| 32 | Concat | supported |
| 29 | Div | supported |
| 24 | Add | supported |
| 18 | Mul | supported |
| 15 | Resize | supported |
| 14 | Transpose | supported |
| **14** | **GridSample** | **NOT supported → CPU fallback** |
| 7 | ConvTranspose | supported |
| 2 | Expand | supported |
| 2 | Sigmoid | supported |
| 2 | Sub | supported |
| 1 | Clip | supported |

The model uses GridSample ×14, all identical: `align_corners=1, mode=linear, padding_mode=border`.

**CORRECTION (2026-07-02) - this flips the earlier assumption:**
The 1712 ms spike blamed GridSample CPU-fallback + "14 roundtrips". But that was an **older
ort-web**. Current **ort-web 1.27.0 ships a WebGPU GridSample kernel** - verified directly in
the bundle the classic `ort.webgpu.min.js` build loads (`ort-wasm-simd-threaded.jsep.mjs`):

```
f.$b("GridSample", a, {align_corners, mode, padding_mode, format: NCHW/NHWC})
["GridSample", [hv, mv]]   // entry in the JSEP (WebGPU) kernel registry
```

The kernel supports exactly our attribute set (linear / border / align_corners, NCHW). So with
current ort-web, **GridSample should run on the GPU - no CPU fallback, no roundtrips.**

**Implication:** ROADMAP Phase 3 (hand-rolled WGSL warp replacing GridSample) is **very likely
unnecessary** - it would duplicate an upstream kernel that now exists. Do NOT start Phase 3
before the real-GPU run below confirms whether GridSample still falls back on current ort-web.
If it runs on GPU, the 1712 ms should collapse and the browser path becomes viable with zero
custom-shader work.

## MEASURED on the real 4060 Ti (2026-07-02) - overturns the optimism above

Ran the WebGPU EP on the actual RTX 4060 Ti (adapter: nvidia / lovelace), current ort-web 1.27.0,
at the fixed 736×1280, via an auto-runner that POSTs results back:

| model | p50 | fps | output |
|-------|----:|----:|--------|
| fp32 webgpu | **1920 ms** | 0.5 | correct (mean 0.4998) |
| fp16 webgpu | **1893 ms** | 0.5 | correct |

Log evidence:
- `WebGPU EP force CPU node count: 0` → GridSample **does run on the GPU** (the kernel exists, as
  found). So "add the missing GridSample kernel" was NOT the fix.
- `Some nodes were not assigned to the preferred EP … ORT explicitly assigns shape related ops to
  CPU` → the Slice/Concat/shape gymnastics around the 14 warps stay on CPU → GPU↔CPU syncs.

**The real conclusion (corrects the "Phase 3 moot" note):**
- ort-web WebGPU is **~1900 ms/frame (0.5 fps)** for this model on a 4060 Ti - **fp16 gives ~0%**.
- Native TensorRT runs the *same model on the same GPU* at ~25 ms (40 fps). The browser is
  **~75× slower** - that gap is ort-web's WebGPU **runtime overhead** (per-op dispatch + the
  shape-op CPU roundtrips), **not** the model and **not** raw GridSample.
- Therefore: a hand-rolled WGSL warp (Phase 3) **alone won't fix it** - it removes GridSample's
  cost, but GridSample already runs on GPU; the loss is the surrounding roundtrips + general
  ort-web dispatch overhead. The levers that actually move it: **(a) own wgpu backend** controlling
  the whole graph (Phase 6, big), **(b) model shrink** (Phase 5 - helps proportionally but ort-web
  per-op overhead persists), or **(c) accept the browser as offline/slow-mo-only, not real-time.**

## Speed-up levers TESTED on the 4060 Ti (2026-07-02) - all cheap ones are DEAD

Measured each on the real GPU (auto-runner → POST). Baseline ~1950 ms/frame.

| lever | what it removes | result | verdict |
|-------|-----------------|-------:|---------|
| fp16 | ½ the FLOPs | 1893 ms | **0%** → not FLOP-bound |
| gpu-buffer output | output download | 1990 ms | **0%** → not IO-bound |
| `enableGraphCapture` + GPU IO-binding | per-op JS→WebGPU dispatch | 1945 ms | **0%** → not JS-dispatch-bound |
| WebNN EP (DirectML) | unfused execution | **558 ms** | **3.5× WIN** (measured, see below) |

**Diagnosis:** fp16 halving the math changed nothing → the cost is **not compute/FLOPs**. Graph
capture removing JS dispatch changed nothing → **not JS overhead**. Output on GPU changed nothing
→ **not transfers**. What's left: the graph runs as ~316 **unfused** WebGPU dispatches with
CPU-side shape ops (Slice/Concat) forcing GPU↔CPU **syncs between segments** - and those syncs +
per-dispatch latency are intrinsic to ort-web's WebGPU EP. Native TensorRT hits ~25 ms because it
**fuses** ~316 nodes into a handful of kernels; ort-web does not fuse.

### WebNN / DirectML - the one cheap lever that WORKS (measured on 4060 Ti, 2026-07-02)

Enabled Chrome flag `#web-machine-learning-neural-network` (WebNN → DirectML on Windows), ran the
same model via `executionProviders:[{name:'webnn',deviceType:'gpu'}]`:

| EP | p50 | fps | output |
|----|----:|----:|--------|
| webgpu (ref) | 1957 ms | 0.5 | correct |
| **webnn (DirectML)** | **558 ms** | **1.8** | correct (mean 0.4998) |
| webnn + wasm fallback | 523 ms | 1.9 | correct |

**~3.5× for free** - DirectML fuses/optimizes the graph where the WebGPU JSEP does not. Output is
correct. Still 22× off native TRT (25 ms), so DirectML-via-WebNN isn't native-class, but it's the
only zero-model-work multiplier that moved the needle.
Caveats: needs the experimental WebNN flag (not on by default in stable Chrome yet, but shipping);
session build is slow (~3.4 s, one-time). Use `ort.all.min.js` (not `ort.webgpu.min.js`) for the
WebNN EP.

**Updated conclusion:** WebGPU JSEP alone has no cheap win, but **WebNN/DirectML gives ~3.5×**.
To go further you still reduce what the runtime dispatches or replace the runtime:
1. **Own fused wgpu backend** (ROADMAP Phase 6, months) - the only path to native-class speed in
   the browser; you control fusion + keep everything GPU-resident.
2. **Model shrink** (Phase 5, weeks) - but since it's dispatch/sync-bound not FLOP-bound, the win
   comes from **fewer nodes**, not fewer FLOPs → sub-linear; halving nodes ≈ ~2× at best.
3. **Lower resolution** (blocked: needs re-export) - helps compute, but we're not compute-bound,
   so expect modest gains.
4. **Accept browser = offline / slow-mo-only** and keep real-time on native TensorRT.

Honest target read: **"at least 2×" (→ ~950 ms) needs real work** (model shrink or own backend) -
no config flag does it. **30 fps real-time in-browser = own fused backend = months.**

## What needs the real GPU (hand-off - run in your Chrome on the 4060 Ti)

The exact per-node placement log and real fps can only come from a browser with a real
WebGPU adapter (Playwright's chromium here has `navigator.gpu === false`).

Harness: `web/probe.html`. It mirrors ort-web's verbose console (node placement +
per-kernel WebGPU profiling) onto the page, and reports p50/p10/fps + output sanity.

Run:
```pwsh
# from repo root
python -m http.server 8123 --bind 127.0.0.1
# open http://127.0.0.1:8123/web/probe.html in Chrome (WebGPU on)
```
Then, for each combo, click Run and copy the log box back:
1. model=fp32, ep=webgpu  → baseline (expect ~1700 ms, GridSample on CPU)
2. model=fp16, ep=webgpu  → does half-precision move the needle?
3. (ep=wasm is the CPU floor; verified here ~9 s at 736×1280)

What to look for in the mirrored log:
- lines mentioning `GridSample` / `fallback` / `CPU` → confirms the fallback set.
- the `webgpu profiling` per-kernel times → the actual hotspots (are the 14 GridSample
  roundtrips the whole cost, or do Conv/Resize also dominate?).
- `RESULT p50=… fps=…` for fp32 vs fp16.

## Resolution curve (measured 2026-07-02, 4060 Ti, webgpu EP, clean runs)

The weights were UNBLOCKED: `tools/restore_pkl.py` rebuilds `flownet.pkl` from
`models/rife_lite.safetensors` (inverse of `convert_weights.py`; verified bit-exact -
PyTorch-on-restored-pkl vs fresh 480p ONNX on CPU EP: mean|Δ|=0, max|Δ|=0). Any-size export:
`python tools/export_onnx.py H W`.

| model (engine shape) | webgpu p50 | fps | vs 720p |
|---|---:|---:|---:|
| 720p (736×1280) | 1957 ms | 0.5 | 1× |
| 480p (480×864)  | 1064 ms | 0.94 | 1.84× |
| 360p (384×640)  | 655 ms  | 1.53 | 3.0× |

Cost scales ~linearly with pixel count (944k → 415k → 246k px) - consistent with
bandwidth-bound, not fixed-dispatch-bound. Extrapolated WebNN/DirectML (÷3.5):
480p ≈ 300 ms (3.3 fps), 360p ≈ 190 ms (5+ fps). The demo/slowmo pages now have a
resolution selector (`web/rife_session.js` `MODELS`).

## Deferred

- The fp16 model is validated (loads + runs, output ≈0.5 on flat 0.5 input) but fp16 was
  measured at 0% gain on WebGPU, so no fp16 variants of the small models were made.

## Artifacts

- `web/probe.html` - the probe harness (console-mirroring, per-model dims, EP × model matrix).
- `tools/restore_pkl.py` - rebuild `flownet.pkl` from `models/rife_lite.safetensors`.
- `tools/export_fp16.py` - reproducible fp32→fp16 ONNX conversion (no torch).
- `assets/rife_lite_fp16.onnx`, `assets/rife_lite_480x854.onnx`, `assets/rife_lite_360x640.onnx`
  - generated models (gitignored, regenerable).
