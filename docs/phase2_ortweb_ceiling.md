# Phase 2 — ort-web WebGPU ceiling probe

Goal (ROADMAP Phase 2): find the exact CPU-fallback nodes on the ort-web WebGPU EP,
measure fp16, measure 480p — decide whether the ort-web path is viable or whether we
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

**CORRECTION (2026-07-02) — this flips the earlier assumption:**
The 1712 ms spike blamed GridSample CPU-fallback + "14 roundtrips". But that was an **older
ort-web**. Current **ort-web 1.27.0 ships a WebGPU GridSample kernel** — verified directly in
the bundle the classic `ort.webgpu.min.js` build loads (`ort-wasm-simd-threaded.jsep.mjs`):

```
f.$b("GridSample", a, {align_corners, mode, padding_mode, format: NCHW/NHWC})
["GridSample", [hv, mv]]   // entry in the JSEP (WebGPU) kernel registry
```

The kernel supports exactly our attribute set (linear / border / align_corners, NCHW). So with
current ort-web, **GridSample should run on the GPU — no CPU fallback, no roundtrips.**

**Implication:** ROADMAP Phase 3 (hand-rolled WGSL warp replacing GridSample) is **very likely
unnecessary** — it would duplicate an upstream kernel that now exists. Do NOT start Phase 3
before the real-GPU run below confirms whether GridSample still falls back on current ort-web.
If it runs on GPU, the 1712 ms should collapse and the browser path becomes viable with zero
custom-shader work.

## What needs the real GPU (hand-off — run in your Chrome on the 4060 Ti)

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

## Blocked / deferred

- **480p measurement:** the models are fixed-shape (736×1280, 1080×1920). A 480p ONNX needs
  a re-export from the RIFE checkpoint. The reference code still exists at
  `%TEMP%\opencode\rife_ref`, but the weights dir `%TEMP%\opencode\rife_m\RIFE_m_train_log`
  is gone. To do 480p: recover `flownet.pkl` and run
  `python tools/export_onnx.py 480 854` (then inline + `tools/export_fp16.py`).
- The fp16 model is validated (loads + runs, output ≈0.5 on flat 0.5 input) but its real-GPU
  speed/quality vs fp32 is the open question the hand-off answers.

## Artifacts

- `web/probe.html` — the probe harness (console-mirroring, fp32/fp16 × webgpu/wasm).
- `tools/export_fp16.py` — reproducible fp32→fp16 ONNX conversion (no torch).
- `assets/rife_lite_fp16.onnx` — generated fp16 model (gitignored, regenerable).
