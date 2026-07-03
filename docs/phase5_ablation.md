# Phase 5 - model shrink: ablations (1) + student [4,2] (2) + aggressive students (3)

> Historical note: webnn_autotest.html and collect_server.py referenced below were
> removed with the rest of the ort-web stack; they live in git history.


## Step 3 result - the student ladder (2026-07-02, 10k steps each, ~35 min per run)

| model | BBB dB | Jelly dB | TRT 720p | TRT 1080p | webgpu 360p | WebNN 360p |
|---|---:|---:|---:|---:|---:|---:|
| full teacher | 41.50 | 37.67 | 21.6 ms | 52.9 ms | - | - |
| student [4,2] | 40.14 | 37.14 | 11.3 ms | 22.1 ms | ~140 ms | 59 ms |
| student [8,4] | 39.54 | 35.68 | **6.6 ms (151 fps)** | **16.5 ms (60.4 fps - 1080p60 PASS)** | - | - |
| student 1blk [4] | 39.68 | 35.25 | **5.8 ms (172 fps)** | - | 84 ms | **29.4 ms (34 fps - real-time)** |

- **1080p60 native: closed** by the [8,4] student (16.5 ms < 16.7 gate).
- **Browser real-time: closed** by the 1-block student on WebNN @360p (29.4 ms), 480p = 57 ms.
- The 1blk student BEATS [8,4] on animation with HALF the nodes; [8,4] wins on live action.
  Ladder visual on the max-motion frame: `docs/student_ladder_bbb.jpg`.
- Web app tiers: `fastest` = student [4,2] (best quality/speed), `turbo` = 1blk student.
- Weights of record: `models/rife_lite_student_{2blk,2blk_s84,1blk}.safetensors` (local; manifests in git).

## u8-IO engines + variable timestep (same day)

`tools/export_u8.py` bakes prepost into the graph (uint8 HWC I/O, bit-exact vs f32:
max|Δ|=0) + a scalar `t` input (RIFEm is natively arbitrary-t). Shim/pipeline support:
`--times N`, `--skip-static`, audio passthrough, reader/writer thread overlap.
**End-to-end wall (720p, student [4,2] u8 engine): 2× = 178.6 fps (7.4 ms/infer -
the u8 engine is FASTER than its f32 twin's 11.3 ms: fused prepost + 4× less PCIe),
4× = 183 fps wall.** Morning baseline end-to-end was 59 fps → 3× on the product path.



## Step 2 result - distilled 2-block student (2026-07-02)

`tools/train_student.py`: student = teacher's block0+block1 (same graph as `2blk_noref`),
frozen full teacher, loss = LapLoss(gt) + 0.2·LapLoss(teacher) + 0.01·L1(flow distill).
20k steps × batch 24 (256² crops), ~76 min on the 4060 Ti, 44,376 triplets from
Sintel + Tears of Steel + Elephants Dream (`tools/extract_frames.py`; eval clips held out).
Best checkpoint @17k. Weights of record: `models/rife_lite_student_2blk.safetensors`.

| model | BBB dB | Jellyfish dB | webgpu 480p | TRT fp16 720p |
|---|---:|---:|---:|---:|
| full teacher | 41.50 | 37.67 | 1064 ms | 21.6 ms (46 fps) |
| `2blk_noref` untrained | 38.90 | 36.88 | 237 ms | 11.2 ms (89 fps) |
| **student (distilled)** | **40.14** | **37.14** | **~250 ms** | **11.3 ms (88 fps, 60fps PASS)** |

Recovered +1.24 dB of the −2.60 on animation (now −1.36 vs teacher), −0.53 on live action,
at identical inference cost. Visual check on the max-motion BBB triplet:
`docs/student_vs_full_bbb.jpg` - the four variants are hard to tell apart at full size.

**WebNN (DirectML) - MEASURED in real Chrome with the feature flag (2026-07-02,
`--enable-features=WebMachineLearningNeuralNetwork`, auto-test `web/webnn_autotest.html`
+ `tools/collect_server.py`):**

| config | webgpu | webnn | webnn fps |
|---|---:|---:|---:|
| 720p full | 1957 ms | 558 ms | 1.8 |
| 480p full | 1064 ms | 240 ms | 4.2 |
| 720p student | ~495 ms | 211 ms | 4.7 |
| 480p student | ~250 ms | **96 ms** | **10.4** |
| 360p student | - | **59 ms** | **16.9** |

The WebNN multiplier shrinks as the graph shrinks (3.5× on the full net, ~2.5× on the
student - DirectML has less dispatch overhead to erase), so the stacked total is
**1957 → 59 ms = 33× in one day**. 360p at ~17 fps is playable; 480p slow-mo precompute
runs at 10 interp/s.
The `fastest` tier in the web app now points at the student ONNX
(`assets/rife_lite_{720,480,360}p_2blk_noref_student.onnx`; regenerate with
`FRAMECAST_WEIGHTS=<dir with flownet.pkl from models/…safetensors + tools/restore_pkl.py>`
`FRAMECAST_SUFFIX=_student python tools/export_ablation.py <H> <W> 2blk_noref`).

Training gotcha (Windows): the system CUDA dir on PATH poisons torch-cu130's bundled cuDNN
(`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`) - train_student.py strips it in-process.

---

# Step 1 - training-free ablation frontier

Before spending weeks on distillation, measure what can be cut from IFNet_m for free.
Variants exported by `tools/export_ablation.py` (custom inference-only forward over the
original weights - no retraining):

| variant | what's cut |
|---|---|
| `noref` | contextnet + unet refinement (output = warped blend) |
| `s842` | IFBlocks run at scales [8,4,2] instead of [4,2,1] (each block ~4× cheaper) |
| `s842_noref` | both |
| `2blk` | block2 dropped (the most expensive IFBlock), scales [4,2] |
| `2blk_noref` | block2 + refinement dropped |

## Quality (PSNR, real triplets: predict middle frame from neighbors)

Harness: `tools/quality_bench.py`, 12 triplets per clip, step 25.
Clips: Big Buck Bunny 720p (animation, smooth motion) + Jellyfish 720p (live action, slow
organic motion) from test-videos.co.uk. **`demo/test_720p.mp4` and `demo/test_10s.mp4` are
NOT usable as quality benchmarks** - the first has almost no motion (trivial neighbor
averaging beats the model there), the second is static for ~100 frames then has ~10 dB
adjacent-frame chaos.

| model | BBB (dB) | Jellyfish (dB) | Δ vs full |
|---|---:|---:|---|
| full (`rife_lite_inlined`) | **41.50** | **37.67** | - |
| `noref` | 40.24 | 37.86 | −1.3 / **+0.2** |
| `s842` | 41.01 | 37.12 | −0.5 / −0.6 |
| `s842_noref` | 39.92 | 37.10 | −1.6 / −0.6 |
| `2blk` | 40.12 | 36.89 | −1.4 / −0.8 |
| `2blk_noref` | 38.91 | 36.88 | −2.6 / −0.8 |
| *floor: avg-neighbors* | *37.35* | *31.28* | |
| *floor: copy-prev* | *32.54* | *26.90* | |

Notes:
- The refinement (contextnet+unet) is worth ~1.3 dB on animation but ~0 on live action.
- Every variant still clearly beats the trivial floors on both clips.

## Speed - browser webgpu EP (4060 Ti, 736×1280, p50 of 6 runs)

| model | p50 | vs full | PSNR cost (worst clip) |
|---|---:|---:|---|
| full | 1957 ms | 1× | - |
| `noref` | 1319 ms | 1.48× | −1.3 dB |
| `s842` | 2043 ms | **0.96× (slower!)** | −0.6 dB |
| `s842_noref` | 810 ms | 2.4× | −1.6 dB |
| `2blk` | 1397 ms | 1.4× | −1.4 dB |
| `2blk_noref` | **495 ms** | **4.0×** | −2.6 dB |

`s842` being *slower* than full confirms the webgpu EP is dispatch-bound: shrinking tensor
sizes doesn't pay for the extra Resize nodes; only *removing nodes* (noref, 2blk) pays.

## Speed - native TensorRT fp16 (4060 Ti, 736×1280, p50 of 50)

(engines: `tools/build_trt_engine.py`; bench: `rife-trt-bench`)

| model | p50 | fps | vs full | 60fps gate |
|---|---:|---:|---:|---|
| full | 21.6 ms | 46 | 1× | FAIL |
| `noref` | 20.5 ms | 49 | 1.05× | FAIL |
| `2blk_noref` | 11.2 ms | 89 | 1.9× | **PASS** |
| `s842_noref` | **9.8 ms** | **102** | **2.2×** | **PASS** |

**The two runtimes reward opposite cuts.** Browser (dispatch-bound): node count is
everything - `2blk_noref` wins 4×, `s842` is useless. TensorRT (bandwidth/compute-bound):
tensor sizes are everything - `s842_noref` wins 2.2×, `noref` is nearly free (fusion
already hid it). Any distilled student must be checked on both axes.

## Browser: measured stack (no training, no flag)

`2blk_noref` @ 480p on plain **webgpu = 237 ms (4.2 fps)** - 8.3× over the day's starting
point (1957 ms). With WebNN (÷3.5, needs flag): expected **~68 ms (~15 fps)** - to be
confirmed in the user's flagged browser. The `fast`/`fastest` tiers are wired into
demo/slowmo as a quality dropdown.

## Training feasibility (checked 2026-07-02)

GPU training on this machine **works**: venv `E:\venvs\rife-train` with
`torch 2.12.1+cu130` sees the 4060 Ti (`cuda.is_available()=True`, matmul smoke-tested).
The cu130 wheel bundles its own CUDA runtime, so the system CUDA-13.1-vs-prebuilt-binaries
curse does not apply. (pip needs `TMP=E:\tmp` - C: is full.)

## Decision input for distillation (step 2)

- Native already passes 60fps@720p with `s842_noref` (−0.6..−1.6 dB) - for the native
  product, distillation is about *quality recovery* at the fast operating points, and 1080p:
  full 1080p was 52.9 ms → `s842_noref` should land ~24 ms ≈ real-time 1080p (verify).
- For the browser the student should have **few blocks** (node count), for TRT **small
  tensors** (scales); a 2-block student trained at native scale [4,2] satisfies both.
- Concrete step-2 proposal: distill a 2-block student (init from teacher's block0/block1)
  with the full 3-block+refine teacher, target = recover the −2.6 dB of `2blk_noref` to
  within ~−0.5 dB at the same node count. Data: real video triplets (BBB/Jellyfish-style
  clips are enough for a first pass; Vimeo-90K if it stalls).
