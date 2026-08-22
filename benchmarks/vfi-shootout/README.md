# VFI shootout

This is a midpoint image-quality shootout for the direct Framegen v7s WebGPU
runtime. It is not an extension product gate: it does not exercise Sharpness,
the presentation canvas, cadence, dropped frames, VRAM use, or temporal
stability.

## Inputs

Obtain Middlebury and SNU-FILM from their official providers and place only the
licensed local evaluation files at the paths in scenes.json. Dataset files and
generated results live under .bench and are intentionally not committed. Every
run records the SHA-256 and size of each input, ground-truth image, runtime,
weight file, manifest, harness, and runner.

The PyTorch reference runner imports code from the repositories passed through
--rife-root and --ifrnet-root. Those imports execute third-party Python. Use
only repositories and checkpoints that you trust. Both repositories must be
clean Git checkouts with readable commits. The runner rejects dirty or non-Git
sources, verifies the exact imported model files, and records their hashes plus
the hashes of every imported Python source under each repository and the exact
checkpoint hashes. This proves local identity; it does not certify
that a repository or checkpoint is an official release.

## WebGPU runs

The launcher uses the pinned Playwright package, installed-Chrome channel, and
headed mode from tools/webgpu_bench_manifest.json. `npx` may fetch that exact
package version on first use. The launcher has no hard-coded host paths, binds
its allowlisted file server only to 127.0.0.1, and removes its temporary browser
session even after launch or page errors. The host still needs the contract's
Chrome channel with WebGPU enabled.

    node tools/run_webgpu_shootout.mjs --res 480 --repetitions 5
    node tools/run_webgpu_shootout.mjs --res 720 --repetitions 5

Each command prints its unique output directory. A raw timing value is one
page-reported median of five synchronized prepPair + runT(0.5) calls. The runner
retains all repeated page medians and reports median, p95, standard deviation,
and coefficient of variation. It deliberately marks a single-revision run as
insufficient for a comparative performance claim. Adapter identity, enabled
WebGPU features, relevant device limits, browser version, and host identity are
recorded and must remain stable across all scenes. Browser APIs may not expose
the installed GPU driver version, which remains a recorded limitation.

## Reference run

    python tools/run_vfi_references.py --rife-root <trusted-practical-rife-repository> --rife-checkpoint <trusted-rife-checkpoint> --ifrnet-root <trusted-ifrnet-repository> --ifrnet-checkpoint <trusted-ifrnet-checkpoint>

Practical-RIFE defaults to CPU and IFRNet defaults to the best available
PyTorch device. Their raw timings are descriptive per method only. Do not rank
their throughput against each other or against WebGPU when devices, frameworks,
precision, or preprocessing differ. Explicit --rife-device and
--ifrnet-device options are available when equivalent local conditions are
possible.

The Practical-RIFE checkpoint must be exactly
`<rife-root>/train_log/flownet.pkl`. Inputs are converted from decoded RGB to
the canonical Practical-RIFE BGR tensor order, and its output is converted back
to RGB for scoring. IFRNet uses the published SNU-FILM evaluation preprocessing:
RGB tensors, divisor-20 replicate padding, and `scale_factor=0.8`. These
contracts are pinned in manifest.json and copied into every reference run.

## Quality report

Pass the three exact run directories printed by the launchers:

    python tools/report_vfi_shootout.py --framegen-480 <framegen-480-run> --framegen-720 <framegen-720-run> --reference <reference-run>

The reporter refuses missing, modified, or dataset-mismatched artifacts. It
also refuses to combine 480p and 720p Framegen runs unless their runtime,
weights, harness, runner, manifest, browser, host, adapter, enabled features,
and device limits match exactly. It reports RGB PSNR, an explicitly named local
11x11 box-window luma SSIM-like
score, and luma gradient-magnitude MAE. It does not combine or rank timing data
and does not claim a quality pass without an explicit baseline and acceptance
thresholds. Exact image matches use a finite PSNR cap of 99 dB.

## Non-GPU checks

    node --test tools/run_webgpu_shootout.test.mjs
    python -m unittest tools/test_vfi_shootout_tools.py
