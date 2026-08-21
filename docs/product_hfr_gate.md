# Product-path HFR gate

## Status

The gate is implemented and has been run twice on the reference 240Hz system.
Both headed product-path reports are valid RED evidence, so no zero-drop x3/x4
product performance claim can be made yet:

- `product-hfr-2026-07-29T06-08-44-670Z.json`: x3 presented 178.36 FPS
  with 27 drops; x4 presented 231.17 FPS with 42 drops and sustained step-down.
- `product-hfr-2026-07-29T06-12-14-436Z.json`: x3 presented 179.37 FPS
  with 9 drops; x4 presented 236.90 FPS with 81 drops.

Average model compute still fits the source interval on the tested RTX GPU, but
the strict product scheduler/compositor contract is not reproducibly green.
The reports bind the exact source hashes used at capture time; run the gate
again after scheduler changes before treating them as current-release evidence.

Run from the repository root:

```pwsh
node tools\run_product_hfr_gate.mjs
```

The command creates a temporary unpacked extension, launches one persistent headed Playwright Chromium context, and measures V7s at 1280x720 with a deterministic 60fps canvas-backed `<video>`. It runs x3 and x4 sequentially. Each factor gets a 5-second warmup followed by a 30-second measurement. Reports are written once under `output/product-hfr`; a valid red report is preserved and the command exits non-zero.

## Frozen product configuration

- V7s (`rt_v7s`) at the 720p model rung.
- Frame generation enabled with fixed x3 or x4.
- Anime dedup, SR, HDR, compare mode, debug UI, and FPS UI disabled.
- Static guard enabled.
- Conv kernels autotuned before warmup; the applied tune is included in evidence.
- Real extension capture, dedup/cut, `prepPair`, lazy `runT`, display queue, rAF pump, and GPUCanvas blit paths are used.

Ordinary extension behavior is unchanged unless the top-level page is loopback HTTP, carries `framegenProductBench=1`, and supplies a random per-run token. Benchmark commands are accepted only through the token-checked page bridge. Benchmark configuration is not written to the user's normal settings.

## Fail-closed contracts

The pure validator rejects malformed accounting and produces a red result for:

- source producer, rVFC callback, or rVFC `mediaTime` cadence outside the frozen 60fps tolerance;
- observed rAF capacity below the requested 180Hz or 240Hz output;
- source busy-skips, rVFC frame gaps or duplicates, stale dedup classifications, factor step-downs, or skipped pairs;
- any source or mid texture-pool exhaustion;
- scheduler or presentation throughput below contract;
- any product queue drop, excessive pending/high-water depth, or late p95/max above contract;
- extension, page, request, or browser errors;
- missing WebGPU features or missing applied autotune;
- source-file, staged-extension, model, fixture, runner, manifest, or validator hash drift.

The report records raw source/rAF/lateness samples so the validator recomputes rates and percentiles instead of trusting page summaries.

## Exact limitations and blockers

`presented` means a successful call from the real rAF pump into the product `present()` path, ending in a GPUCanvas command submission. Chromium does not expose per-canvas compositor scan-out acknowledgement, so this gate cannot prove that every submitted frame became a physical scan-out. The gate uses observed foreground rAF frequency as the closest browser-visible display-capacity check.

A physical foreground display capable of approximately 240Hz is required for the combined x3/x4 gate. x4 must fail closed on a 60/120/144/165/180Hz display. Occlusion, backgrounding, power throttling, or a compositor cap also makes the run invalid by lowering observed rAF Hz.

Google Chrome and Microsoft Edge removed the command-line flags Playwright needs to sideload an unpacked extension. The automated gate therefore uses Playwright's bundled Chromium as required by the official extension-testing guidance: <https://playwright.dev/docs/next/chrome-extensions>. This is a real Chrome-extension product path, but it is not proof of identical performance in the branded Stable Chrome/Dawn build. Before release, repeat the same fixture manually in foreground Stable Chrome with the unpacked or store-installed extension on the target 240Hz display.

The deterministic canvas stream avoids missing codec fixtures and exercises `copyExternalImageToTexture` from a real `<video>`, but it does not cover codec decode cost, CORS reload behavior, DRM, or site-specific player layout. Those remain separate compatibility checks.
