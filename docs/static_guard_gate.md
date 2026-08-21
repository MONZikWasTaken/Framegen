# Static Guard correctness gate

## Status

The Framegen 1.4.1 release candidate passed the headed WebGPU gate on
2026-08-18 in Chrome 151. All 17 frozen checks passed with clean browser
diagnostics and stable source hashes: guard-off parity had zero mismatched
values, text-candidate MAE was `0.286` (maximum `0.5`), and ordinary-motion
MAE was `0`.

Run from the repository root:

```pwsh
node tools\run_static_guard_gate.mjs
```

The command starts a loopback-only static server, uses pinned `@playwright/cli@0.1.17` to open the installed system Chrome channel in headed mode (`--browser chrome`), waits for `window.__STATIC_GUARD_PROMISE__`, and writes one JSON report under `output/static-guard`. It never installs or downloads a browser and has no fallback channel. A valid red report is preserved and the process exits non-zero. `--output` accepts only a unique `.json` path inside that directory; existing evidence is never overwritten.

## Frozen evidence

The report records start/end SHA-256 hashes and sizes for:

- `web/static_guard_fixture.html`;
- `web/rt/rt.js`;
- `extension/assets/rt_v7s.bin`;
- `extension/assets/rt_v7s.json`;
- `tools/run_static_guard_gate.mjs`;
- `tools/static_guard_acceptance.mjs`.

The runtime, weights, and model-manifest hashes emitted by the page must also match the runner's source hashes. Any source drift during execution fails the gate.

## Fail-closed contract

The deterministic 512x288 fixture renders a pixel-defined `FRAMEGEN` label over a moving checker pattern at seven uniformly spaced timesteps. The Node validator independently recomputes the page report and requires:

- byte-identical output when `staticGuard` is omitted versus explicitly disabled;
- agreement with the time-consistent `mix(A, B, t)` candidate in static text cores;
- continuous anchor transitions, temporal linearity, monotonic text values, and retained text contrast;
- no material change relative to guard-off output in high-motion regions;
- no console errors, page errors, failed requests, HTTP errors, fixture errors, identity mismatches, or source drift.

Console warnings and the complete request list are retained for diagnosis but do not independently fail the gate.

## Scope

This is a correctness gate, not a latency benchmark. It does not measure frame-generation throughput, presentation cadence, display scan-out, decode cost, arbitrary subtitle fonts, compression patterns, or V8 training quality. Those remain covered by separate product HFR, quality, and training gates.
