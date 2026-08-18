# Product target-FPS gate

## Status

The schema-v2 gate, deterministic fixture, runner, and CPU validator are
implemented. The validator covers arbitrary numeric targets rather than fixed
60/120 FPS presets. Its unit suite is green.

A green CPU validation result proves that report structure and fail-closed
acceptance rules are coherent. It does not replace the authoritative headed
Chrome/WebGPU run. Do not claim product acceptance until a fresh report from the
real unpacked extension has `passed: true`.

## Authoritative case matrix

All cases use the V7s runtime at 480p internal resolution and a deterministic
1280x720 moving-primitives source:

| Case | Source | Requested target | Expected effective target | Expected clamp |
| --- | ---: | ---: | ---: | --- |
| `source10-target50` | 10 FPS | 50 FPS | 50 FPS | none |
| `source15-target50` | 15 FPS | 50 FPS | 50 FPS | none |
| `source24-target50` | 24 FPS | 50 FPS | 50 FPS | none |
| `source24-request40-floor48` | 24 FPS | 40 FPS | 48 FPS | minimum 2x |
| `source60-target120` | 60 FPS | 120 FPS | 120 FPS | none |
| `source60-target144` | 60 FPS | 144 FPS | 144 FPS | none |
| `source60-request300-display-cap` | 60 FPS | 300 FPS | ~232.8 FPS | display |

The first three cases prove that `factor=target` accepts non-preset output rates.
The 24→40 case proves the product floor: interpolation requests below 2x source
cadence resolve to 2x. The 60→300 case proves that an unsupported request is
reported as a display clamp instead of being presented as achieved. At the
240 Hz ceiling the scheduler deliberately targets about 97% of measured display
service (`232.8` FPS nominally), reserving bounded recovery headroom after an
rAF hitch.

## Authoritative product run

Run from the repository root. If Playwright browsers live outside the default
cache, point `PLAYWRIGHT_BROWSERS_PATH` at that local directory first:

```pwsh
$env:PLAYWRIGHT_BROWSERS_PATH='<path-to-playwright-browsers>' # optional
node tools\run_product_target_fps_gate.mjs
```

The runner stages a temporary unpacked extension and opens one persistent,
headed Chrome-for-Testing context. Every case gets four seconds of warmup and
twelve seconds of measurement. Evidence is written once under
`output/product-target-fps`; both valid red reports and green reports are
preserved.

The product bridge must report the requested target, the measured source floor,
the display capacity, the effective target, and any clamp reason. The validator
also requires the expected source/generated-mid split; duplicating source
anchors cannot satisfy the gate.

## Acceptance contract

Each case requires:

- source producer, `requestVideoFrameCallback`, and media cadence close to the
  declared 10, 15, 24, or 60 FPS source;
- scheduled and presented output close to the effective target;
- source anchors close to source cadence and generated mids close to
  `effective target - source cadence`;
- a factor histogram whose planned mids agree with the scheduled mid work;
- a foreground rAF cadence close to 240 Hz;
- zero queue drops, pool exhaustion, source gaps, duplicate classifications,
  superseded classifications, extension errors, and browser errors;
- bounded pending/high-water depth and presentation lateness;
- the required WebGPU features on a non-integrated GPU.

The report binds the Git state and start/end SHA-256 identities of the content
script, cadence helper, extension manifest, fixture, runtime, weights, runner,
gate manifest, and validator. The staged extension copies are hashed
independently.

## CPU validation

```pwsh
node --test tools\product_target_fps_acceptance.test.mjs
```

The suite includes a complete green schema-v2 case matrix plus deterministic
red mutations for source duplication, incorrect factor provenance, target and
display cadence drift, source delivery drift, drops, queue overflow, texture
pool exhaustion, browser errors, source/hash drift, malformed accounting, and
non-write-once runner behavior.

## Limitation

`presented` means a successful submission through the product rAF/GPUCanvas
path. Chromium does not expose per-canvas physical scan-out acknowledgement, so
the gate proves submitted product cadence rather than literal monitor scan-out
of every canvas frame.
