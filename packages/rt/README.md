# framegen

Real-time neural frame interpolation on **raw WebGPU** - the runtime behind
the [Framegen](https://github.com/MONZikWasTaken/Framegen) extension,
packaged as a library. Hand-written WGSL compute kernels, no ML framework,
~3 ms per generated frame at 720p on a mid-range GPU (RTX 4060 Ti).

Requires a browser with WebGPU and `shader-f16` (Chrome 121+; Apple Silicon
works).

## Install

```
npm i framegen
```

The v7-small weights (2.9 MB) ship inside the package (`weights/`). In a
bundler setup copy them from `node_modules/framegen/weights/`; in the browser
the easiest path is the npm CDN (proper CORS, versioned, cached):

```js
const BASE = 'https://cdn.jsdelivr.net/npm/framegen@1.4.0/weights';
const [bin, manifest] = await Promise.all([
  fetch(`${BASE}/rt_v7s.bin`).then(r => r.arrayBuffer()),
  fetch(`${BASE}/rt_v7s.json`).then(r => r.json()),
]);
```

(GitHub release assets do NOT send CORS headers - fetching them from a page
fails. The CDN route above is the supported one.)

## Interpolate between two frames

```js
import { createRT } from 'framegen';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({
  requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
});

// dimensions must be divisible by 16
const rt = await createRT(device, {
  w: 1280, h: 720,
  weightsBin: bin, weightsManifest: manifest,
  textureInput: true, textureOutput: true,
});

// frameA/frameB are GPUTextures (rgba8unorm, TEXTURE_BINDING);
// out is rgba8unorm with STORAGE_BINDING
rt.prepPair(frameA, frameB);   // t-free trunk, once per pair
rt.runT(0.5, out);             // one mid; call again with any t in (0,1)
```

`prepPair` + `runT` is the real-time path: the trunk runs once per frame
pair, each additional mid costs only the small t-conditioned head - that is
what makes 4x-6x factors affordable.

For one-off use (benchmarks, offline tools) there is also a buffer-mode API:
`rt.run(rgbaA, rgbaB, t)` takes and returns `Uint8Array` RGBA pixels.

### Experimental diagnostics

Exact-GT and artifact harnesses can opt into GPU-resident intermediate outputs.
This path is diagnostic-only and adds no pipelines or dispatches unless
`debugOutputs: true` is passed:

```js
const rt = await createRT(device, { ...opts, debugOutputs: true });
rt.prepPair(frameA, frameB);
rt.runTDebug(0.5, out, { warp0, warp1, flow, mask, refineResidual });
```

The five targets must be full-resolution `rgba16float` textures with
`STORAGE_BINDING`. `flow` stores `(fx0, fy0, fx1, fy1)` in pixels; `mask`
stores `(raw logit, effective compositing mask, max warp disagreement, 1)`;
the effective mask includes `experimentalMaskSharpen` when that A/B is active.
The two warps and the upsampled refine residual use RGB channel order. Add `COPY_SRC` or
`TEXTURE_BINDING` to the target usage when the harness needs readback or
visualization.

The texture-mode final compositor also has a diagnostic-only mask-sharpening
candidate. It is disabled by default and is not enabled by the Framegen
extension:

```js
const rt = await createRT(device, {
  ...opts,
  experimentalMaskSharpen: {
    strength: 4,
    disagreementLow: 0.02,
    disagreementHigh: 0.2,
  },
});
```

This option is for controlled exact-GT A/B tests, not a recommended quality
preset. Sharpening can reduce double images while increasing missing regions,
centroid error, popping, or temporal flicker. Keep the default `null` unless a
candidate passes scene, temporal, and performance gates.

The exact-GT page also exposes a deterministic primitive fixture suite through
`?fixture=`:

- `collision` (default) preserves the public-demo baseline;
- `crossing` exercises identity and fixed depth order;
- `merge_zoom` exercises changing scale and partial occlusion;
- `rotate_thin` stresses rotating sub-/8-resolution boundaries;
- `acceleration` and `bounce` are diagnostic stress cases whose exact internal
  motion is not observable from two anchors alone.

Every fixture runs the same 23 `k/24` positions and publishes its metadata,
focus frame, quality ROI, geometry/tracking aggregates, temporal residual,
flicker excess, motion deficit, and GT-vs-GT sanity result through
`window.__collisionResult`. Unknown fixture names fail explicitly. Do not use
the unobservable-motion fixtures as pairwise-model release blockers; they are
evidence for training-data or temporal-context decisions.

## Squeeze the last 20%

Kernel shapes are GPU-specific. Run the autotuner once per machine and pass
the result in:

```js
import { createRT, tuneConvRB } from 'framegen';

const tune = await tuneConvRB(device, { ci: 192, co: 192, w16: 80, h16: 45 });
localStorage.setItem('fcTune', JSON.stringify(tune));
// ...next session:
const rt = await createRT(device, { ...opts, convTune: JSON.parse(localStorage.getItem('fcTune')) });
```

## What's new in 1.1.0

- **Direct-warp flowout**: the warp samples your source textures through the
  hardware bilinear unit instead of an internal full-res copy - less VRAM,
  less bandwidth, mids resampled once instead of twice (slightly sharper).
- **Occlusion-sparse refine** (tfact2 weights): the refine chain runs only on
  tiles where the two warps disagree, scheduled entirely on the GPU via
  indirect dispatch. Bit-identical output on full-motion frames, up to ~4x
  cheaper refine on calm content. On by default; `sparseRefine: false`
  restores the dense path, `refineThr` tunes the sensitivity (default 0.02).
- **`rt.profileT(a, b, t, outTex)`**: per-stage GPU timings for the texture
  path (needs `timestamp-query` on the device).
- `tuneConvRB` explores more workgroup shapes; pass its result as `convTune`
  exactly as before.

## Example project

A complete working integration - synthetic WebGPU scene boosted in real time,
raw-vs-boosted split, naive-blend comparison, honest GPU timing via
timestamp queries:
live at https://monzikwastaken.github.io/framegen-fps-booster/, source at
https://github.com/MONZikWasTaken/framegen-fps-booster.

## License

**MIT.** Embed it in anything, including commercial products - no strings on
the code. The model weights bundled in this package are licensed separately
(non-commercial - see
[WEIGHTS_LICENSE](https://github.com/MONZikWasTaken/Framegen/blob/main/WEIGHTS_LICENSE.md));
for commercial weight licensing, get in touch.
