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

Weights are **not** bundled - fetch them once from the Framegen release
(2.9 MB) and cache as you like:

```js
const BASE = 'https://github.com/MONZikWasTaken/Framegen/releases/download/v1.0.0';
const [bin, manifest] = await Promise.all([
  fetch(`${BASE}/rt_v7s.bin`).then(r => r.arrayBuffer()),
  fetch(`${BASE}/rt_v7s.json`).then(r => r.json()),
]);
```

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

## License

**LGPL-3.0-or-later.** Embed it in anything, including commercial products;
modifications to the library itself must be published. The model weights are
licensed separately (non-commercial - see
[WEIGHTS_LICENSE](https://github.com/MONZikWasTaken/Framegen/blob/main/WEIGHTS_LICENSE.md));
for commercial weight licensing, get in touch.
