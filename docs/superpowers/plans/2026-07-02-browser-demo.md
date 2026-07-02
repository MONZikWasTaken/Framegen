# Browser demo (two images → middle frame) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser page where you drop two images (or click "demo pair") and get the RIFE-interpolated middle frame via ort-web WebGPU, with a before/after slider.

**Architecture:** A self-contained `web/rife_prepost.js` ES module ports `rife-core::prepost` (BGR / ÷255 / pad-to-32 / crop) to JS over RGBA8; node-run unit tests guard parity. `web/demo.html` orchestrates: image → offscreen canvas → `toInput` → `ort.Tensor` → `session.run` → `fromOutput` → result canvas + slider.

**Tech Stack:** ONNX Runtime Web (WebGPU EP, wasm fallback), vanilla JS ES modules, node 22 (test runner), Python http.server + Playwright (E2E verify).

## Global Constraints

- Color/normalization semantics FIXED, identical to `crates/rife-core/src/prepost.rs`: channel order **BGR**, scale **÷255**, zero-pad **bottom/right**, crop **top-left**, output truncation (`(v*255) clamp[0,255] → int`). Bit-parity with rife-core is the bar.
- ONNX model `assets/rife_lite_inlined.onnx` has fixed input `[1,3,736,1280]`; input names `img0`,`img1`; output `mid`. So the pad target is always engine size **pw=1280, ph=736**; `w,h` are the native image size (≤1280×720).
- Browser pixels are **RGBA8** (4 bytes/px); `toInput` reads R,G,B and ignores A; `fromOutput` writes A=255.
- No build step, no npm deps for the demo itself (ort-web from CDN). `web/package.json` exists only so node treats `web/*.js` as ES modules for tests.
- Commit after each task. Branch `phase1-browser-demo` (already checked out).

---

### Task 1: `web/rife_prepost.js` — JS pre/post port + node parity tests

**Files:**
- Create: `web/package.json`
- Create: `web/rife_prepost.js`
- Create: `web/prepost.test.js`

**Interfaces:**
- Produces (ES module exports):
  - `pad32(x: number) → number` — round up to /32.
  - `toInput(rgba: Uint8ClampedArray|Uint8Array, w, h, pw, ph) → Float32Array` — length `3*pw*ph`, CHW, BGR, ÷255, zero-pad bottom/right.
  - `fromOutput(chw: Float32Array, w, h, pw, ph) → Uint8ClampedArray` — length `4*w*h` RGBA, crop top-left, BGR→RGB, clamp*255 trunc, A=255.

- [ ] **Step 1: Create `web/package.json`** (makes node load `web/*.js` as ESM):

```json
{ "name": "framecast-web", "private": true, "type": "module" }
```

- [ ] **Step 2: Write the failing tests** in `web/prepost.test.js`:

```js
import { pad32, toInput, fromOutput } from './rife_prepost.js';
import assert from 'node:assert/strict';

// pad32
assert.equal(pad32(720), 736);
assert.equal(pad32(1280), 1280);
assert.equal(pad32(1), 32);
assert.equal(pad32(32), 32);
assert.equal(pad32(33), 64);

// toInput: one 2x1 RGBA image, pad to 4x2. Mirrors rife-core to_input_bgr_and_pad.
// pixel0 R=10 G=20 B=30, pixel1 R=40 G=50 B=60 (alpha ignored)
{
  const rgba = new Uint8ClampedArray([10,20,30,255, 40,50,60,255]);
  const [w,h,pw,ph] = [2,1,4,2];
  const out = toInput(rgba, w, h, pw, ph);
  const plane = pw*ph; // 8
  const near = (a,b)=>Math.abs(a-b)<1e-6;
  assert.ok(near(out[0], 30/255), 'B0');       // B plane
  assert.ok(near(out[1], 60/255), 'B1');
  assert.equal(out[2], 0); assert.equal(out[3], 0);        // pad
  assert.ok(near(out[plane], 20/255), 'G0');   // G plane
  assert.ok(near(out[plane+1], 50/255), 'G1');
  assert.ok(near(out[2*plane], 10/255), 'R0'); // R plane
  assert.ok(near(out[2*plane+1], 40/255), 'R1');
  assert.equal(out[4], 0, 'padded row1');
}

// round-trip crop: toInput then fromOutput recovers the RGB (alpha=255).
{
  const rgba = new Uint8ClampedArray([10,20,30,255, 40,50,60,255, 70,80,90,255, 100,110,120,255]); // 2x2
  const [w,h,pw,ph] = [2,2,32,32];
  const chw = toInput(rgba, w, h, pw, ph);
  const back = fromOutput(chw, w, h, pw, ph);
  assert.deepEqual(Array.from(back), [10,20,30,255, 40,50,60,255, 70,80,90,255, 100,110,120,255]);
}

// fromOutput clamp/truncate: one pixel BGR [2.0,-1.0,0.5] -> R=127 G=0 B=255 A=255
{
  const chw = new Float32Array([2.0, -1.0, 0.5]);
  const out = fromOutput(chw, 1, 1, 1, 1);
  assert.deepEqual(Array.from(out), [127, 0, 255, 255]);
}

console.log('rife_prepost.js: ALL TESTS PASSED');
```

- [ ] **Step 3: Run the tests, verify they fail** (module not written yet):

Run: `node web/prepost.test.js`
Expected: FAIL — `Cannot find module` or import error.

- [ ] **Step 4: Implement `web/rife_prepost.js`:**

```js
// JS port of crates/rife-core/src/prepost.rs. Input pixels are RGBA8 (canvas);
// model buffer is CHW f32, BGR order, ÷255, zero-padded bottom/right.

export function pad32(x) {
  return Math.ceil(x / 32) * 32;
}

// RGBA8 HWC (w*h*4) -> CHW f32 (BGR, ÷255), zero-padded bottom/right to pw x ph.
export function toInput(rgba, w, h, pw, ph) {
  const dst = new Float32Array(3 * pw * ph); // zero-filled
  const plane = ph * pw; // row stride is pw
  for (let y = 0; y < h; y++) {
    const row = y * pw;
    const srow = y * w * 4;
    for (let x = 0; x < w; x++) {
      const s = srow + x * 4;
      const o = row + x;
      dst[o] = rgba[s + 2] / 255;             // B
      dst[plane + o] = rgba[s + 1] / 255;     // G
      dst[2 * plane + o] = rgba[s] / 255;     // R
    }
  }
  return dst;
}

// CHW f32 (BGR, pw x ph) -> RGBA8 HWC (w*h*4), crop top-left, BGR->RGB, clamp*255 trunc, A=255.
export function fromOutput(chw, w, h, pw, ph) {
  const dst = new Uint8ClampedArray(4 * w * h);
  const plane = ph * pw;
  const px = (v) => Math.min(255, Math.max(0, v * 255)) | 0; // clamp then truncate
  for (let y = 0; y < h; y++) {
    const row = y * pw;
    const drow = y * w * 4;
    for (let x = 0; x < w; x++) {
      const o = row + x;
      const d = drow + x * 4;
      dst[d] = px(chw[2 * plane + o]);   // R
      dst[d + 1] = px(chw[plane + o]);   // G
      dst[d + 2] = px(chw[o]);           // B
      dst[d + 3] = 255;                  // A
    }
  }
  return dst;
}
```

Note: `| 0` truncates toward zero for non-negative values (matches Rust `as u8` after clamp≥0).

- [ ] **Step 5: Run the tests, verify they pass:**

Run: `node web/prepost.test.js`
Expected: `rife_prepost.js: ALL TESTS PASSED`, exit 0.

- [ ] **Step 6: Commit:**

```bash
git add web/package.json web/rife_prepost.js web/prepost.test.js
git commit -m "feat(web): JS pre/post port of rife-core prepost + node parity tests"
```

---

### Task 2: `web/demo.html` — UI + orchestration

**Files:**
- Create: `web/demo.html`

**Interfaces:**
- Consumes: `./rife_prepost.js` (`toInput`, `fromOutput`); ort-web from CDN; `assets/rife_lite_inlined.onnx`; `demo/I0_0.png`, `demo/I0_1.png`.

- [ ] **Step 1: Create `web/demo.html`:**

```html
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Framecast — browser demo</title>
<style>
  body { font: 14px/1.5 system-ui, sans-serif; margin: 20px; max-width: 1000px; }
  button { font-size: 15px; padding: 6px 14px; margin-right: 8px; }
  .row { display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; margin:12px 0; }
  .card { border:1px solid #ccc; border-radius:8px; padding:8px; }
  canvas, img.thumb { max-width:320px; height:auto; display:block; background:#eee; }
  #status { font:13px monospace; color:#333; white-space:pre-wrap; margin-top:8px; }
  #baWrap { position:relative; max-width:640px; }
  #baWrap canvas { position:absolute; top:0; left:0; max-width:640px; }
  #baAfter { clip-path: inset(0 0 0 50%); }
  #baRange { width:640px; }
  .bad { color:#c00; }
</style>
</head>
<body>
<h3>Framecast — RIFE interpolation in the browser (ort-web WebGPU)</h3>
<p>Drop two frames or use the built-in demo pair. You get the interpolated middle frame.</p>

<div class="row">
  <div class="card"><b>img0</b><br><canvas id="c0" width="320" height="180"></canvas>
    <input type="file" id="f0" accept="image/*"></div>
  <div class="card"><b>img1</b><br><canvas id="c1" width="320" height="180"></canvas>
    <input type="file" id="f1" accept="image/*"></div>
</div>

<div>
  <select id="ep"><option value="webgpu">webgpu</option><option value="wasm">wasm</option></select>
  <button id="demoBtn">Use demo pair</button>
  <button id="runBtn">Interpolate</button>
</div>
<div id="status"></div>

<h4>Result — middle frame (before/after slider)</h4>
<div id="baWrap">
  <canvas id="baBefore"></canvas>
  <canvas id="baAfter"></canvas>
</div>
<input type="range" id="baRange" min="0" max="100" value="50">

<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.webgpu.min.js"></script>
<script type="module">
import { toInput, fromOutput } from './rife_prepost.js';

const EW = 1280, EH = 736;         // engine (padded) size
const MAXW = 1280, MAXH = 720;     // max native size we feed
const st = (m, bad) => { const s=document.getElementById('status');
  s.textContent += m + '\n'; if (bad) s.classList.add('bad'); };

// Loaded native ImageData for each input (RGBA at native w×h ≤ MAXW×MAXH).
const frame = [null, null];

function drawNative(imgOrCanvasSource, canvasEl) {
  // Scale to fit within MAXW×MAXH preserving aspect; draw into an offscreen canvas
  // at native size; also draw a thumbnail into the visible canvasEl.
  const iw = imgOrCanvasSource.width, ih = imgOrCanvasSource.height;
  const s = Math.min(1, MAXW / iw, MAXH / ih);
  const w = Math.max(1, Math.round(iw * s)), h = Math.max(1, Math.round(ih * s));
  const off = new OffscreenCanvas(w, h);
  const octx = off.getContext('2d');
  octx.drawImage(imgOrCanvasSource, 0, 0, w, h);
  // thumbnail
  canvasEl.width = 320; canvasEl.height = Math.round(320 * h / w);
  canvasEl.getContext('2d').drawImage(off, 0, 0, canvasEl.width, canvasEl.height);
  return octx.getImageData(0, 0, w, h); // {data:Uint8ClampedArray, width, height}
}

function loadFileToFrame(file, idx, canvasEl) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => { frame[idx] = drawNative(img, canvasEl); URL.revokeObjectURL(img.src); res(); };
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });
}
function loadUrlToFrame(url, idx, canvasEl) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => { frame[idx] = drawNative(img, canvasEl); res(); };
    img.onerror = rej;
    img.src = url;
  });
}

document.getElementById('f0').onchange = e =>
  loadFileToFrame(e.target.files[0], 0, document.getElementById('c0'));
document.getElementById('f1').onchange = e =>
  loadFileToFrame(e.target.files[0], 1, document.getElementById('c1'));
document.getElementById('demoBtn').onclick = async () => {
  await Promise.all([
    loadUrlToFrame('/demo/I0_0.png', 0, document.getElementById('c0')),
    loadUrlToFrame('/demo/I0_1.png', 1, document.getElementById('c1')),
  ]);
  st('demo pair loaded ('+frame[0].width+'x'+frame[0].height+')');
};

let sess = null, sessEp = null;
async function ensureSession(ep) {
  if (sess && sessEp === ep) return sess;
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  st('creating session on ' + ep + ' ...');
  sess = await ort.InferenceSession.create('/assets/rife_lite_inlined.onnx',
    { executionProviders: [ep], graphOptimizationLevel: 'all' });
  sessEp = ep;
  st('session ready: inputs=' + sess.inputNames + ' output=' + sess.outputNames);
  return sess;
}

document.getElementById('runBtn').onclick = async () => {
  try {
    if (!frame[0] || !frame[1]) { st('load two images first (or click "Use demo pair")', true); return; }
    if (frame[0].width !== frame[1].width || frame[0].height !== frame[1].height) {
      st('the two frames must be the same size', true); return;
    }
    const ep = document.getElementById('ep').value;
    const s = await ensureSession(ep);
    const w = frame[0].width, h = frame[0].height;
    const a = toInput(frame[0].data, w, h, EW, EH);
    const b = toInput(frame[1].data, w, h, EW, EH);
    const feeds = {};
    feeds[s.inputNames[0]] = new ort.Tensor('float32', a, [1, 3, EH, EW]);
    feeds[s.inputNames[1]] = new ort.Tensor('float32', b, [1, 3, EH, EW]);
    const t0 = performance.now();
    const out = await s.run(feeds);
    const dt = performance.now() - t0;
    const chw = out[s.outputNames[0]].data; // Float32Array, CHW BGR, EH×EW
    const rgba = fromOutput(chw, w, h, EW, EH);
    renderResult(frame[0], new ImageData(rgba, w, h));
    st(`interpolated ${w}x${h}: ${dt.toFixed(1)} ms · ep=${ep}`);
  } catch (e) { st('ERROR: ' + (e.stack || e.message || e), true); }
};

function renderResult(beforeImageData, afterImageData) {
  const w = afterImageData.width, h = afterImageData.height;
  for (const id of ['baBefore', 'baAfter']) {
    const c = document.getElementById(id); c.width = w; c.height = h;
    c.style.maxWidth = '640px'; c.style.height = 'auto';
  }
  document.getElementById('baBefore').getContext('2d').putImageData(beforeImageData, 0, 0);
  document.getElementById('baAfter').getContext('2d').putImageData(afterImageData, 0, 0);
  const wrap = document.getElementById('baWrap');
  wrap.style.height = (640 * h / w) + 'px';
  document.getElementById('baRange').oninput = e => {
    document.getElementById('baAfter').style.clipPath = `inset(0 0 0 ${e.target.value}%)`;
  };
}
</script>
</body>
</html>
```

Note: each file input takes its single selected file and maps to its own frame index (f0→0, f1→1). Verify in Step 2 that "Use demo pair" populates both thumbnails.

- [ ] **Step 2: Manual smoke (mechanics)** — serve and open, confirm no JS console errors on load and the demo pair draws thumbnails:

```pwsh
# from repo root
python -m http.server 8123 --bind 127.0.0.1
```
Open `http://127.0.0.1:8123/web/demo.html`, click "Use demo pair" → two thumbnails appear, status shows "demo pair loaded (448x256)". (WebGPU run is verified in Task 3.)

- [ ] **Step 3: Commit:**

```bash
git add web/demo.html
git commit -m "feat(web): demo page — two images to interpolated middle frame + slider"
```

---

### Task 3: E2E validation on wasm (autonomous) + hand-off note

**Files:**
- Create: `web/README.md`

**Interfaces:**
- Consumes: everything from Tasks 1–2.

The WebGPU perf is the user's job (their GPU). What we CAN verify autonomously: the whole
pipeline produces a correct (non-broken) middle frame on the wasm EP, using Playwright.

- [ ] **Step 1: Serve the repo root** (background):

```pwsh
python -m http.server 8123 --bind 127.0.0.1
```

- [ ] **Step 2: Drive the demo on wasm via Playwright.** Navigate to
`http://127.0.0.1:8123/web/demo.html`, then evaluate: set `ep=wasm`, click "Use demo pair",
wait for "demo pair loaded", click "Interpolate", wait until status contains "interpolated"
or "ERROR". Read back the `baAfter` canvas pixel stats:

```js
async () => {
  document.getElementById('ep').value = 'wasm';
  document.getElementById('demoBtn').click();
  const st = document.getElementById('status');
  const wait = async (re, ms) => { const t=Date.now();
    while (Date.now()-t<ms){ if(re.test(st.textContent)) return true; await new Promise(r=>setTimeout(r,300)); } return false; };
  await wait(/demo pair loaded/, 15000);
  document.getElementById('runBtn').click();
  await wait(/interpolated|ERROR/, 180000);
  const c = document.getElementById('baAfter');
  const d = c.getContext('2d').getImageData(0,0,c.width,c.height).data;
  let mn=255,mx=0,s=0,nz=0;
  for (let i=0;i<d.length;i+=4){ const v=d[i]; s+=v; if(v<mn)mn=v; if(v>mx)mx=v; if(v!==0)nz++; }
  const n=d.length/4;
  return { status: st.textContent.split('\n').slice(-3).join(' | '),
           size:[c.width,c.height], meanR:(s/n).toFixed(1), min:mn, max:mx, nonzeroPx:nz };
}
```
Expected: `size:[448,256]`, `max` well above 0, `nonzeroPx` ≈ full frame (output not black), status
contains "interpolated 448x256". If `max===0` / all-zero → pipeline broken, debug before commit.

- [ ] **Step 3: Sanity vs reference (optional but do it).** The candle/PyTorch middle frame for
this pair is `demo/mid_pytorch.png` (448×256). ort-web fp32 should be visually close. If a quick
mean-abs-diff is desired, compare the `baAfter` pixels to that PNG decoded on the page; expect a
small mean diff (different runtimes, not bit-exact). Not a hard gate — the non-broken check in
Step 2 is the gate.

- [ ] **Step 4: Write `web/README.md`** (how to run + hand-off):

```markdown
# Framecast web

Browser demo + probes for the ort-web (WebGPU) path.

## Run
```
python -m http.server 8123 --bind 127.0.0.1
```
- `web/demo.html` — drop two frames (or "Use demo pair") → interpolated middle frame + slider.
- `web/probe.html` — ceiling probe: p50/p10 fps, mirrored ort-web console (node placement +
  per-kernel WebGPU profiling), fp32/fp16 × webgpu/wasm.

## Pre/post
`web/rife_prepost.js` mirrors `crates/rife-core/src/prepost.rs` (BGR, ÷255, pad-to-32, crop).
Unit tests: `node web/prepost.test.js`.

## Notes
- Model `assets/rife_lite_inlined.onnx` is fixed-shape [1,3,736,1280]; smaller frames are
  zero-padded to that and cropped back (native ≤ engine), same as the native TensorRT path.
- Real WebGPU speed must be measured in a browser with a real GPU adapter (Playwright's
  chromium has `navigator.gpu === false`).
```

- [ ] **Step 5: Commit:**

```bash
git add web/README.md
git commit -m "docs(web): README + E2E-verified browser demo"
```

---

## Post-plan verification (whole feature)

- [ ] `node web/prepost.test.js` → ALL TESTS PASSED.
- [ ] Playwright wasm run of `web/demo.html` on the demo pair → non-broken 448×256 middle frame.
- [ ] `web/demo.html` loads with no console errors; "Use demo pair" draws thumbnails.
- [ ] Update memory `browser-webgpu-spike.md` / `intermodule-direction.md`: Phase 1 demo done, JS pre/post ported (option A).

## Self-review notes

- Spec files (`web/demo.html`, `web/rife_prepost.js`, tests) → Tasks 1–3. `prepost.test.html`
  from the spec is realized as node `web/prepost.test.js` (more automatable; same cases).
- Pad target pw=1280/ph=736 and RGBA handling consistent across toInput/fromOutput and demo.
- Option A (hand JS port) implemented; option B (wasm rife-core) explicitly deferred.
- Type names consistent: `toInput(rgba,w,h,pw,ph)`, `fromOutput(chw,w,h,pw,ph)`, `pad32`.
