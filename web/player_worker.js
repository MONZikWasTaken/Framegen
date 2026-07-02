// Framecast player worker: the whole pixel path lives OFF the main thread.
// Receives display-size ImageBitmaps, scales to the active model size, dedups anime
// "twos", runs the model, posts mids back as transferable buffers.
//
// AUTO mode: a quality ladder for the mids (originals never change). The controller
// watches the real inference cost vs the real scene budget (time between unique
// transitions) and walks the ladder: instantly down when over budget, up after a
// stable streak. Sessions are cached; new rungs build in the background.
import * as ortNS from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs';
import { createRT } from './rt/rt.js';
const ort = ortNS.default ?? ortNS;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const SIZE = { 360: [640, 360], 480: [854, 480], 720: [1280, 720] };
// custom-runtime sizes must be /16-divisible (852->848, 360->352, 1080->1072 — sub-1% stretch)
const SIZE_RT = { 352: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1072] };
// mids ladder, best quality first; est = initial ms guess (learned at runtime).
// rt = our own WebGPU runtime (1blk model): bit-exact, no flags, ~3-6x faster than ort.
const LADDER = [
  { key: 'rt@1080',     kind: 'rt',  q: 'turbo',   res: 1080, est: 24 },
  { key: 'fastest@720', kind: 'f32', q: 'fastest', res: 720, est: 150 },
  { key: 'rt@720',      kind: 'rt',  q: 'turbo',   res: 720, est: 11 },
  { key: 'fastest@480', kind: 'f32', q: 'fastest', res: 480, est: 60 },
  { key: 'rt@480',      kind: 'rt',  q: 'turbo',   res: 480, est: 5 },
  { key: 'rt@352',      kind: 'rt',  q: 'turbo',   res: 352, est: 3 },
];
const F32 = {
  turbo:   { 360: { url: '/assets/rife_lite_360p_1blk_s4_student1b.onnx', ew: 640, eh: 384 },
             480: { url: '/assets/rife_lite_480p_1blk_s4_student1b.onnx', ew: 864, eh: 480 },
             720: { url: '/assets/rife_lite_720p_1blk_s4_student1b.onnx', ew: 1280, eh: 736 } },
  fastest: { 360: { url: '/assets/rife_lite_360p_2blk_noref_student.onnx', ew: 640, eh: 384 },
             480: { url: '/assets/rife_lite_480p_2blk_noref_student.onnx', ew: 864, eh: 480 },
             720: { url: '/assets/rife_lite_720p_2blk_noref_student.onnx', ew: 1280, eh: 736 } },
};

// ort-web serializes run() GLOBALLY (even across sessions) — every run, including
// warmups of background-built rungs, must go through this one lock.
let gpuLock = Promise.resolve();
function withGpu(fn) {
  const p = gpuLock.then(fn);
  gpuLock = p.catch(() => {});
  return p;
}

let ep = 'webnn', auto = false;
let sessions = new Map(); // key -> {sess, kind:'u8'|'f32', W, H, ew, eh, ms}
let activeKey = null, buildingKey = null;
let canvases = new Map(); // "WxH" -> {off, ctx}
let last = null, lastKey = null, lastTs = 0, lastUniqueTs = 0, transitionNo = 0;
let busy = false, pending = null;
let uniqueIntervalMs = 42, goodSince = 0, halfRate = false;
let animeMode = true, interpOn = true, factor = 2, maxRes = 720;
// mids above the display resolution are wasted work — the ladder is capped by it
function ladder() { return LADDER.filter(r => r.res <= maxRes + 8); }

function epProvider() {
  return ep === 'webnn'
    ? { name: 'webnn', deviceType: 'gpu', powerPreference: 'high-performance' }
    : ep;
}

function ctxFor(w, h) {
  const k = w + 'x' + h;
  if (!canvases.has(k)) {
    const off = new OffscreenCanvas(w, h);
    canvases.set(k, { off, ctx: off.getContext('2d', { willReadFrequently: true }) });
  }
  return canvases.get(k);
}

// our WebGPU device + weights, shared by all rt rungs (lazy init)
let rtDevice = null, rtWeights = null;
async function ensureRtDevice() {
  if (rtDevice) return;
  if (!navigator.gpu) throw new Error('no WebGPU in worker');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const f16 = adapter.features.has('shader-f16');
  rtDevice = await adapter.requestDevice({ requiredFeatures: f16 ? ['shader-f16'] : [] });
  // slim (120-wide) student: ~3x faster mids at -0.35dB, 4MB of weights;
  // falls back to the full-width blob if the slim one is not deployed
  for (const stem of ['rt_slim', 'rt_1blk']) {
    try {
      const [bin, man] = await Promise.all([
        fetch(`/assets/${stem}.bin`).then(r => { if (!r.ok) throw 0; return r.arrayBuffer(); }),
        fetch(`/assets/${stem}.json`).then(r => { if (!r.ok) throw 0; return r.json(); })]);
      rtWeights = { bin, man };
      postMessage({ type: 'log', msg: 'rt-веса: ' + stem + ' (' + (bin.byteLength >> 20) + 'МБ)' });
      return;
    } catch { /* next */ }
  }
  throw new Error('rt weights not found');
}

// ---- GPU frame path: bitmaps upload straight into textures, dedup runs on the GPU ----
// round-robin depth 3: a pending job keeps its pair alive while the next frame uploads
let frameTex = [], frameTexIdx = 0, texW = 0, texH = 0;
let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupRead = null, dedupSampler = null;
const DEDUP_N = 48 * 27;

function ensureFrameTextures(w, h) {
  if (texW === w && texH === h && frameTex.length === 3) return;
  frameTex.forEach(t => t.destroy());
  frameTex = [];
  for (let i = 0; i < 3; i++) {
    const t = rtDevice.createTexture({
      label: 'frame' + i, size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
    frameTex.push(t);
  }
  texW = w; texH = h; dedupBg.clear();
}

function ensureDedup() {
  if (dedupPipe) return;
  dedupSampler = rtDevice.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  dedupStats = rtDevice.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  dedupRead = rtDevice.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  dedupPipe = rtDevice.createComputePipeline({ layout: 'auto', compute: {
    module: rtDevice.createShaderModule({ code: /* wgsl */`
@group(0) @binding(0) var t0: texture_2d<f32>;
@group(0) @binding(1) var t1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> stats: array<atomic<u32>, 2>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= 48 || y >= 27) { return; }
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(48.0, 27.0);
  let a = textureSampleLevel(t0, samp, uv, 0.0).rgb;
  let b = textureSampleLevel(t1, samp, uv, 0.0).rgb;
  let d = u32(dot(abs(a - b), vec3<f32>(255.0, 255.0, 255.0)));
  atomicAdd(&stats[0], d);
  atomicMax(&stats[1], d);
}`}), entryPoint: 'main' } });
}

async function gpuIsDup(ta, tb) {
  ensureDedup();
  const key = ta.label + '|' + tb.label;
  if (!dedupBg.has(key)) {
    dedupBg.set(key, rtDevice.createBindGroup({ layout: dedupPipe.getBindGroupLayout(0), entries: [
      { binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() },
      { binding: 2, resource: dedupSampler }, { binding: 3, resource: { buffer: dedupStats } }] }));
  }
  rtDevice.queue.writeBuffer(dedupStats, 0, new Uint32Array([0, 0]));
  const enc = rtDevice.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(dedupPipe); pass.setBindGroup(0, dedupBg.get(key));
  pass.dispatchWorkgroups(6, 4);
  pass.end();
  enc.copyBufferToBuffer(dedupStats, 0, dedupRead, 0, 8);
  rtDevice.queue.submit([enc.finish()]);
  await dedupRead.mapAsync(GPUMapMode.READ);
  const s = new Uint32Array(dedupRead.getMappedRange().slice(0));
  dedupRead.unmap();
  const mean = s[0] / DEDUP_N;
  return {
    dup: mean < 2.5 && s[1] < 45,
    cut: mean > 90, // scene cut: a mid would be a ghost-blend of two scenes
  };
}

// NOTE: the u8 rife_web_* graphs are healthy only on MAIN-thread webnn today:
// the webgpu EP rejects them (JSEP conv-channel bug) and in-worker webnn dies on
// MLTensor uploads AND poisons the wasm runtime afterwards (memory OOB on every
// later call). ort rungs therefore run f32 graphs; rt rungs use our own runtime.
async function buildSession(rung) {
  if (rung.kind === 'rt') {
    await ensureRtDevice();
    const [W, H] = SIZE_RT[rung.res];
    const rt = await createRT(rtDevice, { w: W, h: H, textureInput: true,
      weightsBin: rtWeights.bin, weightsManifest: rtWeights.man });
    return { rt, kind: 'rt', W, H, ms: 0 };
  }
  const { q, res } = rung;
  const [W, H] = SIZE[res];
  const m = F32[q][res];
  const s = await ort.InferenceSession.create(m.url,
    { executionProviders: [epProvider()], graphOptimizationLevel: 'all' });
  const z = new Float32Array(3 * m.eh * m.ew);
  const feeds = {};
  feeds[s.inputNames[0]] = new ort.Tensor('float32', z, [1, 3, m.eh, m.ew]);
  feeds[s.inputNames[1]] = new ort.Tensor('float32', z, [1, 3, m.eh, m.ew]);
  await withGpu(() => s.run(feeds));
  return { sess: s, kind: 'f32', W, H, ew: m.ew, eh: m.eh, ms: 0 };
}

async function ensureRung(key) {
  if (sessions.has(key) || buildingKey === key) return;
  buildingKey = key;
  const rung = LADDER.find(r => r.key === key);
  try {
    sessions.set(key, await buildSession(rung));
    postMessage({ type: 'log', msg: 'ступень ' + key + ' готова' });
  } catch (e) {
    postMessage({ type: 'log', msg: 'ступень ' + key + ' недоступна: ' + String(e.message || e).slice(0, 120) });
    rung.est = 1e9; // never pick it again
  }
  buildingKey = null;
}

function estOf(rung) {
  const s = sessions.get(rung.key);
  const base = s && s.ms ? s.ms : rung.est;
  // per-transition cost: rt rungs produce factor-1 mids, ort graphs always one
  return base * (rung.kind === 'rt' ? Math.max(1, factor - 1) : 1);
}

function controllerTick() {
  if (!auto) return;
  const L = ladder();
  const budget = uniqueIntervalMs;
  const act = L.find(r => r.key === activeKey) || L[L.length - 1];
  const now = performance.now();
  if (estOf(act) > budget) {
    // over budget: step down NOW to the best rung that fits (prefer cached)
    const fit = L.filter(r => estOf(r) < budget * 0.85);
    const cachedFit = fit.find(r => sessions.has(r.key));
    if (cachedFit && cachedFit.key !== activeKey) {
      activeKey = cachedFit.key; last = null; goodSince = now;
      postMessage({ type: 'log', msg: 'авто: вниз на ' + activeKey });
    }
    if (fit[0] && !sessions.has(fit[0].key)) ensureRung(fit[0].key);
  } else {
    // headroom: consider the rung ABOVE after 3s of stability
    const idx = L.indexOf(act);
    if (idx > 0) {
      const up = L[idx - 1];
      if (estOf(up) < budget * 0.75) {
        if (!sessions.has(up.key)) { ensureRung(up.key); return; }
        if (now - goodSince > 3000) {
          activeKey = up.key; last = null; goodSince = now;
          postMessage({ type: 'log', msg: 'авто: вверх на ' + activeKey });
        }
      } else goodSince = now;
    }
  }
}

async function runPair(job) {
  busy = true;
  const S = sessions.get(job.key);
  try {
    // rt rungs honor the DLSS-style factor: n-1 mids at t=k/n in ONE batched GPU submit
    // (pair uploaded once); ort f32 graphs have a baked t=0.5 -> single midpoint.
    const n = S.kind === 'rt' ? job.n : 2;
    if (S.kind === 'rt') {
      const ts = [];
      for (let k = 1; k < n; k++) ts.push(k / n);
      const t0 = performance.now();
      const outs = await S.rt.runMulti(job.a, job.b, ts); // GPUTextures straight in
      const ms = (performance.now() - t0) / ts.length;
      S.ms = S.ms ? S.ms * 0.85 + ms * 0.15 : ms;
      outs.forEach((out, i) => {
        const rgba = new Uint8ClampedArray(out.buffer, 0, out.length);
        postMessage({ type: 'mid', rgba: rgba.buffer, w: S.W, h: S.H, ts: job.ts,
                      frac: ts[i], interpMs: S.ms, halfRate, cfg: job.key + (n > 2 ? ' ×' + n : '') },
                    [rgba.buffer]);
      });
    } else {
      const t0 = performance.now();
      const feeds = {};
      feeds[S.sess.inputNames[0]] = new ort.Tensor('float32', toInput(job.a, S), [1, 3, S.eh, S.ew]);
      feeds[S.sess.inputNames[1]] = new ort.Tensor('float32', toInput(job.b, S), [1, 3, S.eh, S.ew]);
      const out = await withGpu(() => S.sess.run(feeds));
      const outRgba = fromOutput(out[S.sess.outputNames[0]].data, S);
      const ms = performance.now() - t0;
      S.ms = S.ms ? S.ms * 0.85 + ms * 0.15 : ms;
      postMessage({ type: 'mid', rgba: outRgba.buffer, w: S.W, h: S.H, ts: job.ts,
                    frac: 0.5, interpMs: S.ms, halfRate, cfg: job.key },
                  [outRgba.buffer]);
    }
  } catch (e) {
    postMessage({ type: 'error', msg: 'interp: ' + (e.message || e) });
  }
  busy = false;
  if (pending) { const p = pending; pending = null; runPair(p); }
}

function toInput(rgba, S) {
  const x = new Float32Array(3 * S.eh * S.ew);
  const plane = S.eh * S.ew;
  for (let y = 0; y < S.H; y++) {
    for (let i = 0; i < S.W; i++) {
      const p = (y * S.W + i) * 4, o = y * S.ew + i;
      x[o] = rgba[p + 2] / 255;
      x[plane + o] = rgba[p + 1] / 255;
      x[2 * plane + o] = rgba[p] / 255;
    }
  }
  return x;
}
function fromOutput(chw, S) {
  const rgba = new Uint8ClampedArray(S.W * S.H * 4);
  const plane = S.eh * S.ew;
  for (let y = 0; y < S.H; y++) {
    for (let i = 0; i < S.W; i++) {
      const o = y * S.ew + i, p = (y * S.W + i) * 4;
      rgba[p] = Math.min(255, Math.max(0, chw[2 * plane + o] * 255));
      rgba[p + 1] = Math.min(255, Math.max(0, chw[plane + o] * 255));
      rgba[p + 2] = Math.min(255, Math.max(0, chw[o] * 255));
      rgba[p + 3] = 255;
    }
  }
  return rgba;
}

function isNearDup(a, b) {
  const px = a.length / 4;
  const stride = Math.max(1, Math.floor(px / 3000)) * 4;
  let sum = 0, mx = 0, cnt = 0;
  for (let i = 0; i < a.length; i += stride) {
    const d = Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
    sum += d;
    if (d > mx) mx = d;
    cnt++;
  }
  return (sum / cnt) < 2.5 && mx < 45;
}

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    ep = m.ep; auto = !!m.auto; animeMode = m.animeMode; interpOn = m.interpOn;
    factor = m.factor || 2;
    maxRes = m.dispH || 720;
    last = null; busy = false; pending = null;
    try {
      let startKey;
      if (auto) {
        // prefer our own runtime; fall back to ort f32 if WebGPU is unavailable in the worker
        for (const k of ['rt@480', 'fastest@480']) {
          await ensureRung(k);
          if (sessions.has(k)) { startKey = k; break; }
        }
      } else {
        startKey = m.quality + '@' + m.res;
        if (!LADDER.find(r => r.key === startKey)) {
          LADDER.push({ key: startKey, kind: 'f32', q: m.quality, res: m.res, est: 60 });
        }
        await ensureRung(startKey);
      }
      if (!startKey || !sessions.has(startKey)) throw new Error('стартовая ступень не собралась');
      activeKey = startKey; goodSince = performance.now();
      postMessage({ type: 'ready', mode: sessions.get(startKey).kind, cfg: startKey });
    } catch (e) {
      postMessage({ type: 'error', msg: 'init: ' + (e.message || e) });
    }
    return;
  }
  if (m.type === 'opts') {
    animeMode = m.animeMode; interpOn = m.interpOn;
    if (m.factor) factor = m.factor;
    return;
  }
  if (m.type === 'flush') { // seek: the previous frame is unrelated now
    last = null; lastTex = null; pending = null;
    return;
  }
  if (m.type === 'frame') {
    // one frame in flight; a backlog of transferred bitmaps (8MB of GPU memory each)
    // grinds everything — excess frames are dropped for interpolation (originals are
    // displayed by the MAIN thread and don't pass through here)
    if (processingFrame) { m.bmp.close(); return; }
    processingFrame = true;
    handleFrame(m)
      .catch(e => postMessage({ type: 'error', msg: 'frame: ' + (e.message || e) }))
      .finally(() => { processingFrame = false; });
    return;
  }
};

let processingFrame = false;
let lastTex = null;

function scheduleTransition(ts, S, job) {
  const du = ts - lastUniqueTs;
  if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
  lastUniqueTs = ts;
  transitionNo++;
  controllerTick();
  // the factor is a CEILING, not a promise: shrink it until the mids fit the live
  // budget — a steady 4x beats a stuttering 6x. Below 2x fall back to half-rate.
  // Margin >1: the 100ms pipeline delay absorbs transient overruns.
  let effN = S.kind === 'rt' ? factor : 2;
  const ms = S.ms || 60;
  while (effN > 2 && (effN - 1) * ms > uniqueIntervalMs * 1.1) effN--;
  halfRate = (effN - 1) * ms > uniqueIntervalMs * 1.05;
  const skip = halfRate && (transitionNo & 1);
  if (interpOn && !skip) {
    job.n = effN;
    if (!busy) runPair(job);
    else { if (pending) postMessage({ type: 'skipped' }); pending = job; }
  }
}

async function handleFrame(m) {
  const S = sessions.get(activeKey);
  if (S.kind === 'rt') {
    // GPU path: bitmap -> texture, dedup on GPU, zero CPU pixel work.
    // Textures are display-sized and session-independent -> rung switches are seamless.
    ensureFrameTextures(m.bmp.width, m.bmp.height);
    const tex = frameTex[frameTexIdx];
    frameTexIdx = (frameTexIdx + 1) % 3;
    rtDevice.queue.copyExternalImageToTexture({ source: m.bmp }, { texture: tex },
      [m.bmp.width, m.bmp.height]);
    m.bmp.close();
    const prevTex = lastTex;
    if (prevTex) {
      const { dup, cut } = await gpuIsDup(prevTex, tex);
      if (cut) {
        postMessage({ type: 'cut', ts: m.ts });
        lastUniqueTs = m.ts; // new scene starts its own rhythm
      } else if (animeMode && dup) {
        postMessage({ type: 'dup', ts: m.ts });
      } else {
        scheduleTransition(m.ts, S, { a: prevTex, b: tex, ts: m.ts, key: activeKey });
      }
    } else {
      lastUniqueTs = m.ts;
    }
    lastTex = tex;
    return;
  }
  // CPU path (ort f32 fallback only): canvas readback + JS dedup
  const capKey = activeKey;
  const { off, ctx } = ctxFor(S.W, S.H);
  ctx.drawImage(m.bmp, 0, 0, S.W, S.H);
  m.bmp.close();
  const rgba = ctx.getImageData(0, 0, S.W, S.H).data;
  const prevFrame = last;
  if (prevFrame && lastKey === capKey) {
    if (animeMode && isNearDup(prevFrame, rgba)) {
      postMessage({ type: 'dup', ts: m.ts });
    } else {
      scheduleTransition(m.ts, sessions.get(capKey), { a: prevFrame, b: rgba, ts: m.ts, key: capKey });
    }
  } else {
    lastUniqueTs = m.ts;
  }
  last = rgba; lastKey = capKey;
}
