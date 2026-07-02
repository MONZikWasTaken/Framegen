// Framecast player worker — fully GPU-resident: video frames arrive as ImageBitmaps,
// go straight into textures, mids are computed INTO textures, and this worker presents
// everything itself on a transferred OffscreenCanvas with its own rAF loop. Nothing
// pixel-shaped ever crosses to the CPU (the sole readback is the 8-byte dedup stat).
//
// AUTO mode: a quality ladder for the mids (originals never change). The controller
// watches the real inference cost vs the real scene budget and walks the ladder.
// Fallback: without WebGPU the worker runs ort f32 graphs and posts mids to the main
// thread, which presents them the old way (present:'main' in the ready message).
import * as ortNS from 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.mjs';
import { createRT } from './rt/rt.js?v=4';
import { createSR } from './rt/sr.js?v=1';
const ort = ortNS.default ?? ortNS;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const DELAY_MS = 100;
const SIZE = { 360: [640, 360], 480: [854, 480], 720: [1280, 720] };
const SIZE_RT = { 352: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1072] };
const LADDER = [
  { key: 'rt@1080',     kind: 'rt',  q: 'turbo',   res: 1080, est: 24, stem: 'rt_slim' },
  { key: 'fastest@720', kind: 'f32', q: 'fastest', res: 720, est: 150 },
  { key: 'rt@720',      kind: 'rt',  q: 'turbo',   res: 720, est: 11, stem: 'rt_slim' },
  { key: 'fastest@480', kind: 'f32', q: 'fastest', res: 480, est: 60 },
  { key: 'rt@480',      kind: 'rt',  q: 'turbo',   res: 480, est: 5, stem: 'rt_slim' },
  { key: 'rt@352',      kind: 'rt',  q: 'turbo',   res: 352, est: 3, stem: 'rt_slim' },
  { key: 'rt60@480',    kind: 'rt',  q: 'turbo',   res: 480, est: 3, stem: 'rt_slim60' },
  { key: 'rt60@352',    kind: 'rt',  q: 'turbo',   res: 352, est: 2, stem: 'rt_slim60' },
];
const F32 = {
  turbo:   { 360: { url: '/assets/rife_lite_360p_1blk_s4_student1b.onnx', ew: 640, eh: 384 },
             480: { url: '/assets/rife_lite_480p_1blk_s4_student1b.onnx', ew: 864, eh: 480 },
             720: { url: '/assets/rife_lite_720p_1blk_s4_student1b.onnx', ew: 1280, eh: 736 } },
  fastest: { 360: { url: '/assets/rife_lite_360p_2blk_noref_student.onnx', ew: 640, eh: 384 },
             480: { url: '/assets/rife_lite_480p_2blk_noref_student.onnx', ew: 864, eh: 480 },
             720: { url: '/assets/rife_lite_720p_2blk_noref_student.onnx', ew: 1280, eh: 736 } },
};

let gpuLock = Promise.resolve(); // ort-web serializes run() globally
function withGpu(fn) { const p = gpuLock.then(fn); gpuLock = p.catch(() => {}); return p; }

let ep = 'webnn', auto = false;
let sessions = new Map();
let activeKey = null, buildingKey = null;
let canvases = new Map();
let last = null, lastKey = null, lastUniqueTs = 0, transitionNo = 0;
let busy = false, pending = null, processingFrame = false;
let uniqueIntervalMs = 42, intervalMs = 42, lastArrival = 0, goodSince = 0, halfRate = false;
let animeMode = true, interpOn = true, factor = 2, maxRes = 720;
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

// ---- device, weights ----
let rtDevice = null;
const rtWeights = new Map();
async function ensureRtDevice() {
  if (rtDevice) return;
  if (!navigator.gpu) throw new Error('no WebGPU in worker');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const f16 = adapter.features.has('shader-f16');
  rtDevice = await adapter.requestDevice({ requiredFeatures: f16 ? ['shader-f16'] : [] });
}
async function ensureWeights(stem) {
  if (rtWeights.has(stem)) return rtWeights.get(stem);
  const tryFetch = async (s) => {
    const [bin, man] = await Promise.all([
      fetch(`/assets/${s}.bin`).then(r => { if (!r.ok) throw 0; return r.arrayBuffer(); }),
      fetch(`/assets/${s}.json`).then(r => { if (!r.ok) throw 0; return r.json(); })]);
    return { bin, man };
  };
  let w;
  try { w = await tryFetch(stem); }
  catch { w = await tryFetch('rt_1blk'); postMessage({ type: 'log', msg: stem + ' нет — беру rt_1blk' }); }
  rtWeights.set(stem, w);
  return w;
}

// ---- presentation: transferred canvas + blit pipeline + display queue + rAF ----
let canvas = null, canvasCtx = null, blitPipe = null, blitSampler = null;
const blitBgCache = new Map();
let queue = []; // {tex, at}
let shown = 0, dropped = 0, dups = 0, cuts = 0;
let fpsWindow = [], statsTimer = 0, presenting = false, midCfg = '';

// ---- optional 2x SR pass on everything presented (anime upscale) ----
let sr = null, srOn = false;
const srTexs = new Map(); // "WxH" -> {ring: [tex], idx}
async function ensureSR() {
  if (sr) return;
  const [bin, man] = await Promise.all([
    fetch('/assets/rt_sr.bin').then(r => { if (!r.ok) throw new Error('rt_sr.bin missing'); return r.arrayBuffer(); }),
    fetch('/assets/rt_sr.json').then(r => r.json())]);
  sr = await createSR(rtDevice, { weightsBin: bin, weightsManifest: man });
  postMessage({ type: 'log', msg: 'SR-апскейлер загружен (' + (bin.byteLength >> 10) + 'КБ)' });
}
function srDstFor(w, h) {
  const k = w + 'x' + h;
  if (!srTexs.has(k)) {
    const ring = [];
    for (let i = 0; i < 4; i++) {
      ring.push(rtDevice.createTexture({ label: 'sr' + k + '#' + i, size: [w * 2, h * 2],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    }
    srTexs.set(k, { ring, idx: 0 });
  }
  const s = srTexs.get(k);
  const t = s.ring[s.idx];
  s.idx = (s.idx + 1) % s.ring.length;
  return t;
}

function ensurePresent() {
  if (blitPipe) return;
  canvasCtx = canvas.getContext('webgpu');
  canvasCtx.configure({ device: rtDevice, format: 'rgba8unorm', alphaMode: 'opaque' });
  blitSampler = rtDevice.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  const mod = rtDevice.createShaderModule({ code: /* wgsl */`
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4(p[i], 0.0, 1.0);
  o.uv = vec2(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, v.uv, 0.0);
}`});
  blitPipe = rtDevice.createRenderPipeline({ layout: 'auto',
    vertex: { module: mod, entryPoint: 'vs' },
    fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] } });
}
function blitBgFor(tex) {
  if (!blitBgCache.has(tex)) {
    blitBgCache.set(tex, rtDevice.createBindGroup({ layout: blitPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: blitSampler }] }));
    if (blitBgCache.size > 48) blitBgCache.clear();
  }
  return blitBgCache.get(tex);
}
function present(texIn) {
  let tex = texIn;
  if (srOn && sr) {
    const dst = srDstFor(texIn.width, texIn.height);
    sr.process(texIn, dst, texIn.width, texIn.height);
    tex = dst;
  }
  const enc = rtDevice.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{
    view: canvasCtx.getCurrentTexture().createView(),
    loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
  pass.setPipeline(blitPipe);
  pass.setBindGroup(0, blitBgFor(tex));
  pass.draw(3);
  pass.end();
  rtDevice.queue.submit([enc.finish()]);
  shown++;
  const now = performance.now();
  fpsWindow.push(now);
  while (fpsWindow.length && fpsWindow[0] < now - 1000) fpsWindow.shift();
}
function pump(now) {
  if (!presenting) return;
  queue.sort((a, b) => a.at - b.at);
  let due = -1;
  for (let i = 0; i < queue.length; i++) if (queue[i].at <= now) due = i;
  if (due >= 0) {
    dropped += due;
    present(queue[due].tex);
    queue = queue.slice(due + 1);
  }
  if (now - statsTimer > 250) {
    statsTimer = now;
    const S = sessions.get(activeKey);
    postMessage({ type: 'stats', fps: fpsWindow.length, interpMs: S && S.ms ? S.ms : 0,
                  cfg: midCfg || activeKey, halfRate, shown, dropped, dups, cuts });
  }
  requestAnimationFrame(pump);
}

// ---- frame textures (originals) + dedup ----
let frameTex = [], frameTexIdx = 0, texW = 0, texH = 0, lastTex = null;
function ensureFrameTextures(w, h) {
  if (texW === w && texH === h && frameTex.length === 8) return;
  frameTex.forEach(t => t.destroy());
  frameTex = [];
  for (let i = 0; i < 8; i++) { // deep ring: items sit ~100ms in the display queue
    frameTex.push(rtDevice.createTexture({
      label: 'frame' + i, size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }));
  }
  texW = w; texH = h; dedupBg.clear(); blitBgCache.clear(); lastTex = null;
}

let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupRead = null, dedupSampler = null;
const DEDUP_N = 48 * 27;
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
  return { dup: mean < 2.5 && s[1] < 45, cut: mean > 90 };
}

// ---- sessions / controller ----
async function buildSession(rung) {
  if (rung.kind === 'rt') {
    await ensureRtDevice();
    const wset = await ensureWeights(rung.stem || 'rt_slim');
    const [W, H] = SIZE_RT[rung.res];
    const rt = await createRT(rtDevice, { w: W, h: H, textureInput: true, textureOutput: true,
      weightsBin: wset.bin, weightsManifest: wset.man });
    const midTexs = [];
    for (let i = 0; i < 12; i++) { // ring: up to (factor-1) in flight + ~100ms in the queue
      midTexs.push(rtDevice.createTexture({ label: rung.key + '#' + i, size: [W, H],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    }
    return { rt, kind: 'rt', W, H, ms: 0, midTexs, midIdx: 0 };
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
    rung.est = 1e9;
  }
  buildingKey = null;
}
function estOf(rung) {
  const s = sessions.get(rung.key);
  const base = s && s.ms ? s.ms : rung.est;
  return base * (rung.kind === 'rt' ? Math.max(1, factor - 1) : 1);
}
function controllerTick() {
  if (!auto) return;
  const L = ladder();
  const budget = uniqueIntervalMs;
  const act = L.find(r => r.key === activeKey) || L[L.length - 1];
  const now = performance.now();
  if (estOf(act) > budget) {
    const fit = L.filter(r => estOf(r) < budget * 0.85);
    const cachedFit = fit.find(r => sessions.has(r.key));
    if (cachedFit && cachedFit.key !== activeKey) {
      activeKey = cachedFit.key; last = null; goodSince = now;
      postMessage({ type: 'log', msg: 'авто: вниз на ' + activeKey });
    }
    if (fit[0] && !sessions.has(fit[0].key)) ensureRung(fit[0].key);
  } else {
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

// ---- f32 prepost (fallback path only) ----
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

// ---- interpolation jobs ----
async function runPair(job) {
  busy = true;
  const S = sessions.get(job.key);
  try {
    const n = S.kind === 'rt' ? job.n : 2;
    if (S.kind === 'rt') {
      const ts = [];
      for (let k = 1; k < n; k++) ts.push(k / n);
      const outs = [];
      for (let k = 0; k < ts.length; k++) {
        outs.push(S.midTexs[S.midIdx]);
        S.midIdx = (S.midIdx + 1) % S.midTexs.length;
      }
      const t0 = performance.now();
      await S.rt.runMulti(job.a, job.b, ts, outs); // submit only — mids stay on the GPU
      rtDevice.queue.onSubmittedWorkDone().then(() => {
        const ms = (performance.now() - t0) / ts.length;
        S.ms = S.ms ? S.ms * 0.85 + ms * 0.15 : ms;
      });
      midCfg = job.key + (n > 2 ? ' ×' + n : '');
      for (let k = 0; k < ts.length; k++) {
        queue.push({ tex: outs[k], at: job.at + ts[k] * intervalMs });
      }
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
                    frac: 0.5, interpMs: S.ms, halfRate, cfg: job.key }, [outRgba.buffer]);
    }
  } catch (e) {
    postMessage({ type: 'error', msg: 'interp: ' + (e.message || e) });
  }
  busy = false;
  if (pending) { const p = pending; pending = null; runPair(p); }
}

function scheduleTransition(S, job) {
  const now = job.ts;
  const du = now - lastUniqueTs;
  if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
  lastUniqueTs = now;
  transitionNo++;
  controllerTick();
  let effN = S.kind === 'rt' ? factor : 2;
  const ms = S.ms || 60;
  while (effN > 2 && (effN - 1) * ms > uniqueIntervalMs * 1.1) effN--;
  halfRate = (effN - 1) * ms > uniqueIntervalMs * 1.05;
  const skip = halfRate && (transitionNo & 1);
  if (interpOn && !skip) {
    job.n = effN;
    if (!busy) runPair(job);
    else { if (pending) dropped++; pending = job; }
  }
}

async function handleFrame(m) {
  const arrival = performance.now();
  const dt = arrival - lastArrival;
  if (dt > 5 && dt < 500) intervalMs = intervalMs * 0.9 + dt * 0.1;
  lastArrival = arrival;
  const S = sessions.get(activeKey);
  if (S.kind === 'rt' && presenting) {
    ensureFrameTextures(m.bmp.width, m.bmp.height);
    const tex = frameTex[frameTexIdx];
    frameTexIdx = (frameTexIdx + 1) % frameTex.length;
    rtDevice.queue.copyExternalImageToTexture({ source: m.bmp }, { texture: tex },
      [m.bmp.width, m.bmp.height]);
    m.bmp.close();
    queue.push({ tex, at: arrival + DELAY_MS }); // the original presents itself here
    const prevTex = lastTex;
    if (prevTex) {
      const { dup, cut } = await gpuIsDup(prevTex, tex);
      if (cut) { cuts++; lastUniqueTs = arrival; }
      else if (animeMode && dup) { dups++; }
      else {
        // mids sit between the previous original (on screen at ~arrival-interval+DELAY)
        // and this one (at arrival+DELAY)
        scheduleTransition(S, { a: prevTex, b: tex, ts: arrival,
                                at: arrival - intervalMs + DELAY_MS, key: activeKey });
      }
    } else {
      lastUniqueTs = arrival;
    }
    lastTex = tex;
    return;
  }
  // ort fallback: CPU pixels, mids posted to the main thread
  const capKey = activeKey;
  const { ctx } = ctxFor(S.W, S.H);
  ctx.drawImage(m.bmp, 0, 0, S.W, S.H);
  m.bmp.close();
  const rgba = ctx.getImageData(0, 0, S.W, S.H).data;
  const prevFrame = last;
  if (prevFrame && lastKey === capKey) {
    if (animeMode && isNearDup(prevFrame, rgba)) { dups++; postMessage({ type: 'dup' }); }
    else scheduleTransition(sessions.get(capKey), { a: prevFrame, b: rgba, ts: m.ts, key: capKey });
  } else {
    lastUniqueTs = m.ts;
  }
  last = rgba; lastKey = capKey;
}

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    ep = m.ep; auto = !!m.auto; animeMode = m.animeMode; interpOn = m.interpOn;
    factor = m.factor || 2;
    maxRes = m.dispH || 720;
    canvas = m.canvas || null;
    last = null; lastTex = null; busy = false; pending = null; queue = [];
    try {
      let startKey;
      if (auto) {
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
      const S = sessions.get(startKey);
      let presentMode = 'main';
      if (S.kind === 'rt' && canvas) {
        srOn = !!m.sr;
        if (srOn) {
          try { await ensureSR(); } catch (e) {
            srOn = false;
            postMessage({ type: 'log', msg: 'SR недоступен: ' + (e.message || e) });
          }
        }
        // with SR the canvas backing store is 2x — real pixels instead of browser upscale
        const mul = srOn ? 2 : 1;
        canvas.width = m.dispW * mul; canvas.height = m.dispH * mul;
        ensurePresent();
        presenting = true;
        requestAnimationFrame(pump);
        presentMode = 'worker';
      }
      postMessage({ type: 'ready', mode: S.kind, present: presentMode, cfg: startKey });
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
  if (m.type === 'flush') {
    last = null; lastTex = null; pending = null; queue = [];
    return;
  }
  if (m.type === 'frame') {
    if (processingFrame) { m.bmp.close(); return; }
    processingFrame = true;
    handleFrame(m)
      .catch(e => postMessage({ type: 'error', msg: 'frame: ' + (e.message || e) }))
      .finally(() => { processingFrame = false; });
    return;
  }
};
