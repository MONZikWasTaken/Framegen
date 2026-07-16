// Framegen player worker - fully GPU-resident and fully OURS: video frames arrive
// as ImageBitmaps, go straight into textures, mids are computed INTO textures by the
// hand-written WGSL runtime (rt.js), and this worker presents everything itself on a
// transferred OffscreenCanvas with its own rAF loop. Nothing pixel-shaped ever
// crosses to the CPU (the sole readback is the 8-byte dedup stat).
//
// ort-web is GONE: WebGPU is a hard requirement now - no wasm/onnx fallback, no CDN.
//
// AUTO mode: a quality ladder for the mids (originals never change). The controller
// watches the real inference cost vs the real scene budget and walks the ladder.
import { createRT, tuneConvRB } from './rt/rt.js?v=8';
import { createSR } from './rt/sr.js?v=4';

const DELAY_MS = 100;
const SIZE_RT = { 352: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1072] };
// rt_v7s = the extension's shipping default (faster than v6 at equal-or-better
// quality); the demo must showcase what users actually get. ensureWeights falls
// back to rt_tfact2 if the v7s set is missing.
const LADDER = [
  { key: 'rt@1080', res: 1080, est: 10, stem: 'rt_v7s' },
  { key: 'rt@720',  res: 720,  est: 4,  stem: 'rt_v7s' },
  { key: 'rt@480',  res: 480,  est: 2,  stem: 'rt_v7s' },
  { key: 'rt@352',  res: 352,  est: 1.6, stem: 'rt_v7s' },
  { key: 'rt60@480', res: 480, est: 3,  stem: 'rt_slim60' },
  { key: 'rt60@352', res: 352, est: 2,  stem: 'rt_slim60' },
];

let auto = false;
let sessions = new Map();
let activeKey = null, buildingKey = null;
let lastTex = null, lastUniqueTs = 0, transitionNo = 0;
let busy = false, pending = null, processingFrame = false;
let uniqueIntervalMs = 42, intervalMs = 42, lastArrival = 0, goodSince = 0, halfRate = false;
let animeMode = true, interpOn = true, factor = 2, maxRes = 720;
function ladder() { return LADDER.filter(r => r.res <= maxRes + 8); }

// ---- device, weights ----
let rtDevice = null;
const rtWeights = new Map();
async function ensureRtDevice() {
  if (rtDevice) return;
  if (!navigator.gpu) throw new Error('no WebGPU - the player only runs on WebGPU browsers');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const feats = adapter.features.has('shader-f16') ? ['shader-f16'] : [];
  if (adapter.features.has('subgroups')) feats.push('subgroups'); // tuner may pick sg kernels
  if (adapter.features.has('timestamp-query')) feats.push('timestamp-query'); // calibration measures on GPU timestamps
  rtDevice = await adapter.requestDevice({ requiredFeatures: feats });
}
let tunes = {}; // res -> {coc, slab, sg}; persisted by the page in localStorage
async function calibrateTune(rung, man, W, H) {
  // one-shot per quality rung: bench kernel variants on the real conv shape;
  // the page saves the winner, it applies on the next session build.
  // EVERY flag the runtime reads persists (dropping w4/v2 here silently kept
  // users on legacy kernels - the extension had the same bug), plus the
  // stride-2 conv0b shape from the s2 sweep.
  try {
    if (tunes[rung.res]) return;
    const wk = man['block0.conv0.1.0.weight'];
    if (!wk) return; // fallback weight sets have a different layout - skip
    const c1k = man['block0.conv0.0.0.weight'];
    const best = await tuneConvRB(rtDevice, { ci: wk.shape[0], co: wk.shape[0],
      w16: W / 16, h16: H / 16, s2ci: c1k ? c1k.shape[0] : 0 });
    tunes[rung.res] = { coc: best.coc, slab: best.slab, sg: !!best.sg,
      wgx: best.wgx || 8, wgy: best.wgy || 8, w4: !!best.w4, v2: !!best.v2 };
    if (best.s2) tunes[rung.res].s2 = { coc: best.s2.coc, w4: !!best.s2.w4 };
    postMessage({ type: 'tune', res: rung.res, tune: tunes[rung.res] });
    postMessage({ type: 'log', msg: 'conv tune ' + rung.res + 'p: ' + JSON.stringify(tunes[rung.res]) });
  } catch { /* tuner is best-effort */ }
}
async function ensureWeights(stem) {
  if (rtWeights.has(stem)) return rtWeights.get(stem);
  const tryFetch = async (s) => {
    const [bin, man] = await Promise.all([
      fetch(`/assets/${s}.bin`).then(r => { if (!r.ok) throw 0; return r.arrayBuffer(); }),
      fetch(`/assets/${s}.json`).then(r => { if (!r.ok) throw 0; return r.json(); })]);
    return { bin, man };
  };
  let w = null;
  for (const s of [stem, 'rt_tfact2', 'rt_tfact', 'rt_slim', 'rt_1blk']) {
    try { w = await tryFetch(s); if (s !== stem) postMessage({ type: 'log', msg: stem + ' missing - falling back to ' + s }); break; }
    catch { /* next */ }
  }
  if (!w) throw new Error('weights not found (' + stem + ')');
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
  const anyTune = Object.values(tunes).find(t => t && t.coc) || null;
  sr = await createSR(rtDevice, { weightsBin: bin, weightsManifest: man,
    convTune: anyTune || { coc: 8, slab: 12, sg: true, w4: true, v2: true } });
  postMessage({ type: 'log', msg: 'SR upscaler loaded (' + (bin.byteLength >> 10) + 'KB)' });
}
function srDstFor(w, h) {
  const k = w + 'x' + h;
  let t = srTexs.get(k);
  if (!t) {
    // ladder switches change size wholesale - destroy old sizes (and their
    // cached blit bind groups) instead of leaking tens of MB per visited size
    if (srTexs.size > 2) {
      for (const [kk, tt] of srTexs) { tt.destroy(); srTexs.delete(kk); }
      blitBgCache.clear();
    }
    t = rtDevice.createTexture({ label: 'sr' + k, size: [w * 2, h * 2],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    srTexs.set(k, t);
  }
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
    // evict BEFORE inserting - clearing after wipes the fresh entry and the
    // present blit binds undefined on the 49th texture
    if (blitBgCache.size > 48) blitBgCache.clear();
    blitBgCache.set(tex, rtDevice.createBindGroup({ layout: blitPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: blitSampler }] }));
  }
  return blitBgCache.get(tex);
}
function present(texIn) {
  let tex = texIn;
  if (srOn && sr) {
    const dst = srDstFor(texIn.width, texIn.height);
    // false while the per-size pipelines compile (async) - show the raw frame
    // instead of a zero-initialized ring texture (black flashes)
    if (sr.process(texIn, dst, texIn.width, texIn.height)) tex = dst;
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
  driveJob(now); // just-in-time mid submission
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
let frameTex = [], frameTexIdx = 0, texW = 0, texH = 0;
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
  queue = []; curJob = null; pairSeq++; // queued entries reference the destroyed pool
}

let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupSampler = null;
let dedupReads = [], dedupReadIdx = 0, pairSeq = 0;
const DEDUP_N = 48 * 27;
function ensureDedup() {
  if (dedupPipe) return;
  dedupSampler = rtDevice.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  dedupStats = rtDevice.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  dedupReads = Array.from({ length: 3 }, () => ({ busy: false,
    buf: rtDevice.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) }));
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
const DEDUP_ZERO = new Uint32Array(2);
async function gpuIsDup(ta, tb) {
  ensureDedup();
  // free readback slot: classifies overlap (handleFrame does not await them);
  // all busy = treat as ordinary motion instead of stalling the frame loop
  let rb = null;
  for (let i = 0; i < dedupReads.length; i++) {
    const c = dedupReads[(dedupReadIdx + i) % dedupReads.length];
    if (!c.busy) { rb = c; dedupReadIdx = (dedupReadIdx + i + 1) % dedupReads.length; break; }
  }
  if (!rb) return { dup: false, cut: false };
  rb.busy = true;
  try {
    const key = ta.label + '|' + tb.label;
    if (!dedupBg.has(key)) {
      dedupBg.set(key, rtDevice.createBindGroup({ layout: dedupPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() },
        { binding: 2, resource: dedupSampler }, { binding: 3, resource: { buffer: dedupStats } }] }));
    }
    // the single stats buffer is safe across overlapping classifies: zero,
    // dispatch and the copy-out are queue-ordered per submit
    rtDevice.queue.writeBuffer(dedupStats, 0, DEDUP_ZERO);
    const enc = rtDevice.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(dedupPipe); pass.setBindGroup(0, dedupBg.get(key));
    pass.dispatchWorkgroups(6, 4);
    pass.end();
    enc.copyBufferToBuffer(dedupStats, 0, rb.buf, 0, 8);
    rtDevice.queue.submit([enc.finish()]);
    await rb.buf.mapAsync(GPUMapMode.READ);
    const s = new Uint32Array(rb.buf.getMappedRange().slice(0));
    rb.buf.unmap();
    const mean = s[0] / DEDUP_N;
    return { dup: mean < 2.5 && s[1] < 45, cut: mean > 90 };
  } finally {
    rb.busy = false;
  }
}

// ---- sessions / controller ----
async function buildSession(rung) {
  await ensureRtDevice();
  const wset = await ensureWeights(rung.stem);
  const [W, H] = SIZE_RT[rung.res];
  const rt = await createRT(rtDevice, { w: W, h: H, textureInput: true, textureOutput: true,
    staticGuard: true, weightsBin: wset.bin, weightsManifest: wset.man,
    convTune: tunes[rung.res] || null });
  if (!tunes[rung.res]) {
    const t0 = performance.now();
    const tick = () => {
      if (tunes[rung.res]) return;
      // idle moment (no frames for a while) or 2 minutes of nonstop playback
      if (performance.now() - lastArrival > 600 || performance.now() - t0 > 120000) {
        calibrateTune(rung, wset.man, W, H);
        return;
      }
      setTimeout(tick, 3000);
    };
    setTimeout(tick, 4000);
  }
  const midTexs = [];
  for (let i = 0; i < 12; i++) { // ring: up to (factor-1) in flight + ~100ms in the queue
    midTexs.push(rtDevice.createTexture({ label: rung.key + '#' + i, size: [W, H],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
  }
  return { rt, W, H, ms: 0, midTexs, midIdx: 0 };
}
function destroySession(key) {
  const S = sessions.get(key);
  if (!S) return;
  sessions.delete(key);
  if (S.rt.destroy) S.rt.destroy();
  S.midTexs.forEach(t => t.destroy());
  blitBgCache.clear(); // cached blit bind groups may reference destroyed mids
}
async function ensureRung(key) {
  if (sessions.has(key) || buildingKey === key) return;
  buildingKey = key;
  const rung = LADDER.find(r => r.key === key);
  try {
    sessions.set(key, await buildSession(rung));
    postMessage({ type: 'log', msg: 'rung ' + key + ' ready' });
    // evict idle rungs: >4 sessions alive is tens of MB of VRAM going nowhere.
    // Only rungs untouched for 5s+ qualify - their mids are long out of the
    // display queue, so destroying textures cannot hit an in-flight present.
    if (sessions.size > 4) {
      const now = performance.now();
      for (const [k, S] of sessions) {
        if (k === key || k === activeKey) continue;
        if ((S.lastUsed || 0) < now - 5000 && (!curJob || curJob.S !== S)) {
          destroySession(k);
          postMessage({ type: 'log', msg: 'rung ' + k + ' evicted (idle)' });
          break;
        }
      }
    }
  } catch (e) {
    postMessage({ type: 'log', msg: 'rung ' + key + ' unavailable: ' + String(e.message || e).slice(0, 120) });
    rung.est = 1e9;
  }
  buildingKey = null;
}
function estOf(rung) {
  const s = sessions.get(rung.key);
  const base = s && s.ms ? s.ms : rung.est;
  return base * Math.max(1, factor - 1);
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
      activeKey = cachedFit.key; goodSince = now;
      postMessage({ type: 'log', msg: 'auto: down to ' + activeKey });
    }
    if (fit[0] && !sessions.has(fit[0].key)) ensureRung(fit[0].key);
  } else {
    const idx = L.indexOf(act);
    if (idx > 0) {
      const up = L[idx - 1];
      if (estOf(up) < budget * 0.75) {
        if (!sessions.has(up.key)) { ensureRung(up.key); return; }
        if (now - goodSince > 3000) {
          activeKey = up.key; goodSince = now;
          postMessage({ type: 'log', msg: 'auto: up to ' + activeKey });
        }
      } else goodSince = now;
    }
  }
}

// ---- interpolation jobs (lazy per-mid submission) ----
// prepPair runs once per frame pair; each mid's compute is submitted just-in-
// time for its display slot from the pump, so present blits interleave with
// computes on the GPU queue instead of the first mid waiting for the whole
// batch (the old runMulti path cost ~(n-1)*ms of extra latency at high factors).
let curJob = null;
function texQueued(t) {
  for (let i = 0; i < queue.length; i++) if (queue[i].tex === t) return true;
  return false;
}
function submitMid() {
  const j = curJob, S = j.S;
  const k = j.next;
  let guard = S.midTexs.length; // never clobber queued mids
  while (guard-- > 0 && texQueued(S.midTexs[S.midIdx])) S.midIdx = (S.midIdx + 1) % S.midTexs.length;
  const out = S.midTexs[S.midIdx];
  S.midIdx = (S.midIdx + 1) % S.midTexs.length;
  const t0 = performance.now();
  try { S.rt.runT(j.ts[k], out); } catch (e) {
    curJob = null;
    postMessage({ type: 'error', msg: 'interp: ' + (e.message || e) });
    return;
  }
  // sample every 4th mid starting at k=1 (mid 0's drain also swallows the trunk
  // prep ahead of it and inflates the estimate); single-mid jobs sample k=0
  if ((k & 3) === 1 || j.ts.length === 1) {
    rtDevice.queue.onSubmittedWorkDone().then(() => {
      const ms = performance.now() - t0;
      S.ms = S.ms ? S.ms * 0.85 + ms * 0.15 : ms;
    });
  }
  queue.push({ tex: out, at: j.at + j.ts[k] * intervalMs });
  j.next++;
  if (j.next >= j.ts.length) curJob = null;
}
function flushJob() {
  while (curJob && curJob.next < curJob.ts.length) submitMid();
  curJob = null;
}
function driveJob(now) {
  if (!curJob) return;
  const lead = 2 * (curJob.S.ms || 10) + 8; // one compute away from the slot
  while (curJob && curJob.next < curJob.ts.length) {
    const disp = curJob.at + curJob.ts[curJob.next] * intervalMs;
    if (disp - now > lead) break;
    submitMid();
  }
}
function runPair(job) {
  const S = sessions.get(job.key);
  try {
    flushJob(); // leftovers of the previous pair go out before the new prep
    const n = job.n;
    const ts = [];
    for (let k = 1; k < n; k++) ts.push(k / n);
    S.rt.prepPair(job.a, job.b);
    S.lastUsed = performance.now();
    curJob = { S, ts, next: 0, at: job.at };
    midCfg = job.key + (n > 2 ? ' ×' + n : '');
  } catch (e) {
    postMessage({ type: 'error', msg: 'interp: ' + (e.message || e) });
  }
}

function scheduleTransition(S, job) {
  const now = job.ts;
  const du = now - lastUniqueTs;
  if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
  lastUniqueTs = now;
  transitionNo++;
  controllerTick();
  let effN = factor;
  const ms = S.ms || 10;
  while (effN > 2 && (effN - 1) * ms > uniqueIntervalMs * 1.1) effN--;
  halfRate = (effN - 1) * ms > uniqueIntervalMs * 1.05;
  const skip = halfRate && (transitionNo & 1);
  if (interpOn && !skip) {
    job.n = effN;
    runPair(job);
  }
}

async function handleFrame(m) {
  const arrival = performance.now();
  const dt = arrival - lastArrival;
  if (dt > 5 && dt < 500) intervalMs = intervalMs * 0.9 + dt * 0.1;
  lastArrival = arrival;
  const S = sessions.get(activeKey);
  if (!S || !presenting) { m.bmp.close(); return; }
  ensureFrameTextures(m.bmp.width, m.bmp.height);
  // never overwrite a texture still queued for presentation or the pair input
  let guard = frameTex.length;
  while (guard-- > 0 && (frameTex[frameTexIdx] === lastTex || texQueued(frameTex[frameTexIdx]))) {
    frameTexIdx = (frameTexIdx + 1) % frameTex.length;
  }
  const tex = frameTex[frameTexIdx];
  frameTexIdx = (frameTexIdx + 1) % frameTex.length;
  rtDevice.queue.copyExternalImageToTexture({ source: m.bmp }, { texture: tex },
    [m.bmp.width, m.bmp.height]);
  m.bmp.close();
  queue.push({ tex, at: arrival + DELAY_MS }); // the original presents itself here
  const prevTex = lastTex;
  if (prevTex) {
    // the dedup readback is a GPU->CPU roundtrip and must not block the frame
    // loop (awaiting it starved 120fps sources of every other input frame).
    // The continuation lands a few ms later, well inside the display buffer;
    // pairSeq guards against a newer pair / pool realloc superseding this one.
    const seq = ++pairSeq;
    gpuIsDup(prevTex, tex).then(({ dup, cut }) => {
      if (!presenting || seq !== pairSeq) return;
      if (cut) { cuts++; lastUniqueTs = arrival; }
      else if (animeMode && dup) { dups++; }
      else {
        // mids sit between the previous original (on screen at ~arrival-interval+DELAY)
        // and this one (at arrival+DELAY)
        scheduleTransition(S, { a: prevTex, b: tex, ts: arrival,
                                at: arrival - intervalMs + DELAY_MS, key: activeKey });
      }
    }).catch(e => postMessage({ type: 'error', msg: 'dedup: ' + (e.message || e) }));
  } else {
    lastUniqueTs = arrival;
  }
  lastTex = tex;
}

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    auto = !!m.auto; animeMode = m.animeMode; interpOn = m.interpOn;
    if (m.tunes) tunes = m.tunes;
    factor = m.factor || 2;
    maxRes = m.dispH || 720;
    canvas = m.canvas || null;
    lastTex = null; busy = false; pending = null; queue = [];
    try {
      if (!canvas) throw new Error('OffscreenCanvas required (present:worker)');
      const rmap = { 360: 352, 352: 352, 480: 480, 720: 720, 1080: 1080 };
      const startKey = auto ? 'rt@480' : ('rt@' + (rmap[+m.res] || 480));
      if (!LADDER.find(r => r.key === startKey)) throw new Error('no rung ' + startKey);
      await ensureRung(startKey);
      if (!sessions.has(startKey)) throw new Error('starting rung failed to build');
      activeKey = startKey; goodSince = performance.now();
      srOn = !!m.sr;
      if (srOn) {
        try { await ensureSR(); } catch (e) {
          srOn = false;
          postMessage({ type: 'log', msg: 'SR unavailable: ' + (e.message || e) });
        }
      }
      // with SR the canvas backing store is 2x - real pixels instead of browser upscale
      const mul = srOn ? 2 : 1;
      canvas.width = m.dispW * mul; canvas.height = m.dispH * mul;
      ensurePresent();
      presenting = true;
      requestAnimationFrame(pump);
      postMessage({ type: 'ready', mode: 'rt', present: 'worker', cfg: startKey });
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
    lastTex = null; pending = null; queue = []; curJob = null; pairSeq++;
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
