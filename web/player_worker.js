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
// custom-runtime sizes must be /16-divisible (852->848, 360->352 — sub-1% stretch)
const SIZE_RT = { 352: [640, 352], 480: [848, 480], 720: [1280, 720] };
// mids ladder, best quality first; est = initial ms guess (learned at runtime).
// rt = our own WebGPU runtime (1blk model): bit-exact, no flags, ~3-6x faster than ort.
const LADDER = [
  { key: 'fastest@720', kind: 'f32', q: 'fastest', res: 720, est: 150 },
  { key: 'rt@720',      kind: 'rt',  q: 'turbo',   res: 720, est: 36 },
  { key: 'fastest@480', kind: 'f32', q: 'fastest', res: 480, est: 60 },
  { key: 'rt@480',      kind: 'rt',  q: 'turbo',   res: 480, est: 16 },
  { key: 'rt@352',      kind: 'rt',  q: 'turbo',   res: 352, est: 9 },
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
let animeMode = true, interpOn = true, factor = 2;

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
  rtDevice = await adapter.requestDevice();
  const [bin, man] = await Promise.all([
    fetch('/assets/rt_1blk.bin').then(r => r.arrayBuffer()),
    fetch('/assets/rt_1blk.json').then(r => r.json())]);
  rtWeights = { bin, man };
}

// NOTE: the u8 rife_web_* graphs are healthy only on MAIN-thread webnn today:
// the webgpu EP rejects them (JSEP conv-channel bug) and in-worker webnn dies on
// MLTensor uploads AND poisons the wasm runtime afterwards (memory OOB on every
// later call). ort rungs therefore run f32 graphs; rt rungs use our own runtime.
async function buildSession(rung) {
  if (rung.kind === 'rt') {
    await ensureRtDevice();
    const [W, H] = SIZE_RT[rung.res];
    const rt = await createRT(rtDevice, { w: W, h: H,
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
  const budget = uniqueIntervalMs;
  const act = LADDER.find(r => r.key === activeKey);
  const now = performance.now();
  if (estOf(act) > budget) {
    // over budget: step down NOW to the best rung that fits (prefer cached)
    const fit = LADDER.filter(r => estOf(r) < budget * 0.85);
    const cachedFit = fit.find(r => sessions.has(r.key));
    if (cachedFit && cachedFit.key !== activeKey) {
      activeKey = cachedFit.key; last = null; goodSince = now;
      postMessage({ type: 'log', msg: 'авто: вниз на ' + activeKey });
    }
    if (fit[0] && !sessions.has(fit[0].key)) ensureRung(fit[0].key);
  } else {
    // headroom: consider the rung ABOVE after 3s of stability
    const idx = LADDER.indexOf(act);
    if (idx > 0) {
      const up = LADDER[idx - 1];
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
    // rt rungs honor the DLSS-style factor: n-1 mids at t=k/n; ort f32 graphs have a
    // baked t=0.5, so they always produce the single midpoint whatever the factor.
    const n = S.kind === 'rt' ? job.n : 2;
    for (let k = 1; k < n; k++) {
      const t0 = performance.now();
      let outRgba;
      if (S.kind === 'rt') {
        const out = await S.rt.run(new Uint8Array(job.a.buffer, job.a.byteOffset, job.a.length),
                                   new Uint8Array(job.b.buffer, job.b.byteOffset, job.b.length),
                                   k / n);
        outRgba = new Uint8ClampedArray(out.buffer, 0, out.length);
      } else {
        const feeds = {};
        feeds[S.sess.inputNames[0]] = new ort.Tensor('float32', toInput(job.a, S), [1, 3, S.eh, S.ew]);
        feeds[S.sess.inputNames[1]] = new ort.Tensor('float32', toInput(job.b, S), [1, 3, S.eh, S.ew]);
        const out = await withGpu(() => S.sess.run(feeds));
        outRgba = fromOutput(out[S.sess.outputNames[0]].data, S);
      }
      const ms = performance.now() - t0;
      S.ms = S.ms ? S.ms * 0.85 + ms * 0.15 : ms;
      postMessage({ type: 'mid', rgba: outRgba.buffer, w: S.W, h: S.H, ts: job.ts,
                    frac: k / n, interpMs: S.ms, halfRate, cfg: job.key + (n > 2 ? ' ×' + n : '') },
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
  if (m.type === 'frame') {
    const capKey = activeKey; // controllerTick may switch mid-handler; this frame's pixels belong to capKey's dims
    const S = sessions.get(capKey);
    const { off, ctx } = ctxFor(S.W, S.H);
    ctx.drawImage(m.bmp, 0, 0, S.W, S.H);
    m.bmp.close();
    const rgba = ctx.getImageData(0, 0, S.W, S.H).data;
    const prevFrame = last; // snapshot: controllerTick() may null the global mid-handler
    if (prevFrame && lastKey === capKey) {
      const dup = animeMode && isNearDup(prevFrame, rgba);
      if (dup) {
        postMessage({ type: 'dup', ts: m.ts });
      } else {
        const du = m.ts - lastUniqueTs;
        if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
        lastUniqueTs = m.ts;
        transitionNo++;
        controllerTick();
        const SS = sessions.get(capKey);
        // the factor is a CEILING, not a promise: shrink it until the mids fit the live
        // budget — a steady 4x beats a stuttering 6x. Below 2x fall back to half-rate.
        // Margin is >1: the 100ms pipeline delay absorbs transient overruns, so demand
        // only sustained throughput, not per-interval slack (0.95 cost a whole rung:
        // 48fps when 72 was sustainable).
        let effN = SS.kind === 'rt' ? factor : 2;
        const ms = SS.ms || 60;
        while (effN > 2 && (effN - 1) * ms > uniqueIntervalMs * 1.1) effN--;
        halfRate = (effN - 1) * ms > uniqueIntervalMs * 1.05;
        const skip = halfRate && (transitionNo & 1);
        if (interpOn && !skip) {
          // the job runs on capKey — the session whose dims match these pixels
          const job = { a: prevFrame, b: rgba, ts: m.ts, key: capKey, n: effN };
          if (!busy) runPair(job);
          else { if (pending) postMessage({ type: 'skipped' }); pending = job; }
        }
      }
    } else {
      lastUniqueTs = m.ts;
    }
    last = rgba; lastKey = capKey;
    return;
  }
};
