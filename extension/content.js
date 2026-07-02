// Framecast content script: real-time frame interpolation for any <video> on the page.
// The whole pipeline is GPU-resident (own WebGPU runtime, weights bundled): video ->
// texture -> interpolation -> overlay canvas. The video keeps playing underneath
// (native audio); the overlay covers it with originals + mids on our own clock.
// DRM (EME) video produces black frames — nothing any extension can do about that.
(() => {
  'use strict';
  if (window.__framecast) return;
  window.__framecast = true;

  const DELAY_MS = 60;   // pipeline delay; audio leads by this much (below lipsync threshold)
  const MODEL_W = 848, MODEL_H = 480; // rt@480 slim — 5ms mids, plenty for 24-30fps sources
  const MAX_FACTOR = 4;

  let rt = null, device = null, videoEl = null;
  let overlay = null, overlayCtx = null, blitPipe = null, blitSampler = null;
  const blitBg = new Map();
  let frameTex = [], frameIdx = 0, texW = 0, texH = 0, lastTex = null;
  let midTexs = [], midIdx = 0;
  let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupRead = null, dedupSampler = null;
  let queue = [], running = false, busy = false, pending = null, processingFrame = false;
  let intervalMs = 42, uniqueIntervalMs = 42, lastArrival = 0, lastUniqueTs = 0;
  let msAvg = 0, shown = 0, dropped = 0, dups = 0, cuts = 0, fpsWin = [];
  let btn = null, hud = null, statsTimer = 0;

  const log = (...a) => console.log('[framecast]', ...a);

  // ---------- device / runtime ----------
  async function ensureRuntime() {
    if (rt) return;
    if (!navigator.gpu) throw new Error('WebGPU недоступен на этой странице');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('нет GPU-адаптера');
    const f16 = adapter.features.has('shader-f16');
    device = await adapter.requestDevice({ requiredFeatures: f16 ? ['shader-f16'] : [] });
    const url = (p) => chrome.runtime.getURL(p);
    const [bin, man] = await Promise.all([
      fetch(url('assets/rt_slim.bin')).then(r => r.arrayBuffer()),
      fetch(url('assets/rt_slim.json')).then(r => r.json())]);
    const { createRT } = await import(url('rt/rt.js'));
    rt = await createRT(device, { w: MODEL_W, h: MODEL_H, textureInput: true, textureOutput: true,
      weightsBin: bin, weightsManifest: man });
    for (let i = 0; i < 12; i++) {
      midTexs.push(device.createTexture({ label: 'fcmid' + i, size: [MODEL_W, MODEL_H],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    }
    log('runtime up (f16:', f16, ')');
  }

  function ensureFrameTextures(w, h) {
    if (texW === w && texH === h && frameTex.length === 8) return;
    frameTex.forEach(t => t.destroy());
    frameTex = [];
    for (let i = 0; i < 8; i++) {
      frameTex.push(device.createTexture({ label: 'fcfr' + i, size: [w, h], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }));
    }
    texW = w; texH = h; dedupBg.clear(); blitBg.clear(); lastTex = null;
  }

  // ---------- dedup / cut (GPU, 8-byte readback) ----------
  function ensureDedup() {
    if (dedupPipe) return;
    dedupSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    dedupStats = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    dedupRead = device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    dedupPipe = device.createComputePipeline({ layout: 'auto', compute: {
      module: device.createShaderModule({ code: `
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
  async function classifyPair(ta, tb) {
    ensureDedup();
    const key = ta.label + '|' + tb.label;
    if (!dedupBg.has(key)) {
      dedupBg.set(key, device.createBindGroup({ layout: dedupPipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() },
        { binding: 2, resource: dedupSampler }, { binding: 3, resource: { buffer: dedupStats } }] }));
    }
    device.queue.writeBuffer(dedupStats, 0, new Uint32Array([0, 0]));
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(dedupPipe); pass.setBindGroup(0, dedupBg.get(key));
    pass.dispatchWorkgroups(6, 4);
    pass.end();
    enc.copyBufferToBuffer(dedupStats, 0, dedupRead, 0, 8);
    device.queue.submit([enc.finish()]);
    await dedupRead.mapAsync(GPUMapMode.READ);
    const s = new Uint32Array(dedupRead.getMappedRange().slice(0));
    dedupRead.unmap();
    const mean = s[0] / (48 * 27);
    return { dup: mean < 2.5 && s[1] < 45, cut: mean > 90, black: s[1] === 0 && mean === 0 };
  }

  // ---------- overlay presentation ----------
  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement('canvas');
    overlay.style.cssText = 'position:absolute; pointer-events:none; z-index:2147483000;';
    document.body.appendChild(overlay);
    overlayCtx = overlay.getContext('webgpu');
    overlayCtx.configure({ device, format: 'rgba8unorm', alphaMode: 'opaque' });
    blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    const mod = device.createShaderModule({ code: `
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
    blitPipe = device.createRenderPipeline({ layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] } });
  }
  function positionOverlay() {
    const r = videoEl.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;
    overlay.style.left = (r.left + scrollX) + 'px';
    overlay.style.top = (r.top + scrollY) + 'px';
    overlay.style.width = r.width + 'px';
    overlay.style.height = r.height + 'px';
    const bw = Math.round(Math.min(r.width * devicePixelRatio, 1920));
    const bh = Math.round(Math.min(r.height * devicePixelRatio, 1080));
    if (overlay.width !== bw || overlay.height !== bh) { overlay.width = bw; overlay.height = bh; }
  }
  function present(tex) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: overlayCtx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(blitPipe);
    if (!blitBg.has(tex)) {
      blitBg.set(tex, device.createBindGroup({ layout: blitPipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: blitSampler }] }));
      if (blitBg.size > 48) blitBg.clear();
    }
    pass.setBindGroup(0, blitBg.get(tex));
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
    shown++;
    const now = performance.now();
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1000) fpsWin.shift();
  }
  function pump(now) {
    if (!running) return;
    positionOverlay();
    queue.sort((a, b) => a.at - b.at);
    let due = -1;
    for (let i = 0; i < queue.length; i++) if (queue[i].at <= now) due = i;
    if (due >= 0) {
      dropped += due;
      present(queue[due].tex);
      queue = queue.slice(due + 1);
    }
    if (now - statsTimer > 500) {
      statsTimer = now;
      hud.textContent = `FC ${fpsWin.length}fps · ${msAvg.toFixed(0)}ms · d${dropped}`;
    }
    requestAnimationFrame(pump);
  }

  // ---------- interpolation ----------
  async function runPair(job) {
    busy = true;
    try {
      const ts = [];
      for (let k = 1; k < job.n; k++) ts.push(k / job.n);
      const outs = [];
      for (let k = 0; k < ts.length; k++) {
        outs.push(midTexs[midIdx]);
        midIdx = (midIdx + 1) % midTexs.length;
      }
      const t0 = performance.now();
      await rt.runMulti(job.a, job.b, ts, outs);
      device.queue.onSubmittedWorkDone().then(() => {
        const ms = (performance.now() - t0) / ts.length;
        msAvg = msAvg ? msAvg * 0.85 + ms * 0.15 : ms;
      });
      for (let k = 0; k < ts.length; k++) {
        queue.push({ tex: outs[k], at: job.at + ts[k] * intervalMs });
      }
    } catch (e) {
      log('interp error', e);
    }
    busy = false;
    if (pending) { const p = pending; pending = null; runPair(p); }
  }

  async function onFrame() {
    if (!running) return;
    videoEl.requestVideoFrameCallback(onFrame);
    if (processingFrame) return;
    processingFrame = true;
    try {
      const arrival = performance.now();
      const dt = arrival - lastArrival;
      if (dt > 5 && dt < 500) intervalMs = intervalMs * 0.9 + dt * 0.1;
      lastArrival = arrival;
      const vw = Math.min(videoEl.videoWidth, 1920), vh = Math.min(videoEl.videoHeight, 1080);
      if (!vw || !vh) return;
      ensureFrameTextures(vw, vh);
      const tex = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: tex }, [vw, vh]);
      queue.push({ tex, at: arrival + DELAY_MS });
      const prev = lastTex;
      if (prev) {
        const { dup, cut, black } = await classifyPair(prev, tex);
        if (black) { /* DRM or covered — keep showing, do nothing */ }
        if (cut) { cuts++; lastUniqueTs = arrival; }
        else if (dup) { dups++; }
        else {
          const du = arrival - lastUniqueTs;
          if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
          lastUniqueTs = arrival;
          let n = MAX_FACTOR;
          const ms = msAvg || 10;
          while (n > 2 && (n - 1) * ms > uniqueIntervalMs * 1.1) n--;
          if ((n - 1) * ms <= uniqueIntervalMs * 1.05) {
            const job = { a: prev, b: tex, at: arrival - intervalMs + DELAY_MS, n };
            if (!busy) runPair(job);
            else pending = job;
          }
        }
      } else {
        lastUniqueTs = arrival;
      }
      lastTex = tex;
    } catch (e) {
      log('frame error', e);
      stop();
      hud.textContent = 'FC: ' + (e.message || e);
    } finally {
      processingFrame = false;
    }
  }

  // ---------- lifecycle / UI ----------
  async function start(v) {
    videoEl = v;
    await ensureRuntime();
    ensureOverlay();
    positionOverlay();
    overlay.style.display = 'block';
    queue = []; lastTex = null; pending = null;
    running = true;
    videoEl.requestVideoFrameCallback(onFrame);
    requestAnimationFrame(pump);
    btn.textContent = 'FC ✓';
    btn.style.background = '#1a7f37';
  }
  function stop() {
    running = false;
    if (overlay) overlay.style.display = 'none';
    queue = []; lastTex = null; pending = null;
    btn.textContent = 'FC ×2';
    btn.style.background = '#333';
  }

  function biggestVideo() {
    let best = null, area = 0;
    for (const v of document.querySelectorAll('video')) {
      const r = v.getBoundingClientRect();
      if (r.width * r.height > area && v.readyState >= 2) { area = r.width * r.height; best = v; }
    }
    return best;
  }

  function injectUI() {
    btn = document.createElement('button');
    btn.textContent = 'FC ×2';
    btn.style.cssText = 'position:fixed; right:14px; bottom:14px; z-index:2147483647;'
      + 'background:#333; color:#fff; border:1px solid #666; border-radius:8px;'
      + 'padding:6px 12px; font:13px monospace; cursor:pointer; opacity:.85;';
    hud = document.createElement('div');
    hud.style.cssText = 'position:fixed; right:14px; bottom:50px; z-index:2147483647;'
      + 'color:#0f0; font:11px monospace; background:rgba(0,0,0,.6); padding:3px 6px; border-radius:6px;';
    btn.onclick = async () => {
      if (running) { stop(); return; }
      const v = biggestVideo();
      if (!v) { hud.textContent = 'FC: видео не найдено'; return; }
      try { await start(v); } catch (e) { hud.textContent = 'FC: ' + (e.message || e); log(e); }
    };
    document.body.appendChild(btn);
    document.body.appendChild(hud);
  }

  // only bother on pages that ever get a <video>
  const boot = () => {
    if (btn) return;
    if (document.querySelector('video')) injectUI();
  };
  boot();
  new MutationObserver(boot).observe(document.documentElement, { childList: true, subtree: true });
})();
