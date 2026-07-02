// Framecast content script: real-time frame interpolation for any <video> on the page.
// GPU-resident pipeline (own WebGPU runtime, weights bundled): video -> texture ->
// interpolation -> overlay canvas (sibling of the video; site controls stay on top).
// DRM (EME) video produces black frames — nothing any extension can do about that.
(() => {
  'use strict';
  if (window.__framecast) return;
  window.__framecast = true;

  const DELAY_MS = 60;
  const SIZES = { 360: [640, 352], 480: [848, 480], 720: [1280, 720] };

  // ---------- settings (chrome.storage.local, live-applied) ----------
  const cfg = { factor: 4, anime: true, debug: false, res: 480, hoverReveal: true };
  try {
    chrome.storage.local.get(cfg, v => Object.assign(cfg, v));
  } catch { /* storage unavailable in some frames */ }
  function saveCfg() {
    try { chrome.storage.local.set(cfg); } catch {}
  }

  let rt = null, rtRes = 0, device = null, videoEl = null;
  let overlay = null, overlayCtx = null, blitPipe = null, blitSampler = null;
  const blitBg = new Map();
  let frameTex = [], frameIdx = 0, texW = 0, texH = 0, lastTex = null;
  let midTexs = [], midIdx = 0;
  let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupRead = null, dedupSampler = null;
  let queue = [], running = false, busy = false, pending = null, processingFrame = false;
  let intervalMs = 42, uniqueIntervalMs = 42, lastArrival = 0, lastUniqueTs = 0;
  let msAvg = 0, shown = 0, dropped = 0, dups = 0, cuts = 0, fpsWin = [], effN = 2, lastStat = null;
  let btn = null, gear = null, hud = null, panel = null, statsTimer = 0;
  const sys = { gpu: '—', f16: false };

  const log = (...a) => console.log('[framecast]', ...a);

  // ---------- device / runtime ----------
  async function ensureRuntime() {
    if (!device) {
      if (!navigator.gpu) throw new Error('WebGPU недоступен');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('нет GPU-адаптера');
      const f16 = adapter.features.has('shader-f16');
      device = await adapter.requestDevice({ requiredFeatures: f16 ? ['shader-f16'] : [] });
      sys.f16 = f16;
      const inf = adapter.info || {};
      sys.gpu = inf.description || [inf.vendor, inf.architecture].filter(Boolean).join(' ') || 'неизвестный GPU';
    }
    if (rt && rtRes === cfg.res) return;
    const url = (p) => chrome.runtime.getURL(p);
    const [bin, man] = await Promise.all([
      fetch(url('assets/rt_slim.bin')).then(r => r.arrayBuffer()),
      fetch(url('assets/rt_slim.json')).then(r => r.json())]);
    const { createRT } = await import(url('rt/rt.js'));
    const [mw, mh] = SIZES[cfg.res];
    rt = await createRT(device, { w: mw, h: mh, textureInput: true, textureOutput: true,
      weightsBin: bin, weightsManifest: man });
    rtRes = cfg.res;
    midTexs.forEach(t => t.destroy());
    midTexs = [];
    for (let i = 0; i < 12; i++) {
      midTexs.push(device.createTexture({ label: 'fcmid' + i, size: [mw, mh],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    }
    blitBg.clear();
    log('runtime up @', cfg.res);
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
    lastStat = { mean, max: s[1] };
    return { dup: mean < 2.5 && s[1] < 45, cut: mean > 90, black: s[1] === 0 };
  }

  // ---------- overlay presentation ----------
  function ensureOverlay() {
    if (overlay) return;
    overlay = document.createElement('canvas');
    // a SIBLING of the video with a modest z-index: above the video, below the controls
    overlay.style.cssText = 'position:absolute; pointer-events:none; z-index:2; transition:clip-path .15s;';
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
    if (overlay.parentElement !== videoEl.parentElement) {
      videoEl.parentElement.insertBefore(overlay, videoEl.nextSibling);
    }
    const uiHost = document.fullscreenElement || document.body;
    if (btn && btn.parentElement !== uiHost) {
      uiHost.appendChild(btn); uiHost.appendChild(gear); uiHost.appendChild(hud); uiHost.appendChild(panel);
    }
    const r = videoEl.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;
    // self-calibrating placement: measure where the overlay actually landed and nudge
    // by the delta — immune to whatever containing block/margins the site uses
    const cur = overlay.getBoundingClientRect();
    const dx = r.left - cur.left, dy = r.top - cur.top;
    if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
      overlay.style.left = ((parseFloat(overlay.style.left) || 0) + dx) + 'px';
      overlay.style.top = ((parseFloat(overlay.style.top) || 0) + dy) + 'px';
    }
    overlay.style.width = r.width + 'px';
    overlay.style.height = r.height + 'px';
    overlay.style.outline = cfg.debug ? '3px solid #19c37d' : 'none';
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
  // hover-reveal: native controls render INSIDE the video element — no z-index can
  // lift them above the overlay. When the mouse is near the BOTTOM of the video we
  // clip only the controls strip out of the overlay: the bar shows through while
  // the rest of the frame keeps playing interpolated.
  let revealUntil = 0;
  document.addEventListener('mousemove', (e) => {
    if (!running || !cfg.hoverReveal) return;
    const r = videoEl.getBoundingClientRect();
    const zone = Math.min(160, r.height * 0.35);
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.bottom - zone && e.clientY <= r.bottom) {
      revealUntil = performance.now() + 1500;
    }
  }, { passive: true });

  function pump(now) {
    if (!running) return;
    positionOverlay();
    { // controls strip cutout + keep the HUD pinned to the video's top-right corner
      const vr = videoEl.getBoundingClientRect();
      const strip = Math.max(56, Math.min(110, vr.height * 0.16));
      overlay.style.clipPath = (cfg.hoverReveal && now < revealUntil) ? `inset(0 0 ${strip}px 0)` : 'none';
      hud.style.left = Math.max(0, vr.right - hud.offsetWidth - 10) + 'px';
      hud.style.top = Math.max(0, vr.top + 10) + 'px';
    }
    queue.sort((a, b) => a.at - b.at);
    let due = -1;
    for (let i = 0; i < queue.length; i++) if (queue[i].at <= now) due = i;
    if (due >= 0) {
      dropped += due;
      present(queue[due].tex);
      queue = queue.slice(due + 1);
    }
    if (now - statsTimer > 400) {
      statsTimer = now;
      const srcFps = intervalMs > 1 ? (1000 / intervalMs) : 0;
      if (cfg.debug) {
        const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
        hud.textContent =
          `видео: ${videoEl.videoWidth}x${videoEl.videoHeight} @ ${srcFps.toFixed(1)}fps\n` +
          `выход: ${fpsWin.length}fps · факт. множитель ×${effN}\n` +
          `вставка: ${msAvg.toFixed(1)}ms @ ${cfg.res}p · бюджет ${uniqueIntervalMs.toFixed(0)}ms\n` +
          `GPU: ${sys.gpu} · загрузка ~${load.toFixed(0)}%\n` +
          `shown ${shown} · drop ${dropped} · dups ${dups} · cuts ${cuts}\n` +
          `diff: mean ${lastStat ? lastStat.mean.toFixed(1) : '—'} max ${lastStat ? lastStat.max : '—'}` +
          `${lastStat && lastStat.max === 0 ? ' (ЧЁРНОЕ — DRM?)' : ''}`;
      } else {
        hud.textContent = `FC ${fpsWin.length}fps ×${effN} · ${msAvg.toFixed(0)}ms`;
      }
      if (panel && panel.style.display === 'block') updateStatus();
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
        const { dup, cut } = await classifyPair(prev, tex);
        if (cut) { cuts++; lastUniqueTs = arrival; }
        else if (cfg.anime && dup) { dups++; }
        else {
          const du = arrival - lastUniqueTs;
          if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
          lastUniqueTs = arrival;
          let n = cfg.factor;
          const ms = msAvg || 10;
          while (n > 2 && (n - 1) * ms > uniqueIntervalMs * 1.1) n--;
          effN = n;
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
      hud.style.display = 'block';
      hud.textContent = 'FC ошибка: ' + (e.message || e);
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
    shown = 0; dropped = 0; dups = 0; cuts = 0;
    running = true;
    hud.style.display = 'block';
    videoEl.requestVideoFrameCallback(onFrame);
    requestAnimationFrame(pump);
    btn.textContent = 'FC ✓';
    btn.style.background = '#1a7f37';
  }
  function stop() {
    running = false;
    if (overlay) overlay.style.display = 'none';
    hud.style.display = 'none';
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

  // live system/status readout in the settings panel: adapter, f16, fps, cost,
  // our estimated GPU load (interp time vs the per-unique-frame budget), VRAM
  function updateStatus() {
    const st = panel && panel.querySelector('#fcStatus');
    if (!st) return;
    const lines = [`GPU: ${sys.gpu}`,
      `f16: ${sys.f16 ? 'да' : 'НЕТ (медленный путь)'} · модель: rt_slim`,
      `статус: ${running ? 'работает' : 'остановлен'}`];
    if (running) {
      const [mw, mh] = SIZES[cfg.res];
      const vramMB = (texW * texH * 4 * frameTex.length + mw * mh * 4 * midTexs.length) / 1048576;
      const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
      lines.push(
        `выход: ${fpsWin.length}fps · множитель ×${effN}`,
        `вставка: ${msAvg.toFixed(1)}ms @ ${cfg.res}p`,
        `нагрузка GPU (наша, оценка): ~${load.toFixed(0)}%`,
        `VRAM текстуры: ~${vramMB.toFixed(0)}MB · очередь ${queue.length}`);
    }
    st.textContent = lines.join('\n');
  }

  function buildPanel() {
    panel = document.createElement('div');
    panel.style.cssText = 'position:fixed; right:14px; bottom:92px; z-index:2147483647;'
      + 'background:#1c1c1c; color:#ddd; border:1px solid #555; border-radius:10px;'
      + 'padding:10px 12px; font:13px/1.9 monospace; display:none; min-width:230px;';
    panel.innerHTML = `
      <div><b>Framecast</b></div>
      <label>множитель (потолок)
        <select id="fcFactor">
          <option value="2">2×</option><option value="3">3×</option>
          <option value="4">4×</option><option value="6">6×</option>
        </select></label><br>
      <label>вставки
        <select id="fcRes">
          <option value="360">360p (быстрее)</option>
          <option value="480">480p</option>
          <option value="720">720p (тяжелее)</option>
        </select></label><br>
      <label><input type="checkbox" id="fcAnime"> аниме-дедуп «двоек»</label><br>
      <label><input type="checkbox" id="fcHover"> контролы плеера при наведении</label><br>
      <label><input type="checkbox" id="fcDebug"> debug (рамка + телеметрия)</label>
      <hr style="border:none;border-top:1px solid #444;margin:8px 0">
      <div id="fcStatus" style="font:11px/1.6 monospace;color:#9c9;white-space:pre">—</div>`;
    document.body.appendChild(panel);
    const F = panel.querySelector('#fcFactor'), R = panel.querySelector('#fcRes');
    const A = panel.querySelector('#fcAnime'), D = panel.querySelector('#fcDebug');
    const Hv = panel.querySelector('#fcHover');
    F.value = String(cfg.factor); R.value = String(cfg.res);
    A.checked = cfg.anime; D.checked = cfg.debug; Hv.checked = cfg.hoverReveal;
    F.onchange = () => { cfg.factor = +F.value; saveCfg(); };
    A.onchange = () => { cfg.anime = A.checked; saveCfg(); };
    D.onchange = () => { cfg.debug = D.checked; saveCfg(); };
    Hv.onchange = () => { cfg.hoverReveal = Hv.checked; saveCfg(); };
    R.onchange = async () => {
      cfg.res = +R.value; saveCfg();
      if (running) { const v = videoEl; stop(); await start(v); } // rebuild the runtime
    };
  }

  function injectUI() {
    btn = document.createElement('button');
    btn.textContent = 'FC ×2';
    btn.style.cssText = 'position:fixed; right:14px; bottom:14px; z-index:2147483647;'
      + 'background:#333; color:#fff; border:1px solid #666; border-radius:8px;'
      + 'padding:6px 12px; font:13px monospace; cursor:pointer; opacity:.85;';
    gear = document.createElement('button');
    gear.textContent = '⚙';
    gear.style.cssText = btn.style.cssText + 'right:78px;';
    hud = document.createElement('div');
    // anchored to the video's top-right corner every pump tick (inside the player)
    hud.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'color:#0f0; font:11px/1.5 monospace; background:rgba(0,0,0,.7); padding:4px 8px;'
      + 'border-radius:6px; white-space:pre; text-align:left; pointer-events:none; display:none;';
    buildPanel();
    btn.onclick = async () => {
      if (running) { stop(); return; }
      const v = biggestVideo();
      if (!v) { hud.style.display = 'block'; hud.textContent = 'FC: видео не найдено'; return; }
      try { await start(v); } catch (e) { hud.style.display = 'block'; hud.textContent = 'FC ошибка: ' + (e.message || e); log(e); }
    };
    gear.onclick = () => {
      panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
      if (panel.style.display === 'block') updateStatus();
    };
    document.body.appendChild(btn);
    document.body.appendChild(gear);
    document.body.appendChild(hud);
  }

  const boot = () => {
    if (btn) return;
    if (document.querySelector('video')) injectUI();
  };
  boot();
  new MutationObserver(boot).observe(document.documentElement, { childList: true, subtree: true });
})();
