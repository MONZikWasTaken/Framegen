// Framecast content script: real-time frame interpolation for any <video> on the page.
// GPU-resident pipeline (own WebGPU runtime, weights bundled): video -> texture ->
// interpolation -> overlay canvas (sibling of the video; site controls stay on top).
// DRM (EME) video produces black frames — nothing any extension can do about that.
(() => {
  'use strict';
  if (window.__framecast) return;
  window.__framecast = true;

  const DELAY_MS = 60;
  // runtime tiles are 16x16 — model dims must be /16 (1088, not 1080; the ~0.7%
  // vertical stretch at present time is invisible)
  const SIZES = { 360: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1088] };

  // ---------- settings (chrome.storage.local, live-applied) ----------
  // factor: 'auto' (smart, self-capped) or a FIXED 2..6 that is never lowered
  const cfg = { factor: 'auto', anime: true, debug: false, res: 480, hoverReveal: true, compare: false,
    fg: true, sr: false, hdr: false };
  function sanitizeCfg() {
    if (cfg.factor !== 'auto' && ![2, 3, 4, 5, 6].includes(cfg.factor)) cfg.factor = 'auto';
    if (!SIZES[cfg.res]) cfg.res = 480;
    cfg.anime = !!cfg.anime; cfg.debug = !!cfg.debug;
    cfg.hoverReveal = !!cfg.hoverReveal; cfg.compare = !!cfg.compare;
    cfg.fg = !!cfg.fg; cfg.sr = !!cfg.sr; cfg.hdr = !!cfg.hdr;
  }
  try {
    // async: the panel may already be built with defaults by the time this lands —
    // ALWAYS resync the UI, otherwise checkboxes show one thing and cfg does another
    chrome.storage.local.get(cfg, v => { Object.assign(cfg, v); sanitizeCfg(); syncPanel(); });
    // settings changed in another tab/frame apply here live
    chrome.storage.onChanged.addListener((ch, area) => {
      if (area !== 'local') return;
      let resChanged = false;
      for (const k in ch) {
        if (!(k in cfg)) continue;
        if (k === 'res' && cfg.res !== ch[k].newValue) resChanged = true;
        cfg[k] = ch[k].newValue;
      }
      sanitizeCfg(); syncPanel();
      if ('hdr' in ch) configureOverlay();
      if (resChanged && running && videoEl && !toggling) {
        toggling = true;
        switchRes().catch(e => log('res sync', e)).finally(() => { toggling = false; });
      }
    });
  } catch { /* storage unavailable in some frames */ }
  function saveCfg() {
    try { chrome.storage.local.set(cfg); } catch {}
  }
  function syncPanel() {
    if (!panel) return;
    panel.querySelector('#fcFactor').value = String(cfg.factor);
    panel.querySelector('#fcRes').value = String(cfg.res);
    panel.querySelector('#fcAnime').checked = cfg.anime;
    panel.querySelector('#fcDebug').checked = cfg.debug;
    panel.querySelector('#fcHover').checked = cfg.hoverReveal;
    panel.querySelector('#fcCompare').checked = cfg.compare;
    panel.querySelector('#fcFG').checked = cfg.fg;
    panel.querySelector('#fcSR').checked = cfg.sr;
    const hd = panel.querySelector('#fcHDR');
    hd.checked = cfg.hdr;
    if (!sys.hdrOk) { hd.disabled = true; hd.style.opacity = '.35'; }
  }

  let rt = null, rtRes = 0, device = null, videoEl = null;
  let overlay = null, overlayCtx = null, blitPipe = null, blitSampler = null;
  const blitBg = new Map();
  let frameTex = [], frameIdx = 0, texW = 0, texH = 0, lastTex = null;
  let midTexs = [], midIdx = 0;
  let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupRead = null, dedupSampler = null;
  let queue = [], running = false, processingFrame = false;
  let intervalMs = 42, uniqueIntervalMs = 42, lastArrival = 0, lastUniqueTs = 0;
  let msAvg = 0, shown = 0, dropped = 0, dups = 0, cuts = 0, fpsWin = [], effN = 2, lastStat = null;
  let btn = null, gear = null, hud = null, panel = null, statsTimer = 0;
  let bar = null, barSeeking = false;
  let rafMs = 0, lastPumpT = 0, warnEl = null, overSince = 0;
  let splitEl = null, splitX = 0.5, toggling = false, autoSkipT = 0;
  let delayMs = DELAY_MS, dropWin = [], switching = false, preloadFailT = -1e9;
  let autoPenalty = 0, penaltyT = 0, dropPressure = 0, lastPressureT = 0;
  const sys = { gpu: '—', f16: false, hdrOk: false, hdrOn: false };
  try { sys.hdrOk = !!(window.matchMedia && matchMedia('(dynamic-range: high)').matches); } catch {}

  const log = (...a) => console.log('[framecast]', ...a);

  // ---------- device / runtime ----------
  // memoized: the hover-preload and the FC click may race — only one build runs
  let rtBuilding = null;
  async function ensureRuntime() {
    while (rtBuilding) await rtBuilding;
    if (device && rt && rtRes === cfg.res) return;
    rtBuilding = buildRuntime();
    try { await rtBuilding; } finally { rtBuilding = null; }
  }
  async function buildRuntime() {
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
    for (let i = 0; i < 24; i++) {
      midTexs.push(device.createTexture({ label: 'fcmid' + i, size: [mw, mh],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
    }
    blitBg.clear();
    log('runtime up @', cfg.res);
  }

  function ensureFrameTextures(w, h) {
    if (texW === w && texH === h && frameTex.length === 12) return;
    frameTex.forEach(t => t.destroy());
    frameTex = [];
    for (let i = 0; i < 12; i++) {
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
    overlay.style.cssText = 'position:absolute; pointer-events:none; z-index:2;'
      + 'opacity:0; transition:opacity .25s;';
    overlayCtx = overlay.getContext('webgpu');
    blitSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    // on native-controls players the overlay OWNS the pointer (pointer-events:auto
    // set in start): dblclick must never reach the native video, whose shadow-DOM
    // handler fullscreens the bare <video> where our canvas cannot exist
    overlay.addEventListener('click', () => {
      if (!videoEl || !videoEl.controls) return;
      if (videoEl.paused) videoEl.play().catch(() => {}); else videoEl.pause();
      flashCenter(svgIcon(videoEl.paused ? 'pause' : 'play', 30));
      updateBar();
    });
    overlay.addEventListener('dblclick', (e) => {
      e.preventDefault();
      if (videoEl && videoEl.controls) toggleFullscreen();
    });
    configureOverlay();
  }

  const BLIT_VS = `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4(p[i], 0.0, 1.0);
  o.uv = vec2(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}`;
  // (re)build the present path: SDR passthrough, or HDR via inverse tone mapping —
  // highlights expand past SDR white on an fp16 canvas in extended tone-mapping mode
  // (same idea as RTX Video HDR; the browser only ever hands us tonemapped SDR)
  function configureOverlay() {
    if (!overlayCtx || !device) return;
    let hdr = !!(cfg.hdr && sys.hdrOk);
    const fmt = hdr ? 'rgba16float' : 'rgba8unorm';
    try {
      overlayCtx.configure({ device, format: fmt, alphaMode: 'opaque',
        ...(hdr ? { colorSpace: 'srgb', toneMapping: { mode: 'extended' } } : {}) });
    } catch (e) {
      log('hdr configure failed, falling back to SDR', e);
      hdr = false;
      overlayCtx.configure({ device, format: 'rgba8unorm', alphaMode: 'opaque' });
    }
    const fs = hdr ? `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  let c = textureSampleLevel(tex, samp, v.uv, 0.0).rgb;
  let lin = pow(max(c, vec3(0.0)), vec3(2.2));
  let y = max(lin.r, max(lin.g, lin.b));
  let t = smoothstep(0.35, 1.0, y);
  let gain = 1.0 + 2.2 * t * t;          // shadows/midtones untouched, peaks ~3.2x SDR white
  return vec4(pow(lin * gain, vec3(1.0 / 2.2)), 1.0);
}` : `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, v.uv, 0.0);
}`;
    const mod = device.createShaderModule({ code: BLIT_VS + fs });
    blitPipe = device.createRenderPipeline({ layout: 'auto',
      vertex: { module: mod, entryPoint: 'vs' },
      fragment: { module: mod, entryPoint: 'fs', targets: [{ format: hdr ? 'rgba16float' : 'rgba8unorm' }] } });
    blitBg.clear(); // bind groups belong to the old pipeline layout
    sys.hdrOn = hdr;
  }
  function positionOverlay() {
    if (overlay.parentElement !== videoEl.parentElement) {
      videoEl.parentElement.insertBefore(overlay, videoEl.nextSibling);
    }
    const uiHost = document.fullscreenElement || document.body;
    if (btn && btn.parentElement !== uiHost) {
      uiHost.appendChild(btn); uiHost.appendChild(gear); uiHost.appendChild(hud); uiHost.appendChild(panel);
      if (bar) uiHost.appendChild(bar);
      if (splitEl) uiHost.appendChild(splitEl);
      if (warnEl) uiHost.appendChild(warnEl);
      if (flashEl) uiHost.appendChild(flashEl);
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
  // ---------- TinySR 2x upscale on the present path ----------
  let sr = null, srBuilding = null;
  const srOut = new Map();
  async function ensureSR() {
    if (sr || !sys.f16 || !device) return; // SR shaders need shader-f16
    if (srBuilding) { await srBuilding; return; }
    srBuilding = (async () => {
      const url = (p) => chrome.runtime.getURL(p);
      const [bin, man] = await Promise.all([
        fetch(url('assets/rt_sr.bin')).then(r => r.arrayBuffer()),
        fetch(url('assets/rt_sr.json')).then(r => r.json())]);
      const { createSR } = await import(url('rt/sr.js'));
      sr = await createSR(device, { weightsBin: bin, weightsManifest: man });
      log('SR up');
    })();
    try { await srBuilding; } finally { srBuilding = null; }
  }

  function present(tex, isMid) {
    // interpolated frames are model-res and look soft next to native source frames;
    // run them through TinySR (2x) when it actually adds pixels toward the canvas.
    // With FG off, SR applies to the source frames instead (pure-upscaler mode).
    if (cfg.sr && (isMid || !cfg.fg)) {
      if (!sr) { ensureSR().catch(e => log('sr', e)); }
      else if (tex.width < overlay.width) {
        const key = tex.width + 'x' + tex.height;
        let out = srOut.get(key);
        if (!out) {
          out = device.createTexture({ label: 'fcsr' + key,
            size: [tex.width * 2, tex.height * 2], format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
          srOut.set(key, out);
          if (srOut.size > 4) {
            for (const [k, t] of srOut) if (k !== key) { t.destroy(); srOut.delete(k); }
            blitBg.clear();
          }
        }
        sr.process(tex, out, tex.width, tex.height);
        tex = out;
      }
    }
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
    if (overlay.style.opacity !== '1') overlay.style.opacity = '1'; // reveal only once pixels exist
    shown++;
    const now = performance.now();
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1000) fpsWin.shift();
  }
  // our own control bar ON TOP of the overlay: native controls render INSIDE the
  // video element and can never show above the canvas, so instead of ever revealing
  // the raw video we drive the <video> ourselves — play/seek/volume/fullscreen as
  // regular DOM above everything. Interpolation is never interrupted.
  let revealUntil = 0, uiVideo = null, uiScan = 0;
  document.addEventListener('mousemove', (e) => {
    const now = performance.now();
    if (!running && now - uiScan > 300) { uiScan = now; uiVideo = biggestVideo(); }
    const v = running ? videoEl : uiVideo;
    if (!v || !btn) return;
    const r = v.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
      revealUntil = now + 2000;
      placeSideButtons(r);
      // hovering a video signals intent: build the runtime NOW (weights fetch +
      // shader compilation, the expensive part) so the FC click lands instantly
      if (!rt && !rtBuilding && now - preloadFailT > 5000) {
        ensureRuntime().catch((err) => { preloadFailT = performance.now(); log('preload', err); });
      }
    }
  }, { passive: true });

  // FC + settings live INSIDE the player: centered vertically at the left edge
  function placeSideButtons(r) {
    const cy = r.top + r.height / 2;
    btn.style.display = gear.style.display = 'block';
    btn.style.left = gear.style.left = (r.left + 12) + 'px';
    btn.style.top = (cy - 42) + 'px';
    gear.style.top = (cy + 4) + 'px';
  }
  setInterval(() => {
    if (!btn) return;
    if (panel && panel.style.display === 'block') { revealUntil = performance.now() + 2000; return; }
    if (performance.now() > revealUntil) {
      btn.style.display = gear.style.display = 'none';
    }
  }, 300);

  // crisp monochrome SVG icons (Feather-style) — no emoji
  const ICONS = {
    play: '<path d="M8 5.5v13a.5.5 0 0 0 .77.42l10.2-6.5a.5.5 0 0 0 0-.84L8.77 5.08A.5.5 0 0 0 8 5.5z" fill="currentColor"/>',
    pause: '<rect x="7" y="5" width="3.4" height="14" rx="1" fill="currentColor"/><rect x="13.6" y="5" width="3.4" height="14" rx="1" fill="currentColor"/>',
    vol: '<path d="M11 5 6.5 9H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h3.5L11 19V5z" fill="currentColor"/>'
      + '<path d="M15 8.6a5 5 0 0 1 0 6.8M17.7 6a9 9 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    volX: '<path d="M11 5 6.5 9H3a1 1 0 0 0-1 1v4a1 1 0 0 0 1 1h3.5L11 19V5z" fill="currentColor"/>'
      + '<path d="m15.5 9.5 5 5m0-5-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    full: '<path d="M8.5 3.5H5a1.5 1.5 0 0 0-1.5 1.5v3.5m17 0V5A1.5 1.5 0 0 0 19 3.5h-3.5m0 17H19a1.5 1.5 0 0 0 1.5-1.5v-3.5m-17 0V19A1.5 1.5 0 0 0 5 20.5h3.5"'
      + ' fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
    gear: '<path d="M4 21v-7m0-4V3m8 18v-9m0-4V3m8 18v-5m0-4V3M1.5 14H7m2-6h6m2.5 8H21"'
      + ' fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  };
  const svgIcon = (name, size = 16) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;

  const fmt = (s) => {
    s = Math.max(0, Math.floor(s || 0));
    const m = Math.floor(s / 60), h = Math.floor(m / 60);
    return (h ? h + ':' + String(m % 60).padStart(2, '0') : m) + ':' + String(s % 60).padStart(2, '0');
  };

  function ensureBar() {
    if (bar) return;
    // floating glass pill, same family as the side buttons
    bar = document.createElement('div');
    bar.style.cssText = 'position:fixed; z-index:2147483646; display:none; align-items:center; gap:10px;'
      + 'background:rgba(15,15,15,.55); backdrop-filter:blur(10px);'
      + 'border:1px solid rgba(255,255,255,.12); border-radius:14px; padding:8px 14px;'
      + 'color:#fff; font:11px system-ui; box-sizing:border-box; user-select:none;'
      + 'box-shadow:0 4px 20px rgba(0,0,0,.4); opacity:0; transform:translateY(8px);'
      + 'transition:opacity .18s, transform .18s; pointer-events:none;';
    bar.innerHTML = `
      <button id="fcPlay" class="fc-btn">${svgIcon('play', 19)}</button>
      <span id="fcCur" style="min-width:34px; text-align:right">0:00</span>
      <input id="fcSeek" class="fc-range" type="range" min="0" max="1000" value="0" style="flex:1">
      <span id="fcDur" style="min-width:34px; color:rgba(255,255,255,.55)">0:00</span>
      <button id="fcMute" class="fc-btn">${svgIcon('vol')}</button>
      <input id="fcVol" class="fc-range" type="range" min="0" max="100" value="100" style="width:60px">
      <button id="fcFull" class="fc-btn">${svgIcon('full')}</button>`;
    document.body.appendChild(bar);
    const q = (id) => bar.querySelector(id);
    // per-button cooldowns: hammering the buttons must never wedge the player
    const guard = (ms) => { let t = 0; return () => {
      const n = performance.now(); if (n - t < ms) return false; t = n; return true; }; };
    const gPlay = guard(180), gMute = guard(120), gFull = guard(400);
    q('#fcPlay').onclick = () => {
      if (!videoEl || !gPlay()) return;
      if (videoEl.paused) videoEl.play().catch(() => {}); else videoEl.pause();
      flashCenter(svgIcon(videoEl.paused ? 'pause' : 'play', 30));
      updateBar();
    };
    q('#fcMute').onclick = () => { if (videoEl && gMute()) { videoEl.muted = !videoEl.muted; updateBar(); } };
    q('#fcVol').oninput = (e) => { if (videoEl) { videoEl.volume = e.target.value / 100; videoEl.muted = false; } };
    q('#fcSeek').addEventListener('pointerdown', () => { barSeeking = true; });
    q('#fcSeek').addEventListener('pointerup', () => { barSeeking = false; });
    q('#fcSeek').oninput = (e) => {
      if (videoEl && videoEl.duration) videoEl.currentTime = e.target.value / 1000 * videoEl.duration;
    };
    q('#fcFull').onclick = () => { if (gFull()) toggleFullscreen(); }; // async transition — no double-fire
    // keep the bar alive while the mouse is on it
    bar.addEventListener('mousemove', () => { revealUntil = performance.now() + 2000; }, { passive: true });
  }

  // big centered ▶/❚❚ splash on play/pause, fades out while scaling up
  let flashEl = null;
  function flashCenter(sym) {
    if (!videoEl) return;
    if (!flashEl) {
      flashEl = document.createElement('div');
      flashEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
        + 'color:#fff; font:600 26px system-ui; background:rgba(15,15,15,.55);'
        + 'backdrop-filter:blur(6px); border-radius:50%; width:72px; height:72px;'
        + 'display:flex; align-items:center; justify-content:center; opacity:0;';
      document.body.appendChild(flashEl);
    }
    const r = videoEl.getBoundingClientRect();
    flashEl.innerHTML = sym;
    flashEl.style.left = (r.left + r.width / 2 - 36) + 'px';
    flashEl.style.top = (r.top + r.height / 2 - 36) + 'px';
    flashEl.style.transition = 'none';
    flashEl.style.opacity = '0.95';
    flashEl.style.transform = 'scale(0.8)';
    requestAnimationFrame(() => {
      flashEl.style.transition = 'opacity .5s ease-out, transform .5s ease-out';
      flashEl.style.opacity = '0';
      flashEl.style.transform = 'scale(1.4)';
    });
  }

  // fullscreen the PARENT (so the overlay comes along) and stretch the video to
  // fill the screen — fullscreening just the container leaves the video at its
  // layout size, which looks like fullscreen "not working"
  let fsByUs = false, fsSaved = '';
  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else if (videoEl) {
      fsByUs = true;
      (videoEl.parentElement || videoEl).requestFullscreen().catch(e => { fsByUs = false; log('fullscreen', e); });
    }
  }
  document.addEventListener('fullscreenchange', () => {
    if (!videoEl || !fsByUs) return;
    if (document.fullscreenElement) {
      fsSaved = videoEl.style.cssText;
      videoEl.style.cssText += ';position:fixed;left:0;top:0;width:100vw;height:100vh;'
        + 'max-width:none;max-height:none;object-fit:contain;background:#000;z-index:1;';
    } else {
      fsByUs = false;
      videoEl.style.cssText = fsSaved;
    }
  });
  const rangeFill = (el, p, color) => {
    el.style.background = `linear-gradient(to right, ${color} ${p}%, rgba(255,255,255,.22) ${p}%)`;
  };
  let barPlayIcon = '', barMuteIcon = '';
  function updateBar() {
    if (!bar || bar.style.display === 'none' || !videoEl) return;
    const pi = videoEl.paused ? 'play' : 'pause';
    if (pi !== barPlayIcon) { barPlayIcon = pi; bar.querySelector('#fcPlay').innerHTML = svgIcon(pi, 19); }
    const mi = (videoEl.muted || videoEl.volume === 0) ? 'volX' : 'vol';
    if (mi !== barMuteIcon) { barMuteIcon = mi; bar.querySelector('#fcMute').innerHTML = svgIcon(mi); }
    const vol = bar.querySelector('#fcVol'), volP = (videoEl.muted ? 0 : videoEl.volume) * 100;
    vol.value = String(Math.round(volP));
    rangeFill(vol, volP, '#fff');
    const d = videoEl.duration || 0, c = videoEl.currentTime || 0;
    bar.querySelector('#fcCur').textContent = fmt(c);
    bar.querySelector('#fcDur').textContent = fmt(d);
    const seek = bar.querySelector('#fcSeek'), p = d ? c / d * 100 : 0;
    if (!barSeeking && d) seek.value = String(Math.round(p * 10));
    rangeFill(seek, p, '#19c37d');
  }

  // compare mode: a draggable divider — raw video shows LEFT of it (the overlay is
  // clipped away), interpolated frames play on the right
  function ensureSplit() {
    if (splitEl) return;
    splitEl = document.createElement('div');
    splitEl.style.cssText = 'position:fixed; z-index:2147483645; width:18px; margin-left:-9px;'
      + 'cursor:ew-resize; touch-action:none; display:none;';
    splitEl.innerHTML = `
      <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; margin-left:-1px;
        background:rgba(255,255,255,.85); box-shadow:0 0 10px rgba(0,0,0,.7)"></div>
      <div style="position:absolute; left:50%; top:50%; width:28px; height:28px; margin:-14px 0 0 -14px;
        border-radius:50%; background:rgba(15,15,15,.62); backdrop-filter:blur(6px);
        border:1px solid rgba(255,255,255,.3); color:#fff; font:12px system-ui;
        display:flex; align-items:center; justify-content:center">⇄</div>
      <div style="position:absolute; right:14px; top:10px; color:#fff; font:10px system-ui;
        background:rgba(15,15,15,.6); border-radius:6px; padding:2px 6px; white-space:nowrap">ориг.</div>
      <div style="position:absolute; left:14px; top:10px; color:#fff; font:10px system-ui;
        background:rgba(25,150,100,.65); border-radius:6px; padding:2px 6px">FC</div>`;
    splitEl.addEventListener('pointerdown', (e) => {
      splitEl.setPointerCapture(e.pointerId);
      const move = (ev) => {
        if (!videoEl) return;
        const r = videoEl.getBoundingClientRect();
        splitX = Math.min(0.98, Math.max(0.02, (ev.clientX - r.left) / r.width));
      };
      move(e);
      splitEl.onpointermove = move;
      splitEl.onpointerup = () => { splitEl.onpointermove = null; splitEl.onpointerup = null; };
      e.preventDefault();
    });
    (document.fullscreenElement || document.body).appendChild(splitEl);
  }

  // overload plate: fixed factors are never lowered, we just tell the user
  function ensureWarn() {
    if (warnEl) return;
    warnEl = document.createElement('div');
    warnEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
      + 'background:rgba(60,16,16,.78); backdrop-filter:blur(8px); color:#ffb4a8;'
      + 'border:1px solid rgba(255,120,100,.35); border-radius:12px; padding:8px 14px;'
      + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4);'
      + 'opacity:0; transform:translateY(-6px); transition:opacity .25s, transform .25s;';
    warnEl.textContent = '⚠ Нагрузка слишком высокая — понизьте множитель или включите «авто»';
    document.body.appendChild(warnEl);
  }
  function updateWarn(now, vr) {
    ensureWarn();
    const load = uniqueIntervalMs > 1 ? msAvg * Math.max(0, effN - 1) / uniqueIntervalMs : 0;
    // fixed factor: over budget OR visibly dropping. auto: even 2x is being skipped
    const dropRate = fpsWin.length ? (dropWin.length / 2) / fpsWin.length : 0; // drops vs shown, per sec
    const dropping = cfg.fg && dropRate > 0.12;
    const fixedOver = cfg.fg && cfg.factor !== 'auto' && (load > 1.02 || dropping);
    const autoOver = cfg.fg && cfg.factor === 'auto' && autoSkipT && now - autoSkipT < 1200;
    if ((fixedOver || autoOver) && !overSince) {
      overSince = now;
      warnEl.textContent = fixedOver
        ? '⚠ Кадры дропаются — понизьте множитель/качество или включите «авто»'
        : '⚠ GPU не успевает даже 2× — поставьте качество «экономное»';
    }
    if (!fixedOver && !autoOver && load < 0.92) overSince = 0;
    const show = overSince && now - overSince > 1500; // sustained, not a warmup blip
    warnEl.style.left = (vr.left + vr.width / 2 - warnEl.offsetWidth / 2) + 'px';
    warnEl.style.top = (vr.top + 12) + 'px';
    warnEl.style.opacity = show ? '1' : '0';
    warnEl.style.transform = show ? 'translateY(0)' : 'translateY(-6px)';
  }

  function pump(now) {
    if (!running) return;
    driveJob(now); // just-in-time mid submission
    if (lastPumpT) {
      const d = now - lastPumpT;
      // pessimist estimator: believe slowdowns fast (40%), speedups slowly (3%) —
      // auto must not re-inflate on every momentary lull
      if (d > 1 && d < 100) rafMs = rafMs ? (d > rafMs ? rafMs * 0.6 + d * 0.4 : rafMs * 0.97 + d * 0.03) : d;
    }
    lastPumpT = now;
    positionOverlay();
    { // our control bar floats above the video bottom, HUD in the top-right corner
      const vr = videoEl.getBoundingClientRect();
      // our bar only where the site relies on native controls (YouTube etc. draw
      // their own DOM controls, which already sit above the overlay)
      if (videoEl.controls && cfg.hoverReveal) {
        const showBar = now < revealUntil;
        const m = Math.max(10, Math.min(16, vr.width * 0.02));
        bar.style.display = 'flex';
        bar.style.left = (vr.left + m) + 'px';
        bar.style.width = (vr.width - 2 * m) + 'px';
        bar.style.top = (vr.bottom - bar.offsetHeight - m) + 'px';
        bar.style.opacity = showBar ? '1' : '0';
        bar.style.transform = showBar ? 'translateY(0)' : 'translateY(8px)';
        bar.style.pointerEvents = showBar ? 'auto' : 'none';
      } else {
        bar.style.display = 'none';
      }
      hud.style.left = Math.max(0, vr.right - hud.offsetWidth - 10) + 'px';
      hud.style.top = Math.max(0, vr.top + 10) + 'px';
      updateWarn(now, vr);
      if (cfg.compare) {
        ensureSplit();
        splitEl.style.display = 'block';
        splitEl.style.left = (vr.left + splitX * vr.width) + 'px';
        splitEl.style.top = vr.top + 'px';
        splitEl.style.height = vr.height + 'px';
        overlay.style.clipPath = `inset(0 0 0 ${(splitX * 100).toFixed(2)}%)`;
      } else {
        if (splitEl) splitEl.style.display = 'none';
        if (overlay.style.clipPath) overlay.style.clipPath = '';
      }
    }
    queue.sort((a, b) => a.at - b.at);
    let due = -1;
    for (let i = 0; i < queue.length; i++) if (queue[i].at <= now) due = i;
    // drop pressure: leaky integrator (tau 300ms) — a burst of drops is visible in
    // milliseconds instead of averaging out over seconds
    dropPressure *= Math.exp((lastPressureT - now) / 300);
    lastPressureT = now;
    if (due >= 0) {
      dropped += due;
      dropPressure += due;
      for (let i = 0; i < due; i++) dropWin.push(now);
      present(queue[due].tex, queue[due].mid);
      queue = queue.slice(due + 1);
    }
    while (dropWin.length && dropWin[0] < now - 2000) dropWin.shift();
    // AIMD controller, evaluated EVERY frame: aggressive decrease on pressure,
    // additive recovery after a long clean stretch
    if (cfg.factor === 'auto') {
      if (dropPressure > 1.2 && autoPenalty < 3 && now - penaltyT > 500) {
        autoPenalty = Math.min(3, autoPenalty + (dropPressure > 3 ? 2 : 1));
        penaltyT = now;
        dropPressure = 0; // consumed by the step
      } else if (autoPenalty > 0 && dropPressure < 0.15 && now - penaltyT > 6000) {
        autoPenalty--; penaltyT = now;
      }
    }
    if (now - statsTimer > 400) {
      statsTimer = now;
      const srcFps = intervalMs > 1 ? (1000 / intervalMs) : 0;
      if (cfg.debug) {
        const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
        hud.textContent =
          `видео: ${videoEl.videoWidth}x${videoEl.videoHeight} @ ${srcFps.toFixed(1)}fps\n` +
          `выход: ${fpsWin.length}fps · множитель ×${effN}${cfg.factor === 'auto' ? (autoPenalty ? ` (авто, −${autoPenalty} за дропы)` : ' (авто)') : ' (фикс)'}\n` +
          `вставка: ${msAvg.toFixed(1)}ms @ ${cfg.res}p · бюджет ${uniqueIntervalMs.toFixed(0)}ms · задержка ${delayMs.toFixed(0)}ms\n` +
          `GPU: ${sys.gpu} · загрузка ~${load.toFixed(0)}%\n` +
          `shown ${shown} · drop ${dropped} · давление ${dropPressure.toFixed(1)} · dups ${dups} · cuts ${cuts}\n` +
          `diff: mean ${lastStat ? lastStat.mean.toFixed(1) : '—'} max ${lastStat ? lastStat.max : '—'}` +
          `${lastStat && lastStat.max === 0 ? ' (ЧЁРНОЕ — DRM?)' : ''}`;
      } else {
        hud.textContent = `FC ${fpsWin.length}fps ×${effN} · ${msAvg.toFixed(0)}ms`;
      }
      if (panel && panel.style.display === 'block') updateStatus();
    }
    updateBar();
    requestAnimationFrame(pump);
  }

  // ---------- interpolation (lazy per-mid submission) ----------
  // prepPair runs once per frame pair; each mid's compute is submitted just-in-time
  // for its display slot, so present blits interleave with computes on the GPU
  // queue instead of the first mid waiting for the whole batch.
  let curJob = null;
  function submitMid() {
    const k = curJob.next;
    const disp = curJob.at + curJob.ts[k] * intervalMs;
    const busyMid = new Set(queue.map(q => q.tex)); // don't clobber queued mids
    let guard = midTexs.length;
    while (guard-- > 0 && busyMid.has(midTexs[midIdx])) midIdx = (midIdx + 1) % midTexs.length;
    const out = midTexs[midIdx];
    midIdx = (midIdx + 1) % midTexs.length;
    const t0 = performance.now();
    try { rt.runT(curJob.ts[k], out); } catch (e) { log('runT', e); curJob = null; return; }
    device.queue.onSubmittedWorkDone().then(() => {
      const ms = performance.now() - t0;
      msAvg = msAvg ? msAvg * 0.85 + ms * 0.15 : ms;
    });
    queue.push({ tex: out, at: disp, mid: true });
    curJob.next++;
    if (curJob.next >= curJob.ts.length) curJob = null;
  }
  function flushJob() {
    while (curJob && curJob.next < curJob.ts.length) submitMid();
    curJob = null;
  }
  function driveJob(now) {
    if (!curJob || switching) return;
    const lead = 2 * (msAvg || 10) + 8; // submit when the display slot is one compute away
    while (curJob && curJob.next < curJob.ts.length) {
      const disp = curJob.at + curJob.ts[curJob.next] * intervalMs;
      if (disp - now > lead) break;
      submitMid();
    }
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
      // note on importExternalTexture (evaluated, rejected): interpolation needs the
      // PREVIOUS frame too, and external textures expire with the video frame — the
      // copy is unavoidable for history and for presenting source frames. Prep/dedup
      // reads scale with MODEL resolution, not source, so reading the external
      // texture instead of the copy saves nothing measurable.
      // NEVER overwrite a texture that is still queued for presentation or needed
      // as an interpolation input — reuse of live textures = timeline soup
      const busyTex = new Set(queue.map(q => q.tex));
      if (lastTex) busyTex.add(lastTex);
      let guard = frameTex.length;
      while (guard-- > 0 && busyTex.has(frameTex[frameIdx])) frameIdx = (frameIdx + 1) % frameTex.length;
      const tex = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: tex }, [vw, vh]);
      // presentation delay must cover the batch compute time (own + the previous
      // batch draining), or high factors drop their early mids as already-stale.
      // Slewed ±2ms/frame so pacing never jumps.
      // lazy submission: a mid only waits for ITS OWN compute, not the whole batch
      const dTarget = Math.min(180, Math.max(60, 2 * (msAvg || 10) + 25));
      delayMs += Math.max(-2, Math.min(2, dTarget - delayMs));
      queue.push({ tex, at: arrival + delayMs, mid: false });
      const prev = lastTex;
      if (!cfg.fg) { // frame generation off: passthrough (SR-only if enabled)
        effN = 1;
        lastUniqueTs = arrival;
      } else if (prev) {
        const { dup, cut } = await classifyPair(prev, tex);
        if (cut) { cuts++; lastUniqueTs = arrival; }
        else if (cfg.anime && dup) { dups++; }
        else {
          const du = arrival - lastUniqueTs;
          if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
          lastUniqueTs = arrival;
          const ms = msAvg || 10;
          let n, run = true;
          if (cfg.factor === 'auto') {
            // smart auto: as much as fits in 85% of the per-unique-frame budget,
            // but never past the display refresh (extra frames would be thrown away)
            n = 6;
            while (n > 2 && (n - 1) * ms > uniqueIntervalMs * 0.85) n--;
            // cap by what the compositor actually presents, no optimism margin
            const dispHz = rafMs > 1 ? 1000 / rafMs : 60;
            while (n > 2 && (1000 / uniqueIntervalMs) * n > dispHz) n--;
            n = Math.max(2, n - autoPenalty); // drop-rate feedback (see pump)
            if ((n - 1) * ms > uniqueIntervalMs * 1.1) { run = false; autoSkipT = arrival; } // even 2x won't fit
          } else {
            n = cfg.factor; // fixed by the user — NEVER lowered
          }
          effN = n;
          if (run && !switching) {
            flushJob(); // leftovers of the previous pair go out before the new prep
            try {
              rt.prepPair(prev, tex);
              const ts = [];
              for (let k = 1; k < n; k++) ts.push(k / n);
              curJob = { ts, next: 0, at: arrival - intervalMs + delayMs };
            } catch (e) { log('prep', e); }
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
    overlay.style.opacity = '0';
    overlay.style.display = 'block';
    // native players: we own clicks (play/pause + our fullscreen). Sites with DOM
    // controls (YouTube etc.) keep the overlay transparent to the pointer.
    overlay.style.pointerEvents = videoEl.controls ? 'auto' : 'none';
    // seed the canvas with the current video frame so the reveal is seamless —
    // no black flash while the first interpolated frames are still in flight
    const vw = Math.min(videoEl.videoWidth, 1920), vh = Math.min(videoEl.videoHeight, 1080);
    if (vw && vh) {
      ensureFrameTextures(vw, vh);
      const seed = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: seed }, [vw, vh]);
      present(seed, false);
    }
    if (cfg.sr) ensureSR().catch(e => log('sr', e));
    queue = []; lastTex = null; curJob = null;
    shown = 0; dropped = 0; dups = 0; cuts = 0;
    running = true;
    hud.style.display = 'block';
    videoEl.requestVideoFrameCallback(onFrame);
    requestAnimationFrame(pump);
    btn.style.background = 'rgba(25,195,125,.9)';
  }
  function stop() {
    running = false;
    if (overlay) { // fade out, then release — the raw video underneath is identical
      overlay.style.opacity = '0';
      setTimeout(() => { if (!running && overlay) overlay.style.display = 'none'; }, 260);
    }
    hud.style.display = 'none';
    if (bar) bar.style.display = 'none';
    overSince = 0;
    if (warnEl) warnEl.style.opacity = '0';
    if (splitEl) splitEl.style.display = 'none';
    queue = []; lastTex = null; curJob = null;
    btn.style.background = '';
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
    const srState = cfg.sr ? (!sys.f16 ? 'недоступен (нет f16)' : (sr ? 'вкл ×2' : 'загрузка…')) : 'выкл';
    const lines = [`GPU: ${sys.gpu}`,
      `f16: ${sys.f16 ? 'да' : 'НЕТ (медленный путь)'} · модель: rt_slim`,
      `FG: ${cfg.fg ? 'вкл' : 'ВЫКЛ'} · SR: ${srState}`,
      `HDR: ${!sys.hdrOk ? 'дисплей не HDR' : (cfg.hdr ? (sys.hdrOn ? 'вкл (ITM)' : 'ошибка, SDR') : 'выкл')}`,
      `статус: ${running ? 'работает' : 'остановлен'}`];
    if (running) {
      const [mw, mh] = SIZES[cfg.res];
      const vramMB = (texW * texH * 4 * frameTex.length + mw * mh * 4 * midTexs.length) / 1048576;
      const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
      lines.push(
        `выход: ${fpsWin.length}fps · множитель ×${effN}${cfg.factor === 'auto' ? ' (авто)' : ' (фикс)'}`,
        `дисплей: ~${rafMs > 1 ? (1000 / rafMs).toFixed(0) : '—'}Гц`,
        `вставка: ${msAvg.toFixed(1)}ms @ ${cfg.res}p`,
        `нагрузка GPU (наша, оценка): ~${load.toFixed(0)}%`,
        `VRAM текстуры: ~${vramMB.toFixed(0)}MB · очередь ${queue.length}`);
    }
    st.textContent = lines.join('\n');
  }

  // hot-swap the runtime on quality change: stop/start reseeded the canvas with a
  // LIVE frame while the pipeline serves ~delayMs-old ones — time visibly jumped
  // forward and snapped back. Instead: drain the in-flight batch, drop queued mids
  // (source frames keep presenting), rebuild rt at the new size, relearn timing.
  async function switchRes() {
    switching = true; // gates onFrame/runPair: NO new mids while textures are being replaced
    try {
      // loop: the user may flip the select again mid-rebuild — converge on the latest
      while (rtRes !== cfg.res) {
        curJob = null; // abandon un-submitted mids of the old pair
        queue = queue.filter((it) => !it.mid);
        msAvg = 0; // cost at the new size is different — relearn
        await ensureRuntime();
        queue = queue.filter((it) => !it.mid); // stragglers that slipped in mid-rebuild
      }
    } finally { switching = false; }
  }

  // keep the whole panel on screen — it grows when "advanced" unfolds
  function clampPanel() {
    if (!panel || panel.style.display !== 'block') return;
    const r = panel.getBoundingClientRect();
    if (r.bottom > innerHeight - 10) panel.style.top = Math.max(10, innerHeight - r.height - 10) + 'px';
    if (r.right > innerWidth - 10) panel.style.left = Math.max(10, innerWidth - r.width - 10) + 'px';
  }

  function buildPanel() {
    panel = document.createElement('div');
    panel.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'background:rgba(16,16,16,.92); color:#ddd; border:1px solid rgba(255,255,255,.14);'
      + 'border-radius:14px; backdrop-filter:blur(10px); box-shadow:0 8px 32px rgba(0,0,0,.5);'
      + 'padding:14px 16px; font:12px/1.5 system-ui; display:none; width:270px; box-sizing:border-box;'
      + 'max-height:calc(100vh - 20px); overflow-y:auto; overscroll-behavior:contain;';
    panel.innerHTML = `
      <div class="fc-title">Framecast <span style="color:#667;font:400 10px system-ui">v0.3.1</span></div>
      <label class="fc-row"><span>Плавность<small>дорисовка кадров нейросетью</small></span>
        <input class="fc-sw" type="checkbox" id="fcFG"></label>
      <label class="fc-row"><span>Чёткость<small>апскейл вставок ×2</small></span>
        <input class="fc-sw" type="checkbox" id="fcSR"></label>
      <label class="fc-row"><span>HDR<small>расширение яркости (нужен HDR-экран)</small></span>
        <input class="fc-sw" type="checkbox" id="fcHDR"></label>
      <label class="fc-row"><span>Качество<small>нагрузка на видеокарту</small></span>
        <select class="fc-sel" id="fcRes">
          <option value="360">экономное</option>
          <option value="480">баланс</option>
          <option value="720">максимум</option>
          <option value="1080">ультра</option>
        </select></label>
      <details class="fc-details">
        <summary>продвинутые настройки</summary>
        <label class="fc-row"><span>Множитель кадров</span>
          <select class="fc-sel" id="fcFactor">
            <option value="auto">авто</option>
            <option value="2">2×</option><option value="3">3×</option>
            <option value="4">4×</option><option value="5">5×</option>
            <option value="6">6×</option>
          </select></label>
        <label class="fc-row"><span>Аниме-дедуп<small>распознавать «двойки» кадров</small></span>
          <input class="fc-sw" type="checkbox" id="fcAnime"></label>
        <label class="fc-row"><span>Контролы при наведении</span>
          <input class="fc-sw" type="checkbox" id="fcHover"></label>
        <label class="fc-row"><span>Сравнение<small>шторка оригинал / FC</small></span>
          <input class="fc-sw" type="checkbox" id="fcCompare"></label>
        <label class="fc-row"><span>Debug<small>рамка + телеметрия</small></span>
          <input class="fc-sw" type="checkbox" id="fcDebug"></label>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,.1);margin:8px 0">
        <div id="fcStatus" style="font:11px/1.6 monospace;color:#9c9;white-space:pre">—</div>
      </details>`;
    document.body.appendChild(panel);
    panel.querySelector('.fc-details').addEventListener('toggle', () => requestAnimationFrame(clampPanel));
    const F = panel.querySelector('#fcFactor'), R = panel.querySelector('#fcRes');
    const A = panel.querySelector('#fcAnime'), D = panel.querySelector('#fcDebug');
    const Hv = panel.querySelector('#fcHover'), Cm = panel.querySelector('#fcCompare');
    syncPanel();
    F.onchange = () => { cfg.factor = F.value === 'auto' ? 'auto' : +F.value; overSince = 0; saveCfg(); };
    A.onchange = () => { cfg.anime = A.checked; saveCfg(); };
    D.onchange = () => { cfg.debug = D.checked; saveCfg(); };
    Hv.onchange = () => { cfg.hoverReveal = Hv.checked; saveCfg(); };
    Cm.onchange = () => { cfg.compare = Cm.checked; saveCfg(); };
    const Fg = panel.querySelector('#fcFG'), Sr = panel.querySelector('#fcSR');
    Fg.onchange = () => { cfg.fg = Fg.checked; overSince = 0; saveCfg(); };
    Sr.onchange = () => {
      cfg.sr = Sr.checked; saveCfg();
      if (cfg.sr && device) ensureSR().catch(e => log('sr', e));
    };
    const Hd = panel.querySelector('#fcHDR');
    Hd.onchange = () => { cfg.hdr = Hd.checked; saveCfg(); configureOverlay(); };
    R.onchange = async () => {
      cfg.res = +R.value; saveCfg();
      if (running && !toggling) { // hot-swap, no visible restart
        toggling = true;
        try { await switchRes(); }
        catch (e) { log('res switch', e); }
        finally { toggling = false; }
      }
    };
  }

  function injectUI() {
    const css = document.createElement('style');
    css.textContent = `
      .fc-btn{background:none;border:none;color:#fff;font:13px/1 system-ui;cursor:pointer;
        padding:4px 6px;opacity:.85;transition:opacity .15s,transform .15s;
        display:inline-flex;align-items:center;justify-content:center}
      .fc-btn:hover{opacity:1;transform:scale(1.15)}
      .fc-side svg{display:block;margin:auto}
      .fc-range{-webkit-appearance:none;appearance:none;height:3px;border-radius:3px;margin:0;
        background:rgba(255,255,255,.22);outline:none;cursor:pointer}
      .fc-range::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;
        border-radius:50%;background:#fff;transition:transform .15s}
      .fc-range:hover::-webkit-slider-thumb{transform:scale(1.35);background:#19c37d}
      .fc-side{position:fixed;z-index:2147483647;width:38px;height:38px;border-radius:50%;
        border:none;background:rgba(15,15,15,.62);color:#fff;cursor:pointer;display:none;
        backdrop-filter:blur(6px);font:600 12px/1 system-ui;box-shadow:0 2px 12px rgba(0,0,0,.4);
        transition:background .15s,transform .15s}
      .fc-side:hover{transform:scale(1.1);background:rgba(45,45,45,.85)}
      .fc-title{font:600 14px system-ui;color:#fff;display:flex;align-items:center;gap:7px;margin-bottom:8px}
      .fc-title::before{content:'';width:8px;height:8px;border-radius:50%;background:#19c37d}
      .fc-row{display:flex;justify-content:space-between;align-items:center;gap:18px;
        padding:7px 0;cursor:pointer;font:12px system-ui;color:#e8e8e8}
      .fc-row small{display:block;color:#8a8f98;font-size:10px;margin-top:1px}
      .fc-sw{appearance:none;-webkit-appearance:none;width:36px;height:20px;border-radius:20px;
        background:#3d4148;position:relative;cursor:pointer;outline:none;margin:0;
        transition:background .2s;flex:none}
      .fc-sw::after{content:'';position:absolute;width:16px;height:16px;border-radius:50%;
        background:#fff;top:2px;left:2px;transition:left .2s;box-shadow:0 1px 3px rgba(0,0,0,.4)}
      .fc-sw:checked{background:#19c37d}
      .fc-sw:checked::after{left:18px}
      /* customizable select (Chrome base-select): button + popup in the same glass */
      .fc-sel, .fc-sel::picker(select){appearance:base-select}
      .fc-sel{background:rgba(255,255,255,.07);color:#eee;border:1px solid rgba(255,255,255,.14);
        border-radius:8px;padding:5px 11px;font:12px system-ui;outline:none;cursor:pointer;
        flex:none;min-width:118px;display:flex;align-items:center;justify-content:space-between;
        gap:8px;transition:background .15s,border-color .15s}
      .fc-sel:hover{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.28)}
      .fc-sel:open{border-color:rgba(25,195,125,.6)}
      .fc-sel::picker-icon{color:#8a8f98;font-size:9px;transition:rotate .15s}
      .fc-sel:open::picker-icon{rotate:180deg}
      .fc-sel::picker(select){background:rgba(20,22,26,.95);backdrop-filter:blur(12px);
        border:1px solid rgba(255,255,255,.14);border-radius:10px;padding:4px;margin-top:4px;
        box-shadow:0 8px 28px rgba(0,0,0,.55)}
      .fc-sel option{padding:5px 10px;border-radius:7px;font:12px system-ui;color:#ddd;
        background:transparent;cursor:pointer}
      .fc-sel option:hover{background:rgba(255,255,255,.09)}
      .fc-sel option:checked{background:rgba(25,195,125,.16);color:#8ee7bd}
      .fc-sel option::checkmark{color:#19c37d}
      .fc-details summary{cursor:pointer;color:#8a8f98;font:11px system-ui;list-style:none;
        display:flex;align-items:center;gap:5px;padding:6px 0 2px;user-select:none}
      .fc-details summary::before{content:'';width:0;height:0;border-left:4px solid #8a8f98;
        border-top:3.5px solid transparent;border-bottom:3.5px solid transparent;transition:transform .15s}
      .fc-details[open] summary::before{transform:rotate(90deg)}
      .fc-details summary::-webkit-details-marker{display:none}`;
    (document.head || document.documentElement).appendChild(css);
    btn = document.createElement('button');
    btn.textContent = 'FC';
    btn.className = 'fc-side';
    gear = document.createElement('button');
    gear.innerHTML = svgIcon('gear', 17);
    gear.className = 'fc-side';
    hud = document.createElement('div');
    // anchored to the video's top-right corner every pump tick (inside the player)
    hud.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'color:#0f0; font:11px/1.5 monospace; background:rgba(0,0,0,.7); padding:4px 8px;'
      + 'border-radius:6px; white-space:pre; text-align:left; pointer-events:none; display:none;';
    buildPanel();
    ensureBar();
    btn.onclick = async () => {
      if (toggling) return; // start/stop in flight — spam-proof
      toggling = true;
      try {
        if (running) { stop(); return; }
        const v = biggestVideo();
        if (!v) { hud.style.display = 'block'; hud.textContent = 'FC: видео не найдено'; return; }
        try { await start(v); } catch (e) { hud.style.display = 'block'; hud.textContent = 'FC ошибка: ' + (e.message || e); log(e); }
      } finally { toggling = false; }
    };
    gear.onclick = () => {
      const open = panel.style.display === 'none';
      panel.style.display = open ? 'block' : 'none';
      if (open) { // dock next to the gear, clamped to the viewport
        const g = gear.getBoundingClientRect();
        panel.style.left = Math.min(g.right + 10, innerWidth - panel.offsetWidth - 10) + 'px';
        panel.style.top = Math.max(10, Math.min(g.top - panel.offsetHeight / 2, innerHeight - panel.offsetHeight - 10)) + 'px';
        updateStatus();
        clampPanel();
      }
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
