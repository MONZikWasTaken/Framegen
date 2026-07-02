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
  let bar = null, barSeeking = false;
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
    overlay.style.cssText = 'position:absolute; pointer-events:none; z-index:2;';
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
      if (bar) uiHost.appendChild(bar);
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
    q('#fcPlay').onclick = () => {
      if (!videoEl) return;
      videoEl.paused ? videoEl.play() : videoEl.pause();
      flashCenter(svgIcon(videoEl.paused ? 'pause' : 'play', 30));
      updateBar();
    };
    q('#fcMute').onclick = () => { if (videoEl) videoEl.muted = !videoEl.muted; updateBar(); };
    q('#fcVol').oninput = (e) => { if (videoEl) { videoEl.volume = e.target.value / 100; videoEl.muted = false; } };
    q('#fcSeek').addEventListener('pointerdown', () => { barSeeking = true; });
    q('#fcSeek').addEventListener('pointerup', () => { barSeeking = false; });
    q('#fcSeek').oninput = (e) => {
      if (videoEl && videoEl.duration) videoEl.currentTime = e.target.value / 1000 * videoEl.duration;
    };
    // fullscreen the PARENT (so the overlay comes along) and stretch the video to
    // fill the screen — fullscreening just the container leaves the video at its
    // layout size, which looks like fullscreen "not working"
    let fsByUs = false, fsSaved = '';
    q('#fcFull').onclick = () => {
      if (document.fullscreenElement) document.exitFullscreen();
      else if (videoEl) {
        fsByUs = true;
        (videoEl.parentElement || videoEl).requestFullscreen().catch(e => { fsByUs = false; log('fullscreen', e); });
      }
    };
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

  function pump(now) {
    if (!running) return;
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
    updateBar();
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
    btn.style.background = 'rgba(25,195,125,.9)';
  }
  function stop() {
    running = false;
    if (overlay) overlay.style.display = 'none';
    hud.style.display = 'none';
    if (bar) bar.style.display = 'none';
    queue = []; lastTex = null; pending = null;
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
    panel.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'background:rgba(16,16,16,.92); color:#ddd; border:1px solid rgba(255,255,255,.14);'
      + 'border-radius:12px; backdrop-filter:blur(8px); box-shadow:0 6px 24px rgba(0,0,0,.45);'
      + 'padding:12px 14px; font:12px/2 system-ui; display:none; min-width:230px;';
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
      .fc-side:hover{transform:scale(1.1);background:rgba(45,45,45,.85)}`;
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
      if (running) { stop(); return; }
      const v = biggestVideo();
      if (!v) { hud.style.display = 'block'; hud.textContent = 'FC: видео не найдено'; return; }
      try { await start(v); } catch (e) { hud.style.display = 'block'; hud.textContent = 'FC ошибка: ' + (e.message || e); log(e); }
    };
    gear.onclick = () => {
      const open = panel.style.display === 'none';
      panel.style.display = open ? 'block' : 'none';
      if (open) { // dock next to the gear, clamped to the viewport
        const g = gear.getBoundingClientRect();
        panel.style.left = Math.min(g.right + 10, innerWidth - panel.offsetWidth - 10) + 'px';
        panel.style.top = Math.max(10, Math.min(g.top - panel.offsetHeight / 2, innerHeight - panel.offsetHeight - 10)) + 'px';
        updateStatus();
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
