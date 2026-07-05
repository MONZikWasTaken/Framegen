// Framecast content script: real-time frame interpolation for any <video> on the page.
// GPU-resident pipeline (own WebGPU runtime, weights bundled): video -> texture ->
// interpolation -> overlay canvas (sibling of the video; site controls stay on top).
// DRM (EME) video produces black frames - nothing any extension can do about that.
(() => {
  'use strict';
  if (window.__framecast) return;
  window.__framecast = true;

  const DELAY_MS = 60;
  // runtime tiles are 16x16 - model dims must be /16 (1088, not 1080; the ~0.7%
  // vertical stretch at present time is invisible)
  const SIZES = { 288: [512, 288], 360: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1088] };

  // ---------- settings (chrome.storage.local, live-applied) ----------
  // factor: 'auto' (smart) or a fixed 2..6 treated as a CEILING under GPU overload
  // model: weight set key from MODELS. v7s is the default: faster (2.57ms vs
  // 3.05ms @480p, 3.75 vs 5.51 @720p) at equal-or-better quality. v6 stays
  // selectable; users with a saved choice keep it.
  const MODELS = { v6: 'rt_tfact2', v7s: 'rt_v7s' };
  const cfg = { factor: 'auto', anime: true, debug: false, res: 480, hoverReveal: true, compare: false,
    fg: true, sr: false, hdr: false, showFps: true, guard: true, model: 'v7s' };
  function sanitizeCfg() {
    if (cfg.factor !== 'auto' && cfg.factor !== 'hz' && ![2, 3, 4, 5, 6].includes(cfg.factor)) cfg.factor = 'auto';
    if (!MODELS[cfg.model]) cfg.model = 'v7s';
    if (!SIZES[cfg.res]) cfg.res = 480;
    cfg.anime = !!cfg.anime; cfg.debug = !!cfg.debug;
    cfg.hoverReveal = !!cfg.hoverReveal; cfg.compare = !!cfg.compare;
    cfg.fg = !!cfg.fg; cfg.sr = !!cfg.sr; cfg.hdr = !!cfg.hdr;
    cfg.showFps = !!cfg.showFps; cfg.guard = !!cfg.guard;
  }
  try {
    // async: the panel may already be built with defaults by the time this lands -
    // ALWAYS resync the UI, otherwise checkboxes show one thing and cfg does another
    chrome.storage.local.get(cfg, v => { Object.assign(cfg, v); sanitizeCfg(); syncPanel(); });
    // settings changed in another tab/frame apply here live
    chrome.storage.onChanged.addListener((ch, area) => {
      if (area !== 'local') return;
      let resChanged = false;
      for (const k in ch) {
        if (!(k in cfg)) continue;
        if ((k === 'res' || k === 'model') && cfg[k] !== ch[k].newValue) resChanged = true;
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
    panel.querySelector('#fcModel').value = cfg.model;
    panel.querySelector('#fcAnime').checked = cfg.anime;
    panel.querySelector('#fcDebug').checked = cfg.debug;
    panel.querySelector('#fcHover').checked = cfg.hoverReveal;
    panel.querySelector('#fcFps').checked = cfg.showFps;
    panel.querySelector('#fcGuard').checked = cfg.guard;
    panel.querySelector('#fcCompare').checked = cfg.compare;
    panel.querySelector('#fcFG').checked = cfg.fg;
    panel.querySelector('#fcSR').checked = cfg.sr;
    const hd = panel.querySelector('#fcHDR');
    hd.checked = cfg.hdr;
    if (!sys.hdrOk) { hd.disabled = true; hd.style.opacity = '.35'; }
  }

  let rt = null, rtRes = 0, rtModel = '', device = null, videoEl = null;
  let overlay = null, overlayCtx = null, blitPipe = null, blitSampler = null;
  const blitBg = new Map();
  let frameTex = [], frameIdx = 0, texW = 0, texH = 0, lastTex = null;
  let midTexs = [], midIdx = 0;
  let dedupPipe = null, dedupBg = new Map(), dedupStats = null, dedupSampler = null;
  let dedupReads = [], dedupReadIdx = 0; // readback ring: classifies overlap now
  let queue = [], running = false, processingFrame = false;
  let pairSeq = 0; // generation counter for in-flight classify continuations
  let hzNext = 0; // display-match mode: absolute time of the next output vsync tick
  let intervalMs = 42, uniqueIntervalMs = 42, lastArrival = 0, lastUniqueTs = 0;
  let msAvg = 0, dropped = 0, dups = 0, cuts = 0, fpsWin = [], effN = 2, lastStat = null;
  let btn = null, gear = null, hud = null, panel = null, statsTimer = 0;
  let bar = null, barSeeking = false, wm = null;
  let rafMs = 0, lastPumpT = 0, warnEl = null, overSince = 0;
  let splitEl = null, splitX = 0.5, toggling = false, autoSkipT = 0;
  let delayMs = DELAY_MS, dropWin = [], switching = false, preloadFailT = -1e9;
  let schedT = 0, rafFloor = 100, uiTick = 0, motionAvg = 0, lateAvg = 0;
  let autoPenalty = 0, penaltyT = 0, dropPressure = 0, lastPressureT = 0;
  const sys = { gpu: '-', f16: false, hdrOk: false, hdrOn: false };
  try { sys.hdrOk = !!(window.matchMedia && matchMedia('(dynamic-range: high)').matches); } catch {}

  const log = (...a) => console.log('[framecast]', ...a);

  // Chrome on Windows IGNORES powerPreference (crbug.com/369219127): on dual-GPU
  // machines we get whatever GPU Chrome runs on. Detect integrated ones and tell
  // the user how to move Chrome to the discrete card.
  function classifyAdapter(adapter) {
    sys.f16 = adapter.features.has('shader-f16');
    const inf = adapter.info || {};
    sys.gpu = inf.description || [inf.vendor, inf.architecture].filter(Boolean).join(' ') || 'unknown GPU';
    sys.integrated = /intel|iris|uhd|graphics 6|vega|radeon\(tm\) graphics|apu/i.test(sys.gpu)
      && !/nvidia|geforce|rtx|gtx|radeon rx|arc a|arc b/i.test(sys.gpu);
  }
  // lightweight probe for the popup: adapter info only, no device, no weights
  let probing = null;
  async function probeAdapter() {
    if (sys.gpu !== '-' || !navigator.gpu) return;
    if (!probing) probing = (async () => {
      try {
        const a = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (a) classifyAdapter(a);
      } catch { /* leave unknown */ }
    })();
    await probing;
  }

  // ---------- device / runtime ----------
  // memoized: the hover-preload and the FC click may race - only one build runs
  let rtBuilding = null;
  async function ensureRuntime() {
    while (rtBuilding) await rtBuilding;
    if (device && rt && rtRes === cfg.res && rtModel === cfg.model) return;
    rtBuilding = buildRuntime();
    try { await rtBuilding; } finally { rtBuilding = null; }
  }
  async function loadConvTune() {
    try {
      const key = 'fcTune|' + sys.gpu + '|' + cfg.res + '|' + MODELS[cfg.model];
      const st = await chrome.storage.local.get('fcTune');
      return (st.fcTune && st.fcTune[key]) || null;
    } catch { return null; }
  }
  async function calibrateConvTune(rtMod) {
    // one-shot per (GPU, quality): bench kernel variants on the real conv shape,
    // persist the winner - picked up on the next runtime build
    try {
      const key = 'fcTune|' + sys.gpu + '|' + cfg.res + '|' + MODELS[cfg.model];
      const st = await chrome.storage.local.get('fcTune');
      const all = st.fcTune || {};
      if (all[key]) return;
      const [mw, mh] = SIZES[cfg.res];
      const best = await rtMod.tuneConvRB(device, { ci: rtC2, co: rtC2, w16: mw / 16, h16: mh / 16 });
      all[key] = { coc: best.coc, slab: best.slab, sg: !!best.sg };
      await chrome.storage.local.set({ fcTune: all });
      log('conv tune', cfg.res, JSON.stringify(best));
    } catch (e) { log('tune skipped', e); }
  }
  let rtC2 = 0;
  async function buildRuntime() {
    if (!device) {
      if (!navigator.gpu) throw new Error('WebGPU unavailable');
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) throw new Error('no GPU adapter');
      const f16 = adapter.features.has('shader-f16');
      const feats = f16 ? ['shader-f16'] : [];
      if (adapter.features.has('subgroups')) feats.push('subgroups'); // tuner may pick sg kernels
      device = await adapter.requestDevice({ requiredFeatures: feats });
      classifyAdapter(adapter);
    }
    if (rt && rtRes === cfg.res && rtModel === cfg.model) return;
    const url = (p) => chrome.runtime.getURL(p);
    // tfact2 family: t-factored student + quarter-res refine head; the runtime
    // autodetects the trunk width from the manifest, so models are weight swaps
    const fetchSet = async (stem) => Promise.all([
      fetch(url('assets/' + stem + '.bin')).then(r => { if (!r.ok) throw 0; return r.arrayBuffer(); }),
      fetch(url('assets/' + stem + '.json')).then(r => { if (!r.ok) throw 0; return r.json(); })]);
    let bin, man;
    try {
      [bin, man] = await fetchSet(MODELS[cfg.model]);
    } catch (e) {
      // a dead runtime (extension was reloaded/updated) is not a missing model -
      // nothing works until the page reloads, so say exactly that and stop
      if (!chrome.runtime?.id) throw new Error('extension reloaded - refresh the page (F5)');
      if (cfg.model === 'v6') throw e;
      log('model ' + cfg.model + ' not bundled - falling back to v6');
      cfg.model = 'v6';
      [bin, man] = await fetchSet(MODELS.v6);
    }
    const rtMod = await import(url('rt/rt.js'));
    const [mw, mh] = SIZES[cfg.res];
    rtC2 = man['block0.conv0.1.0.weight'].shape[0];
    const convTune = await loadConvTune();
    rt = await rtMod.createRT(device, { w: mw, h: mh, textureInput: true, textureOutput: true,
      staticGuard: cfg.guard, weightsBin: bin, weightsManifest: man, convTune });
    rtRes = cfg.res; rtModel = cfg.model;
    if (!convTune) setTimeout(() => calibrateConvTune(rtMod), 4000);
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

  let poolGen = 0; // labels must be unique across reallocations: label-keyed
  // caches (dedup) must never hit an entry built for a destroyed generation
  function ensureFrameTextures(w, h) {
    if (texW === w && texH === h && frameTex.length === 12) return;
    frameTex.forEach(t => t.destroy());
    frameTex = [];
    queue = []; curJob = null; cmpRing = []; // queued entries reference the destroyed pool
    pairSeq++; // in-flight classify continuations must not prep destroyed textures
    poolGen++;
    for (let i = 0; i < 12; i++) {
      frameTex.push(device.createTexture({ label: 'fcfr' + poolGen + '_' + i, size: [w, h], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT }));
    }
    texW = w; texH = h; dedupBg.clear(); blitBg.clear(); lastTex = null;
  }

  // pool dimensions for a source: fit inside FHD, keep the aspect ratio
  function poolDims() {
    const fw = videoEl.videoWidth, fh = videoEl.videoHeight;
    const s = Math.min(1, 1920 / fw, 1080 / fh);
    return [Math.round(fw * s), Math.round(fh * s)];
  }
  // copyExternalImageToTexture copies 1:1 and NEVER scales - for >FHD sources a
  // plain copy grabs the top-left FHD crop of the frame. Capture the full frame
  // into a scratch texture and downscale-blit it into the pool instead.
  let capTex = null, downPipe = null;
  function captureFrame(dst, vw, vh) {
    const fw = videoEl.videoWidth, fh = videoEl.videoHeight;
    if (fw <= 1920 && fh <= 1080) {
      device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: dst }, [vw, vh]);
      return;
    }
    if (!capTex || capTex.width !== fw || capTex.height !== fh) {
      if (capTex) capTex.destroy();
      capTex = device.createTexture({ label: 'fccap', size: [fw, fh], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST
          | GPUTextureUsage.RENDER_ATTACHMENT });
    }
    device.queue.copyExternalImageToTexture({ source: videoEl }, { texture: capTex }, [fw, fh]);
    if (!downPipe) {
      const mod = device.createShaderModule({ code: BLIT_VS + `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, v.uv, 0.0);
}` });
      downPipe = device.createRenderPipeline({ layout: 'auto',
        vertex: { module: mod, entryPoint: 'vs' },
        fragment: { module: mod, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] } });
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: dst.createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(downPipe);
    pass.setBindGroup(0, device.createBindGroup({ layout: downPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: capTex.createView() },
        { binding: 1, resource: blitSampler }] }));
    pass.draw(3);
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  // ---------- dedup / cut (GPU, 8-byte readback) ----------
  function ensureDedup() {
    if (dedupPipe) return;
    dedupSampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    dedupStats = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    dedupReads = Array.from({ length: 3 }, () => ({ busy: false,
      buf: device.createBuffer({ size: 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) }));
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
  const DEDUP_ZERO = new Uint32Array(2);
  async function classifyPair(ta, tb) {
    ensureDedup();
    // free readback slot: classifies overlap (the frame loop does not await
    // them), two can be in flight at 120fps sources. All busy = ordinary motion.
    let rb = null;
    for (let i = 0; i < dedupReads.length; i++) {
      const c = dedupReads[(dedupReadIdx + i) % dedupReads.length];
      if (!c.busy) { rb = c; dedupReadIdx = (dedupReadIdx + i + 1) % dedupReads.length; break; }
    }
    if (!rb) return { dup: false, cut: false, black: false };
    rb.busy = true;
    try {
      const key = ta.label + '|' + tb.label;
      if (!dedupBg.has(key)) {
        dedupBg.set(key, device.createBindGroup({ layout: dedupPipe.getBindGroupLayout(0), entries: [
          { binding: 0, resource: ta.createView() }, { binding: 1, resource: tb.createView() },
          { binding: 2, resource: dedupSampler }, { binding: 3, resource: { buffer: dedupStats } }] }));
      }
      // the single stats buffer is safe across overlapping classifies: zero,
      // dispatch and the copy-out are queue-ordered per submit
      device.queue.writeBuffer(dedupStats, 0, DEDUP_ZERO);
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(dedupPipe); pass.setBindGroup(0, dedupBg.get(key));
      pass.dispatchWorkgroups(6, 4);
      pass.end();
      enc.copyBufferToBuffer(dedupStats, 0, rb.buf, 0, 8);
      device.queue.submit([enc.finish()]);
      await rb.buf.mapAsync(GPUMapMode.READ);
      const s = new Uint32Array(rb.buf.getMappedRange().slice(0));
      rb.buf.unmap();
      const mean = s[0] / (48 * 27);
      lastStat = { mean, max: s[1] };
      // motion EMA feeds the artifact-aware factor cap; dups/cuts don't count as motion
      if (mean < 90) motionAvg = motionAvg * 0.7 + mean * 0.3;
      return { dup: mean < 2.5 && s[1] < 45, cut: mean > 90, black: s[1] === 0 };
    } finally {
      rb.busy = false;
    }
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
  // (re)build the present path: SDR passthrough, or HDR via inverse tone mapping -
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
  // fullscreen renders in the browser's TOP LAYER: anything not inside the
  // fullscreen element is invisible there. Move the whole UI in (and back out) -
  // fired from the fullscreenchange event too, so it works with the SITE's own
  // fullscreen button and while FC is off.
  function reparentUI() {
    const uiHost = document.fullscreenElement || document.body;
    if (uiHost.tagName === 'VIDEO') return; // bare-video fullscreen: nothing can overlay it
    if (btn && btn.parentElement !== uiHost) {
      uiHost.appendChild(btn); uiHost.appendChild(gear); uiHost.appendChild(hud); uiHost.appendChild(panel);
      if (bar) uiHost.appendChild(bar);
      if (splitEl) uiHost.appendChild(splitEl);
      if (warnEl) uiHost.appendChild(warnEl);
      if (flashEl) uiHost.appendChild(flashEl);
      if (wm) uiHost.appendChild(wm);
    }
  }
  document.addEventListener('fullscreenchange', () => {
    reparentUI();
    sbLeft = -1; // force button re-place at the new geometry
    // coords from the OLD geometry are garbage for a moment: hide, let the page
    // reflow (two frames), then re-place against the fresh video rect
    if (btn) { btn.style.display = 'none'; gear.style.display = 'none'; }
    uiScan = 0; // the biggest-video answer may change across fullscreen too
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const v = running ? videoEl : uiVideo;
      if (v && btn && performance.now() < revealUntil) {
        placeSideButtons(v.getBoundingClientRect());
      }
    }));
  });

  function positionOverlay(vrIn) { // caller may pass a fresh video rect to save a forced layout
    if (overlay.parentElement !== videoEl.parentElement) {
      videoEl.parentElement.insertBefore(overlay, videoEl.nextSibling);
    }
    reparentUI();
    const r = vrIn || videoEl.getBoundingClientRect();
    if (r.width < 8 || r.height < 8) return;
    // self-calibrating placement: measure where the overlay actually landed and nudge
    // by the delta - immune to whatever containing block/margins the site uses
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

  let cmpRing = []; // source frames + their due times: compare's left half runs on
  // the ORIGINAL cadence, independent of what the output side presents (hz mode
  // rarely presents raw sources - the left half would freeze otherwise)
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
    // compare: the left half shows the DELAYED source frame (same pipeline clock as
    // the FC half) - revealing the live <video> instead would be off by the delay
    let cmpSrcTex = null;
    if (cfg.compare) {
      const pnow = performance.now();
      while (cmpRing.length > 1 && cmpRing[1].at <= pnow) cmpRing.shift();
      if (cmpRing.length && cmpRing[0].at <= pnow) cmpSrcTex = cmpRing[0].tex;
    }
    if (cfg.compare && cmpSrcTex && cmpSrcTex !== tex) {
      if (!blitBg.has(cmpSrcTex)) {
        blitBg.set(cmpSrcTex, device.createBindGroup({ layout: blitPipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: cmpSrcTex.createView() }, { binding: 1, resource: blitSampler }] }));
      }
      pass.setScissorRect(0, 0, Math.max(1, Math.round(splitX * overlay.width)), overlay.height);
      pass.setBindGroup(0, blitBg.get(cmpSrcTex));
      pass.draw(3);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
    if (overlay.style.opacity !== '1') {
      overlay.style.transition = ''; // back to the stylesheet fade (onSrcChange kills it)
      overlay.style.opacity = '1'; // reveal only once pixels exist
    }
    const now = performance.now();
    fpsWin.push(now);
    while (fpsWin.length && fpsWin[0] < now - 1000) fpsWin.shift();
  }
  // our own control bar ON TOP of the overlay: native controls render INSIDE the
  // video element and can never show above the canvas, so instead of ever revealing
  // the raw video we drive the <video> ourselves - play/seek/volume/fullscreen as
  // regular DOM above everything. Interpolation is never interrupted.
  // sites whose own DOM controls are KNOWN to render above our overlay - there we
  // don't double up with our bar. Everywhere else (jut.su-style players put their
  // bar BELOW the canvas) our controls are the only usable ones.
  const SITE_CONTROLS_OK = /(^|\.)(youtube\.com|youtu\.be|vimeo\.com|twitch\.tv)$/
    .test(location.hostname);
  let revealUntil = 0, uiVideo = null, uiScan = 0, mmLast = 0;
  document.addEventListener('mousemove', (e) => {
    const now = performance.now();
    // gaming mice fire mousemove at up to 1000Hz; the rect read below forces
    // layout - unthrottled that alone janks the main thread while the mouse moves
    if (now - mmLast < 33) return;
    mmLast = now;
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
    } else if (revealUntil > now + 250) {
      revealUntil = now + 250; // pointer left the player: fade soon, not in 2s
    }
  }, { passive: true });

  // FC + settings live INSIDE the player: centered vertically at the left edge
  let sbLeft = -1, sbTop = -1;
  function placeSideButtons(r) {
    btn.style.display = gear.style.display = 'block';
    const left = Math.round(r.left + 12), cy = Math.round(r.top + r.height / 2);
    if (left === sbLeft && cy === sbTop) return; // no writes when nothing moved
    sbLeft = left; sbTop = cy;
    btn.style.left = gear.style.left = left + 'px';
    btn.style.top = (cy - 42) + 'px';
    gear.style.top = (cy + 4) + 'px';
  }
  // vertical feeds: every scroll is an SPA navigation to the next clip. The old
  // stream still dies with a hard stop(), but the user's FC-on intent carries
  // over - re-engage on the new player once it can decode a frame.
  const inFeed = () => /youtube\.com\/shorts|tiktok\.com/.test(location.href);
  let reattachSeq = 0;
  function reattach() {
    const seq = ++reattachSeq, t0 = performance.now();
    const tick = async () => {
      if (seq !== reattachSeq || running || !inFeed()) return;
      if (performance.now() - t0 > 12000) return; // closed player / no video: give up
      const v = biggestVideo();
      if (!v || !v.videoWidth || toggling) { setTimeout(tick, 150); return; }
      toggling = true;
      try { await start(v); }
      catch (e) { log('feed reattach', e); }
      finally { toggling = false; }
    };
    setTimeout(tick, 120);
  }
  let pageHref = location.href;
  setInterval(() => {
    // SPA navigation (YouTube next video, etc): the old stream is dead - showing
    // its frames on the new page is nonsense. Hard-off; the user re-enables -
    // except inside feeds, where the enable carries to the next clip.
    if (location.href !== pageHref) {
      pageHref = location.href;
      if (running) {
        stop();
        if (inFeed()) reattach();
      }
    }
    if (!btn) return;
    if (panel && panel.style.display === 'block') { revealUntil = performance.now() + 2000; return; }
    if (performance.now() > revealUntil) {
      btn.style.display = gear.style.display = 'none';
    }
  }, 300);
  // scrolling moves the video but not our fixed-position buttons - re-pin them
  // (capture: catches scrolling containers, not just the window)
  document.addEventListener('scroll', () => {
    if (!btn || btn.style.display === 'none') return;
    const v = running ? videoEl : uiVideo;
    if (v) placeSideButtons(v.getBoundingClientRect());
  }, { passive: true, capture: true });

  // crisp monochrome SVG icons (Feather-style) - no emoji
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
      + 'background:rgba(16,17,20,.88);'
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
    q('#fcFull').onclick = () => { if (gFull()) toggleFullscreen(); }; // async transition - no double-fire
    // keep the bar alive while the mouse is on it (cheap write, no throttle needed)
    bar.addEventListener('mousemove', () => { revealUntil = performance.now() + 2000; }, { passive: true });
  }

  // big centered ▶/❚❚ splash on play/pause, fades out while scaling up
  let flashEl = null;
  function flashCenter(sym) {
    if (!videoEl) return;
    if (!flashEl) {
      flashEl = document.createElement('div');
      flashEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
        + 'color:#fff; font:600 26px system-ui; background:rgba(16,17,20,.85);'
        + 'border-radius:50%; width:72px; height:72px;'
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
  // fill the screen - fullscreening just the container leaves the video at its
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
  // element refs + last-written values cached: this runs every UI tick, and
  // querySelector lookups + unconditional style writes are wasted work when
  // nothing changed (the gradient string rebuild forces a style recalc)
  let barEls = null, barVolP = -1, barCurT = '', barDurT = '', barSeekV = '', barSeekF = -1;
  function updateBar() {
    if (!bar || bar.style.display === 'none' || !videoEl) return;
    if (!barEls) {
      barEls = { play: bar.querySelector('#fcPlay'), mute: bar.querySelector('#fcMute'),
                 vol: bar.querySelector('#fcVol'), cur: bar.querySelector('#fcCur'),
                 dur: bar.querySelector('#fcDur'), seek: bar.querySelector('#fcSeek') };
    }
    const pi = videoEl.paused ? 'play' : 'pause';
    if (pi !== barPlayIcon) { barPlayIcon = pi; barEls.play.innerHTML = svgIcon(pi, 19); }
    const mi = (videoEl.muted || videoEl.volume === 0) ? 'volX' : 'vol';
    if (mi !== barMuteIcon) { barMuteIcon = mi; barEls.mute.innerHTML = svgIcon(mi); }
    const volP = Math.round((videoEl.muted ? 0 : videoEl.volume) * 100);
    if (volP !== barVolP) { barVolP = volP; barEls.vol.value = String(volP); rangeFill(barEls.vol, volP, '#fff'); }
    const d = videoEl.duration || 0, c = videoEl.currentTime || 0;
    const ct = fmt(c), dt = fmt(d);
    if (ct !== barCurT) { barCurT = ct; barEls.cur.textContent = ct; }
    if (dt !== barDurT) { barDurT = dt; barEls.dur.textContent = dt; }
    const p = d ? c / d * 100 : 0;
    if (!barSeeking && d) {
      const sv = String(Math.round(p * 10));
      if (sv !== barSeekV) { barSeekV = sv; barEls.seek.value = sv; }
    }
    const fp = Math.round(p * 10);
    if (fp !== barSeekF) { barSeekF = fp; rangeFill(barEls.seek, p, '#19c37d'); }
  }

  // compare mode: a draggable divider - raw video shows LEFT of it (the overlay is
  // clipped away), interpolated frames play on the right
  function ensureSplit() {
    if (splitEl) return;
    splitEl = document.createElement('div');
    splitEl.style.cssText = 'position:fixed; z-index:2147483645; width:18px; margin-left:-9px;'
      + 'cursor:ew-resize; touch-action:none; display:none;';
    splitEl.innerHTML = `
      <div style="position:absolute; left:50%; top:0; bottom:0; width:2px; margin-left:-1px;
        background:rgba(255,255,255,.85); box-shadow:0 0 10px rgba(0,0,0,.7)"></div>
      <div style="position:absolute; right:14px; bottom:10px; color:#fff; font:10px system-ui;
        background:rgba(15,15,15,.6); border-radius:6px; padding:2px 6px; white-space:nowrap">orig.</div>
      <div style="position:absolute; left:14px; bottom:10px; color:#fff; font:10px system-ui;
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

  // one-shot amber advisory plate (integrated-GPU hint etc.), positioned above warn
  let adviseEl = null, adviseUntil = 0;
  function advise(text, ms) {
    if (!adviseEl) {
      adviseEl = document.createElement('div');
      adviseEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
        + 'background:rgba(60,45,10,.92); color:#ffd88a;'
        + 'border:1px solid rgba(255,200,90,.35); border-radius:12px; padding:8px 14px;'
        + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4); max-width:70vw;'
        + 'opacity:0; transition:opacity .25s;';
      document.body.appendChild(adviseEl);
    }
    adviseEl.textContent = text;
    adviseUntil = performance.now() + ms;
  }
  function updateAdvise(now, vr) {
    if (!adviseEl) return;
    const show = now < adviseUntil;
    if (show) {
      adviseEl.style.left = Math.max(8, vr.left + vr.width / 2 - adviseEl.offsetWidth / 2) + 'px';
      adviseEl.style.top = (vr.top + 52) + 'px';
    }
    adviseEl.style.opacity = show ? '1' : '0';
  }

  // overload plate: fixed factors are never lowered, we just tell the user
  function ensureWarn() {
    if (warnEl) return;
    warnEl = document.createElement('div');
    warnEl.style.cssText = 'position:fixed; z-index:2147483646; pointer-events:none;'
      + 'background:rgba(60,16,16,.9); color:#ffb4a8;'
      + 'border:1px solid rgba(255,120,100,.35); border-radius:12px; padding:8px 14px;'
      + 'font:12px system-ui; box-shadow:0 4px 20px rgba(0,0,0,.4);'
      + 'opacity:0; transform:translateY(-6px); transition:opacity .25s, transform .25s;';
    warnEl.textContent = '⚠ Load too high - lower the factor or switch to auto';
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
        ? '⚠ Frames are dropping - lower the factor/quality or switch to auto'
        : '⚠ GPU cannot keep up even at 2x - set quality to eco';
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
    // SPA navigation can replace the <video> element entirely: rVFC dies with it
    // and the canvas would keep showing the dead stream's frames forever.
    // Feeds recycle player elements - carry the FC intent to the replacement.
    if (videoEl && !videoEl.isConnected) { stop(); if (inFeed()) reattach(); return; }
    try { pumpBody(now); } catch (e) { log('pump', e); }
    requestAnimationFrame(pump);
  }
  function pumpBody(now) {
    driveJob(now); // just-in-time mid submission
    if (lastPumpT) {
      const d = now - lastPumpT;
      // pessimist estimator: believe slowdowns fast (40%), speedups slowly (3%) -
      // auto must not re-inflate on every momentary lull
      if (d > 1 && d < 100) {
        rafMs = rafMs ? (d > rafMs ? rafMs * 0.6 + d * 0.4 : rafMs * 0.97 + d * 0.03) : d;
        rafFloor = Math.min(rafFloor + 0.02, d); // true vsync: snaps down, creeps up
      }
    }
    lastPumpT = now;
    // UI geometry work (rect reads + style writes force reflows on heavy pages)
    // runs at rAF/4 - presentation below stays per-tick
    uiTick = (uiTick + 1) & 3;
    if (uiTick === 0) {
    { // our control bar floats above the video bottom, HUD in the top-right corner
      const vr = videoEl.getBoundingClientRect(); // read ONCE per tick, shared with positionOverlay
      positionOverlay(vr);
      // our bar everywhere except sites whose own controls verifiably sit above
      // the overlay (see SITE_CONTROLS_OK)
      if (cfg.hoverReveal && (videoEl.controls || !SITE_CONTROLS_OK)) {
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
      // HUD: single line along the top-left of the video
      hud.style.display = (cfg.debug || cfg.showFps) ? 'block' : 'none';
      hud.style.left = (vr.left + 8) + 'px';
      hud.style.top = (vr.top + 8) + 'px';
      hud.style.maxWidth = Math.max(120, vr.width - 16) + 'px';
      // brand mark: bottom-left, always on while running
      wm.style.display = 'block';
      wm.style.left = (vr.left + 10) + 'px';
      wm.style.top = (vr.bottom - 26) + 'px';
      if (btn.style.display !== 'none') placeSideButtons(vr); // stay pinned while running
      updateWarn(now, vr);
      updateAdvise(now, vr);
      if (cfg.compare) {
        ensureSplit();
        splitEl.style.display = 'block';
        splitEl.style.left = (vr.left + splitX * vr.width) + 'px';
        splitEl.style.top = vr.top + 'px';
        splitEl.style.height = vr.height + 'px';
      } else {
        if (splitEl) splitEl.style.display = 'none';
      }
    }
    updateBar();
    } // end of throttled UI block
    if (queue.length > 1) queue.sort((a, b) => a.at - b.at);
    let due = -1;
    for (let i = 0; i < queue.length; i++) if (queue[i].at <= now) due = i;
    // drop pressure: leaky integrator (tau 300ms) - a burst of drops is visible in
    // milliseconds instead of averaging out over seconds
    dropPressure *= Math.exp((lastPressureT - now) / 300);
    lastPressureT = now;
    if (due >= 0) {
      dropped += due;
      dropPressure += due;
      for (let i = 0; i < due; i++) dropWin.push(now);
      // presentation lateness (frames arriving PAST their slot without dropping -
      // external GPU bursts look exactly like this): learn fast, forget slowly,
      // feeds back into the delay target so the buffer grows to absorb bursts
      const late = now - queue[due].at;
      lateAvg = late > lateAvg ? lateAvg * 0.7 + late * 0.3 : lateAvg * 0.985 + late * 0.015;
      present(queue[due].tex, queue[due].mid);
      queue.splice(0, due + 1); // drop in place - slice would allocate per presented frame
    }
    while (dropWin.length && dropWin[0] < now - 2000) dropWin.shift();
    // AIMD controller, evaluated EVERY frame: aggressive decrease on pressure,
    // additive recovery after a long clean stretch
    if (cfg.factor === 'auto') {
      // compositor saturation (frames late by a vsync, not yet dropped) feeds the
      // same controller: rAF stretching 1.7x past the true vsync = strain
      if (rafFloor < 90 && rafMs > rafFloor * 1.7) dropPressure += 0.02;
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
        const mode = cfg.factor === 'auto' ? (autoPenalty ? `auto-${autoPenalty}` : 'auto') : 'fixed';
        hud.textContent = [
          `${videoEl.videoWidth}x${videoEl.videoHeight}@${srcFps.toFixed(0)} → ${fpsWin.length}fps ×${effN} (${mode})`,
          `${msAvg.toFixed(1)}ms@${cfg.res}p`,
          `buf ${delayMs.toFixed(0)}`,
          `GPU ${load.toFixed(0)}%`,
          `raf ${rafMs.toFixed(1)}/${rafFloor.toFixed(1)}`,
          `late ${lateAvg.toFixed(1)}`,
          `drop ${dropped} (${dropPressure.toFixed(1)})`,
          `dup ${dups} cut ${cuts}`,
          `motion ${motionAvg.toFixed(0)}`,
          `diff ${lastStat ? lastStat.mean.toFixed(1) : '-'}/${lastStat ? lastStat.max : '-'}${lastStat && lastStat.max === 0 ? ' DRM?' : ''}`,
        ].join('  ·  ');
      } else {
        hud.textContent = `FC ${fpsWin.length}fps ×${effN} · ${msAvg.toFixed(0)}ms`;
      }
      if (panel && panel.style.display === 'block') updateStatus();
    }
  }

  // ---------- interpolation (lazy per-mid submission) ----------
  // prepPair runs once per frame pair; each mid's compute is submitted just-in-time
  // for its display slot, so present blits interleave with computes on the GPU
  // queue instead of the first mid waiting for the whole batch.
  let curJob = null;
  // queue is tiny (a handful of entries): a linear scan beats allocating a Set,
  // and this runs up to ~19x per source pair in hz mode
  function texQueued(t) {
    for (let i = 0; i < queue.length; i++) if (queue[i].tex === t) return true;
    return false;
  }
  function submitMid() {
    const k = curJob.next;
    const disp = curJob.at + curJob.ts[k] * intervalMs;
    let guard = midTexs.length; // don't clobber queued mids
    while (guard-- > 0 && texQueued(midTexs[midIdx])) midIdx = (midIdx + 1) % midTexs.length;
    const out = midTexs[midIdx];
    midIdx = (midIdx + 1) % midTexs.length;
    const t0 = performance.now();
    try { rt.runT(curJob.ts[k], out); } catch (e) { log('runT', e); curJob = null; return; }
    if ((k & 3) === 0) { // sample every 4th mid: a drain-probe promise per submit adds up at high factors
      device.queue.onSubmittedWorkDone().then(() => {
        const ms = performance.now() - t0;
        msAvg = msAvg ? msAvg * 0.85 + ms * 0.15 : ms;
      });
    }
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
      // PLL-smoothed schedule clock: decode jitter must not shake presentation.
      // Track the arrival rhythm softly (8%), resync hard on seeks/stalls (>80ms off)
      const expected = schedT + intervalMs;
      schedT = (!schedT || Math.abs(arrival - expected) > 80)
        ? arrival : expected + 0.08 * (arrival - expected);
      if (!videoEl.videoWidth || !videoEl.videoHeight) return;
      // letterboxed content (video aspect != player box, e.g. a landscape clip
      // inside a vertical shorts player) renders stretched - disengage cleanly
      // and let the raw player show; resumes by itself when aspects match again
      {
        const br = videoEl.getBoundingClientRect();
        if (br.width > 1 && br.height > 1) {
          const ea = br.width / br.height;
          const va = videoEl.videoWidth / videoEl.videoHeight;
          if (Math.abs(va - ea) / ea > 0.06) {
            if (overlay && overlay.style.opacity !== '0') overlay.style.opacity = '0';
            return;
          }
        }
      }
      const [vw, vh] = poolDims();
      ensureFrameTextures(vw, vh);
      // note on importExternalTexture (evaluated, rejected): interpolation needs the
      // PREVIOUS frame too, and external textures expire with the video frame - the
      // copy is unavoidable for history and for presenting source frames. Prep/dedup
      // reads scale with MODEL resolution, not source, so reading the external
      // texture instead of the copy saves nothing measurable.
      // NEVER overwrite a texture that is still queued for presentation or needed
      // as an interpolation input - reuse of live textures = timeline soup
      let guard = frameTex.length;
      while (guard-- > 0 && (frameTex[frameIdx] === lastTex || texQueued(frameTex[frameIdx]))) {
        frameIdx = (frameIdx + 1) % frameTex.length;
      }
      const tex = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      captureFrame(tex, vw, vh);
      // presentation delay must cover the batch compute time (own + the previous
      // batch draining), or high factors drop their early mids as already-stale.
      // Slewed ±2ms/frame so pacing never jumps.
      // lazy submission: a mid only waits for ITS OWN compute, not the whole batch;
      // sustained lateness (external GPU bursts) buys extra buffer, up to +60ms
      const burstPad = Math.min(60, Math.max(0, (lateAvg - 4) * 2));
      // light, stable runs earn a lower floor: 60ms of buffer is safety for heavy
      // factors and jittery GPUs, but a fast card doing 2-3x with no lateness only
      // needs ~40ms - streams/interactive feel snappier without dropping a frame
      const floorMs = (msAvg && msAvg < 6 && lateAvg < 3 && effN <= 3) ? 42 : 60;
      const dTarget = Math.min(180, Math.max(floorMs, 2 * (msAvg || 10) + 25 + burstPad));
      delayMs += Math.max(-2, Math.min(2, dTarget - delayMs));
      const srcAt = schedT + delayMs;
      if (cfg.compare) { cmpRing.push({ tex, at: srcAt }); if (cmpRing.length > 6) cmpRing.shift(); }
      const hzMode = cfg.factor === 'hz' && cfg.fg;
      if (!hzMode) queue.push({ tex, at: srcAt, mid: false });
      const prev = lastTex;
      if (!cfg.fg) { // frame generation off: passthrough (SR-only if enabled)
        effN = 1;
        lastUniqueTs = arrival;
      } else if (prev) {
        // the dedup readback is a full GPU->CPU roundtrip and must not block
        // the frame loop (awaiting it here starves 120fps sources of every
        // other input frame). The continuation runs a few ms later, well
        // inside the presentation buffer; pairSeq guards against a newer pair
        // (or a pool realloc / source change) having superseded this one.
        const seq = ++pairSeq;
        classifyPair(prev, tex)
          .then((r) => {
            if (!running || seq !== pairSeq) return;
            try { decidePair(r, prev, tex, arrival, srcAt, hzMode); }
            catch (e) { log('decide', e); }
          })
          .catch((e) => log('classify', e));
      } else {
        lastUniqueTs = arrival;
        if (hzMode) queue.push({ tex, at: srcAt, mid: false });
      }
      lastTex = tex;
    } catch (e) {
      if (e.name === 'OperationError') {
        log('frame skipped (decoder gap)'); // transient: no decoded frame this tick
      } else {
        log('frame error', e);
        stop();
        hud.style.display = 'block';
        hud.textContent = 'FC error: ' + (e.message || e);
      }
    } finally {
      processingFrame = false;
    }
  }

  // everything from "is this pair worth interpolating" to prepPair/curJob:
  // runs when the dedup readback lands
  function decidePair({ dup, cut }, prev, tex, arrival, srcAt, hzMode) {
        if (cut) { cuts++; lastUniqueTs = arrival; if (hzMode) queue.push({ tex, at: srcAt, mid: false }); }
        else if (cfg.anime && dup) { dups++; if (hzMode) queue.push({ tex, at: srcAt, mid: false }); }
        else {
          const du = arrival - lastUniqueTs;
          if (du > 5 && du < 500) uniqueIntervalMs = uniqueIntervalMs * 0.85 + du * 0.15;
          lastUniqueTs = arrival;
          const ms = msAvg || 10;
          let n, run = true, hzTs = null;
          if (cfg.factor === 'hz') {
            // display-match: one output frame per vsync tick, exactly on the grid.
            // 24fps@60Hz -> alternating 2/3 mids per source interval (x2.5); source
            // frames display only when a tick lands within 8% of them.
            // the raf-floor estimate jitters (3.2ms "312Hz" moments) - snap it to
            // the nearest REAL refresh rate so the divisor grid stays sane
            const RATES = [240, 165, 144, 120, 100, 75, 60];
            const rawHz = (rafFloor > 2 && rafFloor < 90) ? 1000 / rafFloor : 60;
            let snapHz = 60;
            for (const r of RATES) if (Math.abs(rawHz - r) / r < Math.abs(rawHz - snapHz) / snapHz) snapHz = r;
            const vs = 1000 / snapHz;
            // high-Hz displays: full rate is unaffordable AND unnecessary - any
            // divisor of the display rate keeps a perfect tick grid. Climb to the
            // highest divisor that fits the GPU budget and the x6 product cap.
            const srcFps = 1000 / uniqueIntervalMs;
            let m = 1;
            // hz mode dares past the x6 product cap: the tick grid keeps pacing
            // honest and the budget governor steps down a divisor when needed.
            // x20 ceiling: anime-on-twos has ~12 UNIQUE fps, so a 240Hz grid
            // means 19 mids per real pair (t-step 0.05). The budget governor is
            // the real guard; this cap only fences absurdity.
            while (m < 10 && ((1000 / (vs * m) - srcFps) * ms > 0.85 * 1000
                              || 1000 / (vs * m) > srcFps * 20 + 1)) m++;
            const vsOut = vs * m;
            const T0 = schedT - intervalMs + delayMs, T1 = T0 + intervalMs;
            if (hzNext < T0 - vsOut || hzNext > T1 + vsOut) hzNext = T0 + vsOut; // (re)sync
            hzTs = [];
            while (hzNext < T1 - 0.25 * vsOut) {
              const t = (hzNext - T0) / intervalMs;
              if (t <= 0.08) queue.push({ tex: prev, at: hzNext, mid: false });
              else if (t < 0.97) hzTs.push(t);
              hzNext += vsOut;
            }
            n = hzTs.length + 1;
            if (hzTs.length === 0 || hzTs.length * ms > uniqueIntervalMs * 0.9) {
              queue.push({ tex, at: srcAt, mid: false }); // can't afford / nothing due
              run = false;
            }
          } else if (cfg.factor === 'auto') {
            // smart auto: as much as fits in 85% of the per-unique-frame budget,
            // but never past the display refresh (extra frames would be thrown away)
            n = 6;
            while (n > 2 && (n - 1) * ms > uniqueIntervalMs * 0.85) n--;
            // cap by what the compositor actually presents, no optimism margin
            const dispHz = rafMs > 1 ? 1000 / rafMs : 60;
            while (n > 2 && (1000 / uniqueIntervalMs) * n > dispHz) n--;
            n = Math.max(2, n - autoPenalty); // drop-rate feedback (see pump)
            // fast scenes: interpolation artifacts scale with motion while the eye
            // can't rate smoothness anyway - cap the factor by measured motion
            const mcap = motionAvg > 45 ? 2 : motionAvg > 28 ? 3 : motionAvg > 16 ? 4 : 6;
            if (n > mcap) n = mcap;
            if ((n - 1) * ms > uniqueIntervalMs * 1.1) { run = false; autoSkipT = arrival; } // even 2x won't fit
          } else {
            // fixed by the user = a CEILING: under sustained overload we step down
            // to what actually fits the frame budget (the overload plate explains),
            // because piling up the queue looks far worse than a lower factor
            n = cfg.factor;
            while (n > 2 && (n - 1) * ms > uniqueIntervalMs * 0.9) n--;
            if ((n - 1) * ms > uniqueIntervalMs * 1.15) run = false; // even 2x won't fit
          }
          effN = n;
          if (run && switching && hzTs) queue.push({ tex, at: srcAt, mid: false });
          if (run && !switching) {
            flushJob(); // leftovers of the previous pair go out before the new prep
            try {
              rt.prepPair(prev, tex);
              let ts = hzTs;
              if (!ts) { ts = []; for (let k = 1; k < n; k++) ts.push(k / n); }
              curJob = { ts, next: 0, at: schedT - intervalMs + delayMs };
            } catch (e) {
              log('prep', e);
              if (hzTs) queue.push({ tex, at: srcAt, mid: false });
            }
          }
        }
  }

  // ---------- lifecycle / UI ----------
  // cross-origin video taints the pixel path (SecurityError on copy). Our DNR rule
  // injects ACAO:* on media responses, so reloading the element in CORS mode makes
  // it readable - one reload, playback position preserved, reverted on failure.
  async function makeReadable(v) {
    if (v.crossOrigin === 'anonymous') throw new Error('video unreadable even with CORS');
    const t = v.currentTime, playing = !v.paused;
    v.crossOrigin = 'anonymous';
    v.load();
    try {
      await new Promise((res, rej) => {
        const ok = () => { cleanup(); res(); };
        const bad = () => { cleanup(); rej(new Error('CDN refuses CORS for this video')); };
        const timer = setTimeout(bad, 8000);
        const cleanup = () => {
          clearTimeout(timer);
          v.removeEventListener('loadeddata', ok);
          v.removeEventListener('error', bad);
        };
        v.addEventListener('loadeddata', ok);
        v.addEventListener('error', bad);
      });
    } catch (e) {
      v.removeAttribute('crossorigin'); // put the player back the way it was
      v.load();
      v.currentTime = t;
      if (playing) v.play().catch(() => {});
      throw e;
    }
    v.currentTime = t;
    if (playing) v.play().catch(() => {});
    // loadeddata != decoded frame: copying right away throws "no back resource".
    // Wait for a real presented frame (rVFC), with a timeout so a stalled decoder
    // can't wedge the start path.
    await new Promise((res) => {
      let done = false;
      const fin = () => { if (!done) { done = true; res(); } };
      v.requestVideoFrameCallback(() => fin());
      setTimeout(fin, 1500);
    });
    log('video reloaded with CORS - pixels readable now');
  }

  function onSrcChange() {
    queue = []; curJob = null; lastTex = null; cmpRing = []; schedT = 0; lastUniqueTs = 0; hzNext = 0;
    pairSeq++; // kill in-flight classify continuations from the dead stream
    if (overlay) {
      // hide INSTANTLY: a fade would blend the dead stream's last frame over the
      // new one for 250ms. present() restores the transition on the next real frame
      overlay.style.transition = 'none';
      overlay.style.opacity = '0';
    }
  }
  let srcWatchEl = null;
  async function start(v) {
    if (running && videoEl === v) return; // re-entry insurance: never double-arm the rVFC/rAF loops
    videoEl = v;
    if (srcWatchEl !== v) {
      if (srcWatchEl) srcWatchEl.removeEventListener('emptied', onSrcChange);
      srcWatchEl = v;
      v.addEventListener('emptied', onSrcChange);
    }
    await ensureRuntime();
    ensureOverlay();
    positionOverlay();
    overlay.style.opacity = '0';
    overlay.style.display = 'block';
    // native players: we own clicks (play/pause + our fullscreen). Sites with DOM
    // controls (YouTube etc.) keep the overlay transparent to the pointer.
    overlay.style.pointerEvents = videoEl.controls ? 'auto' : 'none';
    // seed the canvas with the current video frame so the reveal is seamless -
    // no black flash while the first interpolated frames are still in flight
    const [vw, vh] = videoEl.videoWidth && videoEl.videoHeight ? poolDims() : [0, 0];
    if (vw && vh) {
      ensureFrameTextures(vw, vh);
      const seed = frameTex[frameIdx];
      frameIdx = (frameIdx + 1) % frameTex.length;
      try {
        captureFrame(seed, vw, vh);
        present(seed, false);
      } catch (e) {
        if (e.name === 'SecurityError' || String(e).includes('cross-origin')) {
          hud.style.display = 'block';
          hud.textContent = 'FC: video lacks CORS - reloading…';
          await makeReadable(videoEl); // throws a friendly error if the CDN refuses
          // seed is cosmetic (seamless fade-in): if the decoder still has no frame,
          // skip it - the rVFC pipeline below presents the first real frame anyway
          try {
            captureFrame(seed, vw, vh);
            present(seed, false);
          } catch (e2) { log('seed skipped', e2.name); }
        } else if (e.name === 'OperationError') {
          log('seed skipped', e.name); // no decoded frame yet - rVFC will deliver
        } else throw e;
      }
    }
    if (cfg.sr) ensureSR().catch(e => log('sr', e));
    queue = []; lastTex = null; curJob = null; schedT = 0; hzNext = 0;
    dropped = 0; dups = 0; cuts = 0;
    running = true;
    hud.style.display = 'block';
    if (sys.integrated) {
      advise('⚠ Chrome is running on the integrated GPU (' + sys.gpu + '). For full speed: '
        + 'Windows Settings → Display → Graphics → Chrome → High performance, '
        + 'then restart Chrome.', 14000);
    }
    videoEl.requestVideoFrameCallback(onFrame);
    requestAnimationFrame(pump);
    btn.style.background = 'rgba(25,195,125,.9)';
  }
  function stop() {
    running = false;
    if (overlay) { // fade out, then release - the raw video underneath is identical
      overlay.style.opacity = '0';
      setTimeout(() => { if (!running && overlay) overlay.style.display = 'none'; }, 260);
    }
    hud.style.display = 'none';
    if (wm) wm.style.display = 'none';
    if (bar) bar.style.display = 'none';
    overSince = 0;
    if (warnEl) warnEl.style.opacity = '0';
    if (splitEl) splitEl.style.display = 'none';
    queue = []; lastTex = null; curJob = null;
    btn.style.background = '';
  }

  function biggestVideo() {
    // rank by VISIBLE area in the viewport, not raw size: virtualized feeds
    // (TikTok) keep several same-sized players mounted and rotate them through
    // the viewport - an off-screen one must never win. Playing beats paused.
    let best = null, score = 0;
    for (const v of document.querySelectorAll('video')) {
      if (v.readyState < 2) continue;
      const r = v.getBoundingClientRect();
      const vis = Math.max(0, Math.min(r.bottom, innerHeight) - Math.max(r.top, 0))
                * Math.max(0, Math.min(r.right, innerWidth) - Math.max(r.left, 0));
      const s = vis * (v.paused ? 0.5 : 1);
      if (s > score) { score = s; best = v; }
    }
    return best;
  }

  // live system/status readout in the settings panel: adapter, f16, fps, cost,
  // our estimated GPU load (interp time vs the per-unique-frame budget), VRAM
  function updateStatus() {
    const st = panel && panel.querySelector('#fcStatus');
    if (!st) return;
    const srState = cfg.sr ? (!sys.f16 ? 'unavailable (no f16)' : (sr ? 'on x2' : 'loading…')) : 'off';
    const lines = [`GPU: ${sys.gpu}${sys.integrated ? ' ⚠ INTEGRATED' : ''}`,
      `f16: ${sys.f16 ? 'yes' : 'NO (slow path)'} · model: ${rtModel ? MODELS[rtModel] : MODELS[cfg.model] || cfg.model}`,
      `FG: ${cfg.fg ? 'on' : 'OFF'} · SR: ${srState}`,
      `HDR: ${!sys.hdrOk ? 'display not HDR' : (cfg.hdr ? (sys.hdrOn ? 'on (ITM)' : 'failed, SDR') : 'off')}`,
      `status: ${running ? 'running' : 'stopped'}`];
    if (running) {
      const [mw, mh] = SIZES[cfg.res];
      const vramMB = (texW * texH * 4 * frameTex.length + mw * mh * 4 * midTexs.length) / 1048576;
      const load = uniqueIntervalMs > 1 ? Math.min(100, msAvg * Math.max(0, effN - 1) / uniqueIntervalMs * 100) : 0;
      lines.push(
        `out: ${fpsWin.length}fps · factor x${effN}${cfg.factor === 'auto' ? ' (auto)' : ' (fixed)'}`,
        `display: ~${rafMs > 1 ? (1000 / rafMs).toFixed(0) : '-'}Hz`,
        `mid: ${msAvg.toFixed(1)}ms @ ${cfg.res}p`,
        `GPU load (ours, est.): ~${load.toFixed(0)}%`,
        `VRAM textures: ~${vramMB.toFixed(0)}MB · queue ${queue.length}`);
    }
    st.textContent = lines.join('\n');
  }

  // hot-swap the runtime on quality change: stop/start reseeded the canvas with a
  // LIVE frame while the pipeline serves ~delayMs-old ones - time visibly jumped
  // forward and snapped back. Instead: drain the in-flight batch, drop queued mids
  // (source frames keep presenting), rebuild rt at the new size, relearn timing.
  async function switchRes() {
    switching = true; // gates onFrame/runPair: NO new mids while textures are being replaced
    try {
      // loop: the user may flip the select again mid-rebuild - converge on the latest
      while (rtRes !== cfg.res) {
        curJob = null; // abandon un-submitted mids of the old pair
        queue = queue.filter((it) => !it.mid);
        msAvg = 0; // cost at the new size is different - relearn
        await ensureRuntime();
        queue = queue.filter((it) => !it.mid); // stragglers that slipped in mid-rebuild
      }
    } finally { switching = false; }
  }

  // keep the whole panel on screen - it grows when "advanced" unfolds
  function clampPanel() {
    if (!panel || panel.style.display !== 'block') return;
    const r = panel.getBoundingClientRect();
    if (r.bottom > innerHeight - 10) panel.style.top = Math.max(10, innerHeight - r.height - 10) + 'px';
    if (r.right > innerWidth - 10) panel.style.left = Math.max(10, innerWidth - r.width - 10) + 'px';
  }

  function buildPanel() {
    panel = document.createElement('div');
    panel.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'background:rgba(14,15,17,.96); color:#ddd; border:1px solid rgba(255,255,255,.14);'
      + 'border-radius:14px; box-shadow:0 8px 32px rgba(0,0,0,.5);'
      + 'padding:14px 16px; font:12px/1.5 system-ui; display:none; width:270px; box-sizing:border-box;'
      + 'max-height:calc(100vh - 20px); overflow-y:auto; overscroll-behavior:contain;';
    panel.innerHTML = `
      <div class="fc-title">Framecast <span style="color:#667;font:400 10px system-ui">v${VERSION}</span></div>
      <label class="fc-row"><span>Smoothness<small>neural frame generation</small></span>
        <input class="fc-sw" type="checkbox" id="fcFG"></label>
      <label class="fc-row"><span>Sharpness<small>2x upscale of inserted frames</small></span>
        <input class="fc-sw" type="checkbox" id="fcSR"></label>
      <label class="fc-row"><span>HDR<small>brightness expansion (needs an HDR display)</small></span>
        <input class="fc-sw" type="checkbox" id="fcHDR"></label>
      <label class="fc-row"><span>Quality<small>GPU load</small></span>
        <select class="fc-sel" id="fcRes">
          <option value="288">super eco</option><option value="360">eco</option>
          <option value="480">balanced</option>
          <option value="720">max</option>
          <option value="1080">ultra</option>
        </select></label>
      <details class="fc-details">
        <summary>advanced</summary>
        <label class="fc-row"><span>Frame factor</span>
          <select class="fc-sel" id="fcFactor">
            <option value="auto">auto</option>
            <option value="hz">display Hz</option>
            <option value="2">2×</option><option value="3">3×</option>
            <option value="4">4×</option><option value="5">5×</option>
            <option value="6">6×</option>
          </select></label>
        <label class="fc-row"><span>Model<small>interpolation weights</small></span>
          <select class="fc-sel" id="fcModel">
            <option value="v6">v6 (stable)</option>
            <option value="v7s">v7 small</option>
          </select></label>
        <label class="fc-row"><span>Anime dedup<small>detect frames drawn on twos</small></span>
          <input class="fc-sw" type="checkbox" id="fcAnime"></label>
        <label class="fc-row"><span>Hover controls</span>
          <input class="fc-sw" type="checkbox" id="fcHover"></label>
        <label class="fc-row"><span>FPS counter<small>badge in the top-left</small></span>
          <input class="fc-sw" type="checkbox" id="fcFps"></label>
        <label class="fc-row"><span>Subtitle guard<small>static regions are not warped</small></span>
          <input class="fc-sw" type="checkbox" id="fcGuard"></label>
        <label class="fc-row"><span>Compare<small>original / FC split slider</small></span>
          <input class="fc-sw" type="checkbox" id="fcCompare"></label>
        <label class="fc-row"><span>Debug<small>border + telemetry</small></span>
          <input class="fc-sw" type="checkbox" id="fcDebug"></label>
        <hr style="border:none;border-top:1px solid rgba(255,255,255,.1);margin:8px 0">
        <div id="fcStatus" style="font:11px/1.6 monospace;color:#9c9;white-space:pre">-</div>
      </details>`;
    document.body.appendChild(panel);
    panel.querySelector('.fc-details').addEventListener('toggle', () => requestAnimationFrame(clampPanel));
    const F = panel.querySelector('#fcFactor'), R = panel.querySelector('#fcRes');
    const A = panel.querySelector('#fcAnime'), D = panel.querySelector('#fcDebug');
    const Hv = panel.querySelector('#fcHover'), Cm = panel.querySelector('#fcCompare');
    syncPanel();
    F.onchange = () => { cfg.factor = (F.value === 'auto' || F.value === 'hz') ? F.value : +F.value; overSince = 0; saveCfg(); };
    const Md = panel.querySelector('#fcModel');
    Md.onchange = () => { cfg.model = Md.value; saveCfg(); }; // rebuild rides the storage listener, like res
    A.onchange = () => { cfg.anime = A.checked; saveCfg(); };
    D.onchange = () => { cfg.debug = D.checked; saveCfg(); };
    Hv.onchange = () => { cfg.hoverReveal = Hv.checked; saveCfg(); };
    const Fp = panel.querySelector('#fcFps');
    Fp.onchange = () => { cfg.showFps = Fp.checked; saveCfg(); };
    const Gd = panel.querySelector('#fcGuard');
    Gd.onchange = async () => { // the guard is baked into the flow kernel - rebuild
      cfg.guard = Gd.checked; saveCfg();
      if (running && !toggling) {
        toggling = true;
        try { rtRes = -1; await switchRes(); }
        catch (e) { log('guard switch', e); }
        finally { toggling = false; }
      } else { rtRes = -1; }
    };
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

  async function toggleFC() {
    if (toggling) return; // start/stop in flight - spam-proof
    if (!btn) injectUI(); // popup can toggle before the in-page UI ever booted
    toggling = true;
    try {
      if (running) { stop(); return; }
      const v = biggestVideo();
      if (!v) { hud.style.display = 'block'; hud.textContent = 'FC: no video found'; return; }
      try { await start(v); } catch (e) { hud.style.display = 'block'; hud.textContent = 'FC error: ' + (e.message || e); log(e); }
    } finally { toggling = false; }
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
      /* NO backdrop-filter on anything hovering over the RUNNING video: the
         compositor re-blurs the region every frame of a 100+fps canvas - that
         alone janks playback exactly while the cursor summons the UI */
      .fc-side{position:fixed;z-index:2147483647;width:38px;height:38px;border-radius:50%;
        border:none;background:rgba(18,18,20,.88);color:#fff;cursor:pointer;display:none;
        font:600 12px/1 system-ui;box-shadow:0 2px 12px rgba(0,0,0,.4);
        transition:background .15s,transform .15s}
      .fc-side:hover{transform:scale(1.1);background:rgba(45,45,45,.85)}
      .fc-title{font:600 14px system-ui;color:#fff;display:flex;align-items:center;gap:7px;margin-bottom:8px}
      .fc-title::before{content:'';width:8px;height:8px;border-radius:50%;background:#19c37d}
      .fc-row{display:flex;justify-content:space-between;align-items:center;gap:18px;
        padding:7px 0;margin:0;border:0;width:auto;cursor:pointer;
        font:12px/1.4 system-ui;color:#e8e8e8;text-align:left}
      .fc-row>span{display:block;flex:1 1 auto;min-width:0;margin:0;padding:0;
        font:12px/1.4 system-ui;color:#e8e8e8;text-align:left;
        letter-spacing:normal;text-transform:none;white-space:normal}
      .fc-row small{display:block;color:#8a8f98;font:400 10px/1.3 system-ui;
        margin:1px 0 0;padding:0;letter-spacing:normal;text-transform:none}
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
    // single line, pinned to the video's TOP-LEFT; plain dark bar, white text
    hud.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483647;'
      + 'color:#fff; font:11px/1.5 ui-monospace,monospace; background:rgba(0,0,0,.72);'
      + 'padding:3px 9px; white-space:normal; pointer-events:none; display:none;';
    wm = document.createElement('div');
    // permanent brand mark: bottom-left inside the player, bare white text
    wm.style.cssText = 'position:fixed; left:0; top:0; z-index:2147483645;'
      + 'color:#fff; font:600 12px system-ui; opacity:.75; pointer-events:none;'
      + 'text-shadow:0 1px 3px rgba(0,0,0,.8); display:none;';
    wm.textContent = 'Framecast';
    document.body.appendChild(wm);
    buildPanel();
    ensureBar();
    btn.onclick = toggleFC;
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

  // toolbar popup protocol: status snapshot + remote toggle. With all_frames every
  // frame gets the message; the RUNNING frame answers instantly, a frame that merely
  // has a video answers after 120ms, video-less frames after 250ms - first response
  // wins, so the most relevant frame speaks for the tab.
  const VERSION = '1.0.0';
  try {
    chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
      if (msg && msg.type === 'fcStatus') {
        const v = videoEl || biggestVideo();
        const respond = async () => { await probeAdapter(); try { sendResponse({
          version: VERSION, gpu: sys.gpu, integrated: sys.integrated, f16: sys.f16,
          hasVideo: !!v, running, fps: fpsWin.length, effN,
          ms: +(msAvg || 0).toFixed(1), res: cfg.res, factor: cfg.factor,
          drops: dropped, model: rtModel ? MODELS[rtModel] : MODELS[cfg.model] || cfg.model,
        }); } catch {} };
        if (running) respond();
        else setTimeout(respond, v ? 120 : 250);
        return true;
      }
      if (msg && msg.type === 'fcToggle') {
        const v = videoEl || biggestVideo();
        if (!running && !v) return undefined; // let a frame that HAS video take it
        toggleFC().then(() => { try { sendResponse({ running }); } catch {} });
        return true;
      }
    });
  } catch { /* messaging unavailable in some frames */ }

  const boot = () => {
    if (btn) return;
    if (document.querySelector('video')) injectUI();
  };
  boot();
  new MutationObserver(boot).observe(document.documentElement, { childList: true, subtree: true });
})();
