// The pipeline of one page: a <video> into the engine's stream, the Pacer's ticks onto the overlay canvas. Source frames
// arrive through requestVideoFrameCallback, display ticks through requestAnimationFrame; the Pacer says what to present
// on each tick and what to generate for the coming ones (the engine's ARCHITECTURE.md, the Pacer). Everything GPU-side lives on
// the backend's device: the stream's slots, the canvas, the upscaler.
import { log } from './log.js';
import { loadEngine, tuneStore, SIZES } from './engine.js';
import { makeReadable, isCorsError } from './capture.js';
import { Presenter } from './present.js';
import { Upscaler } from './upscale.js';
import { t } from './i18n.js';

const url = p => chrome.runtime.getURL(p);
const ENGINE_URLS = {
  graph: url('engine/graph.json'), manifest: url('engine/flyhighfg.json'), weights: url('engine/flyhighfg.bin'),
};
const SR_URLS = { weights: url('assets/rt_sr.bin'), manifest: url('assets/rt_sr.json') };
const MAX_INPUTS = 24, MAX_OUTPUTS = 20;
// an output not named by a present within this time after its due was dropped or superseded: the slot is free again
const OUTPUT_HOLD_MS = 250;
const UI_INTERVAL_MS = 1000 / 15;
const HUD_INTERVAL_MS = 400;
const TUNE_IDLE_POLL_MS = 3000, TUNE_FORCE_AFTER_MS = 120000;
const rateNumber = value => Number(value.toFixed(2));

function planWarning(plan) {
  switch (plan.state) {
    case 'no_two_x_gpu': return t('state_no_two_x_gpu');
    case 'no_two_x_display': return t('state_no_two_x_display');
    case 'invalid_target': return t('state_invalid_target');
    case 'probing_gpu': return t('state_probing_gpu');
  }
  if (plan.clamp === 'gpu' && plan.requested_hz != null && plan.output_hz != null) {
    return t('warning_gpu_clamp', {
      requested: rateNumber(plan.requested_hz), output: rateNumber(plan.output_hz),
    });
  }
  return plan.warning;
}

export const STREAM_KEYS = ['res', 'anime', 'periodicRepair', 'cadenceSmoothing', 'guard', 'renderCubic', 'sr'];
export const PACER_KEYS = ['factor', 'targetFps', 'fpsLimit', 'limitToDisplay', 'fillDisplay', 'motionAdaptive', 'anime', 'cadenceSmoothing', 'periodicRepair'];
export const PRESENT_KEYS = ['hdr', 'sharpness', 'pictureMode'];
export const MODE_KEYS = ['fg', 'sr', 'hdr', 'sharpness', 'pictureMode', 'compare'];

const isIntegrated = name => /intel|iris|uhd|graphics 6|vega|radeon\(tm\) graphics|apu/i.test(name)
  && !/nvidia|geforce|rtx|gtx|radeon rx|arc a|arc b/i.test(name);

export class Player {
  constructor({ settings, overlay, subtitles, notices, hooks, benchMode = false }) {
    this.settings = settings;
    this.overlay = overlay;
    this.subtitles = subtitles;
    this.notices = notices;
    this.hooks = hooks;
    this.benchMode = benchMode;
    this.backend = null;
    this.backendLoading = null;
    this.kernelsReady = null;
    this.interpolationReady = false;
    this.blocked = null;
    this.sys = { gpu: '-', f16: false, integrated: false, hdrOk: false, hdrOn: false };
    try { this.sys.hdrOk = !!(window.matchMedia && matchMedia('(dynamic-range: high)').matches); } catch { /* no media queries */ }
    this.video = null;
    this.watched = null;
    this.running = false;
    this.epoch = 0;
    this.rvfc = { id: null, video: null };
    this.rafId = null;
    this.stream = null;
    this.streamKey = '';
    this.spec = null;
    this.pacer = null;
    this.pacerKey = '';
    this.recording = false;
    this.source = null;
    this.pictureInput = null;
    this.costs = null;
    this.costSink = null;
    this.presenter = null;
    this.upscaler = null;
    this.poolDims = null;
    this.prev = -1;
    this.recent = [];
    this.outputSlots = [];
    this.compareRing = [];
    this.lastShownSource = -1;
    this.busyEpoch = null;
    this.lastPts = null;
    this.lastPresentedFrames = null;
    this.sourceIntervalMs = 42;
    this.stats = this.freshStats();
    this.nextUiAt = 0;
    this.hudAt = 0;
    this.lastRect = null;
    this.tuneTimer = 0;
    this.tuning = false;
    this.recovering = false;
    this.onSourceChange = () => this.resetPipeline('source change', true);
    this.onRateChange = () => this.resetPipeline('rate change', false);
  }

  freshStats() {
    return { fpsWin: [], dropWin: [], drops: 0, factor: 1, inputExhausted: 0, presents: 0 };
  }

  get cfg() { return this.settings.cfg; }
  isRunning() { return this.running; }

  needsCanvas() {
    const c = this.cfg;
    return c.fg || (c.sr && !!this.backend?.f16) || (c.hdr && this.sys.hdrOk) || c.sharpness > 0 || c.pictureMode !== 'original' || c.compare;
  }

  // ---------- backend ----------
  async ensureBackend() {
    if (this.backend) return this.backend;
    if (!this.backendLoading) {
      this.backendLoading = (async () => {
        await this.settings.ready;
        if (!navigator.gpu) throw new Error(t('gpu_help'));
        // the engine would fail on the same empty answer with wgpu's backend dump instead of something a viewer can act on
        const adapter = await navigator.gpu.requestAdapter().catch(() => null);
        if (!adapter) throw new Error(t('no_adapter_help'));
        let backend;
        try {
          backend = await loadEngine(ENGINE_URLS);
        } catch (e) {
          if (e?.wasmBlocked) {
            // Chrome applies the page's CSP to the content script's world when WebAssembly compiles: without
            // 'wasm-unsafe-eval' or 'unsafe-eval' in its script-src nothing in the extension can run the engine here
            this.blocked = true;
            log(t('blocked_wasm'), e.message);
            throw new Error(t('blocked_wasm'));
          }
          throw e;
        }
        this.sys.gpu = backend.name;
        this.sys.f16 = backend.f16;
        this.sys.integrated = isIntegrated(backend.name);
        backend.device.lost
          .then(info => this.onDeviceLost(backend, info))
          .catch(e => log('device loss handler', e));
        this.backend = backend;
        log('backend', backend.kind, backend.class, backend.name, backend.timestamps ? 'timestamps' : 'no timestamps');
        return backend;
      })().finally(() => { this.backendLoading = null; });
    }
    return this.backendLoading;
  }

  // Adapter info for the popup before any device exists: no device, no weights.
  async probeAdapter() {
    if (this.backend || this.sys.gpu !== '-' || !navigator.gpu) return;
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return;
      const info = adapter.info || {};
      this.sys.gpu = info.description || [info.vendor, info.architecture].filter(Boolean).join(' ');
      this.sys.f16 = adapter.features.has('shader-f16');
      this.sys.integrated = isIntegrated(this.sys.gpu);
    } catch { /* leave unknown */ }
  }

  prepareInterpolation(backend) {
    if (!this.kernelsReady) {
      this.kernelsReady = Promise.all([backend.loadModel(), tuneStore.loaded ? null : tuneStore.load()])
        .then(() => backend.precompile().catch(e => log('precompile', e)))
        .then(() => { if (this.backend === backend) this.interpolationReady = true; })
        .catch(e => {
          if (this.backend === backend) this.kernelsReady = null;
          throw e;
        });
    }
    return this.kernelsReady;
  }

  async preload() {
    if (this.blocked) throw new Error(t('blocked_wasm'));
    const backend = await this.ensureBackend();
    if (this.cfg.fg) await this.prepareInterpolation(backend);
    return backend;
  }

  // ---------- the stream and the pacer ----------
  repairLevel() {
    const c = this.cfg;
    return c.periodicRepair === false ? 0 : c.anime ? 2 : 1;
  }

  // Video mode: one source frame of lookahead and cadence re-timing of drawings held on 2s; anime with cadence
  // smoothing and repair on.
  videoMode() {
    const c = this.cfg;
    return !!(c.anime && c.cadenceSmoothing && c.periodicRepair !== false);
  }

  buildSpec(width, height) {
    const c = this.cfg;
    const spec = { width, height, res: c.res, repair: this.repairLevel(), render: c.renderCubic ? 'bicubic' : 'bilinear',
      guard: !!c.guard, video: this.videoMode(), anime: !!c.anime, tune: null };
    spec.tune = tuneStore.get(this.backend.tuneKey(spec));
    return spec;
  }

  pacerOptions() {
    const c = this.cfg;
    const mode = c.factor === 'auto' ? 'auto' : c.factor === 'hz' ? 'display_hz'
      : c.factor === 'target' ? { target: c.targetFps } : { factor: Number(c.factor) };
    const w = this.spec?.width || 848, h = this.spec?.height || 480;
    return {
      mode,
      fpsLimit: c.factor === 'auto' ? c.fpsLimit : null,
      limitToDisplay: c.limitToDisplay !== false,
      fillDisplay: c.fillDisplay === true,
      motionAdaptive: c.motionAdaptive !== false,
      lookahead: !!this.spec?.video,
      gpuTimestamps: !!this.backend?.timestamps,
      playbackRate: this.playbackRate(),
      // until the first measurement: with the fast render and linear t a generated frame is only the render (about
      // 0.2 to 1 ms), the exact render about 1 ms at the 480 rung on a mid-range GPU. A seed far above the real cost
      // kept fixed factors from generating at all, and held Auto at x2 wherever the measurement did not land
      midCostSeedMs: this.backend?.sessionOptions(this.spec || { width: w, height: h, res: this.cfg.res }).linearT ? 1 : 3,
    };
  }

  playbackRate() {
    return Math.max(0.01, Math.abs(Number(this.video?.playbackRate) || 1));
  }

  ensurePacer(now = performance.now()) {
    const options = this.pacerOptions();
    const key = JSON.stringify(options);
    if (!this.pacer) {
      this.pacer = this.backend.createPacer(options);
      this.pacerKey = key;
      this.syncRecording();
      return;
    }
    if (key === this.pacerKey) return;
    this.pacerKey = key;
    try { this.pacer.setOptions(now, options); } catch (e) { log('pacer options', e); }
  }

  syncRecording() {
    if (!this.pacer) return;
    const wanted = this.benchMode || this.cfg.debug;
    if (wanted && !this.recording) {
      this.pacer.record(`Framegen ${this.hooks.version} ${this.backend.class} ${navigator.userAgent}`);
      this.recording = true;
    } else if (!wanted && this.recording) {
      this.pacer.stopRecording();
      this.recording = false;
    }
  }

  // The stream for the current pool size and settings; rebuilt when either changes (tens of milliseconds).
  ensureStream(width, height) {
    const spec = this.buildSpec(width, height);
    const key = JSON.stringify(spec);
    if (this.stream && key === this.streamKey) return false;
    const replacing = !!this.stream;
    const now = performance.now();
    this.freeStream();
    this.stream = this.backend.openStream(spec);
    this.source = new this.backend.adapters.VideoSource(this.backend.device, this.stream);
    this.spec = spec;
    this.streamKey = key;
    this.warmedUp = false;
    this.presenter?.invalidate();
    this.resetSlots();
    if (this.pacer) {
      this.ensurePacer(now);
      this.pacer.reset(now);
      if (replacing) this.pacer.modeChange(now);
      const pacer = this.pacer, stream = this.stream;
      this.costSink = { costSample: (...sample) =>
        this.stream === stream && this.pacer === pacer && pacer.costSample(...sample) };
      this.costs = new this.backend.adapters.CostSampler(this.costSink, this.backend.nudge);
    }
    log('stream', `${width}x${height}`, spec.video ? 'video' : 'realtime', `rung ${spec.res}`, `repair ${spec.repair}`, spec.tune ? 'tuned' : 'untuned', this.backend.class);
    if (!spec.tune) this.scheduleTune(spec);
    return true;
  }

  freeStream() {
    this.costSink = null;
    if (!this.stream) return;
    try { this.stream.free(); } catch { /* already freed */ }
    this.stream = null;
    this.source = null;
    this.streamKey = '';
  }

  releaseInterpolation() {
    if (!this.stream && !this.pacer) return;
    clearTimeout(this.tuneTimer);
    this.presenter?.invalidate();
    this.freeStream();
    this.pacer?.free();
    this.pacer = null;
    this.pacerKey = '';
    this.recording = false;
    this.costs = null;
    this.spec = null;
    this.resetSlots();
  }

  resetSlots() {
    this.prev = -1;
    this.recent = [];
    this.outputSlots = [];
    this.compareRing = [];
    this.lastShownSource = -1;
    this.lastPts = null;
    this.lastPresentedFrames = null;
  }

  // A seek, a new source, a rate change: forget the pushed frames and the output grid.
  resetPipeline(reason, hideCanvas) {
    const now = performance.now();
    this.resetSlots();
    this.poolDims = null;
    if (this.stream) { try { this.stream.reset(); } catch (e) { log('stream reset', e); } }
    if (this.pacer) {
      this.pacerKey = '';
      this.ensurePacer(now);
      this.pacer.reset(now);
    }
    if (hideCanvas) {
      this.overlay.hideInstant();
      this.overlay.resetGeometry();
    }
    if (this.cfg.debug) log('pipeline reset:', reason);
  }

  // ---------- autotune (idle only: 5 to 15 s of GPU time on the engine, 0.3 s on the previous one) ----------
  scheduleTune(spec) {
    clearTimeout(this.tuneTimer);
    const key = this.backend.tuneKey(spec);
    const t0 = performance.now();
    const tick = () => {
      if (!this.backend || tuneStore.get(key) || this.tuning) return;
      if (!this.spec || this.backend.tuneKey(this.spec) !== key) return;
      const idle = !this.running || !this.video || this.video.paused;
      if (idle || performance.now() - t0 > TUNE_FORCE_AFTER_MS) { this.runTune(spec, key); return; }
      this.tuneTimer = setTimeout(tick, TUNE_IDLE_POLL_MS);
    };
    this.tuneTimer = setTimeout(tick, 4000);
  }

  async runTune(spec, key) {
    this.tuning = true;
    const backend = this.backend;
    try {
      const t0 = performance.now();
      const table = await backend.autotune(spec);
      if (!table) { log('autotune: no timestamp queries, default kernel routing kept'); return; }
      await tuneStore.put(key, table);
      log('autotune', key, `${(performance.now() - t0).toFixed(0)} ms`);
      // the table applies at the next stream build: when idle, now
      if (this.backend === backend && this.stream && !this.running) this.streamKey = '';
    } catch (e) {
      log('autotune skipped', e);
    } finally {
      this.tuning = false;
    }
  }

  // ---------- pool size ----------
  // Presentation-sized pools for the sharp full-resolution render; with the upscaler active and a genuinely smaller
  // source, the native pixels, so it gets real low-resolution input instead of a frame already enlarged to the canvas.
  wantedPoolDims() {
    const v = this.video;
    const fw = v.videoWidth, fh = v.videoHeight;
    const s = Math.min(1, 1920 / fw, 1080 / fh);
    const sw = Math.round(fw * s), sh = Math.round(fh * s);
    const ow = this.overlay.width, oh = this.overlay.height;
    if (this.wantsUpscale() && this.backend.f16 && Upscaler.needed(sw, sh, ow, oh)) return [sw, sh];
    if (ow && oh) return [ow, oh];
    return [sw, sh];
  }

  // Dead-banded against the committed size: the canvas backing is round(rect * DPR) and player chrome, hover controls
  // or a scrollbar nudge it by a few pixels every tick, and a rebuilt stream is a visible freeze.
  poolDimsNow() {
    const dims = this.wantedPoolDims();
    const stable = this.poolDims;
    if (stable) {
      const tolW = Math.max(16, Math.round(dims[0] * 0.02)), tolH = Math.max(16, Math.round(dims[1] * 0.02));
      if (Math.abs(stable[0] - dims[0]) <= tolW && Math.abs(stable[1] - dims[1]) <= tolH) return stable;
    }
    this.poolDims = dims;
    return dims;
  }

  // ---------- presenting ----------
  pictureFrame(video) {
    const [width, height] = this.poolDimsNow();
    if (this.pictureInput?.width !== width || this.pictureInput?.height !== height) {
      this.freePictureInput();
      const { device, adapters } = this.backend;
      const texture = device.createTexture({ label: 'fc-picture-input', size: [width, height], format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
      this.pictureInput = { texture, width, height, source: new adapters.VideoSource(device, { inputTexture: () => texture }) };
    }
    const { texture, source } = this.pictureInput;
    // Capture and presentation are submitted to the same queue in order; nothing holds a picture frame for a later tick.
    source.write(0, video);
    return texture;
  }

  freePictureInput() {
    if (!this.pictureInput) return;
    this.presenter?.invalidate();
    this.pictureInput.texture.destroy();
    this.pictureInput = null;
  }

  ensurePresenter() {
    if (this.presenter) return;
    this.presenter = new Presenter(this.backend.device, this.overlay.ensure(), log);
    this.configurePresenter();
  }

  configurePresenter() {
    if (!this.presenter) return;
    this.sys.hdrOn = this.presenter.configure({ hdr: this.cfg.hdr && this.sys.hdrOk, sharpness: this.cfg.sharpness, pictureMode: this.cfg.pictureMode });
  }

  wantsUpscale() {
    return this.cfg.sr || this.cfg.pictureMode === 'detail';
  }

  ensureUpscaler() {
    if (!this.wantsUpscale() || !this.backend?.f16) return;
    if (!this.upscaler) this.upscaler = new Upscaler(this.backend.device, SR_URLS, { nudge: this.backend.nudge, log });
    this.upscaler.load().catch(e => log('SR', e));
  }

  present(texture, now, { source = -1 } = {}) {
    if (!this.presenter || !this.overlay.fitSupported) return;
    let tex = texture;
    if (this.wantsUpscale() && this.backend.f16 && this.upscaler?.ready
        && Upscaler.needed(texture.width, texture.height, this.overlay.width, this.overlay.height)) {
      tex = this.upscaler.process(texture, now, this.costSink) || texture;
    }
    let compare = null;
    if (this.cfg.compare) {
      if (!this.cfg.fg) {
        // generation off: the frame on screen is the source frame itself, so there is nothing to delay
        compare = texture;
      } else if (this.stream) {
        const ring = this.compareRing;
        while (ring.length > 1 && ring[1].at <= now) ring.shift();
        if (ring.length && ring[0].at <= now) compare = this.stream.inputTexture(ring[0].slot);
      }
    }
    const drawn = this.presenter.present(tex, { compare, splitX: this.hooks.splitX(), width: this.overlay.width, height: this.overlay.height });
    if (!drawn) return;
    if (source >= 0) this.lastShownSource = source;
    if (!this.overlay.visible) this.preparing(false);
    this.overlay.reveal();
    this.stats.presents++;
    const win = this.stats.fpsWin;
    win.push(now);
    while (win.length && win[0] < now - 1000) win.shift();
  }

  // ---------- lifecycle ----------
  cancelLoops() {
    this.epoch++;
    if (this.rvfc.id !== null && this.rvfc.video) {
      try { this.rvfc.video.cancelVideoFrameCallback(this.rvfc.id); } catch { /* gone */ }
    }
    this.rvfc = { id: null, video: null };
    if (this.rafId !== null) { try { cancelAnimationFrame(this.rafId); } catch { /* gone */ } }
    this.rafId = null;
    return this.epoch;
  }

  watch(video) {
    if (this.watched === video) return;
    if (this.watched) {
      this.watched.removeEventListener('emptied', this.onSourceChange);
      this.watched.removeEventListener('seeking', this.onSourceChange);
      this.watched.removeEventListener('ratechange', this.onRateChange);
    }
    this.watched = video;
    video.addEventListener('emptied', this.onSourceChange);
    video.addEventListener('seeking', this.onSourceChange);
    video.addEventListener('ratechange', this.onRateChange);
  }

  async start(video) {
    if (this.running && this.video === video) return;
    this.running = false;
    const epoch = this.cancelLoops();
    this.preparing(true);
    this.video = video;
    this.watch(video);
    await this.ensureBackend();
    if (epoch !== this.epoch || this.video !== video) return;
    const backend = this.backend;
    this.overlay.ensure();
    this.ensurePresenter();
    this.overlay.show();
    this.resetSlots();
    this.poolDims = null;
    this.overlay.resetGeometry();
    this.layoutOverlay(null);
    this.stats = this.freshStats();
    if (!this.cfg.fg) this.releaseInterpolation();

    if (this.needsCanvas() && video.videoWidth && video.videoHeight && this.overlay.fitSupported) {
      const fg = this.cfg.fg;
      await Promise.all([this.presenter.ready, fg ? this.prepareInterpolation(backend) : null]);
      if (epoch !== this.epoch || this.video !== video) return;
      if (fg && this.cfg.fg) {
        this.freePictureInput();
        this.ensurePacer();
        this.ensureStream(...this.poolDimsNow());
      }
      // a cross-origin video fails on its first read: find out (and reload it with CORS) before anything is shown
      try {
        this.probe(video);
      } catch (e) {
        if (isCorsError(e)) {
          this.notices.showCritical(t('cors_reload'));
          await makeReadable(video, log);
          if (epoch !== this.epoch || this.video !== video) return;
          try { this.probe(video); } catch (e2) { log('probe skipped', e2.name || e2); }
        } else if (e?.name === 'OperationError') {
          log('probe skipped (no decoded frame yet)');
        } else throw e;
      }
      if (epoch !== this.epoch || this.video !== video) return;
      if (this.cfg.fg && this.stream && !this.warmedUp) {
        const stream = this.stream;
        const t0 = performance.now();
        try { await backend.warmUp(stream); } catch (e) { if (stream === this.stream) log('warm-up skipped', e); }
        if (epoch !== this.epoch || this.video !== video) return;
        if (stream === this.stream) {
          this.warmedUp = true;
          log('warm-up', `${(performance.now() - t0).toFixed(0)} ms`);
          backend.rememberKernels();
        }
      }
    }
    this.ensureUpscaler();
    this.running = true;
    this.pacer?.reset(performance.now());
    // the first seconds stall while the browser prepares pipelines: not a GPU limit
    this.pacer?.modeChange(performance.now());
    if (!this.needsCanvas()) { this.overlay.passthrough(); this.preparing(false); }
    this.notices.setHud('', this.cfg.debug || this.cfg.showFps);
    if (this.sys.integrated) {
      this.notices.advise(t('integrated_advice', { gpu: this.sys.gpu || t('unknown_gpu') }), 14000);
    }
    this.hooks.onStateChange(true);
    this.armVideoFrames(epoch);
    this.armTicks(epoch);
    if (!this.cfg.fg && video.paused) this.redrawPaused();
  }

  // Reads the current frame into a slot without presenting it: the canvas stays transparent over the playing video until
  // the Pacer's first present, so the kernels' first-use compilation (the warm-up) never shows as a frozen frame.
  probe(video) {
    if (!this.needsCanvas()) return;
    if (!this.cfg.fg) { this.pictureFrame(video); return; }
    if (!this.stream) return;
    const slot = this.stream.nextInput([]);
    if (slot < 0) return;
    this.source.write(slot, video);
  }

  preparing(on) {
    if (this.isPreparing === on) return;
    this.isPreparing = on;
    this.hooks.onPreparing?.(on);
  }

  stop() {
    const wasRunning = this.running;
    this.preparing(false);
    // kernels compiled while running (the classifier, the cadence composite, a tuned conv variant) join the list
    if (this.stream) this.backend?.rememberKernels();
    this.running = false;
    this.cancelLoops();
    this.busyEpoch = null;
    this.releaseInterpolation();
    this.freePictureInput();
    this.overlay.fadeOut(() => !this.running);
    this.notices.hideAll();
    this.subtitles.sync({ active: false, video: this.video, canvas: this.overlay.canvas, canvasVisible: false });
    if (wasRunning) this.hooks.onStateChange(false);
  }

  armVideoFrames(epoch) {
    const video = this.video;
    if (!this.running || epoch !== this.epoch || !video) return;
    const id = video.requestVideoFrameCallback((now, metadata) => {
      if (this.rvfc.id === id) this.rvfc = { id: null, video: null };
      if (!this.running || epoch !== this.epoch || this.video !== video) return;
      this.onVideoFrame(metadata, epoch);
    });
    this.rvfc = { id, video };
  }

  armTicks(epoch) {
    if (!this.running || epoch !== this.epoch) return;
    const id = requestAnimationFrame(now => {
      if (this.rafId === id) this.rafId = null;
      if (!this.running || epoch !== this.epoch) return;
      this.onTick(now, epoch);
    });
    this.rafId = id;
  }

  // ---------- source frames ----------
  async onVideoFrame(metadata, epoch) {
    this.armVideoFrames(epoch);
    const video = this.video;
    if (video.videoWidth !== this.overlay.sourceWidth || video.videoHeight !== this.overlay.sourceHeight) this.layoutOverlay(null);
    if (!this.overlay.fitSupported) return;
    if (!this.needsCanvas()) { this.overlay.passthrough(); return; }
    if (this.busyEpoch !== null) return;
    this.busyEpoch = epoch;
    let stream;
    try {
      const now = performance.now();
      if (!video.videoWidth || !video.videoHeight) return;
      if (this.cfg.fg && !this.interpolationReady && !this.kernelsReady) {
        this.prepareInterpolation(this.backend).catch(e => {
          if (this.running && this.cfg.fg && epoch === this.epoch) this.fail(t('framegen_error', { detail: e?.message || e }), e);
        });
      }
      if (!this.cfg.fg || !this.interpolationReady) {
        this.present(this.pictureFrame(video), now);
        this.stats.factor = 1;
        return;
      }
      if (!this.pacer) {
        this.freePictureInput();
        this.ensurePacer();
        this.pacer.modeChange(performance.now());
      }
      const [w, h] = this.poolDimsNow();
      this.ensureStream(w, h);
      stream = this.stream;
      const pts = Number.isFinite(metadata?.mediaTime) ? metadata.mediaTime * 1000 : now * this.playbackRate();
      const presented = Number(metadata?.presentedFrames);
      let skipped = 0;
      if (Number.isFinite(presented)) {
        if (this.lastPresentedFrames !== null && presented - this.lastPresentedFrames > 1) skipped = presented - this.lastPresentedFrames - 1;
        this.lastPresentedFrames = presented;
      }
      if (this.lastPts !== null && pts > this.lastPts) {
        const d = (pts - this.lastPts) / (skipped + 1);
        if (d > 1 && d < 500) this.sourceIntervalMs = this.sourceIntervalMs * 0.9 + d * 0.1;
      }
      this.lastPts = pts;

      const slot = stream.nextInput(this.inUse());
      if (slot < 0) {
        if (stream.report().inputs < MAX_INPUTS) { stream.addInput(); this.presenter?.invalidate(); }
        this.stats.inputExhausted++;
        return;
      }
      this.source.write(slot, video);
      // the first frame after a start goes up at once: the canvas appears with the picture the video shows and holds
      // it for the presentation delay, after which the Pacer's frames follow it forwards. Appearing with the Pacer's
      // first frame instead stepped the picture back by that delay, and a fade blended the two timelines
      if (!this.overlay.visible) this.present(stream.inputTexture(slot), now, { source: slot });
      let kind = 'motion', motion;
      if (this.prev >= 0) {
        const c = await this.backend.nudge.wrap(stream.classify(this.prev, slot));
        if (!this.running || epoch !== this.epoch || stream !== this.stream) return;
        kind = c.kind;
        motion = c.motion;
      }
      const t = performance.now();
      // media time as it is: the Pacer has the playback rate. Dividing here as well made a 60 fps video at 1.5x read as
      // 135 fps, and Auto stopped generating
      const flush = this.pacer.sourceFrame(t, { id: slot, ptsMs: pts, kind, motion, skipped });
      this.generate(t, flush);
      const p = this.costs.push(stream, slot, pts);
      if (p.statsPending) await this.backend.nudge.wrap(stream.stats());
      if (epoch !== this.epoch || stream !== this.stream) return;
      this.prev = slot;
      const history = Math.max(3, Math.ceil(this.pacer.delayMs() / Math.max(1, this.sourceIntervalMs / this.playbackRate())) + 3);
      this.recent.push(slot);
      while (this.recent.length > history) this.recent.shift();
      if (this.cfg.compare) {
        this.compareRing.push({ slot, at: t + this.pacer.delayMs() });
        while (this.compareRing.length > history) this.compareRing.shift();
      }
    } catch (e) {
      if (epoch !== this.epoch || (stream && stream !== this.stream)) return;
      if (e?.name === 'OperationError') {
        log('frame skipped (decoder gap)');
      } else if (isCorsError(e)) {
        this.fail(t('cors_unreadable'), e);
      } else {
        this.fail(t('framegen_error', { detail: e?.message || e }), e);
      }
    } finally {
      if (this.busyEpoch === epoch) this.busyEpoch = null;
    }
  }

  inUse() {
    const used = this.recent.slice();
    for (const c of this.compareRing) used.push(c.slot);
    if (this.lastShownSource >= 0) used.push(this.lastShownSource);
    return used;
  }

  fail(message, error) {
    log(message, error);
    this.stop();
    this.notices.showCritical(message);
  }

  // ---------- generated frames ----------
  acquireOutput(now, dueMs) {
    const slots = this.outputSlots;
    const count = this.stream.report().outputs;
    while (slots.length < count) slots.push({ token: null, freeAt: 0 });
    let slot = slots.findIndex(output => output.freeAt <= now);
    if (slot < 0) {
      if (count >= MAX_OUTPUTS) return -1;
      slot = this.stream.addOutput();
      slots[slot] = { token: null, freeAt: 0 };
      this.presenter?.invalidate();
    }
    slots[slot].token = null;
    slots[slot].freeAt = Math.max(now, dueMs) + OUTPUT_HOLD_MS;
    return slot;
  }

  generate(now, requests) {
    if (!requests || !requests.length) return;
    for (const r of requests) {
      const out = this.acquireOutput(now, r.due_ms);
      if (out < 0) {
        this.pacer.skipped(r.id);
        continue;
      }
      let written = false;
      try { written = this.costs.generate(this.stream, r, out); } catch (e) { log('generate', e); }
      const slot = this.outputSlots[out];
      if (written) {
        // The Pacer's output ID is an opaque u32. A late tick must name the frame, even after the texture is recycled.
        slot.token = r.id >>> 0;
        this.pacer.generated(now, r.id, slot.token);
      } else {
        slot.freeAt = 0;
        this.pacer.skipped(r.id);
      }
    }
  }

  // ---------- display ticks ----------
  onTick(now, epoch) {
    try { this.tickBody(now); } catch (e) { log('tick', e); }
    finally { this.armTicks(epoch); }
  }

  tickBody(now) {
    const video = this.video;
    if (video && !video.isConnected) { this.stop(); this.hooks.onVideoGone(); return; }
    if (now >= this.nextUiAt) {
      this.nextUiAt = now + UI_INTERVAL_MS;
      const rect = video.getBoundingClientRect();
      this.lastRect = rect;
      this.layoutOverlay(rect);
      this.hooks.onUiTick(now, rect);
    }
    if (!this.needsCanvas()) { this.overlay.passthrough(); return; }
    if (!this.cfg.fg || !this.stream || !this.pacer) return;
    const tick = this.pacer.tick(now);
    if (tick.present) {
      const f = tick.present.frame;
      if ('output' in f) {
        const slot = this.outputSlots.findIndex(output => output.token === f.output);
        if (slot >= 0) {
          this.present(this.stream.outputTexture(slot), now);
          this.outputSlots[slot].token = null;
          this.outputSlots[slot].freeAt = 0;
        } else if (this.lastShownSource >= 0) {
          this.present(this.stream.inputTexture(this.lastShownSource), now, { source: this.lastShownSource });
        } else {
          this.overlay.passthrough();
        }
      } else {
        this.present(this.stream.inputTexture(f.source), now, { source: f.source });
      }
    }
    if (tick.dropped) {
      this.stats.drops += tick.dropped;
      for (let i = 0; i < tick.dropped; i++) this.stats.dropWin.push(now);
    }
    while (this.stats.dropWin.length && this.stats.dropWin[0] < now - 2000) this.stats.dropWin.shift();
    this.stats.factor = tick.factor;
    this.generate(now, tick.generate);
    if (now - this.hudAt > HUD_INTERVAL_MS) {
      this.hudAt = now;
      this.notices.setHud(this.hudText(), this.cfg.debug || this.cfg.showFps);
      this.hooks.onStatus();
    }
  }

  layoutOverlay(rect) {
    const changed = this.overlay.position(this.video, rect, { needsCanvas: this.needsCanvas(), debug: this.cfg.debug });
    if (changed && this.running) this.resetPipeline('canvas geometry', false);
    this.subtitles.sync({ active: this.cfg.subtitleOverlay && this.running && this.needsCanvas() && !!this.video,
      video: this.video, canvas: this.overlay.canvas, canvasVisible: this.overlay.fitSupported && this.overlay.visible });
  }

  // ---------- settings ----------
  applySettings(changed) {
    const has = keys => keys.some(k => changed.has(k));
    if (changed.has('fg') && !this.cfg.fg) this.releaseInterpolation();
    if (has(STREAM_KEYS)) {
      this.streamKey = '';
      this.poolDims = null;
    }
    if (has(PACER_KEYS) && this.pacer && this.backend) {
      this.pacerKey = '';
      this.ensurePacer();
      this.pacer.reset(performance.now());
    }
    if (has(PRESENT_KEYS)) this.configurePresenter();
    if (changed.has('sr') || changed.has('pictureMode')) this.ensureUpscaler();
    if (has(MODE_KEYS) && this.running) {
      if (this.pacer) this.pacer.reset(performance.now());
      if (!this.needsCanvas()) {
        this.freePictureInput();
        this.overlay.passthrough();
      } else this.layoutOverlay(null);
    }
    if (!this.cfg.compare) this.compareRing = [];
    if (changed.has('debug')) this.syncRecording();
    if (changed.has('showFps') || changed.has('debug')) this.notices.hudVisible(this.running && (this.cfg.debug || this.cfg.showFps));
    if (changed.has('showWarnings') && !this.cfg.showWarnings) this.notices.hideWarnings();
    if (changed.has('showWatermark') && !this.cfg.showWatermark) this.notices.watermark.style.display = 'none';
    if (changed.has('subtitleOverlay')) this.layoutOverlay(null);
    if (has(MODE_KEYS)) this.redrawPaused();
  }

  // A paused video delivers no frames, so a new look would not show until playback resumed.
  redrawPaused() {
    const epoch = this.epoch, presenter = this.presenter;
    return presenter?.ready?.then(() => {
      if (!this.running || epoch !== this.epoch || presenter !== this.presenter) return;
      if (!this.video?.paused || !this.video.videoWidth || !this.video.videoHeight || !this.overlay.fitSupported || !this.needsCanvas()) return;
      if (!this.cfg.fg) { this.present(this.pictureFrame(this.video), performance.now()); return; }
      if (!this.stream || !this.source) return;
      const slot = this.stream.nextInput(this.inUse());
      if (slot < 0) return;
      this.source.write(slot, this.video);
      this.present(this.stream.inputTexture(slot), performance.now(), { source: slot });
    }).catch(e => {
      if (epoch === this.epoch && this.running && e?.name !== 'OperationError') this.fail(t('redraw_failed'), e);
    });
  }

  onFullscreenChange() {
    if (this.pacer) this.pacer.modeChange(performance.now());
    this.poolDims = null;
    this.nextUiAt = 0;
  }

  // ---------- device loss ----------
  async onDeviceLost(backend, info) {
    if (this.backend !== backend) return;
    log('device lost', info?.reason || 'unknown', info?.message || '');
    const video = this.video, wasRunning = this.running;
    this.running = false;
    this.cancelLoops();
    clearTimeout(this.tuneTimer);
    this.freePictureInput();
    this.freeStream();
    try { this.presenter?.destroy(); } catch { /* gone */ }
    try { this.upscaler?.destroy(); } catch { /* gone */ }
    try { backend.free(); } catch { /* gone */ }
    this.presenter = null;
    this.upscaler = null;
    this.pacer = null;
    this.pacerKey = '';
    this.recording = false;
    this.backend = null;
    this.kernelsReady = null;
    this.interpolationReady = false;
    this.overlay.hideInstant();
    if (!wasRunning || !video?.isConnected || this.recovering) { this.hooks.onStateChange(false); return; }
    this.recovering = true;
    this.notices.showCritical(t('gpu_recovering'));
    try {
      for (let attempt = 0; attempt < 3; attempt++) {
        try {
          await this.ensureBackend();
          await this.start(video);
          if (this.running) { this.notices.advise(t('gpu_restored'), 3500); return; }
        } catch (e) {
          log('recovery attempt failed', e);
        }
      }
      this.notices.showCritical(t('gpu_recovery_failed'));
      this.hooks.onStateChange(false);
    } finally {
      this.recovering = false;
    }
  }

  // ---------- readouts ----------
  report() {
    try { return this.pacer ? this.pacer.report() : null; } catch { return null; }
  }

  rateLabel() {
    const c = this.cfg;
    if (c.factor === 'auto') return c.fpsLimit !== null ? t('auto_limit', { rate: rateNumber(c.fpsLimit) }) : t('auto');
    if (c.factor === 'hz') return t('match_display');
    if (c.factor === 'target') return t('custom_rate', { rate: rateNumber(c.targetFps) });
    return t('factor_source', { factor: c.factor });
  }

  rateState(r) {
    const label = this.rateLabel();
    const plan = r?.plan;
    if (!plan || plan.state === 'factor') return label;
    const hz = plan.output_hz ? t('rate_approx', { rate: rateNumber(plan.output_hz) }) : '';
    if (plan.state === 'measuring') return `${label} · ${t('measuring')}`;
    if (!plan.interpolation_allowed) return `${label} · ${planWarning(plan) || plan.state.replaceAll('_', ' ')}`;
    if (plan.clamp) return `${label}${hz} · ${t(`clamp_${plan.clamp}`)}`;
    return `${label}${hz}`;
  }

  // The plate the viewer can act on, or null: a rate capped by the display, the 2x floor or the scheduler is the
  // planner doing its job and stays in the status line.
  warningText(r) {
    const c = this.cfg;
    if (!c.fg || !r) return null;
    if (this.source?.path === 'canvas') return t('warning_old_firefox');
    const presents = Math.max(1, this.stats.fpsWin.length);
    const dropping = (this.stats.dropWin.length / 2) / presents > 0.12;
    const fixed = typeof c.factor === 'number';
    if (fixed && dropping) return t('warning_drops');
    const plan = r.plan;
    if (c.factor === 'auto' && plan?.state === 'no_two_x_gpu') return t('warning_auto_gpu');
    if (plan?.warning && !['measuring', 'factor'].includes(plan.state)
        && !['display', 'minimum', 'runtime'].includes(plan.clamp)) return planWarning(plan);
    if (fixed && this.stats.factor > 1 && this.stats.factor < c.factor) {
      return t('warning_fixed_gpu', { actual: this.stats.factor, requested: c.factor });
    }
    return null;
  }

  hudText() {
    const r = this.report();
    const fps = this.stats.fpsWin.length;
    const v = this.video;
    const mid = r ? r.mid_cost_ms : 0;
    if (!this.cfg.debug) return t('hud_fps_rate', { fps, rate: this.rateLabel(), mid: mid.toFixed(0) });
    if (!r) return t('hud_fps', { fps });
    const srcHz = r.source_hz ? r.source_hz.toFixed(0) : '?';
    return [
      t('hud_output', { width: v.videoWidth, height: v.videoHeight, source: srcHz, fps, factor: this.stats.factor, rate: this.rateLabel() }),
      t('hud_render', { pair: r.pair_cost_ms.toFixed(1), mid: mid.toFixed(1), res: this.cfg.res, present: r.present_cost_ms.toFixed(1) }),
      t('hud_delay', { delay: r.delay_ms.toFixed(0), hz: r.display.display_hz, late: r.late_avg_ms.toFixed(1) }),
      t('hud_drops', { drops: this.stats.drops, pressure: r.pressure.toFixed(1), penalty: r.penalty, strain: r.strain_penalty }),
      t('hud_motion', { cuts: r.counters.cuts, holds: r.counters.holds, motion: r.motion_avg.toFixed(0), queue: r.queued }),
      t('hud_pool', { engine: this.backend.class, width: this.spec?.width, height: this.spec?.height, mode: this.spec?.video ? t('hud_video_suffix') : '' }),
    ].join('  ·  ');
  }

  statusLines() {
    const c = this.cfg;
    const r = this.report();
    const video = this.video || this.hooks.candidateVideo();
    const frameSize = this.spec || this.pictureInput;
    const srNeeded = !frameSize || Upscaler.needed(frameSize.width, frameSize.height, this.overlay.width, this.overlay.height);
    let srState = 'off';
    if (this.wantsUpscale()) {
      if (this.backend && !this.backend.f16) srState = 'sr_unavailable';
      else if (!this.upscaler?.ready) srState = 'sr_loading';
      else srState = srNeeded ? 'sr_on' : 'sr_ready';
    }
    const engine = this.backend?.class || (this.blocked
      ? t('engine_unavailable', { reason: t('blocked_wasm') }) : t('engine_not_loaded'));
    const hdr = !this.sys.hdrOk ? 'hdr_no_display' : c.hdr ? (this.sys.hdrOn ? 'on' : 'hdr_sdr') : 'off';
    const lines = [
      t('gpu_line', { gpu: this.sys.gpu || t('unknown_gpu') }) + (this.sys.integrated ? t('integrated_suffix') : ''),
      t('diag_engine', { engine }),
      t('diag_fg', { fg: t(c.fg ? 'on' : 'off'), sr: t(srState) }),
      t('diag_source', { size: video?.videoWidth ? `${video.videoWidth}×${video.videoHeight}` : t('unknown') }),
      t('diag_rate', { rate: this.rateState(r) }),
      t('diag_hdr', { state: t(hdr) }),
      t('diag_sharpness', { value: c.sharpness === 0 ? t('off') : c.sharpness }),
      t('diag_status', { state: t(this.running ? 'running' : 'stopped') }),
    ];
    if (this.running && r) {
      const pool = this.spec ? this.spec.width * this.spec.height * 4 * (this.stream?.report().inputs + this.stream?.report().outputs) / 1048576 : 0;
      lines.push(
        t('diag_output', { fps: this.stats.fpsWin.length, factor: this.stats.factor }),
        t('diag_callbacks', {
          display: r.display.measured ? t('hz_value', { rate: r.display.display_hz }) : t('measuring'),
          source: r.source_hz ? r.source_hz.toFixed(2) : '?',
        }),
        t('diag_render', { pair: r.pair_cost_ms.toFixed(1), mid: r.mid_cost_ms.toFixed(1), res: c.res, present: r.present_cost_ms.toFixed(1) }),
        t('diag_delay', { delay: r.delay_ms.toFixed(0), drops: this.stats.drops, queue: r.queued }),
        t('diag_vram', { size: pool.toFixed(0), mode: this.spec?.video ? t('diag_video_suffix') : '' }));
    }
    return lines;
  }

  status() {
    const r = this.report();
    const frameSize = this.spec || this.pictureInput;
    return {
      running: this.running,
      gpu: this.sys.gpu, integrated: this.sys.integrated, f16: this.sys.f16,
      engine: this.backend ? this.backend.class : null, blocked: this.blocked ? t('blocked_wasm') : null,
      fps: this.stats.fpsWin.length, factor: this.stats.factor,
      midMs: r ? +r.mid_cost_ms.toFixed(1) : 0, pairMs: r ? +r.pair_cost_ms.toFixed(1) : 0,
      drops: this.stats.drops, res: this.cfg.res,
      rateLabel: this.rateLabel(), rateState: this.rateState(r),
      browserCallbackHz: r?.display?.measured ? r.display.raw_hz : null,
      presentationMode: this.needsCanvas() ? 'canvas' : 'native',
      pool: frameSize ? { width: frameSize.width, height: frameSize.height, video: !!frameSize.video } : null,
      pacer: r,
    };
  }
}
