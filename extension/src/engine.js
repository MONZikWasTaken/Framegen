// The FlyHigh engine as Framegen's backend: the wasm module and its browser adapters, one device and one model per
// page, streams and pacers built from a host-neutral stream spec. The autotune tables live in chrome.storage.local
// beside the previous engine's conv tunes (key fcTune).

// Internal network sizes per Quality setting (the model's rungs). The engine clamps them to the frame.
export const SIZES = { 288: [512, 288], 360: [640, 352], 480: [848, 480], 720: [1280, 720], 1080: [1920, 1088] };
export const INPUT_SLOTS = 12;
export const OUTPUT_SLOTS = 8;
// Scene cuts at a mean cell difference above 90 / 765: the threshold the extension has used since 1.0 (the engine's
// default 0.08 flags fast anime action as cuts).
const SCENE_CUT = 90 / 765;
const HELD_FRAMES = { mean: 2.5 / 765, max: 45 / 765 };

function fetchOk(url, as) {
  return fetch(url).then(r => {
    if (!r.ok) throw new Error(`${url} HTTP ${r.status}`);
    return r[as]();
  });
}

export const tuneStore = {
  cache: {},
  loaded: false,
  async load() {
    try {
      const st = await chrome.storage.local.get('fcTune');
      this.cache = st.fcTune || {};
    } catch { this.cache = {}; }
    this.loaded = true;
  },
  get(key) { return this.cache[key] || null; },
  async put(key, value) {
    if (!this.loaded) await this.load();
    this.cache[key] = value;
    try { await chrome.storage.local.set({ fcTune: this.cache }); } catch { /* storage unavailable */ }
  },
};

// The kernels the extension's configurations compile (realtime at 720p and 1080p, the 720 rung, cadence smoothing with
// its composite), recorded with engine 0.1.3; what a device compiles beyond them is remembered in kernelStore.
const DEFAULT_KERNELS = ['avgpool', 'binary', 'blendmix', 'chcopy', 'classify', 'convex', 'film', 'gm_box_h', 'gm_box_v',
  'gm_composite', 'gm_cost', 'gm_fit_reduce', 'gm_fit_solve', 'gm_mask', 'gm_select', 'gm_stats', 'gm_trust', 'grad',
  'interp', 'keep_static', 'keep_static_age', 'keep_static_age_tex', 'luma', 'render_static', 'render_tex', 'sad',
  'skreduce', 'tconv', 'tconv32', 'tex_down', 'tex_in', 'tex_out', 'unary', 'warp'];
// Kernels compiled per animation frame: Chrome compiles in the process that composites the page, about 20 ms a kernel
// on a site whose shader cache is cold (the cache is per site), so a stream's worth at once froze the video for a second.
const KERNELS_PER_FRAME = 2;

export const kernelStore = {
  names: null,
  async load() {
    if (this.names) return this.names;
    let stored = [];
    try { stored = (await chrome.storage.local.get('fhKernels')).fhKernels || []; } catch { /* storage unavailable */ }
    this.names = [...new Set([...DEFAULT_KERNELS, ...stored])];
    return this.names;
  },
  async remember(compiled) {
    const names = await this.load();
    const fresh = compiled.filter(k => !names.includes(k));
    if (!fresh.length) return;
    names.push(...fresh);
    try { await chrome.storage.local.set({ fhKernels: names.filter(k => !DEFAULT_KERNELS.includes(k)) }); } catch { /* storage unavailable */ }
  },
};

const nextFrame = () => new Promise(resolve => {
  // a hidden tab gets no animation frames; the timer keeps the compilation going there
  const timer = setTimeout(resolve, 50);
  requestAnimationFrame(() => { clearTimeout(timer); resolve(); });
});

export async function loadEngine(urls) {
  const [mod, adapters] = await Promise.all([import(urls.module), import(urls.adapters)]);
  try {
    await mod.default();
  } catch (e) {
    // a page CSP without 'wasm-unsafe-eval' governs the content script's world too: the module cannot compile here
    const blocked = new Error('WebAssembly unavailable on this page: ' + (e?.message || e));
    blocked.wasmBlocked = true;
    throw blocked;
  }
  const [weights, manifest, graph] = await Promise.all([
    fetchOk(urls.weights, 'arrayBuffer'), fetchOk(urls.manifest, 'text'), fetchOk(urls.graph, 'text')]);
  const engine = await mod.FhEngine.create({});
  const model = engine.loadModel(graph, manifest, new Uint8Array(weights));
  return new EngineBackend(mod, adapters, engine, model, await adapterName());
}

// The engine reports wgpu's adapter name, which in a browser is the WebGPU implementation; the adapter's own info
// names the GPU (for the status readouts and the autotune keys).
async function adapterName() {
  try {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    const info = adapter?.info || {};
    return info.description || [info.vendor, info.architecture].filter(Boolean).join(' ');
  } catch { return ''; }
}

export class EngineBackend {
  constructor(mod, adapters, engine, model, gpuName) {
    this.kind = 'engine';
    this.freed = false;
    this.mod = mod;
    this.adapters = adapters;
    this.engine = engine;
    this.model = model;
    this.device = engine.device();
    this.info = engine.info();
    this.modelInfo = model.info();
    this.name = gpuName || this.info.name || 'unknown GPU';
    this.class = this.info.class;
    this.f16 = this.info.class !== 'portable-f32';
    this.timestamps = !!this.info.timestamps;
    this.nudge = new adapters.ReadbackNudge(this.device);
  }

  sessionOptions(spec) {
    const [rw, rh] = SIZES[spec.res] || SIZES[480];
    const iw = Math.min(rw, spec.width), ih = Math.min(rh, spec.height);
    const fast = !spec.video && (iw < spec.width || ih < spec.height);
    return {
      width: spec.width,
      height: spec.height,
      internal: [iw, ih],
      k: 1,
      repair: spec.repair,
      render: spec.render,
      staticProtection: spec.guard,
      // mids rendered straight from the source textures with the t-dependent network once per pair: the same picture
      // at a fraction of the GPU time; needs an internal size below the frame, and cadence re-timing (video mode)
      // renders the exact path
      fastRender: fast,
      linearT: fast,
      tune: spec.tune || null,
    };
  }

  streamOptions(spec) {
    return {
      mode: spec.video ? 'video' : 'realtime',
      cadence: spec.video,
      sceneCut: { threshold: SCENE_CUT },
      heldFrames: spec.anime ? HELD_FRAMES : null,
      inputs: INPUT_SLOTS,
      outputs: OUTPUT_SLOTS,
    };
  }

  tuneKey(spec) {
    const s = this.sessionOptions(spec);
    return `fh|${this.name}|${s.internal[0]}x${s.internal[1]}|${s.width}x${s.height}`;
  }

  openStream(spec) {
    return this.engine.stream(this.model, this.sessionOptions(spec), this.streamOptions(spec));
  }

  createPacer(options) {
    return new this.mod.FhPacer({ ...this.adapters.pacerDefaults(), ...options });
  }

  async autotune(spec) {
    const table = await this.engine.autotune(this.model, { ...this.sessionOptions(spec), tune: null });
    return table || null;
  }

  warmUp(stream) {
    return this.adapters.warmUp(this.device, stream);
  }

  report() {
    return this.engine.report();
  }

  // Compiles the remembered kernels a few per frame, so the stream that opens later finds them compiled.
  async precompile() {
    const names = await kernelStore.load();
    for (let i = 0; i < names.length; i += KERNELS_PER_FRAME) {
      if (this.freed) return;
      if (this.engine.precompile(names.slice(i, i + KERNELS_PER_FRAME))) {
        await this.device.queue.onSubmittedWorkDone();
        await nextFrame();
      }
    }
  }

  rememberKernels() {
    try { return kernelStore.remember(this.engine.compiledKernels()); } catch { return Promise.resolve(); }
  }

  free() {
    this.freed = true;
    try { this.model.free(); } catch { /* already freed */ }
    try { this.engine.free(); } catch { /* already freed */ }
  }
}
