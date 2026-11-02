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

  free() {
    try { this.model.free(); } catch { /* already freed */ }
    try { this.engine.free(); } catch { /* already freed */ }
  }
}
