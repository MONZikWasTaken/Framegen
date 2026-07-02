import { toInput, fromOutput } from './rife_prepost.js';

// Fixed-shape models (engine size = padded to /32). Lower res = proportionally faster
// (the model is bandwidth/dispatch-bound): measured webgpu 720p=1957ms / 480p=1064ms / 360p=655ms.
export const MODELS = {
  '720': { url: '/assets/rife_lite_inlined.onnx', ew: 1280, eh: 736, maxw: 1280, maxh: 720 },
  '480': { url: '/assets/rife_lite_480x854.onnx', ew: 864,  eh: 480, maxw: 854,  maxh: 480 },
  '360': { url: '/assets/rife_lite_360x640.onnx', ew: 640,  eh: 384, maxw: 640,  maxh: 360 },
};

// 'webnn' -> DirectML on Windows (fuses the graph; ~3.5x over the WebGPU EP, but needs the
// Chrome flag #web-machine-learning-neural-network and the ort.all bundle). Others pass through.
function epProvider(ep) {
  return ep === 'webnn'
    ? { name: 'webnn', deviceType: 'gpu', powerPreference: 'high-performance' }
    : ep;
}

// Returns a handle { sess, ew, eh, maxw, maxh } for interpolate().
export async function createSession(ep, res = '720') {
  const m = MODELS[res];
  if (!m) throw new Error(`unknown resolution: ${res}`);
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  const sess = await ort.InferenceSession.create(m.url,
    { executionProviders: [epProvider(ep)], graphOptimizationLevel: 'all' });
  return { sess, ew: m.ew, eh: m.eh, maxw: m.maxw, maxh: m.maxh };
}

// Two same-size ImageData -> interpolated middle-frame ImageData.
export async function interpolate(handle, a, b) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`frame size mismatch: ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  }
  const { sess, ew, eh } = handle;
  const w = a.width, h = a.height;
  if (w > ew || h > eh) throw new Error(`frame ${w}x${h} exceeds engine ${ew}x${eh}`);
  const feeds = {};
  feeds[sess.inputNames[0]] = new ort.Tensor('float32', toInput(a.data, w, h, ew, eh), [1, 3, eh, ew]);
  feeds[sess.inputNames[1]] = new ort.Tensor('float32', toInput(b.data, w, h, ew, eh), [1, 3, eh, ew]);
  const out = await sess.run(feeds);
  const chw = out[sess.outputNames[0]].data;
  return new ImageData(fromOutput(chw, w, h, ew, eh), w, h);
}
