import { toInput, fromOutput } from './rife_prepost.js';

// Fixed-shape models (engine size = padded to /32). Lower res = proportionally faster.
// Quality tiers (training-free ablations, see docs/phase5_ablation.md):
//   full    - the whole net
//   fast    - refinement (contextnet+unet) cut: ~1.5x on webgpu, -1.3..+0.2 dB
//   fastest - block2 + refinement cut: ~4x on webgpu, -2.6..-0.8 dB
// Measured webgpu p50 (4060 Ti): 720p full=1957 / 480p full=1064 / 480p fastest=237ms.
export const MODELS = {
  '720': { ew: 1280, eh: 736, maxw: 1280, maxh: 720, files: {
    full: '/assets/rife_lite_inlined.onnx',
    fast: '/assets/rife_lite_720p_noref.onnx',
    fastest: '/assets/rife_lite_720p_2blk_noref.onnx' } },
  '480': { ew: 864, eh: 480, maxw: 854, maxh: 480, files: {
    full: '/assets/rife_lite_480x854.onnx',
    fast: '/assets/rife_lite_480p_noref.onnx',
    fastest: '/assets/rife_lite_480p_2blk_noref.onnx' } },
  '360': { ew: 640, eh: 384, maxw: 640, maxh: 360, files: {
    full: '/assets/rife_lite_360x640.onnx',
    fast: '/assets/rife_lite_360p_noref.onnx',
    fastest: '/assets/rife_lite_360p_2blk_noref.onnx' } },
};

// 'webnn' -> DirectML on Windows (fuses the graph; ~3.5x over the WebGPU EP, but needs the
// Chrome flag #web-machine-learning-neural-network and the ort.all bundle). Others pass through.
function epProvider(ep) {
  return ep === 'webnn'
    ? { name: 'webnn', deviceType: 'gpu', powerPreference: 'high-performance' }
    : ep;
}

// Returns a handle { sess, ew, eh, maxw, maxh } for interpolate().
export async function createSession(ep, res = '720', quality = 'full') {
  const m = MODELS[res];
  if (!m) throw new Error(`unknown resolution: ${res}`);
  const url = m.files[quality];
  if (!url) throw new Error(`unknown quality: ${quality}`);
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  const sess = await ort.InferenceSession.create(url,
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
