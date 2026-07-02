import { toInput, fromOutput } from './rife_prepost.js';

const EW = 1280, EH = 736; // engine (padded) input size

export async function createSession(ep) {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
  return await ort.InferenceSession.create('/assets/rife_lite_inlined.onnx',
    { executionProviders: [ep], graphOptimizationLevel: 'all' });
}

// Two same-size ImageData -> interpolated middle-frame ImageData.
export async function interpolate(session, a, b) {
  if (a.width !== b.width || a.height !== b.height) {
    throw new Error(`frame size mismatch: ${a.width}x${a.height} vs ${b.width}x${b.height}`);
  }
  const w = a.width, h = a.height;
  const feeds = {};
  feeds[session.inputNames[0]] = new ort.Tensor('float32', toInput(a.data, w, h, EW, EH), [1, 3, EH, EW]);
  feeds[session.inputNames[1]] = new ort.Tensor('float32', toInput(b.data, w, h, EW, EH), [1, 3, EH, EW]);
  const out = await session.run(feeds);
  const chw = out[session.outputNames[0]].data;
  return new ImageData(fromOutput(chw, w, h, EW, EH), w, h);
}
