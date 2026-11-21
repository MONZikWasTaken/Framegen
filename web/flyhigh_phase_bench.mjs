import { EngineBackend, loadEngine } from '../extension/src/engine.js';

const root = new URL('../extension/engine/', import.meta.url);
const urls = Object.fromEntries([
  ['graph', 'graph.json'],
  ['manifest', 'flyhighfg.json'], ['weights', 'flyhighfg.bin'],
].map(([key, file]) => [key, new URL(file, root).href]));

export function phaseCases(width, height) {
  const common = EngineBackend.prototype.sessionOptions({
    width, height, res: 480, video: false, repair: 1, render: 'bicubic', guard: true,
  });
  if (!common.fastRender || common.internal.every((n, i) => Math.ceil(n / 32) === Math.ceil([width, height][i] / 32))) {
    throw new Error('The padded network resolution must be smaller than the frame');
  }
  return [
    ['fast-linear', common],
    ['direct-exact', { ...common, linearT: false }],
    ['full-buffer-exact', { ...common, fastRender: false, linearT: false }],
  ];
}

export function frameIndices(count) {
  if (!Number.isSafeInteger(count) || count < 5) throw new Error('Need at least five captured frames');
  return Array.from({ length: Math.floor((count - 1) / 4) }, (_, i) => Array.from({ length: 5 }, (_, j) => 4 * i + j));
}

function captionFrame(frame, ctx, w, h) {
  ctx.putImageData(new ImageData(new Uint8ClampedArray(frame.buffer, frame.byteOffset, frame.length), w, h), 0, 0);
  ctx.font = 'bold 48px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = 'white';
  ctx.strokeStyle = 'black';
  ctx.lineWidth = 6;
  const label = 'The runners round the bend.';
  ctx.strokeText(label, w / 2, h - 80);
  ctx.fillText(label, w / 2, h - 80);
  return ctx.getImageData(0, 0, w, h).data;
}

function pixelErrors(actual, target, before, after) {
  let sum = 0, motionSum = 0, motionCount = 0;
  for (let p = 0; p < target.length; p += 4) {
    const r = actual[p] - target[p], g = actual[p + 1] - target[p + 1], b = actual[p + 2] - target[p + 2];
    const err = r * r + g * g + b * b;
    sum += err;
    if (Math.abs(before[p] - after[p]) + Math.abs(before[p + 1] - after[p + 1]) + Math.abs(before[p + 2] - after[p + 2]) > 48) {
      motionSum += err;
      motionCount += 3;
    }
  }
  const psnr = (e, n) => Math.min(99, 10 * Math.log10(65025 * n / e));
  return { psnr: psnr(sum, target.length * 0.75), motionPsnr: motionCount ? psnr(motionSum, motionCount) : null };
}

function cropFrame(data, width, [x, y, w, h]) {
  const pixels = new Uint8ClampedArray(w * h * 4);
  for (let r = 0; r < h; r++) {
    const start = ((y + r) * width + x) * 4;
    pixels.set(data.subarray(start, start + w * 4), r * w * 4);
  }
  return new ImageData(pixels, w, h);
}

export async function comparePhases(file, {
  width = 1920, height = 1080, fps = 120, path = 'all', caption = false,
  crop = [0, 0, width, height], onPreview = () => {},
} = {}) {
  if (![width, height, ...crop].every(Number.isSafeInteger) || width <= 0 || height <= 0
      || crop[0] < 0 || crop[1] < 0 || crop[2] <= 0 || crop[3] <= 0
      || !Number.isFinite(fps) || fps <= 0 || crop[0] + crop[2] > width || crop[1] + crop[3] > height) {
    throw new Error('Invalid clip dimensions, crop or frame rate');
  }
  const bytes = width * height * 4;
  if (!Number.isSafeInteger(bytes) || file.size % bytes !== 0) throw new Error('Expected packed RGBA8 frames without a header');
  const groups = frameIndices(file.size / bytes);
  const cases = phaseCases(width, height).filter(([name]) => path === 'all' || path === name);
  if (!cases.length) throw new Error('Unknown benchmark render path');
  const backend = await loadEngine(urls);
  const { device } = backend;
  const streams = [], rows = [], previews = new Map();
  let validation;
  device.pushErrorScope('validation');
  try {
    await backend.loadModel();
    for (const [name, options] of cases) {
      const stream = backend.engine.stream(backend.model, options, { mode: 'realtime', inputs: 4, outputs: 1 });
      streams.push([name, stream]);
      await backend.warmUp(stream);
    }
    const ctx = caption ? new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true }) : null;
    const read = async index => {
      const data = new Uint8Array(await file.slice(index * bytes, (index + 1) * bytes).arrayBuffer());
      if (data.length !== bytes) throw new Error(`Unable to read captured frame ${index}`);
      return ctx ? captionFrame(data, ctx, width, height) : data;
    };
    const push = (frame, index) => {
      for (const [, stream] of streams) {
        const slot = stream.nextInput([]);
        if (slot < 0) throw new Error('No free input slot in phase benchmark');
        device.queue.writeTexture({ texture: stream.inputTexture(slot) }, frame, { bytesPerRow: width * 4 }, [width, height]);
        stream.push(slot, index * 1000 / fps);
      }
    };
    let before = await read(0);
    push(before, 0);
    for (const indices of groups) {
      const after = await read(indices[4]);
      push(after, indices[4]);
      for (let phase = 1; phase < 4; phase++) {
        const target = await read(indices[phase]);
        const t = phase / 4, row = { frame: indices[phase], t, scores: {} };
        const output = new Map();
        for (const [name, stream] of streams) {
          if (!stream.generate(t, 0)) throw new Error(`${name} did not generate phase ${t}`);
          const image = await backend.nudge.wrap(stream.readRgba(0));
          if (image.length !== bytes) throw new Error('Unexpected output size');
          row.scores[name] = pixelErrors(image, target, before, after);
          output.set(name, image);
        }
        rows.push(row);
        if (phase === 2) continue;
        const delta = Math.abs((row.scores['direct-exact']?.motionPsnr ?? 0) - (row.scores['fast-linear']?.motionPsnr ?? 0));
        if (!previews.has(t) || delta > previews.get(t).delta) previews.set(t, {
          delta, frame: row.frame, images: [['captured reference', target], ...output].map(([name, image]) =>
            [name, cropFrame(image, width, crop)]),
        });
      }
      before = after;
    }
  } finally {
    try { validation = await device.popErrorScope(); }
    finally { for (const [, stream] of streams) stream.free(); backend.free(); }
  }
  if (validation) throw new Error(validation.message);
  for (const [t, { frame, images }] of previews) onPreview(t, frame, images);
  const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
  const summary = [0.25, 0.5, 0.75].map(t => ({
    t, modes: Object.fromEntries(cases.map(([name]) => {
      const samples = rows.filter(row => row.t === t).map(row => row.scores[name]);
      return [name, { psnr: average(samples.map(s => s.psnr)), motionPsnr: average(samples.flatMap(s => s.motionPsnr ?? [])) }];
    })),
  }));
  return { file: file.name ?? null, width, height, fps, caption, intervals: groups.length,
    adapter: backend.name, precision: backend.class, cases: Object.fromEntries(cases), summary, rows };
}
