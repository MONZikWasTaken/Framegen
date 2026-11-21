import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const english = JSON.parse(readFileSync(new URL('../extension/_locales/en/messages.json', import.meta.url), 'utf8'));
globalThis.chrome = {
  runtime: { getURL: path => path },
  i18n: { getMessage: key => english[key]?.message || '' },
};
globalThis.GPUTextureUsage = { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 };
const { Player } = await import('../extension/src/player.js');

test('compare uses the source texture in Picture Mode without an interpolation stream', () => {
  const texture = { width: 1920, height: 1080 };
  const overlay = { ...texture, fitSupported: true };
  const hooks = { splitX: () => 0.5 };
  const settings = { cfg: { fg: false, compare: true, pictureMode: 'noir' } };
  const player = new Player({ settings, overlay, hooks });
  player.presenter = { present(output, { compare }) {
    assert.equal(output, texture);
    assert.equal(compare, texture);
    return false;
  } };
  assert.equal(player.stream, null);
  player.present(texture, 0);
});

function fixture() {
  const noop = () => {};
  const textures = [], drawn = [], writes = [];
  const cfg = { fg: false, pictureMode: 'vivid', factor: 2, sharpness: 0 };
  const video = { videoWidth: 1920, videoHeight: 1080, paused: false, addEventListener: noop };
  const overlay = {
    width: 1920, height: 1080, fitSupported: true, sourceWidth: 1920, sourceHeight: 1080,
    ensure: noop, show: noop, resetGeometry: noop, fadeOut: noop, position: noop, passthrough: noop,
  };
  const interp = { models: 0, streams: 0, pacers: 0, released: 0, pushed: 0 };
  let pacersFreed = 0;
  let modeChanges = 0;
  const gpuSamples = [];
  const backend = {
    device: {
      createTexture(desc) {
        const [width, height] = desc.size;
        const texture = { desc, width, height, destroyed: false, destroy() { this.destroyed = true; } };
        textures.push(texture);
        return texture;
      },
    },
    adapters: {
      VideoSource: class {
        constructor(device, inputs) { this.inputs = inputs; }
        write(slot, video) { writes.push([this.inputs.inputTexture(slot), video]); }
      },
      CostSampler: class {
        constructor(pacer) { this.pacer = pacer; }
        push() { interp.pushed++; return {}; }
      },
    },
    async loadModel() { interp.models++; },
    precompile: async () => {},
    sessionOptions: () => ({}), tuneKey: () => 'fake',
    openStream({ width, height }) {
      interp.streams++;
      const fgTexture = { width, height };
      return { nextInput: () => 0, inputTexture: () => fgTexture, classify: async () => ({ kind: 'motion' }),
        free() { interp.released++; } };
    },
    createPacer() {
      interp.pacers++;
      return { reset: noop, modeChange() { modeChanges++; }, setOptions: noop, sourceFrame: () => [], delayMs: () => 0,
        costSample(...sample) { gpuSamples.push(sample); return true; },
        free() { pacersFreed++; } };
    },
    warmUp: async () => {}, rememberKernels: noop,
  };
  const player = new Player({
    settings: { cfg, ready: Promise.resolve() }, overlay, subtitles: { sync: noop },
    notices: { setHud: noop, hideAll: noop, showCritical(msg) { assert.fail(msg); } }, hooks: { onStateChange: noop },
  });
  player.ensureBackend = async () => (player.backend = backend);
  player.ensurePresenter = () => { player.presenter = { ready: Promise.resolve(), invalidate: noop }; };
  player.configurePresenter = noop;
  player.present = texture => drawn.push(texture);
  player.armVideoFrames = player.armTicks = player.scheduleTune = noop;
  const frame = (mediaTime = 1) => player.onVideoFrame({ mediaTime }, player.epoch);
  return { player, video, overlay, cfg, backend, interp, textures, drawn, writes, frame, pacersFreed: () => pacersFreed,
    gpuSamples, modeChanges: () => modeChanges };
}

test('Vivid without FG reuses one source texture and does not load or open the interpolation pipeline', async () => {
  const { player, video, interp, textures, drawn, writes, frame } = fixture();
  await player.start(video);
  await frame(1);
  await frame(2);

  assert.deepEqual(interp, { models: 0, streams: 0, pacers: 0, released: 0, pushed: 0 });
  assert.equal(textures.length, 1);
  assert.equal(textures[0].desc.size[0] * textures[0].desc.size[1] * 4, 8_294_400);
  assert.deepEqual(player.status().pool, { width: 1920, height: 1080, video: false });
  assert.deepEqual(drawn, [textures[0], textures[0]]);
  assert.ok(writes.every(([texture, input]) => texture === textures[0] && input === video));
});

for (const fg of [false, true]) {
  test(`Detail Boost feeds TinySR the original 720p frames at 1080p (FG ${fg ? 'on' : 'off'})`, async () => {
    const { player, video, overlay, cfg, backend, interp, drawn, frame } = fixture();
    Object.assign(video, { videoWidth: 1280, videoHeight: 720 });
    Object.assign(cfg, { fg, pictureMode: 'vivid', sr: false });
    backend.f16 = true;
    backend.nudge = { wrap: p => p };
    player.hooks.splitX = () => 0.5;
    overlay.reveal = () => {};
    const srInputs = [], upscaled = { width: 2560, height: 1440 };
    player.upscaler = {
      ready: true,
      load: async () => {},
      process(texture) { srInputs.push([texture.width, texture.height]); return upscaled; },
    };
    const mockPresenter = player.ensurePresenter;
    player.ensurePresenter = () => {
      mockPresenter();
      player.presenter.present = texture => { drawn.push(texture); return true; };
    };
    player.present = Player.prototype.present.bind(player);
    await player.start(video);
    await frame(1);
    assert.equal(drawn.at(-1).width, 1920);

    cfg.pictureMode = 'detail';
    player.applySettings(new Set(['pictureMode']));
    await frame(2);
    assert.deepEqual(srInputs, [[1280, 720]]);
    assert.equal(drawn.at(-1), upscaled);
    assert.match(player.statusLines().find(line => line.startsWith('FG:')), /upscale: on x2/);
    assert.equal(interp.streams, fg ? 2 : 0);

    cfg.pictureMode = 'vivid';
    player.applySettings(new Set(['pictureMode']));
    await frame(3);
    assert.equal(srInputs.length, 1);
    assert.equal(drawn.at(-1).width, 1920);
    player.stop();
  });
}

test('Picture Mode stays live while FG prepares, then can switch back and release interpolation', async () => {
  const { player, video, cfg, backend, interp, textures, drawn, frame } = fixture();
  const { promise, resolve } = Promise.withResolvers();
  backend.loadModel = async () => { interp.models++; await promise; };
  await player.start(video);
  await frame();
  cfg.fg = true;
  player.applySettings(new Set(['fg']));
  await frame(2);
  assert.equal(drawn.at(-1), textures[0]);
  assert.equal(interp.streams, 0);
  resolve();
  await player.kernelsReady;
  await frame(3);
  assert.deepEqual(interp, { models: 1, streams: 1, pacers: 1, released: 0, pushed: 1 });
  assert.equal(textures[0].destroyed, true);

  const classification = Promise.withResolvers();
  player.backend.nudge = { wrap: p => p };
  player.stream.classify = () => classification.promise;
  const interrupted = frame(4);
  cfg.fg = false;
  player.applySettings(new Set(['fg']));
  classification.reject(new Error('retired stream'));
  await interrupted;
  await frame(5);
  assert.equal(interp.released, 1);
  assert.equal(player.pacer, null);
  assert.equal(drawn.at(-1), textures[1]);
});

test('a paused video redraws its Picture Mode, and a resize and stop release the scratch texture', async () => {
  const { player, video, overlay, interp, textures, drawn, frame } = fixture();
  video.paused = true;
  await player.start(video);
  await player.redrawPaused();
  assert.equal(drawn.at(-1), textures[0]);
  assert.equal(interp.streams, 0);

  overlay.width = 1280;
  overlay.height = 720;
  video.paused = false;
  await frame();
  assert.equal(textures[0].destroyed, true);
  assert.equal(drawn.at(-1), textures[1]);
  player.stop();
  assert.equal(textures[1].destroyed, true);
});

test('stopping FG frees the stream and pacer once, and re-enabling does not reload the model', async () => {
  const { player, video, cfg, interp, frame, pacersFreed } = fixture();
  cfg.fg = true;
  await player.start(video);
  await frame();

  player.stop();
  player.stop();
  assert.equal(interp.released, 1);
  assert.equal(pacersFreed(), 1);
  assert.equal(player.status().pool, null);
  assert.equal(player.source, null);

  await player.start(video);
  await frame(2);
  assert.deepEqual(interp, { models: 1, streams: 2, pacers: 2, released: 1, pushed: 2 });
  player.stop();
});

test('rebuilding a stream restarts GPU estimates and ignores samples from retired streams', async () => {
  const { player, video, cfg, frame, gpuSamples, modeChanges } = fixture();
  cfg.fg = true;
  await player.start(video);
  const old = player.costs.pacer;
  assert.equal(old.costSample(1, 'mid', 1), true);
  const initial = modeChanges();

  cfg.res = 720;
  player.applySettings(new Set(['res']));
  await frame(2);
  assert.equal(modeChanges(), initial + 1);
  assert.equal(old.costSample(2, 'mid', 10), false);
  const current = player.costs.pacer;
  assert.equal(current.costSample(3, 'mid', 3), true);
  player.stop();
  assert.equal(current.costSample(4, 'pair', 4), false);
  assert.deepEqual(gpuSamples, [[1, 'mid', 1], [3, 'mid', 3]]);
});

test('a browser callback cap is not reported as the physical display refresh rate', () => {
  const { player, cfg } = fixture();
  Object.assign(cfg, { factor: 'target', targetFps: 120 });
  const display = { measured: true, raw_hz: 59.99, display_hz: 60 };
  const report = { display, plan: { state: 'active', interpolation_allowed: true, clamp: 'display', output_hz: 60 },
    mid_cost_ms: 1, pair_cost_ms: 1 };
  player.report = () => report;
  assert.match(player.rateState(report), /browser callback limit/);
  assert.equal(player.status().browserCallbackHz, 59.99);
  display.measured = false;
  assert.equal(player.status().browserCallbackHz, null);
});

for (const waitingOn of ['classification', 'cadence stats']) {
  test(`FG can restart with old ${waitingOn} pending without unblocking a busy new stream`, async () => {
    const { player, video, cfg, backend, interp, frame } = fixture();
    cfg.fg = true;
    backend.nudge = { wrap: p => p };
    await player.start(video);
    await frame(1);

    const oldReadback = Promise.withResolvers();
    if (waitingOn === 'classification') {
      player.stream.classify = () => oldReadback.promise;
    } else {
      player.stream.classify = async () => ({ kind: 'motion' });
      player.stream.stats = () => oldReadback.promise;
      player.costs.push = () => ({ statsPending: true });
    }
    const oldFrame = frame(2);
    await Promise.resolve();
    player.stop();
    await player.start(video);
    await frame(3);
    assert.equal(interp.pushed, 2);

    const next = Promise.withResolvers();
    let classifyCalls = 0;
    player.stream.classify = () => { classifyCalls++; return next.promise; };
    const newFrame = frame(4);
    oldReadback.resolve({ kind: 'motion' });
    await oldFrame;
    const skippedFrame = frame(5);
    next.resolve({ kind: 'motion' });
    await Promise.all([newFrame, skippedFrame]);
    assert.equal(classifyCalls, 1);
    assert.equal(interp.pushed, 3);
    player.stop();
  });
}
