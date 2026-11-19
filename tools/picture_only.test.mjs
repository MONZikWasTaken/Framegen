import assert from 'node:assert/strict';
import test from 'node:test';

globalThis.chrome = { runtime: { getURL: path => path } };
globalThis.GPUTextureUsage = { TEXTURE_BINDING: 1, COPY_DST: 2, RENDER_ATTACHMENT: 4 };
const { Player } = await import('../extension/src/player.js');

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
  const fgTexture = {};
  const backend = {
    device: {
      createTexture(desc) {
        const texture = { desc, destroyed: false, destroy() { this.destroyed = true; } };
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
        push() { interp.pushed++; return {}; }
      },
    },
    async loadModel() { interp.models++; },
    precompile: async () => {},
    sessionOptions: () => ({}), tuneKey: () => 'fake',
    openStream() {
      interp.streams++;
      return { nextInput: () => 0, inputTexture: () => fgTexture, free() { interp.released++; } };
    },
    createPacer() {
      interp.pacers++;
      return { reset: noop, modeChange: noop, setOptions: noop, sourceFrame: () => [], delayMs: () => 0,
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
  return { player, video, overlay, cfg, backend, interp, textures, drawn, writes, frame, pacersFreed: () => pacersFreed };
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
