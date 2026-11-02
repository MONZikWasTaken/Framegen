// The product benchmark bridge: a page served from loopback with a per-run token drives the real extension path and
// reads the Pacer's report and its trace v1 recording (what `flyhigh-dev replay` reproduces). Unreachable from ordinary
// pages: no benchmark state exists without the gate.
import { errors } from './log.js';
import { Profiles } from './profiles.js';

export function benchToken() {
  try {
    const params = new URL(location.href).searchParams;
    const token = params.get('framegenBenchToken') || '';
    const loopback = location.hostname === '127.0.0.1' || location.hostname === 'localhost';
    return window === window.top && location.protocol === 'http:' && loopback
      && params.get('framegenProductBench') === '1' && /^[a-f0-9]{32,128}$/.test(token)
      ? token : null;
  } catch { return null; }
}

export function installBenchBridge({ token, settings, player, version, biggestVideo }) {
  const reply = (id, ok, value, error) => window.postMessage({
    source: 'framegen-product-bench-extension', token, id, ok, value, error,
  }, location.origin);
  window.addEventListener('message', event => {
    if (event.source !== window) return;
    const message = event.data;
    if (!message || message.source !== 'framegen-product-bench-fixture'
        || message.token !== token || !Number.isInteger(message.id)) return;
    (async () => {
      switch (message.command) {
        case 'ping':
          return { bridgeVersion: 3, extensionVersion: version };
        case 'configure': {
          if (player.isRunning()) throw new Error('configure requires a stopped product path');
          const p = message.payload || {};
          settings.set({ ...Profiles.DEFAULT_SETTINGS, ...p, debug: false, hoverReveal: false, showFps: false,
            showWatermark: false, compare: false, hdr: false }, { persist: false });
          return { ...settings.cfg };
        }
        case 'prepare': {
          if (player.isRunning()) throw new Error('prepare requires a stopped product path');
          const backend = await player.ensureBackend();
          return { engine: backend.class, gpu: backend.name, timestamps: backend.timestamps };
        }
        case 'start': {
          const video = biggestVideo();
          if (!video) throw new Error('no ready benchmark video');
          await player.start(video);
          return { running: player.isRunning(), width: video.videoWidth, height: video.videoHeight };
        }
        case 'snapshot':
          return {
            schemaVersion: 3,
            extensionVersion: version,
            status: player.status(),
            stream: player.stream ? player.stream.report() : null,
            engine: player.backend ? player.backend.report() : null,
            errors: errors.map(e => ({ ...e })),
          };
        case 'trace':
          return player.pacer ? player.pacer.trace() : null;
        case 'stop':
          player.stop();
          return { running: player.isRunning() };
        default:
          throw new Error('unknown product benchmark command');
      }
    })().then(value => reply(message.id, true, value, null))
      .catch(error => reply(message.id, false, null, error.stack || error.message || String(error)));
  });
  window.postMessage({ source: 'framegen-product-bench-extension', token, id: 0, ok: true,
    value: { bridgeVersion: 3, extensionVersion: version }, error: null }, location.origin);
}
