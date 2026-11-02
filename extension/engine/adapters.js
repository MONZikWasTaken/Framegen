// Host adapters for the browser (ARCHITECTURE.md): what every WebGPU host needs around the engine's
// classes and nothing the engine itself should know about: how a <video> gets into a slot in each browser, how a slot
// gets onto a canvas, requestAnimationFrame as the Pacer's ticks, the readbacks Firefox sits on, the GPU cost samples
// the Pacer learns from. A plain ES module, no build step; it is shipped next to flyhigh_web.js.

export const FIREFOX = typeof navigator !== 'undefined' && /firefox/i.test(navigator.userAgent);

/** Pacer option defaults for this browser: Firefox's compositor stalls 40 to 90 ms when two canvas presents come
 * closer than 7 ms. Spread them into your own options. */
export function pacerDefaults() {
  return FIREFOX ? { min_present_interval_ms: 7 } : {};
}

/** Firefox resolves mapAsync only on its GPU polls (70 to 100 ms) unless work is submitted; empty submits while a
 * readback is pending bring it to 5 ms. Wrap every promise of a readback: classify, stats, cost, readRgba. */
export class ReadbackNudge {
  constructor(device, enabled = FIREFOX) {
    this.device = device; this.enabled = enabled; this.pending = 0; this.running = false;
  }
  wrap(promise) {
    if (!this.enabled) return promise;
    this.pending++;
    if (!this.running) {
      this.running = true;
      const tick = () => {
        if (this.pending <= 0) { this.running = false; return; }
        try { this.device.queue.submit([]); } catch { /* the device is gone */ }
        setTimeout(tick, 1);
      };
      setTimeout(tick, 0);
    }
    return promise.finally(() => { this.pending--; });
  }
}

/** Compiles every kernel a session or a stream runs and waits for the GPU, before the first real frame: Firefox
 * compiles pipelines serially in its GPU process, 0.1 to 2 s each, and they would land on the first frames. */
export async function warmUp(device, sessionOrStream) {
  sessionOrStream.warmUp();
  await device.queue.onSubmittedWorkDone();
}

const BLIT = `
struct VOut { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  let p = pos[i]; var o: VOut; o.p = vec4<f32>(p, 0.0, 1.0); o.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); return o;
}
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment fn fs(i: VOut) -> @location(0) vec4<f32> { return vec4<f32>(textureSampleLevel(t, s, i.uv, 0.0).rgb, 1.0); }`;
const BLIT_EXTERNAL = BLIT
  .replace('var t: texture_2d<f32>;', 'var t: texture_external;')
  .replace('textureSampleLevel(t, s, i.uv, 0.0)', 'textureSampleBaseClampToEdge(t, s, i.uv)');

function pipeline(device, code, format) {
  const module = device.createShaderModule({ code });
  return device.createRenderPipeline({
    layout: 'auto',
    vertex: { module, entryPoint: 'vs' },
    fragment: { module, entryPoint: 'fs', targets: [{ format }] },
    primitive: { topology: 'triangle-list' },
  });
}
function draw(device, pipe, bindGroup, view) {
  const enc = device.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }] });
  pass.setPipeline(pipe); pass.setBindGroup(0, bindGroup); pass.draw(3); pass.end();
  device.queue.submit([enc.finish()]);
}

/** Writes a <video>'s current frame into a stream's input slot: copyExternalImageToTexture where the browser takes a
 * video source and the frame has the slot's size, an importExternalTexture render pass into the slot otherwise (Firefox
 * rejects video sources; a frame of another size is scaled by the pass, the copy is 1:1 only). */
export class VideoSource {
  constructor(device, stream) {
    this.device = device; this.stream = stream; this.copyWorks = !FIREFOX; this.pipe = null; this.sampler = null; this.views = new Map();
  }
  write(slot, video) {
    const tex = this.stream.inputTexture(slot);
    if (this.copyWorks && video.videoWidth === tex.width && video.videoHeight === tex.height) {
      try { this.stream.pushVideo(slot, video); return; } catch (e) {
        if (!(e instanceof TypeError)) throw e;
        console.warn('copyExternalImageToTexture rejected the video, using importExternalTexture', e);
        this.copyWorks = false;
      }
    }
    const d = this.device;
    if (!this.pipe) {
      this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear' });
      this.pipe = pipeline(d, BLIT_EXTERNAL, 'rgba8unorm');
    }
    const ext = d.importExternalTexture({ source: video });
    const bg = d.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: ext }, { binding: 1, resource: this.sampler }] });
    let view = this.views.get(slot);
    if (!view || view.texture !== tex) { view = tex.createView(); view.texture = tex; this.views.set(slot, view); }
    draw(d, this.pipe, bg, view);
  }
}

/** Presents the frame a tick names, a source slot or an output slot of the stream, onto a canvas. */
export class CanvasPresenter {
  constructor(device, canvas, stream) {
    this.device = device; this.stream = stream;
    this.ctx = canvas.getContext('webgpu');
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format, alphaMode: 'opaque' });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.pipe = pipeline(device, BLIT, format);
    this.bindGroups = new Map();
  }
  /** `frame` is `tick.present.frame`: `{ source: slot }` or `{ output: slot }`. */
  present(frame) {
    const source = 'source' in frame;
    const slot = source ? frame.source : frame.output;
    const key = (source ? 'i' : 'o') + slot;
    let bg = this.bindGroups.get(key);
    if (!bg) {
      const tex = source ? this.stream.inputTexture(slot) : this.stream.outputTexture(slot);
      bg = this.device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: tex.createView() }, { binding: 1, resource: this.sampler }] });
      this.bindGroups.set(key, bg);
    }
    draw(this.device, this.pipe, bg, this.ctx.getCurrentTexture().createView());
  }
}

/** requestAnimationFrame as the Pacer's display ticks. A tick later than `stallMs` after the previous one is reported
 * as a host stall (a long task, a hidden tab), so the Pacer does not charge the drops to the GPU. `onTick(now, tick)`
 * presents and generates. */
export class TickSource {
  constructor(pacer, onTick, stallMs = 50) {
    this.pacer = pacer; this.onTick = onTick; this.stallMs = stallMs; this.running = false; this.last = 0;
  }
  start() {
    this.running = true; this.last = 0;
    const loop = (now) => {
      if (!this.running) return;
      requestAnimationFrame(loop);
      if (this.last && now - this.last > this.stallMs) this.pacer.hostStall(now, now - this.last);
      this.last = now;
      this.onTick(now, this.pacer.tick(now));
    };
    requestAnimationFrame(loop);
  }
  stop() { this.running = false; }
}

/** GPU cost samples for the Pacer from the frames' own work: a timed push and a timed generate every `every` calls,
 * plus every probe the Pacer asks for. Replaces `stream.push` and `stream.generate` in the host's loop. */
export class CostSampler {
  constructor(pacer, nudge, every = 30) {
    this.pacer = pacer; this.nudge = nudge; this.every = every; this.pushes = 0; this.mids = 0;
  }
  push(stream, slot, pts) {
    if (this.pushes++ % this.every === 0) {
      const p = stream.pushTimed(slot, pts);
      this.take('pair', p.cost);
      return p;
    }
    return stream.push(slot, pts);
  }
  /** Returns whether a frame was written (what `pacer.generated` / `pacer.skipped` want to know). */
  generate(stream, request, out) {
    if (request.probe || this.mids++ % this.every === 0) {
      const g = stream.generateTimed(request.t, out);
      this.take('mid', g.cost);
      return g.written;
    }
    return stream.generate(request.t, out);
  }
  take(stage, cost) {
    if (!cost) return;
    this.nudge.wrap(cost).then((ms) => { if (ms !== null) this.pacer.costSample(performance.now(), stage, ms); }).catch(() => {});
  }
}
