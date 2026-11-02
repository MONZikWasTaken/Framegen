// TinySR, the 2x neural upscale on the present path (rt/sr.js, assets/rt_sr): every presented frame goes through it
// when it adds pixels toward the canvas. Generated frames only was tried and shimmered sharp/soft at display rate.
const UPSCALE_RATIO = 1.15;
const DEFAULT_TUNE = { coc: 8, slab: 12, sg: true, w4: true, v2: true };

export class Upscaler {
  constructor(device, urls, { nudge, log }) {
    this.device = device;
    this.urls = urls;
    this.nudge = nudge;
    this.log = log;
    this.sr = null;
    this.loading = null;
    this.outputs = new Map();
    this.processed = 0;
    this.pendingCost = false;
  }

  get ready() { return !!this.sr; }
  get scale() { return this.sr ? (this.sr.scale || 2) : 2; }

  load() {
    if (this.sr || this.loading) return this.loading || Promise.resolve();
    this.loading = (async () => {
      const [bin, manifest, mod] = await Promise.all([
        fetch(this.urls.weights).then(r => r.arrayBuffer()),
        fetch(this.urls.manifest).then(r => r.json()),
        import(this.urls.module)]);
      // the SR convs run at full video resolution, where the w4/v2 kernel variants measured about 2x; sr.js
      // feature-gates the subgroup variant itself
      const sr = await mod.createSR(this.device, { weightsBin: bin, weightsManifest: manifest, convTune: DEFAULT_TUNE });
      if (this.destroyed) { try { sr.destroy(); } catch { /* gone */ } return; }
      this.sr = sr;
      this.log('SR up');
    })().finally(() => { this.loading = null; });
    return this.loading;
  }

  static needed(sourceWidth, sourceHeight, canvasWidth, canvasHeight) {
    if (!sourceWidth || !sourceHeight || !canvasWidth || !canvasHeight) return false;
    return canvasWidth / sourceWidth > UPSCALE_RATIO || canvasHeight / sourceHeight > UPSCALE_RATIO;
  }

  output(width, height) {
    const key = width + 'x' + height;
    let out = this.outputs.get(key);
    if (out) return out;
    out = this.device.createTexture({ label: 'fc-sr' + key, size: [width * this.scale, height * this.scale], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
    this.outputs.set(key, out);
    if (this.outputs.size > 4) {
      for (const [k, t] of this.outputs) if (k !== key) { try { t.destroy(); } catch { /* gone */ } this.outputs.delete(k); }
      this.invalidated = true;
    }
    return out;
  }

  // Returns the upscaled texture, or null while the per-size pipelines still compile (show the frame as is). Every
  // eighth frame is timed and fed to the Pacer as the present cost.
  process(texture, now, pacer) {
    if (!this.sr) return null;
    const out = this.output(texture.width, texture.height);
    const timed = this.sr.hasGpuTimestamps && !this.pendingCost && (this.processed % 8 === 0);
    let ok;
    if (timed) {
      const result = this.sr.processTimed(texture, out, texture.width, texture.height);
      ok = result !== null;
      if (ok && result.timing && pacer) {
        this.pendingCost = true;
        this.nudge.wrap(result.timing)
          .then(ms => { if (Number.isFinite(ms) && ms > 0) pacer.costSample(performance.now(), 'present', ms); })
          .catch(() => {})
          .finally(() => { this.pendingCost = false; });
      }
    } else {
      ok = this.sr.process(texture, out, texture.width, texture.height);
    }
    if (!ok) return null;
    this.processed++;
    return out;
  }

  destroy() {
    this.destroyed = true;
    try { this.sr?.destroy(); } catch { /* gone */ }
    this.sr = null;
    for (const t of this.outputs.values()) { try { t.destroy(); } catch { /* gone */ } }
    this.outputs.clear();
  }
}
