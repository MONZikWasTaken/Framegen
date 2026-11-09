// The overlay canvas's present path: a frame texture (a source slot, an output slot or an upscaled frame) onto the
// canvas, SDR or HDR (inverse tone mapping on an fp16 canvas in extended tone-mapping mode: highlights expand past SDR
// white, the idea of RTX Video HDR; the browser only ever hands us tonemapped SDR), with the optional sharpening and the
// compare split (the delayed source frame on the left of the divider).
const BLIT_VS = `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: VOut;
  o.pos = vec4(p[i], 0.0, 1.0);
  o.uv = vec2(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}`;

function fragment(sharpness, hdr) {
  const sampleColor = sharpness ? `
fn sampleColor(uv: vec2<f32>) -> vec3<f32> {
  let c = textureSampleLevel(tex, samp, uv, 0.0).rgb;
  let px = 1.0 / vec2<f32>(textureDimensions(tex));
  let blur = (textureSampleLevel(tex, samp, uv + vec2<f32>(px.x, 0.0), 0.0).rgb
    + textureSampleLevel(tex, samp, uv - vec2<f32>(px.x, 0.0), 0.0).rgb
    + textureSampleLevel(tex, samp, uv + vec2<f32>(0.0, px.y), 0.0).rgb
    + textureSampleLevel(tex, samp, uv - vec2<f32>(0.0, px.y), 0.0).rgb) * 0.25;
  return clamp(c + (c - blur) * ${sharpness.toFixed(1)}, vec3<f32>(0.0), vec3<f32>(1.0));
}` : `
fn sampleColor(uv: vec2<f32>) -> vec3<f32> {
  return textureSampleLevel(tex, samp, uv, 0.0).rgb;
}`;
  return sampleColor + (hdr ? `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  let c = sampleColor(v.uv);
  let lin = pow(max(c, vec3(0.0)), vec3(2.2));
  let y = max(lin.r, max(lin.g, lin.b));
  let t = smoothstep(0.35, 1.0, y);
  let gain = 1.0 + 2.2 * t * t;
  return vec4(pow(lin * gain, vec3(1.0 / 2.2)), 1.0);
}` : `
@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
  return vec4<f32>(sampleColor(v.uv), 1.0);
}`);
}

export class Presenter {
  constructor(device, canvas, log) {
    this.device = device;
    this.log = log;
    this.ctx = canvas.getContext('webgpu');
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    this.pipe = null;
    this.hdr = false;
    this.bindGroups = new Map();
    this.presents = 0;
  }

  // Returns whether HDR output is on after this configuration.
  configure({ hdr, sharpness }) {
    let hdrOn = !!hdr;
    try {
      this.ctx.configure({ device: this.device, format: hdrOn ? 'rgba16float' : 'rgba8unorm', alphaMode: 'opaque',
        ...(hdrOn ? { colorSpace: 'srgb', toneMapping: { mode: 'extended' } } : {}) });
    } catch (e) {
      this.log('hdr configure failed, falling back to SDR', e);
      hdrOn = false;
      this.ctx.configure({ device: this.device, format: 'rgba8unorm', alphaMode: 'opaque' });
    }
    const module = this.device.createShaderModule({ code: BLIT_VS + fragment(sharpness, hdrOn) });
    // compiled off the thread that composites the page: a synchronous pipeline on a site whose shader cache is cold
    // held the video for about 0.2 s at the first present
    this.pipe = null;
    this.bindGroups.clear();
    const gen = (this.gen || 0) + 1;
    this.gen = gen;
    this.ready = this.device.createRenderPipelineAsync({ layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: hdrOn ? 'rgba16float' : 'rgba8unorm' }] } })
      .then(pipe => { if (this.gen === gen) { this.pipe = pipe; this.bindGroups.clear(); } })
      .catch(e => this.log('present pipeline', e));
    this.hdr = hdrOn;
    return hdrOn;
  }

  bindGroup(texture) {
    let bg = this.bindGroups.get(texture);
    if (!bg) {
      if (this.bindGroups.size > 48) this.bindGroups.clear();
      bg = this.device.createBindGroup({ layout: this.pipe.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: texture.createView() }, { binding: 1, resource: this.sampler }] });
      this.bindGroups.set(texture, bg);
    }
    return bg;
  }

  // Returns whether anything was drawn (nothing until the pipeline exists).
  present(texture, { compare = null, splitX = 0.5, width = 0, height = 0 } = {}) {
    if (!this.pipe) return false;
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginRenderPass({ colorAttachments: [{
      view: this.ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bindGroup(texture));
    pass.draw(3);
    if (compare && compare !== texture && width && height) {
      pass.setScissorRect(0, 0, Math.max(1, Math.round(splitX * width)), height);
      pass.setBindGroup(0, this.bindGroup(compare));
      pass.draw(3);
    }
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.presents++;
    return true;
  }

  // The slot textures changed (a new stream): the cached bind groups point at destroyed textures.
  invalidate() { this.bindGroups.clear(); }

  destroy() {
    this.bindGroups.clear();
    try { this.ctx.unconfigure(); } catch { /* never configured */ }
  }
}
