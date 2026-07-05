// Framegen tiny 2x super-resolution pass (anime upscale) for the present path.
// Residual-vs-bilinear: out = bilinear2x(src) + detail, detail = 3 tiny convs (c=16).
// Texture in -> texture out (2x size); everything stays on the GPU.
// Weights: assets/rt_sr.{bin,json} (tools/export_sr_weights.py).

import { wgslConvRB, WGSL_TO_F16 } from './rt.js';

const WG = 8;

function wgslIn(C) {
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;   // [C,3,3,3]
@group(0) @binding(3) var<storage, read> bias: array<f32>;  // [C]
@group(0) @binding(4) var<storage, read> alpha: array<f32>; // [C]
@group(0) @binding(5) var<storage, read_write> dst: array<f16>; // [C,H,W]
@group(0) @binding(6) var<storage, read> dims: array<u32>;  // [W,H]

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(dims[0]); let H = i32(dims[1]);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  var px: array<vec3<f32>, 9>;
  for (var ky = 0; ky < 3; ky++) {
    for (var kx = 0; kx < 3; kx++) {
      let sx = clamp(x + kx - 1, 0, W - 1);
      let sy = clamp(y + ky - 1, 0, H - 1);
      let uv = (vec2<f32>(f32(sx), f32(sy)) + 0.5) / vec2<f32>(f32(W), f32(H));
      let c = textureSampleLevel(src, samp, uv, 0.0).rgb;
      px[ky * 3 + kx] = vec3<f32>(c.b, c.g, c.r); // BGR domain like the rest
    }
  }
  for (var co = 0; co < ${C}; co++) {
    var acc = bias[co];
    for (var k = 0; k < 9; k++) {
      let wb = co * 27 + k;
      acc += px[k].x * wgt[wb] + px[k].y * wgt[wb + 9] + px[k].z * wgt[wb + 18];
    }
    let v = select(alpha[co] * acc, acc, acc >= 0.0);
    dst[co * H * W + y * W + x] = f16(v);
  }
}`;
}

// NOTE: torch Conv2d weight is [CO,CI,3,3]; the wgslIn indexing above expects
// [CO][CI][k] flattened as co*27 + ci*9 + k - matches torch layout directly.
// mid convs use the register-blocked kernel from rt.js (2x2 patch x 4 output
// channels per thread, shared-memory tiles) - the naive per-pixel loops cost
// 3.5x more at c=32. The out conv computes ALL 12 pixel-shuffle outputs per
// low-res pixel in one thread - four 2x-quadrant threads would re-read the
// same CxHxW window four times.
function wgslShuffle(W, H) {
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> det: array<f16>;    // [12,H,W] conv output
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x); let oy = i32(gid.y);
  if (ox >= ${W * 2} || oy >= ${H * 2}) { return; }
  let x = ox / 2; let y = oy / 2;
  let sub = (oy & 1) * 2 + (ox & 1);
  let p = y * ${W} + x;
  // detail channels [b,g,r] live at co = ch*4 + sub (torch PixelShuffle layout)
  let db = f32(det[u32(sub) * ${H * W}u + u32(p)]);
  let dg = f32(det[(u32(sub) + 4u) * ${H * W}u + u32(p)]);
  let dr = f32(det[(u32(sub) + 8u) * ${H * W}u + u32(p)]);
  let uv = (vec2<f32>(f32(ox), f32(oy)) + 0.5) / vec2<f32>(${W * 2}.0, ${H * 2}.0);
  let base = textureSampleLevel(srcTex, samp, uv, 0.0).rgb;
  let b = clamp(base.b + db, 0.0, 1.0);
  let g = clamp(base.g + dg, 0.0, 1.0);
  let r = clamp(base.r + dr, 0.0, 1.0);
  textureStore(outTex, vec2<i32>(ox, oy), vec4<f32>(r, g, b, 1.0));
}`;
}

export async function createSR(device, { weightsBin, weightsManifest, channels }) {
  // channel width lives in the weights: c1 is the 3->C input conv
  const C = channels || (weightsManifest['c1.weight'] ? weightsManifest['c1.weight'].shape[0] : 16);
  const bufN = (bytes) => device.createBuffer({
    size: Math.ceil(bytes / 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

  const wbuf = {};
  for (const [name, m] of Object.entries(weightsManifest)) {
    const n = m.shape.reduce((a, b) => a * b, 1);
    wbuf[name] = bufN(n * 4);
    device.queue.writeBuffer(wbuf[name], 0, weightsBin, m.offset * 4, n * 4);
  }
  const sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear',
    addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
  const pipe = (code) => device.createComputePipeline({ layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const pIn = pipe(wgslIn(C));
  // f16 copies of the heavy conv weights (the RB kernels read f16)
  const pToH = pipe(WGSL_TO_F16);
  const wbufH = {};
  {
    const enc = device.createCommandEncoder();
    for (const name of ['c2.weight', 'c3.weight', 'c4.weight']) {
      const n = weightsManifest[name].shape.reduce((a, b) => a * b, 1);
      wbufH[name] = device.createBuffer({ size: Math.ceil(n / 2) * 4,
        usage: GPUBufferUsage.STORAGE });
      const pass = enc.beginComputePass();
      pass.setPipeline(pToH);
      pass.setBindGroup(0, device.createBindGroup({ layout: pToH.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: wbuf[name] } },
          { binding: 1, resource: { buffer: wbufH[name] } }] }));
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
    }
    device.queue.submit([enc.finish()]);
  }
  const onesAlpha = bufN(12 * 4);
  device.queue.writeBuffer(onesAlpha, 0, new Float32Array(12).fill(1));

  // per-input-size state (feature buffers + dims); keyed by "WxH"
  const states = new Map();
  function stateFor(w, h) {
    const k = w + 'x' + h;
    if (!states.has(k)) {
      const dims = bufN(8);
      device.queue.writeBuffer(dims, 0, new Uint32Array([w, h]));
      const fa = bufN(C * w * h * 2);
      const fb = bufN(C * w * h * 2);
      const det = bufN(12 * w * h * 2);
      const pMid = pipe(wgslConvRB(C, C, w, h, w, h, false));
      const pOutConv = pipe(wgslConvRB(C, 12, w, h, w, h, false));
      const pShuf = pipe(wgslShuffle(w, h));
      const midBg = (wname, aname, sBuf, dBuf) => device.createBindGroup({
        layout: pMid.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: sBuf } },
          { binding: 1, resource: { buffer: wbufH[wname] } },
          { binding: 2, resource: { buffer: wbuf[wname.replace('.weight', '.bias')] } },
          { binding: 3, resource: { buffer: wbuf[aname] } },
          { binding: 4, resource: { buffer: dBuf } }] });
      states.set(k, {
        dims, fa, fb, det, pMid, pOutConv, pShuf,
        bgM2: midBg('c2.weight', 'a2.weight', fa, fb),
        bgM3: midBg('c3.weight', 'a3.weight', fb, fa),
        bgOut: device.createBindGroup({ layout: pOutConv.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: fa } },
          { binding: 1, resource: { buffer: wbufH['c4.weight'] } },
          { binding: 2, resource: { buffer: wbuf['c4.bias'] } },
          { binding: 3, resource: { buffer: onesAlpha } },
          { binding: 4, resource: { buffer: det } }] }),
      });
      if (states.size > 6) { // sizes changed wholesale
        for (const [kk, s] of states) if (kk !== k) { s.fa.destroy(); s.fb.destroy(); s.det.destroy(); s.dims.destroy(); states.delete(kk); }
      }
    }
    return states.get(k);
  }

  const gx = (n) => Math.ceil(n / WG);
  // keyed by texture OBJECT, not label: the host rebuilds its texture pools on
  // settings changes reusing the same labels - a label-keyed cache then binds
  // views of DESTROYED textures and every SR'd frame comes out corrupted until
  // page reload. WeakMaps also let dead textures drop their bind groups with GC.
  const bgCache = new WeakMap();

  // srcTex (w x h) -> dstTex (2w x 2h, rgba8unorm STORAGE_BINDING)
  function process(srcTex, dstTex, w, h) {
    const S = stateFor(w, h);
    let perSrc = bgCache.get(srcTex);
    if (!perSrc) { perSrc = new WeakMap(); bgCache.set(srcTex, perSrc); }
    let bgs = perSrc.get(dstTex);
    if (!bgs) {
      const srcView = srcTex.createView();
      bgs = {
        in: device.createBindGroup({ layout: pIn.getBindGroupLayout(0), entries: [
          { binding: 0, resource: srcView }, { binding: 1, resource: sampler },
          { binding: 2, resource: { buffer: wbuf['c1.weight'] } },
          { binding: 3, resource: { buffer: wbuf['c1.bias'] } },
          { binding: 4, resource: { buffer: wbuf['a1.weight'] } },
          { binding: 5, resource: { buffer: S.fa } },
          { binding: 6, resource: { buffer: S.dims } }] }),
        shuf: device.createBindGroup({ layout: S.pShuf.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: S.det } },
          { binding: 1, resource: srcView }, { binding: 2, resource: sampler },
          { binding: 3, resource: dstTex.createView() }] }),
      };
      perSrc.set(dstTex, bgs);
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pIn); pass.setBindGroup(0, bgs.in); pass.dispatchWorkgroups(gx(w), gx(h));
    pass.setPipeline(S.pMid);
    pass.setBindGroup(0, S.bgM2); pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16), C / 4);
    pass.setBindGroup(0, S.bgM3); pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16), C / 4);
    pass.setPipeline(S.pOutConv); pass.setBindGroup(0, S.bgOut);
    pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16), 3);
    pass.setPipeline(S.pShuf); pass.setBindGroup(0, bgs.shuf);
    pass.dispatchWorkgroups(gx(w * 2), gx(h * 2));
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  return { process };
}
