// Framegen tiny 2x super-resolution pass (anime upscale) for the present path.
// Residual-vs-bilinear: out = bilinear2x(src) + detail, detail = 3 tiny convs (c=16).
// Texture in -> texture out (2x size); everything stays on the GPU.
// Weights: assets/rt_sr.{bin,json} (tools/export_sr_weights.py).

import { wgslConvRB, wgslConvRBSg, WGSL_TO_F32, WGSL_TO_F16 } from './rt.js?v=8';

const WG = 8;

function wgslIn(C, tune) {
  const WGX = (tune && tune.wgx) || 8, WGY = (tune && tune.wgy) || 8;
  const TL = !!(tune && tune.tl); // textureLoad: exact texel reads, no sampler/uv math
  return wgslInBody(C, WGX, WGY, TL);
}
function wgslInBody(C, WGX, WGY, TL) {
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<storage, read> wgt: array<f32>;   // [C,3,3,3]
@group(0) @binding(3) var<storage, read> bias: array<f32>;  // [C]
@group(0) @binding(4) var<storage, read> alpha: array<f32>; // [C]
@group(0) @binding(5) var<storage, read_write> dst: array<f16>; // [C,H,W]
@group(0) @binding(6) var<storage, read> dims: array<u32>;  // [W,H]

@compute @workgroup_size(${WGX}, ${WGY})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let W = i32(dims[0]); let H = i32(dims[1]);
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= W || y >= H) { return; }
  var px: array<vec3<f32>, 9>;
  for (var ky = 0; ky < 3; ky++) {
    for (var kx = 0; kx < 3; kx++) {
      let sx = clamp(x + kx - 1, 0, W - 1);
      let sy = clamp(y + ky - 1, 0, H - 1);
      ${TL ? `let c = textureLoad(src, vec2<i32>(sx, sy), 0).rgb;` : `let uv = (vec2<f32>(f32(sx), f32(sy)) + 0.5) / vec2<f32>(f32(W), f32(H));
      let c = textureSampleLevel(src, samp, uv, 0.0).rgb;`}
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
function wgslShuffle(W, H, S) {
  // S = upscale factor: 4 (det [48,H,W]), 2 (det [12,H,W]), or 1 - pure
  // restore/deblock, det is a plain [3,H,W] residual on the source
  const S2 = S * S;
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> det: array<f16>;    // [${3 * S2},${H},${W}] conv output
@group(0) @binding(1) var srcTex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ox = i32(gid.x); let oy = i32(gid.y);
  if (ox >= ${W * S} || oy >= ${H * S}) { return; }
  let x = ox / ${S}; let y = oy / ${S};
  let sub = u32((oy % ${S}) * ${S} + ox % ${S});
  let p = u32(y * ${W} + x);
  // detail channels [b,g,r] live at co = ch*${S2} + sub (torch PixelShuffle layout)
  let db = f32(det[sub * ${H * W}u + p]);
  let dg = f32(det[(sub + ${S2}u) * ${H * W}u + p]);
  let dr = f32(det[(sub + ${2 * S2}u) * ${H * W}u + p]);
  let uv = (vec2<f32>(f32(ox), f32(oy)) + 0.5) / vec2<f32>(${W * S}.0, ${H * S}.0);
  let base = textureSampleLevel(srcTex, samp, uv, 0.0).rgb;
  let b = clamp(base.b + db, 0.0, 1.0);
  let g = clamp(base.g + dg, 0.0, 1.0);
  let r = clamp(base.r + dr, 0.0, 1.0);
  textureStore(outTex, vec2<i32>(ox, oy), vec4<f32>(r, g, b, 1.0));
}`;
}

export async function createSR(device, { weightsBin, weightsManifest, channels, convTune }) {
  // convTune: same shape dict as the interpolation runtime's tuner output -
  // the SR convs run at FULL video resolution, so the w4/v2 kernel variants
  // matter here more than anywhere (these grids are ~256x the trunk's)
  // channel width lives in the weights: c1 is the 3->C input conv
  const C = channels || (weightsManifest['c1.weight'] ? weightsManifest['c1.weight'].shape[0] : 16);
  // upscale factor lives in the out conv: 3*S^2 pixel-shuffle channels
  const C4 = weightsManifest['c4.weight'].shape[0];      // 12 (2x), 48 (4x), 3 (1x restore)
  const SCALE = C4 === 48 ? 4 : C4 === 3 ? 1 : 2;
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
  // async compiles: the RB conv shaders are heavily unrolled and a sync
  // createComputePipeline stalls the main thread for tens of ms - exactly when
  // the user flips SR on. Dawn compiles these on its worker pool instead.
  const pipeAsync = (code) => device.createComputePipelineAsync({ layout: 'auto',
    compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  // f16 copies of the heavy conv weights (the RB kernels read f16)
  const sgOk = convTune && convTune.sg && device.features.has('subgroups');
  const RB = sgOk ? wgslConvRBSg : wgslConvRB;
  const needW4 = !!(convTune && convTune.w4);
  // in-conv: textureLoad (exact texels, same values the linear sampler returns
  // at texel centers) + a 16x8 workgroup - measured -20% over the 8x8 sampler
  // version on Ada
  const IN_TUNE = { tl: true, wgx: 16, wgy: 8 };
  const [pIn, pToH, pToF] = await Promise.all([pipeAsync(wgslIn(C, IN_TUNE)), pipeAsync(WGSL_TO_F16),
    needW4 ? pipeAsync(WGSL_TO_F32) : null]);
  const wbufH = {};
  {
    const enc = device.createCommandEncoder();
    for (const name of ['c2.weight', 'c3.weight', 'c4.weight']) {
      const n = weightsManifest[name].shape.reduce((a, b) => a * b, 1);
      const half = device.createBuffer({ size: Math.ceil(n / 2) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const pass = enc.beginComputePass();
      pass.setPipeline(pToH);
      pass.setBindGroup(0, device.createBindGroup({ layout: pToH.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: wbuf[name] } },
          { binding: 1, resource: { buffer: half } }] }));
      pass.dispatchWorkgroups(Math.ceil(n / 256));
      pass.end();
      if (needW4) { // widen back to f32: bit-exact f32(f16(w)), CVT leaves the hot loop
        const wide = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.STORAGE });
        const p2 = enc.beginComputePass();
        p2.setPipeline(pToF);
        p2.setBindGroup(0, device.createBindGroup({ layout: pToF.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: half } },
            { binding: 1, resource: { buffer: wide } }] }));
        p2.dispatchWorkgroups(Math.ceil(n / 256));
        p2.end();
        wbufH[name] = wide;
      } else {
        wbufH[name] = half;
      }
    }
    device.queue.submit([enc.finish()]);
  }
  const onesAlpha = bufN(C4 * 4);
  device.queue.writeBuffer(onesAlpha, 0, new Float32Array(C4).fill(1));

  // per-input-size state (feature buffers + dims); keyed by "WxH". Built ASYNC:
  // the first process() call at a new size kicks the build and returns null -
  // the caller keeps presenting un-upscaled frames until the pipelines are ready
  // (a sync build here froze the page right as SR engaged).
  const states = new Map();
  // the trunk's convTune was measured at the TRUNK shape (96-240ch, H/16 grid);
  // the SR convs run 16ch at FULL video res where wider channel blocks win
  // (fewer z-slices re-staging the same tile: mid coc16 -11%, out z=1 -34%
  // measured on Ada). Candidates are few - bench them on the real size during
  // the async state build (kernel speed does not depend on weight values) and
  // keep the winner. Same kernel family, same accumulation order: bit-exact.
  async function pickConv(cands, srcB, dstB, wName, w, h) {
    const pipes = await Promise.all(cands.map(v => pipeAsync(
      (v.sg && device.features.has('subgroups') ? wgslConvRBSg : wgslConvRB)(
        v.ci, v.co, w, h, w, h, false, v.tune))));
    let best = null;
    for (let i = 0; i < cands.length; i++) {
      const v = cands[i];
      const bgB = device.createBindGroup({ layout: pipes[i].getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: srcB } },
        { binding: 1, resource: { buffer: wbufH[wName] } },
        { binding: 2, resource: { buffer: wbuf[wName.replace('.weight', '.bias')] } },
        { binding: 3, resource: { buffer: v.alpha } },
        { binding: 4, resource: { buffer: dstB } }] });
      const tx = ((v.tune.wgx || 8) * 2), ty = ((v.tune.wgy || 8) * 2);
      const run = (k) => {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipes[i]); pass.setBindGroup(0, bgB);
        for (let j = 0; j < k; j++) {
          pass.dispatchWorkgroups(Math.ceil(w / tx), Math.ceil(h / ty), v.co / v.tune.coc);
        }
        pass.end();
        device.queue.submit([enc.finish()]);
      };
      run(3); await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      run(20); await device.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t0) / 20;
      if (!best || ms < best.ms) best = { pipe: pipes[i], tune: v.tune, ms };
    }
    return best;
  }
  function stateFor(w, h) {
    const k = w + 'x' + h;
    const st = states.get(k);
    if (st) return st.ready ? st : null;
    const building = { ready: false };
    states.set(k, building);
    (async () => {
      const dims = bufN(8);
      device.queue.writeBuffer(dims, 0, new Uint32Array([w, h]));
      const fa = bufN(C * w * h * 2);
      const fb = bufN(C * w * h * 2);
      const det = bufN(C4 * w * h * 2);
      const outCoc = C4 % 8 === 0 ? 8 : C4 % 4 === 0 ? 4 : 1; // CO must divide
      const baseT = convTune || { coc: Math.min(8, C), slab: C, w4: false, v2: false };
      // slab must fit the 16KB shared budget at this tune's workgroup shape
      const fit = (t, sg) => {
        const ts = (t.wgx || 8) * 2 + 2, th2 = (t.wgy || 8) * 2 + 2;
        let slab = t.slab;
        while (slab > 1 && slab * ts * th2 * 2 + (sg ? 0 : t.coc * slab * 9 * (t.w4 ? 4 : 2)) > 16384) slab--;
        return { ...t, slab };
      };
      const cand = (co, alpha, sg, t) => ({ ci: C, co, sg, alpha, tune: fit(t, sg && device.features.has('subgroups')) });
      const midCands = [cand(C, wbuf['a2.weight'], !!baseT.sg, { ...baseT, slab: C })];
      if (C % 16 === 0) {
        midCands.push(cand(C, wbuf['a2.weight'], !!baseT.sg, { ...baseT, coc: 16, slab: C }));
        midCands.push(cand(C, wbuf['a2.weight'], false, { ...baseT, coc: 16, slab: 8, wgx: 16, wgy: 8 }));
      }
      const outCands = [cand(C4, onesAlpha, !!baseT.sg, { ...baseT, coc: outCoc, slab: C })];
      if (C4 <= 16 && C4 !== outCoc) { // z=1: every channel in one block
        outCands.push(cand(C4, onesAlpha, !!baseT.sg, { ...baseT, coc: C4, slab: C }));
      }
      const [mid, out, pShuf] = await Promise.all([
        pickConv(midCands, fa, fb, 'c2.weight', w, h),
        pickConv(outCands, fa, det, 'c4.weight', w, h),
        pipeAsync(wgslShuffle(w, h, SCALE))]);
      const pMid = mid.pipe, pOutConv = out.pipe;
      const midBg = (wname, aname, sBuf, dBuf) => device.createBindGroup({
        layout: pMid.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: sBuf } },
          { binding: 1, resource: { buffer: wbufH[wname] } },
          { binding: 2, resource: { buffer: wbuf[wname.replace('.weight', '.bias')] } },
          { binding: 3, resource: { buffer: wbuf[aname] } },
          { binding: 4, resource: { buffer: dBuf } }] });
      const dd = (t, co) => [Math.ceil(w / ((t.wgx || 8) * 2)), Math.ceil(h / ((t.wgy || 8) * 2)), co / t.coc];
      Object.assign(building, {
        midDims: dd(mid.tune, C), outDims: dd(out.tune, C4),
        dims, fa, fb, det, pMid, pOutConv, pShuf,
        bgM2: midBg('c2.weight', 'a2.weight', fa, fb),
        bgM3: midBg('c3.weight', 'a3.weight', fb, fa),
        bgOut: device.createBindGroup({ layout: pOutConv.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: fa } },
          { binding: 1, resource: { buffer: wbufH['c4.weight'] } },
          { binding: 2, resource: { buffer: wbuf['c4.bias'] } },
          { binding: 3, resource: { buffer: onesAlpha } },
          { binding: 4, resource: { buffer: det } }] }),
        ready: true,
      });
      if (states.size > 6) { // sizes changed wholesale
        for (const [kk, s] of states) if (kk !== k && s.ready) { s.fa.destroy(); s.fb.destroy(); s.det.destroy(); s.dims.destroy(); states.delete(kk); }
      }
    })().catch(() => states.delete(k)); // failed build: retry on a later frame
    return null;
  }

  const gx = (n) => Math.ceil(n / WG);
  // keyed by texture OBJECT, not label: the host rebuilds its texture pools on
  // settings changes reusing the same labels - a label-keyed cache then binds
  // views of DESTROYED textures and every SR'd frame comes out corrupted until
  // page reload. WeakMaps also let dead textures drop their bind groups with GC.
  const bgCache = new WeakMap();

  // srcTex (w x h) -> dstTex (2w x 2h, rgba8unorm STORAGE_BINDING).
  // Returns false while the per-size pipelines are still compiling - the caller
  // should present the original texture that frame.
  function process(srcTex, dstTex, w, h) {
    const S = stateFor(w, h);
    if (!S) return false;
    let perSrc = bgCache.get(srcTex);
    if (!perSrc) { perSrc = new WeakMap(); bgCache.set(srcTex, perSrc); }
    let bgs = perSrc.get(dstTex);
    // bgs must belong to the CURRENT per-size state: eviction destroys fa/det,
    // and a cached bind group referencing them fails validation forever after
    if (!bgs || bgs.S !== S) {
      const srcView = srcTex.createView();
      bgs = {
        // no sampler entry: the textureLoad in-conv never binds one ('auto'
        // strips the unused binding)
        in: device.createBindGroup({ layout: pIn.getBindGroupLayout(0), entries: [
          { binding: 0, resource: srcView },
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
      bgs.S = S;
      perSrc.set(dstTex, bgs);
    }
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pIn); pass.setBindGroup(0, bgs.in); pass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 8));
    pass.setPipeline(S.pMid);
    pass.setBindGroup(0, S.bgM2); pass.dispatchWorkgroups(S.midDims[0], S.midDims[1], S.midDims[2]);
    pass.setBindGroup(0, S.bgM3); pass.dispatchWorkgroups(S.midDims[0], S.midDims[1], S.midDims[2]);
    pass.setPipeline(S.pOutConv); pass.setBindGroup(0, S.bgOut);
    pass.dispatchWorkgroups(S.outDims[0], S.outDims[1], S.outDims[2]);
    pass.setPipeline(S.pShuf); pass.setBindGroup(0, bgs.shuf);
    pass.dispatchWorkgroups(gx(w * SCALE), gx(h * SCALE));
    pass.end();
    device.queue.submit([enc.finish()]);
    return true;
  }

  return { process, scale: SCALE };
}
