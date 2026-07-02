// Framecast custom WebGPU runtime — hand-rolled forward of the 1-block student
// (block0 of IFNet_m, scale=4, timestep=0.5). The whole frame is ONE command buffer
// of ~13 dispatches; no per-op JS, no runtime glue. Weights: assets/rt_1blk.{bin,json}
// (tools/export_rt_weights.py).
//
// Graph replicated from model/IFNet_m.py:
//   x  = cat(img0,img1,t) BGR /255                     [7,H,W]
//   xq = bilinear(x, 1/4, align_corners=false)         [7,H/4,W/4]
//   f8   = prelu(conv3x3s2(xq))                        [120,H/8,W/8]
//   f16  = prelu(conv3x3s2(f8))                        [240,H/16,W/16]
//   f16  = convblock(f16) + f16   (8x conv3x3s1+prelu, residual)
//   tmp8 = deconv4x4s2(f16)                            [5,H/8,W/8]
//   tmpF = bilinear(tmp8, x8, align_corners=false); flow = tmpF[0:4]*8; mask = tmpF[4]
//   mid  = sigmoid(mask)*warp(img0,flow01) + (1-s)*warp(img1,flow23)
// Requires H,W divisible by 16.

const WG = 8;

// NOTE: with layout:'auto' WebGPU strips unused bindings from the layout, so each
// entry point gets its own shader with exactly the bindings it touches.
function wgslPrepFull(W, H) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> rgba0: array<u32>;
@group(0) @binding(1) var<storage, read> rgba1: array<u32>;
@group(0) @binding(2) var<storage, read_write> imgs: array<f32>;  // [6,${H},${W}] b0 g0 r0 b1 g1 r1

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${W} || y >= ${H}) { return; }
  let o = y * ${W} + x;
  let c0 = unpack4x8unorm(rgba0[o]).xyz;
  let c1 = unpack4x8unorm(rgba1[o]).xyz;
  let P = ${H * W};
  imgs[o] = c0.z; imgs[P + o] = c0.y; imgs[2 * P + o] = c0.x;
  imgs[3 * P + o] = c1.z; imgs[4 * P + o] = c1.y; imgs[5 * P + o] = c1.x;
}`;
}

function wgslPrepQuarter(W, H) {
  const QW = W / 4, QH = H / 4;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> rgba0: array<u32>;
@group(0) @binding(1) var<storage, read> rgba1: array<u32>;
@group(0) @binding(2) var<storage, read_write> xq: array<f32>;    // [7,${QH},${QW}]
@group(0) @binding(3) var<storage, read> tstep: array<f32>;       // [1] timestep

fn px(buf: i32, x: i32, y: i32) -> vec3<f32> {
  let v = select(rgba1[y * ${W} + x], rgba0[y * ${W} + x], buf == 0);
  return unpack4x8unorm(v).xyz; // r,g,b in 0..1
}

fn sampleQ(buf: i32, sx: f32, sy: f32) -> vec3<f32> {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  let xa = clamp(x0, 0, ${W - 1}); let xb = clamp(x0 + 1, 0, ${W - 1});
  let ya = clamp(y0, 0, ${H - 1}); let yb = clamp(y0 + 1, 0, ${H - 1});
  let v00 = px(buf, xa, ya); let v10 = px(buf, xb, ya);
  let v01 = px(buf, xa, yb); let v11 = px(buf, xb, yb);
  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${QW} || y >= ${QH}) { return; }
  // align_corners=false: src = (dst+0.5)*4 - 0.5
  let sx = (f32(x) + 0.5) * 4.0 - 0.5;
  let sy = (f32(y) + 0.5) * 4.0 - 0.5;
  let c0 = sampleQ(0, sx, sy);
  let c1 = sampleQ(1, sx, sy);
  let o = y * ${QW} + x;
  let P = ${QH * QW};
  xq[o] = c0.z; xq[P + o] = c0.y; xq[2 * P + o] = c0.x;
  xq[3 * P + o] = c1.z; xq[4 * P + o] = c1.y; xq[5 * P + o] = c1.x;
  xq[6 * P + o] = tstep[0]; // timestep
}`;
}

// generic conv3x3 (stride 1 or 2) + bias + PReLU, optional residual add (post-activation)
function wgslConv(CI, CO, IW, IH, OW, OH, stride, residual) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;      // [${CI},${IH},${IW}]
@group(0) @binding(1) var<storage, read> wgt: array<f32>;      // [${CO},${CI},3,3]
@group(0) @binding(2) var<storage, read> bias: array<f32>;     // [${CO}]
@group(0) @binding(3) var<storage, read> alpha: array<f32>;    // [${CO}] prelu
@group(0) @binding(4) var<storage, read_write> dst: array<f32>; // [${CO},${OH},${OW}]
${residual ? `@group(0) @binding(5) var<storage, read> res: array<f32>;` : ``}

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y); let co = i32(gid.z);
  if (x >= ${OW} || y >= ${OH}) { return; }
  var acc = bias[co];
  let wbase = co * ${CI * 9};
  for (var ci = 0; ci < ${CI}; ci++) {
    let sbase = ci * ${IH * IW};
    let wb = wbase + ci * 9;
    for (var ky = 0; ky < 3; ky++) {
      let iy = y * ${stride} + ky - 1;
      if (iy < 0 || iy >= ${IH}) { continue; }
      for (var kx = 0; kx < 3; kx++) {
        let ix = x * ${stride} + kx - 1;
        if (ix < 0 || ix >= ${IW}) { continue; }
        acc += src[sbase + iy * ${IW} + ix] * wgt[wb + ky * 3 + kx];
      }
    }
  }
  let v = select(alpha[co] * acc, acc, acc >= 0.0);
  let o = co * ${OH * OW} + y * ${OW} + x;
  dst[o] = ${residual ? `v + res[o]` : `v`};
}`;
}

// ConvTranspose2d 4x4 stride2 pad1, no activation. Weight layout [CI, CO, 4, 4].
function wgslDeconv(CI, CO, IW, IH, OW, OH) {
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read> wgt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y); let co = i32(gid.z);
  if (x >= ${OW} || y >= ${OH}) { return; }
  var acc = bias[co];
  for (var ky = 0; ky < 4; ky++) {
    let ty = y + 1 - ky;
    if (ty < 0 || (ty & 1) != 0) { continue; }
    let iy = ty / 2;
    if (iy >= ${IH}) { continue; }
    for (var kx = 0; kx < 4; kx++) {
      let tx = x + 1 - kx;
      if (tx < 0 || (tx & 1) != 0) { continue; }
      let ix = tx / 2;
      if (ix >= ${IW}) { continue; }
      for (var ci = 0; ci < ${CI}; ci++) {
        acc += src[ci * ${IH * IW} + iy * ${IW} + ix]
             * wgt[ci * ${CO * 16} + co * 16 + ky * 4 + kx];
      }
    }
  }
  dst[co * ${OH * OW} + y * ${OW} + x] = acc;
}`;
}

// upsample tmp8 x8 (align_corners=false), flow*=8, warp both images, sigmoid blend, pack rgba
function wgslFlowOut(W, H) {
  const TW = W / 8, TH = H / 8;
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> tmp8: array<f32>;  // [5,${TH},${TW}]
@group(0) @binding(1) var<storage, read> imgs: array<f32>;  // [6,${H},${W}]
@group(0) @binding(2) var<storage, read_write> outp: array<u32>; // rgba

fn tap(c: i32, x: i32, y: i32) -> f32 {
  return tmp8[c * ${TH * TW} + clamp(y, 0, ${TH - 1}) * ${TW} + clamp(x, 0, ${TW - 1})];
}
fn up(c: i32, sx: f32, sy: f32) -> f32 {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  return mix(mix(tap(c, x0, y0), tap(c, x0 + 1, y0), fx),
             mix(tap(c, x0, y0 + 1), tap(c, x0 + 1, y0 + 1), fx), fy);
}
fn img(plane: i32, x: i32, y: i32) -> f32 {
  return imgs[plane * ${H * W} + clamp(y, 0, ${H - 1}) * ${W} + clamp(x, 0, ${W - 1})];
}
// grid_sample bilinear, border, align_corners=true == pixel-space bilinear with clamped taps
fn warp3(base: i32, sx: f32, sy: f32) -> vec3<f32> {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  var r: vec3<f32>;
  for (var c = 0; c < 3; c++) {
    let p = base + c;
    r[c] = mix(mix(img(p, x0, y0), img(p, x0 + 1, y0), fx),
               mix(img(p, x0, y0 + 1), img(p, x0 + 1, y0 + 1), fx), fy);
  }
  return r; // b,g,r
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${W} || y >= ${H}) { return; }
  // align_corners=false x8 upsample: src = (dst+0.5)/8 - 0.5; flow scaled by 8 (=scale*2*... baked)
  let sx = (f32(x) + 0.5) / 8.0 - 0.5;
  let sy = (f32(y) + 0.5) / 8.0 - 0.5;
  let fx0 = up(0, sx, sy) * 8.0;
  let fy0 = up(1, sx, sy) * 8.0;
  let fx1 = up(2, sx, sy) * 8.0;
  let fy1 = up(3, sx, sy) * 8.0;
  let m = 1.0 / (1.0 + exp(-up(4, sx, sy))); // sigmoid(mask)
  let w0 = warp3(0, f32(x) + fx0, f32(y) + fy0);
  let w1 = warp3(3, f32(x) + fx1, f32(y) + fy1);
  let bgr = w0 * m + w1 * (1.0 - m);
  // BGR -> RGB, *255 truncate (matches rife-core prepost), alpha 255
  let r = u32(clamp(bgr.z, 0.0, 1.0) * 255.0);
  let g = u32(clamp(bgr.y, 0.0, 1.0) * 255.0);
  let b = u32(clamp(bgr.x, 0.0, 1.0) * 255.0);
  outp[y * ${W} + x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}`;
}

export async function createRT(device, { w, h, weightsBin, weightsManifest }) {
  if (w % 16 || h % 16) throw new Error(`rt: dims must be /16 (got ${w}x${h})`);
  const QW = w / 4, QH = h / 4, W8 = w / 8, H8 = h / 8, W16 = w / 16, H16 = h / 16;

  const buf = (n, usage = GPUBufferUsage.STORAGE) =>
    device.createBuffer({ size: n * 4, usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });

  // weights
  const man = weightsManifest;
  const wbuf = {};
  for (const [name, m] of Object.entries(man)) {
    const n = m.shape.reduce((a, b) => a * b, 1);
    wbuf[name] = buf(n);
    device.queue.writeBuffer(wbuf[name], 0, weightsBin, m.offset * 4, n * 4);
  }

  // activations
  const tbuf = buf(1);
  const rgba0 = buf(w * h), rgba1 = buf(w * h);
  const imgs = buf(6 * w * h);
  const xq = buf(7 * QH * QW);
  const f8 = buf(120 * H8 * W8);
  const f16a = buf(240 * H16 * W16), f16b = buf(240 * H16 * W16), f16r = buf(240 * H16 * W16);
  const tmp8 = buf(5 * H8 * W8);
  const outp = buf(w * h);
  const staging = device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const mod = (code) => device.createShaderModule({ code });
  const pipe = (code, entry = 'main') => device.createComputePipeline({
    layout: 'auto', compute: { module: mod(code), entryPoint: entry } });

  const pPrepFull = pipe(wgslPrepFull(w, h));
  const pPrepQ = pipe(wgslPrepQuarter(w, h));
  const pConv0a = pipe(wgslConv(7, 120, QW, QH, W8, H8, 2, false));
  const pConv0b = pipe(wgslConv(120, 240, W8, H8, W16, H16, 2, false));
  const pConvB = pipe(wgslConv(240, 240, W16, H16, W16, H16, 1, false));
  const pConvBR = pipe(wgslConv(240, 240, W16, H16, W16, H16, 1, true));
  const pDeconv = pipe(wgslDeconv(240, 5, W16, H16, W8, H8));
  const pFlow = pipe(wgslFlowOut(w, h));

  const bg = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) });

  const bgPrepFull = bg(pPrepFull, [rgba0, rgba1, imgs]);
  const bgPrepQ = bg(pPrepQ, [rgba0, rgba1, xq, tbuf]);
  const bgConv0a = bg(pConv0a, [xq, wbuf['block0.conv0.0.0.weight'], wbuf['block0.conv0.0.0.bias'], wbuf['block0.conv0.0.1.weight'], f8]);
  const bgConv0b = bg(pConv0b, [f8, wbuf['block0.conv0.1.0.weight'], wbuf['block0.conv0.1.0.bias'], wbuf['block0.conv0.1.1.weight'], f16a]);
  // convblock ping-pong: a->b, b->a, ... 8th conv adds the residual (f16r = copy of f16a)
  const bgB = [];
  let src = f16a, dst = f16b;
  for (let i = 0; i < 8; i++) {
    const wn = `block0.convblock.${i}.0.weight`, bn = `block0.convblock.${i}.0.bias`, an = `block0.convblock.${i}.1.weight`;
    if (i < 7) {
      bgB.push({ p: pConvB, g: bg(pConvB, [src, wbuf[wn], wbuf[bn], wbuf[an], dst]) });
    } else {
      bgB.push({ p: pConvBR, g: bg(pConvBR, [src, wbuf[wn], wbuf[bn], wbuf[an], dst, f16r]) });
    }
    [src, dst] = [dst, src];
  }
  const f16out = src; // after 8 convs
  const bgDeconv = bg(pDeconv, [f16out, wbuf['block0.lastconv.weight'], wbuf['block0.lastconv.bias'], tmp8]);
  const bgFlow = bg(pFlow, [tmp8, imgs, outp]);

  const gx = (n) => Math.ceil(n / WG);

  async function run(rgbaA, rgbaB, t = 0.5) {
    device.queue.writeBuffer(tbuf, 0, new Float32Array([t]));
    device.queue.writeBuffer(rgba0, 0, rgbaA.buffer, rgbaA.byteOffset, w * h * 4);
    device.queue.writeBuffer(rgba1, 0, rgbaB.buffer, rgbaB.byteOffset, w * h * 4);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pPrepFull); pass.setBindGroup(0, bgPrepFull); pass.dispatchWorkgroups(gx(w), gx(h));
    pass.setPipeline(pPrepQ); pass.setBindGroup(0, bgPrepQ); pass.dispatchWorkgroups(gx(QW), gx(QH));
    pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), 120);
    pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), 240);
    pass.end();
    // residual copy AFTER conv0b (f16r = f16a snapshot)
    enc.copyBufferToBuffer(f16a, 0, f16r, 0, 240 * H16 * W16 * 4);
    const pass2 = enc.beginComputePass();
    for (const { p, g } of bgB) {
      pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(gx(W16), gx(H16), 240);
    }
    pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8), 5);
    pass2.setPipeline(pFlow); pass2.setBindGroup(0, bgFlow); pass2.dispatchWorkgroups(gx(w), gx(h));
    pass2.end();
    enc.copyBufferToBuffer(outp, 0, staging, 0, w * h * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint8Array(staging.getMappedRange().slice(0));
    staging.unmap();
    return out;
  }

  return { run, w, h };
}
