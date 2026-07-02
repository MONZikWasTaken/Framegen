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

function wgslPrepQuarter(W, H, f16) {
  const QW = W / 4, QH = H / 4;
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> rgba0: array<u32>;
@group(0) @binding(1) var<storage, read> rgba1: array<u32>;
@group(0) @binding(2) var<storage, read_write> xq: array<${T}>;    // [7,${QH},${QW}]
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
  xq[o] = ${T}(c0.z); xq[P + o] = ${T}(c0.y); xq[2 * P + o] = ${T}(c0.x);
  xq[3 * P + o] = ${T}(c1.z); xq[4 * P + o] = ${T}(c1.y); xq[5 * P + o] = ${T}(c1.x);
  xq[6 * P + o] = ${T}(tstep[0]); // timestep
}`;
}

// conv3x3 (stride 1/2) + bias + PReLU, optional residual add (post-activation).
// v3: COC output channels per thread (src reads amortized), weight slab staged through
// workgroup memory, and optional f16 storage for activations+weights (accumulation
// stays f32) — halves the traffic on the bandwidth-bound conv stack.
function wgslConv(CI, CO, IW, IH, OW, OH, stride, residual, f16) {
  const COC = CO % 4 === 0 ? 4 : 1;       // channels per thread
  const SLAB = Math.min(CI, 30);          // ci per staging round (fits 16KB wg memory)
  const slabFloats = COC * SLAB * 9;
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;      // [${CI},${IH},${IW}]
@group(0) @binding(1) var<storage, read> wgt: array<${T}>;      // [${CO},${CI},3,3]
@group(0) @binding(2) var<storage, read> bias: array<f32>;     // [${CO}]
@group(0) @binding(3) var<storage, read> alpha: array<f32>;    // [${CO}] prelu
@group(0) @binding(4) var<storage, read_write> dst: array<${T}>; // [${CO},${OH},${OW}]
${residual ? `@group(0) @binding(5) var<storage, read> res: array<${T}>;` : ``}

var<workgroup> wsh: array<${T}, ${slabFloats}>; // [COC, SLAB, 9] slab of weights
${stride === 1 ? `var<workgroup> tile: array<${T}, ${SLAB * 100}>; // [SLAB, 10, 10] input tiles (8x8 out + halo)` : ''}

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let x = i32(gid.x); let y = i32(gid.y);
  let lx = i32(lid.x); let ly = i32(lid.y);
  let wx0 = i32(wid.x) * ${WG}; let wy0 = i32(wid.y) * ${WG};
  let cb = i32(gid.z) * ${COC}; // first output channel of this thread's block
  let inb = x < ${OW} && y < ${OH};
  var acc: array<f32, ${COC}>;
  for (var c = 0; c < ${COC}; c++) { acc[c] = bias[cb + c]; }

  for (var s = 0; s < ${CI}; s += ${SLAB}) {
    let sl = min(${SLAB}, ${CI} - s);
    let n = ${COC} * sl * 9;
    workgroupBarrier();
    // cooperative load: weights for [cb..cb+COC) x [s..s+sl) x 9
    var idx = i32(li);
    while (idx < n) {
      let c = idx / (sl * 9);
      let r = idx % (sl * 9);
      wsh[idx] = wgt[(cb + c) * ${CI * 9} + (s + r / 9) * 9 + r % 9];
      idx += ${WG * WG};
    }
    workgroupBarrier();
${stride === 1 ? `
    // stride-1 path: stage 10x10 input tiles for the WHOLE ci-slab at once — barriers
    // per slab (16/conv) instead of per channel (480/conv), values reused 9x from shared
    var ti = i32(li);
    let tn = sl * 100;
    while (ti < tn) {
      let ci = ti / 100;
      let r = ti % 100;
      let ty = wy0 + r / 10 - 1;
      let tx = wx0 + r % 10 - 1;
      var v = ${T}(0.0);
      if (ty >= 0 && ty < ${IH} && tx >= 0 && tx < ${IW}) {
        v = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
      }
      tile[ti] = v;
      ti += ${WG * WG};
    }
    workgroupBarrier();
    if (inb) {
      for (var ci = 0; ci < sl; ci++) {
        let tb = ci * 100;
        for (var ky = 0; ky < 3; ky++) {
          for (var kx = 0; kx < 3; kx++) {
            let sv = f32(tile[tb + (ly + ky) * 10 + lx + kx]);
            let wb = ci * 9 + ky * 3 + kx;
            for (var c = 0; c < ${COC}; c++) {
              acc[c] += sv * f32(wsh[c * (sl * 9) + wb]);
            }
          }
        }
      }
    }
  }` : `
    if (inb) {
      for (var ci = 0; ci < sl; ci++) {
        let sbase = (s + ci) * ${IH * IW};
        for (var ky = 0; ky < 3; ky++) {
          let iy = y * ${stride} + ky - 1;
          if (iy < 0 || iy >= ${IH}) { continue; }
          for (var kx = 0; kx < 3; kx++) {
            let ix = x * ${stride} + kx - 1;
            if (ix < 0 || ix >= ${IW}) { continue; }
            let sv = f32(src[sbase + iy * ${IW} + ix]);
            let wb = ci * 9 + ky * 3 + kx;
            for (var c = 0; c < ${COC}; c++) {
              acc[c] += sv * f32(wsh[c * (sl * 9) + wb]);
            }
          }
        }
      }
    }
  }`}
  if (!inb) { return; }
  for (var c = 0; c < ${COC}; c++) {
    let co = cb + c;
    let v = select(alpha[co] * acc[c], acc[c], acc[c] >= 0.0);
    let o = co * ${OH * OW} + y * ${OW} + x;
    dst[o] = ${T}(${residual ? `v + f32(res[o])` : `v`});
  }
}`;
}

// register-blocked conv3x3 s1 (f16 storage): each thread computes a 2x2 pixel patch x
// 4 output channels (16 accumulators) — every shared read now feeds 4 FMAs instead of ~1.
// Workgroup = 8x8 threads = 16x16 output tile; input tiles 18x18 per ci staged per slab.
function wgslConvRB(CI, CO, IW, IH, OW, OH, residual) {
  const COC = 4, SLAB = 20;
  const slabW = COC * SLAB * 9;      // f16 weights in shared
  const slabT = SLAB * 324;          // 18x18 tiles
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;
${residual ? `@group(0) @binding(5) var<storage, read> res: array<f16>;` : ``}

var<workgroup> wsh: array<f16, ${slabW}>;
var<workgroup> tile: array<f16, ${slabT}>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lx = i32(lid.x); let ly = i32(lid.y);
  let ox0 = i32(wid.x) * 16; let oy0 = i32(wid.y) * 16;   // wg output origin
  let x0 = ox0 + lx * 2; let y0 = oy0 + ly * 2;           // this thread's 2x2 patch
  let cb = i32(wid.z) * ${COC};
  // 16 scalar accumulators (unrolled — arrays may spill out of registers in WGSL)
${Array.from({ length: COC }, (_, c) =>
  `  var a${c}0 = bias[cb + ${c}]; var a${c}1 = a${c}0; var a${c}2 = a${c}0; var a${c}3 = a${c}0;`).join('\n')}

  for (var s = 0; s < ${CI}; s += ${SLAB}) {
    let sl = min(${SLAB}, ${CI} - s);
    workgroupBarrier();
    var idx = i32(li);
    let wn = ${COC} * sl * 9;
    while (idx < wn) {
      let c = idx / (sl * 9);
      let r = idx % (sl * 9);
      wsh[idx] = wgt[(cb + c) * ${CI * 9} + (s + r / 9) * 9 + r % 9];
      idx += 64;
    }
    var ti = i32(li);
    let tn = sl * 324;
    while (ti < tn) {
      let ci = ti / 324;
      let r = ti % 324;
      let ty = oy0 + r / 18 - 1;
      let tx = ox0 + r % 18 - 1;
      var v = f16(0.0);
      if (ty >= 0 && ty < ${IH} && tx >= 0 && tx < ${IW}) {
        v = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
      }
      tile[ti] = v;
      ti += 64;
    }
    workgroupBarrier();
    for (var ci = 0; ci < sl; ci++) {
      let tb = ci * 324 + (ly * 2) * 18 + lx * 2; // top-left of this thread's 4x4 window
      for (var ky = 0; ky < 3; ky++) {
        let rb = tb + ky * 18;
        for (var kx = 0; kx < 3; kx++) {
          let t00 = f32(tile[rb + kx]);
          let t01 = f32(tile[rb + kx + 1]);
          let t10 = f32(tile[rb + kx + 18]);
          let t11 = f32(tile[rb + kx + 19]);
          let wb = ci * 9 + ky * 3 + kx;
${Array.from({ length: COC }, (_, c) => `          {
            let wv = f32(wsh[${c} * (sl * 9) + wb]);
            a${c}0 += t00 * wv; a${c}1 += t01 * wv; a${c}2 += t10 * wv; a${c}3 += t11 * wv;
          }`).join('\n')}
        }
      }
    }
  }
${Array.from({ length: COC }, (_, c) => `  {
    let co = cb + ${c};
    let al = alpha[co];
${[0, 1, 2, 3].map(p => `    {
      let x = x0 + ${p & 1};
      let y = y0 + ${p >> 1};
      if (x < ${OW} && y < ${OH}) {
        let a = a${c}${p};
        let v = select(al * a, a, a >= 0.0);
        let o = co * ${OH * OW} + y * ${OW} + x;
        dst[o] = f16(${residual ? `v + f32(res[o])` : `v`});
      }
    }`).join('\n')}
  }`).join('\n')}
}`;
}

// texture-input prep variants: the video frame lives in a GPU texture (uploaded via
// copyExternalImageToTexture) and never touches the CPU; the sampler also does the
// display->model resize for free. Sampling at texel centers == exact texel values.
function wgslPrepFullTex(W, H) {
  return /* wgsl */`
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> imgs: array<f32>;  // [6,${H},${W}] BGR

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${W} || y >= ${H}) { return; }
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
  let c0 = textureSampleLevel(tex0, samp, uv, 0.0).rgb;
  let c1 = textureSampleLevel(tex1, samp, uv, 0.0).rgb;
  let o = y * ${W} + x;
  let P = ${H * W};
  imgs[o] = c0.b; imgs[P + o] = c0.g; imgs[2 * P + o] = c0.r;
  imgs[3 * P + o] = c1.b; imgs[4 * P + o] = c1.g; imgs[5 * P + o] = c1.r;
}`;
}

function wgslPrepQuarterTex(W, H, f16) {
  const QW = W / 4, QH = H / 4;
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> xq: array<${T}>;   // [7,${QH},${QW}]
@group(0) @binding(4) var<storage, read> tstep: array<f32>;

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${QW} || y >= ${QH}) { return; }
  // quarter of the MODEL grid; the sampler maps through whatever the texture size is
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(${QW}.0, ${QH}.0);
  let c0 = textureSampleLevel(tex0, samp, uv, 0.0).rgb;
  let c1 = textureSampleLevel(tex1, samp, uv, 0.0).rgb;
  let o = y * ${QW} + x;
  let P = ${QH * QW};
  xq[o] = ${T}(c0.b); xq[P + o] = ${T}(c0.g); xq[2 * P + o] = ${T}(c0.r);
  xq[3 * P + o] = ${T}(c1.b); xq[4 * P + o] = ${T}(c1.g); xq[5 * P + o] = ${T}(c1.r);
  xq[6 * P + o] = ${T}(tstep[0]);
}`;
}

// one-shot f32 -> f16 conversion (weights at init)
const WGSL_TO_F16 = /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&src)) { dst[i] = f16(src[i]); }
}`;

// ConvTranspose2d 4x4 stride2 pad1, no activation. Weight layout [CI, CO, 4, 4].
function wgslDeconv(CI, CO, IW, IH, OW, OH, f16src) {
  const T = f16src ? 'f16' : 'f32';
  return /* wgsl */`
${f16src ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;
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
        acc += f32(src[ci * ${IH * IW} + iy * ${IW} + ix])
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

export async function createRT(device, { w, h, weightsBin, weightsManifest, textureInput = false }) {
  if (w % 16 || h % 16) throw new Error(`rt: dims must be /16 (got ${w}x${h})`);
  const QW = w / 4, QH = h / 4, W8 = w / 8, H8 = h / 8, W16 = w / 16, H16 = h / 16;
  const useF16 = device.features.has('shader-f16');
  // channel widths come from the weights themselves (supports slim students)
  const C1 = weightsManifest['block0.conv0.0.0.weight'].shape[0]; // conv0a out (120 full / 60 slim)
  const C2 = weightsManifest['block0.conv0.1.0.weight'].shape[0]; // main width (240 full / 120 slim)
  if (C2 % 4) throw new Error('rt: main width must be /4');

  const bufBytes = (bytes, usage = GPUBufferUsage.STORAGE) => device.createBuffer({
    size: Math.ceil(bytes / 4) * 4,
    usage: usage | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const buf = (n) => bufBytes(n * 4);
  const abuf = (n) => bufBytes(n * (useF16 ? 2 : 4)); // activation dtype

  const mod = (code) => device.createShaderModule({ code });
  const pipe = (code, entry = 'main') => device.createComputePipeline({
    layout: 'auto', compute: { module: mod(code), entryPoint: entry } });
  const bg = (p, entries) => device.createBindGroup({
    layout: p.getBindGroupLayout(0),
    entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) });

  // weights (f32 upload; conv weights get f16 copies when supported)
  const man = weightsManifest;
  const wbuf = {};
  for (const [name, m] of Object.entries(man)) {
    const n = m.shape.reduce((a, b) => a * b, 1);
    wbuf[name] = buf(n);
    device.queue.writeBuffer(wbuf[name], 0, weightsBin, m.offset * 4, n * 4);
  }
  const convW = (name) => {
    if (!useF16) return wbuf[name];
    const n = man[name].shape.reduce((a, b) => a * b, 1);
    const half = bufBytes(n * 2);
    const p = pipe(WGSL_TO_F16);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, bg(p, [wbuf[name], half]));
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();
    device.queue.submit([enc.finish()]);
    return half;
  };

  // activations (f16 when supported), fixed f32 elsewhere
  const tbuf = buf(1);
  const rgba0 = textureInput ? null : buf(w * h);
  const rgba1 = textureInput ? null : buf(w * h);
  const imgs = buf(6 * w * h);
  const xq = abuf(7 * QH * QW);
  const f8 = abuf(C1 * H8 * W8);
  const actBytes = C2 * H16 * W16 * (useF16 ? 2 : 4);
  const f16a = bufBytes(actBytes), f16b = bufBytes(actBytes), f16r = bufBytes(actBytes);
  const tmp8 = buf(5 * H8 * W8);
  const outp = buf(w * h);
  const staging = device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  // slots for batched multi-t runs (factor N: upload once, N-1 mids in ONE submit)
  const MAXT = 5;
  const tbufs = [], stagings = [];
  for (let i = 0; i < MAXT; i++) {
    tbufs.push(buf(1));
    stagings.push(device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  }

  const sampler = textureInput
    ? device.createSampler({ magFilter: 'linear', minFilter: 'linear',
                             addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' })
    : null;
  const pPrepFull = pipe(textureInput ? wgslPrepFullTex(w, h) : wgslPrepFull(w, h));
  const pPrepQ = pipe(textureInput ? wgslPrepQuarterTex(w, h, useF16) : wgslPrepQuarter(w, h, useF16));
  const pConv0a = pipe(wgslConv(7, C1, QW, QH, W8, H8, 2, false, useF16));
  const pConv0b = pipe(wgslConv(C1, C2, W8, H8, W16, H16, 2, false, useF16));
  const pConvB = useF16
    ? pipe(wgslConvRB(C2, C2, W16, H16, W16, H16, false))
    : pipe(wgslConv(C2, C2, W16, H16, W16, H16, 1, false, false));
  const pConvBR = useF16
    ? pipe(wgslConvRB(C2, C2, W16, H16, W16, H16, true))
    : pipe(wgslConv(C2, C2, W16, H16, W16, H16, 1, true, false));
  const pDeconv = pipe(wgslDeconv(C2, 5, W16, H16, W8, H8, useF16));
  const pFlow = pipe(wgslFlowOut(w, h));

  // buffer-input prep bind groups (unused in texture mode)
  const bgPrepFull = textureInput ? null : bg(pPrepFull, [rgba0, rgba1, imgs]);
  const bgPrepQ = textureInput ? null : bg(pPrepQ, [rgba0, rgba1, xq, tbuf]);
  const bgPrepQt = textureInput ? null : tbufs.map(tb => bg(pPrepQ, [rgba0, rgba1, xq, tb]));
  // texture-mode prep bind groups are built per texture pair and cached (ping-pong -> few combos)
  const texBgCache = new Map();
  function texPrepBgs(texA, texB) {
    const key = texA.label + '|' + texB.label;
    if (!texBgCache.has(key)) {
      const va = texA.createView(), vb = texB.createView();
      texBgCache.set(key, {
        full: device.createBindGroup({ layout: pPrepFull.getBindGroupLayout(0), entries: [
          { binding: 0, resource: va }, { binding: 1, resource: vb },
          { binding: 2, resource: sampler }, { binding: 3, resource: { buffer: imgs } }] }),
        q: tbufs.map(tb => device.createBindGroup({ layout: pPrepQ.getBindGroupLayout(0), entries: [
          { binding: 0, resource: va }, { binding: 1, resource: vb },
          { binding: 2, resource: sampler }, { binding: 3, resource: { buffer: xq } },
          { binding: 4, resource: { buffer: tb } }] })),
      });
      if (texBgCache.size > 12) texBgCache.clear(); // texture set changed wholesale
    }
    return texBgCache.get(key);
  }
  const bgConv0a = bg(pConv0a, [xq, convW('block0.conv0.0.0.weight'), wbuf['block0.conv0.0.0.bias'], wbuf['block0.conv0.0.1.weight'], f8]);
  const bgConv0b = bg(pConv0b, [f8, convW('block0.conv0.1.0.weight'), wbuf['block0.conv0.1.0.bias'], wbuf['block0.conv0.1.1.weight'], f16a]);
  // convblock ping-pong: a->b, b->a, ... 8th conv adds the residual (f16r = copy of f16a)
  const bgB = [];
  let src = f16a, dst = f16b;
  for (let i = 0; i < 8; i++) {
    const wn = `block0.convblock.${i}.0.weight`, bn = `block0.convblock.${i}.0.bias`, an = `block0.convblock.${i}.1.weight`;
    if (i < 7) {
      bgB.push({ p: pConvB, g: bg(pConvB, [src, convW(wn), wbuf[bn], wbuf[an], dst]) });
    } else {
      bgB.push({ p: pConvBR, g: bg(pConvBR, [src, convW(wn), wbuf[bn], wbuf[an], dst, f16r]) });
    }
    [src, dst] = [dst, src];
  }
  const f16out = src; // after 8 convs
  const bgDeconv = bg(pDeconv, [f16out, wbuf['block0.lastconv.weight'], wbuf['block0.lastconv.bias'], tmp8]);
  const bgFlow = bg(pFlow, [tmp8, imgs, outp]);

  const gx = (n) => Math.ceil(n / WG);
  // register-blocked convblock kernel covers 16x16 output per workgroup
  const cbX = useF16 ? Math.ceil(W16 / 16) : gx(W16);
  const cbY = useF16 ? Math.ceil(H16 / 16) : gx(H16);

  // per-stage GPU times via timestamp queries (needs 'timestamp-query' on the device)
  async function profile(rgbaA, rgbaB) {
    if (!device.features.has('timestamp-query')) return 'no timestamp-query feature';
    const stages = [
      ['prepFull', pPrepFull, bgPrepFull, [gx(w), gx(h), 1]],
      ['prepQ', pPrepQ, bgPrepQ, [gx(QW), gx(QH), 1]],
      ['conv0a', pConv0a, bgConv0a, [gx(W8), gx(H8), C1 / 4]],
      ['conv0b', pConv0b, bgConv0b, [gx(W16), gx(H16), C2 / 4]],
      ...bgB.map(({ p, g }, i) => [`convB${i}`, p, g, [cbX, cbY, C2 / 4]]),
      ['deconv', pDeconv, bgDeconv, [gx(W8), gx(H8), 5]],
      ['flow', pFlow, bgFlow, [gx(w), gx(h), 1]],
    ];
    const qs = device.createQuerySet({ type: 'timestamp', count: stages.length * 2 });
    const qbuf = device.createBuffer({ size: stages.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const qread = device.createBuffer({ size: stages.length * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    device.queue.writeBuffer(tbuf, 0, new Float32Array([0.5]));
    device.queue.writeBuffer(rgba0, 0, rgbaA.buffer, rgbaA.byteOffset, w * h * 4);
    device.queue.writeBuffer(rgba1, 0, rgbaB.buffer, rgbaB.byteOffset, w * h * 4);
    const enc = device.createCommandEncoder();
    stages.forEach(([name, p, g, d], i) => {
      if (name === 'convB0') enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes);
      const pass = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } });
      pass.setPipeline(p); pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(d[0], d[1], d[2]);
      pass.end();
    });
    enc.resolveQuerySet(qs, 0, stages.length * 2, qbuf, 0);
    enc.copyBufferToBuffer(qbuf, 0, qread, 0, stages.length * 16);
    device.queue.submit([enc.finish()]);
    await qread.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(qread.getMappedRange().slice(0));
    qread.unmap();
    return stages.map(([name], i) =>
      `${name}: ${(Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6).toFixed(2)}ms`).join(' · ');
  }

  async function run(rgbaA, rgbaB, t = 0.5) {
    device.queue.writeBuffer(tbuf, 0, new Float32Array([t]));
    device.queue.writeBuffer(rgba0, 0, rgbaA.buffer, rgbaA.byteOffset, w * h * 4);
    device.queue.writeBuffer(rgba1, 0, rgbaB.buffer, rgbaB.byteOffset, w * h * 4);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pPrepFull); pass.setBindGroup(0, bgPrepFull); pass.dispatchWorkgroups(gx(w), gx(h));
    pass.setPipeline(pPrepQ); pass.setBindGroup(0, bgPrepQ); pass.dispatchWorkgroups(gx(QW), gx(QH));
    pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), C1 / 4);
    pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), C2 / 4);
    pass.end();
    // residual copy AFTER conv0b (f16r = f16a snapshot)
    enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes);
    const pass2 = enc.beginComputePass();
    for (const { p, g } of bgB) {
      pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, C2 / 4);
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

  // batched: upload/bind the pair once, produce mids for every t in ONE submit.
  // Buffer mode: a/b are RGBA arrays. Texture mode: a/b are GPUTextures (zero CPU pixels).
  async function runMulti(a, b, ts) {
    if (ts.length > MAXT) throw new Error('too many timesteps');
    for (let i = 0; i < ts.length; i++) {
      device.queue.writeBuffer(tbufs[i], 0, new Float32Array([ts[i]]));
    }
    let tbg = null;
    if (textureInput) {
      tbg = texPrepBgs(a, b);
    } else {
      device.queue.writeBuffer(rgba0, 0, a.buffer, a.byteOffset, w * h * 4);
      device.queue.writeBuffer(rgba1, 0, b.buffer, b.byteOffset, w * h * 4);
    }
    const enc = device.createCommandEncoder();
    {
      const pass = enc.beginComputePass();
      pass.setPipeline(pPrepFull); pass.setBindGroup(0, tbg ? tbg.full : bgPrepFull); pass.dispatchWorkgroups(gx(w), gx(h));
      pass.end();
    }
    for (let i = 0; i < ts.length; i++) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pPrepQ); pass.setBindGroup(0, tbg ? tbg.q[i] : bgPrepQt[i]); pass.dispatchWorkgroups(gx(QW), gx(QH));
      pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), C1 / 4);
      pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), C2 / 4);
      pass.end();
      enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes);
      const pass2 = enc.beginComputePass();
      for (const { p, g } of bgB) {
        pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, C2 / 4);
      }
      pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8), 5);
      pass2.setPipeline(pFlow); pass2.setBindGroup(0, bgFlow); pass2.dispatchWorkgroups(gx(w), gx(h));
      pass2.end();
      enc.copyBufferToBuffer(outp, 0, stagings[i], 0, w * h * 4);
    }
    device.queue.submit([enc.finish()]);
    const outs = [];
    for (let i = 0; i < ts.length; i++) {
      await stagings[i].mapAsync(GPUMapMode.READ);
      outs.push(new Uint8Array(stagings[i].getMappedRange().slice(0)));
      stagings[i].unmap();
    }
    return outs;
  }

  return { run, runMulti, profile, w, h };
}
