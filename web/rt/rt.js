// Framegen custom WebGPU runtime - hand-rolled forward of the 1-block student
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

// identity stamps for texture-keyed caches. MODULE scope, not per-createRT:
// runtimes get rebuilt (res/model/tune switches) while the caller's textures
// survive with their stamps - a per-instance counter restarting at 1 would
// re-issue ids already stamped on live textures, and the next pool realloc
// would hand destroyed-texture bind groups out of the cache again.
let texBgSeq = 0;
const texBgId = (t) => t.__rtBgId || (t.__rtBgId = ++texBgSeq);

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
// stays f32) - halves the traffic on the bandwidth-bound conv stack.
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
    // stride-1 path: stage 10x10 input tiles for the WHOLE ci-slab at once - barriers
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
// 4 output channels (16 accumulators) - every shared read now feeds 4 FMAs instead of ~1.
// Workgroup = 8x8 threads = 16x16 output tile; input tiles 18x18 per ci staged per slab.
export function wgslConvRB(CI, CO, IW, IH, OW, OH, residual, tune) {
  // tune: {coc, slab} - shared memory must fit slab*324*2 + coc*slab*9*2 <= 16384
  const COC = (tune && tune.coc) || 4, SLAB = (tune && tune.slab) || 20;
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
  // 16 scalar accumulators (unrolled - arrays may spill out of registers in WGSL)
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

// exp/subgroups: like wgslConvRB, but weights are NOT staged through shared
// memory - every lane computes the same weight index, so the first lane loads it
// from global (L2) and subgroupBroadcastFirst hands it to the wave. Saves the
// cooperative staging loop and shrinks the barrier to the input tile only.
export function wgslConvRBSg(CI, CO, IW, IH, OW, OH, residual, tune) {
  const COC = (tune && tune.coc) || 4, SLAB = (tune && tune.slab) || 20;
  const slabT = SLAB * 324;
  return /* wgsl */`
enable f16;
enable subgroups;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;
${residual ? `@group(0) @binding(5) var<storage, read> res: array<f16>;` : ``}

var<workgroup> tile: array<f16, ${slabT}>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lx = i32(lid.x); let ly = i32(lid.y);
  let ox0 = i32(wid.x) * 16; let oy0 = i32(wid.y) * 16;
  let x0 = ox0 + lx * 2; let y0 = oy0 + ly * 2;
  let cb = i32(wid.z) * ${COC};
${Array.from({ length: COC }, (_, c) =>
  `  var a${c}0 = bias[cb + ${c}]; var a${c}1 = a${c}0; var a${c}2 = a${c}0; var a${c}3 = a${c}0;`).join('\n')}

  for (var s = 0; s < ${CI}; s += ${SLAB}) {
    let sl = min(${SLAB}, ${CI} - s);
    workgroupBarrier();
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
      let tb = ci * 324 + (ly * 2) * 18 + lx * 2;
      let wrow = (s + ci) * 9;
      for (var ky = 0; ky < 3; ky++) {
        let rb = tb + ky * 18;
        for (var kx = 0; kx < 3; kx++) {
          let t00 = f32(tile[rb + kx]);
          let t01 = f32(tile[rb + kx + 1]);
          let t10 = f32(tile[rb + kx + 18]);
          let t11 = f32(tile[rb + kx + 19]);
          let wk = wrow + ky * 3 + kx;
${Array.from({ length: COC }, (_, c) => `          {
            let wv = subgroupBroadcastFirst(f32(wgt[(cb + ${c}) * ${CI * 9} + wk]));
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

// t-factored prep: 6 channels, NO timestep - the trunk is t-free, t enters via FiLM
function wgslPrepQuarterTex6(W, H, f16) {
  const QW = W / 4, QH = H / 4;
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> xq: array<${T}>;   // [6,${QH},${QW}]

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${QW} || y >= ${QH}) { return; }
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(${QW}.0, ${QH}.0);
  let c0 = textureSampleLevel(tex0, samp, uv, 0.0).rgb;
  let c1 = textureSampleLevel(tex1, samp, uv, 0.0).rgb;
  let o = y * ${QW} + x;
  let P = ${QH * QW};
  xq[o] = ${T}(c0.b); xq[P + o] = ${T}(c0.g); xq[2 * P + o] = ${T}(c0.r);
  xq[3 * P + o] = ${T}(c1.b); xq[4 * P + o] = ${T}(c1.g); xq[5 * P + o] = ${T}(c1.r);
}`;
}

// FiLM conditioning: x' = x * (1 + scale[c]) + shift[c]; params are the tiny
// t-MLP's output, computed on the CPU per timestep (2*C floats)
function wgslFilm(C, N, f16) {
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;   // [C,N] trunk features
@group(0) @binding(1) var<storage, read> prm: array<f32>;    // [2C] scale, shift
@group(0) @binding(2) var<storage, read_write> dst: array<${T}>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = i32(gid.x);
  if (i >= ${C * N}) { return; }
  let c = i / ${N};
  dst[i] = ${T}(f32(src[i]) * (1.0 + prm[c]) + prm[${C} + c]);
}`;
}

// flowout variant writing straight into a storage texture (GPU-resident presentation:
// the mid never leaves the GPU). rgba8unorm store rounds instead of truncating - ±1 LSB
// vs the buffer path, invisible; the correctness harness keeps using the buffer path.
function wgslFlowOutTex(W, H, staticGuard = false, withRes = false) {
  const TW = W / 8, TH = H / 8, RW = W / 4, RH = H / 4;
  // tfact2: the refine residual (quarter res) is folded straight into this pass -
  // bilinear x4 upsample (align_corners=False grid), added BEFORE the clamp
  const RES_DECL = withRes ? /* wgsl */`
@group(0) @binding(3) var<storage, read> res: array<f32>; // [3,${RH},${RW}]
fn rtap(c: i32, x: i32, y: i32) -> f32 {
  return res[c * ${RH * RW} + clamp(y, 0, ${RH - 1}) * ${RW} + clamp(x, 0, ${RW - 1})];
}
fn rup(c: i32, sx: f32, sy: f32) -> f32 {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  return mix(mix(rtap(c, x0, y0), rtap(c, x0 + 1, y0), fx),
             mix(rtap(c, x0, y0 + 1), rtap(c, x0 + 1, y0 + 1), fx), fy);
}` : '';
  const RES_ADD = withRes ? /* wgsl */`
  let rx = (f32(x) + 0.5) / 4.0 - 0.5;
  let ry = (f32(y) + 0.5) / 4.0 - 0.5;
  bgr = bgr + vec3<f32>(rup(0, rx, ry), rup(1, rx, ry), rup(2, rx, ry));` : '';
  // static-region protection (SVP-style): where A and B are locally identical
  // (subtitles, logos, UI, frozen shots-in-motion) the warp can still DRAG other
  // content there - blend back to the untouched source instead. Soft ramp so
  // moving-edge pixels transition smoothly.
  const GUARD = staticGuard ? /* wgsl */`
  var d = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let xx = x + dx; let yy = y + dy;
      d += max(abs(img(0, xx, yy) - img(3, xx, yy)),
           max(abs(img(1, xx, yy) - img(4, xx, yy)),
               abs(img(2, xx, yy) - img(5, xx, yy))));
    }
  }
  d *= (1.0 / 9.0);
  let wStatic = 1.0 - smoothstep(0.03, 0.09, d);
  if (wStatic > 0.001) {
    let stat = vec3<f32>(
      (img(0, x, y) + img(3, x, y)) * 0.5,
      (img(1, x, y) + img(4, x, y)) * 0.5,
      (img(2, x, y) + img(5, x, y)) * 0.5);
    bgr = mix(bgr, stat, wStatic);
  }` : '';
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> tmp8: array<f32>;  // [5,${TH},${TW}]
@group(0) @binding(1) var<storage, read> imgs: array<f32>;  // [6,${H},${W}]
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
${RES_DECL}
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
  let sx = (f32(x) + 0.5) / 8.0 - 0.5;
  let sy = (f32(y) + 0.5) / 8.0 - 0.5;
  let fx0 = up(0, sx, sy) * 8.0;
  let fy0 = up(1, sx, sy) * 8.0;
  let fx1 = up(2, sx, sy) * 8.0;
  let fy1 = up(3, sx, sy) * 8.0;
  let m = 1.0 / (1.0 + exp(-up(4, sx, sy)));
  let w0 = warp3(0, f32(x) + fx0, f32(y) + fy0);
  let w1 = warp3(3, f32(x) + fx1, f32(y) + fy1);
  var bgr = w0 * m + w1 * (1.0 - m);
${RES_ADD}
  bgr = clamp(bgr, vec3<f32>(0.0), vec3<f32>(1.0));
${GUARD}
  textureStore(outTex, vec2<i32>(x, y), vec4<f32>(bgr.z, bgr.y, bgr.x, 1.0));
}`;
}

// per-pair static-difference plane (texture in/out mode): the guard's d-term
// depends only on the frame PAIR, never on t - computing it once here removes
// 54 full-res buffer reads/px from EVERY flowout dispatch (up to 19 mids in hz
// mode share one pair). Same sampler/uv as the old prepFull, so d is bit-equal.
function wgslDiff(W, H) {
  return /* wgsl */`
@group(0) @binding(0) var tex0: texture_2d<f32>;
@group(0) @binding(1) var tex1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>; // [${H},${W}] max-channel |A-B|

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${W} || y >= ${H}) { return; }
  let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
  let d = abs(textureSampleLevel(tex0, samp, uv, 0.0).rgb
            - textureSampleLevel(tex1, samp, uv, 0.0).rgb);
  dst[y * ${W} + x] = max(d.x, max(d.y, d.z));
}`;
}

// direct-warp flowout (texture in/out mode): the warp samples the SOURCE textures
// through the hardware bilinear unit instead of a model-res f32 copy - one texture
// sample replaces 12 buffer reads, the 6-plane imgs buffer and its per-pair prep
// pass disappear, and the source is resampled ONCE (warp) instead of twice
// (copy then warp) - sharper mids for less bandwidth. Filter precision is the
// sampler's (~8-bit subtexel) - same +-1 LSB class as the rgba8 store above.
function wgslFlowOutTexDirect(W, H, staticGuard = false, withRes = false) {
  const TW = W / 8, TH = H / 8, RW = W / 4, RH = H / 4;
  const RES_DECL = withRes ? /* wgsl */`
@group(0) @binding(2) var<storage, read> res: array<f32>; // [3,${RH},${RW}]
fn rtap(c: i32, x: i32, y: i32) -> f32 {
  return res[c * ${RH * RW} + clamp(y, 0, ${RH - 1}) * ${RW} + clamp(x, 0, ${RW - 1})];
}
fn rup(c: i32, sx: f32, sy: f32) -> f32 {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  return mix(mix(rtap(c, x0, y0), rtap(c, x0 + 1, y0), fx),
             mix(rtap(c, x0, y0 + 1), rtap(c, x0 + 1, y0 + 1), fx), fy);
}` : '';
  const RES_ADD = withRes ? /* wgsl */`
  let rx = (f32(x) + 0.5) / 4.0 - 0.5;
  let ry = (f32(y) + 0.5) / 4.0 - 0.5;
  bgr = bgr + vec3<f32>(rup(0, rx, ry), rup(1, rx, ry), rup(2, rx, ry));` : '';
  // static-region protection: d comes from the per-pair sdiff plane (9 taps of one
  // plane instead of 9x6 image reads); the untouched-source blend samples the pair
  // at the pixel center - identical values to the old imgs-based path
  const GUARD = staticGuard ? /* wgsl */`
  var d = 0.0;
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      d += dtap(x + dx, y + dy);
    }
  }
  d *= (1.0 / 9.0);
  let wStatic = 1.0 - smoothstep(0.03, 0.09, d);
  if (wStatic > 0.001) {
    let stat = (warpT(tex0, f32(x), f32(y)) + warpT(tex1, f32(x), f32(y))) * 0.5;
    bgr = mix(bgr, stat, wStatic);
  }` : '';
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> tmp8: array<f32>;  // [5,${TH},${TW}]
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;
${RES_DECL}
${staticGuard ? /* wgsl */`@group(0) @binding(3) var<storage, read> sdiff: array<f32>; // [${H},${W}]
fn dtap(x: i32, y: i32) -> f32 {
  return sdiff[clamp(y, 0, ${H - 1}) * ${W} + clamp(x, 0, ${W - 1})];
}` : ''}
@group(1) @binding(0) var tex0: texture_2d<f32>;
@group(1) @binding(1) var tex1: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;
fn tap(c: i32, x: i32, y: i32) -> f32 {
  return tmp8[c * ${TH * TW} + clamp(y, 0, ${TH - 1}) * ${TW} + clamp(x, 0, ${TW - 1})];
}
fn up(c: i32, sx: f32, sy: f32) -> f32 {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  return mix(mix(tap(c, x0, y0), tap(c, x0 + 1, y0), fx),
             mix(tap(c, x0, y0 + 1), tap(c, x0 + 1, y0 + 1), fx), fy);
}
// grid_sample bilinear/border via the sampler: clamp-to-edge + hw filtering
fn warpT(t: texture_2d<f32>, sx: f32, sy: f32) -> vec3<f32> {
  let uv = (vec2<f32>(sx, sy) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
  return textureSampleLevel(t, samp, uv, 0.0).bgr; // b,g,r like the buffer path
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${W} || y >= ${H}) { return; }
  let sx = (f32(x) + 0.5) / 8.0 - 0.5;
  let sy = (f32(y) + 0.5) / 8.0 - 0.5;
  let fx0 = up(0, sx, sy) * 8.0;
  let fy0 = up(1, sx, sy) * 8.0;
  let fx1 = up(2, sx, sy) * 8.0;
  let fy1 = up(3, sx, sy) * 8.0;
  let m = 1.0 / (1.0 + exp(-up(4, sx, sy)));
  let w0 = warpT(tex0, f32(x) + fx0, f32(y) + fy0);
  let w1 = warpT(tex1, f32(x) + fx1, f32(y) + fy1);
  var bgr = w0 * m + w1 * (1.0 - m);
${RES_ADD}
  bgr = clamp(bgr, vec3<f32>(0.0), vec3<f32>(1.0));
${GUARD}
  textureStore(outTex, vec2<i32>(x, y), vec4<f32>(bgr.z, bgr.y, bgr.x, 1.0));
}`;
}

// ---- refine head (tfact2): occlusion repair at QUARTER resolution ----
// Gathers [warped0(3), warped1(3), mask(1), flow*(0.25/20)(4)] = 11ch at H/4.
// F.interpolate(x, 0.25, bilinear, align_corners=False) samples the source at
// 4x+1.5 - i.e. the mean of the CENTER 2x2 of each 4x4 block; we warp those
// four full-res positions and average, matching the trainer exactly.
function wgslRefinePrep(W, H, f16) {
  const TW = W / 8, TH = H / 8, HW = W / 4, HH = H / 4;
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> tmp8: array<f32>;  // [5,${TH},${TW}]
@group(0) @binding(1) var<storage, read_write> rin: array<${T}>; // [11,${HH},${HW}]
@group(1) @binding(0) var tex0: texture_2d<f32>;
@group(1) @binding(1) var tex1: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

fn tap(c: i32, x: i32, y: i32) -> f32 {
  return tmp8[c * ${TH * TW} + clamp(y, 0, ${TH - 1}) * ${TW} + clamp(x, 0, ${TW - 1})];
}
fn up(c: i32, sx: f32, sy: f32) -> f32 {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  return mix(mix(tap(c, x0, y0), tap(c, x0 + 1, y0), fx),
             mix(tap(c, x0, y0 + 1), tap(c, x0 + 1, y0 + 1), fx), fy);
}
// direct warp: sample the source texture (see wgslFlowOutTexDirect)
fn warpT(t: texture_2d<f32>, sx: f32, sy: f32) -> vec3<f32> {
  let uv = (vec2<f32>(sx, sy) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
  return textureSampleLevel(t, samp, uv, 0.0).bgr; // b,g,r
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let hx = i32(gid.x); let hy = i32(gid.y);
  if (hx >= ${HW} || hy >= ${HH}) { return; }
  var w0 = vec3<f32>(0.0); var w1 = vec3<f32>(0.0);
  var mk = 0.0; var fl = vec4<f32>(0.0);
  for (var sy = 1; sy <= 2; sy++) {
    for (var sxp = 1; sxp <= 2; sxp++) {
      let X = hx * 4 + sxp; let Y = hy * 4 + sy; // center 2x2 of the 4x4 block
      let gx8 = (f32(X) + 0.5) / 8.0 - 0.5;
      let gy8 = (f32(Y) + 0.5) / 8.0 - 0.5;
      let fx0 = up(0, gx8, gy8) * 8.0; let fy0 = up(1, gx8, gy8) * 8.0;
      let fx1 = up(2, gx8, gy8) * 8.0; let fy1 = up(3, gx8, gy8) * 8.0;
      mk += 1.0 / (1.0 + exp(-up(4, gx8, gy8)));
      w0 += warpT(tex0, f32(X) + fx0, f32(Y) + fy0);
      w1 += warpT(tex1, f32(X) + fx1, f32(Y) + fy1);
      fl += vec4<f32>(fx0, fy0, fx1, fy1);
    }
  }
  let P = ${HH * HW};
  let o = hy * ${HW} + hx;
  let q = 0.25;
  let fn_ = 0.25 * (0.25 / 20.0); // mean * (0.25/FLOW_NORM)
  rin[o] = ${T}(w0.x * q);           rin[P + o] = ${T}(w0.y * q);     rin[2 * P + o] = ${T}(w0.z * q);
  rin[3 * P + o] = ${T}(w1.x * q);   rin[4 * P + o] = ${T}(w1.y * q); rin[5 * P + o] = ${T}(w1.z * q);
  rin[6 * P + o] = ${T}(mk * q);
  rin[7 * P + o] = ${T}(fl.x * fn_); rin[8 * P + o] = ${T}(fl.y * fn_);
  rin[9 * P + o] = ${T}(fl.z * fn_); rin[10 * P + o] = ${T}(fl.w * fn_);
}`;
}

// final refine conv (C->3) + sigmoid*2-1 residual, half res, f32 out
function wgslRefineOut(C, HW, HH, f16) {
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;  // [${C},${HH},${HW}]
@group(0) @binding(1) var<storage, read> wgt: array<f32>;   // [3,${C},3,3]
@group(0) @binding(2) var<storage, read> bias: array<f32>;  // [3]
@group(0) @binding(3) var<storage, read_write> dst: array<f32>; // [3,${HH},${HW}]

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${HW} || y >= ${HH}) { return; }
  for (var co = 0; co < 3; co++) {
    var acc = bias[co];
    for (var ci = 0; ci < ${C}; ci++) {
      let sb = ci * ${HH * HW};
      let wb = (co * ${C} + ci) * 9;
      for (var ky = 0; ky < 3; ky++) {
        let sy = clamp(y + ky - 1, 0, ${HH - 1});
        for (var kx = 0; kx < 3; kx++) {
          let sx = clamp(x + kx - 1, 0, ${HW - 1});
          acc += f32(src[sb + sy * ${HW} + sx]) * wgt[wb + ky * 3 + kx];
        }
      }
    }
    dst[co * ${HH * HW} + y * ${HW} + x] = 2.0 / (1.0 + exp(-acc)) - 1.0;
  }
}`;
}

// one-shot f32 -> f16 conversion (weights at init)
export const WGSL_TO_F16 = /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&src)) { dst[i] = f16(src[i]); }
}`;

// ConvTranspose2d 4x4 stride2 pad1, no activation. Weight layout [CI, CO, 4, 4].
// f16 mode covers src AND weights (accumulation stays f32) - the per-thread
// 4*CI weight reads were the last f32 stream on the per-mid path.
function wgslDeconv(CI, CO, IW, IH, OW, OH, f16src) {
  const T = f16src ? 'f16' : 'f32';
  return /* wgsl */`
${f16src ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;
@group(0) @binding(1) var<storage, read> wgt: array<${T}>;
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
             * f32(wgt[ci * ${CO * 16} + co * 16 + ky * 4 + kx]);
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

export async function createRT(device, { w, h, weightsBin, weightsManifest, convTune,
                                          textureInput = false, textureOutput = false,
                                          staticGuard = false }) {
  if (w % 16 || h % 16) throw new Error(`rt: dims must be /16 (got ${w}x${h})`);
  const QW = w / 4, QH = h / 4, W8 = w / 8, H8 = h / 8, W16 = w / 16, H16 = h / 16;
  const useF16 = device.features.has('shader-f16');
  // channel widths come from the weights themselves (supports slim students)
  const C1 = weightsManifest['block0.conv0.0.0.weight'].shape[0]; // conv0a out (120 full / 60 slim)
  const C2 = weightsManifest['block0.conv0.1.0.weight'].shape[0]; // main width (240 full / 120 slim)
  if (C2 % 4) throw new Error('rt: main width must be /4');
  // t-factored graph: trunk (conv0 + 6 convblocks) is timestep-free and runs once
  // per pair; FiLM(t) + convblocks 6,7 + lastconv run per mid. Detected by the
  // film MLP in the manifest; input prep is 6ch (no t channel).
  const tfact = 'film.2.weight' in weightsManifest;
  const CI0 = weightsManifest['block0.conv0.0.0.weight'].shape[1]; // 7 classic, 6 tfact
  if (tfact && (!textureInput || !textureOutput)) {
    throw new Error('rt: tfact weights need texture input/output mode');
  }
  const refi = tfact && ('refine.c0.weight' in weightsManifest); // tfact2
  // fully GPU-resident path (texture in AND out): warps sample the source textures
  // directly - no model-res imgs copy, no prepFull pass; the guard reads a per-pair
  // sdiff plane. Mixed modes keep the old imgs plumbing.
  const direct = textureInput && textureOutput;
  const Z0A = C1 % 4 === 0 ? C1 / 4 : C1; // conv0a kernel packs 4 channels only when C1 is /4

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
  // one pipeline shared by all weight conversions
  const pToF16 = useF16 ? pipe(WGSL_TO_F16) : null;
  const convW = (name) => {
    if (!useF16) return wbuf[name];
    const n = man[name].shape.reduce((a, b) => a * b, 1);
    const half = bufBytes(n * 2);
    const p = pToF16;
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
  const imgs = direct ? null : buf(6 * w * h);
  const sdiff = direct && staticGuard ? buf(w * h) : null; // per-pair max-channel |A-B|
  const xq = abuf(CI0 * QH * QW);
  const f8 = abuf(C1 * H8 * W8);
  const actBytes = C2 * H16 * W16 * (useF16 ? 2 : 4);
  const f16a = bufBytes(actBytes), f16b = bufBytes(actBytes), f16r = bufBytes(actBytes);
  const tmp8 = buf(5 * H8 * W8);
  // readback plumbing exists only for the buffer-output path (rt_test harness);
  // texture-output callers never read back - ~7*w*h*4 bytes of MAP_READ saved
  const outp = textureOutput ? null : buf(w * h);
  const staging = textureOutput ? null
    : device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  // slots for batched multi-t runs (factor N: upload once, N-1 mids in ONE submit)
  const MAXT = 5;
  const tbufs = [], stagings = [];
  for (let i = 0; i < MAXT; i++) {
    tbufs.push(buf(1));
    if (!textureOutput) {
      stagings.push(device.createBuffer({ size: w * h * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
    }
  }

  const sampler = textureInput
    ? device.createSampler({ magFilter: 'linear', minFilter: 'linear',
                             addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' })
    : null;
  // the heavy shaders compile in PARALLEL on Dawn's worker pool (and without
  // blocking the main thread) - createRT is async anyway, so batch-await them
  const pipeAsync = (code, entry = 'main') => device.createComputePipelineAsync({
    layout: 'auto', compute: { module: mod(code), entryPoint: entry } });
  const sgTuned = convTune && convTune.sg && device.features.has('subgroups');
  const [pPrepFull, pPrepQ, pConv0a, pConv0b, pConvB, pConvBR, pDeconv, pFlow, pDiff] = await Promise.all([
    direct ? null : pipeAsync(textureInput ? wgslPrepFullTex(w, h) : wgslPrepFull(w, h)),
    pipeAsync(tfact ? wgslPrepQuarterTex6(w, h, useF16)
      : (textureInput ? wgslPrepQuarterTex(w, h, useF16) : wgslPrepQuarter(w, h, useF16))),
    pipeAsync(wgslConv(CI0, C1, QW, QH, W8, H8, 2, false, useF16)),
    pipeAsync(wgslConv(C1, C2, W8, H8, W16, H16, 2, false, useF16)),
    pipeAsync(useF16
      ? (sgTuned ? wgslConvRBSg : wgslConvRB)(C2, C2, W16, H16, W16, H16, false, convTune)
      : wgslConv(C2, C2, W16, H16, W16, H16, 1, false, false)),
    pipeAsync(useF16
      ? (sgTuned ? wgslConvRBSg : wgslConvRB)(C2, C2, W16, H16, W16, H16, true, convTune)
      : wgslConv(C2, C2, W16, H16, W16, H16, 1, true, false)),
    pipeAsync(wgslDeconv(C2, 5, W16, H16, W8, H8, useF16)),
    pipeAsync(textureOutput
      ? (direct ? wgslFlowOutTexDirect(w, h, staticGuard, refi) : wgslFlowOutTex(w, h, staticGuard, refi))
      : wgslFlowOut(w, h)),
    sdiff ? pipeAsync(wgslDiff(w, h)) : null,
  ]);
  // texture-output mode: flow bind groups are per output texture (small ring - cache them)
  const flowBgCache = new Map();
  function flowBgFor(tex) {
    if (!flowBgCache.has(tex)) {
      // evict BEFORE inserting - clearing after would wipe the fresh entry and
      // hand setBindGroup an undefined (latent until a caller rings >24 textures)
      if (flowBgCache.size > 24) flowBgCache.clear();
      // direct layout: 0 tmp8, 1 outTex, 2 res?, 3 sdiff? (textures ride group 1);
      // legacy layout: 0 tmp8, 1 imgs, 2 outTex, 3 res?
      const entries = direct
        ? [{ binding: 0, resource: { buffer: tmp8 } },
           { binding: 1, resource: tex.createView() }]
        : [{ binding: 0, resource: { buffer: tmp8 } },
           { binding: 1, resource: { buffer: imgs } },
           { binding: 2, resource: tex.createView() }];
      if (refi) entries.push({ binding: direct ? 2 : 3, resource: { buffer: rRes } }); // tfact2 residual
      if (sdiff) entries.push({ binding: 3, resource: { buffer: sdiff } });
      flowBgCache.set(tex, device.createBindGroup({ layout: pFlow.getBindGroupLayout(0), entries }));
    }
    return flowBgCache.get(tex);
  }

  // buffer-input prep bind groups (unused in texture mode)
  const bgPrepFull = textureInput ? null : bg(pPrepFull, [rgba0, rgba1, imgs]);
  const bgPrepQ = textureInput ? null : bg(pPrepQ, [rgba0, rgba1, xq, tbuf]);
  const bgPrepQt = textureInput ? null : tbufs.map(tb => bg(pPrepQ, [rgba0, rgba1, xq, tb]));
  // texture-mode prep bind groups are built per texture pair and cached (ping-pong -> few combos).
  // Keyed by texture IDENTITY, not label: callers recreate pools reusing the same
  // labels (resolution change), and a label-keyed cache would keep serving bind
  // groups of destroyed textures - every submit fails async validation and the
  // mids silently replay stale content.
  const texBgCache = new Map();
  function texPrepBgs(texA, texB) {
    const key = texBgId(texA) + '|' + texBgId(texB);
    if (!texBgCache.has(key)) {
      // evict BEFORE inserting - clearing after would wipe the fresh entry too
      if (texBgCache.size > 12) texBgCache.clear(); // texture set changed wholesale
      const va = texA.createView(), vb = texB.createView();
      const texEntries = [
        { binding: 0, resource: va }, { binding: 1, resource: vb },
        { binding: 2, resource: sampler }];
      texBgCache.set(key, {
        full: direct ? null : device.createBindGroup({ layout: pPrepFull.getBindGroupLayout(0), entries: [
          ...texEntries, { binding: 3, resource: { buffer: imgs } }] }),
        // direct mode: per-pair companions - the sdiff writer and the group(1)
        // texture bindings of the flow / refine-prep pipelines ('auto' layouts
        // are pipeline-unique, so each needs its own bind group)
        diff: sdiff ? device.createBindGroup({ layout: pDiff.getBindGroupLayout(0), entries: [
          ...texEntries, { binding: 3, resource: { buffer: sdiff } }] }) : null,
        flowTex: direct ? device.createBindGroup({ layout: pFlow.getBindGroupLayout(1), entries: texEntries }) : null,
        rprepTex: refi ? device.createBindGroup({ layout: pRPrep.getBindGroupLayout(1), entries: texEntries }) : null,
        q: tfact
          ? [device.createBindGroup({ layout: pPrepQ.getBindGroupLayout(0), entries: [
              { binding: 0, resource: va }, { binding: 1, resource: vb },
              { binding: 2, resource: sampler }, { binding: 3, resource: { buffer: xq } }] })]
          : tbufs.map(tb => device.createBindGroup({ layout: pPrepQ.getBindGroupLayout(0), entries: [
              { binding: 0, resource: va }, { binding: 1, resource: vb },
              { binding: 2, resource: sampler }, { binding: 3, resource: { buffer: xq } },
              { binding: 4, resource: { buffer: tb } }] })),
      });
    }
    return texBgCache.get(key);
  }

  // tfact-only state: trunk feature buffer + FiLM params (tiny t-MLP runs in JS)
  const hbuf = tfact ? bufBytes(C2 * H16 * W16 * (useF16 ? 2 : 4)) : null;
  const filmBuf = tfact ? buf(2 * C2) : null;
  const pFilm = tfact ? pipe(wgslFilm(C2, H16 * W16, useF16)) : null;
  let bgFilm = null, filmW = null;
  if (tfact) {
    bgFilm = bg(pFilm, [hbuf, filmBuf, f16a]);
    const f32 = (name) => {
      const m = weightsManifest[name];
      return new Float32Array(weightsBin, m.offset * 4, m.shape.reduce((a, b) => a * b, 1));
    };
    filmW = { w0: f32('film.0.weight'), b0: f32('film.0.bias'),
              w2: f32('film.2.weight'), b2: f32('film.2.bias') };
  }
  // tfact2 refine head: occlusion repair at QUARTER res; the residual is folded
  // into the flowout pass (withRes)
  const RW4 = w / 4, RH4 = h / 4;
  let pRPrep = null, pRC0 = null, pRC1 = null, pROut = null;
  let bgRPrep = null, bgRC0 = null, bgRC1 = null, bgRC2 = null, bgROut = null;
  let RC = 0, rRes = null;
  if (refi) {
    RC = man['refine.c0.weight'].shape[0];
    // the refine convs are dispatched with z = RC/4 below, and wgslConv only
    // packs 4-wide when CO % 4 == 0 - a non-/4 head would silently mis-dispatch
    if (RC % 4 !== 0) throw new Error('rt: refine channels must be a multiple of 4, got ' + RC);
    const rIn = abuf(11 * RH4 * RW4);
    const rA2 = abuf(RC * RH4 * RW4);
    const rB2 = abuf(RC * RH4 * RW4);
    rRes = buf(3 * RH4 * RW4);
    [pRPrep, pRC0, pRC1, pROut] = await Promise.all([
      pipeAsync(wgslRefinePrep(w, h, useF16)),
      pipeAsync(wgslConv(11, RC, RW4, RH4, RW4, RH4, 1, false, useF16)),
      pipeAsync(wgslConv(RC, RC, RW4, RH4, RW4, RH4, 1, false, useF16)),
      pipeAsync(wgslRefineOut(RC, RW4, RH4, useF16)),
    ]);
    bgRPrep = bg(pRPrep, [tmp8, rIn]); // sources ride group(1) per pair (rprepTex)
    bgRC0 = bg(pRC0, [rIn, convW('refine.c0.weight'), wbuf['refine.c0.bias'], wbuf['refine.a0.weight'], rA2]);
    bgRC1 = bg(pRC1, [rA2, convW('refine.c1.weight'), wbuf['refine.c1.bias'], wbuf['refine.a1.weight'], rB2]);
    bgRC2 = bg(pRC1, [rB2, convW('refine.c2.weight'), wbuf['refine.c2.bias'], wbuf['refine.a2.weight'], rA2]);
    bgROut = bg(pROut, [rA2, wbuf['refine.c3.weight'], wbuf['refine.c3.bias'], rRes]);
  }

  // scratch reused across calls: these run once per mid (writeBuffer reads the
  // array synchronously, so reuse is safe) - allocating per call is pure GC churn
  const tScratch = new Float32Array(1);
  const filmScratch = tfact ? { out: new Float32Array(2 * C2), h: new Float32Array(filmW.b0.length) } : null;
  function filmParams(t) {
    const HN = filmW.b0.length, { out, h } = filmScratch;
    for (let j = 0; j < HN; j++) h[j] = Math.max(0, filmW.w0[j] * t + filmW.b0[j]);
    for (let k = 0; k < 2 * C2; k++) {
      let s = filmW.b2[k];
      const row = k * HN;
      for (let j = 0; j < HN; j++) s += filmW.w2[row + j] * h[j];
      out[k] = s;
    }
    return out;
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
  const bgDeconv = bg(pDeconv, [f16out, convW('block0.lastconv.weight'), wbuf['block0.lastconv.bias'], tmp8]);
  const bgFlow = textureOutput ? null : bg(pFlow, [tmp8, imgs, outp]);

  const gx = (n) => Math.ceil(n / WG);
  // register-blocked convblock kernel covers 16x16 output per workgroup
  const cbX = useF16 ? Math.ceil(W16 / 16) : gx(W16);
  const cbY = useF16 ? Math.ceil(H16 / 16) : gx(H16);
  const cbZ = useF16 ? C2 / ((convTune && convTune.coc) || 4) : C2 / 4;

  // per-stage GPU times via timestamp queries (needs 'timestamp-query' on the device)
  async function profile(rgbaA, rgbaB) {
    if (textureInput) return 'profile: buffer-input mode only';
    if (!device.features.has('timestamp-query')) return 'no timestamp-query feature';
    const stages = [
      ['prepFull', pPrepFull, bgPrepFull, [gx(w), gx(h), 1]],
      ['prepQ', pPrepQ, bgPrepQ, [gx(QW), gx(QH), 1]],
      ['conv0a', pConv0a, bgConv0a, [gx(W8), gx(H8), Z0A]],
      ['conv0b', pConv0b, bgConv0b, [gx(W16), gx(H16), C2 / 4]],
      ...bgB.map(({ p, g }, i) => [`convB${i}`, p, g, [cbX, cbY, cbZ]]),
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
    if (tfact) throw new Error('rt: tfact weights have no buffer-mode run()');
    device.queue.writeBuffer(tbuf, 0, new Float32Array([t]));
    device.queue.writeBuffer(rgba0, 0, rgbaA.buffer, rgbaA.byteOffset, w * h * 4);
    device.queue.writeBuffer(rgba1, 0, rgbaB.buffer, rgbaB.byteOffset, w * h * 4);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pPrepFull); pass.setBindGroup(0, bgPrepFull); pass.dispatchWorkgroups(gx(w), gx(h));
    pass.setPipeline(pPrepQ); pass.setBindGroup(0, bgPrepQ); pass.dispatchWorkgroups(gx(QW), gx(QH));
    pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), Z0A);
    pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), C2 / 4);
    pass.end();
    // residual copy AFTER conv0b (f16r = f16a snapshot)
    enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes);
    const pass2 = enc.beginComputePass();
    for (const { p, g } of bgB) {
      pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
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
  // With textureOutput, outTexs[i] receives mid i and NOTHING is read back - returns null.
  async function runMulti(a, b, ts, outTexs) {
    if (tfact) { // factored graph: trunk once, head per t
      prepPair(a, b);
      for (let i = 0; i < ts.length; i++) runT(ts[i], outTexs[i]);
      return null;
    }
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
      if (direct) {
        if (sdiff) { pass.setPipeline(pDiff); pass.setBindGroup(0, tbg.diff); pass.dispatchWorkgroups(gx(w), gx(h)); }
      } else {
        pass.setPipeline(pPrepFull); pass.setBindGroup(0, tbg ? tbg.full : bgPrepFull); pass.dispatchWorkgroups(gx(w), gx(h));
      }
      pass.end();
    }
    for (let i = 0; i < ts.length; i++) {
      const pass = enc.beginComputePass();
      pass.setPipeline(pPrepQ); pass.setBindGroup(0, tbg ? tbg.q[i] : bgPrepQt[i]); pass.dispatchWorkgroups(gx(QW), gx(QH));
      pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), Z0A);
      pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), C2 / 4);
      pass.end();
      enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes);
      const pass2 = enc.beginComputePass();
      for (const { p, g } of bgB) {
        pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
      }
      pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8), 5);
      pass2.setPipeline(pFlow);
      pass2.setBindGroup(0, textureOutput ? flowBgFor(outTexs[i]) : bgFlow);
      if (direct) pass2.setBindGroup(1, tbg.flowTex);
      pass2.dispatchWorkgroups(gx(w), gx(h));
      pass2.end();
      if (!textureOutput) enc.copyBufferToBuffer(outp, 0, stagings[i], 0, w * h * 4);
    }
    device.queue.submit([enc.finish()]);
    if (textureOutput) return null; // mids live in outTexs, nothing crosses the bus
    // map all stagings concurrently - sequential awaits cost ~1ms each
    await Promise.all(stagings.slice(0, ts.length).map(s => s.mapAsync(GPUMapMode.READ)));
    const outs = [];
    for (let i = 0; i < ts.length; i++) {
      outs.push(new Uint8Array(stagings[i].getMappedRange().slice(0)));
      stagings[i].unmap();
    }
    return outs;
  }

  // ---- lazy per-mid API (texture in/out mode) ----
  // The queue is FIFO: a mid's present blit executes after EVERYTHING submitted
  // before it. Batching all mids upfront therefore makes the FIRST mid wait for
  // the WHOLE batch on the GPU. prepPair + runT let the caller submit each mid
  // just-in-time so present blits interleave with computes - the required
  // presentation delay shrinks from ~2x batch time to ~one mid time.
  let curPrep = null;
  function prepPair(a, b) {
    if (!textureInput) throw new Error('prepPair: texture-input mode only');
    curPrep = texPrepBgs(a, b);
    const enc = device.createCommandEncoder();
    if (tfact) {
      // the WHOLE t-free trunk runs here, once per pair
      const pass = enc.beginComputePass();
      if (sdiff) { // per-pair guard plane (replaces the old full-res prepFull)
        pass.setPipeline(pDiff); pass.setBindGroup(0, curPrep.diff); pass.dispatchWorkgroups(gx(w), gx(h));
      }
      pass.setPipeline(pPrepQ); pass.setBindGroup(0, curPrep.q[0]); pass.dispatchWorkgroups(gx(QW), gx(QH));
      pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), Z0A);
      pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), C2 / 4);
      pass.end();
      enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes); // feat0 residual for the head
      const pass2 = enc.beginComputePass();
      for (let i = 0; i < 6; i++) {
        pass2.setPipeline(bgB[i].p); pass2.setBindGroup(0, bgB[i].g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
      }
      pass2.end();
      enc.copyBufferToBuffer(f16a, 0, hbuf, 0, actBytes); // trunk features, reused per t
    } else {
      const pass = enc.beginComputePass();
      if (direct) {
        if (sdiff) { pass.setPipeline(pDiff); pass.setBindGroup(0, curPrep.diff); pass.dispatchWorkgroups(gx(w), gx(h)); }
      } else {
        pass.setPipeline(pPrepFull); pass.setBindGroup(0, curPrep.full); pass.dispatchWorkgroups(gx(w), gx(h));
      }
      pass.end();
    }
    device.queue.submit([enc.finish()]);
  }
  function runT(t, outTex) {
    if (!curPrep) throw new Error('runT before prepPair');
    if (!textureOutput) throw new Error('runT: texture-output mode only');
    const enc = device.createCommandEncoder();
    if (tfact) {
      // per-mid: FiLM(t) + convblocks 6,7 (+feat0 residual) + deconv + flow
      device.queue.writeBuffer(filmBuf, 0, filmParams(t));
      const pass = enc.beginComputePass();
      pass.setPipeline(pFilm); pass.setBindGroup(0, bgFilm);
      pass.dispatchWorkgroups(Math.ceil((C2 * H16 * W16) / 256));
      pass.setPipeline(bgB[6].p); pass.setBindGroup(0, bgB[6].g); pass.dispatchWorkgroups(cbX, cbY, cbZ);
      pass.setPipeline(bgB[7].p); pass.setBindGroup(0, bgB[7].g); pass.dispatchWorkgroups(cbX, cbY, cbZ);
      pass.setPipeline(pDeconv); pass.setBindGroup(0, bgDeconv); pass.dispatchWorkgroups(gx(W8), gx(H8), 5);
      if (refi) { // quarter-res refine chain; the flowout below folds the residual in
        pass.setPipeline(pRPrep); pass.setBindGroup(0, bgRPrep);
        pass.setBindGroup(1, curPrep.rprepTex); // source textures for the direct warp
        pass.dispatchWorkgroups(gx(RW4), gx(RH4));
        pass.setPipeline(pRC0); pass.setBindGroup(0, bgRC0); pass.dispatchWorkgroups(gx(RW4), gx(RH4), RC / 4);
        pass.setPipeline(pRC1); pass.setBindGroup(0, bgRC1); pass.dispatchWorkgroups(gx(RW4), gx(RH4), RC / 4);
        pass.setPipeline(pRC1); pass.setBindGroup(0, bgRC2); pass.dispatchWorkgroups(gx(RW4), gx(RH4), RC / 4);
        pass.setPipeline(pROut); pass.setBindGroup(0, bgROut); pass.dispatchWorkgroups(gx(RW4), gx(RH4));
      }
      pass.setPipeline(pFlow); pass.setBindGroup(0, flowBgFor(outTex));
      pass.setBindGroup(1, curPrep.flowTex); // 'auto' layouts are pipeline-unique - rebind
      pass.dispatchWorkgroups(gx(w), gx(h));
      pass.end();
      device.queue.submit([enc.finish()]);
      return;
    }
    // single tbuf is safe: writeBuffer and submits are queue-ordered
    tScratch[0] = t;
    device.queue.writeBuffer(tbufs[0], 0, tScratch);
    const pass = enc.beginComputePass();
    pass.setPipeline(pPrepQ); pass.setBindGroup(0, curPrep.q[0]); pass.dispatchWorkgroups(gx(QW), gx(QH));
    pass.setPipeline(pConv0a); pass.setBindGroup(0, bgConv0a); pass.dispatchWorkgroups(gx(W8), gx(H8), Z0A);
    pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(gx(W16), gx(H16), C2 / 4);
    pass.end();
    enc.copyBufferToBuffer(f16a, 0, f16r, 0, actBytes);
    const pass2 = enc.beginComputePass();
    for (const { p, g } of bgB) {
      pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
    }
    pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8), 5);
    pass2.setPipeline(pFlow); pass2.setBindGroup(0, flowBgFor(outTex));
    pass2.setBindGroup(1, curPrep.flowTex); // direct warp sources (runT implies texture in+out)
    pass2.dispatchWorkgroups(gx(w), gx(h));
    pass2.end();
    device.queue.submit([enc.finish()]);
  }

  return { run, runMulti, prepPair, runT, profile, w, h };
}


// ---- one-shot conv autotune: bench wgslConvRB variants on this device ----
// Returns the fastest {coc, slab, ms}. ~200-400ms of GPU time; call it once per
// (adapter, model width, resolution) and persist the answer - relative ranking
// is what matters, so light background load is tolerable.
export async function tuneConvRB(device, { ci, co, w16, h16 }) {
  const base = [{ coc: 4, slab: 20 }, { coc: 8, slab: 20 }, { coc: 8, slab: 12 }, { coc: 4, slab: 12 }]
    .filter(v => co % v.coc === 0 && (v.slab * 324 + v.coc * v.slab * 9) * 2 <= 16384);
  // subgroup variants (weights via subgroupBroadcastFirst, no shared staging):
  // measured +20% at the 360p grid and -19% at 720p/coc8 on a 4060 Ti - exactly
  // why they go through the tuner instead of being hardcoded
  const variants = device.features.has('subgroups')
    ? [...base, ...base.map(v => ({ ...v, sg: true }))]
    : base;
  const buf = (bytes) => device.createBuffer({ size: Math.ceil(bytes / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const src = buf(ci * w16 * h16 * 2), dst = buf(co * w16 * h16 * 2);
  const wgt = buf(co * ci * 9 * 2), bias = buf(co * 4), alpha = buf(co * 4);
  let best = null;
  for (const v of variants) {
    const gen = v.sg ? wgslConvRBSg : wgslConvRB;
    const p = device.createComputePipeline({ layout: 'auto', compute: {
      module: device.createShaderModule({ code: gen(ci, co, w16, h16, w16, h16, false, v) }),
      entryPoint: 'main' } });
    const bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: wgt } },
      { binding: 2, resource: { buffer: bias } }, { binding: 3, resource: { buffer: alpha } },
      { binding: 4, resource: { buffer: dst } }] });
    const run = (k) => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(p); pass.setBindGroup(0, bg);
      for (let i = 0; i < k; i++) pass.dispatchWorkgroups(Math.ceil(w16 / 16), Math.ceil(h16 / 16), co / v.coc);
      pass.end();
      device.queue.submit([enc.finish()]);
    };
    run(3); await device.queue.onSubmittedWorkDone(); // warm (incl pipeline compile)
    const t0 = performance.now();
    run(30); await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / 30;
    if (!best || ms < best.ms) best = { ...v, ms };
  }
  [src, dst, wgt, bias, alpha].forEach(b => b.destroy());
  return best;
}
