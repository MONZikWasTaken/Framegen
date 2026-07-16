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
export function wgslConv(CI, CO, IW, IH, OW, OH, stride, residual, f16, sparse, cocOpt) {
  // sparse: {lbase, T, txt} - workgroup x indexes a tile LIST (sparse-refine path,
  // dispatched indirectly) instead of the dense grid; stride-1 only
  // cocOpt widens the channel block (fewer z-slices re-staging the same tile)
  const COC = cocOpt || (CO % 4 === 0 ? 4 : 1); // channels per thread
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
${sparse ? `@group(0) @binding(5) var<storage, read> tiles: array<u32>;` : ``/* sparse+residual would collide on binding 5 - refine convs are never residual */}

var<workgroup> wsh: array<${T}, ${slabFloats}>; // [COC, SLAB, 9] slab of weights
${stride === 1 ? `var<workgroup> tile: array<${T}, ${SLAB * 100}>; // [SLAB, 10, 10] input tiles (8x8 out + halo)` : ''}

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lx = i32(lid.x); let ly = i32(lid.y);
${sparse ? /* wgsl */`
  let tid = i32(tiles[${sparse.lbase} + i32(wid.x)]);
  let wx0 = (tid % ${sparse.txt}) * ${WG}; let wy0 = (tid / ${sparse.txt}) * ${WG};
  let x = wx0 + lx; let y = wy0 + ly;
` : `
  let x = i32(gid.x); let y = i32(gid.y);
  let wx0 = i32(wid.x) * ${WG}; let wy0 = i32(wid.y) * ${WG};
`}
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
// COC output channels - every shared read feeds 4 FMAs instead of ~1.
// Workgroup shape is tunable: wgx*wgy threads cover a (2*wgx)x(2*wgy) output tile.
// Bigger workgroups (128/256 threads) trade shared-tile reuse for SM occupancy -
// the 64-thread default parks ~12% occupancy on Ada and stalls on latency, so the
// tuner explores both (measured, never hardcoded).
export function wgslConvRB(CI, CO, IW, IH, OW, OH, residual, tune, film) {
  // tune: {coc, slab, wgx, wgy, w4, v2} - shared memory must fit
  // slab*TS*2 + coc*slab*9*(w4?4:2) <= 16384 where TS = (2*wgx+2)*(2*wgy+2)
  // w4: the thread's 4x4 input window loads into registers ONCE per ci (16 tile
  //     ops instead of 36) and the weight slab stages as f32 - the f16->f32
  //     converts that dominate the legacy inner loop on Ada (quarter-rate CVT)
  //     leave the hot loop. Callers bind pre-widened weights (bit-exact
  //     f32(f16(w)) mirrors), so values and accumulation order are unchanged.
  // v2: on top of w4, the tile stages and reads as vec2<f16> pairs - halves the
  //     shared load instructions (alignment holds: TW2 and lx*2 are both even).
  // film: fuse the FiLM affine (x*(1+scale[ci])+shift[ci], prm=[2*CI] f32) into
  //     the tile load - same arithmetic and f16 rounding as the standalone film
  //     pass. Padding taps stay 0 (film applies to in-bounds texels only).
  const COC = (tune && tune.coc) || 4, SLAB = (tune && tune.slab) || 20;
  const WGX = (tune && tune.wgx) || 8, WGY = (tune && tune.wgy) || 8;
  const W4 = !!(tune && tune.w4), V2 = !!(W4 && tune.v2);
  const WT = W4 ? 'f32' : 'f16';
  const OTW = WGX * 2, OTH = WGY * 2;   // output tile
  const TW2 = OTW + 2;                  // input tile row (halo 1px each side)
  const TS = TW2 * (OTH + 2);           // input tile elems per ci
  const NT = WGX * WGY;                 // threads per wg
  const slabW = COC * SLAB * 9;
  const slabT = SLAB * TS;
  const FILM = (v) => film ? `        ${v} = f16(f32(${v}) * (1.0 + prm[s + ci]) + prm[${CI} + s + ci]);` : '';
  // window preload: 16 registers t{row}{col}, converted to f32 once per ci
  const WIN = [0, 1, 2, 3].map(r => V2
    ? `      let p${r}a = tile[tb2 + ${r * (TW2 / 2)}]; let p${r}b = tile[tb2 + ${r * (TW2 / 2) + 1}];
      let t${r}0 = f32(p${r}a.x); let t${r}1 = f32(p${r}a.y); let t${r}2 = f32(p${r}b.x); let t${r}3 = f32(p${r}b.y);`
    : `      let t${r}0 = f32(tile[tb + ${r * TW2}]); let t${r}1 = f32(tile[tb + ${r * TW2 + 1}]); let t${r}2 = f32(tile[tb + ${r * TW2 + 2}]); let t${r}3 = f32(tile[tb + ${r * TW2 + 3}]);`).join('\n');
  const TAPS = [0, 1, 2].map(ky => [0, 1, 2].map(kx => `      {
        let wb = ci * 9 + ${ky * 3 + kx};
${Array.from({ length: COC }, (_, c) => `        {
          let wv = wsh[${c} * (sl * 9) + wb];
          a${c}0 += t${ky}${kx} * wv; a${c}1 += t${ky}${kx + 1} * wv; a${c}2 += t${ky + 1}${kx} * wv; a${c}3 += t${ky + 1}${kx + 1} * wv;
        }`).join('\n')}
      }`).join('\n')).join('\n');
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<${WT}>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;
${residual ? `@group(0) @binding(5) var<storage, read> res: array<f16>;` : ``}
${film ? `@group(0) @binding(5) var<storage, read> prm: array<f32>; // [2*CI] FiLM scale,shift` : ``}

var<workgroup> wsh: array<${WT}, ${slabW}>;
${V2 ? `var<workgroup> tile: array<vec2<f16>, ${slabT / 2}>;` : `var<workgroup> tile: array<f16, ${slabT}>;`}

@compute @workgroup_size(${WGX}, ${WGY}, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lx = i32(lid.x); let ly = i32(lid.y);
  let ox0 = i32(wid.x) * ${OTW}; let oy0 = i32(wid.y) * ${OTH};   // wg output origin
  let x0 = ox0 + lx * 2; let y0 = oy0 + ly * 2;           // this thread's 2x2 patch
  let cb = i32(wid.z) * ${COC};
  // scalar accumulators (unrolled - arrays may spill out of registers in WGSL)
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
      idx += ${NT};
    }
${V2 ? /* vec2 pair staging: e and e+1 share ci and row because TS and TW2 are even */ `    var ti = i32(li);
    let tn = sl * ${TS / 2};
    while (ti < tn) {
      let e = ti * 2;
      let ci = e / ${TS};
      let r = e % ${TS};
      let ty = oy0 + r / ${TW2} - 1;
      let tx = ox0 + r % ${TW2} - 1;
      var va = f16(0.0); var vb = f16(0.0);
      if (ty >= 0 && ty < ${IH}) {
        if (tx >= 0 && tx < ${IW}) {
          va = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
${FILM('va')}
        }
        if (tx + 1 >= 0 && tx + 1 < ${IW}) {
          vb = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx + 1];
${FILM('vb')}
        }
      }
      tile[ti] = vec2<f16>(va, vb);
      ti += ${NT};
    }` : `    var ti = i32(li);
    let tn = sl * ${TS};
    while (ti < tn) {
      let ci = ti / ${TS};
      let r = ti % ${TS};
      let ty = oy0 + r / ${TW2} - 1;
      let tx = ox0 + r % ${TW2} - 1;
      var v = f16(0.0);
      if (ty >= 0 && ty < ${IH} && tx >= 0 && tx < ${IW}) {
        v = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
${FILM('v')}
      }
      tile[ti] = v;
      ti += ${NT};
    }`}
    workgroupBarrier();
    for (var ci = 0; ci < sl; ci++) {
${W4 ? `${V2 ? `      let tb2 = ci * ${TS / 2} + ly * ${TW2} + lx;` : `      let tb = ci * ${TS} + (ly * 2) * ${TW2} + lx * 2;`}
${WIN}
${TAPS}` : `      let tb = ci * ${TS} + (ly * 2) * ${TW2} + lx * 2; // top-left of this thread's 4x4 window
      for (var ky = 0; ky < 3; ky++) {
        let rb = tb + ky * ${TW2};
        for (var kx = 0; kx < 3; kx++) {
          let t00 = f32(tile[rb + kx]);
          let t01 = f32(tile[rb + kx + 1]);
          let t10 = f32(tile[rb + kx + ${TW2}]);
          let t11 = f32(tile[rb + kx + ${TW2 + 1}]);
          let wb = ci * 9 + ky * 3 + kx;
${Array.from({ length: COC }, (_, c) => `          {
            let wv = f32(wsh[${c} * (sl * 9) + wb]);
            a${c}0 += t00 * wv; a${c}1 += t01 * wv; a${c}2 += t10 * wv; a${c}3 += t11 * wv;
          }`).join('\n')}
        }
      }`}
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
export function wgslConvRBSg(CI, CO, IW, IH, OW, OH, residual, tune, film) {
  const COC = (tune && tune.coc) || 4, SLAB = (tune && tune.slab) || 20;
  const WGX = (tune && tune.wgx) || 8, WGY = (tune && tune.wgy) || 8;
  const W4 = !!(tune && tune.w4), V2 = !!(W4 && tune.v2);
  const WT = W4 ? 'f32' : 'f16';
  const OTW = WGX * 2, OTH = WGY * 2;
  const TW2 = OTW + 2;
  const TS = TW2 * (OTH + 2);
  const NT = WGX * WGY;
  const slabT = SLAB * TS;
  const FILM = (v) => film ? `        ${v} = f16(f32(${v}) * (1.0 + prm[s + ci]) + prm[${CI} + s + ci]);` : '';
  const WIN = [0, 1, 2, 3].map(r => V2
    ? `      let p${r}a = tile[tb2 + ${r * (TW2 / 2)}]; let p${r}b = tile[tb2 + ${r * (TW2 / 2) + 1}];
      let t${r}0 = f32(p${r}a.x); let t${r}1 = f32(p${r}a.y); let t${r}2 = f32(p${r}b.x); let t${r}3 = f32(p${r}b.y);`
    : `      let t${r}0 = f32(tile[tb + ${r * TW2}]); let t${r}1 = f32(tile[tb + ${r * TW2 + 1}]); let t${r}2 = f32(tile[tb + ${r * TW2 + 2}]); let t${r}3 = f32(tile[tb + ${r * TW2 + 3}]);`).join('\n');
  // W4 weights read straight from the pre-widened f32 buffer - no CVT in the loop
  const TAPS = [0, 1, 2].map(ky => [0, 1, 2].map(kx => `      {
${Array.from({ length: COC }, (_, c) => `        {
          let wv = subgroupBroadcastFirst(wgt[(cb + ${c}) * ${CI * 9} + wrow + ${ky * 3 + kx}]);
          a${c}0 += t${ky}${kx} * wv; a${c}1 += t${ky}${kx + 1} * wv; a${c}2 += t${ky + 1}${kx} * wv; a${c}3 += t${ky + 1}${kx + 1} * wv;
        }`).join('\n')}
      }`).join('\n')).join('\n');
  return /* wgsl */`
enable f16;
enable subgroups;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<${WT}>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;
${residual ? `@group(0) @binding(5) var<storage, read> res: array<f16>;` : ``}
${film ? `@group(0) @binding(5) var<storage, read> prm: array<f32>; // [2*CI] FiLM scale,shift` : ``}

${V2 ? `var<workgroup> tile: array<vec2<f16>, ${slabT / 2}>;` : `var<workgroup> tile: array<f16, ${slabT}>;`}

@compute @workgroup_size(${WGX}, ${WGY}, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lx = i32(lid.x); let ly = i32(lid.y);
  let ox0 = i32(wid.x) * ${OTW}; let oy0 = i32(wid.y) * ${OTH};
  let x0 = ox0 + lx * 2; let y0 = oy0 + ly * 2;
  let cb = i32(wid.z) * ${COC};
${Array.from({ length: COC }, (_, c) =>
  `  var a${c}0 = bias[cb + ${c}]; var a${c}1 = a${c}0; var a${c}2 = a${c}0; var a${c}3 = a${c}0;`).join('\n')}

  for (var s = 0; s < ${CI}; s += ${SLAB}) {
    let sl = min(${SLAB}, ${CI} - s);
    workgroupBarrier();
${V2 ? `    var ti = i32(li);
    let tn = sl * ${TS / 2};
    while (ti < tn) {
      let e = ti * 2;
      let ci = e / ${TS};
      let r = e % ${TS};
      let ty = oy0 + r / ${TW2} - 1;
      let tx = ox0 + r % ${TW2} - 1;
      var va = f16(0.0); var vb = f16(0.0);
      if (ty >= 0 && ty < ${IH}) {
        if (tx >= 0 && tx < ${IW}) {
          va = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
${FILM('va')}
        }
        if (tx + 1 >= 0 && tx + 1 < ${IW}) {
          vb = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx + 1];
${FILM('vb')}
        }
      }
      tile[ti] = vec2<f16>(va, vb);
      ti += ${NT};
    }` : `    var ti = i32(li);
    let tn = sl * ${TS};
    while (ti < tn) {
      let ci = ti / ${TS};
      let r = ti % ${TS};
      let ty = oy0 + r / ${TW2} - 1;
      let tx = ox0 + r % ${TW2} - 1;
      var v = f16(0.0);
      if (ty >= 0 && ty < ${IH} && tx >= 0 && tx < ${IW}) {
        v = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
${FILM('v')}
      }
      tile[ti] = v;
      ti += ${NT};
    }`}
    workgroupBarrier();
    for (var ci = 0; ci < sl; ci++) {
      let wrow = (s + ci) * 9;
${W4 ? `${V2 ? `      let tb2 = ci * ${TS / 2} + ly * ${TW2} + lx;` : `      let tb = ci * ${TS} + (ly * 2) * ${TW2} + lx * 2;`}
${WIN}
${TAPS}` : `      let tb = ci * ${TS} + (ly * 2) * ${TW2} + lx * 2;
      for (var ky = 0; ky < 3; ky++) {
        let rb = tb + ky * ${TW2};
        for (var kx = 0; kx < 3; kx++) {
          let t00 = f32(tile[rb + kx]);
          let t01 = f32(tile[rb + kx + 1]);
          let t10 = f32(tile[rb + kx + ${TW2}]);
          let t11 = f32(tile[rb + kx + ${TW2 + 1}]);
          let wk = wrow + ky * 3 + kx;
${Array.from({ length: COC }, (_, c) => `          {
            let wv = subgroupBroadcastFirst(f32(wgt[(cb + ${c}) * ${CI * 9} + wk]));
            a${c}0 += t00 * wv; a${c}1 += t01 * wv; a${c}2 += t10 * wv; a${c}3 += t11 * wv;
          }`).join('\n')}
        }
      }`}
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

// register-blocked conv3x3 STRIDE-2 (f16): 2x2 output patch x COC channels per
// thread with the input tile staged in shared - the plain stride-2 path had no
// input staging at all (conv0b re-read every src texel ~9x from global per
// z-slice). Accumulation order (s, ci, ky, kx) matches wgslConv - bit-exact.
export function wgslConvRBs2(CI, CO, IW, IH, OW, OH, tune) {
  const COC = (tune && tune.coc) || (CO % 8 === 0 ? 8 : 4);
  // w4 (same insight as wgslConvRB): the thread's 5x5 stride-2 input window
  // converts to f32 registers ONCE per ci (25 CVTs instead of 36) and weights
  // stage as pre-widened f32 (0 CVTs instead of 9*COC per ci) - Ada's
  // quarter-rate f16->f32 CVT leaves the hot loop. Bit-exact: window values
  // are the same f32(f16) numbers, accumulation order (s, ci, ky, kx) unchanged.
  const W4 = !!(tune && tune.w4);
  const WT = W4 ? 'f32' : 'f16';
  const WGX = 8, WGY = 8;
  const OTW = WGX * 2, OTH = WGY * 2;      // 16x16 output tile
  const TW2 = OTW * 2 + 1;                 // input tile row: stride 2 + 3-tap halo
  const TS = TW2 * (OTH * 2 + 1);
  const NT = WGX * WGY;
  // tile is the shared hog at stride 2 (33x33/ci) - slab down to fit 16KB
  let SLAB = (tune && tune.slab) || 7;
  while (SLAB > 1 && SLAB * TS * 2 + COC * SLAB * 9 * (W4 ? 4 : 2) > 16384) SLAB--;
  const slabW = COC * SLAB * 9;
  const slabT = SLAB * TS;
  // 5x5 window registers u{r}{c}: outputs (p0..p3) read rows/cols {r,r+2}
  const WIN = Array.from({ length: 5 }, (_, r) =>
    `      let u${r}0 = f32(tile[tb + ${r * TW2}]); let u${r}1 = f32(tile[tb + ${r * TW2 + 1}]); let u${r}2 = f32(tile[tb + ${r * TW2 + 2}]); let u${r}3 = f32(tile[tb + ${r * TW2 + 3}]); let u${r}4 = f32(tile[tb + ${r * TW2 + 4}]);`).join('\n');
  const TAPS = [0, 1, 2].map(ky => [0, 1, 2].map(kx => `      {
        let wb = ci * 9 + ${ky * 3 + kx};
${Array.from({ length: COC }, (_, c) => `        {
          let wv = wsh[${c} * (sl * 9) + wb];
          a${c}0 += u${ky}${kx} * wv; a${c}1 += u${ky}${kx + 2} * wv; a${c}2 += u${ky + 2}${kx} * wv; a${c}3 += u${ky + 2}${kx + 2} * wv;
        }`).join('\n')}
      }`).join('\n')).join('\n');
  return /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<${WT}>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;

var<workgroup> wsh: array<${WT}, ${slabW}>;
var<workgroup> tile: array<f16, ${slabT}>;

@compute @workgroup_size(${WGX}, ${WGY}, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let lx = i32(lid.x); let ly = i32(lid.y);
  let ox0 = i32(wid.x) * ${OTW}; let oy0 = i32(wid.y) * ${OTH};
  let x0 = ox0 + lx * 2; let y0 = oy0 + ly * 2;
  let cb = i32(wid.z) * ${COC};
${Array.from({ length: COC }, (_, c) =>
  `  var a${c}0 = bias[cb + ${c}]; var a${c}1 = a${c}0; var a${c}2 = a${c}0; var a${c}3 = a${c}0;`).join('\n')}
  // input tile origin: first output's leftmost tap = 2*ox0 - 1
  let ix0 = ox0 * 2 - 1; let iy0 = oy0 * 2 - 1;
  for (var s = 0; s < ${CI}; s += ${SLAB}) {
    let sl = min(${SLAB}, ${CI} - s);
    workgroupBarrier();
    var idx = i32(li);
    let wn = ${COC} * sl * 9;
    while (idx < wn) {
      let c = idx / (sl * 9);
      let r = idx % (sl * 9);
      wsh[idx] = wgt[(cb + c) * ${CI * 9} + (s + r / 9) * 9 + r % 9];
      idx += ${NT};
    }
    var ti = i32(li);
    let tn = sl * ${TS};
    while (ti < tn) {
      let ci = ti / ${TS};
      let r = ti % ${TS};
      let ty = iy0 + r / ${TW2};
      let tx = ix0 + r % ${TW2};
      var v = f16(0.0);
      if (ty >= 0 && ty < ${IH} && tx >= 0 && tx < ${IW}) {
        v = src[(s + ci) * ${IH * IW} + ty * ${IW} + tx];
      }
      tile[ti] = v;
      ti += ${NT};
    }
    workgroupBarrier();
    for (var ci = 0; ci < sl; ci++) {
      // this thread's 2x2 outputs read input rows ly*4+ky and ly*4+ky+2,
      // cols lx*4+kx and lx*4+kx+2 (stride-2 neighbors)
      let tb = ci * ${TS} + (ly * 4) * ${TW2} + lx * 4;
${W4 ? `${WIN}
${TAPS}` : `      for (var ky = 0; ky < 3; ky++) {
        let rb = tb + ky * ${TW2};
        for (var kx = 0; kx < 3; kx++) {
          let t00 = f32(tile[rb + kx]);
          let t01 = f32(tile[rb + kx + 2]);
          let t10 = f32(tile[rb + kx + ${TW2 * 2}]);
          let t11 = f32(tile[rb + kx + ${TW2 * 2 + 2}]);
          let wb = ci * 9 + ky * 3 + kx;
${Array.from({ length: COC }, (_, c) => `          {
            let wv = f32(wsh[${c} * (sl * 9) + wb]);
            a${c}0 += t00 * wv; a${c}1 += t01 * wv; a${c}2 += t10 * wv; a${c}3 += t11 * wv;
          }`).join('\n')}
        }
      }`}
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
        dst[o] = f16(v);
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
  // v2: tmp8 lives in two rgba16float textures (flow 4ch + mask) written by the
  // deconv - ONE hw bilinear sample replaces the 4-tap software up() per plane
  // (20 buffer reads -> 2 samples), and the refine residual is a texture too.
  // uv is shared: the /8, /4 and full-res grids all map the pixel center to
  // ((x,y)+0.5)/(W,H). Precision: f16 texel storage + ~8-bit subtexel filtering -
  // same +-1 LSB class as the direct-warp and rgba8-store deviations.
  const RES_ADD = withRes ? /* wgsl */`
  bgr = bgr + textureSampleLevel(resT, samp, uv8, 0.0).xyz;` : '';
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
@group(0) @binding(0) var t8f: texture_2d<f32>;  // flow/8: fx0,fy0,fx1,fy1
@group(0) @binding(1) var t8m: texture_2d<f32>;  // mask logit in .x
@group(0) @binding(2) var outTex: texture_storage_2d<rgba8unorm, write>;
${withRes ? `@group(0) @binding(3) var resT: texture_2d<f32>; // refine residual b,g,r` : ''}
${staticGuard ? /* wgsl */`@group(0) @binding(4) var<storage, read> sdiff: array<f32>; // [${H},${W}]
fn dtap(x: i32, y: i32) -> f32 {
  return sdiff[clamp(y, 0, ${H - 1}) * ${W} + clamp(x, 0, ${W - 1})];
}` : ''}
@group(1) @binding(0) var tex0: texture_2d<f32>;
@group(1) @binding(1) var tex1: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;
// grid_sample bilinear/border via the sampler: clamp-to-edge + hw filtering
fn warpT(t: texture_2d<f32>, sx: f32, sy: f32) -> vec3<f32> {
  let uv = (vec2<f32>(sx, sy) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
  return textureSampleLevel(t, samp, uv, 0.0).bgr; // b,g,r like the buffer path
}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= ${W} || y >= ${H}) { return; }
  let uv8 = (vec2<f32>(f32(x), f32(y)) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
  let fl = textureSampleLevel(t8f, samp, uv8, 0.0) * 8.0;
  let m = 1.0 / (1.0 + exp(-textureSampleLevel(t8m, samp, uv8, 0.0).x));
  let w0 = warpT(tex0, f32(x) + fl.x, f32(y) + fl.y);
  let w1 = warpT(tex1, f32(x) + fl.z, f32(y) + fl.w);
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
function wgslRefinePrep(W, H, f16, sparse) {
  const HW = W / 4, HH = H / 4;
  const TXT = Math.ceil(HW / 8);
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var t8f: texture_2d<f32>;  // flow/8 (see deconv texOut)
@group(0) @binding(1) var t8m: texture_2d<f32>;  // mask logit
@group(0) @binding(2) var<storage, read_write> rin: array<${T}>; // [11,${HH},${HW}]
${sparse ? `@group(0) @binding(3) var<storage, read_write> tstat: array<atomic<u32>>; // [tiles] max warp disagreement (f32 bits)` : ''}
@group(1) @binding(0) var tex0: texture_2d<f32>;
@group(1) @binding(1) var tex1: texture_2d<f32>;
@group(1) @binding(2) var samp: sampler;

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
      let uvS = (vec2<f32>(f32(X), f32(Y)) + 0.5) / vec2<f32>(${W}.0, ${H}.0);
      let f4 = textureSampleLevel(t8f, samp, uvS, 0.0) * 8.0;
      mk += 1.0 / (1.0 + exp(-textureSampleLevel(t8m, samp, uvS, 0.0).x));
      w0 += warpT(tex0, f32(X) + f4.x, f32(Y) + f4.y);
      w1 += warpT(tex1, f32(X) + f4.z, f32(Y) + f4.w);
      fl += f4;
    }
  }
${sparse ? /* wgsl */`
  // occlusion signal for the sparse-refine tile mask: where the two warps agree,
  // the refine residual is ~0 and the whole tile can be skipped downstream
  let dis = abs(w0 - w1) * 0.25;
  let occ = max(dis.x, max(dis.y, dis.z));
  atomicMax(&tstat[(hy / 8) * ${TXT} + hx / 8], bitcast<u32>(occ)); // occ >= 0: u32 order == f32 order
` : ''}
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

// sparse-refine tile scheduler: thresholds the per-tile disagreement stats,
// dilates the active set per chained conv layer (halo validity: rout on A needs
// rc2 on A(+)1 tile, rc2 needs rc1 on A(+)2, rc1 needs rc0 on A(+)3), appends
// tile ids to per-layer lists and writes the dispatchWorkgroupsIndirect args -
// the CPU never sees any of it. Empty scene => refine convs dispatch ZERO groups.
function wgslRefineTiles(T, TXT, TYT, zConv, thr) {
  // lists layout: [4][T] tile ids; ind layout: [4][4] u32 dispatch args (x,y,z,pad)
  // l=0 -> rc0 (radius 3), l=1 -> rc1 (r2), l=2 -> rc2 (r1), l=3 -> rout (r0)
  return /* wgsl */`
@group(0) @binding(0) var<storage, read> tstat: array<u32>; // f32 bits, >= 0
@group(0) @binding(1) var<storage, read_write> lists: array<u32>;
@group(0) @binding(2) var<storage, read_write> ind: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = i32(gid.x);
  if (t < 4) { // y/z of the four indirect dispatches (x is the atomic count)
    atomicStore(&ind[u32(t) * 4u + 1u], 1u);
    atomicStore(&ind[u32(t) * 4u + 2u], select(${zConv}u, 1u, t == 3));
  }
  if (t >= ${T}) { return; }
  let tx = t % ${TXT}; let ty = t / ${TXT};
  const THR: u32 = ${'0x' + new Uint32Array(new Float32Array([thr]).buffer)[0].toString(16)}u; // bitcast<u32>(${thr}f)
  // widest neighborhood once; shrink per layer (r = 3,2,1,0)
  for (var l = 0; l < 4; l++) {
    let r = 3 - l;
    var act = false;
    for (var dy = -r; dy <= r; dy++) {
      for (var dx = -r; dx <= r; dx++) {
        let nx = tx + dx; let ny = ty + dy;
        if (nx < 0 || nx >= ${TXT} || ny < 0 || ny >= ${TYT}) { continue; }
        act = act || (tstat[ny * ${TXT} + nx] > THR);
      }
    }
    if (act) {
      let i = atomicAdd(&ind[u32(l) * 4u], 1u);
      lists[u32(l) * ${T}u + i] = u32(t);
    }
  }
}`;
}

// final refine conv (C->3) + sigmoid*2-1 residual into an rgba16float texture
// (flowout samples it with hw bilinear). One pass over ci feeds all 3 output
// channels - the per-co outer loop re-read the whole window 3x.
function wgslRefineOut(C, HW, HH, f16, sparse) {
  const T = f16 ? 'f16' : 'f32';
  return /* wgsl */`
${f16 ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;  // [${C},${HH},${HW}]
@group(0) @binding(1) var<storage, read> wgt: array<f32>;   // [3,${C},3,3]
@group(0) @binding(2) var<storage, read> bias: array<f32>;  // [3]
@group(0) @binding(3) var outT: texture_storage_2d<rgba16float, write>;
${sparse ? `@group(0) @binding(4) var<storage, read> tiles: array<u32>;` : ``}

@compute @workgroup_size(${WG}, ${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
${sparse ? /* wgsl */`
  let tid = i32(tiles[${sparse.lbase} + i32(wid.x)]);
  let x = (tid % ${sparse.txt}) * ${WG} + i32(lid.x);
  let y = (tid / ${sparse.txt}) * ${WG} + i32(lid.y);
` : `
  let x = i32(gid.x); let y = i32(gid.y);
`}
  if (x >= ${HW} || y >= ${HH}) { return; }
  var a0 = bias[0]; var a1 = bias[1]; var a2 = bias[2];
  for (var ci = 0; ci < ${C}; ci++) {
    let sb = ci * ${HH * HW};
    let wb = ci * 9;
    for (var ky = 0; ky < 3; ky++) {
      let sy = clamp(y + ky - 1, 0, ${HH - 1});
      for (var kx = 0; kx < 3; kx++) {
        let sx = clamp(x + kx - 1, 0, ${HW - 1});
        let sv = f32(src[sb + sy * ${HW} + sx]);
        let wk = wb + ky * 3 + kx;
        a0 += sv * wgt[wk];
        a1 += sv * wgt[${C * 9} + wk];
        a2 += sv * wgt[${2 * C * 9} + wk];
      }
    }
  }
  textureStore(outT, vec2<i32>(x, y), vec4<f32>(
    2.0 / (1.0 + exp(-a0)) - 1.0,
    2.0 / (1.0 + exp(-a1)) - 1.0,
    2.0 / (1.0 + exp(-a2)) - 1.0, 0.0));
}`;
}

// one-shot f16 -> f32 widening (w4 kernels read pre-widened weights: the CVT
// leaves the hot loop, values stay bit-exact f32(f16(w)))
export const WGSL_TO_F32 = /* wgsl */`
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&src)) { dst[i] = f32(src[i]); }
}`;

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
// v2: one thread computes ALL CO channels of its pixel - the old gid.z fan-out
// re-read the same 4-tap src window CO times. Weights stage through workgroup
// memory (every thread consumes the identical sequence; the whole tensor is
// CI*CO*16 elems - slabbed only if it outgrows the 16KB budget). Stride-2
// parity picks the 2x2 valid taps of the 4x4 kernel up front. Accumulation
// order matches v1 (ky, kx ascending, ci inner) - bit-exact when CI fits one slab.
function wgslDeconv(CI, CO, IW, IH, OW, OH, f16src, texOut) {
  // texOut (direct mode): flow (a0..a3) and mask (a4) store into rgba16float
  // textures so flowout/rprep upsample through the hardware bilinear unit
  const T = f16src ? 'f16' : 'f32';
  const SLAB = Math.min(CI, Math.floor(16384 / (CO * 16 * (f16src ? 2 : 4))));
  const accs = Array.from({ length: CO }, (_, c) => c);
  return /* wgsl */`
${f16src ? 'enable f16;' : ''}
@group(0) @binding(0) var<storage, read> src: array<${T}>;
@group(0) @binding(1) var<storage, read> wgt: array<${T}>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
${texOut ? `@group(0) @binding(3) var outFlow: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var outMask: texture_storage_2d<rgba16float, write>;`
: `@group(0) @binding(3) var<storage, read_write> dst: array<f32>;`}

var<workgroup> wsh: array<${T}, ${SLAB * CO * 16}>; // [slab, CO, 4, 4] contiguous

@compute @workgroup_size(${WG}, ${WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let x = i32(gid.x); let y = i32(gid.y);
  let inb = x < ${OW} && y < ${OH};
${accs.map(c => `  var a${c} = bias[${c}];`).join('\n')}
  // ty = y+1-ky must be even and >= 0: ky ranges over {py, py+2}, py = (y+1)&1
  let py = (y + 1) & 1; let px = (x + 1) & 1;
  for (var s = 0; s < ${CI}; s += ${SLAB}) {
    let sl = min(${SLAB}, ${CI} - s);
    let n = sl * ${CO * 16};
    workgroupBarrier();
    var idx = i32(li);
    while (idx < n) { wsh[idx] = wgt[s * ${CO * 16} + idx]; idx += ${WG * WG}; }
    workgroupBarrier();
    if (inb) {
      for (var j = 0; j < 2; j++) {
        let ky = py + j * 2;
        let ty = y + 1 - ky;
        let iy = ty / 2;
        if (ty >= 0 && iy < ${IH}) {
          for (var i = 0; i < 2; i++) {
            let kx = px + i * 2;
            let tx = x + 1 - kx;
            let ix = tx / 2;
            if (tx >= 0 && ix < ${IW}) {
              let kb = ky * 4 + kx;
              for (var ci = 0; ci < sl; ci++) {
                let sv = f32(src[(s + ci) * ${IH * IW} + iy * ${IW} + ix]);
                let wb = ci * ${CO * 16} + kb;
${accs.map(c => `                a${c} += sv * f32(wsh[wb + ${c * 16}]);`).join('\n')}
              }
            }
          }
        }
      }
    }
  }
  if (!inb) { return; }
${texOut ? `  textureStore(outFlow, vec2<i32>(x, y), vec4<f32>(a0, a1, a2, a3));
  textureStore(outMask, vec2<i32>(x, y), vec4<f32>(a4, 0.0, 0.0, 0.0));`
: `  let o = y * ${OW} + x;
${accs.map(c => `  dst[${c} * ${OH * OW} + o] = a${c};`).join('\n')}`}
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
                                          staticGuard = false,
                                          sparseRefine = true, refineThr = 0.02 }) {
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
  const needW4 = useF16 && convTune && convTune.w4;
  // conv0b's stride-2 shape is tuned separately (convTune.s2); tunes persisted
  // before the s2 sweep existed fall back to mirroring the cb w4 flag
  const s2Tune = useF16 ? ((convTune && convTune.s2) || (needW4 ? { w4: true } : null)) : null;
  const s2W4 = !!(s2Tune && s2Tune.w4);
  const pToF32 = (needW4 || s2W4) ? pipe(WGSL_TO_F32) : null;
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
    // the f32 original is never bound again - release it (destroy is deferred
    // past the submitted conversion by the spec). ~3MB (v7s) to 18MB (full) VRAM.
    wbuf[name].destroy();
    wbuf[name] = null;
    return half;
  };
  // convblock weights for w4 kernels: widen the f16 copy back to f32 (bit-exact
  // f32(f16(w)) - the same values the legacy kernel computes with, minus the
  // per-use CVT). The f16 intermediate is released once widened.
  const cbW = (name, widen = needW4) => {
    const half = convW(name);
    if (!widen) return half;
    const n = man[name].shape.reduce((a, b) => a * b, 1);
    const wide = bufBytes(n * 4);
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pToF32);
    pass.setBindGroup(0, bg(pToF32, [half, wide]));
    pass.dispatchWorkgroups(Math.ceil(n / 256));
    pass.end();
    device.queue.submit([enc.finish()]);
    half.destroy();
    return wide;
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
  const tmp8 = direct ? null : buf(5 * H8 * W8);
  // direct mode: deconv output lives in two filterable rgba16float textures
  const mkT8 = () => device.createTexture({ size: [W8, H8], format: 'rgba16float',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
  const t8f = direct ? mkT8() : null, t8m = direct ? mkT8() : null;
  const t8fV = direct ? t8f.createView() : null, t8mV = direct ? t8m.createView() : null;
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
  const [pPrepFull, pPrepQ, pConv0a, pConv0b, pConvB, pConvBR, pDeconv, pFlow, pDiff, pConvB6] = await Promise.all([
    direct ? null : pipeAsync(textureInput ? wgslPrepFullTex(w, h) : wgslPrepFull(w, h)),
    pipeAsync(tfact ? wgslPrepQuarterTex6(w, h, useF16)
      : (textureInput ? wgslPrepQuarterTex(w, h, useF16) : wgslPrepQuarter(w, h, useF16))),
    pipeAsync(wgslConv(CI0, C1, QW, QH, W8, H8, 2, false, useF16)),
    pipeAsync(useF16 ? wgslConvRBs2(C1, C2, W8, H8, W16, H16, s2Tune)
                     : wgslConv(C1, C2, W8, H8, W16, H16, 2, false, false)),
    pipeAsync(useF16
      ? (sgTuned ? wgslConvRBSg : wgslConvRB)(C2, C2, W16, H16, W16, H16, false, convTune)
      : wgslConv(C2, C2, W16, H16, W16, H16, 1, false, false)),
    pipeAsync(useF16
      ? (sgTuned ? wgslConvRBSg : wgslConvRB)(C2, C2, W16, H16, W16, H16, true, convTune)
      : wgslConv(C2, C2, W16, H16, W16, H16, 1, true, false)),
    pipeAsync(wgslDeconv(C2, 5, W16, H16, W8, H8, useF16, direct)),
    pipeAsync(textureOutput
      ? (direct ? wgslFlowOutTexDirect(w, h, staticGuard, refi) : wgslFlowOutTex(w, h, staticGuard, refi))
      : wgslFlowOut(w, h)),
    sdiff ? pipeAsync(wgslDiff(w, h)) : null,
    // tfact cb6 with the FiLM affine fused into its tile load (f16 path only;
    // the non-f16 fallback keeps the standalone film pass)
    tfact && useF16
      ? pipeAsync((sgTuned ? wgslConvRBSg : wgslConvRB)(C2, C2, W16, H16, W16, H16, false, convTune, true))
      : null,
  ]);
  // texture-output mode: flow bind groups are per output texture (small ring - cache them)
  const flowBgCache = new Map();
  function flowBgFor(tex) {
    if (!flowBgCache.has(tex)) {
      // evict BEFORE inserting - clearing after would wipe the fresh entry and
      // hand setBindGroup an undefined (latent until a caller rings >24 textures)
      if (flowBgCache.size > 24) flowBgCache.clear();
      // direct layout: 0 t8f, 1 t8m, 2 outTex, 3 resT?, 4 sdiff? (sources ride group 1);
      // legacy layout: 0 tmp8, 1 imgs, 2 outTex, 3 res?
      const entries = direct
        ? [{ binding: 0, resource: t8fV },
           { binding: 1, resource: t8mV },
           { binding: 2, resource: tex.createView() }]
        : [{ binding: 0, resource: { buffer: tmp8 } },
           { binding: 1, resource: { buffer: imgs } },
           { binding: 2, resource: tex.createView() }];
      if (refi) entries.push(direct ? { binding: 3, resource: rResV } : { binding: 3, resource: { buffer: rRes } });
      if (sdiff) entries.push({ binding: 4, resource: { buffer: sdiff } });
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

  // tfact-only state: trunk feature buffer + FiLM params (tiny t-MLP runs in JS).
  // With f16 the FiLM affine is fused into cb6's tile load (pConvB6) - the
  // standalone film pass exists only on the non-f16 fallback.
  const hbuf = tfact ? bufBytes(C2 * H16 * W16 * (useF16 ? 2 : 4)) : null;
  const filmBuf = tfact ? buf(2 * C2) : null;
  const pFilm = tfact && !useF16 ? pipe(wgslFilm(C2, H16 * W16, useF16)) : null;
  let bgFilm = null, filmW = null;
  if (tfact) {
    if (pFilm) bgFilm = bg(pFilm, [hbuf, filmBuf, f16b]); // cb6 reads f16b in the new chain
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
  let pRPrep = null, pRC0 = null, pRC1 = null, pRC2 = null, pROut = null, pRTiles = null;
  let bgRPrep = null, bgRC0 = null, bgRC1 = null, bgRC2 = null, bgROut = null, bgRTiles = null;
  let RC = 0, RCOC = 4, rRes = null, rResT = null, rResV = null, rStat = null, rInd = null;
  // sparse refine: the occlusion-repair head only matters where the two warps
  // DISAGREE. rprep tags 8x8 tiles by max disagreement, a tiny scheduler pass
  // thresholds + dilates the set (halo validity per chained conv) and writes the
  // dispatchWorkgroupsIndirect args - the conv chain then runs on active tiles
  // only, all GPU-side. Calm scene => the refine convs dispatch ZERO workgroups.
  // rRes is cleared per mid, so skipped tiles get residual 0 == "no repair".
  const rSparse = refi && sparseRefine;
  const RTXT = Math.ceil(RW4 / 8), RTYT = Math.ceil(RH4 / 8), RTT = RTXT * RTYT;
  if (refi) {
    RC = man['refine.c0.weight'].shape[0];
    RCOC = RC % 8 === 0 ? 8 : 4; // wider block halves tile re-staging
    // the refine convs are dispatched with z = RC/4 below, and wgslConv only
    // packs 4-wide when CO % 4 == 0 - a non-/4 head would silently mis-dispatch
    if (RC % 4 !== 0) throw new Error('rt: refine channels must be a multiple of 4, got ' + RC);
    const rIn = abuf(11 * RH4 * RW4);
    const rA2 = abuf(RC * RH4 * RW4);
    const rB2 = abuf(RC * RH4 * RW4);
    rResT = device.createTexture({ size: [RW4, RH4], format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT });
    rResV = rResT.createView();
    const sp = (l) => rSparse ? { lbase: l * RTT, txt: RTXT } : null;
    if (rSparse) {
      rStat = buf(RTT);
      rInd = device.createBuffer({ size: 64, // [4][x,y,z,pad] dispatch args
        usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    }
    [pRPrep, pRC0, pRC1, pRC2, pROut, pRTiles] = await Promise.all([
      pipeAsync(wgslRefinePrep(w, h, useF16, rSparse)),
      pipeAsync(wgslConv(11, RC, RW4, RH4, RW4, RH4, 1, false, useF16, sp(0), RCOC)),
      pipeAsync(wgslConv(RC, RC, RW4, RH4, RW4, RH4, 1, false, useF16, sp(1), RCOC)),
      rSparse ? pipeAsync(wgslConv(RC, RC, RW4, RH4, RW4, RH4, 1, false, useF16, sp(2), RCOC)) : null,
      pipeAsync(wgslRefineOut(RC, RW4, RH4, useF16, sp(3))),
      rSparse ? pipeAsync(wgslRefineTiles(RTT, RTXT, RTYT, RC / RCOC, refineThr)) : null,
    ]);
    const rLists = rSparse ? buf(4 * RTT) : null;
    { // rprep reads the deconv textures; sources ride group(1) per pair (rprepTex)
      const e = [{ binding: 0, resource: t8fV }, { binding: 1, resource: t8mV },
                 { binding: 2, resource: { buffer: rIn } }];
      if (rSparse) e.push({ binding: 3, resource: { buffer: rStat } });
      bgRPrep = device.createBindGroup({ layout: pRPrep.getBindGroupLayout(0), entries: e });
    }
    const spb = (l) => rSparse ? [rLists] : [];
    bgRC0 = bg(pRC0, [rIn, convW('refine.c0.weight'), wbuf['refine.c0.bias'], wbuf['refine.a0.weight'], rA2, ...spb(0)]);
    bgRC1 = bg(pRC1, [rA2, convW('refine.c1.weight'), wbuf['refine.c1.bias'], wbuf['refine.a1.weight'], rB2, ...spb(1)]);
    bgRC2 = bg(rSparse ? pRC2 : pRC1, [rB2, convW('refine.c2.weight'), wbuf['refine.c2.bias'], wbuf['refine.a2.weight'], rA2, ...spb(2)]);
    { // rout stores the residual into the rgba16float texture
      const e = [{ binding: 0, resource: { buffer: rA2 } },
                 { binding: 1, resource: { buffer: wbuf['refine.c3.weight'] } },
                 { binding: 2, resource: { buffer: wbuf['refine.c3.bias'] } },
                 { binding: 3, resource: rResV }];
      if (rSparse) e.push({ binding: 4, resource: { buffer: rLists } });
      bgROut = device.createBindGroup({ layout: pROut.getBindGroupLayout(0), entries: e });
    }
    if (rSparse) bgRTiles = bg(pRTiles, [rStat, rLists, rInd]);
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
  // conv0b lands in f16r directly: the residual is BORN there instead of being
  // snapshotted with a per-pair copy; cb7 reads it untouched (the a/b ping-pong
  // never writes f16r). Likewise tfact's cb5 stashes the trunk straight into
  // hbuf - both copyBufferToBuffer round trips of the old wiring are gone.
  // w4 conv0b reads pre-widened f32 weights (cbW == convW when s2 w4 is off)
  const bgConv0b = bg(pConv0b, [f8, cbW('block0.conv0.1.0.weight', s2W4), wbuf['block0.conv0.1.0.bias'], wbuf['block0.conv0.1.1.weight'], f16r]);
  const bgB = [];
  {
    let cbSrc = f16r, cbDst = f16a;
    for (let i = 0; i < 8; i++) {
      const wn = `block0.convblock.${i}.0.weight`, bn = `block0.convblock.${i}.0.bias`, an = `block0.convblock.${i}.1.weight`;
      let p = i === 7 ? pConvBR : pConvB;
      let extra = i === 7 ? [f16r] : [];
      let s2 = cbSrc, d2 = cbDst;
      if (tfact && i === 5) d2 = hbuf;             // trunk stash
      if (tfact && i === 6) {
        if (useF16) { s2 = hbuf; p = pConvB6; extra = [filmBuf]; } // FiLM fused on load
        else { s2 = f16b; }                        // fallback: film pass fills f16b
      }
      bgB.push({ p, g: bg(p, [s2, cbW(wn), wbuf[bn], wbuf[an], d2, ...extra]) });
      if (i === 0) { cbSrc = f16a; cbDst = f16b; }
      else { [cbSrc, cbDst] = [cbDst, cbSrc]; }
    }
  }
  const f16out = f16b; // both chains end in f16b (see trace above)
  const bgDeconv = direct
    ? device.createBindGroup({ layout: pDeconv.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: f16out } },
        { binding: 1, resource: { buffer: convW('block0.lastconv.weight') } },
        { binding: 2, resource: { buffer: wbuf['block0.lastconv.bias'] } },
        { binding: 3, resource: t8fV }, { binding: 4, resource: t8mV }] })
    : bg(pDeconv, [f16out, convW('block0.lastconv.weight'), wbuf['block0.lastconv.bias'], tmp8]);
  const bgFlow = textureOutput ? null : bg(pFlow, [tmp8, imgs, outp]);

  const gx = (n) => Math.ceil(n / WG);
  // register-blocked convblock kernel covers a (2*wgx)x(2*wgy) output tile per wg
  const cbTX = ((convTune && convTune.wgx) || 8) * 2;
  const cbTY = ((convTune && convTune.wgy) || 8) * 2;
  const cbX = useF16 ? Math.ceil(W16 / cbTX) : gx(W16);
  const cbY = useF16 ? Math.ceil(H16 / cbTY) : gx(H16);
  const cbZ = useF16 ? C2 / ((convTune && convTune.coc) || 4) : C2 / 4;
  // stride-2 RB conv0b covers a 16x16 output tile per wg; COC comes from the tune
  const c0bX = useF16 ? Math.ceil(W16 / 16) : gx(W16);
  const c0bY = useF16 ? Math.ceil(H16 / 16) : gx(H16);
  const c0bZ = useF16 ? C2 / ((s2Tune && s2Tune.coc) || (C2 % 8 === 0 ? 8 : 4)) : C2 / 4;

  // per-stage GPU times via timestamp queries (needs 'timestamp-query' on the device)
  async function profile(rgbaA, rgbaB) {
    if (textureInput) return 'profile: buffer-input mode only';
    if (!device.features.has('timestamp-query')) return 'no timestamp-query feature';
    const stages = [
      ['prepFull', pPrepFull, bgPrepFull, [gx(w), gx(h), 1]],
      ['prepQ', pPrepQ, bgPrepQ, [gx(QW), gx(QH), 1]],
      ['conv0a', pConv0a, bgConv0a, [gx(W8), gx(H8), Z0A]],
      ['conv0b', pConv0b, bgConv0b, [c0bX, c0bY, c0bZ]],
      ...bgB.map(({ p, g }, i) => [`convB${i}`, p, g, [cbX, cbY, cbZ]]),
      ['deconv', pDeconv, bgDeconv, [gx(W8), gx(H8), 1]],
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

  // per-stage GPU times for the tfact TEXTURE path (the extension's real graph) -
  // one pass per stage so timestamps bracket each dispatch. Timing only: the
  // extra pass splits change barrier behavior vs production single-pass encoding.
  async function profileT(texA, texB, t = 0.5, outTex = null) {
    if (!tfact) return 'profileT: tfact weights only';
    if (!device.features.has('timestamp-query')) return 'no timestamp-query feature';
    const tp = texPrepBgs(texA, texB);
    device.queue.writeBuffer(filmBuf, 0, filmParams(t));
    const stages = [];
    const st = (name, fn) => stages.push([name, fn]);
    if (sdiff) st('diff', (p) => { p.setPipeline(pDiff); p.setBindGroup(0, tp.diff); p.dispatchWorkgroups(gx(w), gx(h)); });
    st('prepQ', (p) => { p.setPipeline(pPrepQ); p.setBindGroup(0, tp.q[0]); p.dispatchWorkgroups(gx(QW), gx(QH)); });
    st('conv0a', (p) => { p.setPipeline(pConv0a); p.setBindGroup(0, bgConv0a); p.dispatchWorkgroups(gx(W8), gx(H8), Z0A); });
    st('conv0b', (p) => { p.setPipeline(pConv0b); p.setBindGroup(0, bgConv0b); p.dispatchWorkgroups(c0bX, c0bY, c0bZ); });
    for (let i = 0; i < 6; i++) {
      st(`cb${i}`, (p) => { p.setPipeline(bgB[i].p); p.setBindGroup(0, bgB[i].g); p.dispatchWorkgroups(cbX, cbY, cbZ); });
    }
    if (pFilm) st('film', (p) => { p.setPipeline(pFilm); p.setBindGroup(0, bgFilm); p.dispatchWorkgroups(Math.ceil((C2 * H16 * W16) / 256)); });
    st('cb6', (p) => { p.setPipeline(bgB[6].p); p.setBindGroup(0, bgB[6].g); p.dispatchWorkgroups(cbX, cbY, cbZ); });
    st('cb7', (p) => { p.setPipeline(bgB[7].p); p.setBindGroup(0, bgB[7].g); p.dispatchWorkgroups(cbX, cbY, cbZ); });
    st('deconv', (p) => { p.setPipeline(pDeconv); p.setBindGroup(0, bgDeconv); p.dispatchWorkgroups(gx(W8), gx(H8)); });
    if (refi) {
      st('rprep', (p) => { p.setPipeline(pRPrep); p.setBindGroup(0, bgRPrep); p.setBindGroup(1, tp.rprepTex); p.dispatchWorkgroups(gx(RW4), gx(RH4)); });
      if (rSparse) {
        st('rtiles', (p) => { p.setPipeline(pRTiles); p.setBindGroup(0, bgRTiles); p.dispatchWorkgroups(Math.ceil(RTT / 256)); });
        st('rc0', (p) => { p.setPipeline(pRC0); p.setBindGroup(0, bgRC0); p.dispatchWorkgroupsIndirect(rInd, 0); });
        st('rc1', (p) => { p.setPipeline(pRC1); p.setBindGroup(0, bgRC1); p.dispatchWorkgroupsIndirect(rInd, 16); });
        st('rc2', (p) => { p.setPipeline(pRC2); p.setBindGroup(0, bgRC2); p.dispatchWorkgroupsIndirect(rInd, 32); });
        st('rout', (p) => { p.setPipeline(pROut); p.setBindGroup(0, bgROut); p.dispatchWorkgroupsIndirect(rInd, 48); });
      } else {
        st('rc0', (p) => { p.setPipeline(pRC0); p.setBindGroup(0, bgRC0); p.dispatchWorkgroups(gx(RW4), gx(RH4), RC / RCOC); });
        st('rc1', (p) => { p.setPipeline(pRC1); p.setBindGroup(0, bgRC1); p.dispatchWorkgroups(gx(RW4), gx(RH4), RC / RCOC); });
        st('rc2', (p) => { p.setPipeline(pRC1); p.setBindGroup(0, bgRC2); p.dispatchWorkgroups(gx(RW4), gx(RH4), RC / RCOC); });
        st('rout', (p) => { p.setPipeline(pROut); p.setBindGroup(0, bgROut); p.dispatchWorkgroups(gx(RW4), gx(RH4)); });
      }
    }
    if (outTex) {
      st('flow', (p) => { p.setPipeline(pFlow); p.setBindGroup(0, flowBgFor(outTex)); p.setBindGroup(1, tp.flowTex); p.dispatchWorkgroups(gx(w), gx(h)); });
    }
    const qs = device.createQuerySet({ type: 'timestamp', count: stages.length * 2 });
    const qbuf = device.createBuffer({ size: stages.length * 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const qread = device.createBuffer({ size: stages.length * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    stages.forEach(([name, fn], i) => {
      if (name === 'rprep' && rSparse) {
        enc.clearBuffer(rStat); enc.clearBuffer(rInd);
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: rResV,
          loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
        rp.end();
      }
      const pass = enc.beginComputePass({ timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 } });
      fn(pass);
      pass.end();
    });
    enc.resolveQuerySet(qs, 0, stages.length * 2, qbuf, 0);
    enc.copyBufferToBuffer(qbuf, 0, qread, 0, stages.length * 16);
    device.queue.submit([enc.finish()]);
    await qread.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(qread.getMappedRange().slice(0));
    qread.unmap();
    qs.destroy(); qbuf.destroy(); qread.destroy();
    return stages.map(([name], i) =>
      `${name}: ${(Number(ts[i * 2 + 1] - ts[i * 2]) / 1e6).toFixed(3)}ms`).join(' · ');
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
    pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(c0bX, c0bY, c0bZ);
    pass.end();
    const pass2 = enc.beginComputePass();
    for (const { p, g } of bgB) {
      pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
    }
    pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8));
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
      pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(c0bX, c0bY, c0bZ);
      pass.end();
      const pass2 = enc.beginComputePass();
      for (const { p, g } of bgB) {
        pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
      }
      pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8));
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
      pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(c0bX, c0bY, c0bZ);
      // no residual/trunk copies: conv0b wrote f16r directly, cb5 writes hbuf directly
      for (let i = 0; i < 6; i++) {
        pass.setPipeline(bgB[i].p); pass.setBindGroup(0, bgB[i].g); pass.dispatchWorkgroups(cbX, cbY, cbZ);
      }
      pass.end();
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
      if (rSparse) { // stats/args reset per mid; the residual texture clears via
        // the render-pass fast path (the old clearBuffer moved 691KB per mid)
        enc.clearBuffer(rStat); enc.clearBuffer(rInd);
        const rp = enc.beginRenderPass({ colorAttachments: [{ view: rResV,
          loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
        rp.end();
      }
      const pass = enc.beginComputePass();
      if (pFilm) { // non-f16 fallback; with f16 the FiLM affine rides cb6's tile load
        pass.setPipeline(pFilm); pass.setBindGroup(0, bgFilm);
        pass.dispatchWorkgroups(Math.ceil((C2 * H16 * W16) / 256));
      }
      pass.setPipeline(bgB[6].p); pass.setBindGroup(0, bgB[6].g); pass.dispatchWorkgroups(cbX, cbY, cbZ);
      pass.setPipeline(bgB[7].p); pass.setBindGroup(0, bgB[7].g); pass.dispatchWorkgroups(cbX, cbY, cbZ);
      pass.setPipeline(pDeconv); pass.setBindGroup(0, bgDeconv); pass.dispatchWorkgroups(gx(W8), gx(H8));
      if (refi) { // quarter-res refine chain; the flowout below folds the residual in
        pass.setPipeline(pRPrep); pass.setBindGroup(0, bgRPrep);
        pass.setBindGroup(1, curPrep.rprepTex); // source textures for the direct warp
        pass.dispatchWorkgroups(gx(RW4), gx(RH4));
        if (rSparse) { // GPU-scheduled: convs run on active tiles only
          pass.setPipeline(pRTiles); pass.setBindGroup(0, bgRTiles);
          pass.dispatchWorkgroups(Math.ceil(RTT / 256));
          pass.setPipeline(pRC0); pass.setBindGroup(0, bgRC0); pass.dispatchWorkgroupsIndirect(rInd, 0);
          pass.setPipeline(pRC1); pass.setBindGroup(0, bgRC1); pass.dispatchWorkgroupsIndirect(rInd, 16);
          pass.setPipeline(pRC2); pass.setBindGroup(0, bgRC2); pass.dispatchWorkgroupsIndirect(rInd, 32);
          pass.setPipeline(pROut); pass.setBindGroup(0, bgROut); pass.dispatchWorkgroupsIndirect(rInd, 48);
        } else {
          pass.setPipeline(pRC0); pass.setBindGroup(0, bgRC0); pass.dispatchWorkgroups(gx(RW4), gx(RH4), RC / RCOC);
          pass.setPipeline(pRC1); pass.setBindGroup(0, bgRC1); pass.dispatchWorkgroups(gx(RW4), gx(RH4), RC / RCOC);
          pass.setPipeline(pRC1); pass.setBindGroup(0, bgRC2); pass.dispatchWorkgroups(gx(RW4), gx(RH4), RC / RCOC);
          pass.setPipeline(pROut); pass.setBindGroup(0, bgROut); pass.dispatchWorkgroups(gx(RW4), gx(RH4));
        }
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
    pass.setPipeline(pConv0b); pass.setBindGroup(0, bgConv0b); pass.dispatchWorkgroups(c0bX, c0bY, c0bZ);
    pass.end();
    const pass2 = enc.beginComputePass();
    for (const { p, g } of bgB) {
      pass2.setPipeline(p); pass2.setBindGroup(0, g); pass2.dispatchWorkgroups(cbX, cbY, cbZ);
    }
    pass2.setPipeline(pDeconv); pass2.setBindGroup(0, bgDeconv); pass2.dispatchWorkgroups(gx(W8), gx(H8));
    pass2.setPipeline(pFlow); pass2.setBindGroup(0, flowBgFor(outTex));
    pass2.setBindGroup(1, curPrep.flowTex); // direct warp sources (runT implies texture in+out)
    pass2.dispatchWorkgroups(gx(w), gx(h));
    pass2.end();
    device.queue.submit([enc.finish()]);
  }

  return { run, runMulti, prepPair, runT, profile, profileT, w, h };
}


// ---- one-shot conv autotune: bench wgslConvRB variants on this device ----
// Returns the fastest {coc, slab, ms}. ~200-400ms of GPU time; call it once per
// (adapter, model width, resolution) and persist the answer - relative ranking
// is what matters, so light background load is tolerable.
export async function tuneConvRB(device, { ci, co, w16, h16, s2ci }) {
  const shared = (v) => {
    const ts = ((v.wgx || 8) * 2 + 2) * ((v.wgy || 8) * 2 + 2);
    return v.slab * ts * 2 + (v.sg ? 0 : v.coc * v.slab * 9 * (v.w4 ? 4 : 2));
  };
  const base = [
    { coc: 4, slab: 20 }, { coc: 8, slab: 20 }, { coc: 8, slab: 12 }, { coc: 4, slab: 12 },
    // wider workgroups (128/256 threads): the 64-thread shape can bottom out at
    // ~12% SM occupancy (shared-limited) - these trade tile reuse for latency hiding
    { coc: 4, slab: 12, wgx: 16, wgy: 8 }, { coc: 8, slab: 8, wgx: 16, wgy: 8 },
    { coc: 8, slab: 10, wgx: 16, wgy: 8 }, { coc: 4, slab: 6, wgx: 16, wgy: 16 },
    // coc16: halves the z-slices re-staging the same tile - on Ada it beat coc8
    // by 12% at 1080p (0.408 -> 0.353 isolated); coc24 spills registers, skip
    { coc: 16, slab: 8 }, { coc: 16, slab: 8, wgx: 16, wgy: 8 },
  ].filter(v => co % v.coc === 0 && shared(v) <= 16384);
  // subgroup variants (weights via subgroupBroadcastFirst, no shared staging):
  // measured +20% at the 360p grid and -19% at 720p/coc8 on a 4060 Ti - exactly
  // why they go through the tuner instead of being hardcoded
  const sgList = device.features.has('subgroups')
    ? [...base, ...base.map(v => ({ ...v, sg: true })).filter(v => shared(v) <= 16384)]
    : base;
  // w4: 4x4 window in registers + f32 weights (CVT leaves the hot loop);
  // v2 adds vec2<f16> tile loads. Measured per GPU, never assumed.
  const w4List = sgList.map(v => ({ ...v, w4: true })).filter(v => shared(v) <= 16384);
  const variants = [...sgList, ...w4List, ...w4List.map(v => ({ ...v, v2: true }))];
  const buf = (bytes) => device.createBuffer({ size: Math.ceil(bytes / 4) * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const src = buf(ci * w16 * h16 * 2), dst = buf(co * w16 * h16 * 2);
  const wgt = buf(co * ci * 9 * 2), wgt32 = buf(co * ci * 9 * 4);
  const bias = buf(co * 4), alpha = buf(co * 4);
  // compile ALL variants async up front (Dawn's worker pool) - serial sync
  // compiles blocked the calling thread for the whole calibration
  const pipes = await Promise.all(variants.map(v => {
    const gen = v.sg ? wgslConvRBSg : wgslConvRB;
    return device.createComputePipelineAsync({ layout: 'auto', compute: {
      module: device.createShaderModule({ code: gen(ci, co, w16, h16, w16, h16, false, v) }),
      entryPoint: 'main' } });
  }));
  let best = null;
  for (let vi = 0; vi < variants.length; vi++) {
    const v = variants[vi];
    const p = pipes[vi];
    const bg = device.createBindGroup({ layout: p.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: src } },
      { binding: 1, resource: { buffer: v.w4 ? wgt32 : wgt } },
      { binding: 2, resource: { buffer: bias } }, { binding: 3, resource: { buffer: alpha } },
      { binding: 4, resource: { buffer: dst } }] });
    const run = (k) => {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(p); pass.setBindGroup(0, bg);
      const tx = ((v.wgx || 8) * 2), ty = ((v.wgy || 8) * 2);
      for (let i = 0; i < k; i++) pass.dispatchWorkgroups(Math.ceil(w16 / tx), Math.ceil(h16 / ty), co / v.coc);
      pass.end();
      device.queue.submit([enc.finish()]);
    };
    run(3); await device.queue.onSubmittedWorkDone(); // warm (incl pipeline compile)
    const t0 = performance.now();
    run(30); await device.queue.onSubmittedWorkDone();
    const ms = (performance.now() - t0) / 30;
    if (!best || ms < best.ms) best = { ...v, ms };
  }
  // stride-2 sweep (conv0b shape: s2ci -> co, input 2x grid): the winner rides
  // along as best.s2. Measured on Ada: w4+coc16 took conv0b 0.77 -> 0.46ms @1080p
  // (coc16 halves the z-slices, w4 evicts the quarter-rate CVTs) - but per-GPU
  // truth comes from here, never hardcoded.
  if (s2ci && co % 8 === 0) {
    const iw = w16 * 2, ih = h16 * 2;
    const s2src = buf(s2ci * iw * ih * 2);
    const s2v = [{ coc: 8 }, { coc: 16 }, { coc: 8, w4: true }, { coc: 16, w4: true }]
      .filter(v => co % v.coc === 0);
    const s2p = await Promise.all(s2v.map(v => device.createComputePipelineAsync({
      layout: 'auto', compute: { module: device.createShaderModule({
        code: wgslConvRBs2(s2ci, co, iw, ih, w16, h16, v) }), entryPoint: 'main' } })));
    let s2best = null;
    for (let vi = 0; vi < s2v.length; vi++) {
      const v = s2v[vi];
      const bg = device.createBindGroup({ layout: s2p[vi].getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: s2src } },
        { binding: 1, resource: { buffer: v.w4 ? wgt32 : wgt } },
        { binding: 2, resource: { buffer: bias } }, { binding: 3, resource: { buffer: alpha } },
        { binding: 4, resource: { buffer: dst } }] });
      const run = (k) => {
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(s2p[vi]); pass.setBindGroup(0, bg);
        for (let i = 0; i < k; i++) {
          pass.dispatchWorkgroups(Math.ceil(w16 / 16), Math.ceil(h16 / 16), co / v.coc);
        }
        pass.end();
        device.queue.submit([enc.finish()]);
      };
      run(3); await device.queue.onSubmittedWorkDone();
      const t0 = performance.now();
      run(30); await device.queue.onSubmittedWorkDone();
      const ms = (performance.now() - t0) / 30;
      if (!s2best || ms < s2best.ms) s2best = { ...v, ms };
    }
    s2src.destroy();
    best.s2 = { coc: s2best.coc, w4: !!s2best.w4, ms: s2best.ms };
  }
  [src, dst, wgt, wgt32, bias, alpha].forEach(b => b.destroy());
  return best;
}
