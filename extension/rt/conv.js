// The residual-block conv kernels of TinySR (rt/sr.js): the tunable WGSL generators of the previous engine, kept
// for the upscaler alone. wgslConvRB is the portable kernel, wgslConvRBSg its subgroup variant; the two WGSL_TO_*
// shaders convert the weights once at init.
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
