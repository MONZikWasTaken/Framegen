//! Framecast custom runtime on wgpu - the browser's WGSL kernels (web/rt/rt.js), hosted
//! natively. No TensorRT, no CUDA: runs on Vulkan/DX12/Metal, i.e. any GPU.
//!
//! Buffer-mode forward of the 1-block student (block0 of IFNet_m, scale=4):
//! rgba8 frames in -> interpolated rgba8 frame out, timestep as a parameter.
//! Channel widths come from the weight manifest (full 240 / slim 120 / potato 60).

use anyhow::{anyhow, Context, Result};
use std::collections::HashMap;

const WG: u32 = 8;

#[derive(serde::Deserialize)]
struct TensorMeta {
    shape: Vec<usize>,
    offset: usize,
}

/// Token-substitution templating: WGSL is brace-heavy, `format!` escaping is unreadable.
fn tpl(src: &str, subs: &[(&str, String)]) -> String {
    let mut s = src.to_string();
    for (k, v) in subs {
        s = s.replace(k, v);
    }
    s
}

fn wgsl_prep_full(w: u32, h: u32) -> String {
    tpl(
        r#"
@group(0) @binding(0) var<storage, read> rgba0: array<u32>;
@group(0) @binding(1) var<storage, read> rgba1: array<u32>;
@group(0) @binding(2) var<storage, read_write> imgs: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= $W || y >= $H) { return; }
  let o = y * $W + x;
  let c0 = unpack4x8unorm(rgba0[o]).xyz;
  let c1 = unpack4x8unorm(rgba1[o]).xyz;
  let P = $HW;
  imgs[o] = c0.z; imgs[P + o] = c0.y; imgs[2 * P + o] = c0.x;
  imgs[3 * P + o] = c1.z; imgs[4 * P + o] = c1.y; imgs[5 * P + o] = c1.x;
}"#,
        &[
            ("$HW", (w * h).to_string()),
            ("$W", w.to_string()),
            ("$H", h.to_string()),
        ],
    )
}

fn wgsl_prep_quarter(w: u32, h: u32) -> String {
    tpl(
        r#"
enable f16;
@group(0) @binding(0) var<storage, read> rgba0: array<u32>;
@group(0) @binding(1) var<storage, read> rgba1: array<u32>;
@group(0) @binding(2) var<storage, read_write> xq: array<f16>;
@group(0) @binding(3) var<storage, read> tstep: array<f32>;

fn px(buf: i32, x: i32, y: i32) -> vec3<f32> {
  let v = select(rgba1[y * $W + x], rgba0[y * $W + x], buf == 0);
  return unpack4x8unorm(v).xyz;
}

fn sampleQ(buf: i32, sx: f32, sy: f32) -> vec3<f32> {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  let xa = clamp(x0, 0, $W - 1); let xb = clamp(x0 + 1, 0, $W - 1);
  let ya = clamp(y0, 0, $H - 1); let yb = clamp(y0 + 1, 0, $H - 1);
  let v00 = px(buf, xa, ya); let v10 = px(buf, xb, ya);
  let v01 = px(buf, xa, yb); let v11 = px(buf, xb, yb);
  return mix(mix(v00, v10, fx), mix(v01, v11, fx), fy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= $QW || y >= $QH) { return; }
  let sx = (f32(x) + 0.5) * 4.0 - 0.5;
  let sy = (f32(y) + 0.5) * 4.0 - 0.5;
  let c0 = sampleQ(0, sx, sy);
  let c1 = sampleQ(1, sx, sy);
  let o = y * $QW + x;
  let P = $QHW;
  xq[o] = f16(c0.z); xq[P + o] = f16(c0.y); xq[2 * P + o] = f16(c0.x);
  xq[3 * P + o] = f16(c1.z); xq[4 * P + o] = f16(c1.y); xq[5 * P + o] = f16(c1.x);
  xq[6 * P + o] = f16(tstep[0]);
}"#,
        &[
            ("$QHW", (w / 4 * h / 4).to_string()),
            ("$QW", (w / 4).to_string()),
            ("$QH", (h / 4).to_string()),
            ("$W", w.to_string()),
            ("$H", h.to_string()),
        ],
    )
}

/// stride-2 conv3x3 + bias + PReLU, COC output channels per thread, weights staged
/// through workgroup memory (mirror of the browser v3 kernel, f16 storage).
fn wgsl_conv_s2(ci: u32, co: u32, iw: u32, ih: u32, ow: u32, oh: u32) -> String {
    let coc = if co % 4 == 0 { 4 } else { 1 };
    let slab = ci.min(30);
    tpl(
        r#"
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;

var<workgroup> wsh: array<f16, $SLABF>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let x = i32(gid.x); let y = i32(gid.y);
  let cb = i32(gid.z) * $COC;
  let inb = x < $OW && y < $OH;
  var acc: array<f32, $COC>;
  for (var c = 0; c < $COC; c++) { acc[c] = bias[cb + c]; }

  for (var s = 0; s < $CI; s += $SLAB) {
    let sl = min($SLAB, $CI - s);
    let n = $COC * sl * 9;
    workgroupBarrier();
    var idx = i32(li);
    while (idx < n) {
      let c = idx / (sl * 9);
      let r = idx % (sl * 9);
      wsh[idx] = wgt[(cb + c) * $CI9 + (s + r / 9) * 9 + r % 9];
      idx += 64;
    }
    workgroupBarrier();
    if (inb) {
      for (var ci = 0; ci < sl; ci++) {
        let sbase = (s + ci) * $IHW;
        for (var ky = 0; ky < 3; ky++) {
          let iy = y * 2 + ky - 1;
          if (iy < 0 || iy >= $IH) { continue; }
          for (var kx = 0; kx < 3; kx++) {
            let ix = x * 2 + kx - 1;
            if (ix < 0 || ix >= $IW) { continue; }
            let sv = f32(src[sbase + iy * $IW + ix]);
            let wb = ci * 9 + ky * 3 + kx;
            for (var c = 0; c < $COC; c++) {
              acc[c] += sv * f32(wsh[c * (sl * 9) + wb]);
            }
          }
        }
      }
    }
  }
  if (!inb) { return; }
  for (var c = 0; c < $COC; c++) {
    let cco = cb + c;
    let v = select(alpha[cco] * acc[c], acc[c], acc[c] >= 0.0);
    dst[cco * $OHW + y * $OW + x] = f16(v);
  }
}"#,
        &[
            ("$SLABF", (coc * slab * 9).to_string()),
            ("$SLAB", slab.to_string()),
            ("$CI9", (ci * 9).to_string()),
            ("$CI", ci.to_string()),
            ("$COC", coc.to_string()),
            ("$IHW", (iw * ih).to_string()),
            ("$OHW", (ow * oh).to_string()),
            ("$IW", iw.to_string()),
            ("$IH", ih.to_string()),
            ("$OW", ow.to_string()),
            ("$OH", oh.to_string()),
        ],
    )
}

/// register-blocked stride-1 conv3x3 (2x2 pixels x 4 channels per thread) + residual option
fn wgsl_conv_rb(ci: u32, co: u32, w: u32, h: u32, residual: bool) -> String {
    assert_eq!(co % 4, 0);
    let slab = 20u32.min(ci);
    let mut accs = String::new();
    for c in 0..4 {
        accs.push_str(&format!(
            "  var a{c}0 = bias[cb + {c}]; var a{c}1 = a{c}0; var a{c}2 = a{c}0; var a{c}3 = a{c}0;\n"
        ));
    }
    let mut fma = String::new();
    for c in 0..4 {
        fma.push_str(&format!(
            "          {{ let wv = f32(wsh[{c} * (sl * 9) + wb]);
            a{c}0 += t00 * wv; a{c}1 += t01 * wv; a{c}2 += t10 * wv; a{c}3 += t11 * wv; }}\n"
        ));
    }
    let mut stores = String::new();
    for c in 0..4 {
        for p in 0..4 {
            let res = if residual {
                format!("v + f32(res[o])")
            } else {
                "v".to_string()
            };
            stores.push_str(&format!(
                "  {{ let x = x0 + {px}; let y = y0 + {py};
    if (x < $W && y < $H) {{
      let a = a{c}{p};
      let al = alpha[cb + {c}];
      let v = select(al * a, a, a >= 0.0);
      let o = (cb + {c}) * $HW + y * $W + x;
      dst[o] = f16({res});
    }} }}\n",
                px = p & 1,
                py = p >> 1,
            ));
        }
    }
    let res_bind = if residual {
        "@group(0) @binding(5) var<storage, read> res: array<f16>;"
    } else {
        ""
    };
    tpl(
        &format!(
            r#"
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<f16>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> alpha: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f16>;
{res_bind}

var<workgroup> wsh: array<f16, $SLABW>;
var<workgroup> tile: array<f16, $SLABT>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {{
  let lx = i32(lid.x); let ly = i32(lid.y);
  let ox0 = i32(wid.x) * 16; let oy0 = i32(wid.y) * 16;
  let x0 = ox0 + lx * 2; let y0 = oy0 + ly * 2;
  let cb = i32(wid.z) * 4;
{accs}
  for (var s = 0; s < $CI; s += $SLAB) {{
    let sl = min($SLAB, $CI - s);
    workgroupBarrier();
    var idx = i32(li);
    let wn = 4 * sl * 9;
    while (idx < wn) {{
      let c = idx / (sl * 9);
      let r = idx % (sl * 9);
      wsh[idx] = wgt[(cb + c) * $CI9 + (s + r / 9) * 9 + r % 9];
      idx += 64;
    }}
    var ti = i32(li);
    let tn = sl * 324;
    while (ti < tn) {{
      let ci = ti / 324;
      let r = ti % 324;
      let ty = oy0 + r / 18 - 1;
      let tx = ox0 + r % 18 - 1;
      var v = f16(0.0);
      if (ty >= 0 && ty < $H && tx >= 0 && tx < $W) {{
        v = src[(s + ci) * $HW + ty * $W + tx];
      }}
      tile[ti] = v;
      ti += 64;
    }}
    workgroupBarrier();
    for (var ci = 0; ci < sl; ci++) {{
      let tb = ci * 324 + (ly * 2) * 18 + lx * 2;
      for (var ky = 0; ky < 3; ky++) {{
        let rb = tb + ky * 18;
        for (var kx = 0; kx < 3; kx++) {{
          let t00 = f32(tile[rb + kx]);
          let t01 = f32(tile[rb + kx + 1]);
          let t10 = f32(tile[rb + kx + 18]);
          let t11 = f32(tile[rb + kx + 19]);
          let wb = ci * 9 + ky * 3 + kx;
{fma}
        }}
      }}
    }}
  }}
{stores}
}}"#
        ),
        &[
            ("$SLABW", (4 * slab * 9).to_string()),
            ("$SLABT", (slab * 324).to_string()),
            ("$SLAB", slab.to_string()),
            ("$CI9", (ci * 9).to_string()),
            ("$CI", ci.to_string()),
            ("$HW", (w * h).to_string()),
            ("$W", w.to_string()),
            ("$H", h.to_string()),
        ],
    )
}

fn wgsl_deconv(ci: u32, co: u32, iw: u32, ih: u32, ow: u32, oh: u32) -> String {
    tpl(
        r#"
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f16>;
@group(0) @binding(1) var<storage, read> wgt: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y); let co = i32(gid.z);
  if (x >= $OW || y >= $OH) { return; }
  var acc = bias[co];
  for (var ky = 0; ky < 4; ky++) {
    let ty = y + 1 - ky;
    if (ty < 0 || (ty & 1) != 0) { continue; }
    let iy = ty / 2;
    if (iy >= $IH) { continue; }
    for (var kx = 0; kx < 4; kx++) {
      let tx = x + 1 - kx;
      if (tx < 0 || (tx & 1) != 0) { continue; }
      let ix = tx / 2;
      if (ix >= $IW) { continue; }
      for (var ci = 0; ci < $CI; ci++) {
        acc += f32(src[ci * $IHW + iy * $IW + ix])
             * wgt[ci * $CO16 + co * 16 + ky * 4 + kx];
      }
    }
  }
  dst[co * $OHW + y * $OW + x] = acc;
}"#,
        &[
            ("$CO16", (co * 16).to_string()),
            ("$CI", ci.to_string()),
            ("$IHW", (iw * ih).to_string()),
            ("$OHW", (ow * oh).to_string()),
            ("$IW", iw.to_string()),
            ("$IH", ih.to_string()),
            ("$OW", ow.to_string()),
            ("$OH", oh.to_string()),
        ],
    )
}

fn wgsl_flow_out(w: u32, h: u32) -> String {
    tpl(
        r#"
@group(0) @binding(0) var<storage, read> tmp8: array<f32>;
@group(0) @binding(1) var<storage, read> imgs: array<f32>;
@group(0) @binding(2) var<storage, read_write> outp: array<u32>;

fn tap(c: i32, x: i32, y: i32) -> f32 {
  return tmp8[c * $THW + clamp(y, 0, $TH - 1) * $TW + clamp(x, 0, $TW - 1)];
}
fn up(c: i32, sx: f32, sy: f32) -> f32 {
  let x0 = i32(floor(sx)); let y0 = i32(floor(sy));
  let fx = sx - f32(x0);   let fy = sy - f32(y0);
  return mix(mix(tap(c, x0, y0), tap(c, x0 + 1, y0), fx),
             mix(tap(c, x0, y0 + 1), tap(c, x0 + 1, y0 + 1), fx), fy);
}
fn img(plane: i32, x: i32, y: i32) -> f32 {
  return imgs[plane * $HW + clamp(y, 0, $H - 1) * $W + clamp(x, 0, $W - 1)];
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
  return r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x); let y = i32(gid.y);
  if (x >= $W || y >= $H) { return; }
  let sx = (f32(x) + 0.5) / 8.0 - 0.5;
  let sy = (f32(y) + 0.5) / 8.0 - 0.5;
  let fx0 = up(0, sx, sy) * 8.0;
  let fy0 = up(1, sx, sy) * 8.0;
  let fx1 = up(2, sx, sy) * 8.0;
  let fy1 = up(3, sx, sy) * 8.0;
  let m = 1.0 / (1.0 + exp(-up(4, sx, sy)));
  let w0 = warp3(0, f32(x) + fx0, f32(y) + fy0);
  let w1 = warp3(3, f32(x) + fx1, f32(y) + fy1);
  let bgr = w0 * m + w1 * (1.0 - m);
  let r = u32(clamp(bgr.z, 0.0, 1.0) * 255.0);
  let g = u32(clamp(bgr.y, 0.0, 1.0) * 255.0);
  let b = u32(clamp(bgr.x, 0.0, 1.0) * 255.0);
  outp[y * $W + x] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}"#,
        &[
            ("$THW", (w / 8 * h / 8).to_string()),
            ("$TW", (w / 8).to_string()),
            ("$TH", (h / 8).to_string()),
            ("$HW", (w * h).to_string()),
            ("$W", w.to_string()),
            ("$H", h.to_string()),
        ],
    )
}

const WGSL_TO_F16: &str = r#"
enable f16;
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f16>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < arrayLength(&src)) { dst[i] = f16(src[i]); }
}"#;

pub struct RifeWgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    w: u32,
    h: u32,
    c1: u32,
    c2: u32,
    p_prep_full: wgpu::ComputePipeline,
    p_prep_q: wgpu::ComputePipeline,
    p_conv0a: wgpu::ComputePipeline,
    p_conv0b: wgpu::ComputePipeline,
    p_deconv: wgpu::ComputePipeline,
    p_flow: wgpu::ComputePipeline,
    bg_prep_full: wgpu::BindGroup,
    bg_prep_q: wgpu::BindGroup,
    bg_conv0a: wgpu::BindGroup,
    bg_conv0b: wgpu::BindGroup,
    bg_blocks: Vec<(bool, wgpu::BindGroup)>, // (is_residual_pipeline, bg)
    p_conv_b: wgpu::ComputePipeline,
    p_conv_br: wgpu::ComputePipeline,
    bg_deconv: wgpu::BindGroup,
    bg_flow: wgpu::BindGroup,
    tbuf: wgpu::Buffer,
    rgba0: wgpu::Buffer,
    rgba1: wgpu::Buffer,
    f16a: wgpu::Buffer,
    f16r: wgpu::Buffer,
    act_bytes: u64,
    outp: wgpu::Buffer,
    staging: wgpu::Buffer,
}

impl RifeWgpu {
    pub fn new(w: u32, h: u32, weights_bin: &[u8], weights_json: &str) -> Result<Self> {
        if w % 16 != 0 || h % 16 != 0 {
            return Err(anyhow!("dims must be /16 (got {w}x{h})"));
        }
        let man: HashMap<String, TensorMeta> =
            serde_json::from_str(weights_json).context("weight manifest")?;
        let c1 = man
            .get("block0.conv0.0.0.weight")
            .ok_or_else(|| anyhow!("manifest: conv0.0 missing"))?
            .shape[0] as u32;
        let c2 = man["block0.conv0.1.0.weight"].shape[0] as u32;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("no wgpu adapter: {e}"))?;
        if !adapter.features().contains(wgpu::Features::SHADER_F16) {
            return Err(anyhow!("adapter lacks SHADER_F16"));
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::SHADER_F16,
            ..Default::default()
        }))?;

        let (qw, qh, w8, h8, w16, h16) = (w / 4, h / 4, w / 8, h / 8, w / 16, h / 16);

        let buf = |bytes: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: bytes.div_ceil(4) * 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        // f32 weights upload + GPU f16 conversion for conv weights (as in the browser)
        let mut wbuf: HashMap<String, wgpu::Buffer> = HashMap::new();
        for (name, m) in &man {
            let n: usize = m.shape.iter().product();
            let b = buf((n * 4) as u64);
            queue.write_buffer(&b, 0, &weights_bin[m.offset * 4..m.offset * 4 + n * 4]);
            wbuf.insert(name.clone(), b);
        }
        let pipe = |code: String| -> wgpu::ComputePipeline {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let p_f16 = pipe(WGSL_TO_F16.into());
        let mut conv_w16: HashMap<String, wgpu::Buffer> = HashMap::new();
        {
            let mut enc = device.create_command_encoder(&Default::default());
            // Conv2d weights end with ".0.weight" (PReLU is ".1.weight", deconv is "lastconv.weight")
            for name in man.keys().filter(|k| k.ends_with(".0.weight")) {
                let n: usize = man[name].shape.iter().product();
                let half = buf((n * 2) as u64);
                let bgl = p_f16.get_bind_group_layout(0);
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wbuf[name].as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: half.as_entire_binding() },
                    ],
                });
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&p_f16);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
                drop(pass);
                conv_w16.insert(name.clone(), half);
            }
            queue.submit([enc.finish()]);
        }

        // activations
        let tbuf = buf(4);
        let rgba0 = buf((w * h * 4) as u64);
        let rgba1 = buf((w * h * 4) as u64);
        let imgs = buf((6 * w * h * 4) as u64);
        let xq = buf((7 * qw * qh * 2) as u64);
        let f8 = buf((c1 * w8 * h8 * 2) as u64);
        let act_bytes = (c2 * w16 * h16 * 2) as u64;
        let f16a = buf(act_bytes);
        let f16b = buf(act_bytes);
        let f16r = buf(act_bytes);
        let tmp8 = buf((5 * w8 * h8 * 4) as u64);
        let outp = buf((w * h * 4) as u64);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (w * h * 4) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let p_prep_full = pipe(wgsl_prep_full(w, h));
        let p_prep_q = pipe(wgsl_prep_quarter(w, h));
        let p_conv0a = pipe(wgsl_conv_s2(7, c1, qw, qh, w8, h8));
        let p_conv0b = pipe(wgsl_conv_s2(c1, c2, w8, h8, w16, h16));
        let p_conv_b = pipe(wgsl_conv_rb(c2, c2, w16, h16, false));
        let p_conv_br = pipe(wgsl_conv_rb(c2, c2, w16, h16, true));
        let p_deconv = pipe(wgsl_deconv(c2, 5, w16, h16, w8, h8));
        let p_flow = pipe(wgsl_flow_out(w, h));

        let bg = |p: &wgpu::ComputePipeline, bufs: &[&wgpu::Buffer]| -> wgpu::BindGroup {
            let layout = p.get_bind_group_layout(0);
            let entries: Vec<wgpu::BindGroupEntry> = bufs
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            })
        };

        let g = |n: &str| -> &wgpu::Buffer { &wbuf[n] };
        let g16 = |n: &str| -> &wgpu::Buffer { &conv_w16[n] };

        let bg_prep_full = bg(&p_prep_full, &[&rgba0, &rgba1, &imgs]);
        let bg_prep_q = bg(&p_prep_q, &[&rgba0, &rgba1, &xq, &tbuf]);
        let bg_conv0a = bg(
            &p_conv0a,
            &[&xq, g16("block0.conv0.0.0.weight"), g("block0.conv0.0.0.bias"), g("block0.conv0.0.1.weight"), &f8],
        );
        let bg_conv0b = bg(
            &p_conv0b,
            &[&f8, g16("block0.conv0.1.0.weight"), g("block0.conv0.1.0.bias"), g("block0.conv0.1.1.weight"), &f16a],
        );
        let mut bg_blocks = Vec::new();
        let mut src_is_a = true;
        for i in 0..8 {
            let wn = format!("block0.convblock.{i}.0.weight");
            let bn = format!("block0.convblock.{i}.0.bias");
            let an = format!("block0.convblock.{i}.1.weight");
            let (s, d) = if src_is_a { (&f16a, &f16b) } else { (&f16b, &f16a) };
            if i < 7 {
                bg_blocks.push((false, bg(&p_conv_b, &[s, g16(&wn), g(&bn), g(&an), d])));
            } else {
                bg_blocks.push((true, bg(&p_conv_br, &[s, g16(&wn), g(&bn), g(&an), d, &f16r])));
            }
            src_is_a = !src_is_a;
        }
        let f16out = if src_is_a { &f16a } else { &f16b };
        let bg_deconv = bg(
            &p_deconv,
            &[f16out, g("block0.lastconv.weight"), g("block0.lastconv.bias"), &tmp8],
        );
        let bg_flow = bg(&p_flow, &[&tmp8, &imgs, &outp]);

        Ok(Self {
            device,
            queue,
            w,
            h,
            c1,
            c2,
            p_prep_full,
            p_prep_q,
            p_conv0a,
            p_conv0b,
            p_deconv,
            p_flow,
            bg_prep_full,
            bg_prep_q,
            bg_conv0a,
            bg_conv0b,
            bg_blocks,
            p_conv_b,
            p_conv_br,
            bg_deconv,
            bg_flow,
            tbuf,
            rgba0,
            rgba1,
            f16a,
            f16r,
            act_bytes,
            outp,
            staging,
        })
    }

    pub fn channels(&self) -> (u32, u32) {
        (self.c1, self.c2)
    }

    /// rgba0/rgba1: w*h*4 bytes. Returns the interpolated frame (rgba, w*h*4).
    pub fn run(&self, a: &[u8], b: &[u8], t: f32) -> Result<Vec<u8>> {
        let (w, h) = (self.w, self.h);
        let gx = |n: u32| n.div_ceil(WG);
        let (w8, h8, w16, h16) = (w / 8, h / 8, w / 16, h / 16);
        let z0a = if self.c1 % 4 == 0 { self.c1 / 4 } else { self.c1 };
        let z0b = self.c2 / 4;
        let (cbx, cby) = (w16.div_ceil(16), h16.div_ceil(16));

        self.queue.write_buffer(&self.tbuf, 0, &t.to_le_bytes());
        self.queue.write_buffer(&self.rgba0, 0, a);
        self.queue.write_buffer(&self.rgba1, 0, b);
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.p_prep_full);
            pass.set_bind_group(0, &self.bg_prep_full, &[]);
            pass.dispatch_workgroups(gx(w), gx(h), 1);
            pass.set_pipeline(&self.p_prep_q);
            pass.set_bind_group(0, &self.bg_prep_q, &[]);
            pass.dispatch_workgroups(gx(w / 4), gx(h / 4), 1);
            pass.set_pipeline(&self.p_conv0a);
            pass.set_bind_group(0, &self.bg_conv0a, &[]);
            pass.dispatch_workgroups(gx(w8), gx(h8), z0a);
            pass.set_pipeline(&self.p_conv0b);
            pass.set_bind_group(0, &self.bg_conv0b, &[]);
            pass.dispatch_workgroups(gx(w16), gx(h16), z0b);
        }
        enc.copy_buffer_to_buffer(&self.f16a, 0, &self.f16r, 0, self.act_bytes);
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (residual, g) in &self.bg_blocks {
                pass.set_pipeline(if *residual { &self.p_conv_br } else { &self.p_conv_b });
                pass.set_bind_group(0, g, &[]);
                pass.dispatch_workgroups(cbx, cby, z0b);
            }
            pass.set_pipeline(&self.p_deconv);
            pass.set_bind_group(0, &self.bg_deconv, &[]);
            pass.dispatch_workgroups(gx(w8), gx(h8), 5);
            pass.set_pipeline(&self.p_flow);
            pass.set_bind_group(0, &self.bg_flow, &[]);
            pass.dispatch_workgroups(gx(w), gx(h), 1);
        }
        enc.copy_buffer_to_buffer(&self.outp, 0, &self.staging, 0, (w * h * 4) as u64);
        self.queue.submit([enc.finish()]);

        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| anyhow!("poll: {e:?}"))?;
        rx.recv()??;
        let out = slice
            .get_mapped_range()
            .map_err(|e| anyhow!("map range: {e:?}"))?
            .to_vec();
        self.staging.unmap();
        Ok(out)
    }
}
